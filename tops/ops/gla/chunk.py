import functools

import jax
import jax.experimental.pallas as pl
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu
from tops.utils import pad_to_multiple, prepare_chunk_indices
from tops.ops.utils import is_tpu_runtime
from tops.ops.common.chunk_h import chunk_fwd_h_kernel


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================


def chunk_local_cumsum_ref(
    g: jax.Array,
    chunk_size: int,
    scale: float | None = None,
    reverse: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
) -> jax.Array:
    """Chunk-local cumulative sum of gates.

    Args:
        g: [B, T, H, K] — log-space gates (T must be a multiple of chunk_size)
        chunk_size: block size
        cu_seqlens: unused, kept for interface compatibility

    Returns:
        g_cumsum: [B, T, H, K] — chunk-local cumsum
    """
    B, T, H, K = g.shape
    assert reverse == False, "Reverse mode not supported in chunk_local_cumsum"
    assert T % chunk_size == 0, (
        "T must be a multiple of chunk_size for chunk_local_cumsum"
    )
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % chunk_size == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_local_cumsum"
    )
    g = g.reshape(-1, H, K)
    C = chunk_size
    NT = B * T // C
    g = g.reshape(NT, C, H, K)
    g_cumsum = jnp.cumsum(g, axis=1).reshape(B, T, H, K)
    if scale is not None:
        g_cumsum = g_cumsum * scale
    return g_cumsum


def chunk_cumsum_kernel(
    cu_seqlens_ref,
    chunk_indices_ref,
    s_ref,
    o_ref,
    *,
    BT: int,
    BS: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
    IS_VARLEN: bool,
):
    i_s, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    if IS_VARLEN:
        i_n, local_i_t = chunk_indices_ref[i_t, 0], chunk_indices_ref[i_t, 1]
        bos, eos = cu_seqlens_ref[i_n], cu_seqlens_ref[i_n + 1]
        start_t = bos + local_i_t * BT
    else:
        start_t = i_t * BT

    start_s = i_s * BS

    # Each program handles one (BT, BS) tile.
    s = s_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)]

    if IS_VARLEN:
        T_seq = eos - bos
        valid_len = T_seq - local_i_t * BT
        valid_mask = (jnp.arange(BT) < valid_len).astype(jnp.float32)[:, None]
        s = s.astype(jnp.float32) * valid_mask
    else:
        s = s.astype(jnp.float32)
    T = s.shape[0]

    if REVERSE:
        rows = [s[T - 1]]
        for i in range(T - 2, -1, -1):
            rows.append(rows[-1] + s[i])
        rows.reverse()
        o = jnp.stack(rows, axis=0)

    else:
        rows = [s[0]]
        for i in range(1, T):
            rows.append(rows[-1] + s[i])
        o = jnp.stack(rows, axis=0)

    if HAS_SCALE:
        o = o * scale

    o_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)] = o.astype(o_ref.dtype)


def chunk_local_cumsum_vector(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be power of 2"
    )

    if head_first:
        B, H, T, S = g.shape
        # Normalize to (B*H, T, S) to greatly simplify pointer offsets in the kernel.
        g_flat = g.reshape(B * H, T, S)
    else:
        B, T, H, S = g.shape
        g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

    BT = chunk_size
    BS = 128
    out_dtype = output_dtype or g.dtype
    HAS_SCALE = scale is not None
    scale_val = scale if scale is not None else 1.0

    interpret = not is_tpu_runtime()

    # Pad the S dimension to satisfy TPU shape constraints.
    pad_S = (BS - (S % BS)) % BS
    if pad_S > 0:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))

    S_padded = S + pad_S
    NS = S_padded // BS

    # For fixed-length inputs, synthesize cu_seqlens/chunk_indices to simplify kernel control flow.
    is_varlen = cu_seqlens is not None
    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        NT = (T + BT - 1) // BT
        # Dummy arrays for scalar prefetch (not used in the kernel when IS_VARLEN=False).
        cu_seqlens = jnp.zeros(1, dtype=jnp.int32)
        chunk_indices = jnp.zeros((1, 2), dtype=jnp.int32)
    grid = (NS, NT, B * H)

    # In varlen mode, append BT padding at the end to prevent dslice overflow.
    if is_varlen:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))

    kernel = functools.partial(
        chunk_cumsum_kernel,
        BT=BT,
        BS=BS,
        REVERSE=reverse,
        HAS_SCALE=HAS_SCALE,
        scale=scale_val,
        IS_VARLEN=is_varlen,
    )

    o_flat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=pl.no_block_spec,
            out_specs=pl.no_block_spec,
        ),
        out_shape=jax.ShapeDtypeStruct(g_flat.shape, out_dtype),
        interpret=interpret,
    )(cu_seqlens, chunk_indices, g_flat)

    # Remove the padding added earlier.
    o_flat = o_flat[:, :T, :S]

    # Convert the normalized layout back to the user-facing layout.
    if head_first:
        return o_flat.reshape(B, H, T, S)
    else:
        return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# =============================================================================
# Sub-function 2: chunk_fwd_h
# =============================================================================


def chunk_fwd_h_ref(
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    h0: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Inter-chunk hidden state propagation.

    Computes the hidden state at the start of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        gk: [B, T, H, K] — chunk-local cumsum of gates
        h0: [N, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        cu_seqlens_cpu: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_fwd_h"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )
    # seqlens = jnp.diff(cu_seqlens_cpu) if cu_seqlens_cpu is not None else None

    k = k.reshape(-1, H, K)
    v = v.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None
    h0 = h0.reshape(-1, H, K, V) if h0 is not None else None

    ht = jnp.zeros([N, H, K, V], dtype=jnp.float32)
    h_all = jnp.zeros([B, NT, H, K, V], dtype=k.dtype)
    is_varlen = cu_seqlens_cpu is not None
    for i_n in range(N):
        if not is_varlen:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        h = jnp.zeros((H, K, V), dtype=jnp.float32)
        if h0 is not None:
            h = h + h0[i_n].astype(jnp.float32)

        NT = (eos - bos) // C
        for i_t in range(NT):
            # varlen (B=1): use absolute chunk index; non-varlen: (batch, local_chunk)
            bi = 0 if is_varlen else i_n
            ti = bos // C + i_t if is_varlen else i_t
            h_all = h_all.at[bi, ti].set(h.astype(h_all.dtype))
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]
            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= jnp.exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * jnp.exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            # h += jnp.einsum("chk,chv->hkv")
            h = h + lax.dot_general(
                b_k,
                b_v,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        if output_final_state:
            ht = ht.at[i_n].set(h.astype(ht.dtype))

    return h_all, ht


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk_ref(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    scale: float,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Intra-chunk attention matrix with causal mask.

    Args:
        q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
        k: [B, T, H, K] — keys
        g: [B, T, H, K] — chunk-local cumsum of gates
        scale: scaling factor
        cu_seqlens: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        A: [B, NT, C, H, C] — intra-chunk causal attention matrix
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    g_c = g.reshape(B, NT, C, H, K)

    # Numerical stabilization: reference point g_n per chunk (first row)
    g_n = g_c[:, :, 0:1, :, :]  # [B, NT, 1, H, K]
    q_gated = q_c * jnp.exp(g_c - g_n)
    k_gated = k_c * jnp.exp(g_n - g_c)

    # [B, NT, H, C, K] @ [B, NT, H, K, C] -> [B, NT, H, C, C] -> [B, NT, C, H, C]
    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_gated, k_gated,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ) * scale
    A = A.reshape(B, T, H, C)

    return A


# =============================================================================
# Pallas kernel: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk_pl(
    q_ref,
    k_ref,
    g_ref,  # in
    A_ref,  # out
    *,
    BT,
    scale,
):
    """GLA forward intra-chunk attention matrix Pallas kernel.

    Grid: (H, total_NT) where total_NT = B * NT.
    Refs (after block spec indexing):
      q_ref/k_ref/g_ref: (1, 1, BT, K)
      A_ref: (1, 1, BT, BT)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_g = g_ref[0, 0].astype(jnp.float32)  # (BT, K)

    b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_kg = (b_k * jnp.exp(-b_g)).astype(b_k.dtype)

    b_A = (
        jnp.dot(
            b_qg,
            b_kg.T,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    A_ref[0, 0] = b_A.astype(A_ref.dtype)


def chunk_gla_fwd_intra_gk(
    q: jax.Array,  # [B, T, H, K]
    k: jax.Array,  # [B, T, H, K]
    g: jax.Array,  # [B, T, H, K]
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Launcher for chunk_gla_fwd_intra_gk Pallas kernel.

    Pre-reshapes inputs to (H, total_NT, ...) so the kernel is
    agnostic to batch/varlen structure.

    Returns:
        A: [B, T, H, BT] — intra-chunk attention matrix (float32)
    """
    B, T, H, K = q.shape
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    # Reshape: [B, T, H, K] -> [B, NT, BT, H, K] -> [H, B*NT, BT, K]
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)

    # Block specs — grid = (H, total_NT)
    spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    A_shape = jax.ShapeDtypeStruct([H, total_NT, BT, BT], jnp.float32)

    A = pl.pallas_call(
        functools.partial(chunk_gla_fwd_intra_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=A_shape,
        in_specs=[spec, spec, spec],
        out_specs=A_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
    )(_q, _k, _g)

    # Post-reshape: [H, total_NT, BT, BT] -> [B, T, H, BT]
    A = A.reshape(H, B, NT, BT, BT)
    A = A.transpose(1, 0, 2, 3, 4)  # (B, H, NT, BT, BT)
    A = A.reshape(B, H, NT * BT, BT)  # (B, H, T, BT)
    A = A.transpose(0, 2, 1, 3)  # (B, T, H, BT)
    return A


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk_ref(
    q: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    A: jax.Array,
    h: jax.Array,
    scale: float,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Combine inter-chunk and intra-chunk contributions to produce output.

    Args:
        q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
        v: [B, T, H, V] — values
        gk: [B, T, H, K] — chunk-local cumsum of gates
        A: [B, T, H, BT] — intra-chunk attention matrix
        h: [B, NT, H, K, V] — hidden state at start of each chunk
        scale: scaling factor
        cu_seqlens: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = B * T // C
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_gla_fwd_o_gk_ref"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )

    q = q.reshape(-1, C, H, K)
    v = v.reshape(-1, C, H, V)
    gk = gk.reshape(-1, C, H, K)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    qg = q * jnp.exp(gk)

    # Inter-chunk: o_inter = scale * (q_gated @ h)
    o_inter = scale * jnp.einsum("nchk,nhkv->nchv", qg, h)  # [C, K] @ [K, V] -> [C, V]

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))[
        :, None, :
    ]  # (C, 1, C) → broadcasts to (NT, C, H, C)
    n_A = jnp.where(causal_mask, A, 0.0)

    # [C, C] @ [C, V] -> [C, V]
    # Intra-chunk: o_intra = A @ v, contract over j (key position within chunk)
    o_intra = jnp.einsum("nihj,njhv->nihv", n_A, v)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


# =============================================================================
# Backward sub-function 1: chunk_gla_bwd_dA
# =============================================================================


def chunk_gla_bwd_dA_ref(
    v: jax.Array,
    do: jax.Array,
    scale: float,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Gradient of the intra-chunk attention matrix.

    Args:
        v:  [B, T, H, V] — values
        do: [B, T, H, V] — output gradient
        scale: scaling factor
        chunk_size: block size

    Returns:
        dA: [B, T, H, C] — lower-triangular masked gradient
    """
    B, T, H, V = v.shape
    C = chunk_size
    NT = T // C

    v_c = v.reshape(B, NT, C, H, V)
    do_c = do.reshape(B, NT, C, H, V)

    # dA[i,j] = scale * do[i] . v[j]  for j <= i
    dA = (
        jnp.einsum("bnihv,bnjhv->bnihj", do_c, v_c, precision=lax.Precision.HIGHEST)
        * scale
    )

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    dA = jnp.where(causal_mask[None, None, :, None, :], dA, 0.0)

    dA = dA.reshape(B, T, H, C)
    return dA


# =============================================================================
# Backward sub-function 2: chunk_gla_bwd_dv
# =============================================================================


def chunk_gla_bwd_dv_ref(
    k: jax.Array,
    g_cumsum: jax.Array,
    A: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Gradient of v: intra-chunk (A^T @ do) + inter-chunk (k_decay @ dh).

    Args:
        k:        [B, T, H, K]
        g_cumsum: [B, T, H, K]
        A:        [B, T, H, C] — intra-chunk attention matrix
        do:       [B, T, H, V]
        dh:       [B, NT, H, K, V]
        chunk_size: block size

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = k.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.reshape(B, NT, C, H, K)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    A_c = A.reshape(B, NT, C, H, C)

    # Intra-chunk: dv[j] = sum_{i>=j} A[i,j] * do[i]
    # A is lower-triangular (nonzero when i >= j), keep those entries
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A_masked = jnp.where(causal_mask[None, None, :, None, :], A_c, 0.0)
    dv_intra = jnp.einsum(
        "bnihj,bnihv->bnjhv", A_masked, do_c, precision=lax.Precision.HIGHEST
    )

    # Inter-chunk: k_decay @ dh
    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K] — gate cumsum at chunk end
    k_decay = k_c * jnp.exp(gn[:, :, None, :, :] - gc_c)  # [B, NT, C, H, K]
    dv_inter = jnp.einsum(
        "bnchk,bnhkv->bnchv", k_decay, dh, precision=lax.Precision.HIGHEST
    )

    dv = (dv_intra + dv_inter).reshape(B, T, H, V)
    return dv


# =============================================================================
# Backward sub-function 3: chunk_gla_bwd_dqk_intra
# =============================================================================


def chunk_gla_bwd_dqk_intra_ref(
    q: jax.Array,
    k: jax.Array,
    g_cumsum: jax.Array,
    dA: jax.Array,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Intra-chunk dq, dk from the attention matrix gradient dA.

    Args:
        q:        [B, T, H, K]
        k:        [B, T, H, K]
        g_cumsum: [B, T, H, K]
        dA:       [B, T, H, C]
        chunk_size: block size

    Returns:
        dq: [B, T, H, K]
        dk: [B, T, H, K]
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    dA_c = dA.reshape(B, NT, C, H, C)

    # dq[i] = exp(gc[i]) * sum_{j<=i} dA[i,j] * k[j] * exp(-gc[j])
    # dA is already lower-triangular masked, so causal constraint is embedded
    k_neg = k_c * jnp.exp(-gc_c)
    dq = jnp.exp(gc_c) * jnp.einsum(
        "bnihj,bnjhk->bnihk", dA_c, k_neg, precision=lax.Precision.HIGHEST
    )

    # dk[j] = exp(-gc[j]) * sum_{i>=j} dA[i,j] * q[i] * exp(gc[i])
    q_pos = q_c * jnp.exp(gc_c)
    dk = jnp.exp(-gc_c) * jnp.einsum(
        "bnihj,bnihk->bnjhk", dA_c, q_pos, precision=lax.Precision.HIGHEST
    )

    dq = dq.reshape(B, T, H, K)
    dk = dk.reshape(B, T, H, K)
    return dq, dk


# =============================================================================
# Backward sub-function 4: chunk_gla_bwd_dqkg
# =============================================================================


def chunk_gla_bwd_dqkg_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    g_cumsum: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    dq: jax.Array,
    dk: jax.Array,
    scale: float,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Inter-chunk dq, dk contributions + gate gradient dg.

    Args:
        q:        [B, T, H, K]
        k:        [B, T, H, K]
        v:        [B, T, H, V]
        h:        [B, NT, H, K, V] — hidden states at chunk starts
        g_cumsum: [B, T, H, K]
        do:       [B, T, H, V]
        dh:       [B, NT, H, K, V]
        dq:       [B, T, H, K] — intra-chunk dq
        dk:       [B, T, H, K] — intra-chunk dk
        scale: scaling factor
        chunk_size: block size

    Returns:
        dq: [B, T, H, K] — intra + inter combined
        dk: [B, T, H, K] — intra + inter combined
        dg: [B, T, H, K] — gate gradient
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    dq_c = dq.reshape(B, NT, C, H, K)
    dk_c = dk.reshape(B, NT, C, H, K)

    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K]

    # Inter-chunk dq: scale * exp(gc) * (do @ h^T over V)
    dq_inter = (
        scale
        * jnp.exp(gc_c)
        * jnp.einsum("bnchv,bnhkv->bnchk", do_c, h, precision=lax.Precision.HIGHEST)
    )

    # Inter-chunk dk: exp(gn - gc) * (v @ dh^T over V)
    dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * jnp.einsum(
        "bnchv,bnhkv->bnchk",
        v_c,
        dh,
        precision=lax.Precision.HIGHEST,
    )

    # Combine intra + inter
    dq_total = dq_c + dq_inter
    dk_total = dk_c + dk_inter

    # Gate gradient
    # dgk_inter = exp(gn) * sum_v(h * dh) + sum_t(dk_inter * k)
    dgk_inter = jnp.exp(gn) * jnp.einsum(
        "bnhkv,bnhkv->bnhk", h, dh, precision=lax.Precision.HIGHEST
    ) + jnp.sum(dk_inter * k_c, axis=2)  # [B, NT, H, K]

    # dg_raw = q * dq_total - k * dk_total
    dg_raw = q_c * dq_total - k_c * dk_total  # [B, NT, C, H, K]


    # Reverse cumsum over time dimension within each chunk
    dg = (
        jnp.cumsum(dg_raw[:, :, ::-1, :, :], axis=2)[:, :, ::-1, :, :]
        + dgk_inter[:, :, None, :, :]
    )

    dq_out = dq_total.reshape(B, T, H, K)
    dk_out = dk_total.reshape(B, T, H, K)
    dg_out = dg.reshape(B, T, H, K)
    return dq_out, dk_out, dg_out


# =============================================================================
# Varlen padding helpers (JAX)
# =============================================================================


def _pad_varlen_seqs_jax(
    tensors: list[jax.Array],
    cu_seqlens_cpu: jax.Array,
    chunk_size: int,
) -> tuple[list[jax.Array], jax.Array, list[int] | None, list[int] | None]:
    """Pad each variable-length segment along dim=1 to a multiple of chunk_size.

    Args:
        tensors: list of [1, T_total, ...] arrays sharing the same sequence layout
        cu_seqlens_cpu: [N+1] cumulative sequence lengths on CPU
        chunk_size: block size

    Returns:
        (padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens)
        If no padding is needed, returns (tensors, cu_seqlens, None, None).
    """
    N = len(cu_seqlens_cpu) - 1
    C = chunk_size
    orig_seqlens = jnp.diff(cu_seqlens_cpu).tolist()
    padded_seqlens = [((L + C - 1) // C) * C for L in orig_seqlens]

    if orig_seqlens == padded_seqlens:
        return tensors, cu_seqlens_cpu, None, None

    padded = [[] for _ in tensors]
    for i in range(N):
        bos = int(cu_seqlens_cpu[i])
        L = orig_seqlens[i]
        pad = padded_seqlens[i] - L
        for j, t in enumerate(tensors):
            seg = t[:, bos : bos + L]
            if pad > 0:
                pw = ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2)
                seg = jnp.pad(seg, pw)
            padded[j].append(seg)

    padded_tensors = [jnp.concatenate(p, axis=1) for p in padded]
    offsets = [0]
    for pl in padded_seqlens:
        offsets.append(offsets[-1] + pl)
    new_cu_seqlens = jnp.array(offsets, dtype=jnp.int32)

    return padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens


def _unpad_varlen_seqs_jax(
    tensor: jax.Array,
    orig_seqlens: list[int],
    padded_seqlens: list[int],
) -> jax.Array:
    """Remove per-segment padding from a variable-length array along dim=1."""
    parts = []
    offset = 0
    for L, PL in zip(orig_seqlens, padded_seqlens):
        parts.append(tensor[:, offset : offset + L])
        offset += PL
    return jnp.concatenate(parts, axis=1)


# =============================================================================
# Orchestrator: chunk_gla_bwd
# =============================================================================


def chunk_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    g_cumsum: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    h: jax.Array | None,
    A: jax.Array | None,
    do: jax.Array,
    dht: jax.Array | None,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Chunk GLA backward orchestrator.

    Follows the FLA/Triton convention: h and A are passed from the forward
    pass to avoid recomputation. If None, they are recomputed internally.

    Pads inputs to a multiple of chunk_size, then calls the sub-functions.
    For varlen mode, each segment is individually padded to a multiple of
    chunk_size and cu_seqlens is updated accordingly.

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        g:  [B, T, H, K] — raw log-space gates (or None if g_gamma is used)
        g_gamma: broadcastable to [B, T, H, K] or None — constant gate
        g_cumsum: [B, T, H, K] or None — pre-computed chunk-local cumsum
        scale: scaling factor
        initial_state: [B, H, K, V] or [N, H, K, V] (varlen) or None
        h:  [B, NT, H, K, V] or None — hidden states from forward
        A:  [B, T, H, C] or None — intra-chunk attention from forward
        do: [B, T, H, V] — output gradient
        dht: [B, H, K, V] or [N, H, K, V] (varlen) or None — terminal state gradient
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1. Segment lengths need not be multiples
            of chunk_size — they are padded internally.
        chunk_size: block size

    Returns:
        (dq, dk, dv, dg, dh0)
    """
    from tops.ops.common.chunk_h import chunk_bwd_dh_ref

    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Record if g was originally None
    g_orig = g

    # Broadcast g_gamma into full g if g is not provided
    if g is None:
        if g_gamma is not None:
            g = jnp.broadcast_to(g_gamma, q.shape)
        else:
            g = jnp.zeros_like(q)

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    if cu_seqlens is not None:
        assert B == 1
        [q, k, v, g, do], cu_seqlens, orig_seqlens, padded_seqlens = (
            _pad_varlen_seqs_jax([q, k, v, g, do], cu_seqlens, C)
        )
    else:
        NT = (T + C - 1) // C
        T_padded = NT * C
        if T_padded > T:
            pad = T_padded - T
            pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
            q = jnp.pad(q, pad_width)
            k = jnp.pad(k, pad_width)
            v = jnp.pad(v, pad_width)
            g = jnp.pad(g, pad_width)
            do = jnp.pad(do, pad_width)

    # 1. Chunk-local cumsum
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum_ref(g, C, cu_seqlens_cpu=cu_seqlens)

    # 2. Forward replay to get h
    if h is None:
        h, _ = chunk_fwd_h_ref(
            k,
            v,
            gk=g_cumsum,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens_cpu=cu_seqlens,
            chunk_size=C,
        )

    # 3. Backward hidden state gradients
    dh, dh0 = chunk_bwd_dh_ref(
        q,
        k,
        v,
        g_cumsum,
        do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens_cpu=cu_seqlens,
        chunk_size=C,
    )

    # 4. dv (uses A from forward)
    if A is None:
        A = chunk_gla_fwd_intra_gk_ref(q, k, g_cumsum, scale, chunk_size=C)
    dv = chunk_gla_bwd_dv_ref(k, g_cumsum, A, do, dh, cu_seqlens_cpu=cu_seqlens, chunk_size=C)

    # 5. dA
    dA = chunk_gla_bwd_dA_ref(v, do, scale, cu_seqlens_cpu=cu_seqlens, chunk_size=C)

    # 6. Intra-chunk dq, dk
    dq, dk = chunk_gla_bwd_dqk_intra_ref(q, k, g_cumsum, dA, cu_seqlens_cpu=cu_seqlens, chunk_size=C)

    # 7. Inter-chunk dq, dk + gate gradient
    dq, dk, dg = chunk_gla_bwd_dqkg_ref(
        q,
        k,
        v,
        h,
        g_cumsum,
        do,
        dh,
        dq,
        dk,
        scale,
        cu_seqlens_cpu=cu_seqlens,
        chunk_size=C,
    )

    # --- unpadding ---
    if orig_seqlens is not None:
        dq = _unpad_varlen_seqs_jax(dq, orig_seqlens, padded_seqlens)
        dk = _unpad_varlen_seqs_jax(dk, orig_seqlens, padded_seqlens)
        dv = _unpad_varlen_seqs_jax(dv, orig_seqlens, padded_seqlens)
        dg = _unpad_varlen_seqs_jax(dg, orig_seqlens, padded_seqlens)
    else:
        dq = dq[:, :T]
        dk = dk[:, :T]
        dv = dv[:, :T]
        dg = dg[:, :T]

    if g_orig is None and g_gamma is not None:
        # Sum-reduce dg to match g_gamma's shape
        for i, (d_full, d_gamma) in enumerate(zip(dg.shape[::-1], g_gamma.shape[::-1])):
            if d_full != d_gamma:
                dg = jnp.sum(dg, axis=len(dg.shape) - 1 - i, keepdims=True)
        missing_dims = len(dg.shape) - len(g_gamma.shape)
        if missing_dims > 0:
            dg = jnp.sum(dg, axis=tuple(range(missing_dims)))
        dg = dg.reshape(g_gamma.shape)

    return dq, dk, dv, dg, dh0


def chunk_gla_bwd_with_pl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    g_cumsum: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    h: jax.Array | None,
    A: jax.Array | None,
    do: jax.Array,
    dht: jax.Array | None,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Chunk GLA backward orchestrator using Pallas kernels."""
    from tops.ops.common.chunk_h import chunk_bwd_dh_kernel

    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Record if g was originally None
    g_orig = g

    # Broadcast g_gamma into full g if g is not provided
    if g is None:
        if g_gamma is not None:
            g = jnp.broadcast_to(g_gamma, q.shape)
        else:
            g = jnp.zeros_like(q)

    assert T % C == 0, "T must be a multiple of chunk_size for chunk_gla_bwd_with_pl"
    assert (cu_seqlens is None) or (cu_seqlens % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_gla_bwd_with_pl"
    )

    # 1. Chunk-local cumsum
    if g_cumsum is None:
        if g_gamma is not None and cu_seqlens is None:
            _, T_pad, _, _ = q.shape
            pos = jnp.arange(1, C + 1, dtype=jnp.float32)
            pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
            g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)
        else:
            g_cumsum = chunk_local_cumsum_vector(g, C, cu_seqlens=cu_seqlens)

    # 2. Forward replay to get h
    if h is None:
        h, _ = chunk_fwd_h_kernel(
            k,
            v,
            gk=g_cumsum,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=C,
        )
        if cu_seqlens is None:
            h = h.reshape(B, T // C, H, K, V)

    # 3. Backward hidden state gradients
    dh, dh0 = chunk_bwd_dh_kernel(
        q,
        k,
        v,
        gk=g_cumsum,
        do=do,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
    )
    if cu_seqlens is None:
        dh = dh.reshape(B, T // C, H, K, V)
        if dh0 is not None:
            dh0 = dh0.reshape(B, H, K, V)

    # 4. Fused backward pass for dq, dk, dv, dg
    dq, dk, dv, dg = chunk_gla_bwd_fused_pl(
        q, k, v, g_cumsum, h, do, dh, scale=scale, chunk_size=C
    )

    if g_orig is None and g_gamma is not None:
        # Sum-reduce dg to match g_gamma's shape
        for i, (d_full, d_gamma) in enumerate(zip(dg.shape[::-1], g_gamma.shape[::-1])):
            if d_full != d_gamma:
                dg = jnp.sum(dg, axis=len(dg.shape) - 1 - i, keepdims=True)
        missing_dims = len(dg.shape) - len(g_gamma.shape)
        if missing_dims > 0:
            dg = jnp.sum(dg, axis=tuple(range(missing_dims)))
        dg = dg.reshape(g_gamma.shape)

    return dq, dk, dv, dg, dh0


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    g_cumsum: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Chunk GLA forward orchestrator.

    Pads inputs to a multiple of chunk_size (T dim) and 128 (K/V dims),
    then calls the 4 sub-functions.
    For varlen mode, each segment is individually padded to a multiple of
    chunk_size and cu_seqlens is updated accordingly.

    Returns:
        (g_cumsum, A, h, ht, o)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    if g is None:
        if g_gamma is not None:
            g = jnp.broadcast_to(g_gamma, q.shape)
        else:
            g = jnp.zeros_like(q)

    if cu_seqlens is not None:
        assert B == 1
        [q, k, v, g], cu_seqlens, orig_seqlens, padded_seqlens = (
            _pad_varlen_seqs_jax([q, k, v, g], cu_seqlens, C)
        )
    elif T % C != 0:
        q, k, v, g = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v, g))

    # K/V padding (chunk_fwd_h_kernel requires K%128==0, V%128==0)
    q, k, g = (pad_to_multiple(x, 128, axis=3, val=0) for x in (q, k, g))
    v = pad_to_multiple(v, 128, axis=3, val=0)
    if initial_state is not None:
        initial_state = pad_to_multiple(initial_state, [128, 128], axis=[2, 3], val=0)

    if g_cumsum is None:
        if g_gamma is not None and cu_seqlens is None:
            # Constant g_gamma: compute chunk-local cumsum analytically.
            # For constant g, cumsum within each chunk = g_gamma * [1, 2, ..., C].
            # This avoids chunk_local_cumsum_vector whose pallas kernel uses
            # no_block_spec and loads the full tensor into VMEM.
            _, T_pad, _, _ = q.shape
            pos = jnp.arange(1, C + 1, dtype=jnp.float32)
            pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
            g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)
        else:
            g_cumsum = chunk_local_cumsum_vector(g, C, cu_seqlens=cu_seqlens)

    h, ht = chunk_fwd_h_kernel(
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        gk=g_cumsum,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
    )
    if cu_seqlens is None:
        h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])
    else:
        h = h.reshape(1, -1, h.shape[1], h.shape[2], h.shape[3])

    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    o = chunk_gla_fwd_o_gk(
        q, v, g_cumsum, A, h, scale, chunk_size=C,
    )

    # --- unpadding ---
    o = o[..., :V]
    g_cumsum = g_cumsum[..., :K]
    h = h[..., :K, :V]
    if ht is not None:
        ht = ht[..., :K, :V]
    # T unpadding
    if orig_seqlens is not None:
        o = _unpad_varlen_seqs_jax(o, orig_seqlens, padded_seqlens)
    else:
        o = o[:, :T]
    return g_cumsum, A, h, ht, o


# =============================================================================
# Public API: chunk_gla
# =============================================================================


def chunk_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
    """Chunked GLA — pure JAX implementation.

    Splits the sequence into blocks of chunk_size and computes in parallel
    within each block, propagating hidden states across blocks.

    Either ``g`` or ``g_gamma`` (or both) may be provided:
    - ``g``: [B, T, H, K] — per-step log-space gates.
    - ``g_gamma``: broadcastable to [B, T, H, K] — constant gate that is
      broadcast across the sequence.  Used only when ``g is None``.
    If neither is given, gates default to zero (no decay).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] or None — gates (log-space, after logsigmoid)
        g_gamma: broadcastable to [B, T, H, K] or None — constant gate
        scale: scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V]
        output_final_state: whether to return final state
        cu_seqlens: [N+1] variable-length cumulative lengths
        chunk_size: block size, default 16

    Returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] or None
    """
    dtype = q.dtype
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    if g is not None:
        g = g.astype(jnp.float32)
    if g_gamma is not None:
        g_gamma = g_gamma.astype(jnp.float32)
    B, T, H, K = q.shape

    if scale is None:
        scale = K**-0.5

    _, _, _, ht, o = chunk_gla_fwd(
        q,
        k,
        v,
        g,
        g_gamma=g_gamma,
        g_cumsum=None,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    final_state = ht if output_final_state else None
    return o.astype(dtype), final_state


# =============================================================================
# Pallas kernel: chunk_gla_fwd_o_gk (unified, handles both varlen and non-varlen)
# =============================================================================


def chunk_gla_fwd_o_gk_pl_kernel(
    q_ref,
    v_ref,
    g_ref,
    h_ref,
    A_ref,
    o_ref,
    *,
    BT,
    scale,
    USE_EXP2,
):
    """Unified GLA forward O+GK Pallas kernel.

    Block specs deliver exactly one chunk's data per grid point.
    No varlen logic, no K/V tiling, no pl.ds needed.

    Grid: (H, total_NT) where total_NT = B * NT.
    Refs (after block spec indexing):
      q_ref: (1, 1, BT, K)   g_ref: (1, 1, BT, K)
      v_ref: (1, 1, BT, V)   A_ref: (1, 1, BT, BT)
      h_ref: (1, 1, K, V)    o_ref: (1, 1, BT, V)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_g = g_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, V)
    b_h = h_ref[0, 0]  # (K, V)
    b_A = A_ref[0, 0]  # (BT, BT)

    # Inter-chunk: scale * (q * exp(g)) @ h
    b_g_f32 = b_g.astype(jnp.float32)
    if USE_EXP2:
        b_qg = (b_q * jnp.exp2(b_g_f32)).astype(b_q.dtype)
    else:
        b_qg = (b_q * jnp.exp(b_g_f32)).astype(b_q.dtype)
    b_o = jnp.dot(
        b_qg,
        b_h.astype(b_qg.dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_o *= scale

    # Intra-chunk: tril(A) @ v
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(m_s, b_A, 0.0).astype(b_A.dtype)
    b_o += jnp.dot(
        b_A,
        b_v,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0] = b_o.astype(o_ref.dtype)


def chunk_gla_fwd_o_gk_pl(
    q: jax.Array,  # [B, T, H, K]
    v: jax.Array,  # [B, T, H, V]
    g: jax.Array,  # [B, T, H, K]
    A: jax.Array,  # [B, T, H, BT]
    h: jax.Array,  # [B, NT, H, K, V]
    scale: float,
    chunk_size: int,
    use_exp2: bool,
) -> jax.Array:
    """Unified launcher for chunk_gla_fwd_o_gk Pallas kernel.

    Pre-reshapes all inputs to (H, total_NT, ...) so the kernel is
    completely agnostic to batch/varlen structure.
    Works for both varlen (B=1, cu_seqlens aligned to BT) and non-varlen.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    # Reshape: [B, T, H, K] -> [B, NT, BT, H, K] -> [H, B*NT, BT, K]
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _A = (
        A.reshape(B, NT, BT, H, BT)
        .transpose(3, 0, 1, 2, 4)
        .reshape(H, total_NT, BT, BT)
    )
    # h: [B, NT, H, K, V] -> [H, B*NT, K, V]
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    # Block specs — grid = (H, total_NT)
    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    grid = (H, total_NT)
    o = pl.pallas_call(
        functools.partial(
            chunk_gla_fwd_o_gk_pl_kernel,
            BT=BT,
            scale=scale,
            USE_EXP2=use_exp2,
        ),
        grid=grid,
        out_shape=o_shape,
        in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
    )(_q, _v, _g, _h, _A)

    # Post-process: (H, total_NT, BT, V) -> (B, T, H, V)
    # total_NT = B * NT
    o = o.reshape(H, B, NT, BT, V)
    o = o.transpose(1, 0, 2, 3, 4)  # (B, H, NT, BT, V)
    o = o.reshape(B, H, NT * BT, V)  # (B, H, T, V)
    o = o.transpose(0, 2, 1, 3)  # (B, T, H, V)

    return o


def chunk_gla_fwd_o_gk(
    q: jax.Array,  # [B, T, H, K]
    v: jax.Array,  # [B, T, H, V]
    g: jax.Array,  # [B, T, H, K]
    A: jax.Array,  # [B, T, H, BT]
    h: jax.Array,  # [B, NT, H, K, V]
    scale: float,
    cu_seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    use_exp2: bool = False,
) -> jax.Array:
    """Dispatch chunk_gla_fwd_o_gk to the unified Pallas kernel.

    Both varlen and non-varlen take the same path — the launcher
    reshapes inputs so the kernel sees (H, total_NT, ...).
    """
    B, T, H, K = q.shape
    assert T % chunk_size == 0

    return chunk_gla_fwd_o_gk_pl(
        q,
        v,
        g,
        A,
        h,
        scale,
        chunk_size,
        use_exp2,
    )


# =============================================================================
# Pallas kernel: chunk_gla_bwd_fused
# =============================================================================

def chunk_gla_bwd_fused_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, a_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref, dg_ref,
    *,
    BT: int,
    scale: float,
):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)
    b_h = h_ref[0, 0].astype(jnp.float32)
    b_a = a_ref[0, 0].astype(jnp.float32)  # [BT, BT] — true intra-chunk attention
    b_do = do_ref[0, 0]
    b_dh = dh_ref[0, 0].astype(jnp.float32)

    b_gn = b_g[BT - 1, :]

    # 1. dA = do @ v^T * scale  (gradient of loss w.r.t. A; used for dq, dk)
    b_A_do = b_do.astype(b_v.dtype)
    b_dA = jnp.dot(b_A_do, b_v.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA, 0.0)

    # 2. dv — intra uses true A (not dA): dv[j] = sum_{i>=j} A[i,j] * do[i]
    b_a_masked = jnp.where(mask, b_a, 0.0)
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn[None, :] - b_g)).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    b_dv = b_dv_intra + b_dv_inter
    dv_ref[0, 0] = b_dv.astype(dv_ref.dtype)

    # 3. dq
    k_neg = (b_k * jnp.exp(-b_g)).astype(b_k.dtype)
    b_dq_intra = jnp.dot(b_dA.astype(k_neg.dtype), k_neg, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) * jnp.exp(b_g)
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g))
    b_dq = b_dq_intra + b_dq_inter
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # 4. dk
    q_pos = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_dk_intra = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) * jnp.exp(-b_g)
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) * jnp.exp(b_gn[None, :] - b_g)
    b_dk = b_dk_intra + b_dk_inter
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

    # 5. dg
    dgk_inter = jnp.exp(b_gn) * jnp.sum(b_h * b_dh, axis=1) + jnp.sum(b_dk_inter * b_k.astype(jnp.float32), axis=0)
    dg_raw = b_q.astype(jnp.float32) * b_dq - b_k.astype(jnp.float32) * b_dk

    # Use upper triangular matrix multiplication for reverse cumsum
    mask_upper = jnp.arange(BT)[None, :] >= jnp.arange(BT)[:, None]
    M_upper = jnp.where(mask_upper, 1.0, 0.0).astype(jnp.float32)
    dg_rev_cumsum = jnp.dot(M_upper, dg_raw, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    b_dg = dg_rev_cumsum + dgk_inter[None, :]
    dg_ref[0, 0] = b_dg.astype(dg_ref.dtype)


def chunk_gla_bwd_fused_pl(
    q: jax.Array,  # [B, T, H, K]
    k: jax.Array,  # [B, T, H, K]
    v: jax.Array,  # [B, T, H, V]
    g: jax.Array,  # [B, T, H, K]
    h: jax.Array,  # [B, NT, H, K, V]
    do: jax.Array, # [B, T, H, V]
    dh: jax.Array, # [B, NT, H, K, V]
    scale: float,
    chunk_size: int,
    A: jax.Array | None = None,  # [B, T, H, chunk_size] — intra-chunk attention from fwd
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    if A is None:
        A = chunk_gla_fwd_intra_gk_ref(q, k, g, scale, chunk_size=BT)

    # Reshape
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    # A: [B, T, H, BT] -> [B, NT, BT, H, BT] -> [H, total_NT, BT, BT]
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)

    # BlockSpecs
    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

    dq_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype)
    dk_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype)
    dv_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
    dg_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], g.dtype)

    dq, dk, dv, dg = pl.pallas_call(
        functools.partial(chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[dq_shape, dk_shape, dv_shape, dg_shape],
        in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_A, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V, spec_K],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024, # 32 MB limit to be safe
        ),
    )(_q, _k, _v, _g, _h, _A, _do, _dh)

    # Post-process
    def _unreshape(x, shape):
        x = x.reshape(H, B, NT, BT, shape[-1])
        x = x.transpose(1, 0, 2, 3, 4)
        x = x.reshape(B, H, T, shape[-1])
        return x.transpose(0, 2, 1, 3)

    dq = _unreshape(dq, [B, T, H, K])
    dk = _unreshape(dk, [B, T, H, K])
    dv = _unreshape(dv, [B, T, H, V])
    dg = _unreshape(dg, [B, T, H, K])

    return dq, dk, dv, dg
