import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu
from src.utils import prepare_chunk_indices
from src.ops.utils import is_tpu_runtime
from src.ops.common.chunk_h import chunk_fwd_h_kernel


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================


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

    i_n, local_i_t = chunk_indices_ref[i_t, 0], chunk_indices_ref[i_t, 1]

    bos, eos = cu_seqlens_ref[i_n], cu_seqlens_ref[i_n + 1]

    start_t, start_s = bos + local_i_t * BT, i_s * BS

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

    if IS_VARLEN:
        o = (o * valid_mask).astype(o_ref.dtype)
    else:
        o = o.astype(o_ref.dtype)
    o_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)] = o


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
    BS = min(128, S)
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
    if cu_seqlens is None:
        cu_seqlens = jnp.arange(0, B * T + 1, BT, dtype=jnp.int32)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)
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
    for i_n in range(N):
        if cu_seqlens_cpu is None:
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
            h_all[i_n, i_t] = h.astype(h_all.dtype)
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]
            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= jnp.exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * jnp.exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            h = h + jnp.einsum("chk,chv->hkv", b_k, b_v)
        if output_final_state:
            ht[i_n] = h.astype(ht.dtype)

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

    q_gated = q_c * jnp.exp(g_c)
    k_gated = k_c * jnp.exp(-g_c)

    # [B, NT, H, C, K] @ [B, NT, H, K, C] -> [B, NT, H, C, C] -> [B, NT, C, H, C]
    A = jnp.einsum("bnihk,bnjhk->bnihj", q_gated, k_gated) * scale
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
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    g_cumsum: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Chunk GLA forward orchestrator.

    Pads inputs to a multiple of chunk_size, then calls the 4 sub-functions.

    Returns:
        (g_cumsum, A, h, ht, o)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = (T + C - 1) // C
    T_padded = NT * C

    if T_padded > T:
        pad = T_padded - T
        pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, pad_width)
        g = jnp.pad(g, pad_width)

    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum_vector(g, C, cu_seqlens=cu_seqlens)

    h, ht = chunk_fwd_h_kernel(
        k,
        v,
        gk=g_cumsum,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
        interpret=not is_tpu_runtime(),
    )
    if cu_seqlens is None:
        h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])
    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    o = chunk_gla_fwd_o_gk(
        q, v, g_cumsum, A, h, scale, chunk_size=C, cu_seqlens_cpu=cu_seqlens
    )

    o = o[:, :T]
    return g_cumsum, A, h, ht, o


# =============================================================================
# Public API: chunk_gla
# =============================================================================


def chunk_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
    """Chunked GLA — pure JAX implementation.

    Splits the sequence into blocks of chunk_size and computes in parallel
    within each block, propagating hidden states across blocks.
    Mathematically equivalent to naive_recurrent_gla.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — gates (log-space, after logsigmoid)
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
    q, k, v, g = (x.astype(jnp.float32) for x in (q, k, v, g))
    B, T, H, K = q.shape

    if scale is None:
        scale = K**-0.5

    _, _, _, ht, o = chunk_gla_fwd(
        q,
        k,
        v,
        g,
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
