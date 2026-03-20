# pallas-kernel/tops/ops/simple_gla/chunk.py
import functools

import jax
import jax.experimental.pallas as pl
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from tops.utils import pad_to_multiple
from tops.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_fwd_h_ref
from tops.ops.common.chunk_o import chunk_fwd_o
from tops.ops.gla.chunk import chunk_gla_bwd
from tops.ops.utils import is_tpu_runtime


# =============================================================================
# Reference implementations (pure JAX, no Pallas)
# =============================================================================


def chunk_simple_gla_fwd_intra_ref(
    q: jax.Array,
    k: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Intra-chunk attention for Simple GLA (reference, pure JAX).

    Uses standard matmul + Toeplitz decay mask instead of per-K-dim gating.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head
        scale: scaling factor
        chunk_size: block size

    Returns:
        A: [B, T, H, C] — intra-chunk attention matrix
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)

    # Standard attention (no per-element gating)
    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_c, k_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ) * scale

    # Toeplitz decay mask: exp(g_gamma[h] * (i - j))
    g_h = g_gamma.reshape(H)
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    # decay[i, h, j] = exp(g_h[h] * (pos[i] - pos[j]))
    decay = jnp.exp(g_h[None, :, None] * (pos[:, None, None] - pos[None, None, :]))
    A = A * decay[None, None]  # broadcast over B, NT

    A = A.reshape(B, T, H, C)
    return A


def chunk_simple_gla_fwd_o_ref(
    q: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    A: jax.Array,
    h: jax.Array,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Output combination for Simple GLA (reference, pure JAX).

    Inter-chunk: q @ h * exp(g_gamma * pos) * scale
    Intra-chunk: tril(A) @ v

    Args:
        q: [B, T, H, K]
        v: [B, T, H, V]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head
        A: [B, T, H, C] — intra-chunk attention matrix
        h: [B, NT, H, K, V] — hidden state at start of each chunk
        scale: scaling factor
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = B * T // C

    q = q.reshape(-1, C, H, K)
    v = v.reshape(-1, C, H, V)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    # Inter-chunk: scale * q * exp(g_cumsum) @ h
    g_h = g_gamma.reshape(H)
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    g_exp = jnp.exp(g_h[None, :] * pos[:, None])  # (C, H)
    qg = q * g_exp[None, :, :, None]  # (NT, C, H, K) * (1, C, H, 1)

    o_inter = scale * jnp.einsum("nchk,nhkv->nchv", qg, h)

    # Intra-chunk: tril(A) @ v
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))[:, None, :]
    n_A = jnp.where(causal_mask, A, 0.0)
    o_intra = jnp.einsum("nihj,njhv->nihv", n_A, v)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


def chunk_simple_gla_fwd_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jax.Array | None, jax.Array]:
    """Full Simple GLA forward (reference, pure JAX).

    Returns:
        (ht, o) — final state and output
    """
    B, T, H, K = q.shape
    C = chunk_size

    # Pad T
    if T % C != 0:
        q, k, v = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v))

    g_gamma_1d = g_gamma.reshape(-1)
    assert g_gamma_1d.shape[0] == H

    # Stage 1: state propagation using g_gamma
    _, T_pad = q.shape[:2]
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)

    h, ht = chunk_fwd_h_ref(
        k, v, gk=g_cumsum, h0=initial_state,
        output_final_state=output_final_state, chunk_size=C,
    )

    # Stage 2: intra-chunk attention (Simple GLA)
    A = chunk_simple_gla_fwd_intra_ref(q, k, g_gamma, scale, chunk_size=C)

    # Stage 3: output (Simple GLA)
    o = chunk_simple_gla_fwd_o_ref(q, v, g_gamma, A, h, scale, chunk_size=C)

    o = o[:, :T]
    return ht, o


# =============================================================================
# Pallas kernel: chunk_simple_gla_fwd_intra
# =============================================================================


def _chunk_simple_gla_fwd_intra_kernel(
    q_ref,
    k_ref,
    g_gamma,  # [H] via SMEM
    A_ref,  # out
    *,
    BT,
    scale,
):
    """Simple GLA intra-chunk attention Pallas kernel.

    Standard matmul + Toeplitz decay mask (no per-K-dim gating).

    Grid: (H, total_NT).
    Refs (after block spec):
      q_ref/k_ref: (1, 1, BT, K)
      A_ref: (1, 1, BT, BT)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)

    # Standard matmul (no per-element gating on q, k)
    b_A = (
        jnp.dot(
            b_q,
            b_k.T,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    # Toeplitz decay mask: exp(gamma * (i - j))
    head_idx = pl.program_id(0)
    gamma = g_gamma[head_idx]
    pos = (jnp.arange(BT) + 1).astype(jnp.float32)
    decay = jnp.exp(gamma * (pos[:, None] - pos[None, :]))  # (BT, BT)
    b_A = b_A * decay

    A_ref[0, 0] = b_A.astype(A_ref.dtype)


def chunk_simple_gla_fwd_intra(
    q: jax.Array,  # [B, T, H, K]
    k: jax.Array,  # [B, T, H, K]
    g_gamma: jax.Array,  # (1, 1, H, 1) or (H,)
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Launcher for Simple GLA intra-chunk attention Pallas kernel.

    Returns:
        A: [B, T, H, BT] — intra-chunk attention matrix (float32)
    """
    B, T, H, K = q.shape
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    g_gamma_1d = g_gamma.reshape(-1)  # (H,)

    interpret = not is_tpu_runtime()

    # Reshape: [B, T, H, K] -> [H, B*NT, BT, K]
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)

    spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    A_shape = jax.ShapeDtypeStruct([H, total_NT, BT, BT], jnp.float32)

    # SMEM only available on TPU; use plain BlockSpec in interpret mode.
    if interpret:
        g_gamma_spec = pl.BlockSpec(memory_space=pltpu.ANY)
    else:
        g_gamma_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    A = pl.pallas_call(
        functools.partial(_chunk_simple_gla_fwd_intra_kernel, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=A_shape,
        in_specs=[spec, spec, g_gamma_spec],
        out_specs=A_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        interpret=interpret,
    )(_q, _k, g_gamma_1d)

    # Post-reshape: [H, total_NT, BT, BT] -> [B, T, H, BT]
    A = A.reshape(H, B, NT, BT, BT)
    A = A.transpose(1, 0, 2, 3, 4)
    A = A.reshape(B, H, NT * BT, BT)
    A = A.transpose(0, 2, 1, 3)
    return A


# =============================================================================
# Pallas kernel: chunk_simple_gla_fwd_o
# =============================================================================


def _chunk_simple_gla_fwd_o_kernel(
    q_ref,
    v_ref,
    h_ref,
    A_ref,
    g_gamma,  # [H] via SMEM
    o_ref,  # out
    *,
    BT,
    scale,
):
    """Simple GLA output combination Pallas kernel.

    Inter-chunk: q @ h * exp(g_gamma * pos) * scale
    Intra-chunk: tril(A) @ v

    Grid: (H, total_NT).
    Refs (after block spec):
      q_ref: (1, 1, BT, K)   v_ref: (1, 1, BT, V)
      h_ref: (1, 1, K, V)    A_ref: (1, 1, BT, BT)
      o_ref: (1, 1, BT, V)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, V)
    b_h = h_ref[0, 0]  # (K, V)
    b_A = A_ref[0, 0]  # (BT, BT)

    # Inter-chunk: q @ h * exp(g_cumsum) * scale
    # g_cumsum = g_gamma * [1, 2, ..., BT] — scalar per row
    head_idx = pl.program_id(0)
    gamma = g_gamma[head_idx]
    pos = (jnp.arange(BT) + 1).astype(jnp.float32)
    g_exp = jnp.exp(gamma * pos)  # (BT,)

    b_o = jnp.dot(
        b_q,
        b_h.astype(b_q.dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_o = b_o * (scale * g_exp[:, None])  # scale + gate in one multiply

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


def chunk_simple_gla_fwd_o(
    q: jax.Array,  # [B, T, H, K]
    v: jax.Array,  # [B, T, H, V]
    A: jax.Array,  # [B, T, H, BT]
    h: jax.Array,  # [B, NT, H, K, V]
    g_gamma: jax.Array,  # (1, 1, H, 1) or (H,)
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Launcher for Simple GLA output combination Pallas kernel.

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    g_gamma_1d = g_gamma.reshape(-1)  # (H,)

    interpret = not is_tpu_runtime()

    # Reshape: [B, T, H, X] -> [H, B*NT, BT, X]
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)
    # h: [B, NT, H, K, V] -> [H, B*NT, K, V]
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    o_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    # SMEM only available on TPU; use plain BlockSpec in interpret mode.
    if interpret:
        g_gamma_spec = pl.BlockSpec(memory_space=pltpu.ANY)
    else:
        g_gamma_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    o = pl.pallas_call(
        functools.partial(_chunk_simple_gla_fwd_o_kernel, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=o_shape,
        in_specs=[q_spec, v_spec, h_spec, A_spec, g_gamma_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        interpret=interpret,
    )(_q, _v, _h, _A, g_gamma_1d)

    # Post-reshape: (H, total_NT, BT, V) -> (B, T, H, V)
    o = o.reshape(H, B, NT, BT, V)
    o = o.transpose(1, 0, 2, 3, 4)
    o = o.reshape(B, H, NT * BT, V)
    o = o.transpose(0, 2, 1, 3)
    return o


# =============================================================================
# Pallas-based forward orchestrator (internal)
# =============================================================================


def chunk_simple_gla_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jax.Array | None, jax.Array]:
    """Simple GLA forward with Pallas TPU kernels.

    Uses scalar-per-head gates (no [B,T,H,K] gate tensors).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final state
        chunk_size: chunk size

    Returns:
        (ht, o) — final state [B, H, K, V] or None, output [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # --- T padding ---
    if T % C != 0:
        q, k, v = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v))

    # --- K/V padding (chunk_fwd_h_kernel requires K%128==0, V%128==0) ---
    q, k = (pad_to_multiple(x, 128, axis=3, val=0) for x in (q, k))
    v = pad_to_multiple(v, 128, axis=3, val=0)
    if initial_state is not None:
        initial_state = pad_to_multiple(initial_state, [128, 128], axis=[2, 3], val=0)

    g_gamma_1d = g_gamma.reshape(-1)
    assert g_gamma_1d.shape[0] == H

    # Stage 1: Inter-chunk state propagation (reuse existing Pallas kernel)
    h, ht = chunk_fwd_h_kernel(
        k=k,
        v=v,
        g=None,
        g_gamma=g_gamma_1d,
        gk=None,
        h0=initial_state,
        output_final_state=output_final_state,
        chunk_size=C,
        interpret=not is_tpu_runtime(),
    )
    h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])

    # Stage 2: Intra-chunk attention (Simple GLA Pallas kernel)
    A = chunk_simple_gla_fwd_intra(q, k, g_gamma, scale, chunk_size=C)

    # Stage 3: Output combination (Simple GLA Pallas kernel)
    o = chunk_simple_gla_fwd_o(q, v, A, h, g_gamma, scale, chunk_size=C)

    # --- unpadding ---
    o = o[..., :V]
    if ht is not None:
        ht = ht[..., :K, :V]
    o = o[:, :T]

    return ht, o


# =============================================================================
# Public forward (simple version using common kernels)
# =============================================================================


def chunk_simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    B, T, H, K, V = *q.shape, v.shape[-1]

    assert (B, T, H, K) == k.shape
    assert (B, T, H, V) == v.shape
    assert (g is None) or ((B, T, H) == g.shape)
    assert (g_gamma is None) or ((H,) == g_gamma.shape)
    assert (h0 is None) or ((B, H, K, V) == h0.shape)
    assert (cu_seqlens is None) or ((B + 1,) == cu_seqlens.shape)
    assert T % chunk_size == 0
    assert (cu_seqlens is None) or (cu_seqlens % chunk_size == 0).all()
    assert (K % 128 == 0) and (V % 128 == 0)

    h, ht = chunk_fwd_h_kernel(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=h0,
        output_final_state=use_ht,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens_cpu=cu_seqlens,
        chunk_size=chunk_size,
    )
    return o, ht


# =============================================================================
# Backward (delegates to existing chunk_gla_bwd)
# =============================================================================


def chunk_simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    initial_state: jax.Array | None,
    do: jax.Array,
    dht: jax.Array | None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Simple GLA backward, delegating to chunk_gla_bwd.

    Correct because full GLA with constant g across all K dims is
    mathematically identical to Simple GLA.

    Returns:
        (dq, dk, dv, dg_gamma, dh0)
    """
    dq, dk, dv, dg, dh0 = chunk_gla_bwd(
        q, k, v,
        g=None, g_gamma=g_gamma, g_cumsum=None,
        scale=scale, initial_state=initial_state,
        h=None, A=None,
        do=do, dht=dht,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, dg, dh0


# =============================================================================
# Public API
# =============================================================================


def chunk_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Chunked Simple GLA — scalar-per-head gates using Pallas TPU kernels.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head (log-space)
        scale: scaling factor (default K^{-0.5})
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final state
        chunk_size: chunk size

    Returns:
        (o, final_state) — output [B, T, H, V], state [B, H, K, V] or None
    """
    dtype = q.dtype
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    g_gamma = g_gamma.astype(jnp.float32)
    B, T, H, K = q.shape

    if scale is None:
        scale = K ** -0.5

    ht, o = chunk_simple_gla_pallas_fwd(
        q, k, v, g_gamma, scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    final_state = ht if output_final_state else None
    return o.astype(dtype), final_state
