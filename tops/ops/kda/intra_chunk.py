"""Pallas KDA Intra-Chunk Kernel (Forward + Backward).

Implements the intra-chunk computation of Kimi Delta Attention (KDA) using
JAX Pallas primitives for TPU acceleration.

The intra-chunk kernel computes within each chunk of the chunked KDA algorithm:
  - u: corrected value after triangular solve
  - w: corrected key-gate product for inter-chunk state update
  - qg, kg: gated query/key for inter-chunk computation
  - Aqk: causal query-key attention matrix
  - Akk_inv: inverse of the key-key interaction matrix

Key features:
  - Block-based forward substitution solver (block size 16)
  - g-centering factorization for numerical stability
  - segment_ids support for variable-length sequences
  - Both forward and backward passes

Tensor layout: [B, H, T, D] (batch, heads, time, head_dim).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def kda_intra_chunk_bwd_kernel(
    # Inputs (Ref)
    q_ref,
    k_ref,
    g_ref,
    beta_ref,
    segment_ids_ref,
    dAqk_ref,
    dAkk_ref,
    # Outputs (Ref)
    dq_ref,
    dk_ref,
    dg_ref,
    dbeta_ref,
    # Config
    chunk_size: int,
    head_dim: int,
    scale: float,
):
    """Pallas kernel for KDA intra-chunk backward computation."""
    dtype = q_ref.dtype
    q = q_ref[0, 0, 0]
    k = k_ref[0, 0, 0]
    g = g_ref[0, 0, 0]
    beta = beta_ref[0, 0, 0]
    segment_ids = segment_ids_ref[0, 0, 0, :, 0]

    dAqk = dAqk_ref[0, 0, 0]
    dAkk = dAkk_ref[0, 0, 0]

    # Recompute states with g-centering
    g_ref_idx = chunk_size // 2
    g_ref_val = g[g_ref_idx][None, :]
    g_centered = g.astype(jnp.float32) - g_ref_val.astype(jnp.float32)

    q_state = q * jnp.exp2(g_centered).astype(q.dtype)
    k_state_q = k * jnp.exp2(g_centered).astype(k.dtype)
    k_state_k = k * jnp.exp2(-g_centered).astype(k.dtype)

    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    causal_mask = idx[:, None] >= idx[None, :]
    causal_mask_qk = idx[:, None] >= idx[None, :]
    segment_mask = segment_ids[:, None] == segment_ids[None, :]

    mask_akk = causal_mask & segment_mask
    mask_aqk = causal_mask_qk & segment_mask

    dAqk_masked = jnp.where(mask_aqk, dAqk, 0.0) * scale
    dAkk_masked = jnp.where(mask_akk, dAkk, 0.0)

    Akk_raw = jax.lax.dot_general(
        k_state_q,
        k_state_k,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    dbeta = jnp.sum(dAkk_masked * Akk_raw, axis=1, keepdims=True).astype(
        beta.dtype
    )

    dAkk_raw = dAkk_masked * beta

    dq_state = jax.lax.dot_general(
        dAqk_masked,
        k_state_k,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    dk_state_k_1 = jax.lax.dot_general(
        dAqk_masked,
        q_state,
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    dk_state_q = jax.lax.dot_general(
        dAkk_raw,
        k_state_k,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    exp_g = jnp.exp2(g_centered).astype(dtype)
    exp_neg_g = jnp.exp2(-g_centered).astype(dtype)

    dk_state_k_2 = jax.lax.dot_general(
        dAkk_raw,
        k_state_q,
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    dk_state_k = dk_state_k_1 + dk_state_k_2
    dq = (dq_state * exp_g).astype(dtype)
    dk = (dk_state_q * exp_g + dk_state_k * exp_neg_g).astype(dtype)
    dg_c = dq_state * q_state + dk_state_q * k_state_q - dk_state_k * k_state_k

    # Handle g_ref gradient subtraction
    dg_ref_grad = -jnp.sum(dg_c, axis=0, keepdims=False)  # (D,)
    dg = dg_c

    idx_range = jnp.arange(chunk_size, dtype=jnp.int32)
    mask_ref_bool = idx_range == g_ref_idx
    mask_ref = jnp.reshape(mask_ref_bool.astype(dg.dtype), (chunk_size, 1))
    dg = dg + mask_ref * dg_ref_grad[None, :].astype(dg.dtype)

    dq_ref[0, 0, 0] = dq
    dk_ref[0, 0, 0] = dk
    dg_ref[0, 0, 0] = dg.astype(dtype)
    dbeta_ref[0, 0, 0] = dbeta


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def kda_intra_chunk_bwd(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array | None,
    dAqk: jax.Array,
    dAkk: jax.Array,
    scale: float = 1.0,
    chunk_size: int = 128,
):
    """Pallas implementation of KDA Intra-Chunk Backward Pass.

    Propagates upstream gradients dAqk and dAkk through the intra-chunk
    attention matrix construction back to q, k, g, and beta. Uses exp2 (log2
    base) for gate computation.

    This is the Pallas kernel counterpart of the CPU reference
    ``chunk_kda_bwd_intra`` in ``tops/cpu/ops/kda/chunk.py``.

    Args:
        q: (B, T, H, D) Query.
        k: (B, T, H, D) Key.
        g: (B, T, H, D) Cumulative sum of log-decay gates (log2 base).
        beta: (B, T, H) Delta-rule update coefficient.
        segment_ids: (B, T) Segment IDs for variable-length sequences, or None.
                     Tokens with different IDs will not attend to each other.
        dAqk: (B, T, H, C) Upstream gradient for Aqk (lower triangular).
        dAkk: (B, T, H, C) Upstream gradient for Akk (lower triangular,
               positive-beta convention).
        scale: Attention scale factor applied to dAqk internally.
        chunk_size: C — chunk size. T must be divisible by C.

    Returns:
        dq:    (B, T, H, D) Gradient w.r.t. q.
        dk:    (B, T, H, D) Gradient w.r.t. k.
        dg:    (B, T, H, D) Gradient w.r.t. g.
        dbeta: (B, T, H)    Gradient w.r.t. beta.
    """
    B, T, H, D = k.shape
    C = chunk_size

    assert q.ndim == 4, f"q must be 4D [B,T,H,D], got {q.ndim}D"
    assert q.shape == (B, T, H, D), f"q shape {q.shape} != k shape {k.shape}"
    assert g.shape == (B, T, H, D), f"g shape {g.shape} != ({B}, {T}, {H}, {D})"
    assert beta.shape == (B, T, H), f"beta shape {beta.shape} != ({B}, {T}, {H})"
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"

    NT = T // C

    assert dAqk.shape == (B, T, H, C), (
        f"dAqk shape {dAqk.shape} != ({B}, {T}, {H}, {C})"
    )
    assert dAkk.shape == (B, T, H, C), (
        f"dAkk shape {dAkk.shape} != ({B}, {T}, {H}, {C})"
    )
    if segment_ids is not None:
        assert segment_ids.shape == (B, T), (
            f"segment_ids shape {segment_ids.shape} != ({B}, {T})"
        )

    if segment_ids is None:
        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

    # [B, T, H, D] -> [H, B, NT, C, D] for internal kernel (H outermost)
    def _to_kernel(x):
        # [B, T, H, D] -> [H, B, T, D] -> [H, B, NT, C, D]
        return x.transpose(2, 0, 1, 3).reshape(H, B, NT, C, -1)

    q_reshaped = _to_kernel(q)
    k_reshaped = _to_kernel(k)
    g_reshaped = _to_kernel(g)
    # beta: [B, T, H] -> [H, B, T] -> [H, B, NT, C, 1]
    beta_reshaped = beta.transpose(2, 0, 1).reshape(H, B, NT, C, 1)
    # segment_ids: [B, T] -> [1, B, NT, C, 1] (broadcast over H)
    segment_ids_reshaped = segment_ids.reshape(1, B, NT, C, 1)
    # dAqk/dAkk: [B, T, H, C] -> [H, B, NT, C, C]
    dAqk_reshaped = _to_kernel(dAqk).reshape(H, B, NT, C, C)
    dAkk_reshaped = _to_kernel(dAkk).reshape(H, B, NT, C, C)

    grid = (H, B, NT)

    dq_reshaped, dk_reshaped, dg_reshaped, dbeta_reshaped = pl.pallas_call(
        functools.partial(
            kda_intra_chunk_bwd_kernel,
            chunk_size=chunk_size,
            head_dim=D,
            scale=scale,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=(H, B, NT, chunk_size, D), dtype=k.dtype
            ),  # dq
            jax.ShapeDtypeStruct(
                shape=(H, B, NT, chunk_size, D), dtype=k.dtype
            ),  # dk
            jax.ShapeDtypeStruct(
                shape=(H, B, NT, chunk_size, D), dtype=k.dtype
            ),  # dg
            jax.ShapeDtypeStruct(
                shape=(H, B, NT, chunk_size, 1), dtype=k.dtype
            ),  # dbeta
        ],
        in_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # q
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # k
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # g
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, 1),
            ),  # beta
            pl.BlockSpec(
                index_map=lambda i, j, l: (0, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, 1),
            ),  # segment_ids (broadcast over H)
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, chunk_size),
            ),  # dAqk
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, chunk_size),
            ),  # dAkk
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # dq
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # dk
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, D),
            ),  # dg
            pl.BlockSpec(
                index_map=lambda i, j, l: (i, j, l, 0, 0),
                block_shape=(1, 1, 1, chunk_size, 1),
            ),  # dbeta
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(
        q_reshaped,
        k_reshaped,
        g_reshaped,
        beta_reshaped,
        segment_ids_reshaped,
        dAqk_reshaped,
        dAkk_reshaped,
    )

    # [H, B, NT, C, D] -> [B, T, H, D]
    def _to_flat(x):
        # [H, B, NT, C, D] -> [H, B, T, D] -> [B, T, H, D]
        return x.reshape(H, B, T, -1).transpose(1, 2, 0, 3)

    return (
        _to_flat(dq_reshaped),
        _to_flat(dk_reshaped),
        _to_flat(dg_reshaped),
        dbeta_reshaped.reshape(H, B, T).transpose(1, 2, 0),
    )
