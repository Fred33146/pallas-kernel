"""KDA chunked attention: forward and backward orchestrators.

Decomposes KDA into 4 stages:
  1. Gate cumsum (gate.py)
  2. Intra-chunk triangular solve (chunk_intra.py)
  3. Inter-chunk delta-rule state propagation (common/chunk_delta_h.py)
  4. Output computation (gla/chunk.py:chunk_gla_fwd_o_gk)

The chunk_kda function wraps these with jax.custom_vjp for efficient
backward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from tops.ops.gla.chunk import chunk_gla_fwd_o_gk
from tops.ops.kda.chunk_intra import kda_intra_chunk_fwd
from tops.ops.kda.gate import kda_gate_chunk_cumsum
from tops.utils import assert_shape, assert_shape_or_none, pad_to_multiple


def chunk_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array]:
    """KDA chunked forward pass orchestrator.

    Decomposes KDA (Kimi Delta Attention) into four pipelined stages:
      1. **Gate cumsum** — convert per-element gates from natural log
         space to log2 space and compute chunk-local cumulative sums.
      2. **Intra-chunk solve** — solve the lower-triangular delta-rule
         system within each chunk via block forward substitution,
         producing delta-corrected values, correction weights, gated
         projections, and the attention / inverse matrices.
      3. **Inter-chunk state propagation** — propagate the hidden state
         across chunks using the delta-rule recurrence, applying the
         correction from each chunk's w and u.
      4. **Output computation** — combine inter-chunk (query @ state)
         and intra-chunk (attention @ value) contributions.

    Args:
        q:     [B, T, H, K] — query vectors.
        k:     [B, T, H, K] — key vectors.
        v:     [B, T, H, V] — value vectors.
        g:     [B, T, H, K] — per-element gate in natural log space.
        beta:  [B, T, H]    — per-token scalar mixing coefficient for
                               the delta-rule update.
        scale: float or None — attention scale factor. Defaults to
               K ** -0.5 when None.
        initial_state: [B, H, K, V] or None — initial hidden state
                       carried over from a previous segment.
        output_final_state: bool — whether to return the final hidden
                            state after processing all chunks.
        cu_seqlens: [N+1] or None — cumulative sequence lengths for
                    variable-length mode (not yet supported).
        chunk_size: int — tile size for chunked computation. T must be
                    divisible by chunk_size.

    Returns:
        o: [B, T, H, V]           — output tensor.
        final_state: [B, H, K, V] — final hidden state, or None if
                     output_final_state is False.
        g_cumsum: [B, T, H, K]    — chunk-local cumsum of g in log2 space.
        Aqk_flat: [B, T, H, BT]   — intra-chunk attention matrix (flattened).
        Akk_flat: [B, T, H, BT]   — Akk inverse matrix (flattened).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    # --- Input validation ---
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(g, (B, T, H, K), "g")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (B, H, K, V), "initial_state")
    assert T % chunk_size == 0, (
        f"Sequence length T={T} must be divisible by chunk_size={chunk_size}"
    )

    if scale is None:
        scale = K ** -0.5

    # ----------------------------------------------------------------
    # Step 1: Gate cumsum
    # ----------------------------------------------------------------
    # g is in natural log space → kda_gate_chunk_cumsum converts to
    # log2 space and applies chunk-local cumsum.
    # g_cumsum: [B, T, H, K]  (log2 space, float32)
    g_cumsum = kda_gate_chunk_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

    # ----------------------------------------------------------------
    # Step 2: Intra-chunk triangular solve
    # ----------------------------------------------------------------
    # kda_intra_chunk_fwd expects [B, H, T, D] layout.
    # Transpose from public [B, T, H, D] → kernel [B, H, T, D].
    q_bht = jnp.transpose(q, (0, 2, 1, 3))           # [B, H, T, K]
    k_bht = jnp.transpose(k, (0, 2, 1, 3))           # [B, H, T, K]
    g_cumsum_bht = jnp.transpose(g_cumsum, (0, 2, 1, 3))  # [B, H, T, K]
    beta_bht = jnp.transpose(beta, (0, 2, 1))         # [B, H, T]
    v_bht = jnp.transpose(v, (0, 2, 1, 3))           # [B, H, T, V]

    u_bht, w_bht, qg_bht, kg_bht, Aqk, Akk_inv = kda_intra_chunk_fwd(
        q=q_bht,
        k=k_bht,
        g=g_cumsum_bht,
        beta=beta_bht,
        v=v_bht,
        scale=scale,
        chunk_size=chunk_size,
    )
    # u_bht:  [B, H, T, V]
    # w_bht:  [B, H, T, K]
    # qg_bht: [B, H, T, K]  (not used — step 4 re-gates q internally)
    # kg_bht: [B, H, T, K]
    # Aqk:    [B, H, NC, C, C]
    # Akk_inv:[B, H, NC, C, C]

    # ----------------------------------------------------------------
    # Step 3: Inter-chunk delta-rule state propagation
    # ----------------------------------------------------------------
    # chunk_gated_delta_rule_fwd_h_ref expects [B, T, H, D] layout.
    # Transpose intra-chunk outputs back: [B, H, T, D] → [B, T, H, D].
    kg_bthk = jnp.transpose(kg_bht, (0, 2, 1, 3))   # [B, T, H, K]
    w_bthk = jnp.transpose(w_bht, (0, 2, 1, 3))     # [B, T, H, K]
    u_bthv = jnp.transpose(u_bht, (0, 2, 1, 3))     # [B, T, H, V]

    # gk is the chunk-local cumsum in log2 space, already [B, T, H, K].
    # The delta-rule kernel uses gk for state decay (per-element gate).
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg_bthk,
        w=w_bthk,
        u=u_bthv,
        gk=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    # h:           [B, NT, H, K, V]  — hidden state before each chunk
    # v_new:       [B, T, H, V]      — delta-corrected values
    # final_state: [B, H, K, V] or None

    # ----------------------------------------------------------------
    # Step 4: Output computation
    # ----------------------------------------------------------------
    # chunk_gla_fwd_o_gk expects:
    #   q:  [B, T, H, K]
    #   v:  [B, T, H, V]  — use v_new (delta-corrected)
    #   g:  [B, T, H, K]  — g_cumsum
    #   A:  [B, T, H, BT] — intra-chunk attention matrix
    #   h:  [B, NT, H, K, V]
    #
    # Reshape Aqk from [B, H, NC, C, C] → [B, T, H, C]:
    #   [B, H, NC, C, C] → [B, H, NC*C, C] → [B, T, H, C]  (transpose H and T)
    NC = T // chunk_size
    C = chunk_size
    Aqk_reshaped = Aqk.reshape(B, H, NC * C, C)       # [B, H, T, C]
    Aqk_reshaped = jnp.transpose(Aqk_reshaped, (0, 2, 1, 3))  # [B, T, H, C]

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g_cumsum,
        A=Aqk_reshaped,
        h=h,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=True,
    )
    # o: [B, T, H, V]

    # --- Flatten Aqk and Akk_inv for backward ---
    # [B, H, NC, C, C] → [B, H, T, C] → [B, T, H, C]
    Aqk_flat = Aqk.reshape(B, H, NC * C, C)
    Aqk_flat = jnp.transpose(Aqk_flat, (0, 2, 1, 3))  # [B, T, H, BT]
    Akk_flat = Akk_inv.reshape(B, H, NC * C, C)
    Akk_flat = jnp.transpose(Akk_flat, (0, 2, 1, 3))  # [B, T, H, BT]

    return o, final_state, g_cumsum, Aqk_flat, Akk_flat


@jax.custom_vjp
def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None,
              output_final_state=False, cu_seqlens=None, chunk_size=64):
    """Chunked KDA forward+backward with custom_vjp.

    Args:
        q: [B, T, H, K] — Queries.
        k: [B, T, H, K] — Keys.
        v: [B, T, H, V] — Values.
        g: [B, T, H, K] — Per-element gate in natural log space.
        beta: [B, T, H] — Learning rate for delta rule.
        scale: Attention scale. Defaults to K ** -0.5.
        initial_state: [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return final hidden state.
        cu_seqlens: [N+1] — Cumulative sequence lengths for varlen. Optional.
        chunk_size: Chunk size. T must be divisible by chunk_size.

    Returns:
        o: [B, T, H, V] — Output.
        final_state: [B, H, K, V] if output_final_state else None.
    """
    o, final_state, _, _, _ = chunk_kda_fwd(q, k, v, g, beta, scale, initial_state,
                         output_final_state, cu_seqlens, chunk_size)
    return o, final_state


def _chunk_kda_fwd_custom(q, k, v, g, beta, scale, initial_state,
                           output_final_state, cu_seqlens, chunk_size):
    o, final_state, g_cumsum, Aqk_flat, Akk_flat = chunk_kda_fwd(
        q, k, v, g, beta, scale, initial_state,
        output_final_state, cu_seqlens, chunk_size,
    )
    saved = (q, k, v, beta, g_cumsum, Aqk_flat, Akk_flat,
             scale, initial_state, chunk_size)
    return (o, final_state), saved


def _chunk_kda_bwd_custom(saved, grad_outputs):
    do, d_final_state = grad_outputs
    (q, k, v, beta, g_cumsum, Aqk, Akk,
     scale, initial_state, chunk_size) = saved

    from tops.ops.kda.chunk_bwd import chunk_kda_bwd

    dq, dk, dv, db, dg, dh0, _, _ = chunk_kda_bwd(
        q=q, k=k, v=v, beta=beta,
        Aqk=Aqk, Akk=Akk,
        scale=scale, initial_state=initial_state,
        do=do, dht=d_final_state,
        g=g_cumsum, chunk_size=chunk_size,
    )

    # dg is gradient w.r.t. g_cumsum (log2 space).
    # g_cumsum = cumsum(g / ln2), so dg_orig = dg / ln2.
    _LN2 = jnp.log(jnp.array(2.0, dtype=jnp.float32))
    dg = dg / _LN2

    # Return order matches chunk_kda args:
    # (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size)
    return dq, dk, dv, dg, db, None, dh0, None, None, None


chunk_kda.defvjp(_chunk_kda_fwd_custom, _chunk_kda_bwd_custom)
