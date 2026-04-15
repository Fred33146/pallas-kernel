import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common.utils import cdiv, pad_to_multiple
from tops.ops.utils import exp2
from tops.utils import assert_shape, next_power_of_2
from tops.cpu.ops.common.chunk_delta_h import (
  chunk_gated_delta_rule_bwd_dhu,
  chunk_gated_delta_rule_fwd_h,
)
from tops.cpu.ops.kda.chunk_intra import chunk_kda_bwd_intra
from tops.cpu.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from tops.cpu.ops.kda.wy_fast import recompute_w_u_fwd
from tops.cpu.ops.common.chunk_o import chunk_local_cumsum

DEFAULT_BT = 64
DEFAULT_BK = 64
DEFAULT_BV = 64
RCP_LN2 = 1.4426950216


@cpu_reference
def chunk_kda_bwd_wy_dqkg_fused(
  q: jax.Array,  # [B, T, H, K]
  k: jax.Array,  # [B, T, H, K]
  v: jax.Array,  # [B, T, H, V]
  v_new: jax.Array,  # [B, T, H, V]
  g: jax.Array,  # [B, T, H, K]  (log-space cumsum gate, base-2 scaled)
  beta: jax.Array,  # [B, T, H]
  A: jax.Array,  # [B, T, H, BT]  (Akk inverse)
  h: jax.Array,  # [B, NT, H, K, V]  (per-chunk hidden states)
  do: jax.Array,  # [B, T, H, V]
  dh: jax.Array,  # [B, NT, H, K, V]  (hidden state gradients)
  dv: jax.Array,  # [B, T, H, V]
  scale: float | None = None,
  chunk_size: int = DEFAULT_BT,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Fused WY backward kernel: compute dq, dk, dv, db, dg, dA.

  This mirrors chunk_kda_bwd_kernel_wy_dqkg_fused from the Triton code.

  Args:
      q:      [B, T, H, K]      query tensor.
      k:      [B, T, H, K]      key tensor.
      v:      [B, T, H, V]      original value tensor.
      v_new:  [B, T, H, V]      WY-transformed value tensor.
      g:      [B, T, H, K]      log-space cumsum gate (base-2 scaled).
      beta:   [B, T, H]         WY beta coefficients.
      A:      [B, T, H, BT]     Akk inverse matrix (chunk_size = BT).
      h:      [B*NT, H, K, V]   per-chunk hidden states.
      do:     [B, T, H, V]      output gradient.
      dh:     [B*NT, H, K, V]   hidden state gradients.
      dv:     [B, T, H, V]      value gradient.
      scale:  float              softmax scaling factor.
      chunk_size: int            chunk size (BT).

  Returns:
      dq:  [B, T, H, K]  (float32)
      dk:  [B, T, H, K]  (float32)
      dv:  [B, T, H, V]  (same dtype as v)
      db:  [B, T, H]     (float32)
      dg:  [B, T, H, K]  (float32)
      dA:  [B, T, H, BT] (float32)
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)

  # =================== input shape assertions ===================
  assert_shape(q, (B, T, H, K), "q")
  assert_shape(k, (B, T, H, K), "k")
  assert_shape(v, (B, T, H, V), "v")
  assert_shape(v_new, (B, T, H, V), "v_new")
  assert_shape(g, (B, T, H, K), "g")
  assert_shape(beta, (B, T, H), "beta")
  assert_shape(A, (B, T, H, BT), "A")
  assert_shape(h, (B, NT, H, K, V), "h")
  assert_shape(do, (B, T, H, V), "do")
  assert_shape(dh, (B, NT, H, K, V), "dh")
  assert_shape(dv, (B, T, H, V), "dv")

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  # ==============================================================
  BK = min(next_power_of_2(K), DEFAULT_BK)
  BV = min(next_power_of_2(V), DEFAULT_BV)

  def _per_batch_head(
    q_bh, k_bh, v_bh, v_new_bh, g_bh, beta_bh, A_bh, h_bh, do_bh, dh_bh, dv_bh, i_bh
  ):
    """
    Process one (batch, head) pair.
    q_bh, k_bh, g_bh: [T, K]
    v_bh, v_new_bh, do_bh, dv_bh: [T, V]
    beta_bh: [T]
    A_bh: [T, BT]
    h_bh: [NT, K, V]   (per-chunk states for this batch-head)
    dh_bh: [NT, K, V]
    """
    T_actual = q_bh.shape[0]
    T_padded = NT * BT

    # Pad to multiple of BT
    q_bh = pad_to_multiple(q_bh, T_padded, 0)
    k_bh = pad_to_multiple(k_bh, T_padded, 0)
    v_bh = pad_to_multiple(v_bh, T_padded, 0)
    v_new_bh = pad_to_multiple(v_new_bh, T_padded, 0)
    g_bh = pad_to_multiple(g_bh, T_padded, 0)
    beta_bh = pad_to_multiple(beta_bh, T_padded, 0)
    A_bh = pad_to_multiple(A_bh, T_padded, 0)
    do_bh = pad_to_multiple(do_bh, T_padded, 0)
    dv_bh = pad_to_multiple(dv_bh, T_padded, 0)

    # Reshape into chunks
    q_c = q_bh.reshape(NT, BT, K)
    k_c = k_bh.reshape(NT, BT, K)
    v_c = v_bh.reshape(NT, BT, V)
    v_new_c = v_new_bh.reshape(NT, BT, V)
    g_c = g_bh.reshape(NT, BT, K)
    beta_c = beta_bh.reshape(NT, BT)
    A_c = A_bh.reshape(NT, BT, BT)
    do_c = do_bh.reshape(NT, BT, V)
    dv_c = dv_bh.reshape(NT, BT, V)

    def process_chunk(
      i_t, q_t, k_t, v_t, v_new_t, g_t, beta_t, A_t, h_t, do_t, dh_t, dv_t
    ):
      """Process a single chunk. All shapes are per-chunk."""
      o_t = jnp.arange(BT)
      # The last valid token index in this chunk
      # For simplicity, assume full chunks (T divisible by BT)
      last_idx = BT - 1

      b_beta = beta_t  # [BT]

      # Load A (the Akk-inverse matrix, lower-triangular structure)
      b_A = A_t  # [BT, BT]

      b_dA = jnp.zeros([BT, BT], dtype=jnp.float32)
      b_db = jnp.zeros([BT], dtype=jnp.float32)

      # Allocate outputs per K-block
      dq_out = jnp.zeros([BT, K], dtype=jnp.float32)
      dk_out = jnp.zeros([BT, K], dtype=jnp.float32)
      dg_out = jnp.zeros([BT, K], dtype=jnp.float32)
      dv2_out = jnp.zeros([BT, V], dtype=v_t.dtype)

      def k_block_body(i_k, carry):
        dq_acc, dk_acc, dg_acc, dv2_acc, dA_acc, db_acc = carry
        k_start = i_k * BK

        b_k = lax.dynamic_slice(k_t, (0, k_start), (BT, BK))  # [BT, BK]
        b_g = lax.dynamic_slice(g_t, (0, k_start), (BT, BK)).astype(jnp.float32)

        # Gate value at last position in chunk
        b_gn = lax.dynamic_slice(g_t, (last_idx, k_start), (1, BK))[0].astype(
          jnp.float32
        )  # [BK]

        b_dq = jnp.zeros([BT, BK], dtype=jnp.float32)
        b_dk = jnp.zeros([BT, BK], dtype=jnp.float32)
        b_dw = jnp.zeros([BT, BK], dtype=jnp.float32)
        b_dgk = jnp.zeros([BK], dtype=jnp.float32)

        def v_block_body(i_v, inner_carry):
          b_dq_i, b_dk_i, b_dw_i, b_dgk_i, dA_i, db_i, dv2_i = inner_carry
          v_start = i_v * BV

          b_v_new = lax.dynamic_slice(v_new_t, (0, v_start), (BT, BV))
          b_do_blk = lax.dynamic_slice(do_t, (0, v_start), (BT, BV))
          # h_t is [K, V]; Triton loads as transposed [V, K].
          # We load [BK, BV] and transpose in dot products.
          b_h = lax.dynamic_slice(h_t, (k_start, v_start), (BK, BV))  # [BK, BV]
          b_dh = lax.dynamic_slice(dh_t, (k_start, v_start), (BK, BV))  # [BK, BV]
          b_dv_blk = lax.dynamic_slice(dv_t, (0, v_start), (BT, BV))

          # Accumulate dgk: sum(h * dh) along V dim
          b_dgk_i = b_dgk_i + jnp.sum(b_h * b_dh, axis=1)  # [BK]

          # dq += do @ h^T  (Triton: dot(do, h_transposed))
          b_dq_i = b_dq_i + jnp.dot(
            b_do_blk.astype(jnp.float32), b_h.astype(jnp.float32).T
          )  # [BT, BK]
          # dk += v_new @ dh^T
          b_dk_i = b_dk_i + jnp.dot(
            b_v_new.astype(jnp.float32), b_dh.astype(jnp.float32).T
          )  # [BT, BK]
          # dw += dv @ h^T
          b_dw_i = b_dw_i + jnp.dot(
            b_dv_blk.astype(jnp.float32), b_h.astype(jnp.float32).T
          )  # [BT, BK]

          # Only for i_k == 0: compute dA from dv and v, dv2, db
          def do_v_path(_):
            b_v_orig = lax.dynamic_slice(v_t, (0, v_start), (BT, BV))

            # dA += dv @ v^T
            dA_new = dA_i + jnp.dot(
              b_dv_blk.astype(jnp.float32), b_v_orig.astype(jnp.float32).T
            )  # [BT, BT]

            # dv2 = A^T @ dv * beta (Triton loads A transposed)
            b_dvb = jnp.dot(
              b_A.T.astype(jnp.float32), b_dv_blk.astype(jnp.float32)
            )  # [BT, BV]
            b_dv2 = (b_dvb * b_beta[:, None]).astype(v_t.dtype)

            # db += sum(dvb * v, axis=-1)
            db_new = db_i + jnp.sum(b_dvb * b_v_orig.astype(jnp.float32), axis=1)

            dv2_new = lax.dynamic_update_slice(dv2_i, b_dv2, (0, v_start))
            return dA_new, db_new, dv2_new

          def skip_v_path(_):
            return dA_i, db_i, dv2_i

          dA_i, db_i, dv2_i = lax.cond(i_k == 0, do_v_path, skip_v_path, None)

          return b_dq_i, b_dk_i, b_dw_i, b_dgk_i, dA_i, db_i, dv2_i

        num_vb = cdiv(V, BV)
        (b_dq, b_dk, b_dw, b_dgk, dA_acc, db_acc, dv2_acc) = lax.fori_loop(
          0, num_vb, v_block_body, (b_dq, b_dk, b_dw, b_dgk, dA_acc, db_acc, dv2_acc)
        )

        # Apply gate exponentials
        b_gk_exp = exp2(b_g)  # [BT, BK]
        b_gb = b_gk_exp * b_beta[:, None]  # [BT, BK]
        b_dgk = b_dgk * exp2(b_gn)  # [BK]
        b_dq = b_dq * b_gk_exp * scale  # [BT, BK]
        b_dk = b_dk * exp2(b_gn[None, :] - b_g)  # [BT, BK]

        b_kg = b_k.astype(jnp.float32) * b_gk_exp  # [BT, BK]

        # dA contribution from dw (negative)
        b_dw_neg = -b_dw
        dA_acc = dA_acc + jnp.dot(b_dw_neg.astype(b_A.dtype), b_kg.astype(b_A.dtype).T)

        # db contribution from dkgb (A^T @ dw_neg)
        b_dkgb = jnp.dot(b_A.T.astype(jnp.float32), b_dw_neg.astype(jnp.float32))
        db_acc = db_acc + jnp.sum(b_dkgb * b_kg, axis=1)

        # Load q for dg computation
        b_q = lax.dynamic_slice(q_t, (0, k_start), (BT, BK))

        b_kdk = b_k.astype(jnp.float32) * b_dk
        b_dgk_sum = b_dgk + jnp.sum(b_kdk, axis=0)  # [BK]

        # dg for this K-block
        m_last = o_t == last_idx
        b_dg_block = (
          b_q.astype(jnp.float32) * b_dq
          - b_kdk
          + m_last[:, None] * b_dgk_sum[None, :]
          + b_kg * b_dkgb * b_beta[:, None]
        )

        # dk final: add dkgb * gb
        b_dk = b_dk + b_dkgb * b_gb

        # Accumulate into full-K output
        dq_acc = lax.dynamic_update_slice(dq_acc, b_dq, (0, k_start))
        dk_acc = lax.dynamic_update_slice(dk_acc, b_dk, (0, k_start))
        dg_acc = lax.dynamic_update_slice(dg_acc, b_dg_block, (0, k_start))

        return dq_acc, dk_acc, dg_acc, dv2_acc, dA_acc, db_acc

      num_kb = cdiv(K, BK)
      (dq_out, dk_out, dg_out, dv2_out, b_dA, b_db) = lax.fori_loop(
        0, num_kb, k_block_body, (dq_out, dk_out, dg_out, dv2_out, b_dA, b_db)
      )

      # Compute matrix-inverse gradient for dA (Akk inverse)
      # dA_masked -> dA @ A -> A @ dA -> negate
      m_t_valid = o_t < BT
      m_A_lower = (
        (o_t[:, None] > o_t[None, :]) & m_t_valid[:, None] & m_t_valid[None, :]
      )

      b_dA = jnp.where(m_A_lower, b_dA * b_beta[None, :], 0.0)
      b_dA = jnp.dot(b_dA.astype(b_A.dtype), b_A.T)
      b_dA = jnp.dot(b_A.T, b_dA.astype(b_A.dtype))
      b_dA = jnp.where(m_A_lower, -b_dA, 0.0)

      return dq_out, dk_out, dv2_out, b_db, dg_out, b_dA

    # Compute per-batch-head index for h/dh indexing
    # For fixed-length: chunk i_t of batch-head i_bh -> h index = (i_b * NT + i_t)
    # h_bh is already [NT, K, V] for this batch-head

    def chunk_fn(i_t):
      return process_chunk(
        i_t,
        q_c[i_t],
        k_c[i_t],
        v_c[i_t],
        v_new_c[i_t],
        g_c[i_t],
        beta_c[i_t],
        A_c[i_t],
        h_bh[i_t],
        do_c[i_t],
        dh_bh[i_t],
        dv_c[i_t],
      )

    # vmap over chunks
    dq_chunks, dk_chunks, dv2_chunks, db_chunks, dg_chunks, dA_chunks = jax.vmap(
      lambda i: chunk_fn(i)
    )(jnp.arange(NT))

    # Reshape back
    dq_out = dq_chunks.reshape(T_padded, K)[:T_actual]
    dk_out = dk_chunks.reshape(T_padded, K)[:T_actual]
    dv2_out = dv2_chunks.reshape(T_padded, V)[:T_actual]
    db_out = db_chunks.reshape(T_padded)[:T_actual]
    dg_out = dg_chunks.reshape(T_padded, K)[:T_actual]
    dA_out = dA_chunks.reshape(T_padded, BT)[:T_actual]

    return dq_out, dk_out, dv2_out, db_out, dg_out, dA_out

  # Reshape for vmap over (batch, head)
  # [B, T, H, D] -> [B*H, T, D] via transpose + reshape
  def _to_bh(x, d):
    return x.transpose(0, 2, 1, 3).reshape(B * H, T, d)

  q_flat = _to_bh(q, K)
  k_flat = _to_bh(k, K)
  v_flat = _to_bh(v, V)
  v_new_flat = _to_bh(v_new, V)
  g_flat = _to_bh(g, K)
  beta_flat = beta.transpose(0, 2, 1).reshape(B * H, T)  # [B*H, T]
  A_flat = _to_bh(A, BT)
  do_flat = _to_bh(do, V)
  dv_flat = _to_bh(dv, V)

  # h, dh: [NT_total, H, K, V]
  # For fixed-length: NT_total = B * NT, reshape to [B*H, NT, K, V]
  # h is stored as [B*NT, H, K, V] in the original code
  # Rearrange: [B*NT, H, K, V] -> [B, NT, H, K, V] -> [B, H, NT, K, V] -> [B*H, NT, K, V]
  h_reshaped = (
    h.reshape(B, NT, H, K, V).transpose(0, 2, 1, 3, 4).reshape(B * H, NT, K, V)
  )
  dh_reshaped = (
    dh.reshape(B, NT, H, K, V).transpose(0, 2, 1, 3, 4).reshape(B * H, NT, K, V)
  )

  i_bh_indices = jnp.arange(B * H)

  results = jax.vmap(_per_batch_head)(
    q_flat,
    k_flat,
    v_flat,
    v_new_flat,
    g_flat,
    beta_flat,
    A_flat,
    h_reshaped,
    do_flat,
    dh_reshaped,
    dv_flat,
    i_bh_indices,
  )
  dq_flat, dk_flat, dv2_flat, db_flat, dg_flat, dA_flat = results

  # Reshape back to [B, T, H, D]
  def _from_bh(x, d):
    return x.reshape(B, H, T, d).transpose(0, 2, 1, 3)

  dq = _from_bh(dq_flat, K)
  dk = _from_bh(dk_flat, K)
  dv2 = _from_bh(dv2_flat, V)
  db = db_flat.reshape(B, H, T).transpose(0, 2, 1)  # [B, T, H]
  dg = _from_bh(dg_flat, K)
  dA = _from_bh(dA_flat, BT)

  return dq, dk, dv2, db, dg, dA


# =============================================================================
# Stage 1: chunk_kda_bwd_dAv
# =============================================================================


@cpu_reference
def chunk_kda_bwd_dAv(
    q: jax.Array,         # [B, T, H, K]
    k: jax.Array,         # [B, T, H, K]
    v: jax.Array,         # [B, T, H, V]
    do: jax.Array,        # [B, T, H, V]
    A: jax.Array | None = None,         # [B, T, H, BT]
    scale: float | None = None,
    chunk_size: int = DEFAULT_BT,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute dA (attention gradient) and dv (value gradient) for the backward pass.

    This kernel computes the first stage of the KDA backward pass, computing
    the gradient of the attention matrix A and the value gradient dv from the
    output gradient do.

    Args:
        q:     [B, T, H, K]   query tensor.
        k:     [B, T, H, K]   key tensor.
        v:     [B, T, H, V]   value tensor (v_new in the full backward).
        do:    [B, T, H, V]   output gradient.
        A:     [B, T, H, BT]  attention matrix (Aqk).
        scale: float           softmax scaling factor.
        chunk_size: int        chunk size (BT).

    Returns:
        dA: [B, T, H, BT]  attention gradient (float32).
        dv: [B, T, H, V]   value gradient (same dtype as do).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = cdiv(T, BT)
    BV = min(next_power_of_2(V), DEFAULT_BV)

    # =================== input shape assertions ===================
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(do, (B, T, H, V), "do")
    assert_shape(A, (B, T, H, BT), "A")
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    # ==============================================================

    # Reshape to [B*H, T, D] for easier per-head processing
    # The Triton kernel uses (bos*H + i_h)*D strides, equivalent to [B, T, H, D]
    # We process each (batch, head) pair independently

    def _per_batch_head(q_bh, k_bh, v_bh, do_bh, A_bh):
        """Process one (batch, head) pair. Shapes: [T, D]."""
        # Pad T to multiple of BT
        T_actual = q_bh.shape[0]
        T_padded = NT * BT

        q_bh = pad_to_multiple(q_bh, T_padded, axis=0)
        k_bh = pad_to_multiple(k_bh, T_padded, axis=0)
        v_bh = pad_to_multiple(v_bh, T_padded, axis=0)
        do_bh = pad_to_multiple(do_bh, T_padded, axis=0)
        A_bh = pad_to_multiple(A_bh, T_padded, axis=0)

        # Reshape into chunks: [NT, BT, D]
        q_chunks = q_bh.reshape(NT, BT, K)
        k_chunks = k_bh.reshape(NT, BT, K)
        v_chunks = v_bh.reshape(NT, BT, V)
        do_chunks = do_bh.reshape(NT, BT, V)
        A_chunks = A_bh.reshape(NT, BT, BT)

        # Process each chunk with vmap over the NT dimension
        def process_chunk(q_c, k_c, v_c, A_c, do_c):
            dv_c = jnp.zeros([BT, V], dtype=do.dtype)
            dA_c = jnp.zeros([BT, BT], dtype=jnp.float32)

            o_t = jnp.arange(BT)
            # Causal mask for A: keep lower-tri + diagonal (i >= j)
            # Triton loads A transposed, so its upper-tri mask on A^T = lower-tri on A
            m_A = (o_t[:, None] >= o_t[None, :])
            b_A = jnp.where(m_A, A_c, 0.0).astype(do_c.dtype)

            # Loop over V blocks
            def v_block_body(i_v, carry):
                dA_acc, dv_acc = carry
                v_start = i_v * BV
                # Slice V block
                b_v = lax.dynamic_slice(v_c, (0, v_start), (BT, BV))
                b_do = lax.dynamic_slice(do_c, (0, v_start), (BT, BV))

                # dA += do @ v^T
                dA_acc = dA_acc + jnp.dot(b_do.astype(jnp.float32), b_v.astype(jnp.float32).T)

                # dv = A^T @ do (Triton loads A transposed, so dot(A_T, do);
                # here A is not transposed, so we use A.T)
                b_dv = jnp.dot(b_A.T, b_do)
                dv_acc = lax.dynamic_update_slice(dv_acc, b_dv, (0, v_start))
                return dA_acc, dv_acc

            num_vb = cdiv(V, BV)
            dA_c, dv_c = lax.fori_loop(0, num_vb, v_block_body, (dA_c, dv_c))

            # Apply causal mask and scale to dA
            m_causal = (o_t[:, None] >= o_t[None, :])
            dA_c = jnp.where(m_causal, dA_c * scale, 0.0)

            return dA_c, dv_c

        dA_chunks, dv_chunks = jax.vmap(process_chunk)(
            q_chunks, k_chunks, v_chunks, A_chunks, do_chunks
        )
        # Reshape back: [NT, BT, ...] -> [T_padded, ...]
        dA_out = dA_chunks.reshape(T_padded, BT)[:T_actual]
        dv_out = dv_chunks.reshape(T_padded, V)[:T_actual]
        return dA_out, dv_out

    # vmap over batch and head: reshape [B, T, H, D] -> [B*H, T, D]
    q_flat = q.transpose(0, 2, 1, 3).reshape(B * H, T, K)      # [B*H, T, K]
    k_flat = k.transpose(0, 2, 1, 3).reshape(B * H, T, K)
    v_flat = v.transpose(0, 2, 1, 3).reshape(B * H, T, V)
    do_flat = do.transpose(0, 2, 1, 3).reshape(B * H, T, V)
    A_flat = A.transpose(0, 2, 1, 3).reshape(B * H, T, BT)

    dA_flat, dv_flat = jax.vmap(_per_batch_head)(q_flat, k_flat, v_flat, do_flat, A_flat)

    # Reshape back to [B, T, H, ...]
    dA = dA_flat.reshape(B, H, T, BT).transpose(0, 2, 1, 3)    # [B, T, H, BT]
    dv = dv_flat.reshape(B, H, T, V).transpose(0, 2, 1, 3)     # [B, T, H, V]

    return dA, dv

# =============================================================================
# Full backward orchestrator
# =============================================================================



def chunk_kda_bwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  beta: jax.Array,
  Aqk: jax.Array,
  Akk: jax.Array,
  scale: float,
  initial_state: jax.Array | None,
  do: jax.Array,
  dht: jax.Array | None,
  *,
  g: jax.Array | None = None,
  g_org: jax.Array | None = None,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  safe_gate: bool = False,
  lower_bound: float | None = None,
  use_gate_in_kernel: bool = False,
  A_log: jax.Array | None = None,
  dt_bias: jax.Array | None = None,
  disable_recompute: bool = False,
  cp_context=None,
  transpose_state_layout: bool = False,
  **kwargs,
) -> tuple[
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array | None,
  jax.Array | None,
]:
  """Full chunk KDA backward, composed of 6 sub-functions matching Triton.

  Stage 0: recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h
  Stage 1: chunk_kda_bwd_dAv -> dAqk, dv
  Stage 2: chunk_gated_delta_rule_bwd_dhu -> dh, dh0, dv
  Stage 3: chunk_kda_bwd_wy_dqkg_fused -> dq, dk, dv, db, dg, dAkk
  Stage 4: chunk_kda_bwd_intra -> refine dq, dk, db, dg
  Stage 5: reverse cumsum on dg + optional kda_gate_bwd

  Args:
      q:     [B, T, H, K]
      k:     [B, T, H, K]
      v:     [B, T, H, V]
      beta:  [B, T, H]
      Aqk:   [B, T, H, BT] — intra-chunk attention matrix (from fwd)
      Akk:   [B, T, H, BT] — Akk inverse matrix (from fwd)
      scale: query scale
      initial_state: [B, H, K, V] or None
      do:    [B, T, H, V] — output gradient
      dht:   [B, H, K, V] or None — final state gradient
      g:     [B, T, H, K] — post gate+cumsum in log2 space (optional)
      g_org: [B, T, H, K] — raw gate input (for gate bwd)
      cu_seqlens: not implemented
      chunk_indices: not implemented
      chunk_size: block size
      safe_gate: not implemented
      lower_bound: not implemented
      use_gate_in_kernel: whether to run gate backward
      A_log: [H] — gate parameter
      dt_bias: [H*K] — gate bias
      disable_recompute: not used for CPU ref
      cp_context: not implemented
      transpose_state_layout: not implemented

  Returns:
      dq:    [B, T, H, K]
      dk:    [B, T, H, K]
      dv:    [B, T, H, V]
      db:    [B, T, H]
      dg:    [B, T, H, K]
      dh0:   [B, H, K, V] or None
      dA:    [H] or None
      dbias: [H*K] or None
  """
  del cu_seqlens, chunk_indices, safe_gate, cp_context
  del transpose_state_layout

  # ---- Stage 0: Recompute forward intermediates ----
  if disable_recompute:
    w = kwargs["w"]
    u = kwargs["u"]
    qg = kwargs["qg"]
    kg = kwargs["kg"]
    v_new = kwargs["v_new"]
    h = kwargs["h"]
  else:
    if use_gate_in_kernel:
      assert g_org is not None
      assert A_log is not None
      g = kda_gate_chunk_cumsum(
        g_org, A_log,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        dt_bias=dt_bias, lower_bound=lower_bound,
      )

    assert g is not None, "g must be provided if not disabled recompute"
    w, u, qg, kg = recompute_w_u_fwd(k, v, beta, Akk, q=q, gk=g)

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
      kg, w, u,
      gk=g, initial_state=initial_state, output_final_state=False,
      chunk_size=chunk_size, use_exp2=True,
    )

  # ---- Stage 1: dAqk and initial dv ----
  dAqk, dv_init = chunk_kda_bwd_dAv(
    q, k, v_new, do, A=Aqk, scale=scale, chunk_size=chunk_size,
  )

  # ---- Stage 2: dh, dh0, dv (reverse recurrence) ----
  dh, dh0, dv_stage2 = chunk_gated_delta_rule_bwd_dhu(
    qg, kg, w, do, dv_init,
    gk=g, h0=initial_state, dht=dht, scale=scale, chunk_size=chunk_size,
    use_exp2=True,
  )

  # ---- Stage 3: Fused WY backward ----
  dq, dk, dv_out, db, dg_raw, dAkk = chunk_kda_bwd_wy_dqkg_fused(
    q, k, v, v_new, g, beta, A=Akk, h=h, do=do, dh=dh, dv=dv_stage2,
    scale=scale, chunk_size=chunk_size,
  )

  # ---- Stage 4: Intra-chunk backward ----
  dq, dk, db, dg_intra = chunk_kda_bwd_intra(
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg_raw, chunk_size=chunk_size,
  )

  # ---- Stage 5: Reverse cumsum on dg ----
  # ---- Stage 6: Gate backward (optional) ----
  dA_log, dbias = None, None
  dg_final = chunk_local_cumsum(dg_intra, chunk_size, reverse=True)
  if use_gate_in_kernel and g_org is not None:
    dg_final, dA_log, dbias = kda_gate_bwd(
      g_org, A_log, dt_bias=dt_bias, dyg=dg_final, lower_bound=lower_bound,
    )

  return dq, dk, dv_out, db, dg_final, dh0, dA_log, dbias


__all__ = [
  "chunk_kda_bwd",
  "chunk_kda_bwd_dAv",
  "chunk_kda_bwd_wy_dqkg_fused",
]
