import jax


import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from tops.cpu.ops.common.utils import cdiv, pad_to_multiple
from tops.ops.utils import exp2
from tops.cpu.ops import cpu_reference
from tops.utils import assert_shape, next_power_of_2

DEFAULT_BT = 64
DEFAULT_BK = 64
DEFAULT_BV = 64


@cpu_reference
def chunk_kda_bwd_wy_dqkg_fused(
    q: jax.Array,         # [B, T, H, K]
    k: jax.Array,         # [B, T, H, K]
    v: jax.Array,         # [B, T, H, V]
    v_new: jax.Array,     # [B, T, H, V]
    g: jax.Array,         # [B, T, H, K]  (log-space cumsum gate, base-2 scaled)
    beta: jax.Array,      # [B, T, H]
    A: jax.Array,         # [B, T, H, BT]  (Akk inverse)
    h: jax.Array,         # [B, NT, H, K, V]  (per-chunk hidden states)
    do: jax.Array,        # [B, T, H, V]
    dh: jax.Array,        # [B, NT, H, K, V]  (hidden state gradients)
    dv: jax.Array,        # [B, T, H, V]
    scale: float,
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
    assert_shape(dh, (B , NT, H, K, V), "dh")
    assert_shape(dv, (B, T, H, V), "dv")

    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    # ==============================================================
    BK = min(next_power_of_2(K), DEFAULT_BK)
    BV = min(next_power_of_2(V), DEFAULT_BV)

    def _per_batch_head(q_bh, k_bh, v_bh, v_new_bh, g_bh, beta_bh, A_bh,
                        h_bh, do_bh, dh_bh, dv_bh, i_bh):
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

        def process_chunk(i_t, q_t, k_t, v_t, v_new_t, g_t, beta_t, A_t,
                          h_t, do_t, dh_t, dv_t):
            """Process a single chunk. All shapes are per-chunk."""
            o_t = jnp.arange(BT)
            # The last valid token index in this chunk
            # For simplicity, assume full chunks (T divisible by BT)
            last_idx = BT - 1

            b_beta = beta_t                       # [BT]

            # Load A (the Akk-inverse matrix, lower-triangular structure)
            b_A = A_t                             # [BT, BT]

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

                b_k = lax.dynamic_slice(k_t, (0, k_start), (BT, BK))        # [BT, BK]
                b_g = lax.dynamic_slice(g_t, (0, k_start), (BT, BK)).astype(jnp.float32)

                # Gate value at last position in chunk
                b_gn = lax.dynamic_slice(g_t, (last_idx, k_start), (1, BK))[0].astype(jnp.float32)  # [BK]

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
                    b_h = lax.dynamic_slice(h_t, (k_start, v_start), (BK, BV))    # [BK, BV]
                    b_dh = lax.dynamic_slice(dh_t, (k_start, v_start), (BK, BV))  # [BK, BV]
                    b_dv_blk = lax.dynamic_slice(dv_t, (0, v_start), (BT, BV))

                    # Accumulate dgk: sum(h * dh) along V dim 
                    b_dgk_i = b_dgk_i + jnp.sum(b_h * b_dh, axis=1)  # [BK]

                    # dq += do @ h^T  (Triton: dot(do, h_transposed))
                    b_dq_i = b_dq_i + jnp.dot(b_do_blk.astype(jnp.float32),
                                               b_h.astype(jnp.float32).T)  # [BT, BK]
                    # dk += v_new @ dh^T
                    b_dk_i = b_dk_i + jnp.dot(b_v_new.astype(jnp.float32),
                                              b_dh.astype(jnp.float32).T)  # [BT, BK]
                    # dw += dv @ h^T
                    b_dw_i = b_dw_i + jnp.dot(b_dv_blk.astype(jnp.float32),
                                              b_h.astype(jnp.float32).T)   # [BT, BK]

                    # Only for i_k == 0: compute dA from dv and v, dv2, db
                    def do_v_path(_):
                        b_v_orig = lax.dynamic_slice(v_t, (0, v_start), (BT, BV))

                        # dA += dv @ v^T
                        dA_new = dA_i + jnp.dot(b_dv_blk.astype(jnp.float32),
                                                b_v_orig.astype(jnp.float32).T) # [BT, BT]

                        # dv2 = A^T @ dv * beta (Triton loads A transposed)
                        b_dvb = jnp.dot(b_A.T.astype(jnp.float32),
                                       b_dv_blk.astype(jnp.float32))       # [BT, BV]
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
                (b_dq, b_dk, b_dw, b_dgk,
                 dA_acc, db_acc, dv2_acc) = lax.fori_loop(
                    0, num_vb, v_block_body,
                    (b_dq, b_dk, b_dw, b_dgk, dA_acc, db_acc, dv2_acc)
                )

                # Apply gate exponentials
                b_gk_exp = exp2(b_g)                                    # [BT, BK]
                b_gb = b_gk_exp * b_beta[:, None]                       # [BT, BK]
                b_dgk = b_dgk * exp2(b_gn)                             # [BK]
                b_dq = b_dq * b_gk_exp * scale                          # [BT, BK]
                b_dk = b_dk * exp2(b_gn[None, :] - b_g)                # [BT, BK]

                b_kg = b_k.astype(jnp.float32) * b_gk_exp               # [BT, BK]

                # dA contribution from dw (negative)
                b_dw_neg = -b_dw
                dA_acc = dA_acc + jnp.dot(b_dw_neg.astype(b_A.dtype),
                                          b_kg.astype(b_A.dtype).T)

                # db contribution from dkgb (A^T @ dw_neg)
                b_dkgb = jnp.dot(b_A.T.astype(jnp.float32), b_dw_neg.astype(jnp.float32))
                db_acc = db_acc + jnp.sum(b_dkgb * b_kg, axis=1)

                # Load q for dg computation
                b_q = lax.dynamic_slice(q_t, (0, k_start), (BT, BK))

                b_kdk = b_k.astype(jnp.float32) * b_dk
                b_dgk_sum = b_dgk + jnp.sum(b_kdk, axis=0)             # [BK]

                # dg for this K-block
                m_last = (o_t == last_idx)
                b_dg_block = (b_q.astype(jnp.float32) * b_dq
                              - b_kdk
                              + m_last[:, None] * b_dgk_sum[None, :]
                              + b_kg * b_dkgb * b_beta[:, None])

                # dk final: add dkgb * gb
                b_dk = b_dk + b_dkgb * b_gb

                # Accumulate into full-K output
                dq_acc = lax.dynamic_update_slice(dq_acc, b_dq, (0, k_start))
                dk_acc = lax.dynamic_update_slice(dk_acc, b_dk, (0, k_start))
                dg_acc = lax.dynamic_update_slice(dg_acc, b_dg_block, (0, k_start))

                return dq_acc, dk_acc, dg_acc, dv2_acc, dA_acc, db_acc

            num_kb = cdiv(K, BK)
            (dq_out, dk_out, dg_out, dv2_out,
             b_dA, b_db) = lax.fori_loop(
                0, num_kb, k_block_body,
                (dq_out, dk_out, dg_out, dv2_out, b_dA, b_db)
            )

            # Compute matrix-inverse gradient for dA (Akk inverse)
            # dA_masked -> dA @ A -> A @ dA -> negate
            m_t_valid = o_t < BT
            m_A_lower = ((o_t[:, None] > o_t[None, :])
                         & m_t_valid[:, None] & m_t_valid[None, :])

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
                q_c[i_t], k_c[i_t], v_c[i_t], v_new_c[i_t],
                g_c[i_t], beta_c[i_t], A_c[i_t],
                h_bh[i_t], do_c[i_t], dh_bh[i_t], dv_c[i_t],
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
    beta_flat = beta.transpose(0, 2, 1).reshape(B * H, T)       # [B*H, T]
    A_flat = _to_bh(A, BT)
    do_flat = _to_bh(do, V)
    dv_flat = _to_bh(dv, V)

    # h, dh: [NT_total, H, K, V]
    # For fixed-length: NT_total = B * NT, reshape to [B*H, NT, K, V]
    # h is stored as [B*NT, H, K, V] in the original code
    # Rearrange: [B*NT, H, K, V] -> [B, NT, H, K, V] -> [B, H, NT, K, V] -> [B*H, NT, K, V]
    h_reshaped = h.reshape(B, NT, H, K, V).transpose(0, 2, 1, 3, 4).reshape(B * H, NT, K, V)
    dh_reshaped = dh.reshape(B, NT, H, K, V).transpose(0, 2, 1, 3, 4).reshape(B * H, NT, K, V)

    i_bh_indices = jnp.arange(B * H)

    results = jax.vmap(_per_batch_head)(
        q_flat, k_flat, v_flat, v_new_flat, g_flat, beta_flat, A_flat,
        h_reshaped, do_flat, dh_reshaped, dv_flat, i_bh_indices,
    )
    dq_flat, dk_flat, dv2_flat, db_flat, dg_flat, dA_flat = results

    # Reshape back to [B, T, H, D]
    def _from_bh(x, d):
        return x.reshape(B, H, T, d).transpose(0, 2, 1, 3)

    dq = _from_bh(dq_flat, K)
    dk = _from_bh(dk_flat, K)
    dv2 = _from_bh(dv2_flat, V)
    db = db_flat.reshape(B, H, T).transpose(0, 2, 1)            # [B, T, H]
    dg = _from_bh(dg_flat, K)
    dA = _from_bh(dA_flat, BT)

    return dq, dk, dv2, db, dg, dA
