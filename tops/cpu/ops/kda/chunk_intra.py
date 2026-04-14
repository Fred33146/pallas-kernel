"""JAX CPU reference for KDA intra-chunk backward.

Mirrors FLA's ``chunk_kda_bwd_kernel_intra`` from ``fla.ops.kda.chunk_intra``.
Processes diagonal and off-diagonal BC*BC sub-blocks within each C-sized chunk,
propagating gradients from dAqk and dAkk into dq, dk, dg, dbeta.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops.common.utils import acc_dtype


def chunk_kda_bwd_intra(
  q: jax.Array,
  k: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  dAqk: jax.Array,
  dAkk: jax.Array,
  dq: jax.Array | None = None,
  dk: jax.Array | None = None,
  db: jax.Array | None = None,
  dg: jax.Array | None = None,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  safe_gate: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Stage 4: Intra-chunk backward through sub-chunk triangular system.

  For each chunk, processes NC sub-chunks of size BC. For each pair (i, j):
    - Off-diagonal (j < i) for dAqk:
        dq[i] += dAqk[i,j] @ k[j] * exp2(g[i]-g[j])
        dk[j] += dAqk[i,j]^T @ q[i] * exp2(g[i]-g[j])
    - Off-diagonal (j < i) for dAkk:
        dk[i] += dAkk[i,j] @ k[j] * beta[j] * exp2(g_diff)
        dk[j] += dAkk[i,j]^T @ k[i] * beta[i] * exp2(g_diff)
    - Diagonal blocks handle within sub-chunk interactions

  Args:
      q:     [B, T, H, K]
      k:     [B, T, H, K]
      g:     [B, T, H, K] — log-space cumsum gate (base-2 scaled).
      beta:  [B, T, H]
      dAqk:  [B, T, H, BT] — gradient of Aqk attention matrix
      dAkk:  [B, T, H, BT] — gradient of Akk matrix
      dq:    [B, T, H, K] — incoming dq (from stage 3)
      dk:    [B, T, H, K] — incoming dk (from stage 3)
      db:    [B, T, H]   — incoming db (from stage 3)
      dg:    [B, T, H, K] — incoming dg (from stage 3)
      cu_seqlens: not implemented
      chunk_indices: not implemented
      chunk_size: block size (default 64)
      safe_gate: not implemented

  Returns:
      dq:  [B, T, H, K] (accumulated with intra contributions)
      dk:  [B, T, H, K] (accumulated)
      db:  [B, T, H] (accumulated)
      dg:  [B, T, H, K] (accumulated)
  """
  del cu_seqlens, chunk_indices, safe_gate

  B, T, H, K = q.shape
  BT = chunk_size
  BC = min(16, BT)
  NT = T // BT
  NC = BT // BC
  acc = acc_dtype(q.dtype)

  # Default incoming gradients to zeros if not provided (standalone mode)
  if dq is None:
    dq = jnp.zeros_like(q)
  if dk is None:
    dk = jnp.zeros_like(k)
  if db is None:
    db = jnp.zeros((B, T, H), dtype=q.dtype)
  if dg is None:
    dg = jnp.zeros_like(q)

  # Reshape to [B, NT, BT, H, K/1]
  q_c = q.reshape(B, NT, BT, H, K).astype(acc)
  k_c = k.reshape(B, NT, BT, H, K).astype(acc)
  g_c = g.reshape(B, NT, BT, H, K).astype(acc)
  beta_c = beta.reshape(B, NT, BT, H).astype(acc)
  dAqk_c = dAqk.reshape(B, NT, BT, H, BT).astype(acc)
  dAkk_c = dAkk.reshape(B, NT, BT, H, BT).astype(acc)

  dq_out = jnp.zeros((B, NT, BT, H, K), dtype=acc)
  dk_out = jnp.zeros((B, NT, BT, H, K), dtype=acc)
  # Track Akk row-direction dk separately for correct dg sign.
  # In exp2(g[i]-g[j]), g[i] is positive for the row (i looks at j<i),
  # but negative for the column (j is looked at by i>j).
  # FLA splits these as dk2 (row/Phase1) vs dkt (col/Phase2).
  dk_qside = jnp.zeros((B, NT, BT, H, K), dtype=acc)
  db_out = jnp.zeros((B, NT, BT, H), dtype=acc)
  dg_out = jnp.zeros((B, NT, BT, H, K), dtype=acc)

  for n in range(NT):
    for i_sc in range(NC):
      i_start = i_sc * BC
      i_end = i_start + BC

      # Sub-chunk slices
      q_i = q_c[:, n, i_start:i_end]  # [B, BC, H, K]
      k_i = k_c[:, n, i_start:i_end]
      g_i = g_c[:, n, i_start:i_end]
      beta_i = beta_c[:, n, i_start:i_end]  # [B, BC, H]

      # --- Diagonal block (i_sc == j_sc): within sub-chunk ---
      dAqk_diag = dAqk_c[:, n, i_start:i_end, :, i_start:i_end]

      # Pairwise gate difference: exp(g_i[a] - g_i[b])
      exp_g_diff_diag = jnp.exp2(
        g_i[:, :, None, :, :] - g_i[:, None, :, :, :]
      )  # [B, BC, BC, H, K]
      causal_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.bool_))

      # dq from dAqk diagonal
      k_i_exp = k_i[:, None, :, :, :] * exp_g_diff_diag
      k_i_exp = jnp.where(causal_bc[:, :, None, None], k_i_exp, 0.0)
      dq_diag = jnp.einsum("bihj,bijhk->bihk", dAqk_diag, k_i_exp)
      dq_out = dq_out.at[:, n, i_start:i_end].add(dq_diag)

      # dk from dAqk diagonal
      q_i_exp = q_i[:, :, None, :, :] * exp_g_diff_diag
      q_i_exp = jnp.where(causal_bc[:, :, None, None], q_i_exp, 0.0)
      dk_diag_qk = jnp.einsum("bihj,bijhk->bjhk", dAqk_diag, q_i_exp)
      dk_out = dk_out.at[:, n, i_start:i_end].add(dk_diag_qk)

      # dAkk diagonal
      dAkk_diag = dAkk_c[:, n, i_start:i_end, :, i_start:i_end]
      strict_lower_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.bool_), k=-1)

      # dk_row from dAkk diagonal: NO beta (applied after db)
      k_exp_row = k_i[:, None, :, :, :] * exp_g_diff_diag
      k_exp_row = jnp.where(strict_lower_bc[:, :, None, None], k_exp_row, 0.0)
      dk_kk_row = jnp.einsum("bihj,bijhk->bihk", dAkk_diag, k_exp_row)

      # db from dAkk diagonal: db[i] = sum(dk_row[i] * k[i]) before beta
      db_diag = jnp.sum(dk_kk_row * k_i, axis=-1)  # [B, BC, H]
      db_out = db_out.at[:, n, i_start:i_end].add(db_diag)

      # Now apply beta[i] to dk_row
      dk_kk_row = dk_kk_row * beta_i[:, :, :, None]
      dk_out = dk_out.at[:, n, i_start:i_end].add(dk_kk_row)
      dk_qside = dk_qside.at[:, n, i_start:i_end].add(dk_kk_row)

      # dk_col from dAkk diagonal: beta[row] applied inline
      # gate: exp2(g[row] - g[col]) where row > col (strict lower)
      k_beta_exp_col = k_i[:, :, None, :, :] * beta_i[:, :, None, :, None] * jnp.exp2(
        g_i[:, :, None, :, :] - g_i[:, None, :, :, :]
      )
      k_beta_exp_col = jnp.where(strict_lower_bc[:, :, None, None], k_beta_exp_col, 0.0)
      dk_kk_col = jnp.einsum("bihj,bijhk->bjhk", dAkk_diag, k_beta_exp_col)
      dk_out = dk_out.at[:, n, i_start:i_end].add(dk_kk_col)

      # --- Off-diagonal blocks (j_sc < i_sc) ---
      for j_sc in range(i_sc):
        j_start = j_sc * BC
        j_end = j_start + BC

        k_j = k_c[:, n, j_start:j_end]
        g_j = g_c[:, n, j_start:j_end]
        beta_j = beta_c[:, n, j_start:j_end]

        exp_g_ij = jnp.exp2(g_i[:, :, None, :, :] - g_j[:, None, :, :, :])

        # dAqk off-diagonal
        dAqk_off = dAqk_c[:, n, i_start:i_end, :, j_start:j_end]

        # dq[i] += dAqk[i,j] @ k[j] * exp(g[i]-g[j])
        k_j_exp = k_j[:, None, :, :, :] * exp_g_ij
        dq_off = jnp.einsum("bihj,bijhk->bihk", dAqk_off, k_j_exp)
        dq_out = dq_out.at[:, n, i_start:i_end].add(dq_off)

        # dk[j] += dAqk[i,j]^T @ q[i] * exp(g[i]-g[j])
        q_i_exp_off = q_i[:, :, None, :, :] * exp_g_ij
        dk_off = jnp.einsum("bihj,bijhk->bjhk", dAqk_off, q_i_exp_off)
        dk_out = dk_out.at[:, n, j_start:j_end].add(dk_off)

        # dAkk off-diagonal
        dAkk_off = dAkk_c[:, n, i_start:i_end, :, j_start:j_end]

        # dk_row[i] from dAkk: NO beta (applied after db)
        k_j_exp = k_j[:, None, :, :, :] * exp_g_ij
        dk_kk_row_off = jnp.einsum("bihj,bijhk->bihk", dAkk_off, k_j_exp)

        # db[i] from off-diagonal: db[i] = sum(dk_row[i] * k[i]) before beta
        db_off = jnp.sum(dk_kk_row_off * k_i, axis=-1)  # [B, BC_i, H]
        db_out = db_out.at[:, n, i_start:i_end].add(db_off)

        # dk[i] = dk_row * beta[i]
        dk_kk_row_off = dk_kk_row_off * beta_i[:, :, :, None]
        dk_out = dk_out.at[:, n, i_start:i_end].add(dk_kk_row_off)
        dk_qside = dk_qside.at[:, n, i_start:i_end].add(dk_kk_row_off)

        # dk_col[j] from dAkk: beta[i] (row index) applied inline
        k_i_beta_exp_off = k_i[:, :, None, :, :] * beta_i[:, :, None, :, None] * exp_g_ij
        dk_kk_j = jnp.einsum("bihj,bijhk->bjhk", dAkk_off, k_i_beta_exp_off)
        dk_out = dk_out.at[:, n, j_start:j_end].add(dk_kk_j)

    # dg: gate gradient from exp2(g_i - g_j).
    # Row direction (i looks at j<i): g_i has positive sign → +k*dk_qside
    # Col direction (j looked at by i>j): g_j has negative sign → -k*dk_tside
    # dk_tside = dk_out - dk_qside, so:
    # dg = q*dq + k*(dk_qside - dk_tside) = q*dq + k*(2*dk_qside - dk_out)
    dg_chunk = (
      q_c[:, n] * dq_out[:, n]
      + k_c[:, n] * (2 * dk_qside[:, n] - dk_out[:, n])
    )
    dg_out = dg_out.at[:, n].set(dg_chunk)

  # Add to incoming gradients
  dq_final = dq.astype(acc) + dq_out.reshape(B, T, H, K)
  dk_final = dk.astype(acc) + dk_out.reshape(B, T, H, K)
  db_final = db.astype(acc) + db_out.reshape(B, T, H)
  dg_final = dg.astype(acc) + dg_out.reshape(B, T, H, K)

  return (
    dq_final.astype(dq.dtype),
    dk_final.astype(dk.dtype),
    db_final.astype(db.dtype),
    dg_final.astype(dg.dtype),
  )


__all__ = ["chunk_kda_bwd_intra"]
