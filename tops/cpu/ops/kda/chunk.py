"""JAX CPU reference for KDA (Kernel Delta Attention) chunk operations.

This module implements the chunk-based KDA algorithm, directly porting
FLA's naive_chunk_kda to JAX. The algorithm processes sequences in chunks
and uses a triangular system solve for intra-chunk dependencies arising
from the delta rule.

Key algorithm steps:
  1. Reshape inputs into chunks and compute chunk-local cumsum of g
  2. Build intra-chunk interaction matrix A (key-key alignment with decay)
  3. Solve the lower-triangular dependency system iteratively
  4. Compute effective keys (w) and values (u) via the solved system
  5. Propagate inter-chunk state with delta rule correction

KDA differences from Simple GLA:
  - Gate: per-element g: [B,T,H,K] instead of per-head scalar g: [B,T,H]
  - Delta rule: beta learning rate, state update subtracts k^T S from v
  - Intra-chunk: requires solving a linear system, not just masked attention

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation: fp32 (or fp64 for fp64 inputs)
  Final output:
    o: cast back to original v.dtype               [fp64 mode: fp64]
    final_state S: fp32                             [fp64 mode: fp64]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common.utils import acc_dtype as _acc_dtype
from tops.cpu.ops.common.utils import cdiv as _cdiv
from tops.cpu.ops.common.utils import pad_to_multiple as _pad_to_multiple


# =============================================================================
# Internal forward: returns intermediates needed for backward
# =============================================================================


def chunk_kda_fwd(
  q_c: jax.Array,
  k_c: jax.Array,
  v_c: jax.Array,
  g_c: jax.Array,
  beta_c: jax.Array,
  scale: float,
  initial_state: jax.Array | None,
  output_final_state: bool,
  C: int,
  acc_dt: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Internal chunk KDA forward, returning intermediates for backward.

  All inputs are pre-processed: reshaped to [B, H, NT, C, D], cast to
  acc_dt, g already chunk-local cumsummed, q already scaled.

  Args:
      q_c:   [B, H, NT, C, K] — queries (scaled by ``scale``)
      k_c:   [B, H, NT, C, K] — keys
      v_c:   [B, H, NT, C, V] — values
      g_c:   [B, H, NT, C, K] — chunk-local cumsummed gates
      beta_c:[B, H, NT, C]    — learning rate
      scale: float
      initial_state: [B, H, K, V] or None
      output_final_state: bool
      C:     chunk size
      acc_dt: accumulator dtype

  Returns:
      o:     [B, H, NT, C, V] — output (acc_dt)
      Aqk:   [B, H, NT, C, C] — query-key attention (lower tri, with scale)
      Akk:   [B, H, NT, C, C] — (I + A_raw)^{-1}, pure inverse (no beta)
      final_state: [B, H, K, V] or None
  """
  B, H, NT = q_c.shape[:3]
  K = q_c.shape[-1]
  V = v_c.shape[-1]

  # =========================================================================
  # Step 1: Build interaction matrix A
  # =========================================================================
  mask_upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))

  A = jnp.zeros((B, H, NT, C, C), dtype=acc_dt)
  for i in range(C):
    k_i = k_c[..., i, :]
    g_i = g_c[..., i:i + 1, :]
    A = A.at[..., i].set(
      jnp.einsum('...cd,...d->...c', k_c * jnp.exp(g_c - g_i), k_i)
    )

  A = A * beta_c[..., None]
  A = jnp.where(mask_upper[None, None, None], 0, -A)

  # =========================================================================
  # Step 2: Solve lower-triangular dependency system
  # =========================================================================
  for i in range(1, C):
    update = jnp.einsum(
      '...j,...ji->...i', A[..., i, :], A[..., :, :i]
    )
    A = A.at[..., i, :i].set(A[..., i, :i] + update)

  # Akk = (I + A_raw)^{-1}, pure inverse without beta
  Akk = A + jnp.eye(C, dtype=acc_dt)

  # A with beta for computing w, u
  A_beta = Akk * beta_c[..., None, :]

  # =========================================================================
  # Step 3: Compute effective keys (w) and values (u)
  # =========================================================================
  w = jnp.einsum('...ij,...jk->...ik', A_beta, jnp.exp(g_c) * k_c)
  u = jnp.einsum('...ij,...jv->...iv', A_beta, v_c)

  # =========================================================================
  # Step 4: Inter-chunk recurrence with delta rule correction
  # =========================================================================
  S = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if initial_state is not None:
    S = S + initial_state.astype(acc_dt)
  o = jnp.zeros_like(v_c)

  mask_strict_upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=1)
  Aqk_list = []

  for i in range(NT):
    q_i = q_c[:, :, i]
    k_i = k_c[:, :, i]
    u_i = u[:, :, i]
    g_i = g_c[:, :, i]
    w_i = w[:, :, i]

    # Intra-chunk attention matrix A_qk
    A_qk = jnp.zeros((B, H, C, C), dtype=acc_dt)
    for j in range(C):
      k_j = k_c[:, :, i, j]
      g_j = g_c[:, :, i, j:j + 1, :]
      A_qk = A_qk.at[..., j].set(
        jnp.einsum('...cd,...d->...c', q_i * jnp.exp(g_i - g_j), k_j)
      )
    A_qk = jnp.where(mask_strict_upper[None, None], 0, A_qk)
    Aqk_list.append(A_qk)

    # Delta rule correction
    v_corrected = u_i - jnp.einsum('...ck,...kv->...cv', w_i, S)

    # Output
    o = o.at[:, :, i].set(
      jnp.einsum('...ck,...kv->...cv', q_i * jnp.exp(g_i), S)
      + jnp.einsum('...ij,...jv->...iv', A_qk, v_corrected)
    )

    # Update state
    g_last = g_i[:, :, -1]
    S = S * jnp.exp(g_last)[..., None]
    k_weighted = jnp.exp(g_last[:, :, None, :] - g_i) * k_i
    S = S + jnp.einsum('...ck,...cv->...kv', k_weighted, v_corrected)

  Aqk = jnp.stack(Aqk_list, axis=2)  # [B, H, NT, C, C]
  final_state = S if output_final_state else None
  return o, Aqk, Akk, final_state


# =============================================================================
# Stage 4 standalone: chunk_kda_bwd_intra
# =============================================================================

@cpu_reference
def _chunk_kda_bwd_intra(
  q_c: jax.Array,
  k_c: jax.Array,
  g_c: jax.Array,
  beta_c: jax.Array,
  dAqk: jax.Array,
  dAkk: jax.Array,
  C: int,
  acc_dt: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Internal chunked-layout implementation of bwd_intra.

  All inputs in [B, H, NT, C, D] layout. See chunk_kda_bwd_intra for details.
  """
  NT = q_c.shape[2]

  dq_intra = jnp.zeros_like(q_c)
  dk_intra = jnp.zeros_like(k_c)
  db_intra = jnp.zeros_like(beta_c)
  dg_intra = jnp.zeros_like(g_c)

  for t in range(NT):
    g_t = g_c[:, :, t]        # [B, H, C, K]
    k_t = k_c[:, :, t]        # [B, H, C, K]
    q_t = q_c[:, :, t]        # [B, H, C, K]
    b_t = beta_c[:, :, t]     # [B, H, C]
    dAqk_t = dAqk[:, :, t]    # [B, H, C, C]
    dAkk_t = dAkk[:, :, t]    # [B, H, C, C]

    dq_t2 = jnp.zeros_like(q_t)
    dk_left = jnp.zeros_like(k_t)
    dk_right = jnp.zeros_like(k_t)
    db_t2 = jnp.zeros_like(b_t)
    dg_qk_pos = jnp.zeros_like(g_t)
    dg_qk_neg = jnp.zeros_like(g_t)

    for j in range(C):
      k_j = k_t[:, :, j]            # [B, H, K]
      g_j = g_t[:, :, j:j + 1, :]   # [B, H, 1, K]
      exp_diff = jnp.exp(g_t - g_j)  # [B, H, C, K]

      # --- From dAqk ---
      dAqk_col = dAqk_t[:, :, :, j:j + 1]  # [B, H, C, 1]

      # Section 1 (row): dq[r] += dAqk[r,j] * exp(g[r]-g[j]) * k[j]
      dq_t2 = dq_t2 + dAqk_col * exp_diff * k_j[:, :, None, :]

      # Section 2 (col): dk[j] += sum_r dAqk[r,j] * q[r] * exp(g[r]-g[j])
      dk_right_j = (dAqk_col * q_t * exp_diff).sum(axis=-2)
      dk_right = dk_right.at[:, :, j].set(dk_right[:, :, j] + dk_right_j)

      # dg from Aqk: +dg[r], -dg[j]
      contrib = dAqk_col * q_t * exp_diff * k_j[:, :, None, :]
      dg_qk_pos = dg_qk_pos + contrib
      dg_qk_neg = dg_qk_neg.at[:, :, j].set(
        dg_qk_neg[:, :, j] - contrib.sum(axis=-2)
      )

      # --- From dAkk ---
      dAkk_col = dAkk_t[:, :, :, j:j + 1]  # [B, H, C, 1]

      # Section 1 (row): dk_left[r] += dAkk[r,j] * (-beta[r]) * exp(g[r]-g[j]) * k[j]
      dk_left = dk_left + dAkk_col * (-b_t[..., None]) * exp_diff * k_j[:, :, None, :]

      # Section 2 (col): dk_right[j] += sum_r dAkk[r,j] * (-beta[r]) * k[r] * exp(g[r]-g[j])
      dk_right_j_kk = (
        dAkk_col * (-b_t[..., None]) * k_t * exp_diff
      ).sum(axis=-2)
      dk_right = dk_right.at[:, :, j].set(dk_right[:, :, j] + dk_right_j_kk)

      # db: db[r] += dAkk[r,j] * (-a[r,j])
      a_rj = (k_t * exp_diff * k_j[:, :, None, :]).sum(axis=-1)  # [B,H,C]
      db_t2 = db_t2 + (dAkk_col.squeeze(-1) * (-a_rj))

      # dg from Akk: +dg[r], -dg[j]
      contrib_kk = dAkk_col * (-b_t[..., None]) * k_t * exp_diff * k_j[:, :, None, :]
      dg_qk_pos = dg_qk_pos + contrib_kk
      dg_qk_neg = dg_qk_neg.at[:, :, j].set(
        dg_qk_neg[:, :, j] - contrib_kk.sum(axis=-2)
      )

    dq_intra = dq_intra.at[:, :, t].set(dq_t2)
    dk_intra = dk_intra.at[:, :, t].set(dk_left + dk_right)
    db_intra = db_intra.at[:, :, t].set(db_t2)
    dg_intra = dg_intra.at[:, :, t].set(dg_qk_pos + dg_qk_neg)

  return dq_intra, dk_intra, db_intra, dg_intra


@cpu_reference
def chunk_kda_bwd_intra(
  q: jax.Array,
  k: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  dAqk: jax.Array,
  dAkk: jax.Array,
  chunk_size: int,
  acc_dt: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward through intra-chunk attention matrices Aqk and Akk.

  Propagates upstream gradients dAqk and dAkk (from earlier backward stages)
  through the intra-chunk attention matrix construction:

      Aqk(r, j) = q_r^T [exp(g_r - g_j) . k_j]     (lower triangular)
      Akk(r, j) = -beta_r * k_r^T [exp(g_r - g_j) . k_j]  (strictly lower tri)

  This is Stage 4 of the 6-stage chunk KDA backward pipeline.

  Args:
      q:     [B, T, H, K] — queries
      k:     [B, T, H, K] — keys
      g:     [B, T, H, K] — chunk-local cumsummed gates (natural log)
      beta:  [B, T, H]    — learning rate / step size
      dAqk:  [B, T, H, C] — upstream gradient for Aqk (lower triangular)
      dAkk:  [B, T, H, C] — upstream gradient for Akk (strictly lower triangular)
      chunk_size: C — chunk size (T must be divisible by C)
      acc_dt: accumulator dtype

  Returns:
      dq:   [B, T, H, K] — gradient w.r.t. q
      dk:   [B, T, H, K] — gradient w.r.t. k
      db:   [B, T, H]    — gradient w.r.t. beta
      dg:   [B, T, H, K] — gradient w.r.t. g (before reverse cumsum)
  """
  B, T, H, K = q.shape
  C = chunk_size

  assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
  assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
  assert g.shape == q.shape, f"g shape {g.shape} != q shape {q.shape}"
  assert beta.shape == (B, T, H), f"beta shape {beta.shape} != ({B}, {T}, {H})"
  assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"
  assert dAqk.shape == (B, T, H, C), f"dAqk shape {dAqk.shape} != ({B}, {T}, {H}, {C})"
  assert dAkk.shape == (B, T, H, C), f"dAkk shape {dAkk.shape} != ({B}, {T}, {H}, {C})"

  NT = T // C

  # [B, T, H, D] -> [B, H, NT, C, D]
  def _to_chunked(x):
    return jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, -1)

  q_c = _to_chunked(q)
  k_c = _to_chunked(k)
  g_c = _to_chunked(g)
  beta_c = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt).reshape(B, H, NT, C)
  dAqk_c = _to_chunked(dAqk).reshape(B, H, NT, C, C)
  dAkk_c = _to_chunked(dAkk).reshape(B, H, NT, C, C)

  dq_c, dk_c, db_c, dg_c = _chunk_kda_bwd_intra(
    q_c, k_c, g_c, beta_c, dAqk_c, dAkk_c, C, acc_dt,
  )

  # [B, H, NT, C, K] -> [B, T, H, K]
  def _to_flat(x):
    return jnp.transpose(x.reshape(B, H, T, -1), (0, 2, 1, 3))

  return _to_flat(dq_c), _to_flat(dk_c), \
    jnp.transpose(db_c.reshape(B, H, T), (0, 2, 1)), _to_flat(dg_c)


# =============================================================================
# Internal backward: matches FLA chunk_kda_bwd 6-stage pipeline
# =============================================================================


def chunk_kda_bwd(
  q_c: jax.Array,
  k_c: jax.Array,
  v_c: jax.Array,
  g_c: jax.Array,
  beta_c: jax.Array,
  Aqk: jax.Array,
  Akk: jax.Array,
  scale: float,
  initial_state: jax.Array | None,
  do_c: jax.Array,
  dht: jax.Array | None,
  C: int,
  acc_dt: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
           jax.Array | None]:
  """Chunk KDA backward, matching FLA's 6-stage pipeline.

  All inputs are pre-processed: reshaped to [B, H, NT, C, D], cast to
  acc_dt, g already chunk-local cumsummed. q_c does NOT have scale
  (unlike the forward where q_c = q * scale).

  Args:
      q_c:   [B, H, NT, C, K] — queries (NO scale)
      k_c:   [B, H, NT, C, K] — keys
      v_c:   [B, H, NT, C, V] — values
      g_c:   [B, H, NT, C, K] — chunk-local cumsummed gates
      beta_c:[B, H, NT, C]    — learning rate
      Aqk:   [B, H, NT, C, C] — query-key attention (lower tri, with scale)
      Akk:   [B, H, NT, C, C] — (I + A_raw)^{-1}, pure inverse (no beta)
      scale: float
      initial_state: [B, H, K, V] or None
      do_c:  [B, H, NT, C, V] — output gradient
      dht:   [B, H, K, V] or None — final state gradient
      C:     chunk size
      acc_dt: accumulator dtype

  Returns:
      dq: [B, H, NT, C, K]
      dk: [B, H, NT, C, K]
      dv: [B, H, NT, C, V]
      db: [B, H, NT, C]
      dg: [B, H, NT, C, K]
      dh0: [B, H, K, V] or None
  """
  B, H, NT = q_c.shape[:3]
  K = q_c.shape[-1]
  V = v_c.shape[-1]

  # =========================================================================
  # Stage 0: Recompute forward intermediates
  # (matches FLA: recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h)
  # =========================================================================
  A_beta = Akk * beta_c[..., None, :]
  w = jnp.einsum('...ij,...jk->...ik', A_beta, jnp.exp(g_c) * k_c)
  u = jnp.einsum('...ij,...jv->...iv', A_beta, v_c)

  # qg = q * exp(g), no scale
  qg = q_c * jnp.exp(g_c)
  # kg = k * exp(g_last - g), keys normalized to chunk end
  g_last = g_c[:, :, :, -1:, :]  # [B, H, NT, 1, K]
  kg = k_c * jnp.exp(g_last - g_c)

  # Forward scan: h[t] and v_new[t]
  S = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if initial_state is not None:
    S = S + initial_state.astype(acc_dt)
  h_list = []
  v_new = jnp.zeros_like(v_c)

  for t in range(NT):
    h_list.append(S)
    v_new_t = u[:, :, t] - jnp.einsum('...ck,...kv->...cv', w[:, :, t], S)
    v_new = v_new.at[:, :, t].set(v_new_t)
    g_C = g_c[:, :, t, -1]  # [B, H, K]
    S = S * jnp.exp(g_C)[..., None]
    S = S + jnp.einsum('...ck,...cv->...kv', kg[:, :, t], v_new_t)

  h = jnp.stack(h_list, axis=2)  # [B, H, NT, K, V]

  # =========================================================================
  # Stage 1: chunk_kda_bwd_dAv (output backward)
  # dAqk = tril(do @ v_new^T) * scale   -> gradient w.r.t. raw Aqk
  # dv = Aqk^T @ do                     -> gradient w.r.t. v_new
  # =========================================================================
  mask_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
  dAqk = jnp.zeros((B, H, NT, C, C), dtype=acc_dt)
  dv = jnp.zeros_like(v_c)

  for t in range(NT):
    # dAqk = tril(do @ v_new^T) * scale
    dA_t = jnp.einsum('...rv,...jv->...rj', do_c[:, :, t], v_new[:, :, t])
    dA_t = jnp.where(mask_lower[None, None], dA_t * scale, 0.0)
    dAqk = dAqk.at[:, :, t].set(dA_t)

    # dv = Aqk^T @ do: [j,v] = sum_r Aqk[r,j] * do[r,v]
    dv_t = jnp.einsum('...rj,...rv->...jv', Aqk[:, :, t], do_c[:, :, t])
    dv = dv.at[:, :, t].set(dv_t)

  # =========================================================================
  # Stage 2: chunk_gated_delta_rule_bwd_dhu (reverse time recurrence)
  # =========================================================================
  dh_state = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if dht is not None:
    dh_state = dh_state + dht.astype(acc_dt)

  dh_list = [None] * NT

  for t in range(NT - 1, -1, -1):
    dh_list[t] = dh_state

    # Update dv: add inter-chunk contribution dv' = kg @ dh + dv_intra
    # kg[c,k] @ dh[k,v] -> [c,v]
    dv_inter = jnp.einsum('...ck,...kv->...cv', kg[:, :, t], dh_state)
    dv = dv.at[:, :, t].set(dv[:, :, t] + dv_inter)

    # Update dh for next iteration (going to t-1)
    g_C = g_c[:, :, t, -1]  # [B, H, K]
    dh_state = dh_state * jnp.exp(g_C)[..., None]
    # Output path: scale * qg^T @ do
    dh_state = dh_state + scale * jnp.einsum(
      '...ck,...cv->...kv', qg[:, :, t], do_c[:, :, t]
    )
    # Delta rule path: -w^T @ dv'
    dh_state = dh_state - jnp.einsum(
      '...ck,...cv->...kv', w[:, :, t], dv[:, :, t]
    )

  dh = jnp.stack(dh_list, axis=2)  # [B, H, NT, K, V]
  dh0 = dh_state if initial_state is not None else None

  # =========================================================================
  # Stage 3: chunk_kda_bwd_wy_dqkg_fused
  # (WY representation backward + q/k/g gradient fusion)
  # =========================================================================
  dq = jnp.zeros_like(q_c)
  dk = jnp.zeros_like(k_c)
  dv_out = jnp.zeros_like(v_c)
  db = jnp.zeros_like(beta_c)
  dg = jnp.zeros_like(g_c)
  dAkk = jnp.zeros((B, H, NT, C, C), dtype=acc_dt)

  mask_strict_lower = jnp.tril(jnp.ones((C, C), dtype=acc_dt), k=-1)

  for t in range(NT):
    h_t = h[:, :, t]        # [B, H, K, V]
    dh_t = dh[:, :, t]      # [B, H, K, V]
    dv_t = dv[:, :, t]      # [B, H, C, V] (updated dv' from stage 2)
    g_t = g_c[:, :, t]      # [B, H, C, K]
    g_last_t = g_t[:, :, -1:]  # [B, H, 1, K]
    b_t = beta_c[:, :, t]   # [B, H, C]

    # --- V-loop accumulation (vectorized) ---
    # dq_inter = do @ h^T: [C,V] @ [V,K] -> [C,K]
    dq_t = jnp.einsum('...cv,...kv->...ck', do_c[:, :, t], h_t)
    # dk_inter = v_new @ dh^T: [C,V] @ [V,K] -> [C,K]
    dk_t = jnp.einsum('...cv,...kv->...ck', v_new[:, :, t], dh_t)
    # dw = dv' @ h^T: [C,V] @ [V,K] -> [C,K]
    dw_t = jnp.einsum('...cv,...kv->...ck', dv_t, h_t)
    # dgk = sum(h * dh, dim=V): [K]
    dgk_t = (h_t * dh_t).sum(axis=-1)  # [B, H, K]

    # --- Akk gradient from v path ---
    # dAkk_raw += dv' @ v^T: [C,V] @ [V,C] -> [C,C]
    dAkk_t = jnp.einsum('...cv,...jv->...cj', dv_t, v_c[:, :, t])

    # --- dv2 through WY: (Akk^T @ dv') * beta ---
    # Akk^T @ dv': [j,c] @ [c,v] -> [j,v] = sum_c Akk[c,j] * dv[c,v]
    Akk_T_dv = jnp.einsum('...cj,...cv->...jv', Akk[:, :, t], dv_t)
    dv2_t = Akk_T_dv * b_t[..., None]
    # db from v path: sum((Akk^T @ dv') * v, dim=V)
    db_v = (Akk_T_dv * v_c[:, :, t]).sum(axis=-1)  # [B, H, C]

    # --- Apply gating to dq, dk ---
    dq_t = dq_t * jnp.exp(g_t) * scale
    dk_t = dk_t * jnp.exp(g_last_t - g_t)

    # --- Negate dw (actual gradient: dw = -dv' @ h^T) ---
    dw_t = -dw_t

    # --- Akk gradient from w path: dw @ (k*exp(g))^T ---
    kg_exp = k_c[:, :, t] * jnp.exp(g_t)  # k * exp(g)
    dAkk_t = dAkk_t + jnp.einsum('...ck,...jk->...cj', dw_t, kg_exp)

    # --- dk from WY: Akk^T @ dw ---
    Akk_T_dw = jnp.einsum('...cj,...ck->...jk', Akk[:, :, t], dw_t)
    db_w = (Akk_T_dw * kg_exp).sum(axis=-1)  # [B, H, C]

    # --- Matrix inverse gradient ---
    # Akk = (I + A_raw)^{-1} where A_raw = -beta*a (strictly lower).
    # dL/d(I+A_raw) = -Akk^T @ dL/dAkk_pure @ Akk^T
    # dL/dA_raw = strictly_lower(dL/d(I+A_raw))
    # We store -dL/dA_raw = dL/d(beta*a) so that bwd_intra can propagate
    # to k, beta, g through the positive entries beta*a directly.
    dAkk_pure = dAkk_t * b_t[..., None, :] * mask_strict_lower[None, None]
    # Compute Akk^T @ dAkk_pure @ Akk^T
    # Step 1: dAkk_pure @ Akk^T = sum_j dAkk_pure[c,j] * Akk[m,j]
    temp = jnp.einsum('...cj,...mj->...cm', dAkk_pure, Akk[:, :, t])
    # Step 2: Akk^T @ temp = sum_c Akk[c,j] * temp[c,m]
    temp = jnp.einsum('...cj,...cm->...jm', Akk[:, :, t], temp)
    # Store dL/d(beta*a) = -dL/dA_raw = +strictly_lower(Akk^T @ G @ Akk^T)
    dAkk_final = temp * mask_strict_lower[None, None]
    dAkk = dAkk.at[:, :, t].set(dAkk_final)

    # --- dg from WY + inter-chunk ---
    k_dk = k_c[:, :, t] * dk_t
    dgk_t = dgk_t * jnp.exp(g_last_t.squeeze(-2))
    dgk_t = dgk_t + k_dk.sum(axis=-2)  # sum over C -> [B, H, K]
    # m_last: only the last position in the chunk
    m_last = jnp.zeros(C, dtype=acc_dt).at[-1].set(1.0)
    dg_t = (
      q_c[:, :, t] * dq_t
      - k_dk
      + m_last[None, None, :, None] * dgk_t[:, :, None, :]
      + kg_exp * Akk_T_dw * b_t[..., None]
    )

    # --- dk: add WY contribution ---
    dk_t = dk_t + Akk_T_dw * b_t[..., None] * jnp.exp(g_t)

    # --- Store results ---
    dq = dq.at[:, :, t].set(dq_t)
    dk = dk.at[:, :, t].set(dk_t)
    dv_out = dv_out.at[:, :, t].set(dv2_t)
    db = db.at[:, :, t].set(db_v + db_w)
    dg = dg.at[:, :, t].set(dg_t)

  # =========================================================================
  # Stage 4: chunk_kda_bwd_intra
  # (propagate dAqk and dAkk gradients to q, k, beta, g)
  # =========================================================================
  dq_intra, dk_intra, db_intra, dg_intra = _chunk_kda_bwd_intra(
    q_c, k_c, g_c, beta_c, dAqk, dAkk, C, acc_dt,
  )

  # Merge stage 3 and stage 4 results
  dq = dq + dq_intra
  dk = dk + dk_intra
  db = db + db_intra
  dg = dg + dg_intra

  # Reverse cumsum for dg (adjoint of chunk-local cumsum in forward)
  dg = jnp.cumsum(dg[:, :, :, ::-1, :], axis=3)[:, :, :, ::-1, :]

  return dq, dk, dv_out, db, dg, dh0


# =============================================================================
# Public API: chunk_kda
# =============================================================================


@cpu_reference
def chunk_kda(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
  """Chunk KDA with FLA-exact dtype behavior.

  Processes the sequence in chunks of size ``chunk_size``, solving the
  intra-chunk delta rule dependencies via a lower-triangular system.

  Algorithm overview:
    1. Split sequence into chunks, cumsum g within each chunk
    2. Build interaction matrix A[c,i] = sum_k k[c,k]*exp(g[c,k]-g[i,k])*k[i,k]
    3. Apply beta, mask upper triangle, negate, solve triangular system
    4. Compute effective updates: w = A @ (exp(g)*k), u = A @ v
    5. For each chunk: output = (q*exp(g)) @ S + A_qk @ (u - w @ S)
       then update S for next chunk

  Core recurrence (equivalent to naive):
      S' = S_{t-1} * exp(g_t)
      S_t = S' + beta_t * k_t ⊗ (v_t - k_t^T @ S')
      o_t = (q_t * scale)^T @ S_t

  Args:
      q:               [B, T, H, K] — Queries
      k:               [B, T, H, K] — Keys
      v:               [B, T, H, V] — Values
      g:               [B, T, H, K] — Per-element gate in log-space
      beta:            [B, T, H]    — Learning rate for delta rule
      scale:           Scalar query scale. Defaults to K ** -0.5.
      initial_state:   [B, H, K, V] — Initial hidden state. Optional.
      output_final_state: Whether to return the final hidden state.
      chunk_size:      Block size for chunked computation.

  Returns:
      o:           [B, T, H, V] — Output (v.dtype)
      final_state: [B, H, K, V] in fp32 (or fp64), or None
  """
  orig_dtype = v.dtype
  acc_dt = _acc_dtype(q.dtype)
  B, T_orig, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # Shape assertions (project coding standard)
  assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
  assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
  assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
    f"v shape {v.shape} incompatible with q shape {q.shape}"
  )
  assert g.ndim == 4 and g.shape == q.shape, (
    f"g shape {g.shape} != q shape {q.shape}"
  )
  assert beta.ndim == 3 and beta.shape == q.shape[:3], (
    f"beta shape {beta.shape} != {q.shape[:3]}"
  )
  if initial_state is not None:
    assert initial_state.shape == (B, H, K, V), (
      f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"
    )

  if scale is None:
    scale = K ** -0.5

  # --- Pad T to multiple of chunk_size ---
  T = T_orig
  T_padded = _cdiv(T_orig, C) * C
  if T_padded > T_orig:
    q = _pad_to_multiple(q, C, axis=1)
    k = _pad_to_multiple(k, C, axis=1)
    v = _pad_to_multiple(v, C, axis=1)
    g = _pad_to_multiple(g, C, axis=1)
    beta = _pad_to_multiple(beta, C, axis=1)
    T = T_padded

  NT = T // C

  # --- Reshape to [B, H, NT, C, D] and cast to accumulator dtype ---
  q_c = jnp.transpose(q, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K) * scale
  k_c = jnp.transpose(k, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K)
  v_c = jnp.transpose(v, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, V)
  g_c = jnp.transpose(g, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, C, K)
  beta_c = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt).reshape(B, H, NT, C)

  # --- Chunk-local cumulative sum of g ---
  g_c = g_c.cumsum(axis=3)

  # --- Forward ---
  o, Aqk, Akk, final_state = chunk_kda_fwd(
    q_c, k_c, v_c, g_c, beta_c,
    scale=scale,
    initial_state=initial_state,
    output_final_state=output_final_state,
    C=C,
    acc_dt=acc_dt,
  )

  # --- Reshape output back ---
  o = o.reshape(B, H, T, V)
  o = jnp.transpose(o, (0, 2, 1, 3))[:, :T_orig].astype(orig_dtype)

  return o, final_state
