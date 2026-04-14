"""JAX CPU reference for KDA WY representation forward/backward.
Mirrors FLA's ``fla/ops/kda/wy_fast``:
- ``recompute_w_u_fwd``: recompute effective keys w, values u, gated q/k
- ``prepare_wy_repr_bwd``: backward through WY representation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops.common.utils import acc_dtype


def recompute_w_u_fwd(
  k: jax.Array,
  v: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  q: jax.Array | None = None,
  gk: jax.Array | None = None,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
  """Recompute effective keys w, values u, and optionally gated q/k.
  For each chunk of size BT:
    u = A @ (v * beta[:, None])       — WY-transformed value
    w = A @ (k * beta[:, None] * exp2(gk))  — WY-transformed key (gated)
    qg = q * exp2(gk)                — gated query (if q provided)
    kg = k * exp2(gn - gk)           — gated key (if gk provided)
  Args:
      k:    [B, T, H, K] — keys
      v:    [B, T, H, V] — values
      beta: [B, T, H]    — learning rate
      A:    [B, T, H, BT] — Akk^{-1} matrix (lower triangular)
      q:    [B, T, H, K] — queries (optional)
      gk:   [B, T, H, K] — chunk-local cumsummed gates (log2 space, optional)
      cu_seqlens: not implemented in v1
      chunk_indices: not implemented in v1
  Returns:
      w:  [B, T, H, K]
      u:  [B, T, H, V]
      qg: [B, T, H, K] or None
      kg: [B, T, H, K] or None
  """
  del cu_seqlens, chunk_indices

  B, T, H, K = k.shape
  V = v.shape[-1]
  BT = A.shape[-1]
  NT = T // BT
  acc = acc_dtype(k.dtype)

  # Reshape to chunks
  k_c = k.reshape(B, NT, BT, H, K).astype(acc)
  v_c = v.reshape(B, NT, BT, H, V).astype(acc)
  beta_c = beta.reshape(B, NT, BT, H).astype(acc)
  A_c = A.reshape(B, NT, BT, H, BT).astype(acc)

  # u = A @ (v * beta)
  v_beta = v_c * beta_c[..., None]  # [B, NT, BT, H, V]
  # A: [B, NT, BT_i, H, BT_j], v_beta: [B, NT, BT_j, H, V]
  # u[b,n,i,h,v] = sum_j A[b,n,i,h,j] * v_beta[b,n,j,h,v]
  u = jnp.einsum("bnihj,bnjhv->bnihv", A_c, v_beta)

  # w = A @ (k * beta * exp2(gk))
  if gk is not None:
    gk_c = gk.reshape(B, NT, BT, H, K).astype(acc)
    k_beta_gated = k_c * beta_c[..., None] * jnp.exp2(gk_c)
  else:
    k_beta_gated = k_c * beta_c[..., None]
  w = jnp.einsum("bnihj,bnjhk->bnihk", A_c, k_beta_gated)

  # qg = q * exp2(gk)
  qg = None
  if q is not None and gk is not None:
    qg = (q * jnp.exp2(gk)).astype(q.dtype)
  elif q is not None:
    qg = q

  # kg = k * exp2(gn - gk)
  kg = None
  if gk is not None:
    gk_c2 = gk.reshape(B, NT, BT, H, K).astype(acc)
    gn = gk_c2[:, :, -1:, :, :]  # [B, NT, 1, H, K]
    kg = (k_c * jnp.exp2(gn - gk_c2)).reshape(B, T, H, K).astype(k.dtype)

  return (
    w.reshape(B, T, H, K).astype(k.dtype),
    u.reshape(B, T, H, V).astype(v.dtype),
    qg,
    kg,
  )


def prepare_wy_repr_bwd(
  k: jax.Array,
  v: jax.Array,
  beta: jax.Array,
  gk: jax.Array,
  A: jax.Array,
  dk: jax.Array,
  dw: jax.Array,
  du: jax.Array,
  dg: jax.Array,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward for WY representation step.
  Computes gradients from dw/du back through the WY transform,
  including the matrix inverse gradient (dA correction).
  For keys branch (w = A @ k_beta_gated):
    dA += dw @ k_beta_gated^T
    dk_bg = A^T @ dw
    dk += dk_bg * exp(gk) * beta
    db += sum(dk_bg * k * exp(gk), dim=-1)
    dg += k * beta * exp(gk) * dk_bg
  For values branch (u = A @ v_beta):
    dA += du @ v_beta^T
    dv_beta = A^T @ du
    dv = dv_beta * beta
    db += sum(dv_beta * v, dim=-1)
  Matrix inverse gradient:
    dA = tril_strict(dA)
    dA = -A^T @ dA @ A^T
  Args:
      k:    [B, T, H, K]
      v:    [B, T, H, V]
      beta: [B, T, H]
      gk:   [B, T, H, K]
      A:    [B, T, H, BT]
      dk:   [B, T, H, K] — incoming dk
      dw:   [B, T, H, K] — gradient of w
      du:   [B, T, H, V] — gradient of u
      dg:   [B, T, H, K] — incoming dg
      cu_seqlens: not implemented
      chunk_indices: not implemented
  Returns:
      dk:  [B, T, H, K] — updated
      dv:  [B, T, H, V]
      db:  [B, T, H]
      dg:  [B, T, H, K] — updated
      dA:  [B, T, H, BT]
  """
  del cu_seqlens, chunk_indices

  B, T, H, K = k.shape
  V = v.shape[-1]
  BT = A.shape[-1]
  NT = T // BT
  acc = acc_dtype(k.dtype)

  k_c = k.reshape(B, NT, BT, H, K).astype(acc)
  v_c = v.reshape(B, NT, BT, H, V).astype(acc)
  beta_c = beta.reshape(B, NT, BT, H).astype(acc)
  gk_c = gk.reshape(B, NT, BT, H, K).astype(acc)
  A_c = A.reshape(B, NT, BT, H, BT).astype(acc)
  dw_c = dw.reshape(B, NT, BT, H, K).astype(acc)
  du_c = du.reshape(B, NT, BT, H, V).astype(acc)

  exp_gk = jnp.exp2(gk_c)

  # k_beta_gated = k * beta * exp2(gk)
  k_bg = k_c * beta_c[..., None] * exp_gk

  # v_beta = v * beta
  v_beta = v_c * beta_c[..., None]

  # --- Keys branch ---
  # dA from w: dA[i,j] = sum_k dw[i,k] * k_bg[j,k]
  dA = jnp.einsum("bnihk,bnjhk->bnihj", dw_c, k_bg)

  # dk_bg = A^T @ dw: dk_bg[j] = sum_i A[i,j] * dw[i]
  dk_bg = jnp.einsum("bnihj,bnihk->bnjhk", A_c, dw_c)

  # dk += dk_bg * exp(gk) * beta
  dk_new = dk_bg * exp_gk * beta_c[..., None]

  # db from keys: sum(dk_bg * k * exp(gk), dim=-1)
  db_k = jnp.sum(dk_bg * k_c * exp_gk, axis=-1)  # [B, NT, BT, H]

  # dg from keys: k * beta * exp(gk) * dk_bg
  dg_k = k_c * beta_c[..., None] * exp_gk * dk_bg

  # --- Values branch ---
  # dA from u: dA[i,j] += sum_v du[i,v] * v_beta[j,v]
  dA = dA + jnp.einsum("bnihv,bnjhv->bnihj", du_c, v_beta)

  # dv_beta = A^T @ du
  dv_beta = jnp.einsum("bnihj,bnihv->bnjhv", A_c, du_c)

  # dv = dv_beta * beta
  dv = (dv_beta * beta_c[..., None]).reshape(B, T, H, V).astype(v.dtype)

  # db from values: sum(dv_beta * v, dim=-1)
  db_v = jnp.sum(dv_beta * v_c, axis=-1)  # [B, NT, BT, H]

  db = (db_k + db_v).reshape(B, T, H).astype(beta.dtype)

  # --- Matrix inverse gradient ---
  # Mask to strict lower triangular (diagonal excluded because A has 1s on diagonal)
  strict_lower = jnp.tril(jnp.ones((BT, BT), dtype=jnp.bool_), k=-1)
  dA = jnp.where(strict_lower[None, None, :, None, :], dA, 0.0)

  # dA_final = -A^T @ dA_raw @ A^T
  # Reshape to batch matmul: [B*NT*H, BT, BT]
  A_mat = A_c.transpose(0, 1, 3, 2, 4).reshape(-1, BT, BT)  # [B*NT*H, BT, BT]
  dA_mat = dA.transpose(0, 1, 3, 2, 4).reshape(-1, BT, BT)
  At = A_mat.transpose(0, 2, 1)  # A^T
  dA_final = -(At @ dA_mat @ At)
  dA_out = dA_final.reshape(B, NT, H, BT, BT).transpose(0, 1, 3, 2, 4)

  dk_out = (
    (dk.reshape(B, NT, BT, H, K).astype(acc) + dk_new)
    .reshape(B, T, H, K)
    .astype(dk.dtype)
  )
  dg_out = (
    (dg.reshape(B, NT, BT, H, K).astype(acc) + dg_k)
    .reshape(B, T, H, K)
    .astype(dg.dtype)
  )

  return dk_out, dv, db, dg_out, dA_out.reshape(B, T, H, BT).astype(A.dtype)


__all__ = [
  "recompute_w_u_fwd",
  "prepare_wy_repr_bwd",
]
