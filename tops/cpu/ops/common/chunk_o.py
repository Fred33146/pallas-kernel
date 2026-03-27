"""Shared chunk_fwd_o and chunk_local_cumsum for CPU reference kernels.

chunk_local_cumsum: works for both 3D [B,T,H] (Simple GLA) and 4D [B,T,H,K] (GLA).
chunk_fwd_o: fused inter+intra output computation for Simple GLA (scalar g / g_gamma).
  GLA uses separate intra (chunk_gla_fwd_intra_gk) and output (chunk_gla_fwd_o_gk)
  functions due to its per-element K-dim gating, so chunk_fwd_o is Simple GLA only.
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops.common.utils import acc_dtype, dot


def chunk_local_cumsum(
  g: jnp.ndarray,
  chunk_size: int,
  reverse: bool = False,
  cu_seqlens: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Chunk-local cumulative sum of gates.

  Works for both 3D [B,T,H] (Simple GLA scalar gates) and 4D [B,T,H,K]
  (GLA per-element gates). Cumsum is along the chunk dimension (axis 2
  after reshaping to [B, NT, C, ...]).

  When cu_seqlens is provided, the cumsum is applied independently per
  segment so that it does not bleed across segment boundaries.

  Args:
      g: [B, T, H] or [B, T, H, K] — log-space gates
         (T must be a multiple of chunk_size)
      chunk_size: block size
      reverse: if True, flip chunk axis before cumsum and flip back after
      cu_seqlens: [N+1] cumulative sequence lengths defining segment
          boundaries. If provided, cumsum is applied independently per
          segment. cu_seqlens[0] must be 0 and cu_seqlens[-1] must equal T.

  Returns:
      g_cumsum: same shape as g, in fp32 (or fp64 for fp64 input)
  """
  if cu_seqlens is not None:
    # Input is chunked layout: [total_NT, C, ...] — each chunk independent
    acc = acc_dtype(g.dtype)
    g_cast = g.astype(acc)
    if reverse:
      g_cast = jnp.flip(g_cast, axis=1)
    g_cumsum = g_cast.cumsum(axis=1)
    if reverse:
      g_cumsum = jnp.flip(g_cumsum, axis=1)
    return g_cumsum

  C = chunk_size
  T = g.shape[1]
  assert T % C == 0, f"T ({T}) must be a multiple of chunk_size ({C})"
  NT = T // C
  acc = acc_dtype(g.dtype)
  g_cast = g.astype(acc)

  # Reshape: insert chunk dimension at axis 2
  # 3D [B,T,H] -> [B,NT,C,H]; 4D [B,T,H,K] -> [B,NT,C,H,K]
  shape = list(g_cast.shape)
  shape[1:2] = [NT, C]
  g_reshaped = g_cast.reshape(shape)

  if reverse:
    g_reshaped = jnp.flip(g_reshaped, axis=2)
  g_cumsum = g_reshaped.cumsum(axis=2)
  if reverse:
    g_cumsum = jnp.flip(g_cumsum, axis=2)

  return g_cumsum.reshape(g.shape).astype(acc)


def chunk_fwd_o(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  h: jnp.ndarray,
  *,
  g: jnp.ndarray | None = None,
  g_gamma: jnp.ndarray | None = None,
  scale: float = 1.0,
  chunk_size: int = 64,
) -> jnp.ndarray:
  """Fused inter-chunk + intra-chunk output computation (Simple GLA).

  Matches FLA chunk_fwd_kernel_o with USE_G and USE_G_GAMMA.
  In this kernel, USE_G_GAMMA overwrites b_g (unlike chunk_fwd_h where
  USE_G overwrites). So when both are active, each gate applies its own
  transformation independently — which is correct multiplicative composition.

  Args:
      q: [B, T, H, K] — queries (input dtype)
      k: [B, T, H, K] — keys (input dtype)
      v: [B, T, H, V] — values (input dtype)
      h: [B, NT, H, K, V] — hidden states
      g: [B, T, H] — cumsummed scalar gates (fp32). Optional.
      g_gamma: [H] — fixed per-head log-decay. Optional.
      scale: scaling factor
      chunk_size: block size

  Returns:
      o: [B, T, H, V] — v.dtype
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  NT = T // C
  acc = acc_dtype(q.dtype)

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  v_c = v.reshape(B, NT, C, H, V)
  if g is not None:
    g_c = g.reshape(B, NT, C, H)

  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))

  o_chunks = []
  for i in range(NT):
    b_q = q_c[:, i]  # [B, C, H, K]
    b_k = k_c[:, i]  # [B, C, H, K]
    b_v = v_c[:, i]  # [B, C, H, V]
    b_h = h[:, i]  # [B, H, K, V]

    # Inter: q @ h
    b_o = dot("bchk,bhkv->bchv", b_q, b_h.astype(q.dtype), acc)

    # Intra: q @ k.T -> attention matrix
    b_A = dot("bchk,bjhk->bchj", b_q, b_k, acc)

    # --- USE_G ---
    if g is not None:
      b_g = g_c[:, i]  # [B, C, H]
      b_o = b_o * jnp.exp(b_g[..., None])
      b_g_key = jnp.transpose(b_g, (0, 2, 1))  # [B, H, C]
      b_A = b_A * jnp.exp(b_g[:, :, :, None] - b_g_key[:, None, :, :])

    # --- USE_G_GAMMA (overwrites b_g in FLA, but we use separate var) ---
    if g_gamma is not None:
      b_g_gamma = g_gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]
      b_o = b_o * jnp.exp(b_g_gamma[None, :, :, None])
      b_g_gamma_key = b_g_gamma.T  # [H, C]
      b_A = b_A * jnp.exp(b_g_gamma[None, :, :, None] - b_g_gamma_key[None, None, :, :])

    b_A = jnp.where(causal_mask[None, :, None, :], b_A, 0)
    b_o = b_o * scale + dot("bihj,bjhv->bihv", b_A.astype(v.dtype), b_v, acc) * scale

    o_chunks.append(b_o.astype(v.dtype))

  o = jnp.stack(o_chunks, axis=1).reshape(B, T, H, V)
  return o
