"""Shared chunk_fwd_o and chunk_local_cumsum for CPU reference kernels.

chunk_local_cumsum: works for both 3D [B,T,H] (Simple GLA) and 4D [B,T,H,K] (GLA).
chunk_fwd_o: fused inter+intra output computation for Simple GLA (scalar g / g_gamma).
  GLA uses separate intra (chunk_gla_fwd_intra_gk) and output (chunk_gla_fwd_o_gk)
  functions due to its per-element K-dim gating, so chunk_fwd_o is Simple GLA only.
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops.common.utils import acc_dtype, cdiv, dot, pad_to_multiple, read_chunk, write_chunk
from tops.utils import prepare_chunk_indices


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
  Two layouts are supported:
    - Packed layout: g.shape[0] == 1, g is [1, T, H, ...] — iterates per
      segment using cu_seqlens boundaries, pads each segment to chunk_size
      multiples internally.
    - Chunked layout: g.shape[0] > 1, g is [total_NT, C, ...] — each chunk
      is independent (legacy path for GLA gather+flatten).

  Args:
      g: [B, T, H] or [B, T, H, K] or [total_NT, C, ...] — log-space gates
      chunk_size: block size
      reverse: if True, flip chunk axis before cumsum and flip back after
      cu_seqlens: [N+1] cumulative sequence lengths defining segment
          boundaries. If provided, cumsum is applied independently per
          segment.

  Returns:
      g_cumsum: same shape as g, in fp32 (or fp64 for fp64 input)
  """
  if cu_seqlens is not None:
    acc = acc_dtype(g.dtype)

    if g.shape[0] == 1:
      # Packed layout: [1, T, H, ...] — iterate per segment
      N = len(cu_seqlens) - 1
      C = chunk_size
      result = jnp.zeros_like(g, dtype=acc)
      for i_n in range(N):
        bos = int(cu_seqlens[i_n])
        eos = int(cu_seqlens[i_n + 1])
        T_seg = eos - bos
        if T_seg == 0:
          continue
        NT_seg = cdiv(T_seg, C)

        seg_g = g[0, bos:eos].astype(acc)                    # [T_seg, ...]
        # Pad to multiple of C
        T_padded = NT_seg * C
        if T_padded > T_seg:
          seg_g = pad_to_multiple(seg_g, C, axis=0)
        # Reshape to chunks: [NT_seg, C, ...]
        shape = [NT_seg, C] + list(seg_g.shape[1:])
        seg_g = seg_g.reshape(shape)
        if reverse:
          seg_g = jnp.flip(seg_g, axis=1)
        seg_g = seg_g.cumsum(axis=1)
        if reverse:
          seg_g = jnp.flip(seg_g, axis=1)
        # Flatten and trim to T_seg
        flat = seg_g.reshape(-1, *g.shape[2:])[:T_seg]
        result = result.at[0, bos:eos].set(flat)
      return result
    else:
      # Chunked layout: [total_NT, C, ...] — each chunk independent (GLA path)
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
  cu_seqlens: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Fused inter-chunk + intra-chunk output computation (Simple GLA).

  Matches FLA chunk_fwd_kernel_o with USE_G and USE_G_GAMMA.
  In this kernel, USE_G_GAMMA overwrites b_g (unlike chunk_fwd_h where
  USE_G overwrites). So when both are active, each gate applies its own
  transformation independently — which is correct multiplicative composition.

  When cu_seqlens is provided, iterates over global chunk indices from
  prepare_chunk_indices, reading q/k/v/g from packed tensors via read_chunk
  and writing output via write_chunk. Matches Triton chunk_fwd_kernel_o.

  Args:
      q: [B, T, H, K] — queries (input dtype). Packed layout when varlen.
      k: [B, T, H, K] — keys (input dtype). Packed layout when varlen.
      v: [B, T, H, V] — values (input dtype). Packed layout when varlen.
      h: [B, total_NT, H, K, V] — hidden states (total_NT chunks across
          all sequences when varlen, else T//chunk_size)
      g: [B, T, H] — cumsummed scalar gates (fp32). Optional.
          Packed layout when varlen.
      g_gamma: [H] — fixed per-head log-decay. Optional.
      scale: scaling factor
      chunk_size: block size
      cu_seqlens: [N+1] cumulative sequence lengths for variable-length
          packing. When provided, B must be 1.

  Returns:
      o: [B, T, H, V] — v.dtype. Packed layout when varlen.
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  acc = acc_dtype(q.dtype)

  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))

  # =====================================================================
  # Varlen path: iterate over global chunk indices
  # =====================================================================
  if cu_seqlens is not None:
    assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    chunk_indices = prepare_chunk_indices(cu_seqlens, C)
    total_NT = chunk_indices.shape[0]

    o_buf = jnp.zeros((1, T, H, V), dtype=v.dtype)

    for i_tg in range(total_NT):
      i_n = int(chunk_indices[i_tg, 0])
      i_t = int(chunk_indices[i_tg, 1])
      bos = int(cu_seqlens[i_n])
      eos = int(cu_seqlens[i_n + 1])
      T_seq = eos - bos
      valid_len = min(C, T_seq - i_t * C)

      start = bos + i_t * C
      b_q = read_chunk(q, start, valid_len, C)  # [1, C, H, K]
      b_k = read_chunk(k, start, valid_len, C)  # [1, C, H, K]
      b_v = read_chunk(v, start, valid_len, C)  # [1, C, H, V]
      b_h = h[:, i_tg]  # [1, H, K, V]

      # Inter: q @ h
      b_o = dot("bchk,bhkv->bchv", b_q, b_h.astype(q.dtype), acc)

      # Intra: q @ k.T -> attention matrix
      b_A = dot("bchk,bjhk->bchj", b_q, b_k, acc)

      # --- USE_G ---
      if g is not None:
        b_g = read_chunk(g, start, valid_len, C)  # [1, C, H]
        b_o = b_o * jnp.exp(b_g[..., None])
        b_g_key = jnp.transpose(b_g, (0, 2, 1))  # [1, H, C]
        b_A = b_A * jnp.exp(b_g[:, :, :, None] - b_g_key[:, None, :, :])

      # --- USE_G_GAMMA ---
      if g_gamma is not None:
        b_g_gamma = g_gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]
        b_o = b_o * jnp.exp(b_g_gamma[None, :, :, None])
        b_g_gamma_key = b_g_gamma.T  # [H, C]
        b_A = b_A * jnp.exp(b_g_gamma[None, :, :, None] - b_g_gamma_key[None, None, :, :])

      b_A = jnp.where(causal_mask[None, :, None, :], b_A, 0)
      # Keep b_A in fp32 for precision; upcast b_v instead.
      b_o = b_o * scale + dot("bihj,bjhv->bihv", b_A, b_v.astype(jnp.float32), acc) * scale

      o_buf = write_chunk(o_buf, b_o.astype(v.dtype), start, valid_len)

    return o_buf

  # =====================================================================
  # Non-varlen path: standard uniform-chunk iteration
  # =====================================================================
  NT = T // C

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  v_c = v.reshape(B, NT, C, H, V)
  if g is not None:
    g_c = g.reshape(B, NT, C, H)

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
    # Keep b_A in fp32 for precision; upcast b_v instead.
    b_o = b_o * scale + dot("bihj,bjhv->bihv", b_A, b_v.astype(jnp.float32), acc) * scale

    o_chunks.append(b_o.astype(v.dtype))

  o = jnp.stack(o_chunks, axis=1).reshape(B, T, H, V)
  return o
