"""JAX CPU reference for Simple GLA chunk operations with FLA-triton-exact dtype behavior.

This module contains Simple GLA-specific backward functions (chunk_bwd_dqkwg,
chunk_bwd_dv) and orchestrators. Shared functions (chunk_fwd_h, chunk_bwd_dh,
chunk_fwd_o, chunk_local_cumsum) are imported from tops.cpu.ops.common.

Key differences from GLA chunk:
- Gate shape: GLA uses gk: [B,T,H,K] (per-element). Simple GLA uses g: [B,T,H]
  (per-head scalar) and/or g_gamma: [H]
- Gate target: GLA gates k in chunk_fwd_h. Simple GLA gates v in chunk_fwd_h
- g and g_gamma are mutually exclusive in backward (if/elif)

fp64 mode: When inputs are fp64, all precision casts are skipped (they become
no-ops) and accumulation uses fp64 throughout. This provides a high-precision
reference that exceeds Triton's fp32 accumulation.

Dtype contract (matching FLA Triton for bf16/fp16/fp32; all fp64 for fp64):
  Forward:
    g_cumsum: fp32 (chunk_local_cumsum output)     [fp64 mode: fp64]
    h:        k.dtype if states_in_fp32=False, else fp32  [fp64 mode: fp64]
    o:        v.dtype                               [fp64 mode: fp64]
    ht:       fp32 (final hidden state)             [fp64 mode: fp64]
  Backward:
    h (recomputed): fp32 (states_in_fp32=True)     [fp64 mode: fp64]
    dh:       fp32                                  [fp64 mode: fp64]
    dq, dk:   input dtype                           [fp64 mode: fp64]
    dv:       do.dtype                              [fp64 mode: fp64]
    dg:       fp32 (NOT cast to g.dtype)            [fp64 mode: fp64]
    dh0:      fp32                                  [fp64 mode: fp64]
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common.utils import acc_dtype as _acc_dtype
from tops.cpu.ops.common.utils import cdiv as _cdiv
from tops.cpu.ops.common.utils import dot as _dot
from tops.cpu.ops.common.utils import pad_to_multiple as _pad_to_multiple
from tops.cpu.ops.common.utils import gather_chunks, scatter_chunks
from tops.utils import prepare_chunk_indices, prepare_lens
from tops.utils import cdiv as _cdiv_top
from tops.cpu.ops.common.chunk_h import chunk_fwd_h, chunk_bwd_dh
from tops.cpu.ops.common.chunk_o import chunk_fwd_o, chunk_local_cumsum


# =============================================================================
# Orchestrator: chunk_simple_gla_fwd
# =============================================================================


def chunk_simple_gla_fwd(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  g: jnp.ndarray | None,
  g_gamma: jnp.ndarray | None,
  scale: float,
  initial_state: jnp.ndarray | None,
  output_final_state: bool,
  chunk_size: int = 64,
  cu_seqlens: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Chunk Simple GLA forward orchestrator. No blanket dtype upcast.

  Orchestrates the chunk-based forward pass:
  1. Varlen pad (if cu_seqlens provided)
  2. Pad T to multiple of chunk_size
  3. Compute chunk-local cumsum of g (if provided)
  4. Propagate hidden states via chunk_fwd_h
  5. Compute output via chunk_fwd_o
  6. Unpad output to original T

  Args:
      q: [B, T, H, K] -- queries (input dtype)
      k: [B, T, H, K] -- keys (input dtype)
      v: [B, T, H, V] -- values (input dtype)
      g: [B, T, H] -- per-head scalar gates (any dtype). Optional.
      g_gamma: [H] -- fixed per-head log-decay. Optional.
      scale: scaling factor
      initial_state: [B, H, K, V] or [N, H, K, V] -- initial hidden state (fp32). Optional.
      output_final_state: whether to return final state
      chunk_size: block size
      cu_seqlens: [N+1] cumulative sequence lengths for variable-length segments.
          When provided, B must be 1 and sequences are treated independently.

  Returns:
      (o, ht) -- o in v.dtype unpadded to original T, ht in fp32 or None
  """
  T_orig = q.shape[1]
  H = q.shape[2]
  K = q.shape[3]
  V = v.shape[-1]
  C = chunk_size

  # --- Varlen gather+flatten ---
  is_varlen = cu_seqlens is not None
  if is_varlen:
    chunk_indices = prepare_chunk_indices(cu_seqlens, C)
    total_NT = chunk_indices.shape[0]

    # Gather to chunked layout [total_NT, C, ...]
    q_c, valid_lens = gather_chunks(q, cu_seqlens, chunk_indices, C)
    k_c, _ = gather_chunks(k, cu_seqlens, chunk_indices, C)
    v_c, _ = gather_chunks(v, cu_seqlens, chunk_indices, C)
    if g is not None:
      g_c, _ = gather_chunks(g, cu_seqlens, chunk_indices, C)
    # g_gamma is [H], NOT gathered

    # chunk_local_cumsum on chunked layout
    if g is not None:
      g_c = chunk_local_cumsum(g_c, C, cu_seqlens=cu_seqlens)

    # Flatten to [1, total_NT*C, ...]
    q = q_c.reshape(1, total_NT * C, H, K)
    k = k_c.reshape(1, total_NT * C, H, K)
    v = v_c.reshape(1, total_NT * C, H, V)
    if g is not None:
      g = g_c.reshape(1, total_NT * C, H)

    # Build flat cu_seqlens for boundary reset
    lens = prepare_lens(cu_seqlens)
    orig_seqlens = [int(l) for l in lens]
    n_chunks_per_seq = jnp.array([int(_cdiv_top(int(l), C)) for l in lens])
    flat_cu_seqlens = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks_per_seq * C),
    ])

    T_padded = total_NT * C
  else:
    flat_cu_seqlens = None
    orig_seqlens = None

    T = q.shape[1]
    # Padding
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
      q = _pad_to_multiple(q, C, axis=1)
      k = _pad_to_multiple(k, C, axis=1)
      v = _pad_to_multiple(v, C, axis=1)
      if g is not None:
        g = _pad_to_multiple(g, C, axis=1)

    # Chunk-local cumsum of g
    if g is not None:
      g = chunk_local_cumsum(g, C)

  # Hidden state propagation: states_in_fp32=False -> h in k.dtype
  h, ht = chunk_fwd_h(
    k,
    v,
    g=g,
    g_gamma=g_gamma,
    h0=initial_state,
    output_final_state=output_final_state,
    chunk_size=C,
    states_in_fp32=False,
    original_T=T_orig if not is_varlen else None,
    cu_seqlens=flat_cu_seqlens if is_varlen else None,
    orig_seqlens=orig_seqlens,
  )

  # Output computation
  o = chunk_fwd_o(q, k, v, h, g=g, g_gamma=g_gamma, scale=scale, chunk_size=C)

  # Scatter output back to packed layout
  if is_varlen:
    o_c = o.reshape(total_NT, C, H, V)
    o = scatter_chunks(
      jnp.zeros((1, T_orig, H, V), dtype=o.dtype),
      o_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
  else:
    o = o[:, :T_orig]

  return o, ht


# =============================================================================
# Backward sub-function 2: chunk_bwd_dqkwg
# =============================================================================


def chunk_bwd_dqkwg(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  h: jnp.ndarray,
  dh: jnp.ndarray,
  do: jnp.ndarray,
  g: jnp.ndarray | None = None,
  g_gamma: jnp.ndarray | None = None,
  scale: float = 1.0,
  chunk_size: int = 64,
  original_T: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
  """Backward gradients for q, k, and g (Simple GLA).

  Phase 1: V-loop computes b_ds, b_dq, b_dk (and b_dg_last for USE_G).
  Phase 2: Gate application (g / g_gamma / else are MUTUALLY EXCLUSIVE).

  Args:
      q:  [B, T, H, K] -- input dtype
      k:  [B, T, H, K] -- input dtype
      v:  [B, T, H, V] -- input dtype
      h:  [B, NT, H, K, V] -- hidden states (fp32, from bwd recompute)
      dh: [B, NT, H, K, V] -- hidden state gradients (fp32)
      do: [B, T, H, V] -- output gradient (input dtype)
      g:  [B, T, H] -- cumsummed gates (fp32). Optional.
      g_gamma: [H] -- fixed per-head log-decay. Optional.
      scale: scaling factor
      chunk_size: block size
      original_T: original unpadded T (for g_gamma chunk_len computation)

  Returns:
      dq: [B, T, H, K] -- input dtype
      dk: [B, T, H, K] -- input dtype
      dg: [B, T, H] -- fp32, or None (when only g_gamma or no gate)
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(q.dtype)
  if original_T is None:
    original_T = T

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  v_c = v.reshape(B, NT, C, H, V)
  do_c = do.reshape(B, NT, C, H, V)
  if g is not None:
    g_c = g.reshape(B, NT, C, H)

  # Causal mask [C, C]: lower-triangular (i >= j)
  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))

  dq_chunks = []
  dk_chunks = []
  dg_chunks = [] if g is not None else None

  for i in range(NT):
    b_q = q_c[:, i]  # [B, C, H, K]
    b_k = k_c[:, i]  # [B, C, H, K]
    b_v = v_c[:, i]  # [B, C, H, V]
    b_h = h[:, i]  # [B, H, K, V] fp32
    b_dh = dh[:, i]  # [B, H, K, V] fp32
    b_do = do_c[:, i]  # [B, C, H, V]

    # Phase 1: V-loop
    # b_ds = do @ v^T: [B, i, H, j]
    b_ds = _dot("bihv,bjhv->bihj", b_do, b_v, acc)

    # b_dq = do @ h^T: [B, C, H, K]
    b_dq = _dot("bchv,bhkv->bchk", b_do, b_h.astype(do.dtype), acc)

    # b_dk = v @ dh^T: [B, C, H, K]
    b_dk = _dot("bchv,bhkv->bchk", b_v, b_dh.astype(v.dtype), acc)

    # Phase 2: Gate application (mutually exclusive)
    if g is not None:
      gc = g_c[:, i]  # [B, C, H] fp32 (cumsummed)
      g_last = gc[:, -1]  # [B, H]
      # Transpose gc for key-position broadcasting: [B, H, C]
      gc_key = jnp.transpose(gc, (0, 2, 1))

      # b_dg_last from h * dh: sum over K, V -> [B, H]
      b_dg_last = jnp.sum(
        b_h.astype(acc) * b_dh,
        axis=(-2, -1),
      )
      b_dg_last = b_dg_last * jnp.exp(g_last)

      # dq: gate with exp(g), then scale
      b_dq = b_dq * jnp.exp(gc[..., None]) * scale
      b_dg = jnp.sum(b_dq * b_q.astype(acc), axis=-1)  # [B, C, H]

      # dk: gate with exp(-g + g_last)
      b_dk = b_dk * jnp.exp(-gc[..., None] + g_last[:, None, :, None])
      b_dg -= jnp.sum(b_k.astype(acc) * b_dk, axis=-1)  # [B, C, H]
      # b_dg_last: sum dk*k over C and K dims -> [B, H]
      b_dg_last = b_dg_last + jnp.sum(
        b_dk * b_k.astype(acc),
        axis=(1, 3),
      )

      # ds: causal mask + gate
      # b_ds [B, i, H, j] * exp(g_i - g_j)
      # gc[:, :, :, None] = [B, C_i, H, 1]; gc_key[:, None, :, :] = [B, 1, H, C_j]
      b_ds = (
        jnp.where(
          causal_mask[None, :, None, :],
          b_ds * jnp.exp(gc[:, :, :, None] - gc_key[:, None, :, :]),
          0,
        )
        * scale
      )

      # dg from ds: ds2 = ds * (q @ k^T element-wise in attention)
      # b_ds2[b,i,h,j] = b_ds[b,i,h,j] * sum_k(q[b,i,h,k] * k[b,j,h,k])
      b_qk = _dot("bihk,bjhk->bihj", b_q.astype(acc), b_k.astype(acc), acc)
      b_ds2 = b_ds * b_qk
      # dg[t] += sum_j ds2[t,j] - sum_i ds2[i,t]
      # sum over j (axis=-1): [B, C_i, H]
      # sum over i (axis=1): [B, H, C_j] -> transpose to [B, C, H]
      b_dg = (
        b_dg
        + jnp.sum(b_ds2, axis=-1)
        - jnp.transpose(jnp.sum(b_ds2, axis=1), (0, 2, 1))
      )

      # dq += ds @ k
      b_dq = b_dq + _dot("bihj,bjhk->bihk", b_ds.astype(k.dtype), b_k, acc)
      # dk += ds^T @ q
      b_dk = b_dk + _dot(
        "bjhi,bihk->bjhk", jnp.transpose(b_ds, (0, 3, 2, 1)).astype(q.dtype), b_q, acc
      )

      # Merge dg_last at last position of this chunk
      # b_dg[:, -1, :] += b_dg_last
      last_mask = jnp.zeros((C,), dtype=acc)
      last_mask = last_mask.at[-1].set(1.0)
      b_dg = b_dg + b_dg_last[:, None, :] * last_mask[None, :, None]

      dg_chunks.append(b_dg)

    elif g_gamma is not None:
      gamma = g_gamma  # [H]
      chunk_len = min(C, original_T - i * C)
      b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]
      g_gamma_last = gamma * chunk_len  # [H]
      # Transpose for key-position broadcasting: [H, C]
      b_g_gamma_key = b_g_gamma.T

      # dq: gate with exp(g_gamma), then scale
      b_dq = b_dq * jnp.exp(b_g_gamma[None, :, :, None]) * scale

      # dk: gate with exp(-g_gamma + g_gamma_last)
      b_dk = b_dk * jnp.exp(
        -b_g_gamma[None, :, :, None] + g_gamma_last[None, None, :, None]
      )

      # ds: causal mask + gate
      # b_g_gamma[None, :, :, None] = [1, C_i, H, 1]
      # b_g_gamma_key[None, None, :, :] = [1, 1, H, C_j]
      b_ds = (
        jnp.where(
          causal_mask[None, :, None, :],
          b_ds * jnp.exp(b_g_gamma[None, :, :, None] - b_g_gamma_key[None, None, :, :]),
          0,
        )
        * scale
      )

      # dq += ds @ k
      b_dq = b_dq + _dot("bihj,bjhk->bihk", b_ds.astype(k.dtype), b_k, acc)
      # dk += ds^T @ q
      b_dk = b_dk + _dot(
        "bjhi,bihk->bjhk", jnp.transpose(b_ds, (0, 3, 2, 1)).astype(q.dtype), b_q, acc
      )

    else:
      # No gate: exact scale order matters
      b_ds = jnp.where(causal_mask[None, :, None, :], b_ds, 0).astype(k.dtype)

      # dq += ds @ k; dk += ds^T @ q * scale; dq *= scale
      b_dq = b_dq + _dot("bihj,bjhk->bihk", b_ds, b_k, acc)
      b_dk = (
        b_dk
        + _dot("bjhi,bihk->bjhk", jnp.transpose(b_ds, (0, 3, 2, 1)), b_q, acc) * scale
      )
      b_dq = b_dq * scale

    dq_chunks.append(b_dq.astype(q.dtype))
    dk_chunks.append(b_dk.astype(k.dtype))

  dq = jnp.stack(dq_chunks, axis=1).reshape(B, T, H, K)
  dk = jnp.stack(dk_chunks, axis=1).reshape(B, T, H, K)
  dg = jnp.stack(dg_chunks, axis=1).reshape(B, T, H) if dg_chunks is not None else None

  return dq, dk, dg


# =============================================================================
# Backward sub-function 3: chunk_bwd_dv
# =============================================================================


def chunk_bwd_dv(
  q: jnp.ndarray,
  k: jnp.ndarray,
  do: jnp.ndarray,
  dh: jnp.ndarray,
  g: jnp.ndarray | None = None,
  g_gamma: jnp.ndarray | None = None,
  scale: float = 1.0,
  chunk_size: int = 64,
  original_T: int | None = None,
) -> jnp.ndarray:
  """Backward gradient for v (Simple GLA).

  Uses if g / elif g_gamma (mutually exclusive) for gate application.
  The attention matrix here uses UPPER-triangular mask (i >= j), which is
  the transpose of the forward causal mask.

  Args:
      q:  [B, T, H, K] -- input dtype
      k:  [B, T, H, K] -- input dtype
      do: [B, T, H, V] -- output gradient (input dtype)
      dh: [B, NT, H, K, V] -- hidden state gradients (fp32)
      g:  [B, T, H] -- cumsummed gates (fp32). Optional.
      g_gamma: [H] -- fixed per-head log-decay. Optional.
      scale: scaling factor
      chunk_size: block size
      original_T: original unpadded T (for g_gamma chunk_len computation)

  Returns:
      dv: [B, T, H, V] -- do.dtype
  """
  B, T, H, K = q.shape
  V = do.shape[-1]
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(q.dtype)
  if original_T is None:
    original_T = T

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  do_c = do.reshape(B, NT, C, H, V)
  if g is not None:
    g_c = g.reshape(B, NT, C, H)

  # Upper-triangular mask [C, C]: i >= j (transpose of forward causal)
  upper_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))

  dv_chunks = []

  for i in range(NT):
    b_q = q_c[:, i]  # [B, C, H, K]
    b_k = k_c[:, i]  # [B, C, H, K]
    b_do = do_c[:, i]  # [B, C, H, V]
    b_dh = dh[:, i]  # [B, H, K, V] fp32

    # Intra: A = k @ q^T [B, j, H, i] -- note reversed from forward
    b_A = _dot("bjhk,bihk->bjhi", b_k, b_q, acc)

    # Inter: k @ dh -> dv [B, C, H, V]
    b_dv = _dot("bchk,bhkv->bchv", b_k, b_dh.astype(k.dtype), acc)

    if g is not None:
      gc = g_c[:, i]  # [B, C, H]
      g_last = gc[:, -1]  # [B, H]
      # Transpose for i-position broadcasting: [B, H, C]
      gc_i = jnp.transpose(gc, (0, 2, 1))

      # Upper-triangular mask with gate: exp(g_i - g_j) where i >= j
      # b_A is [B, j, H, i]
      # gc[:, :, :, None] = [B, C_j, H, 1]; gc_i[:, None, :, :] = [B, 1, H, C_i]
      b_A = jnp.where(
        upper_mask[None, :, None, :],  # [1, j, 1, i]
        b_A * jnp.exp(gc_i[:, None, :, :] - gc[:, :, :, None]) * scale,
        0,
      ).astype(do.dtype)

      # Gate dv with exp(-g + g_last)
      b_dv = b_dv * jnp.exp(-gc[..., None] + g_last[:, None, :, None])

    elif g_gamma is not None:
      gamma = g_gamma  # [H]
      chunk_len = min(C, original_T - i * C)
      b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]
      g_gamma_last = gamma * chunk_len  # [H]
      # Transpose for i-position broadcasting: [H, C]
      b_g_gamma_i = b_g_gamma.T

      # Upper-triangular mask with gate: exp(g_gamma_i - g_gamma_j)
      # b_g_gamma[None, :, :, None] = [1, C_j, H, 1]
      # b_g_gamma_i[None, None, :, :] = [1, 1, H, C_i]
      b_A = jnp.where(
        upper_mask[None, :, None, :],
        b_A
        * jnp.exp(b_g_gamma_i[None, None, :, :] - b_g_gamma[None, :, :, None])
        * scale,
        0,
      ).astype(do.dtype)

      # Gate dv
      b_dv = b_dv * jnp.exp(
        -b_g_gamma[None, :, :, None] + g_gamma_last[None, None, :, None]
      )

    else:
      # No gate
      b_A = jnp.where(upper_mask[None, :, None, :], b_A * scale, 0).astype(do.dtype)

    # dv += A^T @ do  (A is [B, j, H, i], do is [B, i, H, V])
    b_dv = b_dv + _dot("bjhi,bihv->bjhv", b_A, b_do, acc)

    dv_chunks.append(b_dv.astype(do.dtype))

  dv = jnp.stack(dv_chunks, axis=1).reshape(B, T, H, V)
  return dv


# =============================================================================
# Backward orchestrator: chunk_simple_gla_bwd
# =============================================================================


def chunk_simple_gla_bwd(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  g: jnp.ndarray | None,
  g_gamma: jnp.ndarray | None,
  initial_state: jnp.ndarray | None,
  do: jnp.ndarray,
  dht: jnp.ndarray | None,
  scale: float,
  chunk_size: int = 64,
  cu_seqlens: jnp.ndarray | None = None,
) -> tuple[
  jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None
]:
  """Chunk Simple GLA backward orchestrator.

  Orchestrates the backward pass:
  1. Varlen pad (if cu_seqlens provided)
  2. Pad inputs to multiple of chunk_size
  3. Compute chunk-local cumsum of g (if provided)
  4. Recompute h with states_in_fp32=True
  5. Compute dh via chunk_bwd_dh
  6. Compute dq, dk, dg via chunk_bwd_dqkwg
  7. Compute dv via chunk_bwd_dv
  8. If dg is not None: reverse cumsum (stays fp32)
  9. Unpad

  Args:
      q:  [B, T, H, K] -- input dtype
      k:  [B, T, H, K] -- input dtype
      v:  [B, T, H, V] -- input dtype
      g:  [B, T, H] -- raw per-head scalar gates (NOT cumsummed). Optional.
      g_gamma: [H] -- fixed per-head log-decay. Optional.
      initial_state: [B, H, K, V] or [N, H, K, V] -- fp32. Optional.
      do: [B, T, H, V] -- output gradient (input dtype)
      dht: [B, H, K, V] or [N, H, K, V] -- terminal hidden state gradient (fp32). Optional.
      scale: scaling factor
      chunk_size: block size
      cu_seqlens: [N+1] cumulative sequence lengths for variable-length segments.
          When provided, B must be 1 and sequences are treated independently.

  Returns:
      dq:  [B, T, H, K] -- input dtype
      dk:  [B, T, H, K] -- input dtype
      dv:  [B, T, H, V] -- do.dtype
      dg:  [B, T, H] -- fp32, or None
      dh0: [B, H, K, V] or [N, H, K, V] -- fp32, or None
  """
  T_orig = q.shape[1]
  H = q.shape[2]
  K = q.shape[3]
  V = v.shape[-1]
  C = chunk_size

  assert not (g is not None and g_gamma is not None), (
    "chunk_simple_gla_bwd does not support both g and g_gamma simultaneously. "
    "chunk_bwd_dqkwg/chunk_bwd_dv use if/elif (mutually exclusive), so "
    "gradients would be incorrect. Use only one of g or g_gamma."
  )

  # --- Varlen gather+flatten ---
  is_varlen = cu_seqlens is not None
  if is_varlen:
    chunk_indices = prepare_chunk_indices(cu_seqlens, C)
    total_NT = chunk_indices.shape[0]

    # Gather to chunked layout [total_NT, C, ...]
    q_c, valid_lens = gather_chunks(q, cu_seqlens, chunk_indices, C)
    k_c, _ = gather_chunks(k, cu_seqlens, chunk_indices, C)
    v_c, _ = gather_chunks(v, cu_seqlens, chunk_indices, C)
    do_c, _ = gather_chunks(do, cu_seqlens, chunk_indices, C)
    if g is not None:
      g_c, _ = gather_chunks(g, cu_seqlens, chunk_indices, C)
    # g_gamma is [H], NOT gathered

    # chunk_local_cumsum on chunked layout
    if g is not None:
      g_c = chunk_local_cumsum(g_c, C, cu_seqlens=cu_seqlens)

    # Flatten to [1, total_NT*C, ...]
    q = q_c.reshape(1, total_NT * C, H, K)
    k = k_c.reshape(1, total_NT * C, H, K)
    v = v_c.reshape(1, total_NT * C, H, V)
    do = do_c.reshape(1, total_NT * C, H, V)
    if g is not None:
      g = g_c.reshape(1, total_NT * C, H)

    # Build flat cu_seqlens for boundary reset
    lens = prepare_lens(cu_seqlens)
    orig_seqlens = [int(l) for l in lens]
    n_chunks_per_seq = jnp.array([int(_cdiv_top(int(l), C)) for l in lens])
    flat_cu_seqlens = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks_per_seq * C),
    ])

    T_padded = total_NT * C
  else:
    flat_cu_seqlens = None
    orig_seqlens = None

    T = q.shape[1]
    # Padding
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
      q = _pad_to_multiple(q, C, axis=1)
      k = _pad_to_multiple(k, C, axis=1)
      v = _pad_to_multiple(v, C, axis=1)
      do = _pad_to_multiple(do, C, axis=1)
      if g is not None:
        g = _pad_to_multiple(g, C, axis=1)

    # Chunk-local cumsum of g (same as forward orchestrator)
    if g is not None:
      g = chunk_local_cumsum(g, C)

  # Recompute h with states_in_fp32=True (backward needs fp32 states)
  h, _ = chunk_fwd_h(
    k,
    v,
    g=g,
    g_gamma=g_gamma,
    h0=initial_state,
    output_final_state=False,
    chunk_size=C,
    states_in_fp32=True,
    original_T=T_orig if not is_varlen else None,
    cu_seqlens=flat_cu_seqlens if is_varlen else None,
    orig_seqlens=orig_seqlens,
  )

  # dh: backward hidden state gradient propagation
  dh, dh0 = chunk_bwd_dh(
    q,
    do,
    g=g,
    g_gamma=g_gamma,
    h0=initial_state,
    dht=dht,
    scale=scale,
    chunk_size=C,
    original_T=T_orig if not is_varlen else None,
    cu_seqlens=flat_cu_seqlens if is_varlen else None,
    orig_seqlens=orig_seqlens,
  )

  # dq, dk, dg
  dq, dk, dg = chunk_bwd_dqkwg(
    q,
    k,
    v,
    h,
    dh,
    do,
    g=g,
    g_gamma=g_gamma,
    scale=scale,
    chunk_size=C,
    original_T=T_orig if not is_varlen else None,
  )

  # dv
  dv = chunk_bwd_dv(
    q,
    k,
    do,
    dh,
    g=g,
    g_gamma=g_gamma,
    scale=scale,
    chunk_size=C,
    original_T=T_orig if not is_varlen else None,
  )

  # Reverse cumsum of dg (keep fp32, matching FLA internal bwd + GLA CPU ref;
  # note: FLA autograd wrapper casts dg to g.dtype via .to(g), we intentionally
  # keep fp32 for gradient precision, same as GLA CPU ref)
  if dg is not None:
    dg = chunk_local_cumsum(dg, C, reverse=True)

  # Scatter gradients back to packed layout
  if is_varlen:
    dq_c = dq.reshape(total_NT, C, H, K)
    dk_c = dk.reshape(total_NT, C, H, K)
    dv_c = dv.reshape(total_NT, C, H, V)
    dq = scatter_chunks(
      jnp.zeros((1, T_orig, H, K), dtype=dq.dtype),
      dq_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
    dk = scatter_chunks(
      jnp.zeros((1, T_orig, H, K), dtype=dk.dtype),
      dk_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
    dv = scatter_chunks(
      jnp.zeros((1, T_orig, H, V), dtype=dv.dtype),
      dv_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
    if dg is not None:
      dg_c = dg.reshape(total_NT, C, H)
      dg = scatter_chunks(
        jnp.zeros((1, T_orig, H), dtype=dg.dtype),
        dg_c, cu_seqlens, chunk_indices, C, valid_lens,
      )
  else:
    dq = dq[:, :T_orig]
    dk = dk[:, :T_orig]
    dv = dv[:, :T_orig]
    if dg is not None:
      dg = dg[:, :T_orig]

  return dq, dk, dv, dg, dh0


# =============================================================================
# Public API: chunk_simple_gla
# =============================================================================


@cpu_reference
def chunk_simple_gla(
  q: jnp.ndarray,
  k: jnp.ndarray,
  v: jnp.ndarray,
  g: jnp.ndarray | None = None,
  g_gamma: jnp.ndarray | None = None,
  scale: float | None = None,
  initial_state: jnp.ndarray | None = None,
  output_final_state: bool = False,
  cu_seqlens: jnp.ndarray | None = None,
  chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Chunk Simple GLA with FLA-triton-exact dtype behavior.

  No blanket upcast -- inputs stay in their original dtype.

  Simple GLA uses per-head scalar gates (g: [B,T,H]) instead of per-element
  gates (gk: [B,T,H,K]) as in standard GLA. The scalar gate is broadcast
  over both K and V dimensions when updating the hidden state.

  Core recurrence (chunk-based):
      h_t = h_{t-1} * exp(g_t) [* exp(g_gamma)] + k_t^T @ v_t
      o_t = (q_t * scale)^T @ h_t

  Args:
      q:               [B, T, H, K] -- Queries
      k:               [B, T, H, K] -- Keys
      v:               [B, T, H, V] -- Values
      g:               [B, T, H]    -- Per-head scalar gate in log-space.
                       Optional. If None, no learnable decay is applied.
      g_gamma:         [H]          -- Fixed per-head log-decay.
                       Only one of g or g_gamma is recommended; using both
                       causes state explosion due to FLA variable-reuse.
                       Must be in acc_dtype (fp32 or fp64). Optional.
      scale:           Scalar query scale. Defaults to K ** -0.5.
      initial_state:   [B, H, K, V] or [N, H, K, V] -- Initial hidden state. Optional.
                       When cu_seqlens is provided, shape is [N, H, K, V].
      output_final_state: Whether to return the final hidden state.
      cu_seqlens:      [N+1] cumulative sequence lengths for variable-length segments.
                       When provided, B must be 1 and sequences are treated independently.
      chunk_size:      Block size for chunked computation.

  Returns:
      o:           [B, T, H, V] -- Output (v.dtype)
      final_state: [B, H, K, V] or [N, H, K, V] in fp32 (or fp64), or None
  """
  B, T, H, K = q.shape

  # Shape assertions (project coding standard)
  assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
  assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
  assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
    f"v shape {v.shape} incompatible with q shape {q.shape}"
  )
  if g is not None:
    assert g.ndim == 3 and g.shape == q.shape[:3], f"g shape {g.shape} != {q.shape[:3]}"
  if g_gamma is not None:
    assert g_gamma.ndim == 1 and g_gamma.shape[0] == H, (
      f"g_gamma shape {g_gamma.shape} != (H={H},)"
    )
  if cu_seqlens is not None:
    assert B == 1, f"B must be 1 for varlen, got {B}"
    N = len(cu_seqlens) - 1
    lens = jnp.diff(cu_seqlens)
    assert jnp.all(lens > 0), "Empty segments not supported"
    if initial_state is not None:
      assert initial_state.shape == (N, H, K, v.shape[-1]), (
        f"initial_state shape {initial_state.shape} != ({N}, {H}, {K}, {v.shape[-1]})"
      )
  else:
    if initial_state is not None:
      assert initial_state.shape == (B, H, K, v.shape[-1]), (
        f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {v.shape[-1]})"
      )

  if scale is None:
    scale = K**-0.5
  if initial_state is not None:
    initial_state = initial_state.astype(_acc_dtype(q.dtype))

  o, ht = chunk_simple_gla_fwd(
    q,
    k,
    v,
    g,
    g_gamma,
    scale,
    initial_state,
    output_final_state,
    chunk_size,
    cu_seqlens=cu_seqlens,
  )
  return o, (ht if output_final_state else None)
