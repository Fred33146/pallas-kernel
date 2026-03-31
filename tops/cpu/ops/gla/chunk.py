"""JAX CPU reference for GLA chunk operations with FLA-triton-exact dtype behavior.

This implementation precisely matches the mixed-precision behavior of the
FLA Triton kernels, including:
- Keeping inputs in their original dtype (bf16/fp16/fp32), NO blanket upcast
- Selective upcasting to fp32 for accumulation and exp operations
- Explicit casting back to input dtype where Triton does so

fp64 mode: When inputs are fp64, all precision casts are skipped (they become
no-ops) and accumulation uses fp64 throughout. This provides a high-precision
reference that exceeds Triton's fp32 accumulation.

Dtype contract (matching FLA Triton for bf16/fp16/fp32; all fp64 for fp64):
  Forward:
    g_cumsum: fp32 (chunk_local_cumsum output)     [fp64 mode: fp64]
    h:        k.dtype if states_in_fp32=False, else fp32  [fp64 mode: fp64]
    A:        fp32 (intra-chunk attention matrix)   [fp64 mode: fp64]
    o:        v.dtype                               [fp64 mode: fp64]
    ht:       fp32 (final hidden state)             [fp64 mode: fp64]
  Backward:
    h (recomputed): fp32 (states_in_fp32=True)      [fp64 mode: fp64]
    dh:       fp32                                  [fp64 mode: fp64]
    dA:       fp32                                  [fp64 mode: fp64]
    dq, dk:   fp32 (from chunk_gla_bwd)             [fp64 mode: fp64]
    dv:       do.dtype                              [fp64 mode: fp64]
    dg:       fp32 (NOT cast to g.dtype — FLA keeps fp32)  [fp64 mode: fp64]
    dh0:      fp32                                  [fp64 mode: fp64]
"""

from __future__ import annotations

import jax
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
from tops.cpu.ops.common.chunk_o import chunk_local_cumsum


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk(
  q: jax.Array,
  k: jax.Array,
  g: jax.Array,
  scale: float,
  chunk_size: int = 64,
) -> jax.Array:
  """Intra-chunk attention matrix.

  FLA Triton: g_cumsum is fp32, so q*exp(g) and k*exp(-g) are fp32
  (natural type promotion). Matmul in fp32. A output: always fp32.

  Args:
      q: [B, T, H, K] — queries (input dtype)
      k: [B, T, H, K] — keys (input dtype)
      g: [B, T, H, K] — chunk-local cumsum (fp32)
      scale: scaling factor
      chunk_size: block size

  Returns:
      A: [B, T, H, C] — fp32
  """
  B, T, H, K = q.shape
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(q.dtype)

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  g_c = g.reshape(B, NT, C, H, K)

  # Reference point for numerical stability
  g_n = g_c[:, :, 0:1, :, :]

  # input_dtype * fp32_exp → fp32 (natural promotion); fp64 stays fp64
  q_gated = q_c * jnp.exp(g_c - g_n)
  k_gated = k_c * jnp.exp(g_n - g_c)

  A = _dot("bnihk,bnjhk->bnihj", q_gated, k_gated, acc) * scale

  return A.reshape(B, T, H, C)


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk(
  q: jax.Array,
  v: jax.Array,
  g: jax.Array,
  A: jax.Array,
  h: jax.Array,
  scale: float,
  chunk_size: int = 64,
) -> jax.Array:
  """Combine inter-chunk and intra-chunk to produce output.

  FLA Triton dtype behavior:
  - qg = (q * exp(g_fp32)).to(q.dtype) — explicit cast back!
  - o_inter: einsum(qg_input_dtype, h_input_dtype) → fp32 accum
  - A cast to v.dtype for intra matmul
  - o stored in v.dtype

  Args:
      q: [B, T, H, K] — queries (input dtype)
      v: [B, T, H, V] — values (input dtype)
      g: [B, T, H, K] — chunk-local cumsum (fp32)
      A: [B, T, H, C] — attention matrix (fp32)
      h: [B, NT, H, K, V] — hidden states (k.dtype or fp32)
      scale: scaling factor
      chunk_size: block size

  Returns:
      o: [B, T, H, V] — v.dtype
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  acc = _acc_dtype(q.dtype)

  q_c = q.reshape(-1, C, H, K)
  v_c = v.reshape(-1, C, H, V)
  g_c = g.reshape(-1, C, H, K)
  h_c = h.reshape(-1, H, K, V)
  A_c = A.reshape(-1, C, H, C)

  # qg = (q * exp(g_fp32)).to(q.dtype) — triton explicit cast back
  qg = (q_c * jnp.exp(g_c)).astype(q.dtype)

  # Inter: qg(input_dtype) @ h.to(qg.dtype), acc dtype accumulation
  o_inter = scale * _dot("nchk,nhkv->nchv", qg, h_c.astype(qg.dtype), acc)

  # Intra: A.to(v.dtype) @ v, acc dtype accumulation
  A_v = A_c.astype(v.dtype)
  o_intra = _dot("nihj,njhv->nihv", A_v, v_c, acc)

  o = (o_inter + o_intra).astype(v.dtype)
  return o.reshape(B, T, H, V)


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  g_cumsum: jax.Array | None,
  scale: float,
  initial_state: jax.Array | None,
  output_final_state: bool,
  chunk_size: int = 64,
  cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
  """Chunk GLA forward orchestrator. No blanket dtype upcast.

  Returns:
      (g_cumsum, A, h, ht, o) — o unpadded to original T
  """
  B, T_orig, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # --- Varlen gather+flatten ---
  is_varlen = cu_seqlens is not None
  if is_varlen:
    chunk_indices = prepare_chunk_indices(cu_seqlens, C)
    total_NT = chunk_indices.shape[0]

    # Gather to chunked layout [total_NT, C, H, K]
    q_c, valid_lens = gather_chunks(q, cu_seqlens, chunk_indices, C)
    k_c, _ = gather_chunks(k, cu_seqlens, chunk_indices, C)
    v_c, _ = gather_chunks(v, cu_seqlens, chunk_indices, C)
    g_c, _ = gather_chunks(g, cu_seqlens, chunk_indices, C)

    # chunk_local_cumsum on chunked layout (cu_seqlens triggers chunked branch)
    if g_cumsum is None:
      g_cumsum_c = chunk_local_cumsum(g_c, C, cu_seqlens=cu_seqlens)
    else:
      g_cumsum_c = g_cumsum  # already computed

    # Flatten to [1, total_NT*C, H, K]
    q = q_c.reshape(1, total_NT * C, H, K)
    k = k_c.reshape(1, total_NT * C, H, K)
    v = v_c.reshape(1, total_NT * C, H, V)
    g_cumsum = g_cumsum_c.reshape(1, total_NT * C, *g_cumsum_c.shape[2:])

    # Build flat cu_seqlens for boundary reset
    lens = prepare_lens(cu_seqlens)
    n_chunks_per_seq = jnp.array([int(_cdiv_top(int(l), C)) for l in lens])
    flat_cu_seqlens = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks_per_seq * C),
    ])

    T_padded = total_NT * C
  else:
    flat_cu_seqlens = None

    T = q.shape[1]
    # Existing padding for non-chunk-aligned T
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
      q = _pad_to_multiple(q, C, axis=1)
      k = _pad_to_multiple(k, C, axis=1)
      v = _pad_to_multiple(v, C, axis=1)
      g = _pad_to_multiple(g, C, axis=1)

    if g_cumsum is None:
      g_cumsum = chunk_local_cumsum(g, C)

  NT = T_padded // C

  # Forward: states_in_fp32=False → h in k.dtype
  h, ht = chunk_fwd_h(
    k,
    v,
    gk=g_cumsum,
    h0=initial_state,
    output_final_state=output_final_state,
    chunk_size=C,
    states_in_fp32=False,
    cu_seqlens=flat_cu_seqlens if is_varlen else None,
  )

  # A: fp32
  A = chunk_gla_fwd_intra_gk(q, k, g=g_cumsum, scale=scale, chunk_size=C)

  # Causal mask
  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
  A_5d = A.reshape(B, -1, C, H, C)
  A = jnp.where(causal_mask[None, None, :, None, :], A_5d, 0.0).reshape(
    B, T_padded, H, C
  )

  # Output: v.dtype
  o = chunk_gla_fwd_o_gk(q, v, g=g_cumsum, A=A, h=h, scale=scale, chunk_size=C)

  # Scatter output back to packed layout
  if is_varlen:
    o_c = o.reshape(total_NT, C, H, V)
    o = scatter_chunks(
      jnp.zeros((1, T_orig, H, V), dtype=o.dtype),
      o_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
    g_cumsum_out = g_cumsum  # keep flat for potential bwd use
    A_out = A
  else:
    o = o[:, :T_orig]
    g_cumsum_out = g_cumsum[:, :T_orig]
    A_out = A[:, :T_orig]

  return g_cumsum_out, A_out, h, ht, o


# =============================================================================
# Backward sub-function 2: chunk_gla_bwd_dA
# =============================================================================


def chunk_gla_bwd_dA(
  v: jax.Array,
  do: jax.Array,
  scale: float,
  chunk_size: int = 64,
) -> jax.Array:
  """Gradient of intra-chunk attention matrix.

  FLA Triton: dA output in fp32. Matmul accumulates in fp32.

  Returns:
      dA: [B, T, H, C] — fp32, lower-triangular masked
  """
  B, T, H, V = v.shape
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(v.dtype)

  v_c = v.reshape(B, NT, C, H, V)
  do_c = do.reshape(B, NT, C, H, V)

  dA = _dot("bnihv,bnjhv->bnihj", do_c, v_c, acc) * scale

  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
  dA = jnp.where(causal_mask[None, None, :, None, :], dA, 0.0)
  return dA.reshape(B, T, H, C)


# =============================================================================
# Backward sub-function 3: chunk_gla_bwd_dv
# =============================================================================


def chunk_gla_bwd_dv(
  k: jax.Array,
  g_cumsum: jax.Array,
  A: jax.Array,
  do: jax.Array,
  dh: jax.Array,
  chunk_size: int = 64,
) -> jax.Array:
  """Gradient of v.

  FLA Triton dtype:
  - Intra: dot(A_fp32, do.to(fp32)) — fp32, allow_tf32=False
  - Inter: k_decay.to(k.dtype) @ dh.to(k.dtype) — input dtype, fp32 accum
  - dv output: do.dtype

  Returns:
      dv: [B, T, H, V] — do.dtype
  """
  B, T, H, K = k.shape
  V = do.shape[-1]
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(k.dtype)

  k_c = k.reshape(B, NT, C, H, K)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  do_c = do.reshape(B, NT, C, H, V)
  A_c = A.reshape(B, NT, C, H, C)

  # Intra: A(acc) @ do.to(acc), allow_tf32=False → HIGHEST precision
  dv_intra = _dot("bnihj,bnihv->bnjhv", A_c, do_c.astype(acc), acc)

  # Inter: k_decay @ dh
  gn = gc_c[:, :, -1, :, :]
  k_decay = (k_c * jnp.exp(gn[:, :, None, :, :] - gc_c)).astype(k.dtype)

  dv_inter = _dot("bnchk,bnhkv->bnchv", k_decay, dh.astype(k.dtype), acc)

  dv = (dv_intra + dv_inter).astype(do.dtype)
  return dv.reshape(B, T, H, V)


# =============================================================================
# Backward sub-function 4: chunk_gla_bwd_dqk_intra
# =============================================================================


def chunk_gla_bwd_dqk_intra(
  q: jax.Array,
  k: jax.Array,
  g_cumsum: jax.Array,
  dA: jax.Array,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
  """Intra-chunk dq, dk from dA.

  FLA Triton: k, gk explicitly cast to fp32. dq, dk output in fp32.

  Returns:
      dq, dk: [B, T, H, K] — fp32
  """
  B, T, H, K = q.shape
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(q.dtype)

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  dA_c = dA.reshape(B, NT, C, H, C)

  # k and gk explicitly cast to acc dtype in triton intra kernel
  k_neg = k_c.astype(acc) * jnp.exp(-gc_c)
  dq = jnp.exp(gc_c) * _dot("bnihj,bnjhk->bnihk", dA_c, k_neg, acc)

  q_pos = q_c.astype(acc) * jnp.exp(gc_c)
  dk = jnp.exp(-gc_c) * _dot("bnihj,bnihk->bnjhk", dA_c, q_pos, acc)

  return dq.reshape(B, T, H, K), dk.reshape(B, T, H, K)


# =============================================================================
# Backward sub-function 5: chunk_gla_bwd_dqkg
# =============================================================================


def chunk_gla_bwd_dqkg(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  h: jax.Array,
  g_cumsum: jax.Array,
  do: jax.Array,
  dh: jax.Array,
  dq: jax.Array,
  dk: jax.Array,
  scale: float,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Inter-chunk dq, dk + gate gradient dg.

  FLA Triton dtype:
  - h: fp32 (states_in_fp32=True in bwd). Cast to do.dtype for matmul.
  - dh: fp32. Cast to v.dtype for matmul.
  - All intermediates in fp32.

  Returns:
      dq, dk, dg: [B, T, H, K] — fp32
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  NT = T // C
  acc = _acc_dtype(q.dtype)

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  v_c = v.reshape(B, NT, C, H, V)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  do_c = do.reshape(B, NT, C, H, V)
  dq_c = dq.reshape(B, NT, C, H, K)
  dk_c = dk.reshape(B, NT, C, H, K)

  gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K] acc dtype

  # dq_inter: do(input) @ h.to(do.dtype), * scale * exp(gc)
  dq_inter = (
    scale * jnp.exp(gc_c) * _dot("bnchv,bnhkv->bnchk", do_c, h.astype(do.dtype), acc)
  )

  # dk_inter: v(input) @ dh.to(v.dtype), * exp(gn - gc)
  dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * _dot(
    "bnchv,bnhkv->bnchk", v_c, dh.astype(v.dtype), acc
  )

  dq_total = dq_c + dq_inter
  dk_total = dk_c + dk_inter

  # Gate gradient
  dgk_inter = jnp.exp(gn) * _dot("bnhkv,bnhkv->bnhk", h.astype(acc), dh, acc) + jnp.sum(
    dk_inter * k_c.astype(acc), axis=2
  )

  dg_raw = q_c.astype(acc) * dq_total - k_c.astype(acc) * dk_total
  # Reverse cumsum: flip → cumsum → flip
  dg = jnp.cumsum(dg_raw[:, :, ::-1], axis=2)[:, :, ::-1] + dgk_inter[:, :, None, :, :]

  return (
    dq_total.reshape(B, T, H, K),
    dk_total.reshape(B, T, H, K),
    dg.reshape(B, T, H, K),
  )


# =============================================================================
# Backward orchestrator: chunk_gla_bwd
# =============================================================================


def chunk_gla_bwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  g_cumsum: jax.Array | None,
  scale: float,
  initial_state: jax.Array | None,
  h: jax.Array | None,
  A: jax.Array | None,
  do: jax.Array,
  dht: jax.Array | None,
  chunk_size: int = 64,
  cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Chunk GLA backward orchestrator.

  FLA Triton:
  - Recomputes h with states_in_fp32=True (fp32)
  - dh: fp32
  - Returns: dq(fp32), dk(fp32), dv(do.dtype), dg(fp32), dh0(fp32)

  Returns:
      (dq, dk, dv, dg, dh0)
  """
  B, T_orig, H, K = q.shape
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
    g_c, _ = gather_chunks(g, cu_seqlens, chunk_indices, C)
    do_c, _ = gather_chunks(do, cu_seqlens, chunk_indices, C)

    # chunk_local_cumsum on chunked layout
    if g_cumsum is None:
      g_cumsum_c = chunk_local_cumsum(g_c, C, cu_seqlens=cu_seqlens)
    else:
      g_cumsum_c = g_cumsum

    # Flatten to [1, total_NT*C, ...]
    q = q_c.reshape(1, total_NT * C, H, K)
    k = k_c.reshape(1, total_NT * C, H, K)
    v = v_c.reshape(1, total_NT * C, H, V)
    do = do_c.reshape(1, total_NT * C, H, V)
    g_cumsum = g_cumsum_c.reshape(1, total_NT * C, *g_cumsum_c.shape[2:])

    # Build flat cu_seqlens for boundary reset
    lens = prepare_lens(cu_seqlens)
    n_chunks_per_seq = jnp.array([int(_cdiv_top(int(l), C)) for l in lens])
    flat_cu_seqlens = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks_per_seq * C),
    ])

    T_padded = total_NT * C
  else:
    flat_cu_seqlens = None

    T = q.shape[1]
    # Existing padding
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
      q = _pad_to_multiple(q, C, axis=1)
      k = _pad_to_multiple(k, C, axis=1)
      v = _pad_to_multiple(v, C, axis=1)
      g = _pad_to_multiple(g, C, axis=1)
      do = _pad_to_multiple(do, C, axis=1)

    # Recompute g_cumsum
    if g_cumsum is None:
      g_cumsum = chunk_local_cumsum(g, C)

  # Recompute h with states_in_fp32=True (FLA backward behavior)
  if h is None:
    h, _ = chunk_fwd_h(
      k,
      v,
      gk=g_cumsum,
      h0=initial_state,
      output_final_state=False,
      chunk_size=C,
      states_in_fp32=True,
      cu_seqlens=flat_cu_seqlens if is_varlen else None,
    )

  # dh: fp32
  dh, dh0 = chunk_bwd_dh(
    q,
    do,
    gk=g_cumsum,
    h0=initial_state,
    dht=dht,
    scale=scale,
    chunk_size=C,
    cu_seqlens=flat_cu_seqlens if is_varlen else None,
  )

  # Recompute A if not provided
  if A is None:
    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(
      causal_mask[None, None, :, None, :], A.reshape(B, -1, C, H, C), 0.0
    ).reshape(B, T_padded, H, C)
  else:
    A = A.reshape(B, T_padded, H, C)

  # dv: do.dtype
  dv = chunk_gla_bwd_dv(k, g_cumsum, A, do, dh, chunk_size=C)

  # dA: fp32
  dA = chunk_gla_bwd_dA(v, do, scale, chunk_size=C)

  # dq, dk intra: fp32
  dq, dk = chunk_gla_bwd_dqk_intra(q, k, g_cumsum, dA, chunk_size=C)

  # dq, dk inter + dg: fp32
  dq, dk, dg = chunk_gla_bwd_dqkg(
    q,
    k,
    v,
    h,
    g_cumsum,
    do,
    dh,
    dq,
    dk,
    scale,
    chunk_size=C,
  )

  # Scatter gradients back to packed layout
  if is_varlen:
    dq_c = dq.reshape(total_NT, C, H, K)
    dk_c = dk.reshape(total_NT, C, H, K)
    dv_c = dv.reshape(total_NT, C, H, V)
    dg_c = dg.reshape(total_NT, C, H, K)
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
    dg = scatter_chunks(
      jnp.zeros((1, T_orig, H, K), dtype=dg.dtype),
      dg_c, cu_seqlens, chunk_indices, C, valid_lens,
    )
  else:
    dq = dq[:, :T_orig]
    dk = dk[:, :T_orig]
    dv = dv[:, :T_orig]
    dg = dg[:, :T_orig]

  # Match FLA output dtypes: dq(fp32), dk(fp32), dv(do.dtype), dg(fp32), dh0(fp32)
  # FLA keeps dg in fp32 (g_cumsum dtype), does NOT cast to g.dtype

  return dq, dk, dv, dg, dh0


# =============================================================================
# Public API: chunk_gla
# =============================================================================


@cpu_reference
def chunk_gla(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None = None,
  scale: float | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
  """Chunk GLA with FLA-triton-exact dtype behavior.

  No blanket upcast — inputs stay in their original dtype.

  Args:
      q: [B, T, H, K]
      k: [B, T, H, K]
      v: [B, T, H, V]
      g: [B, T, H, K] — log-space gates (any dtype); if None, defaults to zeros (no decay)
      scale: scaling factor (default: K^-0.5)
      initial_state: [B, H, K, V] (fp32); or [N, H, K, V] when cu_seqlens is provided
      output_final_state: whether to return final state
      cu_seqlens: [N+1] cumulative sequence lengths for variable-length segments.
          When provided, B must be 1 and sequences are treated independently.
      chunk_size: block size

  Returns:
      (o, final_state) — o in v.dtype, final_state in fp32 or None
  """
  B, T, H, K = q.shape
  V = v.shape[-1]

  if cu_seqlens is not None:
    assert B == 1, f"B must be 1 for varlen, got {B}"
    N = len(cu_seqlens) - 1
    lens = jnp.diff(cu_seqlens)
    assert jnp.all(lens > 0), "Empty segments not supported"
    if initial_state is not None:
      assert initial_state.shape == (N, H, K, V)

  if g is None:
    g = jnp.zeros_like(q)
  if scale is None:
    scale = K**-0.5
  if initial_state is not None:
    initial_state = initial_state.astype(_acc_dtype(q.dtype))

  _, _, _, ht, o = chunk_gla_fwd(
    q,
    k,
    v,
    g,
    None,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
  )
  return o, ht if output_final_state else None
