from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype, cdiv as _cdiv, pad_to_multiple as _pad_to_multiple


@cpu_reference
def naive_recurrent_kda(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """
  Core recurrence (per timestep):
      S' = S_{t-1} * exp(g_t)                            decay
      residual = v_t - k_t^T @ S'                        prediction error
      S_t = S' + beta_t * k_t ⊗ residual                 delta update
      o_t = (q_t * scale)^T @ S_t                        output

  Dtype behavior (matching FLA):
    - All inputs cast to fp32 for computation
    - Hidden state S is fp32 accumulator
    - Output o computed in fp32, cast back to original dtype
    - Final state S stays in fp32
    - fp64 mode: all computation in fp64, no precision cast

  Args:
      q:               [B, T, H, K] — Queries
      k:               [B, T, H, K] — Keys
      v:               [B, T, H, V] — Values
      g:               [B, T, H, K] — Per-element gate in log-space (e.g., -exp(A)*softplus(g))
      beta:            [B, T, H]    — Learning rate / step size for delta rule
      scale:           Scalar query scale. Defaults to K ** -0.5.
      initial_state:   [B, H, K, V] — Initial hidden state. Optional.
      output_final_state: Whether to return the final hidden state.

  Returns:
      o:           [B, T, H, V] — Output (original input dtype)
      final_state: [B, H, K, V] in fp32 (or fp64), or None
  """
  orig_dtype, acc_dt = v.dtype, _acc_dtype(q.dtype)

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

  B, T, H, K, V = *q.shape, v.shape[-1]

  if initial_state is not None:
    assert initial_state.shape == (B, H, K, V), (
      f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"
    )

  if scale is None:
    scale = K ** -0.5

  # [B, T, H, K] -> [B, H, T, K], cast to acc_dt
  q, k, v, g = (
    jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt)
    for x in (q, k, v, g)
  )
  # q: [B, H, T, K]   k: [B, H, T, K]   v: [B, H, T, V]   g: [B, H, T, K]

  # [B, T, H] -> [B, H, T]
  beta = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt)  # [B, H, T]

  q = q * scale  # [B, H, T, K]

  S = jnp.zeros((B, H, K, V), dtype=acc_dt)   # [B, H, K, V] hidden state
  if initial_state is not None:
    S += initial_state.astype(acc_dt)           # [B, H, K, V]
  o = jnp.zeros((B, H, T, V), dtype=acc_dt)   # [B, H, T, V] output buffer

  for i in range(T):
    q_i = q[:, :, i]     # [B, H, K]
    k_i = k[:, :, i]     # [B, H, K]
    v_i = v[:, :, i]     # [B, H, V]
    g_i = g[:, :, i]     # [B, H, K]
    b_i = beta[:, :, i]  # [B, H]

    # 1. Decay the state
    # exp(g_i): [B, H, K] -> [B, H, K, 1] via broadcast
    S = S * jnp.exp(g_i)[..., None]  # [B, H, K, V]

    # 2. Delta rule update
    # k_i[..., None]: [B, H, K, 1],  k_i[..., None] * S: [B, H, K, V]
    v_predicted = (k_i[..., None] * S).sum(-2)    # [B, H, V]
    residual = v_i - v_predicted                  # [B, H, V]

    # b_i[..., None] * k_i: [B, H, K],  einsum -> [B, H, K, V]
    S = S + jnp.einsum(
      'bhk,bhv->bhkv', b_i[..., None] * k_i, residual
    )  # [B, H, K, V]

    # 3. Compute output: einsum [B,H,K] x [B,H,K,V] -> [B, H, V]
    o = o.at[:, :, i].set(
      jnp.einsum('bhk,bhkv->bhv', q_i, S)  # [B, H, V]
    )

  final_state = S if output_final_state else None  # [B, H, K, V] or None
  # [B, H, T, V] -> [B, T, H, V], cast back to orig_dtype
  return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state


@cpu_reference
def naive_chunk_kda(
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
  """
  Processes the sequence in chunks of size ``chunk_size``, solving
  intra-chunk delta rule dependencies via a lower-triangular forward
  substitution.  Mathematically equivalent to ``naive_recurrent_kda`` but operates
  on blocks, yielding better cache behaviour.

  Algorithm:
    1. Reshape inputs into [B, H, NT, C, ...] and cumsum g within chunks
    2. Build interaction matrix A[c,i] = sum_k k[c,k]*exp(g[c][k]-g[i][k])*k[i,k]
    3. Solve lower-triangular system via forward substitution
    4. Compute effective keys w = A @ (exp(g)*k) and values u = A @ v
    5. Inter-chunk loop: o = (q*scale*exp(g)) @ S + A_qk @ (u - w @ S)

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
      final_state: [B, H, K, V] in accumulator dtype, or None
  """

  orig_dtype = v.dtype
  acc_dt = _acc_dtype(q.dtype)

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

  B, T, H, K, V = *q.shape, v.shape[-1]
  BT = chunk_size

  if initial_state is not None:
    assert initial_state.shape == (B, H, K, V), (
      f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"
    )

  if scale is None:
    scale = K ** -0.5

  # --- Pad T to multiple of chunk_size ---
  assert BT > 0, f"chunk_size must be positive, got {BT}"
  T_orig = T
  T_padded = _cdiv(T, BT) * BT
  if T_padded > T:
    q = _pad_to_multiple(q, BT, axis=1)      # [B, T_padded, H, K]
    k = _pad_to_multiple(k, BT, axis=1)      # [B, T_padded, H, K]
    v = _pad_to_multiple(v, BT, axis=1)      # [B, T_padded, H, V]
    g = _pad_to_multiple(g, BT, axis=1)      # [B, T_padded, H, K]
    beta = _pad_to_multiple(beta, BT, axis=1) # [B, T_padded, H]
    T = T_padded

  NT = T // BT  # number of chunks

  # [B, T, H, D] -> [B, H, NT, BT, D], cast to acc_dt
  q, k, v, g = (
    jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt).reshape(B, H, NT, BT, -1)
    for x in (q, k, v, g)
  )
  # q: [B, H, NT, BT, K]   k: [B, H, NT, BT, K]
  # v: [B, H, NT, BT, V]   g: [B, H, NT, BT, K]

  # [B, T, H] -> [B, H, NT, BT]
  beta = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt).reshape(B, H, NT, BT)  # [B, H, NT, BT]

  q = q * scale           # [B, H, NT, BT, K]
  g = g.cumsum(axis=3)    # [B, H, NT, BT, K]  cumulative gate within each chunk

  # =========================================================================
  # Step 1: Build interaction matrix A
  # A[..., c, i] = sum_k k[c,k] * exp(g[c,k] - g[i,k]) * k[i,k]
  # Note: diagonal is masked.
  # =========================================================================
  mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.bool_))  # [BT, BT]

  A = jnp.zeros((*q.shape[:-1], BT), dtype=acc_dt)  # [B, H, NT, BT, BT]
  for i in range(BT):
    k_i = k[..., i, :]         # [B, H, NT, K]
    g_i = g[..., i:i + 1, :]   # [B, H, NT, 1, K]
    # k * exp(g - g_i): [B, H, NT, BT, K],  einsum -> [B, H, NT, BT]
    A = A.at[..., i].set(
      jnp.einsum('...cd,...d->...c', k * jnp.exp(g - g_i), k_i)  # [B, H, NT, BT]
    )
  A = A * beta[..., None]  # [B, H, NT, BT, BT]  scale by beta

  # =========================================================================
  # Step 2: Solve lower-triangular dependency system
  # Forward substitution: A[i, :i] += A[i, :] @ A[:, :i]
  # =========================================================================
  A = jnp.where(mask, 0, -A)  # [B, H, NT, BT, BT]  zero upper triangle, negate lower
  for i in range(1, BT):
    # A[..., i, :, None]: [B, H, NT, BT, 1],  A[..., :, :i]: [B, H, NT, BT, i]
    update = jnp.einsum('...m,...mj->...j', A[..., i, :], A[..., :, :i])  # [B, H, NT, i]
    A = A.at[..., i, :i].set(A[..., i, :i] + update)       # [B, H, NT, i]
  # beta[..., None, :]: [B, H, NT, 1, BT]
  A = (A + jnp.eye(BT, dtype=acc_dt)) * beta[..., None, :]  # [B, H, NT, BT, BT]

  # =========================================================================
  # Step 3: Compute effective keys (w) and corrected values (u)
  #   u = A @ v  →  β·e when S=0 (intra-chunk only)
  #   w = A @ (exp(g)·k)  →  effective keys for inter-chunk correction
  #   v_corr = u - w @ S  →  β·e with inter-chunk state accounted for
  # =========================================================================
  # exp(g) * k: [B, H, NT, BT, K]
  w = A @ (jnp.exp(g) * k)   # [B, H, NT, BT, K]  effective keys
  u = A @ v                  # [B, H, NT, BT, V]  β·e (assuming S=0)

  # =========================================================================
  # Step 4: Inter-chunk recurrence with delta rule correction
  #   o_c = q_c·exp(G_c) @ S  +  A_qk @ v_corr   (inter + intra)
  #   S_new = S·exp(G_last)  +  Σ exp(G_last-G_j)·k_j·v_corr_j
  # =========================================================================
  S = jnp.zeros((B, H, K, V), dtype=acc_dt)  # [B, H, K, V] hidden state
  if initial_state is not None:
    S = S + initial_state.astype(acc_dt)      # [B, H, K, V]
  o = jnp.zeros_like(v)                       # [B, H, NT, BT, V] output buffer

  mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.bool_), k=1)  # [BT, BT] strict upper triangle
  for i in range(NT):
    q_i = q[:, :, i]   # [B, H, BT, K]
    k_i = k[:, :, i]   # [B, H, BT, K]
    u_i = u[:, :, i]   # [B, H, BT, V]
    g_i = g[:, :, i]   # [B, H, BT, K]
    w_i = w[:, :, i]   # [B, H, BT, K]

    # --- Intra-chunk attention matrix ---
    # A_qk[c,j] = (q_c · exp(G_c)) · (k_j · exp(-G_j))
    # query-key attention within chunk, causal masked (lower triangle + diagonal)
    A_qk = jnp.einsum(
      "...ck,...jk->...cj",
      q_i * jnp.exp(g_i),     # [B, H, BT, K]
      k_i * jnp.exp(-g_i),    # [B, H, BT, K]
    )  # [B, H, BT, BT]
    A_qk = jnp.where(mask, 0, A_qk)  # [B, H, BT, BT]

    # --- Delta rule correction ---
    # v_corr = β·e with inter-chunk state correction
    # u_i is β·e assuming S=0; w_i @ S corrects for historical state
    v_corr = u_i - w_i @ S  # [B, H, BT, V]

    # --- Output: o_c = q_c·exp(G_c) @ S  +  A_qk @ v_corr ---
    # ① inter-chunk: query reads historical state S
    # ② intra-chunk: query attends to corrected values within chunk
    o = o.at[:, :, i].set(
      jnp.einsum('...ck,...kv->...cv', q_i * jnp.exp(g_i), S)  # ① [B, H, BT, V]
      + A_qk @ v_corr                                           # ② [B, H, BT, V]
    )

    # --- Update state for next chunk ---
    g_last = g_i[:, :, -1]   # [B, H, K]  last gate in chunk
    # exp(g_last)[..., None]: [B, H, K, 1]
    S = S * jnp.exp(g_last)[..., None]  # [B, H, K, V]  decay state
    # g_last[:, :, None, :]: [B, H, 1, K],  exp(...) * k_i: [B, H, BT, K]
    S = S + jnp.einsum(
      '...ck,...cv->...kv',
      jnp.exp(g_last[:, :, None, :] - g_i) * k_i,  # [B, H, BT, K]
      v_corr,                                        # [B, H, BT, V]
    )  # [B, H, K, V]

  if not output_final_state:
    S = None

  # [B, H, NT, BT, V] -> [B, T, H, V], trim padding, cast back
  o = o.reshape(B, H, T, V)                                    # [B, H, T, V]
  o = jnp.transpose(o, (0, 2, 1, 3))[:, :T_orig].astype(orig_dtype)  # [B, T_orig, H, V]
  return o, S
