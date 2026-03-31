"""JAX CPU reference for Simple GLA naive recurrent operations with FLA-exact dtype behavior.

Precisely matches the FLA naive recurrent Simple GLA dtype behavior:

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  Internal computation:
    q, k, v, g: cast to fp32 for computation     [fp64 mode: fp64]
    h (hidden state): fp32 accumulator             [fp64 mode: fp64]
    o (output buffer): fp32 during computation     [fp64 mode: fp64]
  Final output:
    o: cast back to original input dtype           [fp64 mode: fp64]
    final_state h: fp32                            [fp64 mode: fp64]

Simple GLA vs GLA:
  - GLA uses per-element gates gk: [B, T, H, K]
  - Simple GLA uses per-head scalar gates g: [B, T, H] broadcast over K
  - Optionally combines with fixed per-head decay g_gamma: [H]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype


@cpu_reference
def naive_simple_gla(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None = None,
  g_gamma: jax.Array | None = None,
  scale: float | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Naive recurrent Simple GLA — JAX CPU reference with FLA-exact dtype behavior.

  Simple GLA uses per-head scalar gates (g: [B,T,H]) instead of per-element
  gates (gk: [B,T,H,K]) as in standard GLA. The scalar gate is broadcast
  over both K and V dimensions when updating the hidden state.

  Core recurrence (per timestep):
      gate_t = exp(g_t) [* exp(g_gamma)]       scalar per (B, H)
      h_t = h_{t-1} * gate_t + k_t^T @ v_t    gate broadcast over [K, V]
      o_t = (q_t * scale)^T @ h_t              sum over K dimension

  Dtype behavior (matching FLA):
    - All inputs cast to fp32 for computation (`.float()` in PyTorch)
    - Hidden state h is fp32 accumulator
    - Output o computed in fp32, cast back to original dtype
    - Final state h stays in fp32
    - fp64 mode: all computation in fp64, no precision cast

  Uses ``jax.lax.scan`` for the time-step loop so that ``jax.grad`` can
  differentiate the recurrence efficiently (reverse-mode scan) instead of
  unrolling T Python loop iterations into the computation graph.

  Args:
      q:               [B, T, H, K] — Queries
      k:               [B, T, H, K] — Keys
      v:               [B, T, H, V] — Values
      g:               [B, T, H]    — Per-head scalar gate in log-space (e.g., logsigmoid)
                       Optional. If None and g_gamma is None, gate defaults to 1.0.
      g_gamma:         [H]          — Fixed per-head log-decay, applied multiplicatively
                       with g (i.e., combined gate = exp(g + g_gamma)).
                       Must be in acc_dtype (fp32 or fp64). Optional.
      scale:           Scalar query scale. Defaults to K ** -0.5.
      initial_state:   [B, H, K, V] — Initial hidden state. Optional.
      output_final_state: Whether to return the final hidden state.

  Returns:
      o:           [B, T, H, V] — Output (original input dtype)
      final_state: [B, H, K, V] in fp32 (or fp64), or None
  """
  orig_dtype = q.dtype
  acc_dt = _acc_dtype(orig_dtype)

  # Shape assertions (project coding standard)
  assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
  assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
  assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
    f"v shape {v.shape} incompatible with q shape {q.shape}"
  )
  if g is not None:
    assert g.ndim == 3 and g.shape == q.shape[:3], f"g shape {g.shape} != {q.shape[:3]}"
  if g_gamma is not None:
    assert g_gamma.ndim == 1 and g_gamma.shape[0] == q.shape[2], (
      f"g_gamma shape {g_gamma.shape} != (H={q.shape[2]},)"
    )

  # FLA: q, k, v = map(lambda x: x.transpose(1, 2).float(), ...)
  # Transpose [B, T, H, D] -> [B, H, T, D] and cast to accumulator dtype
  q_f, k_f, v_f = (jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt) for x in (q, k, v))
  B, H, T, K = q_f.shape
  V = v_f.shape[-1]

  # Default scale: K ** -0.5
  if scale is None:
    scale = K**-0.5

  # Precompute per-step gate: [B, H, T] → exp → [B, H, T]
  # When no gate is active, use ones (multiplication identity).
  if g is not None:
    g_f = jnp.transpose(g, (0, 2, 1)).astype(acc_dt)  # [B, H, T]
    if g_gamma is not None:
      g_gamma_f = g_gamma.astype(acc_dt)
      gate = jnp.exp(g_f + g_gamma_f[None, :, None])  # [B, H, T]
    else:
      gate = jnp.exp(g_f)  # [B, H, T]
  elif g_gamma is not None:
    g_gamma_f = g_gamma.astype(acc_dt)
    gate = jnp.broadcast_to(jnp.exp(g_gamma_f)[None, :, None], (B, H, T))  # [B, H, T]
  else:
    gate = jnp.ones((B, H, T), dtype=acc_dt)

  # Initial hidden state
  h0 = jnp.zeros((B, H, K, V), dtype=acc_dt)
  if initial_state is not None:
    h0 = h0 + initial_state.astype(acc_dt)

  # Prepare scan inputs: move T axis to leading position
  gate_seq = jnp.transpose(gate, (2, 0, 1))  # [T, B, H]
  k_seq = jnp.transpose(k_f, (2, 0, 1, 3))  # [T, B, H, K]
  v_seq = jnp.transpose(v_f, (2, 0, 1, 3))  # [T, B, H, V]
  q_seq = jnp.transpose(q_f * scale, (2, 0, 1, 3))  # [T, B, H, K]

  def step(h, xs):
    gate_t, k_t, v_t, q_t = xs
    # gate_t: [B, H], k_t: [B, H, K], v_t: [B, H, V], q_t: [B, H, K]
    kv_t = k_t[..., None] * v_t[..., None, :]  # [B, H, K, V]
    h = h * gate_t[:, :, None, None] + kv_t  # [B, H, K, V]
    o_t = (q_t[..., None] * h).sum(-2)  # [B, H, V]
    return h, o_t

  h_final, o_seq = jax.lax.scan(step, h0, (gate_seq, k_seq, v_seq, q_seq))
  # o_seq: [T, B, H, V] → [B, T, H, V]
  o = jnp.transpose(o_seq, (1, 0, 2, 3))

  final_state = h_final if output_final_state else None
  return o.astype(orig_dtype), final_state
