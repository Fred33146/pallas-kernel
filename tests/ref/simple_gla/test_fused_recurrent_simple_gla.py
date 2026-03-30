"""fused_recurrent_simple_gla: JAX CPU ref (tops.cpu.ops.simple_gla) tests.

Tests:
  1. Dtype verification (no GPU)
  2. Cross-validation: fused_recurrent vs naive (no GPU)
  3. Cross-validation: fused_recurrent vs chunk_simple_gla (no GPU)
  4. Varlen: fused_recurrent with cu_seqlens vs per-segment calls (no GPU)
  5. Reverse mode produces different output from forward (no GPU)
  6. CPU ref vs FLA Triton (GPU, when available)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp

from tops.cpu.ops.simple_gla import (
  naive_simple_gla,
  chunk_simple_gla,
  fused_recurrent_simple_gla,
)
from tests.utils import compare_tensor

HAS_CUDA = False
try:
  import torch
  import torch.nn.functional as F

  HAS_CUDA = torch.cuda.is_available()
  if HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except ImportError:
  pass

try:
  from fla.ops.simple_gla import fused_recurrent_simple_gla as triton_fused_recurrent

  HAS_FLA = True
except ImportError:
  HAS_FLA = False

requires_triton = pytest.mark.skipif(
  not (HAS_CUDA and HAS_FLA),
  reason="Requires CUDA device and flash-linear-attention",
)

# ============================================================================
# Dtype maps and gate modes
# ============================================================================

_JAX_DTYPES = {
  "float64": jnp.float64,
  "float32": jnp.float32,
  "float16": jnp.float16,
  "bfloat16": jnp.bfloat16,
}

ALL_DTYPES = [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]
GATES = ["g", "g_gamma"]


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False, gate="g"):
  """Generate test inputs.

  Args:
      gate: 'g' (per-step scalar), 'g_gamma' (fixed decay)
  """
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 6)
  q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
  k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
  v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
  acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
  g_val = (
    jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H))).astype(dtype)
    if gate == "g"
    else None
  )
  g_gamma_val = (
    -jax.nn.softplus(jax.random.normal(keys[4], (H,))).astype(acc)
    if gate == "g_gamma"
    else None
  )
  h0_arr = jax.random.normal(keys[5], (B, H, K, V), dtype=acc) if h0 else None
  return q, k, v, g_val, g_gamma_val, h0_arr


def _torch_to_jax(t, dtype):
  return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


# ============================================================================
# 1. Dtype verification (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("gate", GATES)
def test_fused_recurrent_dtypes(dtype, gate):
  """Output dtype should match q.dtype; final state should be fp32/fp64."""
  B, T, H, K, V = 2, 32, 4, 32, 64
  q, k, v, g, g_gamma, _ = _make_inputs(B, T, H, K, V, dtype, gate=gate)
  o, ht = fused_recurrent_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, output_final_state=True
  )
  acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
  assert o.dtype == dtype, f"o.dtype={o.dtype}, expected {dtype}"
  assert ht.dtype == acc, f"ht.dtype={ht.dtype}, expected {acc}"


def test_fused_recurrent_no_final_state():
  """output_final_state=False should return None final state."""
  B, T, H, K, V = 2, 32, 4, 32, 64
  q, k, v, g, _, _ = _make_inputs(B, T, H, K, V, jnp.float32, gate="g")
  o, ht = fused_recurrent_simple_gla(q, k, v, g=g, output_final_state=False)
  assert ht is None


# ============================================================================
# 2. Cross-validation: fused_recurrent vs naive (no GPU needed)
# ============================================================================

_CROSS_SHAPES = [
  dict(B=2, T=32, H=4, K=32, V=64, seed=42),
  dict(B=1, T=64, H=2, K=32, V=64, seed=7),
  dict(B=2, T=37, H=4, K=16, V=32, seed=40),
  dict(B=2, T=1, H=4, K=32, V=64, seed=50),
  dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
  dict(B=2, T=64, H=4, K=32, V=64, seed=99),
]


def _cross_id(c):
  parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
  if c.get("h0"):
    parts.append("h0")
  return "-".join(parts)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize(
  "cfg", _CROSS_SHAPES, ids=[_cross_id(c) for c in _CROSS_SHAPES]
)
def test_fused_recurrent_vs_naive_fp64(cfg, gate):
  """fp64: fused_recurrent should match naive to near machine precision."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  q, k, v, g, g_gamma, h0 = _make_inputs(
    B, T, H, K, V, jnp.float64, seed=cfg["seed"], gate=gate, h0=cfg.get("h0", False)
  )
  # naive expects initial_state [B, H, K, V]; fused_recurrent expects [N, H, K, V]
  # In non-varlen, N=B so they're equivalent.
  o_naive, s_naive = naive_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  o_fr, s_fr = fused_recurrent_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  assert compare_tensor(
    "output", o_naive, o_fr, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
  )
  assert compare_tensor(
    "state", s_naive, s_fr, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
  )


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize(
  "cfg", _CROSS_SHAPES, ids=[_cross_id(c) for c in _CROSS_SHAPES]
)
def test_fused_recurrent_vs_naive_fp32(cfg, gate):
  """fp32: fused_recurrent should match naive within fp32 tolerance."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  q, k, v, g, g_gamma, h0 = _make_inputs(
    B, T, H, K, V, jnp.float32, seed=cfg["seed"], gate=gate, h0=cfg.get("h0", False)
  )
  o_naive, s_naive = naive_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  o_fr, s_fr = fused_recurrent_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  assert compare_tensor(
    "output", o_naive, o_fr, atol=1e-5, rtol=1e-5, compare_dtype=np.float64
  )
  assert compare_tensor(
    "state", s_naive, s_fr, atol=1e-5, rtol=1e-5, compare_dtype=np.float64
  )


# ============================================================================
# 3. Cross-validation: fused_recurrent vs chunk_simple_gla (no GPU needed)
# ============================================================================


def test_fused_recurrent_vs_chunk_fp64():
  """fp64: fused_recurrent and chunk should agree."""
  B, T, H, K, V = 2, 64, 4, 32, 64
  q, k, v, g, _, h0 = _make_inputs(
    B, T, H, K, V, jnp.float64, seed=42, gate="g", h0=True
  )

  o_fr, s_fr = fused_recurrent_simple_gla(
    q, k, v, g=g, initial_state=h0, output_final_state=True
  )
  o_chunk, s_chunk = chunk_simple_gla(
    q, k, v, g=g, initial_state=h0, output_final_state=True
  )
  assert compare_tensor(
    "output", o_fr, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
  )
  assert compare_tensor(
    "state", s_fr, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
  )


# ============================================================================
# 4. Varlen: cu_seqlens vs per-segment calls (no GPU needed)
# ============================================================================


def test_fused_recurrent_cu_seqlens_fp64_g():
  """Varlen (g gate): cu_seqlens result should match per-segment calls."""
  B, H, K, V = 1, 2, 32, 64
  T1, T2 = 20, 30
  T_total = T1 + T2

  q, k, v, g, _, _ = _make_inputs(B, T_total, H, K, V, jnp.float64, seed=42, gate="g")
  cu_seqlens = np.array([0, T1, T_total])
  h0 = jax.random.normal(jax.random.PRNGKey(99), (2, H, K, V), dtype=jnp.float64)

  o_var, s_var = fused_recurrent_simple_gla(
    q, k, v, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens
  )

  o1, s1 = fused_recurrent_simple_gla(
    q[:, :T1],
    k[:, :T1],
    v[:, :T1],
    g=g[:, :T1],
    initial_state=h0[0:1],
    output_final_state=True,
  )
  o2, s2 = fused_recurrent_simple_gla(
    q[:, T1:],
    k[:, T1:],
    v[:, T1:],
    g=g[:, T1:],
    initial_state=h0[1:2],
    output_final_state=True,
  )
  o_manual = jnp.concatenate([o1, o2], axis=1)
  s_manual = jnp.concatenate([s1, s2], axis=0)

  np.testing.assert_allclose(
    np.array(o_var), np.array(o_manual), atol=1e-12, rtol=1e-12
  )
  np.testing.assert_allclose(
    np.array(s_var), np.array(s_manual), atol=1e-12, rtol=1e-12
  )


def test_fused_recurrent_cu_seqlens_fp64_g_gamma():
  """Varlen (g_gamma gate): cu_seqlens result should match per-segment calls."""
  B, H, K, V = 1, 2, 32, 64
  T1, T2 = 15, 25
  T_total = T1 + T2

  q, k, v, _, g_gamma, _ = _make_inputs(
    B, T_total, H, K, V, jnp.float64, seed=77, gate="g_gamma"
  )
  cu_seqlens = np.array([0, T1, T_total])
  h0 = jax.random.normal(jax.random.PRNGKey(88), (2, H, K, V), dtype=jnp.float64)

  o_var, s_var = fused_recurrent_simple_gla(
    q,
    k,
    v,
    g_gamma=g_gamma,
    initial_state=h0,
    output_final_state=True,
    cu_seqlens=cu_seqlens,
  )

  o1, s1 = fused_recurrent_simple_gla(
    q[:, :T1],
    k[:, :T1],
    v[:, :T1],
    g_gamma=g_gamma,
    initial_state=h0[0:1],
    output_final_state=True,
  )
  o2, s2 = fused_recurrent_simple_gla(
    q[:, T1:],
    k[:, T1:],
    v[:, T1:],
    g_gamma=g_gamma,
    initial_state=h0[1:2],
    output_final_state=True,
  )
  o_manual = jnp.concatenate([o1, o2], axis=1)
  s_manual = jnp.concatenate([s1, s2], axis=0)

  np.testing.assert_allclose(
    np.array(o_var), np.array(o_manual), atol=1e-12, rtol=1e-12
  )
  np.testing.assert_allclose(
    np.array(s_var), np.array(s_manual), atol=1e-12, rtol=1e-12
  )


# ============================================================================
# 5. Reverse mode (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("gate", GATES)
def test_fused_recurrent_reverse_differs(gate):
  """Reverse mode should produce different results from forward mode."""
  B, T, H, K, V = 2, 32, 4, 32, 64
  q, k, v, g, g_gamma, _ = _make_inputs(B, T, H, K, V, jnp.float64, seed=42, gate=gate)

  o_fwd, _ = fused_recurrent_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, reverse=False, output_final_state=True
  )
  o_rev, _ = fused_recurrent_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, reverse=True, output_final_state=True
  )
  assert not np.allclose(np.array(o_fwd), np.array(o_rev), atol=1e-6)


# ============================================================================
# 6. CPU ref vs FLA Triton (GPU, when available)
# ============================================================================

_TRITON_SHAPES = [
  dict(B=2, T=32, H=4, K=32, V=64, seed=42),
  dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
  dict(B=1, T=64, H=2, K=32, V=64, seed=7),
  dict(B=2, T=64, H=4, K=32, V=64, seed=99),
]

_TRITON_DTYPE_TOLS = {
  "float32": dict(atol=5e-5, rtol=5e-5),
  "float16": dict(atol=5e-3, rtol=5e-3),
  "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

TRITON_CASES = [
  {**s, "dtype": d, **t, "gate": g}
  for s in _TRITON_SHAPES
  for d, t in _TRITON_DTYPE_TOLS.items()
  for g in GATES
]


def _triton_case_id(c):
  parts = [
    f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}_{c['gate']}"
  ]
  if c.get("h0"):
    parts.append("h0")
  return "-".join(parts)


@requires_triton
@pytest.mark.parametrize(
  "cfg", TRITON_CASES, ids=[_triton_case_id(c) for c in TRITON_CASES]
)
def test_cpu_vs_triton_fwd(cfg):
  """Compare CPU fused_recurrent vs FLA Triton fused_recurrent_simple_gla."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  atol, rtol = cfg["atol"], cfg["rtol"]
  gate = cfg["gate"]
  jax_dtype = _JAX_DTYPES[cfg["dtype"]]
  torch_dtype = getattr(torch, cfg["dtype"])

  torch.manual_seed(cfg["seed"])
  q_t = torch.randn(B, T, H, K).to(torch_dtype)
  k_t = torch.randn(B, T, H, K).to(torch_dtype)
  v_t = torch.randn(B, T, H, V).to(torch_dtype)
  g_t = F.logsigmoid(torch.randn(B, T, H)).to(torch_dtype) if gate == "g" else None
  g_gamma_t = -F.softplus(torch.randn(H)).float() if gate == "g_gamma" else None
  h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

  kwargs = dict(output_final_state=True)
  if h0_t is not None:
    kwargs["initial_state"] = h0_t.float().cuda()
  o_tri, s_tri = triton_fused_recurrent(
    q_t.cuda(),
    k_t.cuda(),
    v_t.cuda(),
    g=g_t.cuda() if g_t is not None else None,
    g_gamma=g_gamma_t.cuda() if g_gamma_t is not None else None,
    **kwargs,
  )
  o_tri = o_tri.cpu()
  s_tri = s_tri.cpu() if s_tri is not None else None

  q_j = _torch_to_jax(q_t, jax_dtype)
  k_j = _torch_to_jax(k_t, jax_dtype)
  v_j = _torch_to_jax(v_t, jax_dtype)
  g_j = _torch_to_jax(g_t, jax_dtype) if g_t is not None else None
  g_gamma_j = _torch_to_jax(g_gamma_t, jnp.float32) if g_gamma_t is not None else None
  h0_j = _torch_to_jax(h0_t, jnp.float32) if h0_t is not None else None

  o_cpu, s_cpu = fused_recurrent_simple_gla(
    q_j,
    k_j,
    v_j,
    g=g_j,
    g_gamma=g_gamma_j,
    initial_state=h0_j,
    output_final_state=True,
  )

  assert compare_tensor(
    "output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64
  )
  if s_tri is not None:
    assert compare_tensor(
      "final_state", s_tri, s_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64
    )
  assert o_cpu.dtype == jax_dtype


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
