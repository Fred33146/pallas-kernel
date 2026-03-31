"""chunk_simple_gla: JAX CPU ref (tops.cpu.ops.simple_gla) vs FLA Triton GPU, multi-precision.

Tests:
  1. Intermediate dtype verification (no GPU)
  2. Cross-validation: naive vs chunk (no GPU)
  3. Backward cross-validation: chunk_bwd vs jax.grad(naive) (no GPU)
  4. CPU ref vs FLA Triton forward (GPU, when available)
  5. CPU ref vs FLA Triton backward (GPU, when available)
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

from tops.cpu.ops.simple_gla import naive_simple_gla, chunk_simple_gla
from tops.cpu.ops.simple_gla.chunk import chunk_simple_gla_bwd
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
  from fla.ops.simple_gla.naive import naive_recurrent_simple_gla as triton_naive
  from fla.ops.simple_gla import chunk_simple_gla as triton_chunk

  HAS_FLA = True
except ImportError:
  HAS_FLA = False

requires_triton = pytest.mark.skipif(
  not (HAS_CUDA and HAS_FLA),
  reason="Requires CUDA device and flash-linear-attention",
)

# ============================================================================
# Dtype / tolerance maps
# ============================================================================

_JAX_DTYPES = {
  "float64": jnp.float64,
  "float32": jnp.float32,
  "float16": jnp.float16,
  "bfloat16": jnp.bfloat16,
}

_DTYPE_TOLS = {
  "float32": dict(atol=5e-5, rtol=5e-5),
  "float16": dict(atol=7e-3, rtol=7e-3),
  "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

ALL_DTYPES = [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]

# Gate modes:
#   forward supports g+g_gamma combined ("both"), but chunk matches FLA's
#   variable-reuse behavior (differs from naive's correct multiplicative
#   composition), so naive-vs-chunk cross-validation excludes "both"
#   backward uses if/elif (mutually exclusive), no "both"
GATES_FWD = ["g", "g_gamma", "none"]
GATES_BWD = ["g", "g_gamma", "none"]
# Triton chunk comparison: g and g_gamma are mutually exclusive.
# "both" mode is excluded because FLA's chunk_fwd_h has a variable-reuse
# bug when both are active, causing state explosion.
GATES_TRITON = ["g", "g_gamma", "none"]

# ============================================================================
# Cross-validation shape configs (no GPU needed)
#
# Each dict: B, T, H, K, V, seed (required); h0 (optional)
# ============================================================================

_XVAL_SHAPES = [
  # ── standard ──
  dict(B=2, T=32, H=4, K=32, V=64, seed=42),
  dict(B=1, T=64, H=2, K=32, V=64, seed=7),
  # ── non-aligned T (padding test) ──
  dict(B=2, T=37, H=4, K=16, V=32, seed=40),
  # ── T < chunk_size edge case ──
  dict(B=2, T=1, H=4, K=32, V=64, seed=50),
  # ── with initial state ──
  dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
  # ── larger ──
  dict(B=2, T=64, H=4, K=32, V=64, seed=99),
]

# ============================================================================
# Triton comparison shape configs (GPU required)
# ============================================================================

_TRITON_SHAPES = [
  # ── standard ──
  dict(B=2, T=32, H=4, K=32, V=64, seed=42),
  dict(B=1, T=64, H=2, K=32, V=64, seed=7),
  # ── larger ──
  dict(B=2, T=64, H=4, K=32, V=64, seed=99),
  # ── with initial state ──
  dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
]

TRITON_CASES = [
  {**s, "dtype": d, **t} for s in _TRITON_SHAPES for d, t in _DTYPE_TOLS.items()
]

# Backward Triton comparison: fp32 + bf16 only.
# Backward accumulation order differs more than forward, so tolerances are
# looser. bf16 tolerances scale with number of chunks (more sequential
# accumulation → more float error drift).
_BWD_DTYPE_TOLS = {
    "float32": dict(atol=5e-4, rtol=5e-4),
    "bfloat16": dict(atol=5e-1, rtol=5e-1),
}

# B1_T64_H2 and B2_T64_H4 excluded from backward: FLA Triton backward
# kernel (chunk_bwd_kernel_dqkwg) triggers CUDA_ERROR_ILLEGAL_ADDRESS
# during autotuner benchmarking for T=64 shapes with bfloat16.
# T=64 backward correctness is still covered by the CPU-only
# cross-validation tests (test_chunk_bwd_vs_autograd_fp64).
_BWD_TRITON_SHAPES = [
  dict(B=2, T=32, H=4, K=32, V=64, seed=42),
  # dict(B=2, T=64, H=4, K=32, V=64, seed=99),  # excluded: CUDA_ERROR_ILLEGAL_ADDRESS with bfloat16
  dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
]

BWD_TRITON_CASES = [
    {**s, "dtype": d, **t}
    for s in _BWD_TRITON_SHAPES
    for d, t in _BWD_DTYPE_TOLS.items()
]


# ============================================================================
# ID generators
# ============================================================================


def _shape_id(c):
  parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
  if c.get("h0"):
    parts.append("h0")
  return "-".join(parts)


def _case_id(c):
  parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}"]
  if c.get("h0"):
    parts.append("h0")
  return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False, gate="g"):
  """Generate test inputs.

  Args:
      gate: 'g', 'g_gamma', 'both', 'none'
  """
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 6)
  q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
  k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
  v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
  acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
  # g: always negative for stability (logsigmoid)
  g_val = (
    jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H))).astype(dtype)
    if gate in ("g", "both")
    else None
  )
  g_gamma_val = (
    -jax.nn.softplus(jax.random.normal(keys[4], (H,))).astype(acc)
    if gate in ("g_gamma", "both")
    else None
  )
  h0_arr = jax.random.normal(keys[5], (B, H, K, V), dtype=acc) if h0 else None
  return q, k, v, g_val, g_gamma_val, h0_arr


def _torch_to_jax(t, dtype):
  """Convert torch tensor to JAX array via fp32 intermediate."""
  return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


# ============================================================================
# 1. Dtype verification (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("gate", GATES_FWD)
def test_naive_dtypes(dtype, gate):
  """Verify output and state dtypes match FLA contract."""
  B, T, H, K, V = 2, 32, 4, 32, 64
  q, k, v, g, g_gamma, _ = _make_inputs(B, T, H, K, V, dtype, gate=gate)
  o, ht = naive_simple_gla(q, k, v, g=g, g_gamma=g_gamma, output_final_state=True)
  acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
  assert o.dtype == dtype, f"o.dtype={o.dtype}, expected {dtype}"
  assert ht.dtype == acc, f"ht.dtype={ht.dtype}, expected {acc}"


def test_naive_output_final_state_false():
  """When output_final_state=False, final_state should be None."""
  B, T, H, K, V = 2, 32, 4, 32, 64
  q, k, v, g, _, _ = _make_inputs(B, T, H, K, V, jnp.float32, gate="g")
  o, ht = naive_simple_gla(q, k, v, g=g, output_final_state=False)
  assert ht is None


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("gate", GATES_FWD)
def test_chunk_dtypes(dtype, gate):
  """Verify chunk output and state dtypes match FLA contract."""
  B, T, H, K, V = 2, 64, 4, 32, 64
  q, k, v, g, g_gamma, _ = _make_inputs(B, T, H, K, V, dtype, gate=gate)
  o, ht = chunk_simple_gla(q, k, v, g=g, g_gamma=g_gamma, output_final_state=True)
  acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
  assert o.dtype == dtype
  assert ht.dtype == acc


# ============================================================================
# 2. Cross-validation: naive vs chunk (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("gate", GATES_FWD)
@pytest.mark.parametrize("cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES])
def test_naive_vs_chunk_fp64(cfg, gate):
  """Cross-validate naive vs chunk in fp64 (high precision)."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  q, k, v, g, g_gamma, h0 = _make_inputs(
    B, T, H, K, V, jnp.float64, seed=cfg["seed"], gate=gate, h0=cfg.get("h0", False)
  )
  o_naive, s_naive = naive_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  o_chunk, s_chunk = chunk_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  assert compare_tensor(
    "output", o_naive, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
  )
  if s_naive is not None:
    assert compare_tensor(
      "state", s_naive, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64
    )


@pytest.mark.parametrize("gate", GATES_FWD)
@pytest.mark.parametrize("cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES])
def test_naive_vs_chunk_fp32(cfg, gate):
  """Cross-validate naive vs chunk in fp32."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  q, k, v, g, g_gamma, h0 = _make_inputs(
    B, T, H, K, V, jnp.float32, seed=cfg["seed"], gate=gate, h0=cfg.get("h0", False)
  )
  o_naive, s_naive = naive_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  o_chunk, s_chunk = chunk_simple_gla(
    q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=True
  )
  assert compare_tensor(
    "output", o_naive, o_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64
  )
  if s_naive is not None:
    assert compare_tensor(
      "state", s_naive, s_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64
    )


# ============================================================================
# 3. Backward cross-validation: chunk_bwd vs jax.grad(naive) (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("gate", GATES_BWD)
@pytest.mark.parametrize("cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES])
def test_chunk_bwd_vs_autograd_fp64(cfg, gate):
  """fp64: chunk backward should match jax.grad(naive) to high precision."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  q, k, v, g, g_gamma, h0 = _make_inputs(
    B, T, H, K, V, jnp.float64, seed=cfg["seed"], gate=gate, h0=cfg.get("h0", False)
  )

  # Reference: jax.grad on naive_simple_gla
  def loss_fn(q, k, v, g, g_gamma, h0):
    o, _ = naive_simple_gla(
      q, k, v, g=g, g_gamma=g_gamma, initial_state=h0, output_final_state=False
    )
    return o.sum()

  argnums = [0, 1, 2]
  if g is not None:
    argnums.append(3)
  ref_grads = jax.grad(loss_fn, argnums=argnums)(q, k, v, g, g_gamma, h0)

  # chunk backward
  from tops.cpu.ops.simple_gla.chunk import chunk_simple_gla_bwd

  scale = K**-0.5
  do = jnp.ones((B, T, H, V), dtype=jnp.float64)
  dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
    q,
    k,
    v,
    g,
    g_gamma,
    h0,
    do=do,
    dht=None,
    scale=scale,
    chunk_size=16,
  )

  assert compare_tensor(
    "dq", ref_grads[0], dq, atol=1e-8, rtol=1e-8, compare_dtype=np.float64
  )
  assert compare_tensor(
    "dk", ref_grads[1], dk, atol=1e-8, rtol=1e-8, compare_dtype=np.float64
  )
  assert compare_tensor(
    "dv", ref_grads[2], dv, atol=1e-8, rtol=1e-8, compare_dtype=np.float64
  )
  if g is not None:
    assert compare_tensor(
      "dg", ref_grads[3], dg, atol=1e-8, rtol=1e-8, compare_dtype=np.float64
    )
  else:
    assert dg is None

  if gate == "g_gamma":
    assert dg is None, "dg must be None when only g_gamma is provided"


# ============================================================================
# 4. CPU ref vs FLA Triton (GPU, when available)
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", TRITON_CASES, ids=[_case_id(c) for c in TRITON_CASES])
def test_cpu_naive_vs_triton_naive(cfg):
  """Compare CPU naive vs FLA Triton naive (g mode only).

  FLA's naive_recurrent_simple_gla only supports the g (per-head scalar gate)
  mode, so g_gamma is not tested here.
  """
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  atol, rtol = cfg["atol"], cfg["rtol"]
  jax_dtype = _JAX_DTYPES[cfg["dtype"]]
  triton_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
  }[cfg["dtype"]]

  torch.manual_seed(cfg["seed"])
  q_t = torch.randn(B, T, H, K).to(triton_dtype)
  k_t = torch.randn(B, T, H, K).to(triton_dtype)
  v_t = torch.randn(B, T, H, V).to(triton_dtype)
  g_t = F.logsigmoid(torch.randn(B, T, H)).to(triton_dtype)
  h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

  # Run Triton
  kwargs = dict(output_final_state=True)
  if h0_t is not None:
    kwargs["initial_state"] = h0_t.float().cuda()
  o_tri, s_tri = triton_naive(q_t.cuda(), k_t.cuda(), v_t.cuda(), g_t.cuda(), **kwargs)
  o_tri = o_tri.cpu()
  s_tri = s_tri.cpu() if s_tri is not None else None

  # Run CPU
  o_cpu, s_cpu = naive_simple_gla(
    _torch_to_jax(q_t, jax_dtype),
    _torch_to_jax(k_t, jax_dtype),
    _torch_to_jax(v_t, jax_dtype),
    g=_torch_to_jax(g_t, jax_dtype),
    initial_state=_torch_to_jax(h0_t, jnp.float32) if h0_t is not None else None,
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


@requires_triton
@pytest.mark.parametrize("gate", GATES_TRITON)
@pytest.mark.parametrize("cfg", TRITON_CASES, ids=[_case_id(c) for c in TRITON_CASES])
def test_cpu_chunk_vs_triton_chunk(cfg, gate):
  """Compare CPU chunk vs FLA Triton chunk (g, g_gamma, none gate modes)."""
  B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
  atol, rtol = cfg["atol"], cfg["rtol"]
  jax_dtype = _JAX_DTYPES[cfg["dtype"]]
  triton_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
  }[cfg["dtype"]]

  torch.manual_seed(cfg["seed"])
  q_t = torch.randn(B, T, H, K).to(triton_dtype)
  k_t = torch.randn(B, T, H, K).to(triton_dtype)
  v_t = torch.randn(B, T, H, V).to(triton_dtype)

  g_t = (
    F.logsigmoid(torch.randn(B, T, H)).to(triton_dtype)
    if gate in ("g", "both")
    else None
  )
  g_gamma_t = (
    -F.softplus(torch.randn(H)).float() if gate in ("g_gamma", "both") else None
  )
  h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

  # Run Triton
  kwargs = dict(output_final_state=True)
  if h0_t is not None:
    kwargs["initial_state"] = h0_t.float().cuda()
  o_tri, s_tri = triton_chunk(
    q_t.cuda(),
    k_t.cuda(),
    v_t.cuda(),
    g=g_t.cuda() if g_t is not None else None,
    g_gamma=g_gamma_t.cuda() if g_gamma_t is not None else None,
    **kwargs,
  )
  o_tri = o_tri.cpu()
  s_tri = s_tri.cpu() if s_tri is not None else None

  # Run CPU
  o_cpu, s_cpu = chunk_simple_gla(
    _torch_to_jax(q_t, jax_dtype),
    _torch_to_jax(k_t, jax_dtype),
    _torch_to_jax(v_t, jax_dtype),
    g=_torch_to_jax(g_t, jax_dtype) if g_t is not None else None,
    g_gamma=_torch_to_jax(g_gamma_t, jnp.float32) if g_gamma_t is not None else None,
    initial_state=_torch_to_jax(h0_t, jnp.float32) if h0_t is not None else None,
    output_final_state=True,
  )

  assert compare_tensor(
    "output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64
  )
  if s_tri is not None:
    assert compare_tensor(
      "final_state", s_tri, s_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64
    )


# ============================================================================
# 5. CPU ref vs FLA Triton backward (GPU, when available)
# ============================================================================


@requires_triton
@pytest.mark.parametrize("gate", GATES_BWD)
@pytest.mark.parametrize("cfg", BWD_TRITON_CASES, ids=[_case_id(c) for c in BWD_TRITON_CASES])
def test_cpu_chunk_bwd_vs_triton_chunk_bwd(cfg, gate):
    """Compare CPU chunk backward vs FLA Triton backward (via autograd).

    Triton gradients are obtained by calling loss.backward() on the FLA
    forward output.  CPU gradients come from chunk_simple_gla_bwd directly.
    """
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    jax_dtype = _JAX_DTYPES[cfg["dtype"]]
    triton_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg["dtype"]]

    torch.manual_seed(cfg["seed"])
    q_t = torch.randn(B, T, H, K, dtype=triton_dtype)
    k_t = torch.randn(B, T, H, K, dtype=triton_dtype)
    v_t = torch.randn(B, T, H, V, dtype=triton_dtype)
    do_t = torch.randn(B, T, H, V, dtype=triton_dtype)

    g_t = F.logsigmoid(torch.randn(B, T, H)).to(triton_dtype) if gate == "g" else None
    g_gamma_t = -F.softplus(torch.randn(H)).float() if gate == "g_gamma" else None
    h0_t = torch.randn(B, H, K, V).float() if cfg.get("h0") else None

    # ── Triton forward + backward via autograd ──
    q_g = q_t.clone().cuda().requires_grad_()
    k_g = k_t.clone().cuda().requires_grad_()
    v_g = v_t.clone().cuda().requires_grad_()
    g_g = g_t.clone().cuda().requires_grad_() if g_t is not None else None
    do_g = do_t.clone().cuda()
    g_gamma_g = g_gamma_t.clone().cuda() if g_gamma_t is not None else None
    h0_g = h0_t.clone().cuda() if h0_t is not None else None

    o_g, _ = triton_chunk(
        q_g, k_g, v_g,
        g=g_g, g_gamma=g_gamma_g,
        initial_state=h0_g, output_final_state=False,
    )
    loss = (o_g * do_g).sum()
    loss.backward()

    dq_tri = q_g.grad.cpu()
    dk_tri = k_g.grad.cpu()
    dv_tri = v_g.grad.cpu()
    dg_tri = g_g.grad.cpu() if g_g is not None else None

    # ── CPU JAX backward ──
    q_j = _torch_to_jax(q_t, jax_dtype)
    k_j = _torch_to_jax(k_t, jax_dtype)
    v_j = _torch_to_jax(v_t, jax_dtype)
    do_j = _torch_to_jax(do_t, jax_dtype)
    g_j = _torch_to_jax(g_t, jax_dtype) if g_t is not None else None
    g_gamma_j = _torch_to_jax(g_gamma_t, jnp.float32) if g_gamma_t is not None else None
    h0_j = _torch_to_jax(h0_t, jnp.float32) if h0_t is not None else None

    scale = K ** -0.5
    dq_j, dk_j, dv_j, dg_j, dh0_j = chunk_simple_gla_bwd(
        q_j, k_j, v_j, g_j, g_gamma_j, h0_j,
        do=do_j, dht=None, scale=scale,
    )

    assert compare_tensor("dq", dq_tri, dq_j, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_tri, dk_j, atol=atol, rtol=rtol)
    assert compare_tensor("dv", dv_tri, dv_j, atol=atol, rtol=rtol)
    if dg_tri is not None:
        assert compare_tensor("dg", dg_tri, dg_j, atol=atol, rtol=rtol)
    else:
        assert dg_j is None


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
