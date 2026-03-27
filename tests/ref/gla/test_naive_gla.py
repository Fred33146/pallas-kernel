"""naive_recurrent_gla / fused_chunk_gla: JAX CPU ref (tops.cpu.ops.gla) tests.

Tests:
  1. Intermediate dtype verification (no GPU)
  2. Cross-validation: naive vs chunk_gla (no GPU)
  3. Cross-validation: fused_chunk_gla vs chunk_gla (no GPU)
  4. CPU ref vs FLA Triton (GPU, when available)
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

from tops.cpu.ops.gla import naive_recurrent_gla, fused_chunk_gla, chunk_gla


# ============================================================================
# Helpers
# ============================================================================


def _make_jax_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H, K))).astype(dtype)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    h0_arr = jax.random.normal(keys[4], (B, H, K, V), dtype=acc) if h0 else None
    return q, k, v, g, h0_arr


# ============================================================================
# 1. Intermediate dtype verification (no GPU needed)
# ============================================================================


def test_naive_dtypes_bf16():
    """bf16 input: internal fp32 computation, output cast to bf16, state fp32."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.bfloat16)
    o, ht = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    assert o.dtype == jnp.bfloat16, f"o.dtype={o.dtype}, expected bfloat16"
    assert ht.dtype == jnp.float32, f"ht.dtype={ht.dtype}, expected float32"


def test_naive_dtypes_fp16():
    """fp16 input: internal fp32 computation, output cast to fp16, state fp32."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float16)
    o, ht = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    assert o.dtype == jnp.float16, f"o.dtype={o.dtype}, expected float16"
    assert ht.dtype == jnp.float32, f"ht.dtype={ht.dtype}, expected float32"


def test_naive_dtypes_fp32():
    """fp32 input: all fp32."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float32)
    o, ht = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    assert o.dtype == jnp.float32, f"o.dtype={o.dtype}, expected float32"
    assert ht.dtype == jnp.float32, f"ht.dtype={ht.dtype}, expected float32"


def test_naive_dtypes_fp64():
    """fp64 input: all fp64, no precision cast."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float64)
    o, ht = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    assert o.dtype == jnp.float64, f"o.dtype={o.dtype}, expected float64"
    assert ht.dtype == jnp.float64, f"ht.dtype={ht.dtype}, expected float64"


def test_naive_output_final_state_false():
    """When output_final_state=False, final_state should be None."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float32)
    o, ht = naive_recurrent_gla(q, k, v, g, output_final_state=False)
    assert ht is None


# ============================================================================
# 2. Cross-validation: naive vs chunk_gla (no GPU needed)
# ============================================================================


_NAIVE_VS_CHUNK_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]


def _nc_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


@pytest.mark.parametrize(
    "cfg", _NAIVE_VS_CHUNK_SHAPES,
    ids=[_nc_case_id(c) for c in _NAIVE_VS_CHUNK_SHAPES],
)
def test_naive_vs_chunk_fp64(cfg):
    """fp64: naive and chunk should agree to near machine precision."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, g, h0 = _make_jax_inputs(
        B, T, H, K, V, jnp.float64, seed=cfg["seed"], h0=cfg.get("h0", False),
    )
    o_naive, s_naive = naive_recurrent_gla(
        q, k, v, g, initial_state=h0, output_final_state=True,
    )
    o_chunk, s_chunk = chunk_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True,
    )
    from tests.utils import compare_tensor

    assert compare_tensor("output", o_naive, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


@pytest.mark.parametrize(
    "cfg", _NAIVE_VS_CHUNK_SHAPES,
    ids=[_nc_case_id(c) for c in _NAIVE_VS_CHUNK_SHAPES],
)
def test_naive_vs_chunk_fp32(cfg):
    """fp32: naive and chunk should agree within fp32 tolerance."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, g, h0 = _make_jax_inputs(
        B, T, H, K, V, jnp.float32, seed=cfg["seed"], h0=cfg.get("h0", False),
    )
    o_naive, s_naive = naive_recurrent_gla(
        q, k, v, g, initial_state=h0, output_final_state=True,
    )
    o_chunk, s_chunk = chunk_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True,
    )
    from tests.utils import compare_tensor

    assert compare_tensor("output", o_naive, o_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64)


# ============================================================================
# 3. Cross-validation: fused_chunk_gla vs chunk_gla (no GPU needed)
# ============================================================================


def test_fused_chunk_is_chunk_fp64():
    """fused_chunk_gla should produce identical results to chunk_gla."""
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, h0 = _make_jax_inputs(B, T, H, K, V, jnp.float64, seed=42, h0=True)

    o_fused, s_fused = fused_chunk_gla(
        q, k, v, g, initial_state=h0, output_final_state=True,
    )
    o_chunk, s_chunk = chunk_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True,
    )
    np.testing.assert_array_equal(np.array(o_fused), np.array(o_chunk))
    np.testing.assert_array_equal(np.array(s_fused), np.array(s_chunk))


def test_fused_chunk_is_chunk_bf16():
    """fused_chunk_gla should produce identical results to chunk_gla in bf16."""
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.bfloat16, seed=42)

    o_fused, s_fused = fused_chunk_gla(q, k, v, g, output_final_state=True)
    o_chunk, s_chunk = chunk_gla(q, k, v, g=g, output_final_state=True)
    np.testing.assert_array_equal(np.array(o_fused), np.array(o_chunk))
    np.testing.assert_array_equal(np.array(s_fused), np.array(s_chunk))


# ============================================================================
# 4. CPU ref vs FLA Triton (GPU, when available)
# ============================================================================

HAS_CUDA = False
triton_imports_available = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass

try:
    from fla.ops.gla.naive import naive_recurrent_gla as triton_naive_gla

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

_JAX_DTYPES = {
    "float64": jnp.float64,
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

_DTYPE_TOLS = {
    "float64": dict(atol=5e-5, rtol=5e-5),
    "float32": dict(atol=5e-5, rtol=5e-5),
    "float16": dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

_TRITON_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]

TRITON_CASES = [
    {**s, "dtype": d, **t}
    for s in _TRITON_SHAPES
    for d, t in _DTYPE_TOLS.items()
]


def _triton_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


@requires_triton
@pytest.mark.parametrize("cfg", TRITON_CASES, ids=[_triton_case_id(c) for c in TRITON_CASES])
def test_cpu_naive_vs_triton_naive(cfg):
    """Compare CPU naive vs FLA Triton naive."""
    import torch
    import torch.nn.functional as F
    from tests.utils import compare_tensor

    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]

    _TORCH_DTYPES = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    # Triton doesn't support fp64; use fp32 as Triton reference
    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q_t = torch.randn(B, T, H, K).to(triton_dtype)
    k_t = torch.randn(B, T, H, K).to(triton_dtype)
    v_t = torch.randn(B, T, H, V).to(triton_dtype)
    gk_t = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

    # Run Triton naive
    kwargs = dict(output_final_state=True)
    if h0_t is not None:
        kwargs["initial_state"] = h0_t.float().cuda()
    o_tri, s_tri = triton_naive_gla(
        q_t.cuda(), k_t.cuda(), v_t.cuda(), gk_t.cuda(), **kwargs,
    )
    o_tri = o_tri.cpu()
    s_tri = s_tri.cpu() if s_tri is not None else None

    # Run CPU naive
    def _to_jax(t, dt):
        return jnp.array(t.detach().cpu().float().numpy(), dtype=dt)

    q_j = _to_jax(q_t, jax_dtype)
    k_j = _to_jax(k_t, jax_dtype)
    v_j = _to_jax(v_t, jax_dtype)
    gk_j = _to_jax(gk_t, jax_dtype)
    h0_j = _to_jax(h0_t, jnp.float32) if h0_t is not None else None

    o_cpu, s_cpu = naive_recurrent_gla(
        q_j, k_j, v_j, gk_j, initial_state=h0_j, output_final_state=True,
    )

    assert compare_tensor("output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    if s_tri is not None:
        assert compare_tensor("final_state", s_tri, s_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert o_cpu.dtype == jax_dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
