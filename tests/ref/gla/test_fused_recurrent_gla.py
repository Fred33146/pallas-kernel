"""fused_recurrent_gla: JAX CPU ref (tops.cpu.ops.gla) tests.

Tests:
  1. Intermediate dtype verification (no GPU)
  2. Cross-validation: fused_recurrent vs naive (no GPU)
  3. Cross-validation: fused_recurrent vs chunk_gla (no GPU)
  4. Backward dtype verification (no GPU)
  5. CPU ref vs FLA Triton fwd (GPU)
  6. CPU ref vs FLA Triton bwd (GPU)
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

from tops.cpu.ops.gla import (
    fused_recurrent_fwd,
    fused_recurrent_bwd,
    fused_recurrent_gla,
    naive_recurrent_gla,
    chunk_gla,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_jax_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    gk = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H, K))).astype(dtype)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    h0_arr = jax.random.normal(keys[4], (B, H, K, V), dtype=acc) if h0 else None
    return q, k, v, gk, h0_arr


# ============================================================================
# 1. Forward dtype verification (no GPU needed)
# ============================================================================


def test_fused_recurrent_fwd_dtypes_bf16():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.bfloat16)
    # fused_recurrent_fwd returns fp32 (matching FLA's Python launcher)
    o, ht = fused_recurrent_fwd(q, k, v, gk=gk, output_final_state=True)
    assert o.dtype == jnp.float32, f"o.dtype={o.dtype}, expected float32"
    assert ht.dtype == jnp.float32, f"ht.dtype={ht.dtype}, expected float32"
    # fused_recurrent_gla casts to q.dtype (matching FusedRecurrentFunction.forward)
    o2, _ = fused_recurrent_gla(q, k, v, gk=gk, output_final_state=True)
    assert o2.dtype == jnp.bfloat16, f"o2.dtype={o2.dtype}, expected bfloat16"


def test_fused_recurrent_fwd_dtypes_fp16():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.float16)
    # fused_recurrent_fwd returns fp32 (matching FLA's Python launcher)
    o, ht = fused_recurrent_fwd(q, k, v, gk=gk, output_final_state=True)
    assert o.dtype == jnp.float32, f"o.dtype={o.dtype}, expected float32"
    assert ht.dtype == jnp.float32, f"ht.dtype={ht.dtype}, expected float32"
    # fused_recurrent_gla casts to q.dtype (matching FusedRecurrentFunction.forward)
    o2, _ = fused_recurrent_gla(q, k, v, gk=gk, output_final_state=True)
    assert o2.dtype == jnp.float16, f"o2.dtype={o2.dtype}, expected float16"


def test_fused_recurrent_fwd_dtypes_fp32():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.float32)
    o, ht = fused_recurrent_fwd(q, k, v, gk=gk, output_final_state=True)
    assert o.dtype == jnp.float32
    assert ht.dtype == jnp.float32


def test_fused_recurrent_fwd_dtypes_fp64():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.float64)
    o, ht = fused_recurrent_fwd(q, k, v, gk=gk, output_final_state=True)
    assert o.dtype == jnp.float64
    assert ht.dtype == jnp.float64


def test_fused_recurrent_fwd_no_final_state():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.float32)
    o, ht = fused_recurrent_fwd(q, k, v, gk=gk, output_final_state=False)
    assert ht is None


# ============================================================================
# 2. Backward dtype verification (no GPU needed)
# ============================================================================


def test_fused_recurrent_bwd_dtypes_bf16():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, h0 = _make_jax_inputs(B, T, H, K, V, jnp.bfloat16, h0=True)
    o, _ = fused_recurrent_fwd(q, k, v, gk=gk, initial_state=h0, output_final_state=True)
    do = jax.random.normal(jax.random.PRNGKey(99), (B, T, H, V), dtype=jnp.bfloat16)
    dht = jax.random.normal(jax.random.PRNGKey(100), (B, H, K, V), dtype=jnp.float32)

    dq, dk, dv, dgk, dgv, dh0 = fused_recurrent_bwd(
        q, k, v, gk=gk, do=do, dht=dht, initial_state=h0,
    )
    # FLA: dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dgk.to(gk)
    assert dq.dtype == jnp.bfloat16, f"dq.dtype={dq.dtype}"
    assert dk.dtype == jnp.bfloat16, f"dk.dtype={dk.dtype}"
    assert dv.dtype == jnp.bfloat16, f"dv.dtype={dv.dtype}"
    assert dgk.dtype == jnp.bfloat16, f"dgk.dtype={dgk.dtype}"
    assert dgv is None  # no gv used
    assert dh0.dtype == jnp.float32, f"dh0.dtype={dh0.dtype}"


def test_fused_recurrent_bwd_dtypes_fp64():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, h0 = _make_jax_inputs(B, T, H, K, V, jnp.float64, h0=True)
    o, _ = fused_recurrent_fwd(q, k, v, gk=gk, initial_state=h0, output_final_state=True)
    do = jax.random.normal(jax.random.PRNGKey(99), (B, T, H, V), dtype=jnp.float64)
    dht = jax.random.normal(jax.random.PRNGKey(100), (B, H, K, V), dtype=jnp.float64)

    dq, dk, dv, dgk, dgv, dh0 = fused_recurrent_bwd(
        q, k, v, gk=gk, do=do, dht=dht, initial_state=h0,
    )
    assert dq.dtype == jnp.float64
    assert dk.dtype == jnp.float64
    assert dv.dtype == jnp.float64
    assert dgk.dtype == jnp.float64
    assert dh0.dtype == jnp.float64


# ============================================================================
# 3. Cross-validation: fused_recurrent vs naive (no GPU needed)
# ============================================================================


_CROSS_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]


def _cross_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


@pytest.mark.parametrize("cfg", _CROSS_SHAPES, ids=[_cross_id(c) for c in _CROSS_SHAPES])
def test_fused_recurrent_vs_naive_fp64(cfg):
    """fp64: fused_recurrent and naive should agree to near machine precision."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, gk, h0 = _make_jax_inputs(
        B, T, H, K, V, jnp.float64, seed=cfg["seed"], h0=cfg.get("h0", False),
    )
    scale = K**-0.5

    o_fr, s_fr = fused_recurrent_gla(
        q, k, v, gk=gk, scale=scale, initial_state=h0, output_final_state=True,
    )
    o_naive, s_naive = naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True,
    )
    from tests.utils import compare_tensor

    assert compare_tensor("output", o_naive, o_fr, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_fr, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


@pytest.mark.parametrize("cfg", _CROSS_SHAPES, ids=[_cross_id(c) for c in _CROSS_SHAPES])
def test_fused_recurrent_vs_naive_fp32(cfg):
    """fp32: fused_recurrent and naive should agree within fp32 tolerance."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, gk, h0 = _make_jax_inputs(
        B, T, H, K, V, jnp.float32, seed=cfg["seed"], h0=cfg.get("h0", False),
    )
    scale = K**-0.5

    o_fr, s_fr = fused_recurrent_gla(
        q, k, v, gk=gk, scale=scale, initial_state=h0, output_final_state=True,
    )
    o_naive, s_naive = naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True,
    )
    from tests.utils import compare_tensor

    assert compare_tensor("output", o_naive, o_fr, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_fr, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)


# ============================================================================
# 4. Cross-validation: fused_recurrent vs chunk_gla (no GPU needed)
# ============================================================================


def test_fused_recurrent_vs_chunk_fp64():
    """fp64: fused_recurrent and chunk should agree."""
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, gk, h0 = _make_jax_inputs(B, T, H, K, V, jnp.float64, seed=42, h0=True)
    scale = K**-0.5

    o_fr, s_fr = fused_recurrent_gla(
        q, k, v, gk=gk, scale=scale, initial_state=h0, output_final_state=True,
    )
    o_chunk, s_chunk = chunk_gla(
        q, k, v, g=gk, scale=scale, initial_state=h0, output_final_state=True,
    )
    from tests.utils import compare_tensor

    assert compare_tensor("output", o_fr, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("state", s_fr, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


# ============================================================================
# 5. CPU ref vs FLA Triton (GPU, when available)
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
    from fla.ops.gla import fused_recurrent_gla as triton_fused_recurrent_gla
    from fla.ops.common.fused_recurrent import fused_recurrent_bwd as triton_fused_recurrent_bwd

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

_FWD_DTYPE_TOLS = {
    "float64": dict(atol=5e-5, rtol=5e-5),
    "float32": dict(atol=5e-5, rtol=5e-5),
    "float16": dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

_BWD_DTYPE_TOLS = {
    "float64": dict(atol=5e-5, rtol=5e-5),
    "float32": dict(atol=5e-5, rtol=5e-5),
}

_TRITON_FWD_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]

TRITON_FWD_CASES = [
    {**s, "dtype": d, **t}
    for s in _TRITON_FWD_SHAPES
    for d, t in _FWD_DTYPE_TOLS.items()
]

_TRITON_BWD_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=14, dht=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]

TRITON_BWD_CASES = [
    {**s, "dtype": d, **t}
    for s in _TRITON_BWD_SHAPES
    for d, t in _BWD_DTYPE_TOLS.items()
]


def _triton_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


def _to_jax(t, dt):
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dt)


@requires_triton
@pytest.mark.parametrize("cfg", TRITON_FWD_CASES, ids=[_triton_case_id(c) for c in TRITON_FWD_CASES])
def test_cpu_vs_triton_fwd(cfg):
    """Compare CPU fused_recurrent_fwd vs FLA Triton fused_recurrent_gla."""
    import torch
    import torch.nn.functional as F
    from tests.utils import compare_tensor

    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]

    _TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q_t = torch.randn(B, T, H, K).to(triton_dtype)
    k_t = torch.randn(B, T, H, K).to(triton_dtype)
    v_t = torch.randn(B, T, H, V).to(triton_dtype)
    gk_t = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

    # Run Triton
    kwargs = dict(output_final_state=True)
    if h0_t is not None:
        kwargs["initial_state"] = h0_t.float().cuda()
    o_tri, s_tri = triton_fused_recurrent_gla(
        q_t.cuda(), k_t.cuda(), v_t.cuda(), gk=gk_t.cuda(), **kwargs,
    )
    o_tri = o_tri.cpu()
    s_tri = s_tri.cpu() if s_tri is not None else None

    # Run CPU ref
    q_j = _to_jax(q_t, jax_dtype)
    k_j = _to_jax(k_t, jax_dtype)
    v_j = _to_jax(v_t, jax_dtype)
    gk_j = _to_jax(gk_t, jax_dtype)
    h0_j = _to_jax(h0_t, jnp.float32) if h0_t is not None else None

    o_cpu, s_cpu = fused_recurrent_gla(
        q_j, k_j, v_j, gk=gk_j, initial_state=h0_j, output_final_state=True,
    )

    assert compare_tensor("output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    if s_tri is not None:
        assert compare_tensor("final_state", s_tri, s_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert o_cpu.dtype == jax_dtype


@requires_triton
@pytest.mark.parametrize("cfg", TRITON_BWD_CASES, ids=[_triton_case_id(c) for c in TRITON_BWD_CASES])
def test_cpu_vs_triton_bwd(cfg):
    """Compare CPU fused_recurrent_bwd vs FLA Triton fused_recurrent_bwd."""
    import torch
    import torch.nn.functional as F
    from tests.utils import compare_tensor

    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]

    _TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q_t = torch.randn(B, T, H, K).to(triton_dtype)
    k_t = torch.randn(B, T, H, K).to(triton_dtype)
    v_t = torch.randn(B, T, H, V).to(triton_dtype)
    gk_t = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    do_t = torch.randn(B, T, H, V).to(triton_dtype)
    h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht_t = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # Run Triton fwd first (needed for o in bwd)
    kwargs_fwd = dict(output_final_state=True)
    if h0_t is not None:
        kwargs_fwd["initial_state"] = h0_t.float().cuda()
    o_tri, _ = triton_fused_recurrent_gla(
        q_t.cuda(), k_t.cuda(), v_t.cuda(), gk=gk_t.cuda(), **kwargs_fwd,
    )

    # Run Triton bwd
    dq_tri, dk_tri, dv_tri, _, dgk_tri, _, dh0_tri = triton_fused_recurrent_bwd(
        q=q_t.cuda(), k=k_t.cuda(), v=v_t.cuda(),
        gk=gk_t.cuda(), o=o_tri,
        do=do_t.cuda(), scale=K**-0.5,
        initial_state=h0_t.float().cuda() if h0_t is not None else None,
        dht=dht_t.float().cuda() if dht_t is not None else None,
    )
    dq_tri = dq_tri.cpu()
    dk_tri = dk_tri.cpu()
    dv_tri = dv_tri.cpu()
    dgk_tri = dgk_tri.cpu() if dgk_tri is not None else None
    dh0_tri = dh0_tri.cpu() if dh0_tri is not None else None

    # Run CPU ref
    q_j = _to_jax(q_t, jax_dtype)
    k_j = _to_jax(k_t, jax_dtype)
    v_j = _to_jax(v_t, jax_dtype)
    gk_j = _to_jax(gk_t, jax_dtype)
    do_j = _to_jax(do_t, jax_dtype)
    h0_j = _to_jax(h0_t, jnp.float32) if h0_t is not None else None
    dht_j = _to_jax(dht_t, jnp.float32) if dht_t is not None else None

    o_cpu, _ = fused_recurrent_fwd(
        q_j, k_j, v_j, gk=gk_j, initial_state=h0_j, output_final_state=True,
    )

    dq_cpu, dk_cpu, dv_cpu, dgk_cpu, _, dh0_cpu = fused_recurrent_bwd(
        q_j, k_j, v_j, gk=gk_j, o=o_cpu, do=do_j, dht=dht_j,
        scale=K**-0.5, initial_state=h0_j,
    )

    assert compare_tensor("dq", dq_tri, dq_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("dk", dk_tri, dk_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("dv", dv_tri, dv_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    if dgk_tri is not None:
        assert compare_tensor("dgk", dgk_tri, dgk_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    if cfg.get("h0") and dh0_tri is not None:
        assert compare_tensor("dh0", dh0_tri, dh0_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)


# ============================================================================
# Variable-length test (no GPU)
# ============================================================================


def test_fused_recurrent_cu_seqlens_fp64():
    """Variable-length via cu_seqlens should match per-segment results."""
    B, H, K, V = 1, 2, 32, 64
    T1, T2 = 20, 30
    T_total = T1 + T2

    q, k, v, gk, _ = _make_jax_inputs(B, T_total, H, K, V, jnp.float64, seed=42)
    cu_seqlens = np.array([0, T1, T_total])
    h0 = jax.random.normal(jax.random.PRNGKey(99), (2, H, K, V), dtype=jnp.float64)
    scale = K**-0.5

    o_var, s_var = fused_recurrent_gla(
        q, k, v, gk=gk, scale=scale, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )

    o1, s1 = fused_recurrent_gla(
        q[:, :T1], k[:, :T1], v[:, :T1], gk=gk[:, :T1],
        scale=scale, initial_state=h0[0:1], output_final_state=True,
    )
    o2, s2 = fused_recurrent_gla(
        q[:, T1:], k[:, T1:], v[:, T1:], gk=gk[:, T1:],
        scale=scale, initial_state=h0[1:2], output_final_state=True,
    )
    o_manual = jnp.concatenate([o1, o2], axis=1)
    s_manual = jnp.concatenate([s1, s2], axis=0)

    np.testing.assert_allclose(np.array(o_var), np.array(o_manual), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.array(s_var), np.array(s_manual), atol=1e-12, rtol=1e-12)


# ============================================================================
# Reverse mode test (no GPU)
# ============================================================================


def test_fused_recurrent_reverse_fp64():
    """Reverse mode should produce different results from forward."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, gk, _ = _make_jax_inputs(B, T, H, K, V, jnp.float64)

    o_fwd, _ = fused_recurrent_gla(q, k, v, gk=gk, reverse=False, output_final_state=True)
    o_rev, _ = fused_recurrent_gla(q, k, v, gk=gk, reverse=True, output_final_state=True)

    # Should NOT be the same (different processing order)
    assert not np.allclose(np.array(o_fwd), np.array(o_rev), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
