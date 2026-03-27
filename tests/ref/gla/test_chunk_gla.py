"""chunk_gla: JAX CPU ref (tops.cpu.ops.gla) vs FLA Triton GPU, multi-precision."""

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
import torch
import torch.nn.functional as F
import jax.numpy as jnp

from tops.cpu.ops.gla import chunk_gla, chunk_gla_fwd, chunk_gla_bwd
from tests.utils import compare_tensor

try:
    from fla.ops.gla import chunk_gla as triton_chunk_gla
    from fla.ops.gla.chunk import chunk_gla_bwd as triton_chunk_gla_bwd
    from fla.ops.gla.chunk import chunk_gla_fwd as triton_chunk_gla_fwd
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

requires_cuda_and_fla = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)

_JAX_DTYPES = {
    "float64": jnp.float64,
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# ============================================================================
# Forward test configs
#
# Each dict: B, T, H, K, V, seed (required); dtype, h0, atol, rtol (optional)
# Triton 不支持 fp64，fp64 用例以 Triton fp32 输出作为参考
# ============================================================================

_FWD_SHAPES = [
    # ── standard ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── K != V ──
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # ── large batch / heads ──
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    # ── with h0 ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    # ── long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
]

_DTYPE_TOLS = {
    "float64":  dict(atol=5e-5, rtol=5e-5),
    "float32":  dict(atol=5e-5, rtol=5e-5),
    "float16":  dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

FWD_CASES = [
    {**s, "dtype": d, **t}
    for s in _FWD_SHAPES
    for d, t in _DTYPE_TOLS.items()
]

# ============================================================================
# Backward test configs
# ============================================================================

_BWD_SHAPES = [
    # ── standard ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── K != V ──
    dict(B=2, T=64, H=4, K=16, V=128, seed=20),
    dict(B=2, T=64, H=4, K=128, V=16, seed=21),
    # ── large batch / heads ──
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    # ── with h0 / dht ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=2, T=64, H=4, K=32, V=64, seed=14, dht=True),
    dict(B=2, T=64, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    # ── long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
]

BWD_CASES = [
    {**s, "dtype": d, **t}
    for s in _BWD_SHAPES
    for d, t in _DTYPE_TOLS.items()
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t, dtype):
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


def _run_triton_fwd(q, k, v, g, *, h0=None, scale=None):
    kwargs = dict(output_final_state=True)
    if h0 is not None:
        kwargs["initial_state"] = h0.float().to(DEVICE)
    if scale is not None:
        kwargs["scale"] = scale
    o, s = triton_chunk_gla(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE),
        g=g.to(DEVICE), **kwargs,
    )
    return o.cpu(), s.cpu() if s is not None else None


def _run_cpu_fwd(q, k, v, g, jax_dtype, *, h0=None, scale=None):
    q_j = _torch_to_jax(q, jax_dtype)
    k_j = _torch_to_jax(k, jax_dtype)
    v_j = _torch_to_jax(v, jax_dtype)
    g_j = _torch_to_jax(g, jax_dtype)
    h0_j = _torch_to_jax(h0, jnp.float32) if h0 is not None else None
    return chunk_gla(
        q_j, k_j, v_j, g=g_j,
        initial_state=h0_j, output_final_state=True, scale=scale,
    )


def _run_triton_bwd(q, k, v, g, do, *, h0=None, dht=None, scale=None,
                    chunk_size=64):
    B, T, H, K = q.shape
    C = chunk_size
    T_padded = ((T + C - 1) // C) * C

    q_t, k_t, v_t, g_t, do_t = q, k, v, g, do
    if T_padded > T:
        pad = T_padded - T
        q_t = F.pad(q, (0, 0, 0, 0, 0, pad))
        k_t = F.pad(k, (0, 0, 0, 0, 0, pad))
        v_t = F.pad(v, (0, 0, 0, 0, 0, pad))
        g_t = F.pad(g, (0, 0, 0, 0, 0, pad))
        do_t = F.pad(do, (0, 0, 0, 0, 0, pad))

    def _dev(t):
        return t.float().contiguous().to(DEVICE) if t is not None else None

    q_g, k_g, v_g, g_g, do_g = _dev(q_t), _dev(k_t), _dev(v_t), _dev(g_t), _dev(do_t)
    h0_g, dht_g = _dev(h0), _dev(dht)

    g_cumsum_g, A_g, h_g, _, _ = triton_chunk_gla_fwd(
        q_g, k_g, v_g, g_g, None, scale, h0_g,
        output_final_state=False, chunk_size=C,
    )
    dq, dk, dv, dg, dh0_out = triton_chunk_gla_bwd(
        q_g, k_g, v_g, g_g, g_cumsum_g, scale, h0_g, h_g, A_g, do_g, dht_g,
        chunk_size=C,
    )
    return (
        dq[:, :T].cpu(), dk[:, :T].cpu(), dv[:, :T].cpu(), dg[:, :T].cpu(),
        dh0_out.cpu() if dh0_out is not None else None,
    )


def _run_cpu_bwd(q, k, v, g, do, jax_dtype, *, h0=None, dht=None,
                 scale=None, chunk_size=16):
    q_j = _torch_to_jax(q, jax_dtype)
    k_j = _torch_to_jax(k, jax_dtype)
    v_j = _torch_to_jax(v, jax_dtype)
    g_j = _torch_to_jax(g, jax_dtype)
    do_j = _torch_to_jax(do, jax_dtype)
    h0_j = _torch_to_jax(h0, jax_dtype) if h0 is not None else None
    dht_j = _torch_to_jax(dht, jax_dtype) if dht is not None else None
    return chunk_gla_bwd(
        q_j, k_j, v_j, g_j,
        g_cumsum=None, scale=scale, initial_state=h0_j,
        h=None, A=None, do=do_j, dht=dht_j,
        chunk_size=chunk_size,
    )


def _naive_gla(q, k, v, g, scale, *, h0=None):
    """Step-by-step GLA recurrence: h_t = h_{t-1} * exp(g_t) + k_t^T v_t."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    h = jnp.zeros((B, H, K, V), dtype=q.dtype)
    if h0 is not None:
        h = h + h0

    outs = []
    for t in range(T):
        h = h * jnp.exp(g[:, t, :, :, None]) + k[:, t, :, :, None] * v[:, t, :, None, :]
        outs.append(jnp.sum(q[:, t, :, :, None] * h, axis=2) * scale)
    return jnp.stack(outs, axis=1), h


# ============================================================================
# Intermediate dtype verification (no GPU needed)
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


def test_intermediate_dtypes_bf16():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.bfloat16)

    g_cumsum, A, h, ht, o = chunk_gla_fwd(q, k, v, g, None, K**-0.5, None, True, 16)
    assert g_cumsum.dtype == jnp.float32
    assert A.dtype == jnp.float32
    assert h.dtype == jnp.bfloat16
    assert ht.dtype == jnp.float32
    assert o.dtype == jnp.bfloat16

    do = jax.random.normal(jax.random.PRNGKey(99), (B, T, H, V), dtype=jnp.bfloat16)
    dq, dk, dv, dg, dh0 = chunk_gla_bwd(q, k, v, g, None, K**-0.5, None, None, None, do, None, 16)
    assert dq.dtype == jnp.float32
    assert dk.dtype == jnp.float32
    assert dv.dtype == jnp.bfloat16
    assert dg.dtype == jnp.float32
    assert dh0 is None


def test_intermediate_dtypes_fp16():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float16)

    g_cumsum, A, h, ht, o = chunk_gla_fwd(q, k, v, g, None, K**-0.5, None, True, 16)
    assert g_cumsum.dtype == jnp.float32
    assert A.dtype == jnp.float32
    assert h.dtype == jnp.float16
    assert ht.dtype == jnp.float32
    assert o.dtype == jnp.float16


def test_intermediate_dtypes_fp64():
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float64)

    g_cumsum, A, h, ht, o = chunk_gla_fwd(q, k, v, g, None, K**-0.5, None, True, 16)
    assert g_cumsum.dtype == jnp.float64
    assert A.dtype == jnp.float64
    assert h.dtype == jnp.float64
    assert ht.dtype == jnp.float64
    assert o.dtype == jnp.float64

    do = jax.random.normal(jax.random.PRNGKey(99), (B, T, H, V), dtype=jnp.float64)
    dq, dk, dv, dg, _ = chunk_gla_bwd(q, k, v, g, None, K**-0.5, None, None, None, do, None, 16)
    assert dq.dtype == jnp.float64
    assert dk.dtype == jnp.float64
    assert dv.dtype == jnp.float64
    assert dg.dtype == jnp.float64


# ============================================================================
# Forward: CPU ref vs Triton (requires GPU)
# ============================================================================


@requires_cuda_and_fla
@pytest.mark.parametrize("cfg", FWD_CASES, ids=[_case_id(c) for c in FWD_CASES])
def test_cpu_vs_triton_chunk_fwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]

    # Triton 不支持 fp64，用 fp32 跑 Triton 作参考
    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K).to(triton_dtype)
    k = torch.randn(B, T, H, K).to(triton_dtype)
    v = torch.randn(B, T, H, V).to(triton_dtype)
    g = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None

    o_tri, s_tri = _run_triton_fwd(q, k, v, g, h0=h0)
    o_cpu, s_cpu = _run_cpu_fwd(q, k, v, g, jax_dtype, h0=h0)

    assert compare_tensor("output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("final_state", s_tri, s_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert o_cpu.dtype == jax_dtype


# ============================================================================
# Backward: CPU ref vs Triton (requires GPU)
# ============================================================================


@requires_cuda_and_fla
@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_case_id(c) for c in BWD_CASES])
def test_cpu_vs_triton_chunk_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]
    scale = K ** -0.5

    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K).to(triton_dtype)
    k = torch.randn(B, T, H, K).to(triton_dtype)
    v = torch.randn(B, T, H, V).to(triton_dtype)
    g = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    do = torch.randn(B, T, H, V).to(triton_dtype)
    h0 = torch.randn(B, H, K, V).to(triton_dtype) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V).to(triton_dtype) if cfg.get("dht") else None

    dq_tri, dk_tri, dv_tri, dg_tri, dh0_tri = _run_triton_bwd(
        q, k, v, g, do, h0=h0, dht=dht, scale=scale,
    )
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = _run_cpu_bwd(
        q, k, v, g, do, jax_dtype, h0=h0, dht=dht, scale=scale,
    )

    assert compare_tensor("dq", dq_tri, dq_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("dk", dk_tri, dk_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("dv", dv_tri, dv_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    assert compare_tensor("dg", dg_tri, dg_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)
    if cfg.get("h0") and dh0_tri is not None and dh0_cpu is not None:
        assert compare_tensor("dh0", dh0_tri, dh0_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64)


# ============================================================================
# Naive cross-validation (no GPU needed)
# ============================================================================


def test_chunk_vs_naive_fp64():
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float64, seed=42)

    o_chunk, s_chunk = chunk_gla(q, k, v, g=g, scale=K**-0.5, output_final_state=True)
    o_naive, s_naive = _naive_gla(q, k, v, g, K**-0.5)

    assert compare_tensor("output", o_naive, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


def test_chunk_vs_naive_fp64_with_h0():
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, h0 = _make_jax_inputs(B, T, H, K, V, jnp.float64, seed=42, h0=True)

    o_chunk, s_chunk = chunk_gla(q, k, v, g=g, scale=K**-0.5, initial_state=h0, output_final_state=True)
    o_naive, s_naive = _naive_gla(q, k, v, g, K**-0.5, h0=h0)

    assert compare_tensor("output", o_naive, o_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_chunk, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


def test_chunk_vs_naive_fp32():
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, _ = _make_jax_inputs(B, T, H, K, V, jnp.float32, seed=42)

    o_chunk, s_chunk = chunk_gla(q, k, v, g=g, scale=K**-0.5, output_final_state=True)
    o_naive, s_naive = _naive_gla(q, k, v, g, K**-0.5)

    assert compare_tensor("output", o_naive, o_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64)
    assert compare_tensor("state", s_naive, s_chunk, atol=5e-3, rtol=5e-3, compare_dtype=np.float64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
