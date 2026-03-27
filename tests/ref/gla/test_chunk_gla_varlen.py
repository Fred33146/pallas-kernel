"""Varlen tests for chunk_gla: varlen output should match independent-batch.

Tests:
  1. CPU-only: varlen vs independent-batch (no GPU needed)
  2. GPU: CPU ref varlen vs FLA Triton varlen (requires CUDA + FLA)
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
import jax.numpy as jnp
import pytest
from tops.cpu.ops.gla import chunk_gla
from tests.utils import compare_tensor

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass

try:
    from fla.ops.gla import chunk_gla as triton_chunk_gla

    HAS_FLA = True
except ImportError:
    HAS_FLA = False

requires_cuda_and_fla = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)

# ============================================================================
# Configs
# ============================================================================

_VARLEN_SHAPES = [
    # ── equal segments, chunk-aligned ──
    dict(T=32, H=2, K=32, V=64, cu_seqlens=[0, 16, 32], C=16, seed=42),
    # ── unequal segments, chunk-aligned ──
    dict(T=48, H=4, K=32, V=64, cu_seqlens=[0, 16, 48], C=16, seed=7),
    # ── non-chunk-aligned (triggers per-segment padding) ──
    dict(T=30, H=2, K=32, V=64, cu_seqlens=[0, 10, 30], C=16, seed=11),
    # ── single token segment ──
    dict(T=17, H=2, K=32, V=64, cu_seqlens=[0, 1, 17], C=16, seed=400),
    # ── many short segments ──
    dict(T=40, H=2, K=32, V=64, cu_seqlens=[0, 5, 15, 25, 40], C=16, seed=99),
]

_DTYPE_TOLS = {
    "float64":  dict(atol=5e-5, rtol=5e-5),
    "float32":  dict(atol=5e-5, rtol=5e-5),
    "float16":  dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

_JAX_DTYPES = {
    "float64": jnp.float64,
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

GPU_VARLEN_CASES = [
    {**s, "dtype": d, **t}
    for s in _VARLEN_SHAPES
    for d, t in _DTYPE_TOLS.items()
    if d != "float64"  # Triton does not support fp64
]


def _varlen_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}"


def _gpu_case_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}_{c['dtype']}"


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t, dtype):
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


def _run_independent_batch(
    q, k, v, g, cu_seqlens, chunk_size, initial_state=None, output_final_state=False
):
    """Run each segment as independent [1, T_i, ...] call, concatenate."""
    N = len(cu_seqlens) - 1
    outputs = []
    final_states = []
    for n in range(N):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        h0_n = initial_state[n : n + 1] if initial_state is not None else None
        o_n, ht_n = chunk_gla(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g=g[:, bos:eos] if g is not None else None,
            initial_state=h0_n,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        outputs.append(o_n)
        if ht_n is not None:
            final_states.append(ht_n[0])
    o_cat = jnp.concatenate(outputs, axis=1)
    ht_cat = jnp.stack(final_states, axis=0) if final_states else None
    return o_cat, ht_cat


# ============================================================================
# 1. CPU-only: varlen vs independent-batch (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("cfg", _VARLEN_SHAPES, ids=[_varlen_id(c) for c in _VARLEN_SHAPES])
def test_gla_varlen_fwd(cfg):
    """Varlen forward should match running each segment independently."""
    T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
    cu = jnp.array(cfg["cu_seqlens"])
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(keys[0], (1, T, H, K))
    k = jax.random.normal(keys[1], (1, T, H, K))
    v = jax.random.normal(keys[2], (1, T, H, V))
    g = -jax.nn.softplus(jax.random.normal(keys[3], (1, T, H, K))).astype(q.dtype)

    o_var, _ = chunk_gla(q, k, v, g=g, cu_seqlens=cu, chunk_size=C)
    o_ind, _ = _run_independent_batch(q, k, v, g, cu, C)

    assert jnp.allclose(o_var, o_ind, atol=1e-5)


def test_gla_varlen_fwd_with_h0():
    """Varlen forward with per-sequence h0."""
    H, K, V, C = 2, 32, 64, 16
    cu = jnp.array([0, 16, 32])
    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q = jax.random.normal(keys[0], (1, 32, H, K))
    k = jax.random.normal(keys[1], (1, 32, H, K))
    v = jax.random.normal(keys[2], (1, 32, H, V))
    g = -jax.nn.softplus(jax.random.normal(keys[3], (1, 32, H, K))).astype(q.dtype)
    h0 = jax.random.normal(keys[4], (2, H, K, V))

    o_var, ht_var = chunk_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu, chunk_size=C
    )
    o_ind, ht_ind = _run_independent_batch(
        q, k, v, g, cu, C, initial_state=h0, output_final_state=True
    )

    assert jnp.allclose(o_var, o_ind, atol=1e-5)
    assert ht_var.shape == (2, H, K, V)
    assert jnp.allclose(ht_var, ht_ind, atol=1e-5)


def test_gla_varlen_fp64():
    """Varlen forward in fp64 (high-precision reference)."""
    H, K, V, C = 2, 32, 64, 16
    cu = jnp.array([0, 10, 30])
    keys = jax.random.split(jax.random.PRNGKey(123), 4)
    q = jax.random.normal(keys[0], (1, 30, H, K), dtype=jnp.float64)
    k = jax.random.normal(keys[1], (1, 30, H, K), dtype=jnp.float64)
    v = jax.random.normal(keys[2], (1, 30, H, V), dtype=jnp.float64)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (1, 30, H, K), dtype=jnp.float64))

    o_var, _ = chunk_gla(q, k, v, g=g, cu_seqlens=cu, chunk_size=C)
    o_ind, _ = _run_independent_batch(q, k, v, g, cu, C)

    assert jnp.allclose(o_var, o_ind, atol=1e-10)


# ============================================================================
# 2. GPU: CPU ref varlen vs FLA Triton varlen
# ============================================================================


@requires_cuda_and_fla
@pytest.mark.parametrize("cfg", GPU_VARLEN_CASES, ids=[_gpu_case_id(c) for c in GPU_VARLEN_CASES])
def test_gla_varlen_cpu_vs_triton(cfg):
    """Compare CPU ref chunk_gla(cu_seqlens=...) against FLA Triton chunk_gla(cu_seqlens=...)."""
    T, H, K, V, C = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"]
    seed = cfg["seed"]
    jax_dtype = _JAX_DTYPES[cfg["dtype"]]
    torch_dtype = getattr(torch, cfg["dtype"])
    atol, rtol = cfg["atol"], cfg["rtol"]
    cu_list = cfg["cu_seqlens"]

    torch.manual_seed(seed)
    q_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    k_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    v_t = torch.randn(1, T, H, V, dtype=torch_dtype)
    g_t = F.logsigmoid(torch.randn(1, T, H, K)).to(torch_dtype)

    cu_t = torch.tensor(cu_list, dtype=torch.long)

    # FLA Triton with cu_seqlens
    o_tri, ht_tri = triton_chunk_gla(
        q_t.to("cuda"), k_t.to("cuda"), v_t.to("cuda"),
        g=g_t.to("cuda"),
        cu_seqlens=cu_t.to("cuda"),
        output_final_state=True,
    )
    o_tri = o_tri.cpu()
    ht_tri = ht_tri.cpu() if ht_tri is not None else None

    # CPU ref with cu_seqlens
    q_j = _torch_to_jax(q_t, jax_dtype)
    k_j = _torch_to_jax(k_t, jax_dtype)
    v_j = _torch_to_jax(v_t, jax_dtype)
    g_j = _torch_to_jax(g_t, jax_dtype)
    cu_j = jnp.array(cu_list)

    o_cpu, ht_cpu = chunk_gla(
        q_j, k_j, v_j, g=g_j,
        cu_seqlens=cu_j,
        output_final_state=True,
        chunk_size=C,
    )

    assert compare_tensor("output", o_tri.float().numpy(), np.array(o_cpu, dtype=np.float32),
                   atol=atol, rtol=rtol)

    if ht_tri is not None and ht_cpu is not None:
        assert compare_tensor("final_state", ht_tri.float().numpy(), np.array(ht_cpu, dtype=np.float32),
                       atol=atol, rtol=rtol)


@requires_cuda_and_fla
def test_gla_varlen_cpu_vs_triton_with_h0():
    """Varlen with per-sequence h0: CPU ref vs FLA Triton."""
    H, K, V, C = 2, 32, 64, 16
    cu_list = [0, 16, 48]
    T = 48
    N = 2

    torch.manual_seed(42)
    q_t = torch.randn(1, T, H, K, dtype=torch.float32)
    k_t = torch.randn(1, T, H, K, dtype=torch.float32)
    v_t = torch.randn(1, T, H, V, dtype=torch.float32)
    g_t = F.logsigmoid(torch.randn(1, T, H, K)).to(torch.float32)
    h0_t = torch.randn(N, H, K, V, dtype=torch.float32)
    cu_t = torch.tensor(cu_list, dtype=torch.long)

    # FLA Triton
    o_tri, ht_tri = triton_chunk_gla(
        q_t.to("cuda"), k_t.to("cuda"), v_t.to("cuda"),
        g=g_t.to("cuda"),
        initial_state=h0_t.to("cuda"),
        cu_seqlens=cu_t.to("cuda"),
        output_final_state=True,
    )
    o_tri = o_tri.cpu()
    ht_tri = ht_tri.cpu()

    # CPU ref
    q_j = _torch_to_jax(q_t, jnp.float32)
    k_j = _torch_to_jax(k_t, jnp.float32)
    v_j = _torch_to_jax(v_t, jnp.float32)
    g_j = _torch_to_jax(g_t, jnp.float32)
    h0_j = _torch_to_jax(h0_t, jnp.float32)
    cu_j = jnp.array(cu_list)

    o_cpu, ht_cpu = chunk_gla(
        q_j, k_j, v_j, g=g_j,
        initial_state=h0_j,
        cu_seqlens=cu_j,
        output_final_state=True,
        chunk_size=C,
    )

    assert compare_tensor("output", o_tri.float().numpy(), np.array(o_cpu, dtype=np.float32),
                   atol=5e-5, rtol=5e-5)
    assert compare_tensor("final_state", ht_tri.float().numpy(), np.array(ht_cpu, dtype=np.float32),
                   atol=5e-5, rtol=5e-5)
