"""chunk_simple_gla varlen: verify cu_seqlens-based variable-length processing
matches independent per-sequence batch execution.

Tests:
  1. CPU-only: varlen vs independent batch (no GPU needed)
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
from tops.cpu.ops.simple_gla import chunk_simple_gla
from tests.utils import compare_tensor

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
    from fla.ops.simple_gla import chunk_simple_gla as triton_chunk_simple_gla

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
    dict(T=128, H=2, K=32, V=64, cu_seqlens=[0, 64, 128], C=64, seed=42),
    # ── unequal segments, chunk-aligned ──
    dict(T=192, H=4, K=32, V=64, cu_seqlens=[0, 64, 192], C=64, seed=7),
    # ── non-chunk-aligned (triggers per-segment padding) ──
    dict(T=100, H=2, K=32, V=64, cu_seqlens=[0, 30, 100], C=64, seed=11),
    # ── single token segment ──
    dict(T=65, H=2, K=32, V=64, cu_seqlens=[0, 1, 65], C=64, seed=400),
]

_DTYPE_TOLS = {
    "float32":  dict(atol=5e-5, rtol=5e-5),
    "float16":  dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

GATE_MODES = ["g", "g_gamma", "none"]
GPU_GATE_MODES = ["g", "none"]  # g_gamma not in FLA's simple_gla API with cu_seqlens

GPU_VARLEN_CASES = [
    {**s, "dtype": d, "gate": gm, **t}
    for s in _VARLEN_SHAPES
    for d, t in _DTYPE_TOLS.items()
    for gm in GPU_GATE_MODES
]


def _varlen_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}"


def _gpu_case_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}_{c['dtype']}_{c['gate']}"


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t, dtype):
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


def _run_independent_batch(
    q, k, v, cu_seqlens, chunk_size,
    g=None, g_gamma=None, initial_state=None, output_final_state=False,
):
    """Run chunk_simple_gla independently per sequence and concatenate results."""
    N = len(cu_seqlens) - 1
    outputs = []
    final_states = []
    for n in range(N):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        h0_n = initial_state[n : n + 1] if initial_state is not None else None
        g_n = g[:, bos:eos] if g is not None else None
        o_n, ht_n = chunk_simple_gla(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g=g_n,
            g_gamma=g_gamma,
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
# 1. CPU-only: varlen vs independent batch (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("gate", GATE_MODES)
@pytest.mark.parametrize("cfg", _VARLEN_SHAPES, ids=[_varlen_id(c) for c in _VARLEN_SHAPES])
def test_simple_gla_varlen_fwd(cfg, gate):
    """Varlen forward should match independent per-sequence execution."""
    T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
    cu = jnp.array(cfg["cu_seqlens"])
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(keys[0], (1, T, H, K))
    k = jax.random.normal(keys[1], (1, T, H, K))
    v = jax.random.normal(keys[2], (1, T, H, V))

    g = jax.random.normal(keys[3], (1, T, H)) * 0.1 if gate == "g" else None
    g_gamma = jnp.full((H,), -0.05, dtype=jnp.float32) if gate == "g_gamma" else None

    o_var, _ = chunk_simple_gla(
        q, k, v, g=g, g_gamma=g_gamma, cu_seqlens=cu, chunk_size=C
    )
    o_ind, _ = _run_independent_batch(q, k, v, cu, C, g=g, g_gamma=g_gamma)

    assert compare_tensor("output", o_var, o_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)


# ============================================================================
# 2. CPU-only: varlen with initial state h0
# ============================================================================


def test_simple_gla_varlen_with_h0():
    """Varlen forward with initial state should match independent execution."""
    H, K, V, C = 2, 32, 64, 64
    cu = jnp.array([0, 64, 128])
    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q = jax.random.normal(keys[0], (1, 128, H, K))
    k = jax.random.normal(keys[1], (1, 128, H, K))
    v = jax.random.normal(keys[2], (1, 128, H, V))
    g = jax.random.normal(keys[3], (1, 128, H)) * 0.1
    h0 = jax.random.normal(keys[4], (2, H, K, V))

    o_var, ht_var = chunk_simple_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu, chunk_size=C
    )
    o_ind, ht_ind = _run_independent_batch(
        q, k, v, cu, C, g=g, initial_state=h0, output_final_state=True
    )

    assert compare_tensor("output", o_var, o_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)
    assert ht_var.shape == (2, H, K, V)
    assert compare_tensor("final_state", ht_var, ht_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)


# ============================================================================
# 3. GPU: CPU ref varlen vs FLA Triton varlen
# ============================================================================


@requires_cuda_and_fla
@pytest.mark.parametrize("cfg", GPU_VARLEN_CASES, ids=[_gpu_case_id(c) for c in GPU_VARLEN_CASES])
def test_simple_gla_varlen_cpu_vs_triton(cfg):
    """Compare CPU ref chunk_simple_gla(cu_seqlens=...) against FLA Triton."""
    T, H, K, V, C = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"]
    seed = cfg["seed"]
    gate = cfg["gate"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    torch_dtype = getattr(torch, cfg["dtype"])
    jax_dtype = getattr(jnp, cfg["dtype"])
    cu_list = cfg["cu_seqlens"]

    torch.manual_seed(seed)
    q_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    k_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    v_t = torch.randn(1, T, H, V, dtype=torch_dtype)
    g_t = torch.randn(1, T, H, dtype=torch_dtype) * 0.1 if gate == "g" else None

    cu_t = torch.tensor(cu_list, dtype=torch.long)

    # FLA Triton with cu_seqlens
    o_tri, ht_tri = triton_chunk_simple_gla(
        q_t.to("cuda"), k_t.to("cuda"), v_t.to("cuda"),
        g=g_t.to("cuda") if g_t is not None else None,
        cu_seqlens=cu_t.to("cuda"),
        output_final_state=True,
    )
    o_tri = o_tri.cpu()
    ht_tri = ht_tri.cpu() if ht_tri is not None else None

    # CPU ref with cu_seqlens
    q_j = _torch_to_jax(q_t, jax_dtype)
    k_j = _torch_to_jax(k_t, jax_dtype)
    v_j = _torch_to_jax(v_t, jax_dtype)
    g_j = _torch_to_jax(g_t, jax_dtype) if g_t is not None else None
    cu_j = jnp.array(cu_list)

    o_cpu, ht_cpu = chunk_simple_gla(
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
