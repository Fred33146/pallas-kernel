"""chunk_simple_gla: forward (Triton vs JAX) & backward (Triton vs JAX Pallas)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ["TRITON_F32_DEFAULT"] = "ieee"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid OOM errors with large T and chunk_size=128
os.environ["PALLAS_INTERPRET"]="1"

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import numpy as np
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd, chunk_simple_gla_bwd
from tests.utils import compare_tensor, torch_to_jax, make_alibi_g_gamma

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This test requires CUDA. Please run on a machine with an NVIDIA GPU."

from fla.ops.simple_gla.chunk import (
    chunk_simple_gla_fwd as triton_fwd,
    chunk_simple_gla_bwd as triton_bwd,
)

fp32_atol = 1e-4
fp32_rtol = 1e-4
bf16_atol = 1e-3
bf16_rtol = 1e-3
CHUNK_SIZE = 64
default_dtype = torch.bfloat16

def _tol(T, chunk_size=CHUNK_SIZE, base_atol=bf16_atol, base_ulp=2):
    """Compute NT-scaled tolerance for cross-platform bf16 comparison.

    Errors compound across chunks due to different matmul tiling/precision
    paths between Triton GPU tensor core and JAX GPU interpret mode.
    """
    NT = T // chunk_size
    return dict(atol=max(base_atol, base_atol * NT), max_ulp=max(base_ulp, base_ulp * NT))

# ============================================================================
# Test configs
#
# Constraints for JAX chunk path:
#   - K, V must be multiples of 128 (chunk_fwd_h_kernel Pallas requirement)
#   - T must be a multiple of chunk_size (default 64)
#   - gate: only "g_gamma" or "none" (JAX chunk_fwd_h does not support scalar g yet)
# ============================================================================

CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, **_tol(64, base_atol=5e-2)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13, **_tol(128, base_atol=5e-2)),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7, **_tol(256, base_atol=5e-2)),
    # ── with h0 ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, h0=True, **_tol(64, base_atol=5e-2)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13, h0=True, **_tol(128, base_atol=5e-2)),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7, h0=True, **_tol(256, base_atol=5e-2)),
    # ── single head ──
    dict(B=2, T=64, H=1, K=128, V=128, seed=10, **_tol(64, base_atol=5e-2)),
    # ── K != V (both multiples of 128) ──
    dict(B=2, T=64, H=4, K=128, V=256, seed=20, **_tol(64, base_atol=5e-2)),
    dict(B=2, T=64, H=4, K=256, V=128, seed=21, **_tol(64, base_atol=5e-2)),
    # ── minimal T = chunk_size ──
    dict(B=2, T=64, H=2, K=128, V=128, seed=30, **_tol(64, base_atol=5e-2)),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=128, V=128, seed=50, **_tol(64, base_atol=5e-2)),
    # ── many heads ──
    dict(B=1, T=128, H=16, K=128, V=128, seed=60, **_tol(128, base_atol=5e-2)),
    # ── no gate ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=100, gate="none", **_tol(64, base_atol=5e-2)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=101, gate="none", **_tol(128, base_atol=5e-2)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=102, gate="none", h0=True, **_tol(64, base_atol=5e-2)),
    # ── g_gamma only ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=150, gate="g_gamma", **_tol(64, base_atol=5e-2)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=151, gate="g_gamma", **_tol(128, base_atol=5e-2)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=152, gate="g_gamma", h0=True, **_tol(64, base_atol=5e-2)),
    dict(B=1, T=256, H=2, K=128, V=128, seed=153, gate="g_gamma", **_tol(256, base_atol=5e-2)),
    dict(B=1, T=64, H=2, K=128, V=256, seed=154, gate="g_gamma", **_tol(64, base_atol=5e-2)),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=200, scale=0.1, **_tol(64, base_atol=5e-2)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=201, scale=0.1, h0=True, **_tol(64, base_atol=5e-2)),
    # ── longer sequences ──
    dict(B=1, T=512, H=2, K=128, V=128, seed=300, **_tol(512, base_atol=5e-2)),
    dict(B=1, T=512, H=2, K=128, V=128, seed=301, h0=True, **_tol(512, base_atol=5e-2)),
    dict(B=1, T=512, H=2, K=128, V=128, seed=302, gate="g_gamma", **_tol(512, base_atol=5e-2)),
    dict(B=1, T=512, H=2, K=128, V=128, seed=303, gate="g_gamma", h0=True, **_tol(512, base_atol=5e-2)),
    # ── multi-batch + long ──
    dict(B=4, T=256, H=2, K=128, V=128, seed=360, **_tol(256, base_atol=5e-2)),
    dict(B=2, T=256, H=4, K=128, V=128, seed=361, gate="g_gamma", **_tol(256, base_atol=5e-2)),
    dict(B=2, T=4096, H=16, K=128, V=128, seed=362, gate="g_gamma", h0=True, **_tol(4096, base_atol=5e-2)),
    dict(B=8, T=4096, H=16, K=128, V=128, seed=363, gate="g_gamma", h0=True, chunk_size=64, **_tol(4096, 64, base_atol=5e-2)),
    dict(B=8, T=4096, H=16, K=128, V=128, seed=364, gate="g_gamma", h0=True, chunk_size=128, **_tol(4096, 128, base_atol=5e-2)),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "none")
    if gate != "none":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    if c.get("chunk_size", CHUNK_SIZE) != CHUNK_SIZE:
        parts.append(f"C{c['chunk_size']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _run_triton_fwd(q, k, v, *, g=None, g_gamma=None, h0=None, scale=None,
                    chunk_size=CHUNK_SIZE):
    """Run Triton chunk_simple_gla_fwd directly."""
    o, ht = triton_fwd(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE),
        g=g.to(DEVICE) if g is not None else None,
        g_gamma=g_gamma.to(DEVICE) if g_gamma is not None else None,
        scale=scale,
        initial_state=h0.to(DEVICE) if h0 is not None else None,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    return o.cpu(), ht.cpu() if ht is not None else None


def _run_triton_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None, dht=None,
                    scale=None, chunk_size=CHUNK_SIZE):
    """Run Triton chunk_simple_gla_bwd directly."""
    dq, dk, dv, _, dh0 = triton_bwd(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE),
        g=g.to(DEVICE) if g is not None else None,
        g_gamma=g_gamma.to(DEVICE) if g_gamma is not None else None,
        initial_state=h0.to(DEVICE) if h0 is not None else None,
        do=do.to(DEVICE),
        dht=dht.to(DEVICE) if dht is not None else None,
        scale=scale,
        chunk_size=chunk_size,
    )
    return dq.cpu(), dk.cpu(), dv.cpu(), dh0.cpu() if dh0 is not None else None


def _run_jax_fwd(q, k, v, *, g_gamma=None, h0=None, scale=None,
                 chunk_size=CHUNK_SIZE):
    """Run JAX chunk_simple_gla_fwd."""
    q_j = torch_to_jax(q)
    k_j = torch_to_jax(k)
    v_j = torch_to_jax(v)
    g_gamma_j = torch_to_jax(g_gamma) if g_gamma is not None else None
    h0_j = torch_to_jax(h0) if h0 is not None else None

    o, ht = chunk_simple_gla_fwd(
        q_j, k_j, v_j,
        g=None,
        g_gamma=g_gamma_j,
        scale=scale,
        h0=h0_j,
        use_ht=True,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Parametrized test — Triton chunk (gold) vs JAX chunk
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_triton_chunk_vs_jax_chunk(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    chunk_size = cfg.get("chunk_size", CHUNK_SIZE)
    dtype = cfg.get("dtype", default_dtype)

    default_atol = bf16_atol if dtype == torch.bfloat16 else fp32_atol
    default_rtol = bf16_rtol if dtype == torch.bfloat16 else fp32_rtol
    atol = cfg.get("atol", default_atol)
    rtol = cfg.get("rtol", default_rtol)
    max_ulp = cfg.get("max_ulp", 2)
    gate = cfg.get("gate", "none")
    scale = cfg.get("scale", None)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)

    g_gamma = None
    if gate == "g_gamma":
        g_gamma = torch.from_numpy(np.array(make_alibi_g_gamma(H, 32, 0)))

    N = B
    h0 = torch.randn(N, H, K, V, dtype=dtype) if cfg.get("h0") else None

    o_tri, ht_tri = _run_triton_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=chunk_size)
    o_jax, ht_jax = _run_jax_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=chunk_size)

    assert compare_tensor("output", o_tri, o_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if ht_tri is not None and ht_jax is not None:
        assert compare_tensor("final_state", ht_tri, ht_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)


# ============================================================================
# Backward: JAX Pallas vs Triton (g_gamma only)
# ============================================================================

BWD_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7, **_tol(128, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=13, h0=True, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=1, K=128, V=128, seed=10, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=20, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=21, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=400, **_tol(128, base_atol=5e-2, base_ulp=40)),
    dict(B=1, T=64, H=2, K=128, V=128, seed=41, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=1, T=256, H=2, K=128, V=128, seed=300, **_tol(256, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=99, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=128, H=4, K=128, V=128, seed=503, chunk_size=64, **_tol(128, 64, base_atol=5e-2, base_ulp=40)),
    # ── with dht (terminal state gradient) ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=600, dht=True, **_tol(64, base_atol=5e-2, base_ulp=40)),
    dict(B=1, T=128, H=2, K=128, V=128, seed=601, dht=True, **_tol(128, base_atol=5e-2, base_ulp=40)),
    dict(B=2, T=64, H=4, K=128, V=128, seed=602, h0=True, dht=True, **_tol(64, base_atol=5e-2, base_ulp=80)),
    dict(B=1, T=256, H=2, K=128, V=128, seed=603, h0=True, dht=True, **_tol(256, base_atol=5e-2, base_ulp=40)),
]


def _bwd_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_bwd_case_id(c) for c in BWD_CASES])
def test_simple_gla_bwd_gamma(cfg):
    """JAX chunk_simple_gla_bwd should match Triton backward (g_gamma only)."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = cfg.get("chunk_size", CHUNK_SIZE)
    dtype = cfg.get("dtype", default_dtype)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    do = torch.randn(B, T, H, V, dtype=dtype)
    g_gamma = torch.from_numpy(np.array(make_alibi_g_gamma(H, 32, 0)))

    N = B
    h0 = torch.randn(N, H, K, V, dtype=dtype) if cfg.get("h0") else None
    dht = torch.randn(N, H, K, V, dtype=dtype) if cfg.get("dht") else None

    # Triton gold (direct fwd/bwd)
    dq_gold, dk_gold, dv_gold, dh0_gold = _run_triton_bwd(
        q, k, v, do,
        g_gamma=g_gamma, h0=h0, dht=dht, scale=scale, chunk_size=C,
    )

    # JAX Pallas
    q_j = torch_to_jax(q)
    k_j = torch_to_jax(k)
    v_j = torch_to_jax(v)
    do_j = torch_to_jax(do)
    g_gamma_j = torch_to_jax(g_gamma)
    h0_j = torch_to_jax(h0) if h0 is not None else None
    dht_j = torch_to_jax(dht) if dht is not None else None

    dq_jax, dk_jax, dv_jax, dh0_jax = chunk_simple_gla_bwd(
        q_j, k_j, v_j, do_j,
        g_gamma=g_gamma_j, scale=scale, h0=h0_j, dht=dht_j, chunk_size=C,
    )

    default_atol = bf16_atol if dtype == torch.bfloat16 else fp32_atol
    default_rtol = bf16_rtol if dtype == torch.bfloat16 else fp32_rtol
    atol = cfg.get("atol", default_atol)
    rtol = cfg.get("rtol", default_rtol)
    max_ulp = cfg.get("max_ulp", 2)
    assert compare_tensor("dq", dq_gold, dq_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dk", dk_gold, dk_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dv", dv_gold, dv_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if dh0_gold is not None and dh0_jax is not None:
        assert compare_tensor("dh0", dh0_gold, dh0_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)


# ============================================================================
# NaN stability: chunk_size=128 with realistic ALiBi g_gamma
# ============================================================================


NAN_CASES = [
    dict(B=8, T=4096, H=16, K=128, V=128, num_layers=20, layer_idx=1, seed=42),
    dict(B=8, T=4096, H=16, K=128, V=128, num_layers=20, layer_idx=0, seed=43),
    dict(B=8, T=4096, H=16, K=128, V=128, num_layers=20, layer_idx=5, seed=44),
]


def _nan_case_id(c):
    return f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_layer{c['layer_idx']}"


@pytest.mark.parametrize("cfg", NAN_CASES, ids=[_nan_case_id(c) for c in NAN_CASES])
def test_chunk128_no_nan(cfg):
    """chunk_size=128 with ALiBi g_gamma should not produce NaN/Inf."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    jax.clear_caches()

    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    g_gamma = make_alibi_g_gamma(H, cfg["num_layers"], cfg["layer_idx"])

    key = jax.random.PRNGKey(cfg["seed"])
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    do = jax.random.normal(k4, (B, T, H, V), dtype=jnp.bfloat16)

    # Forward
    o, _ = chunk_simple_gla_fwd(q, k_arr, v, g_gamma=g_gamma, chunk_size=128)
    assert not jnp.any(jnp.isnan(o)), f"NaN in forward (layer {cfg['layer_idx']})"
    assert not jnp.any(jnp.isinf(o)), f"Inf in forward (layer {cfg['layer_idx']})"

    # Backward
    dq, dk, dv, _ = chunk_simple_gla_bwd(q, k_arr, v, do, g_gamma=g_gamma, chunk_size=128)
    for name, grad in [("dq", dq), ("dk", dk), ("dv", dv)]:
        assert not jnp.any(jnp.isnan(grad)), f"NaN in {name} (layer {cfg['layer_idx']})"
        assert not jnp.any(jnp.isinf(grad)), f"Inf in {name} (layer {cfg['layer_idx']})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
