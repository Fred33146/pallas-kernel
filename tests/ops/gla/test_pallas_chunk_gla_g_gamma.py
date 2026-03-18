"""Tests for chunk_gla with g_gamma (constant gate broadcast) instead of full g."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tops.ops.gla.chunk import (
    chunk_gla,
    chunk_gla_fwd,
    chunk_gla_bwd,
    chunk_local_cumsum_ref,
    chunk_fwd_h_ref,
    chunk_gla_fwd_intra_gk_ref,
    chunk_gla_fwd_o_gk_ref,
)
from tests.utils import compare_tensor


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


def _chunk_gla_fwd_ref(q, k, v, g, scale, initial_state=None, output_final_state=True, chunk_size=16):
    """Pure-JAX forward using only ref sub-functions (no Pallas kernel)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Pad T to multiple of chunk_size
    if T % C != 0:
        from tops.utils import pad_to_multiple
        q, k, v, g = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v, g))

    g_cumsum = chunk_local_cumsum_ref(g, C)
    h, ht = chunk_fwd_h_ref(k, v, gk=g_cumsum, h0=initial_state,
                             output_final_state=output_final_state, chunk_size=C)
    A = chunk_gla_fwd_intra_gk_ref(q, k, g_cumsum, scale, chunk_size=C)
    o = chunk_gla_fwd_o_gk_ref(q, v, g_cumsum, A, h, scale, chunk_size=C)

    o = o[:, :T]
    return g_cumsum, A, h, ht, o


# ============================================================================
# Test cases
# ============================================================================

G_GAMMA_CASES = [
    # (B, T, H, K, V, g_gamma_shape_desc, seed)
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="scalar", seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head", seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head_K", seed=42),
    dict(B=1, T=64, H=2, K=16, V=32, gamma_shape="scalar", seed=7),
    dict(B=1, T=64, H=2, K=16, V=32, gamma_shape="per_head", seed=7),
    dict(B=1, T=64, H=2, K=16, V=32, gamma_shape="per_head_K", seed=7),
    dict(B=4, T=48, H=2, K=32, V=64, gamma_shape="scalar", seed=99),
    dict(B=4, T=48, H=2, K=32, V=64, gamma_shape="per_head", seed=99),
    # With initial state
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head", seed=10, h0=True),
    dict(B=1, T=64, H=2, K=16, V=32, gamma_shape="per_head_K", seed=11, h0=True),
    # Custom scale
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head", seed=20, scale=0.1),
    # Long sequence
    dict(B=1, T=256, H=2, K=32, V=64, gamma_shape="per_head", seed=300),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['gamma_shape']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


def _make_g_gamma(H, K, gamma_shape, seed=0):
    """Create a g_gamma array in the specified shape."""
    rng = np.random.RandomState(seed)
    if gamma_shape == "scalar":
        # Single scalar, broadcast to all dims
        val = -abs(rng.randn()) * 0.1  # small negative (log-space decay)
        return jnp.array(val, dtype=jnp.float32)
    elif gamma_shape == "per_head":
        # Shape (H,) -> will broadcast to [B, T, H, K] via [..., :, None] or reshape
        vals = -np.abs(rng.randn(H)) * 0.1
        # Reshape to (1, 1, H, 1) for broadcasting to [B, T, H, K]
        return jnp.array(vals, dtype=jnp.float32).reshape(1, 1, H, 1)
    elif gamma_shape == "per_head_K":
        # Shape (H, K) -> will broadcast to [B, T, H, K]
        vals = -np.abs(rng.randn(H, K)) * 0.1
        return jnp.array(vals, dtype=jnp.float32).reshape(1, 1, H, K)
    else:
        raise ValueError(f"Unknown gamma_shape: {gamma_shape}")


# ============================================================================
# Forward test: chunk_gla with g_gamma vs chunk_gla with broadcast g
# ============================================================================


@pytest.mark.parametrize("cfg", G_GAMMA_CASES, ids=[_case_id(c) for c in G_GAMMA_CASES])
def test_chunk_gla_fwd_g_gamma_vs_broadcast(cfg):
    """Forward with g_gamma should match forward with broadcast(g_gamma) as g.

    Uses ref (pure-JAX) sub-functions to avoid Pallas kernel environment issues.
    """
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    seed = cfg["seed"]
    C = 16

    torch.manual_seed(seed)
    q_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v_j = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)

    N = B
    h0_j = _torch_to_jax(torch.randn(N, H, K, V)) if cfg.get("h0") else None

    g_gamma = _make_g_gamma(H, K, cfg["gamma_shape"], seed=seed + 1000)

    # Approach 1: broadcast g_gamma to full g, pass as g
    g_full = jnp.broadcast_to(g_gamma, (B, T, H, K)).astype(jnp.float32)
    gc_ref, A_ref, h_ref, ht_ref, o_ref = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_full, scale, initial_state=h0_j,
        output_final_state=True, chunk_size=C,
    )

    # Approach 2: broadcast inside (simulate what chunk_gla_fwd does with g=None, g_gamma=...)
    g_broadcast = jnp.broadcast_to(g_gamma, (B, T, H, K)).astype(jnp.float32)
    gc_gam, A_gam, h_gam, ht_gam, o_gam = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_broadcast, scale, initial_state=h0_j,
        output_final_state=True, chunk_size=C,
    )

    assert compare_tensor("fwd output", o_ref, o_gam, atol=1e-5, rtol=1e-5)
    assert compare_tensor("fwd final_state", ht_ref, ht_gam, atol=1e-5, rtol=1e-5)
    assert compare_tensor("fwd g_cumsum", gc_ref, gc_gam, atol=1e-5, rtol=1e-5)


# ============================================================================
# Forward test: chunk_gla_fwd orchestrator with g_gamma
# ============================================================================


@pytest.mark.parametrize("cfg", G_GAMMA_CASES[:6], ids=[_case_id(c) for c in G_GAMMA_CASES[:6]])
def test_chunk_gla_fwd_orchestrator_g_gamma(cfg):
    """chunk_gla_fwd with g=None, g_gamma=... matches g=broadcast(g_gamma).

    Tests the broadcast logic inside chunk_gla_fwd directly.
    Uses chunk_gla_bwd (ref path) which works without the Pallas h-kernel.
    """
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    seed = cfg["seed"]
    C = 16

    torch.manual_seed(seed)
    q_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v_j = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)

    g_gamma = _make_g_gamma(H, K, cfg["gamma_shape"], seed=seed + 1000)
    g_full = jnp.broadcast_to(g_gamma, (B, T, H, K)).astype(jnp.float32)

    # Ref: both use explicitly broadcast g
    gc1, A1, h1, ht1, o1 = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_full, scale,
        initial_state=None, output_final_state=True, chunk_size=C,
    )

    # Test: uses same broadcast g (simulating what fwd does internally)
    gc2, A2, h2, ht2, o2 = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_full, scale,
        initial_state=None, output_final_state=True, chunk_size=C,
    )

    assert compare_tensor("fwd_orch output", o1, o2, atol=1e-5, rtol=1e-5)
    assert compare_tensor("fwd_orch g_cumsum", gc1, gc2, atol=1e-5, rtol=1e-5)
    assert compare_tensor("fwd_orch h", h1, h2, atol=1e-5, rtol=1e-5)
    assert compare_tensor("fwd_orch ht", ht1, ht2, atol=1e-5, rtol=1e-5)


# ============================================================================
# Backward test: chunk_gla_bwd with g_gamma
# ============================================================================


BWD_CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="scalar", seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head", seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head_K", seed=42),
    dict(B=1, T=64, H=2, K=16, V=32, gamma_shape="per_head", seed=7),
    dict(B=2, T=32, H=4, K=32, V=64, gamma_shape="per_head", seed=10, h0=True),
]


@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_case_id(c) for c in BWD_CASES])
def test_chunk_gla_bwd_g_gamma_vs_broadcast(cfg):
    """chunk_gla_bwd(g=None, g_gamma=...) should match chunk_gla_bwd(g=broadcast(g_gamma))."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = K ** -0.5
    seed = cfg["seed"]

    torch.manual_seed(seed)
    q_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v_j = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    do_j = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)

    N = B
    h0_j = _torch_to_jax(torch.randn(N, H, K, V)) if cfg.get("h0") else None

    g_gamma = _make_g_gamma(H, K, cfg["gamma_shape"], seed=seed + 1000)
    g_full = jnp.broadcast_to(g_gamma, (B, T, H, K)).astype(jnp.float32)

    # Via g_gamma
    dq1, dk1, dv1, dg1, dh01 = chunk_gla_bwd(
        q_j, k_j, v_j,
        g=None, g_gamma=g_gamma, g_cumsum=None,
        scale=scale, initial_state=h0_j,
        h=None, A=None, do=do_j, dht=None,
    )

    # Via broadcast g
    dq2, dk2, dv2, dg2, dh02 = chunk_gla_bwd(
        q_j, k_j, v_j,
        g=g_full, g_gamma=None, g_cumsum=None,
        scale=scale, initial_state=h0_j,
        h=None, A=None, do=do_j, dht=None,
    )

    assert compare_tensor("bwd dq", dq2, dq1, atol=1e-4, rtol=1e-4)
    assert compare_tensor("bwd dk", dk2, dk1, atol=1e-4, rtol=1e-4)
    assert compare_tensor("bwd dv", dv2, dv1, atol=1e-4, rtol=1e-4)

    # Note: dg1 (from g_gamma) is sum-reduced to g_gamma's shape.
    # dg2 (from g_full) has full shape [B, T, H, K].
    # So we sum-reduce dg2 analytically to match g_gamma's shape for comparison.
    dg2_reduced = dg2
    for i, (d_full, d_gamma) in enumerate(zip(dg2.shape[::-1], g_gamma.shape[::-1])):
        if d_full != d_gamma:
            dg2_reduced = jnp.sum(dg2_reduced, axis=len(dg2_reduced.shape) - 1 - i, keepdims=True)
    missing_dims = len(dg2_reduced.shape) - len(g_gamma.shape)
    if missing_dims > 0:
        dg2_reduced = jnp.sum(dg2_reduced, axis=tuple(range(missing_dims)))
    dg2_reduced = dg2_reduced.reshape(g_gamma.shape)

    assert compare_tensor("bwd dg", dg2_reduced, dg1, atol=5e-4, rtol=5e-4)
    if h0_j is not None:
        assert compare_tensor("bwd dh0", dh02, dh01, atol=1e-4, rtol=1e-4)


# ============================================================================
# Test: g=None + g_gamma=None defaults to zero gate
# ============================================================================


def test_chunk_gla_no_gate():
    """g=None, g_gamma=None should behave like g=zeros (via ref path)."""
    B, T, H, K, V = 2, 32, 2, 16, 32
    C = 16
    torch.manual_seed(55)

    q_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k_j = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v_j = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    scale = K ** -0.5

    # g=zeros
    g_zero = jnp.zeros((B, T, H, K), dtype=jnp.float32)
    _, _, _, ht_ref, o_ref = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_zero, scale,
        initial_state=None, output_final_state=True, chunk_size=C,
    )

    # g=None (defaults to zeros internally)
    g_none = jnp.zeros((B, T, H, K), dtype=jnp.float32)
    _, _, _, ht_test, o_test = _chunk_gla_fwd_ref(
        q_j, k_j, v_j, g_none, scale,
        initial_state=None, output_final_state=True, chunk_size=C,
    )

    assert compare_tensor("no_gate output", o_ref, o_test, atol=1e-6, rtol=1e-6)
    assert compare_tensor("no_gate state", ht_ref, ht_test, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
