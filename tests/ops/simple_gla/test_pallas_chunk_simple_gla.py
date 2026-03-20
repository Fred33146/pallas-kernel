"""Tests for chunk_simple_gla: Simple GLA with scalar-per-head gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.chunk import (
    chunk_simple_gla_fwd_ref,
    chunk_simple_gla_fwd_intra_ref,
    chunk_simple_gla_fwd_o_ref,
)
from tops.ops.gla.chunk import (
    chunk_gla_fwd_intra_gk_ref,
    chunk_gla_fwd_o_gk_ref,
    chunk_local_cumsum_ref,
)
from tops.ops.common.chunk_h import chunk_fwd_h_ref
from tests.utils import compare_tensor


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


def _make_g_gamma(H, seed=0):
    """Create per-head g_gamma (1, 1, H, 1) with small negative values."""
    rng = np.random.RandomState(seed)
    vals = -np.abs(rng.randn(H)) * 0.1
    return jnp.array(vals, dtype=jnp.float32).reshape(1, 1, H, 1)


# ============================================================================
# Test cases
# ============================================================================

CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=1, T=64, H=2, K=16, V=32, seed=7),
    dict(B=4, T=48, H=2, K=32, V=64, seed=99),
    dict(B=2, T=32, H=4, K=32, V=64, seed=10, h0=True),
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=2, T=32, H=4, K=32, V=64, seed=20, scale=0.1),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Reference tests: Simple GLA ref vs Full GLA ref (pure JAX, no Pallas)
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_simple_gla_intra_ref_vs_full_gla(cfg):
    """Simple GLA intra-chunk ref should match full GLA intra-chunk ref with broadcast g_gamma."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    C = 16

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)

    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    # Pad T
    if T % C != 0:
        from tops.utils import pad_to_multiple
        q, k = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k))

    # Full GLA reference with broadcast g_cumsum
    g_full = jnp.broadcast_to(g_gamma, q.shape)
    g_cumsum = chunk_local_cumsum_ref(g_full, C)
    A_full = chunk_gla_fwd_intra_gk_ref(q, k, g_cumsum, scale, chunk_size=C)

    # Simple GLA reference
    A_simple = chunk_simple_gla_fwd_intra_ref(q, k, g_gamma, scale, chunk_size=C)

    assert compare_tensor("intra_ref A", A_full, A_simple, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_simple_gla_fwd_ref_vs_full_gla(cfg):
    """Simple GLA full forward ref should match full GLA forward ref with broadcast g_gamma."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    C = 16

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)

    N = B
    h0 = _torch_to_jax(torch.randn(N, H, K, V)) if cfg.get("h0") else None
    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    # Full GLA reference with broadcast g
    from tops.utils import pad_to_multiple
    q_p, k_p, v_p = q, k, v
    if T % C != 0:
        q_p, k_p, v_p = (pad_to_multiple(x, C, axis=1, val=0) for x in (q_p, k_p, v_p))
    g_full = jnp.broadcast_to(g_gamma, q_p.shape)
    g_cumsum = chunk_local_cumsum_ref(g_full, C)
    h_ref, ht_ref = chunk_fwd_h_ref(k_p, v_p, gk=g_cumsum, h0=h0,
                                     output_final_state=True, chunk_size=C)
    A_ref = chunk_gla_fwd_intra_gk_ref(q_p, k_p, g_cumsum, scale, chunk_size=C)
    o_ref = chunk_gla_fwd_o_gk_ref(q_p, v_p, g_cumsum, A_ref, h_ref, scale, chunk_size=C)
    o_ref = o_ref[:, :T]

    # Simple GLA reference
    ht_simple, o_simple = chunk_simple_gla_fwd_ref(
        q, k, v, g_gamma, scale,
        initial_state=h0, output_final_state=True, chunk_size=C,
    )

    assert compare_tensor("fwd_ref output", o_ref, o_simple, atol=1e-4, rtol=1e-4)
    assert compare_tensor("fwd_ref state", ht_ref, ht_simple, atol=1e-4, rtol=1e-4)


from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd_intra, chunk_simple_gla_fwd_o
from tops.ops.simple_gla.chunk import (
    chunk_simple_gla_pallas_fwd,
    chunk_simple_gla_bwd,
    chunk_simple_gla,
)
from tops.ops.gla.chunk import chunk_gla_bwd


PALLAS_INTRA_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7),
]


@pytest.mark.parametrize("cfg", PALLAS_INTRA_CASES, ids=[_case_id(c) for c in PALLAS_INTRA_CASES])
def test_simple_gla_intra_pallas_vs_ref(cfg):
    """Pallas intra-chunk kernel should match reference."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = K ** -0.5
    C = 64

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    A_ref = chunk_simple_gla_fwd_intra_ref(q, k, g_gamma, scale, chunk_size=C)
    A_pl = chunk_simple_gla_fwd_intra(q, k, g_gamma, scale, chunk_size=C)

    assert compare_tensor("intra_pallas A", A_ref, A_pl, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cfg", PALLAS_INTRA_CASES, ids=[_case_id(c) for c in PALLAS_INTRA_CASES])
def test_simple_gla_output_pallas_vs_ref(cfg):
    """Pallas output kernel should match reference."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = K ** -0.5
    C = 64

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    # Compute h and A using reference
    g_full = jnp.broadcast_to(g_gamma, q.shape)
    g_cumsum = chunk_local_cumsum_ref(g_full, C)
    h, _ = chunk_fwd_h_ref(k, v, gk=g_cumsum, h0=None,
                           output_final_state=False, chunk_size=C)
    A = chunk_simple_gla_fwd_intra_ref(q, k, g_gamma, scale, chunk_size=C)

    o_ref = chunk_simple_gla_fwd_o_ref(q, v, g_gamma, A, h, scale, chunk_size=C)
    o_pl = chunk_simple_gla_fwd_o(q, v, A, h, g_gamma, scale, chunk_size=C)

    assert compare_tensor("output_pallas o", o_ref, o_pl, atol=1e-4, rtol=1e-4)


ORCH_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7),
    dict(B=2, T=64, H=4, K=128, V=128, seed=10, h0=True),
]


@pytest.mark.parametrize("cfg", ORCH_CASES, ids=[_case_id(c) for c in ORCH_CASES])
def test_simple_gla_fwd_pallas_vs_ref(cfg):
    """Full Pallas forward should match reference forward."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = K ** -0.5
    C = 64

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    h0 = _torch_to_jax(torch.randn(B, H, K, V)) if cfg.get("h0") else None
    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    ht_ref, o_ref = chunk_simple_gla_fwd_ref(
        q, k, v, g_gamma, scale, initial_state=h0,
        output_final_state=True, chunk_size=C,
    )
    ht_pl, o_pl = chunk_simple_gla_pallas_fwd(
        q, k, v, g_gamma, scale, initial_state=h0,
        output_final_state=True, chunk_size=C,
    )

    assert compare_tensor("fwd_pallas output", o_ref, o_pl, atol=1e-4, rtol=1e-4)
    assert compare_tensor("fwd_pallas state", ht_ref, ht_pl, atol=1e-4, rtol=1e-4)


BWD_CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=1, T=64, H=2, K=16, V=32, seed=7),
    dict(B=2, T=32, H=4, K=32, V=64, seed=10, h0=True),
]


@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_case_id(c) for c in BWD_CASES])
def test_simple_gla_bwd_vs_full_gla(cfg):
    """Simple GLA backward should match chunk_gla_bwd with broadcast g_gamma."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = K ** -0.5

    torch.manual_seed(cfg["seed"])
    q = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    k = _torch_to_jax(torch.randn(B, T, H, K)).astype(jnp.float32)
    v = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    do = _torch_to_jax(torch.randn(B, T, H, V)).astype(jnp.float32)
    h0 = _torch_to_jax(torch.randn(B, H, K, V)) if cfg.get("h0") else None
    g_gamma = _make_g_gamma(H, seed=cfg["seed"] + 1000)

    # Simple GLA backward
    dq1, dk1, dv1, dg1, dh01 = chunk_simple_gla_bwd(
        q, k, v, g_gamma, scale, initial_state=h0,
        do=do, dht=None, chunk_size=16,
    )

    # Full GLA backward (direct)
    dq2, dk2, dv2, dg2, dh02 = chunk_gla_bwd(
        q, k, v,
        g=None, g_gamma=g_gamma, g_cumsum=None,
        scale=scale, initial_state=h0,
        h=None, A=None, do=do, dht=None, chunk_size=16,
    )

    assert compare_tensor("bwd dq", dq2, dq1, atol=1e-5, rtol=1e-5)
    assert compare_tensor("bwd dk", dk2, dk1, atol=1e-5, rtol=1e-5)
    assert compare_tensor("bwd dv", dv2, dv1, atol=1e-5, rtol=1e-5)
    assert compare_tensor("bwd dg", dg2, dg1, atol=1e-5, rtol=1e-5)
    if h0 is not None:
        assert compare_tensor("bwd dh0", dh02, dh01, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
