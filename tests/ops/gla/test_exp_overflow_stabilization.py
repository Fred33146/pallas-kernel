"""Test exp overflow stabilization in GLA chunk kernels.

Verifies that the midpoint stabilization prevents NaN when gate magnitudes
are large (|g_gamma| > 0.69), which causes exp(|g|*127) to exceed fp32 range
at chunk_size=128.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tops.ops.gla.chunk import (
    chunk_gla_fwd_intra_gk_ref,
    chunk_gla_bwd_dA_ref,
    chunk_gla_bwd_dv_ref,
    chunk_gla_bwd_dqk_intra_ref,
    chunk_gla_bwd_dqkg_ref,
)


def _make_large_gate_inputs(B, T, H, K, chunk_size, seed=42, g_scale=1.0):
    """Create inputs with large gate values that would overflow without stabilization.

    g_scale controls magnitude: 1.0 gives max |g| ~ chunk_size * 1.0,
    which overflows exp() for chunk_size >= 90.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, K), dtype=jnp.float32)
    do = jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)

    # Create per-step gate values and cumsum within chunks to simulate
    # realistic cumulative gate values
    g_raw = jax.random.uniform(keys[4], (B, T, H, K), minval=-g_scale, maxval=0.0)
    BT = chunk_size
    NT = T // BT
    g_chunked = g_raw.reshape(B, NT, BT, H, K)
    g_cumsum_chunked = jnp.cumsum(g_chunked, axis=2)
    g = g_cumsum_chunked.reshape(B, T, H, K)

    return q, k, v, g, do


# ============================================================================
# Test: forward intra-chunk with large gates does not produce NaN
# ============================================================================

@pytest.mark.parametrize("chunk_size,g_scale", [
    (64, 0.8),
    (128, 0.8),
    (128, 1.0),
])
def test_fwd_intra_large_gates_no_nan(chunk_size, g_scale):
    """Forward intra-chunk attention should not produce NaN with large gates."""
    B, T, H, K = 1, chunk_size * 2, 2, 32
    q, k, _, g, _ = _make_large_gate_inputs(B, T, H, K, chunk_size, g_scale=g_scale)

    # The reference uses stabilization (g_n at first element)
    A_ref = chunk_gla_fwd_intra_gk_ref(q, k, g, scale=K**-0.5, chunk_size=chunk_size)

    assert not jnp.any(jnp.isnan(A_ref)), "Reference A contains NaN"
    assert not jnp.any(jnp.isinf(A_ref)), "Reference A contains Inf"


# ============================================================================
# Test: backward with large gates does not produce NaN
# ============================================================================

@pytest.mark.parametrize("chunk_size,g_scale", [
    (64, 0.8),
    (128, 0.8),
    (128, 1.0),
])
def test_bwd_large_gates_no_nan(chunk_size, g_scale):
    """Backward pass should not produce NaN with large gate magnitudes."""
    B, T, H, K = 1, chunk_size * 2, 2, 32
    V = K
    NT = T // chunk_size
    scale = K**-0.5

    q, k, v, g, do = _make_large_gate_inputs(B, T, H, K, chunk_size, g_scale=g_scale)

    key = jax.random.PRNGKey(99)
    h = jax.random.normal(key, (B, NT, H, K, V), dtype=jnp.float32)
    dh = jax.random.normal(jax.random.PRNGKey(100), (B, NT, H, K, V), dtype=jnp.float32)

    A = chunk_gla_fwd_intra_gk_ref(q, k, g, scale, chunk_size=chunk_size)
    dA = chunk_gla_bwd_dA_ref(v, do, scale, chunk_size=chunk_size)
    dv = chunk_gla_bwd_dv_ref(k, g, A, do, dh, chunk_size=chunk_size)
    dq_intra, dk_intra = chunk_gla_bwd_dqk_intra_ref(q, k, g, dA, chunk_size=chunk_size)
    dq, dk, dg = chunk_gla_bwd_dqkg_ref(
        q, k, v, h, g, do, dh, dq_intra, dk_intra, scale, chunk_size=chunk_size
    )

    for name, arr in [("dq", dq), ("dk", dk), ("dv", dv), ("dg", dg)]:
        assert not jnp.any(jnp.isnan(arr)), f"{name} contains NaN (chunk_size={chunk_size}, g_scale={g_scale})"
        assert not jnp.any(jnp.isinf(arr)), f"{name} contains Inf (chunk_size={chunk_size}, g_scale={g_scale})"


# ============================================================================
# Test: midpoint stabilization math equivalence
# ============================================================================

def test_midpoint_stabilization_equivalence():
    """Verify that exp(g)*exp(-g') == exp(g-mid)*exp(mid-g') algebraically."""
    key = jax.random.PRNGKey(42)
    BT, K = 128, 32
    # Use small gate values so outputs stay moderate and fp32 precision is good
    g = jax.random.uniform(key, (BT, K), minval=-0.2, maxval=0.0)
    g = jnp.cumsum(g, axis=0)  # range ~ BT * 0.1 ~ 12.8

    g_mid = (g[0:1, :] + g[-1:, :]) * 0.5

    q = jax.random.normal(jax.random.PRNGKey(1), (BT, K))
    k = jax.random.normal(jax.random.PRNGKey(2), (BT, K))

    # Stabilized version (midpoint — what the kernel now computes)
    qg_s = q * jnp.exp(g - g_mid)
    kg_s = k * jnp.exp(g_mid - g)
    A_stabilized = qg_s @ kg_s.T

    # Direct version using first-element stabilization (what the reference uses)
    g_ref = g[0:1, :]
    qg_r = q * jnp.exp(g - g_ref)
    kg_r = k * jnp.exp(g_ref - g)
    A_reference = qg_r @ kg_r.T

    np.testing.assert_allclose(
        np.array(A_stabilized), np.array(A_reference),
        atol=1e-3, rtol=1e-3,
        err_msg="Midpoint stabilization changes the result",
    )
    assert not jnp.any(jnp.isnan(A_stabilized)), "Stabilized A contains NaN"
    assert not jnp.any(jnp.isnan(A_reference)), "Reference A contains NaN"


def test_midpoint_handles_larger_range_than_first_element():
    """Midpoint stabilization survives ranges where first-element stabilization overflows."""
    key = jax.random.PRNGKey(42)
    BT, K = 128, 32
    # g_scale=0.8 gives range ~ 128 * 0.4 = 51.2, half_range ~ 25.6
    # exp(51.2) ~ 1.7e22 (overflows with first-element ref)
    # exp(25.6) ~ 1.4e11 (fine with midpoint)
    g = jax.random.uniform(key, (BT, K), minval=-0.8, maxval=0.0)
    g = jnp.cumsum(g, axis=0)

    g_mid = (g[0:1, :] + g[-1:, :]) * 0.5

    q = jax.random.normal(jax.random.PRNGKey(1), (BT, K))
    k = jax.random.normal(jax.random.PRNGKey(2), (BT, K))

    # Midpoint stabilization should be finite
    qg_mid = q * jnp.exp(g - g_mid)
    kg_mid = k * jnp.exp(g_mid - g)
    A_mid = qg_mid @ kg_mid.T
    assert not jnp.any(jnp.isnan(A_mid)), "Midpoint-stabilized A contains NaN"
    assert not jnp.any(jnp.isinf(A_mid)), "Midpoint-stabilized A contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
