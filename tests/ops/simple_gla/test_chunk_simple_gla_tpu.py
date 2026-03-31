"""chunk_simple_gla: Pallas kernel accuracy vs JAX CPU reference.

Forward: chunk_simple_gla_fwd (tops.ops, Pallas) vs chunk_simple_gla_fwd (tops.cpu, pure JAX CPU)
Backward: chunk_simple_gla_bwd (tops.ops, Pallas) vs chunk_simple_gla_bwd (tops.cpu, pure JAX CPU)

The tops.cpu implementations are the ground-truth reference.
This test validates that the Pallas kernels produce equivalent results
on both CPU (interpret mode) and TPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd, chunk_simple_gla_bwd
from tops.cpu.ops.simple_gla import chunk_simple_gla_fwd as cpu_chunk_simple_gla_fwd
from tops.cpu.ops.simple_gla import chunk_simple_gla_bwd as cpu_chunk_simple_gla_bwd
from tests.utils import compare_tensor

# ============================================================================
# Test configs
#
# Constraints for JAX chunk path (tops.ops):
#   - K, V must be multiples of 128 (Pallas kernel alignment requirement)
#   - T must be a multiple of chunk_size (default 64)
#   - gate: only "g_gamma" or "none" (chunk path does not support scalar g)
# ============================================================================

CHUNK_SIZE = 64

FWD_CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=150, gate="g_gamma"),
    dict(B=2, T=128, H=4, K=128, V=128, seed=151, gate="g_gamma"),
    dict(B=1, T=256, H=2, K=128, V=128, seed=153, gate="g_gamma"),
    # ── with h0 ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=152, gate="g_gamma", h0=True),
    dict(B=2, T=128, H=4, K=128, V=128, seed=160, gate="g_gamma", h0=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=161, gate="g_gamma", h0=True),
    # ── single head ──
    dict(B=2, T=64, H=1, K=128, V=128, seed=162, gate="g_gamma"),
    # ── K != V (both multiples of 128) ──
    dict(B=1, T=64, H=2, K=128, V=256, seed=154, gate="g_gamma"),
    dict(B=2, T=64, H=4, K=256, V=128, seed=163, gate="g_gamma"),
    # ── minimal T = chunk_size ──
    dict(B=2, T=64, H=2, K=128, V=128, seed=164, gate="g_gamma"),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=128, V=128, seed=165, gate="g_gamma"),
    # ── many heads ──
    dict(B=1, T=128, H=16, K=128, V=128, seed=166, gate="g_gamma"),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=200, gate="g_gamma", scale=0.1),
    dict(B=2, T=64, H=4, K=128, V=128, seed=201, gate="g_gamma", scale=0.1, h0=True),
    # ── longer sequences ──
    dict(B=1, T=512, H=2, K=128, V=128, seed=302, gate="g_gamma"),
    dict(B=1, T=512, H=2, K=128, V=128, seed=303, gate="g_gamma", h0=True),
    dict(B=2, T=4096, H=16, K=128, V=128, seed=304, gate="g_gamma", h0=True),
    dict(B=2, T=8192, H=16, K=128, V=128, seed=305, gate="g_gamma", h0=True),
    # ── multi-batch + long ──
    dict(B=4, T=256, H=2, K=128, V=128, seed=360, gate="g_gamma"),
    dict(B=2, T=256, H=4, K=128, V=128, seed=361, gate="g_gamma"),
]


def _fwd_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "none")
    if gate != "none":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(cfg, *, dtype=jnp.float32):
    """Generate random inputs as JAX arrays with the given config."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)

    gate = cfg.get("gate", "none")
    g_gamma = None
    if gate == "g_gamma":
        # negative log-decay per head
        g_gamma = -jnp.abs(jax.random.normal(keys[3], (H,), dtype=jnp.float32)) * 0.5

    N = B
    h0 = None
    if cfg.get("h0"):
        h0 = jax.random.normal(keys[4], (N, H, K, V), dtype=dtype)

    return q, k, v, g_gamma, h0


def _run_cpu_chunk_fwd(q, k, v, *, g_gamma=None, h0=None, scale=None,
                      chunk_size=CHUNK_SIZE):
    """Run tops.cpu chunk_simple_gla_fwd (CPU chunk reference)."""
    K = q.shape[-1]
    s = scale if scale is not None else K**-0.5
    o, ht = cpu_chunk_simple_gla_fwd(
        q, k, v,
        g=None,
        g_gamma=g_gamma,
        scale=s,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    return o, ht


def _run_pallas_fwd(q, k, v, *, g_gamma=None, h0=None, scale=None,
                    chunk_size=CHUNK_SIZE):
    """Run tops.ops chunk_simple_gla_fwd (Pallas on TPU, interpret on CPU)."""
    o, ht = chunk_simple_gla_fwd(
        q, k, v,
        g=None,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        use_ht=True,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Forward: Pallas chunk vs CPU naive reference
# ============================================================================


@pytest.mark.parametrize("cfg", FWD_CASES, ids=[_fwd_case_id(c) for c in FWD_CASES])
def test_chunk_fwd_vs_cpu(cfg):
    """chunk_simple_gla_fwd (Pallas) should match cpu chunk_simple_gla_fwd reference."""
    T = cfg["T"]
    NT = T // CHUNK_SIZE
    # Chunked accumulation introduces rounding differences; scale tolerance
    # with the number of chunks.
    # bf16 matmul accumulation order differs; errors compound across chunks.
    # Base tolerance accounts for bf16 precision (~7.8e-3 relative) and
    # cancellation in K=128-dim dot products.  Additional per-chunk term
    # covers inter-chunk state propagation error.
    atol = cfg.get("atol", min(5e-2, 1e-2 + 1e-2 * max(NT, 1)))
    rtol = cfg.get("rtol", 5e-2)
    max_ulp = 4
    scale = cfg.get("scale", None)

    q, k, v, g_gamma, h0 = _make_inputs(cfg, dtype=jnp.bfloat16)

    o_ref, ht_ref = _run_cpu_chunk_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale)
    o_pl, ht_pl = _run_pallas_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale)

    assert compare_tensor("output", o_ref, o_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if ht_ref is not None and ht_pl is not None:
        # Final state accumulates rounding errors over all T timesteps
        # (not just one chunk), so use a larger tolerance.
        ht_atol = max(atol, 5e-2)
        assert compare_tensor(
            "final_state", ht_ref, ht_pl, atol=ht_atol, rtol=rtol, max_ulp=max_ulp
        )


# ============================================================================
# Backward: Pallas chunk_bwd vs CPU chunk_bwd — g_gamma only
# ============================================================================

BWD_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7),
    dict(B=2, T=64, H=4, K=128, V=128, seed=13, h0=True),
    dict(B=2, T=64, H=1, K=128, V=128, seed=10),
    dict(B=2, T=64, H=4, K=128, V=128, seed=20),
    dict(B=2, T=128, H=4, K=128, V=128, seed=400),
    dict(B=1, T=64, H=2, K=128, V=128, seed=41),
    dict(B=1, T=256, H=2, K=128, V=128, seed=300),
    dict(B=2, T=64, H=4, K=128, V=128, seed=99),
    # ── with dht (terminal state gradient) ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=600, dht=True),
    dict(B=1, T=128, H=2, K=128, V=128, seed=601, dht=True),
    dict(B=2, T=64, H=4, K=128, V=128, seed=602, h0=True, dht=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=603, h0=True, dht=True),
    # ── longer ──
    dict(B=1, T=512, H=2, K=128, V=128, seed=700),
    dict(B=1, T=512, H=2, K=128, V=128, seed=701, h0=True, dht=True),
]


def _bwd_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


def _run_cpu_bwd(q, k, v, do, *, g_gamma, h0=None, dht=None, scale=None,
                 chunk_size=CHUNK_SIZE):
    """Run tops.cpu chunk_simple_gla_bwd (CPU reference backward)."""
    dq, dk, dv, _dg, dh0 = cpu_chunk_simple_gla_bwd(
        q,
        k,
        v,
        g=None,
        g_gamma=g_gamma,
        initial_state=h0,
        do=do,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, dh0


@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_bwd_case_id(c) for c in BWD_CASES])
def test_chunk_bwd_vs_cpu(cfg):
    """chunk_simple_gla_bwd (Pallas) should match cpu chunk_simple_gla_bwd reference."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = cfg.get("chunk_size", CHUNK_SIZE)

    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 6)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    do = jax.random.normal(keys[3], (B, T, H, V), dtype=jnp.bfloat16)
    # negative log-decay per head
    g_gamma = -jnp.abs(jax.random.normal(keys[4], (H,), dtype=jnp.float32)) * 0.5

    N = B
    h0 = jax.random.normal(keys[5], (N, H, K, V), dtype=jnp.bfloat16) if cfg.get("h0") else None
    dht = jax.random.normal(jax.random.PRNGKey(cfg["seed"] + 1000),
                            (N, H, K, V), dtype=jnp.bfloat16) if cfg.get("dht") else None

    # CPU reference backward
    dq_ref, dk_ref, dv_ref, dh0_ref = _run_cpu_bwd(
        q, k, v, do, g_gamma=g_gamma, h0=h0, dht=dht, scale=scale, chunk_size=C,
    )

    # Pallas chunk backward
    dq_pl, dk_pl, dv_pl, dh0_pl = chunk_simple_gla_bwd(
        q, k, v, do,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        dht=dht,
        chunk_size=C,
    )

    NT = T // C
    # bf16 accumulation order differs; backward errors compound across chunks.
    # Backward kernels have more intermediate bf16 casts (A, dh, etc.),
    # so use a moderately loose tolerance.
    atol = cfg.get("atol", min(5e-1, 1e-1 * max(NT, 1)))
    rtol = cfg.get("rtol", 5e-2)
    max_ulp = cfg.get("max_ulp", 60)

    assert compare_tensor("dq", dq_ref, dq_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dk", dk_ref, dk_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dv", dv_ref, dv_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if dh0_ref is not None and dh0_pl is not None:
        assert compare_tensor("dh0", dh0_ref, dh0_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
