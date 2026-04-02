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
import time

from tops.ops.simple_gla.fused_chunk import fused_chunk_simple_gla_fwd
from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd, chunk_simple_gla_bwd, chunk_simple_gla
from tops.ops.common.chunk_h import chunk_fwd_h_kernel as chunk_fwd_h
from tops.ops.common.chunk_h import chunk_bwd_dh_kernel as chunk_bwd_dh
from tops.ops.common.chunk_o import chunk_simple_gla_bwd_o_pl
from tops.cpu.ops.simple_gla import chunk_simple_gla_fwd as cpu_chunk_simple_gla_fwd
from tops.cpu.ops.simple_gla import chunk_simple_gla_bwd as cpu_chunk_simple_gla_bwd
from tests.utils import compare_tensor

# from tops.ops.gla.chunk_fused_kernels import chunk_fwd_fused_g_gamma

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
    # ── chunk_size=128 ──
    dict(B=1, T=256, H=2, K=128, V=128, seed=800, gate="g_gamma", chunk_size=128),
    dict(B=2, T=256, H=4, K=128, V=128, seed=801, gate="g_gamma", chunk_size=128),
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
    if c.get("chunk_size") is not None:
        parts.append(f"C={c['chunk_size']}")
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
    o, ht = fused_chunk_simple_gla_fwd(
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
def test_fused_chunk_fwd_vs_cpu(cfg):
    """chunk_simple_gla_fwd (Pallas) should match cpu chunk_simple_gla_fwd reference."""
    T = cfg["T"]
    C = cfg.get("chunk_size", CHUNK_SIZE)
    NT = T // C
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

    # jax.profiler.start_trace('fused_simple_gla_opt_prof')
    # time.sleep(1)
    o_pl, ht_pl = _run_pallas_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=C)
    # time.sleep(1)
    # jax.profiler.stop_trace()

    o_ref, ht_ref = _run_cpu_chunk_fwd(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=C)

    assert compare_tensor("output", o_ref, o_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if ht_ref is not None and ht_pl is not None:
        # Final state accumulates rounding errors over all T timesteps
        # (not just one chunk), so use a larger tolerance.
        ht_atol = max(atol, 5e-2)
        assert compare_tensor(
            "final_state", ht_ref, ht_pl, atol=ht_atol, rtol=rtol, max_ulp=max_ulp
        )


if __name__ == "__main__":
    # test_fused_chunk_fwd_vs_cpu(FWD_CASES[17])
    pytest.main([__file__, "-v"])
