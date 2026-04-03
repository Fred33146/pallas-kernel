"""Chunk Pallas kernel tests: forward (fused_chunk_fwd) + backward (chunk_bwd_o).

Forward: fused_chunk_fwd (Pallas, fused h+o)
     vs chunk_fwd_h_ref + chunk_fwd_o_ref (two-step reference)

Backward: chunk_simple_gla_bwd_o_pl (Pallas, fused dq/dk/dv)
      vs chunk_bwd_dqkwg + chunk_bwd_dv (CPU reference)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from tops.ops.common.fused_chunk import fused_chunk_fwd
from tops.ops.common.chunk_h import chunk_fwd_h_ref
from tops.ops.common.chunk_o import chunk_fwd_o_ref
from tops.ops.common.chunk_o import chunk_simple_gla_bwd_o_pl
from tops.cpu.ops.common import chunk_fwd_h as cpu_chunk_fwd_h
from tops.cpu.ops.common import chunk_bwd_dh as cpu_chunk_bwd_dh
from tops.cpu.ops.simple_gla import chunk_bwd_dqkwg, chunk_bwd_dv
from tests.utils import compare_tensor

# ============================================================================
# Forward configs
#
# fused_chunk_fwd constraints: K%128==0, V%128==0, T%chunk_size==0
# Supports: g (chunk-local cumsum), g_gamma, both, neither
# ============================================================================

FWD_CHUNK_SIZE = 64

FWD_GATE_CASES = [
    dict(B=1, T=128, H=2, K=128, V=128, seed=10, gates=[]),
    dict(B=1, T=128, H=2, K=128, V=128, seed=11, gates=["g"]),
    dict(B=1, T=128, H=2, K=128, V=128, seed=12, gates=["g_gamma"]),
    dict(B=1, T=128, H=2, K=128, V=128, seed=13, gates=["g", "g_gamma"]),
]

FWD_SHAPE_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=20, gates=["g_gamma"]),
    dict(B=1, T=256, H=2, K=128, V=128, seed=21, gates=["g_gamma"]),
    dict(B=1, T=128, H=2, K=256, V=128, seed=22, gates=["g_gamma"]),
    dict(B=1, T=128, H=2, K=128, V=256, seed=23, gates=["g_gamma"]),
]

FWD_H0_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=30, gates=["g_gamma"], h0=True),
    dict(B=1, T=128, H=2, K=128, V=128, seed=31, gates=["g_gamma"], h0=True),
]

FWD_STATE_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=40, gates=["g_gamma"], use_ht=True),
    dict(B=2, T=64, H=4, K=128, V=128, seed=41, gates=["g_gamma"], use_ht=False),
]

# ============================================================================
# Backward configs
#
# chunk_simple_gla_bwd_o_pl supports g_gamma ONLY
# Constraints: T%chunk_size==0
# ============================================================================

BWD_CHUNK_SIZE = 64

BWD_STANDARD_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=43),
    dict(B=2, T=128, H=4, K=128, V=128, seed=45),
]

BWD_K_NEQ_V_CASES = [
    dict(B=1, T=64, H=2, K=128, V=256, seed=50),
    dict(B=1, T=64, H=2, K=256, V=128, seed=51),
]


# ============================================================================
# ID helpers
# ============================================================================


def _fwd_id(c):
    V = c.get("V", c["K"])
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{V}"]
    gates = c.get("gates", [])
    parts.append("gates=" + "+".join(gates) if gates else "nogates")
    if c.get("h0"):
        parts.append("h0")
    if c.get("use_ht") is not None:
        parts.append(f"ht={c['use_ht']}")
    return "-".join(parts)


def _bwd_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("chunk_size") is not None:
        parts.append(f"C={c['chunk_size']}")
    return "-".join(parts)


# ============================================================================
# Forward helpers
# ============================================================================


def _make_fwd_inputs(cfg, dtype=jnp.bfloat16):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    V = cfg.get("V", K)
    gates = cfg.get("gates", [])
    chunk_size = cfg.get("chunk_size", FWD_CHUNK_SIZE)

    key = jax.random.PRNGKey(cfg.get("seed", 42))
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)

    g = (jax.nn.log_sigmoid(
        jax.random.normal(keys[3], (B, T, H), dtype=jnp.float32)
    ) if "g" in gates else None)
    g_gamma = (
        -(8 / H * (1 - 1 / 8)) * jnp.arange(H, dtype=jnp.float32)
        if "g_gamma" in gates else None
    )

    h0 = (jax.random.normal(keys[4], (B, H, K, V), dtype=dtype)
           if cfg.get("h0") else None)

    return dict(q=q, k=k, v=v, g=g, g_gamma=g_gamma, h0=h0,
                chunk_size=chunk_size)


def _run_fwd_and_compare(cfg, atol=None, rtol=5e-2, max_ulp=50):
    inputs = _make_fwd_inputs(cfg)
    K = cfg["K"]
    scale = K ** -0.5
    use_ht = cfg.get("use_ht", True)
    T = cfg["T"]
    C = cfg.get("chunk_size", FWD_CHUNK_SIZE)
    NT = T // C

    if atol is None:
        atol = min(2e-1, 5e-2 + 3e-2 * max(NT, 1))

    # Two-step reference
    h, ht_ref = chunk_fwd_h_ref(
        inputs["k"], inputs["v"],
        g=inputs["g"], g_gamma=inputs["g_gamma"], h0=inputs["h0"],
        output_final_state=use_ht, chunk_size=C,
    )
    o_ref = chunk_fwd_o_ref(
        inputs["q"], inputs["k"], inputs["v"], h,
        g=inputs["g"], g_gamma=inputs["g_gamma"],
        scale=scale, chunk_size=C,
    )

    # Fused kernel
    o_fused, ht_fused = fused_chunk_fwd(
        inputs["q"], inputs["k"], inputs["v"],
        g=inputs["g"], g_gamma=inputs["g_gamma"], h0=inputs["h0"],
        scale=scale, use_ht=use_ht, chunk_size=C,
    )

    assert compare_tensor("output", o_ref, o_fused, atol=atol, rtol=rtol,
                          max_ulp=max_ulp)
    if use_ht:
        if ht_ref is not None and ht_fused is not None:
            assert compare_tensor("final_state", ht_ref, ht_fused,
                                  atol=max(atol, 5e-2), rtol=rtol,
                                  max_ulp=max_ulp)
    else:
        assert ht_fused is None


# ============================================================================
# Backward helpers
# ============================================================================


def _make_bwd_inputs(cfg, dtype=jnp.bfloat16):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    C = cfg.get("chunk_size", BWD_CHUNK_SIZE)
    scale = K ** -0.5

    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    do = jax.random.normal(keys[3], (B, T, H, V), dtype=dtype)
    g_gamma = -jnp.abs(jax.random.normal(keys[4], (H,), dtype=jnp.float32)) * 0.5

    h, _ = cpu_chunk_fwd_h(
        k, v, g_gamma=g_gamma, output_final_state=False,
        states_in_fp32=True, chunk_size=C,
    )
    dh, _ = cpu_chunk_bwd_dh(
        q, do, g_gamma=g_gamma, scale=scale, chunk_size=C,
    )

    return q, k, v, do, g_gamma, h, dh, scale, C


def _run_bwd_and_compare(cfg):
    q, k, v, do, g_gamma, h, dh, scale, C = _make_bwd_inputs(cfg)
    T = cfg["T"]
    NT = T // C

    # CPU reference
    dq_ref, dk_ref, _dg = chunk_bwd_dqkwg(
        q, k, v, h, dh, do, g_gamma=g_gamma, scale=scale, chunk_size=C,
    )
    dv_ref = chunk_bwd_dv(
        q, k, do, dh, g_gamma=g_gamma, scale=scale, chunk_size=C,
    )

    # Pallas kernel
    dq_pl, dk_pl, dv_pl = chunk_simple_gla_bwd_o_pl(
        q, k, v, g_gamma, h, do, dh, scale=scale, chunk_size=C,
    )

    atol = min(5e-1, 1e-1 * max(NT, 1))
    rtol = 5e-2
    max_ulp = 60

    assert compare_tensor("dq", dq_ref, dq_pl, atol=atol, rtol=rtol,
                          max_ulp=max_ulp)
    assert compare_tensor("dk", dk_ref, dk_pl, atol=atol, rtol=rtol,
                          max_ulp=max_ulp)
    assert compare_tensor("dv", dv_ref, dv_pl, atol=atol, rtol=rtol,
                          max_ulp=max_ulp)


# ============================================================================
# Forward: gate combinations
# ============================================================================


@pytest.mark.parametrize("cfg", FWD_GATE_CASES,
                         ids=[_fwd_id(c) for c in FWD_GATE_CASES])
def test_fwd_gate_combinations(cfg):
    """Test all 4 subsets of {g, g_gamma}."""
    _run_fwd_and_compare(cfg)


# ============================================================================
# Forward: shape variations
# ============================================================================


@pytest.mark.parametrize("cfg", FWD_SHAPE_CASES,
                         ids=[_fwd_id(c) for c in FWD_SHAPE_CASES])
def test_fwd_shapes(cfg):
    """Various B, T, H, K, V combinations."""
    _run_fwd_and_compare(cfg)


# ============================================================================
# Forward: initial state
# ============================================================================


@pytest.mark.parametrize("cfg", FWD_H0_CASES,
                         ids=[_fwd_id(c) for c in FWD_H0_CASES])
def test_fwd_initial_state(cfg):
    """With initial state h0."""
    _run_fwd_and_compare(cfg)


# ============================================================================
# Forward: final state output
# ============================================================================


@pytest.mark.parametrize("cfg", FWD_STATE_CASES,
                         ids=[_fwd_id(c) for c in FWD_STATE_CASES])
def test_fwd_final_state(cfg):
    """use_ht True/False."""
    _run_fwd_and_compare(cfg)


# ============================================================================
# Forward: assertion validation (negative tests)
# ============================================================================


def test_fwd_k_not_multiple_128_raises():
    """K not a multiple of 128 should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 64))
    k = jax.random.normal(key, (1, 64, 2, 64))
    v = jax.random.normal(key, (1, 64, 2, 128))
    with pytest.raises(AssertionError, match="K.*128"):
        fused_chunk_fwd(q, k, v, chunk_size=64)


def test_fwd_v_not_multiple_128_raises():
    """V not a multiple of 128 should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 64))
    with pytest.raises(AssertionError, match="V.*128"):
        fused_chunk_fwd(q, k, v, chunk_size=64)


def test_fwd_t_not_divisible_by_chunk_size_raises():
    """T not divisible by chunk_size should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 100, 2, 128))
    k = jax.random.normal(key, (1, 100, 2, 128))
    v = jax.random.normal(key, (1, 100, 2, 128))
    with pytest.raises(AssertionError, match="divisible by chunk_size"):
        fused_chunk_fwd(q, k, v, chunk_size=64)


# ============================================================================
# Forward: NaN stability
# ============================================================================


@pytest.mark.parametrize("g_gamma_val", [-0.5, -0.8, -1.0])
def test_fwd_nan_stability(g_gamma_val):
    """Large |g_gamma| should not produce NaN/Inf."""
    B, T, H, K, V = 1, 256, 2, 128, 128
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = jnp.full((H,), g_gamma_val, dtype=jnp.float32)

    o, ht = fused_chunk_fwd(
        q, k, v, g_gamma=g_gamma, use_ht=True, chunk_size=64,
    )

    assert not jnp.any(jnp.isnan(o)), f"output NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(o)), f"output Inf (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isnan(ht)), f"state NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(ht)), f"state Inf (g_gamma={g_gamma_val})"


# ============================================================================
# Backward: standard cases
# ============================================================================


@pytest.mark.parametrize("cfg", BWD_STANDARD_CASES,
                         ids=[_bwd_id(c) for c in BWD_STANDARD_CASES])
def test_bwd_standard(cfg):
    """chunk_simple_gla_bwd_o_pl should match CPU ref backward."""
    _run_bwd_and_compare(cfg)


# ============================================================================
# Backward: K != V
# ============================================================================


@pytest.mark.parametrize("cfg", BWD_K_NEQ_V_CASES,
                         ids=[_bwd_id(c) for c in BWD_K_NEQ_V_CASES])
def test_bwd_k_neq_v(cfg):
    """K != V shapes."""
    _run_bwd_and_compare(cfg)


# ============================================================================
# Backward: NaN stability
# ============================================================================


@pytest.mark.parametrize("g_gamma_val", [-0.5, -0.8, -1.0])
def test_bwd_nan_stability(g_gamma_val):
    """Pallas backward should not produce NaN with large |g_gamma|."""
    B, T, H, K, V = 1, 256, 2, 128, 128
    C = 128
    scale = K ** -0.5

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    do = jax.random.normal(keys[3], (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = jnp.full((H,), g_gamma_val, dtype=jnp.float32)

    h, _ = cpu_chunk_fwd_h(
        k, v, g_gamma=g_gamma, output_final_state=False,
        states_in_fp32=True, chunk_size=C,
    )
    dh, _ = cpu_chunk_bwd_dh(
        q, do, g_gamma=g_gamma, scale=scale, chunk_size=C,
    )

    dq, dk, dv = chunk_simple_gla_bwd_o_pl(
        q, k, v, g_gamma, h, do, dh, scale=scale, chunk_size=C,
    )

    for name, arr in [("dq", dq), ("dk", dk), ("dv", dv)]:
        assert not jnp.any(jnp.isnan(arr)), (
            f"{name} NaN (g_gamma={g_gamma_val}, C={C})"
        )
        assert not jnp.any(jnp.isinf(arr)), (
            f"{name} Inf (g_gamma={g_gamma_val}, C={C})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
