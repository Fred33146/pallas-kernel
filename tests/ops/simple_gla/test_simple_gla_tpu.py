"""simple_gla TPU tests: implementations + unified API cross-validation.

Implementations:
  - Naive: simple_gla_naive (tops.ops) vs naive_simple_gla (tops.cpu)
  - Fused recurrent: fused_recurrent_simple_gla (tops.ops) vs CPU reference

Unified API cross-validation:
  - chunk vs fused_chunk (g_gamma, K/V % 128)
  - naive vs fused_recurrent (g, g_gamma, both)
  - Gradient NaN tests
  - Error handling for invalid parameters
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.naive import simple_gla_naive
from tops.cpu.ops.simple_gla import naive_simple_gla
from tops.ops.simple_gla import simple_gla, fused_recurrent_simple_gla
from tops.ops.simple_gla.chunk import chunk_simple_gla
from tops.cpu.ops.simple_gla import fused_recurrent_simple_gla as cpu_fused_recurrent
from tests.utils import compare_tensor


# ============================================================================
# Shared helper
# ============================================================================


def _make_inputs(cfg, *, dtype=None, include_cu_seqlens=False):
    """Generate q, k, v, g, g_gamma, h0, and optionally cu_seqlens.

    Args:
        cfg: dict with B, T, H, K, V, seed, gate, h0, cu_seqlens, dtype keys.
        dtype: override dtype (default: cfg['dtype'] or float32).
        include_cu_seqlens: if True, also return cu_seqlens from cfg.

    Returns:
        (q, k, v, g, g_gamma, h0) or (q, k, v, g, g_gamma, h0, cu_seqlens).
    """
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    if dtype is None:
        dtype_str = cfg.get("dtype", "float32")
        dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32

    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)

    gate = cfg.get("gate", "none")
    g = None
    g_gamma = None
    if gate in ("g", "both"):
        g = jax.nn.log_sigmoid(
            jax.random.normal(keys[3], (B, T, H), dtype=jnp.float32)
        )
    if gate in ("g_gamma", "both"):
        g_gamma = -jnp.abs(
            jax.random.normal(keys[3], (H,), dtype=jnp.float32)
        ) * 0.5

    cu_seqlens = cfg.get("cu_seqlens")
    if cu_seqlens is not None:
        cu_seqlens = np.array(cu_seqlens, dtype=np.int32)

    N = B
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1

    h0 = None
    if cfg.get("h0"):
        h0 = jax.random.normal(keys[4], (N, H, K, V), dtype=dtype)

    if include_cu_seqlens:
        return q, k, v, g, g_gamma, h0, cu_seqlens
    return q, k, v, g, g_gamma, h0


# ============================================================================
# Naive: simple_gla_naive (tops.ops) vs naive_simple_gla (tops.cpu)
# ============================================================================

NAIVE_GATE_CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42, gate="none"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=43, gate="g"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=44, gate="g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=45, gate="both"),
]

NAIVE_H0_CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=50, gate="g", h0=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=53, gate="g_gamma", output_final_state=False),
]

NAIVE_SHAPE_CASES = [
    dict(B=1, T=1, H=2, K=32, V=64, seed=70, gate="g_gamma"),
    dict(B=2, T=3, H=4, K=16, V=32, seed=71, gate="g"),
    dict(B=1, T=16, H=2, K=64, V=16, seed=76, gate="both"),
]

NAIVE_DTYPE_CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=90, gate="g_gamma", dtype="bfloat16"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=91, gate="g_gamma", dtype="float32"),
]


def _naive_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "none")
    if gate != "none":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("output_final_state") is not None:
        parts.append(f"ofs={c['output_final_state']}")
    if c.get("dtype"):
        parts.append(c["dtype"])
    return "-".join(parts)


def _run_naive(cfg, atol=1e-5, rtol=1e-5):
    q, k, v, g, g_gamma, h0 = _make_inputs(cfg)
    output_final_state = cfg.get("output_final_state", True)

    o_ref, ht_ref = naive_simple_gla(
        q, k, v, g=g, g_gamma=g_gamma,
        initial_state=h0, output_final_state=output_final_state,
    )
    o_ops, ht_ops = simple_gla_naive(
        q, k, v, g=g, g_gamma=g_gamma,
        initial_state=h0, output_final_state=output_final_state,
    )

    assert compare_tensor("output", o_ref, o_ops, atol=atol, rtol=rtol)
    if output_final_state:
        assert ht_ref is not None and ht_ops is not None
        assert compare_tensor("final_state", ht_ref, ht_ops, atol=atol, rtol=rtol)
    else:
        assert ht_ops is None


@pytest.mark.parametrize("cfg", NAIVE_GATE_CASES,
                         ids=[_naive_id(c) for c in NAIVE_GATE_CASES])
def test_naive_gate_combinations(cfg):
    """Test all 4 subsets of {g, g_gamma}."""
    _run_naive(cfg)


@pytest.mark.parametrize("cfg", NAIVE_H0_CASES,
                         ids=[_naive_id(c) for c in NAIVE_H0_CASES])
def test_naive_initial_state(cfg):
    """With/without initial state, output_final_state variations."""
    _run_naive(cfg)


@pytest.mark.parametrize("cfg", NAIVE_SHAPE_CASES,
                         ids=[_naive_id(c) for c in NAIVE_SHAPE_CASES])
def test_naive_shape_edge_cases(cfg):
    """T=1, odd T, K!=V."""
    _run_naive(cfg)


@pytest.mark.parametrize("cfg", NAIVE_DTYPE_CASES,
                         ids=[_naive_id(c) for c in NAIVE_DTYPE_CASES])
def test_naive_dtype(cfg):
    """bf16 and float32 input precision."""
    dtype_str = cfg.get("dtype", "float32")
    atol = 1e-4 if dtype_str == "bfloat16" else 1e-5
    rtol = 1e-4 if dtype_str == "bfloat16" else 1e-5
    _run_naive(cfg, atol=atol, rtol=rtol)


@pytest.mark.parametrize("g_gamma_val", [-0.5, -0.8, -1.0])
def test_naive_nan_stability(g_gamma_val):
    """Large |g_gamma| should not produce NaN/Inf."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g_gamma = jnp.full((H,), g_gamma_val, dtype=jnp.float32)

    o, ht = simple_gla_naive(
        q, k, v, g_gamma=g_gamma, output_final_state=True,
    )

    assert not jnp.any(jnp.isnan(o)), f"output NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(o)), f"output Inf (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isnan(ht)), f"state NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(ht)), f"state Inf (g_gamma={g_gamma_val})"


# ============================================================================
# Fused recurrent: fused_recurrent_simple_gla (tops.ops) vs CPU reference
# ============================================================================

RECURRENT_GATE_CASES = [
    dict(B=2, T=16, H=4, K=16, V=8, seed=10, gate="g"),
    dict(B=2, T=16, H=4, K=16, V=8, seed=11, gate="g_gamma"),
    dict(B=2, T=16, H=4, K=16, V=8, seed=12, gate="both"),
]

RECURRENT_LARGE_CASES = [
    dict(B=2, T=128, H=4, K=128, V=128, seed=30, gate="g_gamma"),
    dict(B=1, T=256, H=2, K=128, V=128, seed=32, gate="g_gamma", h0=True),
]

RECURRENT_REVERSE_CASES = [
    dict(B=2, T=16, H=4, K=16, V=8, seed=40, gate="g", reverse=True),
    dict(B=2, T=16, H=4, K=16, V=8, seed=42, gate="both", reverse=True, h0=True),
]

RECURRENT_VARLEN_CASES = [
    dict(B=1, T=7, H=2, K=8, V=4, seed=50, gate="g_gamma",
         cu_seqlens=[0, 3, 7]),
    dict(B=1, T=10, H=2, K=16, V=8, seed=51, gate="g",
         cu_seqlens=[0, 4, 7, 10]),
]

RECURRENT_STATE_CASES = [
    dict(B=2, T=16, H=4, K=16, V=8, seed=60, gate="g", h0=True,
         output_final_state=True),
    dict(B=2, T=16, H=4, K=16, V=8, seed=62, gate="g",
         output_final_state=False),
]


def _recurrent_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    parts.append(f"gate={c.get('gate', 'g')}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("reverse"):
        parts.append("rev")
    if c.get("cu_seqlens") is not None:
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("output_final_state") is not None:
        parts.append(f"ofs={c['output_final_state']}")
    return "-".join(parts)


def _run_recurrent(cfg):
    q, k, v, g, g_gamma, h0, cu_seqlens = _make_inputs(cfg, include_cu_seqlens=True)
    reverse = cfg.get("reverse", False)
    output_final_state = cfg.get("output_final_state", True)

    o_ref, ht_ref = cpu_fused_recurrent(
        q, k, v, g=g, g_gamma=g_gamma,
        initial_state=h0, output_final_state=output_final_state,
        reverse=reverse, cu_seqlens=cu_seqlens,
    )
    o_ops, ht_ops = fused_recurrent_simple_gla(
        q, k, v, g=g, g_gamma=g_gamma,
        initial_state=h0, output_final_state=output_final_state,
        reverse=reverse, cu_seqlens=cu_seqlens,
    )

    assert compare_tensor("output", o_ref, o_ops, atol=1e-5, rtol=1e-5)
    if output_final_state:
        assert ht_ref is not None and ht_ops is not None
        assert compare_tensor("final_state", ht_ref, ht_ops, atol=1e-5, rtol=1e-5)
    else:
        assert ht_ops is None


@pytest.mark.parametrize("cfg", RECURRENT_GATE_CASES,
                         ids=[_recurrent_id(c) for c in RECURRENT_GATE_CASES])
def test_recurrent_gate_combinations(cfg):
    """g-only, g_gamma-only, both."""
    _run_recurrent(cfg)


@pytest.mark.parametrize("cfg", RECURRENT_LARGE_CASES,
                         ids=[_recurrent_id(c) for c in RECURRENT_LARGE_CASES])
def test_recurrent_large_shapes(cfg):
    """Larger shapes for TPU workloads."""
    _run_recurrent(cfg)


@pytest.mark.parametrize("cfg", RECURRENT_REVERSE_CASES,
                         ids=[_recurrent_id(c) for c in RECURRENT_REVERSE_CASES])
def test_recurrent_reverse(cfg):
    """reverse=True mode."""
    _run_recurrent(cfg)


@pytest.mark.parametrize("cfg", RECURRENT_VARLEN_CASES,
                         ids=[_recurrent_id(c) for c in RECURRENT_VARLEN_CASES])
def test_recurrent_varlen(cfg):
    """cu_seqlens variable-length sequences."""
    _run_recurrent(cfg)


@pytest.mark.parametrize("cfg", RECURRENT_STATE_CASES,
                         ids=[_recurrent_id(c) for c in RECURRENT_STATE_CASES])
def test_recurrent_state(cfg):
    """Initial state and output_final_state variations."""
    _run_recurrent(cfg)


@pytest.mark.parametrize("g_gamma_val", [-0.5, -0.8, -1.0])
def test_recurrent_nan_stability(g_gamma_val):
    """Large |g_gamma| should not produce NaN/Inf."""
    B, T, H, K, V = 2, 32, 4, 16, 8
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g_gamma = jnp.full((H,), g_gamma_val, dtype=jnp.float32)

    o, ht = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, output_final_state=True,
    )

    assert not jnp.any(jnp.isnan(o)), f"output NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(o)), f"output Inf (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isnan(ht)), f"state NaN (g_gamma={g_gamma_val})"
    assert not jnp.any(jnp.isinf(ht)), f"state Inf (g_gamma={g_gamma_val})"


def test_recurrent_jit_consistency():
    """jax.jit should produce same results as eager execution."""
    B, T, H, K, V = 2, 16, 4, 16, 8
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g_gamma = -jnp.abs(jax.random.normal(keys[3], (H,), dtype=jnp.float32)) * 0.5

    o_eager, ht_eager = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, output_final_state=True,
    )

    jit_fn = jax.jit(
        lambda q_, k_, v_, gg: fused_recurrent_simple_gla(
            q_, k_, v_, g_gamma=gg, output_final_state=True,
        )
    )
    o_jit, ht_jit = jit_fn(q, k, v, g_gamma)

    assert compare_tensor("o_jit", o_eager, o_jit, atol=1e-6, rtol=1e-6)
    assert compare_tensor("ht_jit", ht_eager, ht_jit, atol=1e-6, rtol=1e-6)


def test_recurrent_varlen_jit_consistency():
    """jax.jit with cu_seqlens should produce same results as eager."""
    B, T, H, K, V = 1, 7, 2, 8, 4
    key = jax.random.PRNGKey(17)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g_gamma = jnp.array([-0.25, -0.55], dtype=jnp.float32)
    cu_seqlens = jnp.array([0, 2, 5, 7], dtype=jnp.int32)
    h0 = jax.random.normal(keys[3], (3, H, K, V), dtype=jnp.float32)

    eager_o, eager_ht = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )
    jit_fn = jax.jit(
        lambda q_, k_, v_, gg, h0_, cu: fused_recurrent_simple_gla(
            q_, k_, v_, g_gamma=gg, initial_state=h0_,
            output_final_state=True, cu_seqlens=cu,
        )
    )
    jit_o, jit_ht = jit_fn(q, k, v, g_gamma, h0, cu_seqlens)

    assert compare_tensor("o_varlen_jit", eager_o, jit_o, atol=1e-5, rtol=1e-5)
    assert compare_tensor("ht_varlen_jit", eager_ht, jit_ht, atol=1e-5, rtol=1e-5)


def test_recurrent_decode_avoids_chunk_padding_decay():
    """T=1 decode: fused_recurrent ht should differ from chunk ht.

    chunk pads T=1 to chunk_size, applying extra decay to the hidden state.
    fused_recurrent processes exactly T=1 with no padding artifact.
    """
    B, T, H, K, V = 1, 1, 2, 128, 128
    key = jax.random.PRNGKey(23)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    h0 = jax.random.normal(keys[3], (B, H, K, V), dtype=jnp.float32)
    g_gamma = jnp.array([-0.1, -0.3], dtype=jnp.float32)

    o_recurrent, ht_recurrent = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0, output_final_state=True,
    )
    _o_chunk, ht_chunk = chunk_simple_gla(
        q, k, v, g_gamma, initial_state=h0, output_final_state=True,
    )

    o_ref, _ = cpu_fused_recurrent(
        q, k, v, g_gamma=g_gamma, initial_state=h0, output_final_state=True,
    )
    assert compare_tensor("decode_o", o_ref, o_recurrent, atol=1e-5, rtol=1e-5)

    ht_r = np.array(ht_recurrent)
    ht_c = np.array(ht_chunk)
    assert not np.allclose(ht_r, ht_c, atol=1e-4, rtol=1e-4), (
        "chunk and fused_recurrent should differ for T=1 decode"
    )


# ============================================================================
# Unified API: chunk vs fused_chunk cross-validation
# ============================================================================

CHUNK_SIZE = 64

CHUNK_VS_FUSED_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=43),
    dict(B=2, T=64, H=4, K=128, V=128, seed=44, h0=True),
]


def _cross_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("chunk_size") is not None:
        parts.append(f"C={c['chunk_size']}")
    return "-".join(parts)


def _make_chunk_inputs(cfg, dtype=jnp.bfloat16):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    g_gamma = -jnp.abs(jax.random.normal(keys[3], (H,), dtype=jnp.float32)) * 0.5

    h0 = None
    if cfg.get("h0"):
        h0 = jax.random.normal(
            jax.random.PRNGKey(cfg["seed"] + 100), (B, H, K, V), dtype=dtype
        )

    return q, k, v, g_gamma, h0


@pytest.mark.parametrize("cfg", CHUNK_VS_FUSED_CASES,
                         ids=[_cross_id(c) for c in CHUNK_VS_FUSED_CASES])
def test_chunk_vs_fused_chunk_fwd(cfg):
    """simple_gla(mode='chunk') and simple_gla(mode='fused_chunk') should match."""
    q, k, v, g_gamma, h0 = _make_chunk_inputs(cfg)
    C = cfg.get("chunk_size", CHUNK_SIZE)
    T = cfg["T"]
    NT = T // C

    o_chunk, ht_chunk = simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, chunk_size=C, mode="chunk",
    )
    o_fused, ht_fused = simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, chunk_size=C, mode="fused_chunk",
    )

    atol = min(5e-2, 1e-2 + 1e-2 * max(NT, 1))
    rtol = 5e-2

    assert compare_tensor("o_cross", o_chunk, o_fused, atol=atol, rtol=rtol)
    if ht_chunk is not None and ht_fused is not None:
        assert compare_tensor("ht_cross", ht_chunk, ht_fused,
                              atol=max(atol, 5e-2), rtol=rtol)


@pytest.mark.parametrize("cfg", CHUNK_VS_FUSED_CASES[:2],
                         ids=[_cross_id(c) for c in CHUNK_VS_FUSED_CASES[:2]])
def test_chunk_vs_fused_chunk_grads(cfg):
    """Gradients from chunk and fused_chunk should match."""
    q, k, v, g_gamma, h0 = _make_chunk_inputs(cfg)
    C = cfg.get("chunk_size", CHUNK_SIZE)
    T = cfg["T"]
    NT = T // C

    def loss_chunk(q_, k_, v_, gg):
        o, _ = simple_gla(q_, k_, v_, g_gamma=gg, initial_state=h0,
                          mode="chunk", chunk_size=C)
        return o.sum()

    def loss_fused(q_, k_, v_, gg):
        o, _ = simple_gla(q_, k_, v_, g_gamma=gg, initial_state=h0,
                          mode="fused_chunk", chunk_size=C)
        return o.sum()

    grads_chunk = jax.grad(loss_chunk, argnums=(0, 1, 2))(q, k, v, g_gamma)
    grads_fused = jax.grad(loss_fused, argnums=(0, 1, 2))(q, k, v, g_gamma)

    atol = min(5e-1, 1e-1 * max(NT, 1))
    rtol = 5e-2

    for name, g_c, g_f in zip(["dq", "dk", "dv"], grads_chunk, grads_fused):
        assert compare_tensor(f"{name}_cross", g_c, g_f, atol=atol, rtol=rtol)


# ============================================================================
# Unified API: naive vs fused_recurrent cross-validation
# ============================================================================

NAIVE_VS_RECURRENT_CASES = [
    dict(B=2, T=16, H=4, K=16, V=8, seed=200, gate="g"),
    dict(B=2, T=16, H=4, K=16, V=8, seed=201, gate="g_gamma"),
    dict(B=2, T=16, H=4, K=16, V=8, seed=202, gate="both"),
]


def _naive_recurrent_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    parts.append(f"gate={c['gate']}")
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


@pytest.mark.parametrize("cfg", NAIVE_VS_RECURRENT_CASES,
                         ids=[_naive_recurrent_id(c) for c in NAIVE_VS_RECURRENT_CASES])
def test_naive_vs_fused_recurrent(cfg):
    """simple_gla(mode='naive') and simple_gla(mode='fused_recurrent') should match."""
    q, k, v, g, g_gamma, h0 = _make_inputs(cfg)

    o_naive, ht_naive = simple_gla(
        q, k, v, g=g, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, mode="naive",
    )
    o_recurrent, ht_recurrent = simple_gla(
        q, k, v, g=g, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, mode="fused_recurrent",
    )

    assert compare_tensor("o_cross", o_naive, o_recurrent, atol=1e-5, rtol=1e-5)
    if ht_naive is not None and ht_recurrent is not None:
        assert compare_tensor("ht_cross", ht_naive, ht_recurrent,
                              atol=1e-5, rtol=1e-5)


# ============================================================================
# Unified API: gradient NaN tests
# ============================================================================

GRAD_NAN_CASES = [
    dict(mode="chunk", g_gamma_val=-0.5, chunk_size=64),
    dict(mode="chunk", g_gamma_val=-0.8, chunk_size=128),
    dict(mode="fused_chunk", g_gamma_val=-0.5, chunk_size=64),
    dict(mode="fused_chunk", g_gamma_val=-0.8, chunk_size=128),
]


def _grad_id(c):
    return f"{c['mode']}-gamma={c['g_gamma_val']}-C={c['chunk_size']}"


@pytest.mark.parametrize("cfg", GRAD_NAN_CASES,
                         ids=[_grad_id(c) for c in GRAD_NAN_CASES])
def test_gradient_no_nan(cfg):
    """jax.grad through simple_gla should not produce NaN."""
    B, T, H, K, V = 1, 256, 2, 128, 128
    mode = cfg["mode"]
    C = cfg["chunk_size"]

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = jnp.full((H,), cfg["g_gamma_val"], dtype=jnp.float32)

    def loss_fn(q_, k_, v_, gg):
        o, _ = simple_gla(q_, k_, v_, g_gamma=gg, chunk_size=C, mode=mode)
        return o.sum()

    grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v, g_gamma)

    for name, arr in zip(["dq", "dk", "dv"], grads):
        assert not jnp.any(jnp.isnan(arr)), (
            f"{name} NaN (mode={mode}, g_gamma={cfg['g_gamma_val']}, C={C})"
        )
        assert not jnp.any(jnp.isinf(arr)), (
            f"{name} Inf (mode={mode}, g_gamma={cfg['g_gamma_val']}, C={C})"
        )


# ============================================================================
# Unified API: error handling
# ============================================================================

_ERR_MODES_REJECT_G = ["chunk", "fused_chunk"]
_ERR_MODES_REJECT_REVERSE = ["chunk", "fused_chunk"]


@pytest.mark.parametrize("mode", _ERR_MODES_REJECT_G)
def test_mode_rejects_g(mode):
    """chunk/fused_chunk should reject per-token gate g."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    g = jax.random.normal(key, (1, 64, 2))
    g_gamma = jnp.zeros((2,))
    with pytest.raises(ValueError, match="does not support per-token gate"):
        simple_gla(q, k, v, g=g, g_gamma=g_gamma, mode=mode)


@pytest.mark.parametrize("mode", _ERR_MODES_REJECT_REVERSE)
def test_mode_rejects_reverse(mode):
    """chunk/fused_chunk should reject reverse=True."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    g_gamma = jnp.zeros((2,))
    with pytest.raises(ValueError, match="does not support.*reverse"):
        simple_gla(q, k, v, g_gamma=g_gamma, reverse=True, mode=mode)


def test_chunk_rejects_cu_seqlens():
    """mode='chunk' should reject cu_seqlens."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    g_gamma = jnp.zeros((2,))
    cu = jnp.array([0, 32, 64], dtype=jnp.int32)
    with pytest.raises(ValueError, match="does not support.*cu_seqlens"):
        simple_gla(q, k, v, g_gamma=g_gamma, cu_seqlens_cpu=cu, mode="chunk")


def test_unknown_mode_raises():
    """Unknown mode should raise ValueError."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    g_gamma = jnp.zeros((2,))
    with pytest.raises(ValueError, match="Unknown mode"):
        simple_gla(q, k, v, g_gamma=g_gamma, mode="invalid")


def test_chunk_requires_g_gamma():
    """mode='chunk' should require g_gamma."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 64, 2, 128))
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    with pytest.raises(AssertionError):
        simple_gla(q, k, v, mode="chunk")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
