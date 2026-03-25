"""chunk_fwd_h: Pallas kernel vs Jax TPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from tops.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_fwd_h_ref
from tests.utils import compare_tensor


PALLAS_CASES = [
    dict(
        B=1,
        T=1024,
        H=4,
        K=128,
        chunk_size=64,
        seed=11,
    ),
    dict(
        B=1,
        T=1024,
        H=4,
        K=256,
        chunk_size=64,
        seed=11,
    ),
    dict(
        B=4,
        T=512,
        H=4,
        K=256,
        chunk_size=64,
        seed=11,
    ),
]


# ---------------------------------------------------------------------------
# Edge-case test configurations
# ---------------------------------------------------------------------------

# 1. Gate combinations — each subset of {g, gk, g_gamma}
GATE_COMBO_CASES = [
    # No gates at all (pure linear attention)
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=[]),
    # gk only (standard GLA)
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["gk"]),
    # g only (scalar gate)
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g"]),
    # g_gamma only (simple GLA)
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g_gamma"]),
    # g + gk
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g", "gk"]),
    # g + g_gamma
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g", "g_gamma"]),
    # g_gamma + gk
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g_gamma", "gk"]),
    # All three gates
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"]),
]

# 2. h0 = None (zero initial state)
H0_NONE_CASES = [
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, use_h0=False, gates=["gk"]),
    dict(B=2, T=256, H=4, K=128, V=128, chunk_size=64, use_h0=False, gates=["g", "gk", "g_gamma"]),
]

# 3. Single chunk (T == chunk_size) — boundary: only one chunk, no inter-chunk propagation
SINGLE_CHUNK_CASES = [
    dict(B=1, T=64, H=2, K=128, V=128, chunk_size=64, gates=["gk"]),
    dict(B=2, T=64, H=4, K=128, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"]),
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=128, gates=["gk"]),
]

# 4. Two chunks (T == 2 * chunk_size) — minimum multi-chunk scenario
TWO_CHUNK_CASES = [
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["gk"]),
    dict(B=1, T=256, H=2, K=256, V=128, chunk_size=128, gates=["g_gamma"]),
]

# 5. V != K — value dimension differs from key dimension
V_NEQ_K_CASES = [
    dict(B=1, T=128, H=2, K=128, V=256, chunk_size=64, gates=["gk"]),
    dict(B=1, T=128, H=2, K=256, V=128, chunk_size=64, gates=["gk"]),
    dict(B=2, T=256, H=4, K=128, V=256, chunk_size=64, gates=["g", "gk", "g_gamma"]),
]

# 6. Single head (H=1)
SINGLE_HEAD_CASES = [
    dict(B=1, T=128, H=1, K=128, V=128, chunk_size=64, gates=["gk"]),
    dict(B=2, T=256, H=1, K=256, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"]),
]

# 7. Large K with multiple tiles (K=384, K=512)
LARGE_K_CASES = [
    dict(B=1, T=128, H=2, K=384, V=128, chunk_size=64, gates=["gk"]),
    dict(B=1, T=128, H=2, K=128, V=384, chunk_size=64, gates=["gk"]),
    dict(B=1, T=128, H=2, K=256, V=256, chunk_size=64, gates=["g", "gk"]),
]

# 8. Different chunk sizes
CHUNK_SIZE_CASES = [
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=32, gates=["gk"]),
    dict(B=1, T=256, H=2, K=128, V=128, chunk_size=128, gates=["gk"]),
    dict(B=1, T=256, H=2, K=128, V=128, chunk_size=256, gates=["gk"]),  # single chunk via large chunk_size
]

# 9. Larger batch sizes
BATCH_CASES = [
    dict(B=8, T=128, H=2, K=128, V=128, chunk_size=64, gates=["gk"]),
    dict(B=4, T=256, H=4, K=256, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"]),
]

# 10. output_final_state = False
NO_FINAL_STATE_CASES = [
    dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["gk"]),
    dict(B=2, T=256, H=4, K=128, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"]),
]


def _case_id(c):
    V = c.get("V", c["K"])
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}"]
    if V != c["K"]:
        parts.append(f"V{V}")
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("chunk_size") is not None:
        parts.append(f"chunk{c['chunk_size']}")
    if c.get("gates") is not None:
        parts.append("gates=" + "+".join(c["gates"]) if c["gates"] else "nogates")
    if c.get("use_h0") is False:
        parts.append("noh0")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _to_jax_cu_seqlens(cu_seqlens) -> jax.Array | None:
    if cu_seqlens is None:
        return None
    return jnp.asarray(cu_seqlens, dtype=jnp.int32)


def _run_tpu(
    k,
    v,
    g_gamma=None,
    gk=None,
    g=None,
    h0=None,
    chunk_size=64,
    *,
    cu_seqlens=None,
):

    cu = _to_jax_cu_seqlens(cu_seqlens)
    h, ht = chunk_fwd_h_ref(
        k,
        v,
        g_gamma=g_gamma,
        gk=gk,
        g=g,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens_cpu=cu,
        output_final_state=True,
    )

    return h, ht


def _run_pallas(
    k,
    v,
    g_gamma=None,
    gk=None,
    h0=None,
    g=None,
    chunk_size=64,
    *,
    cu_seqlens=None,
):
    cu = _to_jax_cu_seqlens(cu_seqlens)
    h, ht = chunk_fwd_h_kernel(
        k,
        v,
        g_gamma=g_gamma,
        gk=gk,
        g=g,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens_cpu=cu,
        output_final_state=True,
    )
    if cu is None:
        h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])
    return h, ht


# ============================================================================
# Parametrized test — native vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_native_tpu_vs_pallas(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    chunk_size = cfg.get("chunk_size", 64)
    cu = cfg.get("cu_seqlens", None)
    N = 1
    if cu is not None:
        N = len(cu) - 1
    else:
        N = B
    key = jax.random.PRNGKey(1)
    k = jax.random.normal(key, (B, T, H, K))
    v = jax.random.normal(key, (B, T, H, K))
    gk = jax.random.normal(key, (B, T, H, K))
    h0 = jax.random.normal(key, (N, H, K, K))
    g = jax.random.normal(key, (B, T, H))
    g_gamma = -(8 / H * (1 - 1 / 8)) * jnp.arange(H)
    h, ht = _run_tpu(
        k,
        v,
        g_gamma=g_gamma,
        gk=gk,
        h0=h0,
        g=g,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )

    pallas_h, pallas_ht = _run_pallas(
        k,
        v,
        g_gamma=g_gamma,
        gk=gk,
        g=g,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )
    assert compare_tensor("h", h, pallas_h, atol=atol, rtol=rtol)
    assert compare_tensor("ht", ht, pallas_ht, atol=atol, rtol=rtol)


# ============================================================================
# Helpers for edge-case tests
# ============================================================================


def _make_inputs(cfg, seed=42, use_h0=True):
    """Build k, v, gates, h0 from a config dict."""
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    V = cfg.get("V", K)
    chunk_size = cfg.get("chunk_size", 64)
    gates = cfg.get("gates", [])
    if cfg.get("use_h0") is False:
        use_h0 = False

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    k = jax.random.normal(keys[0], (B, T, H, K))
    v = jax.random.normal(keys[1], (B, T, H, V))

    # Use log_sigmoid to bound gates to (-inf, 0], so exp(gate) ∈ (0, 1].
    # This matches real GLA where gates come from log_sigmoid, preventing
    # hidden state explosion over long sequential propagation chains.
    gk = jax.nn.log_sigmoid(jax.random.normal(keys[2], (B, T, H, K))) if "gk" in gates else None
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H))) if "g" in gates else None
    g_gamma = (
        -(8 / H * (1 - 1 / 8)) * jnp.arange(H) if "g_gamma" in gates else None
    )
    h0 = jax.random.normal(keys[4], (B, H, K, V)) if use_h0 else None

    return dict(
        k=k,
        v=v,
        gk=gk,
        g=g,
        g_gamma=g_gamma,
        h0=h0,
        chunk_size=chunk_size,
    )


def _run_and_compare(cfg, atol=1e-4, rtol=1e-4):
    """Run ref vs pallas and compare."""
    inputs = _make_inputs(cfg)
    h, ht = _run_tpu(**inputs)
    pallas_h, pallas_ht = _run_pallas(**inputs)
    assert compare_tensor("h", h, pallas_h, atol=atol, rtol=rtol)
    assert compare_tensor("ht", ht, pallas_ht, atol=atol, rtol=rtol)


# ============================================================================
# Edge-case tests — gate combinations
# ============================================================================


@pytest.mark.parametrize(
    "cfg", GATE_COMBO_CASES, ids=[_case_id(c) for c in GATE_COMBO_CASES]
)
def test_gate_combinations(cfg):
    """Test all 2^3 subsets of {g, gk, g_gamma}."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — h0 = None (zero initial state)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", H0_NONE_CASES, ids=[_case_id(c) for c in H0_NONE_CASES]
)
def test_h0_none(cfg):
    """h0=None should behave as zero initial state."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — single chunk (T == chunk_size)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", SINGLE_CHUNK_CASES, ids=[_case_id(c) for c in SINGLE_CHUNK_CASES]
)
def test_single_chunk(cfg):
    """Only one chunk — no inter-chunk propagation."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — two chunks (minimum multi-chunk)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", TWO_CHUNK_CASES, ids=[_case_id(c) for c in TWO_CHUNK_CASES]
)
def test_two_chunks(cfg):
    """Exactly two chunks — minimal inter-chunk propagation."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — V != K
# ============================================================================


@pytest.mark.parametrize(
    "cfg", V_NEQ_K_CASES, ids=[_case_id(c) for c in V_NEQ_K_CASES]
)
def test_v_neq_k(cfg):
    """Value dimension differs from key dimension."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — single head (H=1)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", SINGLE_HEAD_CASES, ids=[_case_id(c) for c in SINGLE_HEAD_CASES]
)
def test_single_head(cfg):
    """H=1 — single head edge case."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — large K/V with multiple tiles
# ============================================================================


@pytest.mark.parametrize(
    "cfg", LARGE_K_CASES, ids=[_case_id(c) for c in LARGE_K_CASES]
)
def test_large_k_v(cfg):
    """K or V requiring multiple 128-tiles."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — different chunk sizes
# ============================================================================


@pytest.mark.parametrize(
    "cfg", CHUNK_SIZE_CASES, ids=[_case_id(c) for c in CHUNK_SIZE_CASES]
)
def test_chunk_sizes(cfg):
    """Various chunk_size values (32, 128, 256)."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — larger batch sizes
# ============================================================================


@pytest.mark.parametrize(
    "cfg", BATCH_CASES, ids=[_case_id(c) for c in BATCH_CASES]
)
def test_batch_sizes(cfg):
    """B > 1 with various configurations."""
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — output_final_state = False
# ============================================================================


@pytest.mark.parametrize(
    "cfg", NO_FINAL_STATE_CASES, ids=[_case_id(c) for c in NO_FINAL_STATE_CASES]
)
def test_no_final_state(cfg):
    """output_final_state=False — ht should be None."""
    inputs = _make_inputs(cfg)
    h, ht = chunk_fwd_h_ref(
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        g_gamma=inputs["g_gamma"],
        gk=inputs["gk"],
        h0=inputs["h0"],
        chunk_size=inputs["chunk_size"],
        output_final_state=False,
    )
    pallas_h, pallas_ht = chunk_fwd_h_kernel(
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        g_gamma=inputs["g_gamma"],
        gk=inputs["gk"],
        h0=inputs["h0"],
        chunk_size=inputs["chunk_size"],
        output_final_state=False,
    )
    pallas_h = pallas_h.reshape(
        inputs["k"].shape[0], -1,
        inputs["k"].shape[2], inputs["k"].shape[3],
        inputs["v"].shape[-1],
    )
    assert compare_tensor("h", h, pallas_h, atol=1e-4, rtol=1e-4)
    assert ht is None
    assert pallas_ht is None


# ============================================================================
# Edge-case tests — states_in_fp32
# ============================================================================


def test_states_in_fp32():
    """states_in_fp32=True should produce float32 hidden states."""
    cfg = dict(B=1, T=128, H=2, K=128, V=128, chunk_size=64, gates=["gk"])
    inputs = _make_inputs(cfg)
    h_ref, ht_ref = chunk_fwd_h_ref(
        inputs["k"],
        inputs["v"],
        gk=inputs["gk"],
        h0=inputs["h0"],
        chunk_size=64,
        output_final_state=True,
        states_in_fp32=True,
    )
    h_pallas, ht_pallas = chunk_fwd_h_kernel(
        inputs["k"],
        inputs["v"],
        gk=inputs["gk"],
        h0=inputs["h0"],
        chunk_size=64,
        output_final_state=True,
        states_in_fp32=True,
    )
    h_pallas = h_pallas.reshape(1, -1, 2, 128, 128)
    assert h_ref.dtype == jnp.float32, f"ref h dtype should be float32, got {h_ref.dtype}"
    assert h_pallas.dtype == jnp.float32, f"pallas h dtype should be float32, got {h_pallas.dtype}"
    assert compare_tensor("h_fp32", h_ref, h_pallas, atol=1e-4, rtol=1e-4)
    assert compare_tensor("ht_fp32", ht_ref, ht_pallas, atol=1e-4, rtol=1e-4)


# ============================================================================
# Edge-case tests — assertion validation (negative tests)
# ============================================================================


def test_k_not_multiple_of_128_raises():
    """K not a multiple of 128 should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (1, 64, 2, 64))
    v = jax.random.normal(key, (1, 64, 2, 128))
    with pytest.raises(AssertionError, match="K % 128"):
        chunk_fwd_h_kernel(k, v, chunk_size=64, output_final_state=True)


def test_v_not_multiple_of_128_raises():
    """V not a multiple of 128 should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 64))
    with pytest.raises(AssertionError, match="V % 128"):
        chunk_fwd_h_kernel(k, v, chunk_size=64, output_final_state=True)


def test_t_not_multiple_of_chunk_size_raises():
    """T not divisible by chunk_size should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (1, 100, 2, 128))
    v = jax.random.normal(key, (1, 100, 2, 128))
    with pytest.raises(AssertionError, match="T mod chunk_size"):
        chunk_fwd_h_kernel(k, v, chunk_size=64, output_final_state=True)


def test_gv_raises():
    """gv is not supported and should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    gv = jax.random.normal(key, (1, 64, 2, 128))
    with pytest.raises(AssertionError, match="gv is currently not supported"):
        chunk_fwd_h_kernel(k, v, gv=gv, chunk_size=64, output_final_state=True)


def test_h0_wrong_shape_raises():
    """h0 with wrong shape should raise AssertionError."""
    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (1, 64, 2, 128))
    v = jax.random.normal(key, (1, 64, 2, 128))
    h0_bad = jax.random.normal(key, (1, 2, 64, 128))  # K dim wrong
    with pytest.raises(AssertionError):
        chunk_fwd_h_kernel(k, v, h0=h0_bad, chunk_size=64, output_final_state=True)


# ============================================================================
# Edge-case tests — many heads
# ============================================================================


def test_many_heads():
    """H=8, more heads than typical."""
    cfg = dict(B=1, T=128, H=8, K=128, V=128, chunk_size=64, gates=["g", "gk", "g_gamma"])
    _run_and_compare(cfg)


# ============================================================================
# Edge-case tests — long sequence
# ============================================================================


def test_long_sequence():
    """T=2048, longer sequence with many chunks."""
    cfg = dict(B=128, T=4096, H=16, K=128, V=128, chunk_size=64, gates=["gk"])
    _run_and_compare(cfg)


if __name__ == "__main__":
    pytest.main([__file__])
