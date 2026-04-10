"""simple_gla varlen forward: Pallas chunk_simple_gla_fwd_varlen vs CPU reference.

Tests that chunk_simple_gla_fwd_varlen (Pallas, cu_seqlens_dev != None) produces
equivalent results to cpu_chunk_simple_gla_fwd run independently per sequence.

Constraints for chunk_simple_gla_fwd_varlen:
  - B must be 1 (packed varlen layout)
  - cu_seqlens_dev values must be multiples of chunk_size
  - T must be a multiple of chunk_size
  - K, V must be multiples of 128
  - cu_seqlens_cpu must be None
  - Only g_gamma gate mode (chunk path does not support per-token g)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from tops.ops.simple_gla import chunk_simple_gla_fwd_varlen
from tops.cpu.ops.simple_gla import chunk_simple_gla_fwd as cpu_chunk_simple_gla_fwd
from tests.utils import compare_tensor

# ============================================================================
# Test configs
#
# Constraints:
#   - B = 1 (varlen requires packed layout with batch=1)
#   - T % chunk_size == 0
#   - K % 128 == 0, V % 128 == 0
#   - All cu_seqlens entries must be multiples of chunk_size
# ============================================================================

CHUNK_SIZE = 64

VARLEN_FWD_CASES = [
    # ── 2 equal segments ──
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=100, gate="g_gamma"),
    # ── 2 unequal segments ──
    dict(T=256, H=4, K=128, V=128, cu_seqlens=[0, 64, 256], seed=101, gate="g_gamma"),
    # ── 3 segments ──
    dict(T=192, H=2, K=128, V=128, cu_seqlens=[0, 64, 128, 192], seed=102, gate="g_gamma"),
    # ── single segment (varlen with 1 seq = regular forward) ──
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 128], seed=103, gate="g_gamma"),
    # ── with h0 ──
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=110, gate="g_gamma", h0=True),
    dict(T=256, H=2, K=128, V=128, cu_seqlens=[0, 64, 256], seed=111, gate="g_gamma", h0=True),
    dict(T=192, H=2, K=128, V=128, cu_seqlens=[0, 64, 128, 192], seed=112, gate="g_gamma", h0=True),
    # ── K != V ──
    dict(T=128, H=2, K=128, V=256, cu_seqlens=[0, 64, 128], seed=120, gate="g_gamma"),
    dict(T=128, H=2, K=256, V=128, cu_seqlens=[0, 64, 128], seed=121, gate="g_gamma"),
    # ── single head ──
    dict(T=128, H=1, K=128, V=128, cu_seqlens=[0, 64, 128], seed=130, gate="g_gamma"),
    # ── many heads ──
    dict(T=128, H=16, K=128, V=128, cu_seqlens=[0, 64, 128], seed=131, gate="g_gamma"),
    # ── longer sequences ──
    dict(T=512, H=2, K=128, V=128, cu_seqlens=[0, 128, 384, 512], seed=140, gate="g_gamma"),
    dict(T=512, H=4, K=128, V=128, cu_seqlens=[0, 256, 512], seed=141, gate="g_gamma", h0=True),
    # ── custom scale ──
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=150, gate="g_gamma", scale=0.1),
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=151, gate="g_gamma", scale=0.1, h0=True),
    # ── chunk_size=128 ──
    dict(T=256, H=2, K=128, V=128, cu_seqlens=[0, 128, 256], seed=160, gate="g_gamma", chunk_size=128),
    dict(T=384, H=4, K=128, V=128, cu_seqlens=[0, 128, 256, 384], seed=161, gate="g_gamma", chunk_size=128),
    # ── many short segments ──
    dict(T=256, H=4, K=128, V=128, cu_seqlens=[0, 64, 128, 192, 256], seed=170, gate="g_gamma"),
    dict(T=256, H=2, K=128, V=128, cu_seqlens=[0, 64, 128, 192, 256], seed=171, gate="g_gamma", h0=True),
    # ── no gate (g_gamma=None) ──
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=180, gate="none"),
    dict(T=128, H=4, K=128, V=128, cu_seqlens=[0, 64, 128], seed=181, gate="none", h0=True),
]


def _varlen_fwd_case_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    parts = [f"T{c['T']}_segs{n_segs}_H{c['H']}_K{c['K']}_V{c['V']}"]
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


def _make_varlen_inputs(cfg, *, dtype=jnp.bfloat16):
    """Generate random inputs for varlen forward test.

    Args:
        cfg: Config dict with T, H, K, V, cu_seqlens, seed, gate, h0, scale.
        dtype: Data type for q, k, v, h0.

    Returns:
        (q, k, v, g_gamma, h0, cu_seqlens_dev) -- all JAX arrays.
    """
    T, H, K, V = cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    cu_seqlens = cfg["cu_seqlens"]
    N = len(cu_seqlens) - 1  # number of sequences

    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (1, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (1, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (1, T, H, V), dtype=dtype)

    gate = cfg.get("gate", "none")
    g_gamma = None
    if gate == "g_gamma":
        g_gamma = -jnp.abs(jax.random.normal(keys[3], (H,), dtype=jnp.float32)) * 0.5

    h0 = None
    if cfg.get("h0"):
        h0 = jax.random.normal(keys[4], (N, H, K, V), dtype=dtype)

    cu_seqlens_dev = jnp.array(cu_seqlens, dtype=jnp.int32)

    return q, k, v, g_gamma, h0, cu_seqlens_dev


def _run_cpu_ref_per_seq(q, k, v, cu_seqlens, *, g_gamma=None, h0=None,
                         scale=None, chunk_size=CHUNK_SIZE):
    """Run cpu_chunk_simple_gla_fwd independently per sequence and concatenate.

    This is the ground-truth reference: each sub-sequence is processed
    independently with its own initial state, guaranteeing no cross-sequence
    contamination.

    Args:
        q: [1, T, H, K] packed queries.
        k: [1, T, H, K] packed keys.
        v: [1, T, H, V] packed values.
        cu_seqlens: list or array of cumulative sequence lengths.
        g_gamma: [H] per-head log-decay, or None.
        h0: [N, H, K, V] per-sequence initial state, or None.
        scale: query scaling factor.
        chunk_size: chunk size.

    Returns:
        (o_cat, ht_cat) -- concatenated output and stacked final states.
    """
    K_dim = q.shape[-1]
    s = scale if scale is not None else K_dim ** -0.5
    N = len(cu_seqlens) - 1

    outputs = []
    final_states = []
    for n in range(N):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        h0_n = h0[n:n + 1] if h0 is not None else None

        o_n, ht_n = cpu_chunk_simple_gla_fwd(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g=None,
            g_gamma=g_gamma,
            scale=s,
            initial_state=h0_n,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        outputs.append(o_n)
        if ht_n is not None:
            final_states.append(ht_n[0])

    o_cat = jnp.concatenate(outputs, axis=1)
    ht_cat = jnp.stack(final_states, axis=0) if final_states else None
    return o_cat, ht_cat


def _run_pallas_varlen_fwd(q, k, v, cu_seqlens_dev, *, g_gamma=None, h0=None,
                           scale=None, chunk_size=CHUNK_SIZE):
    """Run chunk_simple_gla_fwd_varlen (Pallas varlen kernel).

    Args:
        q: [1, T, H, K] packed queries.
        k: [1, T, H, K] packed keys.
        v: [1, T, H, V] packed values.
        cu_seqlens_dev: [N+1] cumulative sequence lengths (device array).
        g_gamma: [H] per-head log-decay, or None.
        h0: [N, H, K, V] per-sequence initial state, or None.
        scale: query scaling factor.
        chunk_size: chunk size.

    Returns:
        (o, ht) -- output [1, T, H, V] and final state [N, H, K, V] or None.
    """
    o, ht = chunk_simple_gla_fwd_varlen(
        q, k, v,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        use_ht=True,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Forward: Pallas varlen vs CPU per-sequence reference
# ============================================================================


@pytest.mark.parametrize(
    "cfg", VARLEN_FWD_CASES,
    ids=[_varlen_fwd_case_id(c) for c in VARLEN_FWD_CASES],
)
def test_varlen_fwd_vs_cpu(cfg):
    """chunk_simple_gla_fwd_varlen (Pallas) should match per-sequence CPU reference."""
    T = cfg["T"]
    C = cfg.get("chunk_size", CHUNK_SIZE)
    NT = T // C
    scale = cfg.get("scale", None)
    cu_seqlens = cfg["cu_seqlens"]

    # Tolerance: bf16 matmul accumulation errors compound across chunks.
    atol = cfg.get("atol", min(5e-2, 2e-2 + 1e-2 * max(NT, 1)))
    rtol = cfg.get("rtol", 5e-2)
    max_ulp = 4

    q, k, v, g_gamma, h0, cu_seqlens_dev = _make_varlen_inputs(cfg, dtype=jnp.bfloat16)

    # CPU reference: run each sub-sequence independently
    o_ref, ht_ref = _run_cpu_ref_per_seq(
        q, k, v, cu_seqlens,
        g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=C,
    )

    # Pallas varlen kernel
    o_pl, ht_pl = _run_pallas_varlen_fwd(
        q, k, v, cu_seqlens_dev,
        g_gamma=g_gamma, h0=h0, scale=scale, chunk_size=C,
    )

    assert compare_tensor("output", o_ref, o_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)

    if ht_ref is not None and ht_pl is not None:
        ht_atol = max(atol, 5e-2)
        assert compare_tensor(
            "final_state", ht_ref, ht_pl, atol=ht_atol, rtol=rtol, max_ulp=max_ulp,
        )


# ============================================================================
# Cross-validation: varlen should match non-varlen for single-sequence input
#
# When cu_seqlens = [0, T] (one sequence spanning entire T), the varlen
# kernel should produce identical results to the regular forward path.
# ============================================================================

SINGLE_SEQ_CASES = [
    dict(T=128, H=4, K=128, V=128, seed=200, gate="g_gamma"),
    dict(T=256, H=2, K=128, V=128, seed=201, gate="g_gamma"),
    dict(T=128, H=4, K=128, V=128, seed=202, gate="g_gamma", h0=True),
    dict(T=256, H=2, K=128, V=128, seed=203, gate="g_gamma", h0=True),
]


def _single_seq_case_id(c):
    parts = [f"T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


@pytest.mark.parametrize(
    "cfg", SINGLE_SEQ_CASES,
    ids=[_single_seq_case_id(c) for c in SINGLE_SEQ_CASES],
)
def test_varlen_single_seq_matches_regular(cfg):
    """Varlen with cu_seqlens=[0,T] should match non-varlen CPU forward exactly."""
    T, H, K, V = cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    C = cfg.get("chunk_size", CHUNK_SIZE)
    NT = T // C

    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (1, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (1, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (1, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(keys[3], (H,), dtype=jnp.float32)) * 0.5

    h0 = None
    if cfg.get("h0"):
        h0 = jax.random.normal(keys[4], (1, H, K, V), dtype=jnp.bfloat16)

    s = K ** -0.5
    cu_seqlens_dev = jnp.array([0, T], dtype=jnp.int32)

    # Non-varlen CPU reference (single sequence, no cu_seqlens)
    o_ref, ht_ref = cpu_chunk_simple_gla_fwd(
        q, k, v,
        g=None, g_gamma=g_gamma, scale=s,
        initial_state=h0, output_final_state=True, chunk_size=C,
    )

    # Varlen Pallas kernel with cu_seqlens=[0, T]
    o_pl, ht_pl = chunk_simple_gla_fwd_varlen(
        q, k, v,
        g_gamma=g_gamma, scale=s, h0=h0,
        use_ht=True, cu_seqlens_dev=cu_seqlens_dev, chunk_size=C,
    )

    atol = min(5e-2, 2e-2 + 1e-2 * max(NT, 1))
    rtol = 5e-2
    max_ulp = 4

    assert compare_tensor("output", o_ref, o_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if ht_ref is not None and ht_pl is not None:
        assert compare_tensor(
            "final_state", ht_ref, ht_pl, atol=max(atol, 5e-2), rtol=rtol, max_ulp=max_ulp,
        )


# ============================================================================
# Sequence isolation: varlen should prevent cross-sequence information leakage
#
# Verify that the output for each sub-sequence in the packed tensor is
# identical regardless of what other sequences are packed alongside it.
# ============================================================================


def test_varlen_sequence_isolation():
    """Output of each sub-sequence should be independent of other packed sequences."""
    H, K, V, C = 4, 128, 128, 64
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    # Create two independent sequences of length 64
    q1 = jax.random.normal(keys[0], (1, 64, H, K), dtype=jnp.bfloat16)
    k1 = jax.random.normal(keys[1], (1, 64, H, K), dtype=jnp.bfloat16)
    v1 = jax.random.normal(keys[2], (1, 64, H, V), dtype=jnp.bfloat16)

    q2 = jax.random.normal(keys[3], (1, 64, H, K), dtype=jnp.bfloat16)
    k2 = jax.random.normal(keys[4], (1, 64, H, K), dtype=jnp.bfloat16)
    v2 = jax.random.normal(keys[5], (1, 64, H, V), dtype=jnp.bfloat16)

    g_gamma = jnp.full((H,), -0.3, dtype=jnp.float32)
    s = K ** -0.5
    cu = jnp.array([0, 64, 128], dtype=jnp.int32)

    # Pack [seq1, seq2] and run varlen
    q_packed = jnp.concatenate([q1, q2], axis=1)
    k_packed = jnp.concatenate([k1, k2], axis=1)
    v_packed = jnp.concatenate([v1, v2], axis=1)

    o_packed, _ = chunk_simple_gla_fwd_varlen(
        q_packed, k_packed, v_packed,
        g_gamma=g_gamma, scale=s,
        cu_seqlens_dev=cu, chunk_size=C,
    )

    # Run seq1 alone (non-varlen reference)
    o_seq1_ref, _ = cpu_chunk_simple_gla_fwd(
        q1, k1, v1,
        g=None, g_gamma=g_gamma, scale=s,
        initial_state=None, output_final_state=False, chunk_size=C,
    )

    # Now pack [seq1, different_seq2] -- seq1's output should be the same
    q3 = jax.random.normal(jax.random.PRNGKey(999), (1, 64, H, K), dtype=jnp.bfloat16)
    k3 = jax.random.normal(jax.random.PRNGKey(998), (1, 64, H, K), dtype=jnp.bfloat16)
    v3 = jax.random.normal(jax.random.PRNGKey(997), (1, 64, H, V), dtype=jnp.bfloat16)

    q_packed2 = jnp.concatenate([q1, q3], axis=1)
    k_packed2 = jnp.concatenate([k1, k3], axis=1)
    v_packed2 = jnp.concatenate([v1, v3], axis=1)

    o_packed2, _ = chunk_simple_gla_fwd_varlen(
        q_packed2, k_packed2, v_packed2,
        g_gamma=g_gamma, scale=s,
        cu_seqlens_dev=cu, chunk_size=C,
    )

    # seq1 output should be identical regardless of what seq2 is
    assert compare_tensor(
        "isolation_packed_vs_ref", o_seq1_ref, o_packed[:, :64], atol=5e-2, rtol=5e-2, max_ulp=4,
    )
    assert compare_tensor(
        "isolation_cross_pack", o_packed[:, :64], o_packed2[:, :64], atol=1e-6, rtol=1e-6, max_ulp=1,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
