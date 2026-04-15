"""Pallas KDA Intra-Chunk Backward Kernel vs JAX CPU Reference.

Compares the Pallas TPU kernel against the pure JAX CPU reference:
  - CPU ref: float32
  - Pallas (TPU): bfloat16

Usage:
    uv run python -m pytest tests/ops/kda/test_chunk_kda_tpu.py -v
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tops.cpu.ops.kda import chunk_kda_bwd_intra
from tests.utils import compare_tensor

try:
    from tops.ops.kda import chunk_kda_bwd_intra as pallas_chunk_kda_bwd_intra

    HAS_PALLAS = True
except ImportError:
    HAS_PALLAS = False

# ---------------------------------------------------------------------------
# Backward: Pallas kda_intra_chunk_bwd vs CPU ref chunk_kda_bwd_intra
# ---------------------------------------------------------------------------

_BWD_SHAPES = [
    # Basic shapes
    dict(B=2, H=4, NT=2, C=16, K=32, seed=42),
    dict(B=1, H=2, NT=4, C=16, K=16, seed=7),
    dict(B=2, H=4, NT=1, C=32, K=16, seed=40),
    dict(B=1, H=2, NT=2, C=64, K=32, seed=99),
    # K >> C: head dim much larger than chunk size
    dict(B=1, H=2, NT=2, C=16, K=64, seed=51),
    dict(B=1, H=2, NT=2, C=16, K=128, seed=52),
    # H=1
    dict(B=2, H=1, NT=2, C=16, K=32, seed=61),
    dict(B=1, H=1, NT=4, C=32, K=64, seed=62),
    # Odd NT (non-power-of-2)
    dict(B=1, H=2, NT=3, C=16, K=32, seed=71),
    dict(B=2, H=4, NT=5, C=16, K=16, seed=72),
    dict(B=1, H=2, NT=7, C=32, K=32, seed=73),
    # Large B
    dict(B=4, H=4, NT=2, C=16, K=32, seed=81),
    dict(B=8, H=2, NT=2, C=16, K=16, seed=82),
    # More C=16 K=16 boundary variants
    dict(B=2, H=8, NT=4, C=16, K=16, seed=91),
    dict(B=1, H=4, NT=8, C=16, K=16, seed=92),
    # Larger shape to stress-test
    dict(B=2, H=16, NT=64, C=64, K=128, seed=128),
]


def _bwd_shape_id(c):
    return f"B{c['B']}_H{c['H']}_NT{c['NT']}_C{c['C']}_K{c['K']}"


def _make_bwd_inputs(B, H, NT, C, K, seed=42):
    """Generate inputs for bwd_intra comparison.

    Returns float32 tensors for CPU ref and bfloat16 tensors for Pallas,
    both in [B, T, H, K] layout. CPU ref uses natural-log gates, Pallas
    uses log2 gates.

    Args:
        B: batch size
        H: number of heads
        NT: number of chunks
        C: chunk size
        K: head dimension
        seed: random seed

    Returns:
        cpu_inputs: (q, k, g, beta, dAqk, dAkk) float32 [B,T,H,K]
        pallas_inputs: (q, k, g, beta, dAqk, dAkk) bfloat16 [B,T,H,K]
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    T = NT * C

    # --- Generate in [B, H, NT, C, K] layout, float32 ---
    q_c = jax.random.normal(keys[0], (B, H, NT, C, K), dtype=jnp.float32) * 0.1
    k_c = jax.random.normal(keys[1], (B, H, NT, C, K), dtype=jnp.float32) * 0.1

    # g: negative for stability, chunk-local cumsum
    g_raw = (
        -jnp.abs(jax.random.normal(keys[2], (B, H, NT, C, K), dtype=jnp.float32))
        * 0.1
    )
    g_c = g_raw.cumsum(axis=3)  # chunk-local cumsum, natural-log base

    beta_c = (
        jax.nn.sigmoid(jax.random.normal(keys[3], (B, H, NT, C), dtype=jnp.float32))
    )

    # dAqk: lower triangular (including diagonal)
    mask_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    dAqk_raw = jax.random.normal(keys[4], (B, H, NT, C, C), dtype=jnp.float32) * 0.1
    dAqk = jnp.where(mask_lower[None, None, None], dAqk_raw, 0.0)

    # dAkk: strictly lower triangular (excluding diagonal)
    mask_strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)
    dAkk_raw = jax.random.normal(keys[5], (B, H, NT, C, C), dtype=jnp.float32) * 0.1
    dAkk = jnp.where(mask_strict_lower[None, None, None], dAkk_raw, 0.0)

    # --- Convert to flat layout [B, T, H, K] ---
    def _to_flat(x):
        return x.reshape(B, H, T, -1).transpose(0, 2, 1, 3)

    q_flat = _to_flat(q_c)             # [B, T, H, K]
    k_flat = _to_flat(k_c)             # [B, T, H, K]
    g_flat = _to_flat(g_c)             # [B, T, H, K] (natural-log)
    beta_flat = beta_c.reshape(B, H, T).transpose(0, 2, 1)  # [B, T, H]

    # dAqk/dAkk: [B, H, NT, C, C] -> [B, T, H, C]
    dAqk_flat = dAqk.reshape(B, H, T, C).transpose(0, 2, 1, 3)
    dAkk_flat = dAkk.reshape(B, H, T, C).transpose(0, 2, 1, 3)

    # --- CPU ref inputs (float32, natural-log) ---
    cpu_inputs = (q_flat, k_flat, g_flat, beta_flat, dAqk_flat, dAkk_flat)

    # --- Pallas inputs (bfloat16, log2, negated dAkk) ---
    g_pallas = g_flat / jnp.log(2.0)
    # Sign convention: CPU ref uses -beta internally, Pallas uses +beta
    # -> pass -dAkk to Pallas so that (-dAkk)*(+beta) = dAkk*(-beta)
    dAkk_pallas = -dAkk_flat

    pallas_inputs = (
        q_flat.astype(jnp.bfloat16),
        k_flat.astype(jnp.bfloat16),
        g_pallas.astype(jnp.bfloat16),
        beta_flat.astype(jnp.bfloat16),
        dAqk_flat.astype(jnp.bfloat16),
        dAkk_pallas.astype(jnp.bfloat16),
    )

    return cpu_inputs, pallas_inputs


@pytest.mark.skipif(not HAS_PALLAS, reason="tops.ops.kda not available")
class TestPallasIntraChunkBwd:
    """kda_intra_chunk_bwd (Pallas TPU, bf16) vs chunk_kda_bwd_intra (CPU ref, fp32)."""

    @pytest.mark.parametrize(
        "cfg", _BWD_SHAPES, ids=[_bwd_shape_id(c) for c in _BWD_SHAPES]
    )
    def test_bwd_basic(self, cfg):
        """Compare Pallas backward kernel (bf16) against CPU reference (fp32)."""
        B, H, NT, C, K = cfg["B"], cfg["H"], cfg["NT"], cfg["C"], cfg["K"]

        cpu_inputs, pallas_inputs = _make_bwd_inputs(
            B, H, NT, C, K, seed=cfg["seed"]
        )
        q, k, g_cpu, beta, dAqk, dAkk_cpu = cpu_inputs
        q_pl, k_pl, g_pl, beta_pl, dAqk_pl, dAkk_pl = pallas_inputs

        # --- CPU ref (float32) ---
        # CPU ref now uses exp2 (log2 gates) and same sign convention as Pallas,
        # so pass log2 gates and same dAkk as Pallas.
        g_cpu_log2 = g_cpu / jnp.log(2.0)
        dq_cpu, dk_cpu, db_cpu, dg_cpu = chunk_kda_bwd_intra(
            q, k, g_cpu_log2, beta, dAqk, -dAkk_cpu, chunk_size=C,
        )

        # --- Pallas TPU (bfloat16) ---
        dq_pl, dk_pl, db_pl, dg_pl = pallas_chunk_kda_bwd_intra(
            q=q_pl,
            k=k_pl,
            g=g_pl,
            beta=beta_pl,
            dAqk=dAqk_pl,
            dAkk=dAkk_pl,
            chunk_size=C,
        )

        # bf16 vs fp32: relaxed tolerance
        atol, rtol = 5e-3, 5e-2
        assert compare_tensor(
            "dq", dq_cpu, dq_pl,
            atol=atol, rtol=rtol, compare_dtype=np.float32,
        )
        assert compare_tensor(
            "dk", dk_cpu, dk_pl,
            atol=atol, rtol=rtol, compare_dtype=np.float32,
        )
        assert compare_tensor(
            "db", db_cpu, db_pl,
            atol=atol, rtol=rtol, compare_dtype=np.float32,
        )
        assert compare_tensor(
            "dg", dg_cpu, dg_pl,
            atol=atol, rtol=rtol, compare_dtype=np.float32,
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
