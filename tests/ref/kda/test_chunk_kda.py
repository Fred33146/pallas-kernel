"""chunk_kda_bwd_intra: JAX CPU ref cross-validation tests.

Tests:
  1. Shape & non-trivial output
  2. Symmetry: zero dAqk/dAkk produce expected zero gradients
  3. Edge cases: zero gates, extreme beta, minimal dimensions
  4. Linearity: gradient scales linearly with upstream dA
  5. CPU ref vs FLA Triton chunk_kda_bwd_intra (GPU, when available)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp

from tops.cpu.ops.kda import chunk_kda_bwd_intra
from tests.utils import compare_tensor

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    torch = None

HAS_FLA = False
try:
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as triton_chunk_kda_bwd_intra

    HAS_FLA = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(B, H, NT, C, K, dtype, seed=42):
    """Generate inputs in flat [B, T, H, K] layout with chunk-local cumsummed g.

    Args:
        B: batch size
        H: number of heads
        NT: number of chunks
        C: chunk size
        K: head dimension
        dtype: JAX dtype
        seed: random seed

    Returns:
        q:    [B, T, H, K] — query
        k:    [B, T, H, K] — key
        g:    [B, T, H, K] — chunk-local cumsummed gates
        beta: [B, T, H]   — mixing coefficient
    """
    T = NT * C
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)

    # Generate g and apply chunk-local cumsum
    g_raw = jax.nn.log_sigmoid(
        jax.random.normal(keys[2], (B, T, H, K))
    ).astype(dtype)
    # Reshape to chunks, cumsum within each chunk, reshape back
    g = g_raw.reshape(B, NT, C, H, K).cumsum(axis=2).reshape(B, T, H, K)

    beta = jax.nn.sigmoid(
        jax.random.normal(keys[3], (B, T, H))
    ).astype(dtype)
    return q, k, g, beta


def _make_dA(B, H, NT, C, dtype, seed=100):
    """Generate random upstream gradients dAqk and dAkk with proper masking.

    dAqk: lower triangular (including diagonal)
    dAkk: strictly lower triangular (excluding diagonal)

    Args:
        B: batch size
        H: number of heads
        NT: number of chunks
        C: chunk size
        dtype: JAX dtype
        seed: random seed

    Returns:
        dAqk: [B, T, H, C] — T = NT * C
        dAkk: [B, T, H, C]
    """
    T = NT * C
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)

    mask_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    mask_strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)

    # Generate in [B, NT, C, H, C] then reshape to [B, T, H, C]
    dAqk_raw = jax.random.normal(keys[0], (B, NT, C, H, C), dtype=dtype)
    dAqk = jnp.where(mask_lower[None, None, :, None, :], dAqk_raw, 0.0)
    dAqk = dAqk.reshape(B, T, H, C)

    dAkk_raw = jax.random.normal(keys[1], (B, NT, C, H, C), dtype=dtype)
    dAkk = jnp.where(mask_strict_lower[None, None, :, None, :], dAkk_raw, 0.0)
    dAkk = dAkk.reshape(B, T, H, C)

    return dAqk, dAkk


# ============================================================================
# Shape configs — shared by CPU ref and Triton tests
# ============================================================================


_SHAPES = [
    dict(B=2, NT=2, H=4, C=16, K=32, seed=42),
    dict(B=1, NT=4, H=2, C=16, K=16, seed=7),
    dict(B=2, NT=1, H=4, C=32, K=16, seed=40),
    dict(B=1, NT=2, H=2, C=64, K=32, seed=99),
    # Edge: minimal dimensions
    dict(B=1, NT=1, H=1, C=4, K=1, seed=11),
    dict(B=1, NT=2, H=1, C=2, K=8, seed=22),
    # 1. K >> C: head dim much larger than chunk size
    dict(B=1, NT=2, H=2, C=16, K=64, seed=51),
    dict(B=1, NT=2, H=2, C=16, K=128, seed=52),
    # 2. H=1 with Triton-compatible dims (C>=16, K>=16)
    dict(B=2, NT=2, H=1, C=16, K=32, seed=61),
    dict(B=1, NT=4, H=1, C=32, K=64, seed=62),
    # 3. Odd NT (non-power-of-2)
    dict(B=1, NT=3, H=2, C=16, K=32, seed=71),
    dict(B=2, NT=5, H=4, C=16, K=16, seed=72),
    dict(B=1, NT=7, H=2, C=32, K=32, seed=73),
    # 4. Large B
    dict(B=4, NT=2, H=4, C=16, K=32, seed=81),
    dict(B=8, NT=2, H=2, C=16, K=16, seed=82),
    # 5. More C=16 K=16 boundary variants
    dict(B=2, NT=4, H=8, C=16, K=16, seed=91),
    dict(B=1, NT=8, H=4, C=16, K=16, seed=92),
    # Larger shape to stress-test; kept within ~4GB GPU budget.
    dict(B=2, NT=64, H=16, C=64, K=128, seed=128),
]


def _shape_id(c):
    return f"B{c['B']}_NT{c['NT']}_H{c['H']}_C{c['C']}_K{c['K']}"


# ============================================================================
# 1. Shape & non-trivial output
# ============================================================================


@pytest.mark.parametrize("cfg", _SHAPES, ids=[_shape_id(c) for c in _SHAPES])
def test_standalone_matches_full_backward_fp64(cfg):
    """Verify standalone chunk_kda_bwd_intra produces correct shapes and non-trivial output."""
    B, NT, H, C, K = cfg["B"], cfg["NT"], cfg["H"], cfg["C"], cfg["K"]
    T = NT * C
    acc_dt = jnp.float64

    q, k, g, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=cfg["seed"])
    dAqk, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=cfg["seed"] + 1)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk, chunk_size=C,
    )

    assert dq.shape == (B, T, H, K)
    assert dk.shape == (B, T, H, K)
    assert db.shape == (B, T, H)
    assert dg.shape == (B, T, H, K)

    assert jnp.abs(dq).max() > 0, "dq is all zeros"
    assert jnp.abs(dk).max() > 0, "dk is all zeros"
    assert jnp.abs(db).max() > 0, "db is all zeros"
    assert jnp.abs(dg).max() > 0, "dg is all zeros"


# ============================================================================
# 2. Symmetry: zero dAqk/dAkk produce expected zero gradients
# ============================================================================


@pytest.mark.parametrize("cfg", _SHAPES, ids=[_shape_id(c) for c in _SHAPES])
def test_standalone_symmetry_fp64(cfg):
    """Verify that zero dAkk produces zero db, and zero dAqk produces zero dq."""
    B, NT, H, C, K = cfg["B"], cfg["NT"], cfg["H"], cfg["C"], cfg["K"]
    T = NT * C
    acc_dt = jnp.float64

    q, k, g, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=cfg["seed"])

    # dAqk only (dAkk=0): should produce nonzero dq but zero db
    dAqk, _ = _make_dA(B, H, NT, C, jnp.float64, seed=cfg["seed"] + 1)
    dAkk_zero = jnp.zeros((B, T, H, C), dtype=acc_dt)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk_zero, chunk_size=C,
    )
    assert jnp.abs(dq).max() > 0, "dq should be nonzero with dAqk"
    assert jnp.allclose(db, 0.0, atol=1e-15), "db should be zero when dAkk=0"

    # dAkk only (dAqk=0): should produce zero dq
    dAqk_zero = jnp.zeros((B, T, H, C), dtype=acc_dt)
    _, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=cfg["seed"] + 2)

    dq2, dk2, db2, dg2 = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk_zero, dAkk, chunk_size=C,
    )
    assert jnp.allclose(dq2, 0.0, atol=1e-15), "dq should be zero when dAqk=0"
    assert jnp.abs(db2).max() > 0, "db should be nonzero with dAkk"


# ============================================================================
# 3. Edge cases: zero gates, extreme beta, minimal dimensions
# ============================================================================


def test_zero_gates_fp64():
    """With g=0, exp(g_r - g_j)=1 everywhere, reducing to plain dot-product gradients."""
    B, NT, H, C, K = 1, 2, 2, 8, 4
    T = NT * C
    acc_dt = jnp.float64

    q, k, _, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=77)
    g_zero = jnp.zeros((B, T, H, K), dtype=jnp.float64)
    dAqk, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=78)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g_zero, beta, dAqk, dAkk, chunk_size=C,
    )

    assert dq.shape == (B, T, H, K)
    assert jnp.abs(dq).max() > 0, "dq should be nonzero"
    assert jnp.abs(dk).max() > 0, "dk should be nonzero"
    assert jnp.abs(db).max() > 0, "db should be nonzero"
    # dg should still be nonzero (gradient of exp at g=0 is 1)
    assert jnp.abs(dg).max() > 0, "dg should be nonzero even when g=0"


def test_beta_zero_fp64():
    """With beta=0, dk from Akk path is zero (dk_left = -beta * ...), but db is
    nonzero because db = dL/d(beta) doesn't depend on beta's current value."""
    B, NT, H, C, K = 1, 2, 2, 8, 4
    T = NT * C
    acc_dt = jnp.float64

    q, k, g, _ = _make_inputs(B, H, NT, C, K, jnp.float64, seed=55)
    beta_zero = jnp.zeros((B, T, H), dtype=jnp.float64)
    # Use only dAkk to isolate the beta-dependent path
    dAqk_zero = jnp.zeros((B, T, H, C), dtype=acc_dt)
    _, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=56)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g, beta_zero, dAqk_zero, dAkk, chunk_size=C,
    )

    # dq should be zero (only dAqk path contributes to dq, and dAqk=0)
    assert jnp.allclose(dq, 0.0, atol=1e-15), "dq should be zero when dAqk=0"
    # db = dL/d(beta) is nonzero regardless of beta's value
    assert jnp.abs(db).max() > 0, "db should be nonzero (gradient w.r.t. beta)"
    # dk = (-beta)*exp(g)*Mkk + exp(-g)*N where N includes (-beta),
    # so dk should be zero when beta=0 and dAqk=0
    assert jnp.allclose(dk, 0.0, atol=1e-15), "dk should be zero when beta=0 and dAqk=0"


def test_beta_one_fp64():
    """With beta=1, the Akk path is fully active; all gradients should be nonzero."""
    B, NT, H, C, K = 1, 2, 2, 8, 4
    T = NT * C
    acc_dt = jnp.float64

    q, k, g, _ = _make_inputs(B, H, NT, C, K, jnp.float64, seed=66)
    beta_one = jnp.ones((B, T, H), dtype=jnp.float64)
    dAqk, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=67)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g, beta_one, dAqk, dAkk, chunk_size=C,
    )

    assert jnp.abs(dq).max() > 0
    assert jnp.abs(dk).max() > 0
    assert jnp.abs(db).max() > 0
    assert jnp.abs(dg).max() > 0


def test_chunk_size_one_fp64():
    """C=1: single-element chunks. dAkk is always zero (strictly lower tri of 1x1)."""
    B, NT, H, C, K = 1, 8, 2, 1, 4
    T = NT * C
    acc_dt = jnp.float64

    q, k, g, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=33)
    dAqk, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=34)

    # dAkk should be all zeros for C=1 (strictly lower tri of 1x1 = empty)
    assert jnp.allclose(dAkk, 0.0)

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk, chunk_size=C,
    )

    assert dq.shape == (B, T, H, K)
    assert jnp.abs(dq).max() > 0, "dq from diagonal dAqk should be nonzero"
    assert jnp.allclose(db, 0.0, atol=1e-15), "db should be zero when dAkk=0 (C=1)"


# ============================================================================
# 4. Linearity: gradient scales linearly with upstream dA
# ============================================================================


def test_linearity_dAqk_fp64():
    """dq should scale linearly with dAqk (dAkk=0)."""
    B, NT, H, C, K = 1, 2, 2, 8, 4
    T = NT * C
    acc_dt = jnp.float64
    alpha = 3.14

    q, k, g, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=88)
    dAqk, _ = _make_dA(B, H, NT, C, jnp.float64, seed=89)
    dAkk_zero = jnp.zeros((B, T, H, C), dtype=acc_dt)

    dq1, dk1, db1, dg1 = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk_zero, chunk_size=C,
    )
    dq2, dk2, db2, dg2 = chunk_kda_bwd_intra(
        q, k, g, beta, alpha * dAqk, dAkk_zero, chunk_size=C,
    )

    np.testing.assert_allclose(
        np.asarray(dq2), np.asarray(alpha * dq1), atol=1e-10, rtol=1e-10,
        err_msg="dq not linear in dAqk",
    )
    np.testing.assert_allclose(
        np.asarray(dk2), np.asarray(alpha * dk1), atol=1e-10, rtol=1e-10,
        err_msg="dk not linear in dAqk",
    )
    np.testing.assert_allclose(
        np.asarray(dg2), np.asarray(alpha * dg1), atol=1e-10, rtol=1e-10,
        err_msg="dg not linear in dAqk",
    )


def test_linearity_dAkk_fp64():
    """dk, db, dg should scale linearly with dAkk (dAqk=0)."""
    B, NT, H, C, K = 1, 2, 2, 8, 4
    T = NT * C
    acc_dt = jnp.float64
    alpha = 2.71

    q, k, g, beta = _make_inputs(B, H, NT, C, K, jnp.float64, seed=88)
    _, dAkk = _make_dA(B, H, NT, C, jnp.float64, seed=90)
    dAqk_zero = jnp.zeros((B, T, H, C), dtype=acc_dt)

    dq1, dk1, db1, dg1 = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk_zero, dAkk, chunk_size=C,
    )
    dq2, dk2, db2, dg2 = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk_zero, alpha * dAkk, chunk_size=C,
    )

    np.testing.assert_allclose(
        np.asarray(dk2), np.asarray(alpha * dk1), atol=1e-10, rtol=1e-10,
        err_msg="dk not linear in dAkk",
    )
    np.testing.assert_allclose(
        np.asarray(db2), np.asarray(alpha * db1), atol=1e-10, rtol=1e-10,
        err_msg="db not linear in dAkk",
    )
    np.testing.assert_allclose(
        np.asarray(dg2), np.asarray(alpha * dg1), atol=1e-10, rtol=1e-10,
        err_msg="dg not linear in dAkk",
    )


# ============================================================================
# 5. CPU ref vs FLA Triton chunk_kda_bwd_intra (GPU, when available)
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", _SHAPES, ids=[_shape_id(c) for c in _SHAPES])
def test_cpu_ref_vs_triton_bwd_intra(cfg):
    """Compare JAX CPU ref chunk_kda_bwd_intra vs FLA Triton."""
    B, NT, H, C, K = cfg["B"], cfg["NT"], cfg["H"], cfg["C"], cfg["K"]
    # Triton tl.dot requires K >= 16 and C >= 16
    if C < 16 or K < 16:
        pytest.skip(f"Triton tl.dot requires C >= 16 and K >= 16 (got C={C}, K={K})")
    T = NT * C
    dtype = jnp.float32

    q, k, g, beta = _make_inputs(
        B, H, NT, C, K, dtype, seed=cfg["seed"]
    )
    dAqk, dAkk = _make_dA(B, H, NT, C, dtype, seed=cfg["seed"] + 1)

    # ---- JAX CPU ref ----
    dq_jax, dk_jax, db_jax, dg_jax = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk, chunk_size=C,
    )

    def _to_torch(x, dtype=torch.float32):
        return torch.tensor(np.asarray(x), dtype=dtype, device="cuda")

    q_t = _to_torch(q)
    k_t = _to_torch(k)
    g_t = _to_torch(g)
    beta_t = _to_torch(beta)
    dAqk_t = _to_torch(dAqk)
    dAkk_t = _to_torch(dAkk)

    dq_t = torch.zeros_like(q_t)
    dk_t = torch.zeros_like(k_t)
    db_t = torch.zeros(B, T, H, dtype=torch.float32, device="cuda")
    dg_t = torch.zeros_like(q_t)

    # ---- FLA Triton ----
    dq_t, dk_t, db_t, dg_t = triton_chunk_kda_bwd_intra(
        q=q_t,
        k=k_t,
        g=g_t,
        beta=beta_t,
        dAqk=dAqk_t,
        dAkk=dAkk_t,
        dq=dq_t,
        dk=dk_t,
        db=db_t,
        dg=dg_t,
        chunk_size=C,
    )

    # ---- Compare in [B, T, H, K] layout ----
    def _to_np(t):
        return t.detach().cpu().float().numpy()

    atol, rtol = 5e-4, 5e-4
    assert compare_tensor(
        "dq", _to_np(dq_t), np.asarray(dq_jax),
        atol=atol, rtol=rtol, compare_dtype=np.float32,
    )
    assert compare_tensor(
        "dk", _to_np(dk_t), np.asarray(dk_jax),
        atol=atol, rtol=rtol, compare_dtype=np.float32,
    )
    assert compare_tensor(
        "db", _to_np(db_t), np.asarray(db_jax),
        atol=atol, rtol=rtol, compare_dtype=np.float32,
    )
    assert compare_tensor(
        "dg", _to_np(dg_t), np.asarray(dg_jax),
        atol=atol, rtol=rtol, compare_dtype=np.float32,
    )


@requires_triton
@pytest.mark.parametrize("label,beta_override,g_override", [
    ("g_zero", None, "zero"),
    ("beta_zero", "zero", None),
    ("beta_one", "one", None),
])
def test_cpu_ref_vs_triton_special_inputs(label, beta_override, g_override):
    """Cross-validate special input patterns (g=0, beta=0, beta=1) against Triton."""
    B, NT, H, C, K = 2, 2, 4, 16, 32
    T = NT * C
    dtype = jnp.float32

    q, k, g, beta = _make_inputs(B, H, NT, C, K, dtype, seed=200)
    dAqk, dAkk = _make_dA(B, H, NT, C, dtype, seed=201)

    if g_override == "zero":
        g = jnp.zeros_like(g)
    if beta_override == "zero":
        beta = jnp.zeros_like(beta)
    elif beta_override == "one":
        beta = jnp.ones_like(beta)

    # ---- JAX CPU ref ----
    dq_jax, dk_jax, db_jax, dg_jax = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk, chunk_size=C,
    )
    def _to_torch(x, dtype=torch.float32):
        return torch.tensor(np.asarray(x), dtype=dtype, device="cuda")

    q_t = _to_torch(q)
    k_t = _to_torch(k)
    g_t = _to_torch(g)
    beta_t = _to_torch(beta)
    dAqk_t = _to_torch(dAqk)
    dAkk_t = _to_torch(dAkk)

    dq_t = torch.zeros_like(q_t)
    dk_t = torch.zeros_like(k_t)
    db_t = torch.zeros(B, T, H, dtype=torch.float32, device="cuda")
    dg_t = torch.zeros_like(q_t)

    dq_t, dk_t, db_t, dg_t = triton_chunk_kda_bwd_intra(
        q=q_t, k=k_t, g=g_t, beta=beta_t,
        dAqk=dAqk_t, dAkk=dAkk_t,
        dq=dq_t, dk=dk_t, db=db_t, dg=dg_t,
        chunk_size=C,
    )

    def _to_np(t):
        return t.detach().cpu().float().numpy()

    atol, rtol = 5e-4, 5e-4
    assert compare_tensor("dq", _to_np(dq_t), np.asarray(dq_jax), atol=atol, rtol=rtol, compare_dtype=np.float32)
    assert compare_tensor("dk", _to_np(dk_t), np.asarray(dk_jax), atol=atol, rtol=rtol, compare_dtype=np.float32)
    assert compare_tensor("db", _to_np(db_t), np.asarray(db_jax), atol=atol, rtol=rtol, compare_dtype=np.float32)
    assert compare_tensor("dg", _to_np(dg_t), np.asarray(dg_jax), atol=atol, rtol=rtol, compare_dtype=np.float32)


if __name__ == "__main__":
    print("HAS_FLA = ", HAS_FLA)
    pytest.main([__file__, "-V", "-s"])
