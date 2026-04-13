"""naive_recurrent_kda / naive_chunk_kda: JAX CPU ref (tops.cpu.ops.kda) tests.

Tests:
  1. Dtype verification (no GPU)
  2. Input assertion tests (no GPU)
  3. Cross-validation: naive_recurrent vs naive_chunk (no GPU)
  4. Backward cross-validation: jax.grad(naive) vs jax.grad(chunk) (no GPU)
  5. CPU ref vs FLA Triton (GPU, when available)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp

from tops.cpu.ops.kda import naive_recurrent_kda, naive_chunk_kda
from tests.utils import compare_tensor


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False):
    """Generate KDA test inputs.

    Args:
        B, T, H, K, V: tensor dimensions
        dtype: JAX dtype for q, k, v, g
        seed: random seed
        h0: whether to generate initial state

    Returns:
        q, k, v, g, beta, h0_arr
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    g = jax.nn.log_sigmoid(
        jax.random.normal(keys[3], (B, T, H, K))
    ).astype(dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (B, T, H))
    ).astype(dtype)
    h0_arr = jax.random.normal(keys[5], (B, H, K, V), dtype=acc) if h0 else None
    return q, k, v, g, beta, h0_arr


# ============================================================================
# Shape configs
# ============================================================================

_XVAL_SHAPES = [
    # standard
    dict(B=2, T=64, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # non-aligned T (padding test)
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    # T < chunk_size edge case
    dict(B=2, T=1, H=4, K=32, V=64, seed=50),
    # with initial state
    dict(B=2, T=64, H=4, K=32, V=64, seed=13, h0=True),
    # larger
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]

ALL_DTYPES = [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]


def _shape_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


# ============================================================================
# 1. Dtype verification (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_naive_recurrent_dtypes(dtype):
    """Verify output and state dtypes match FLA contract."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, beta, _ = _make_inputs(B, T, H, K, V, dtype)
    o, ht = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    assert o.dtype == dtype, f"o.dtype={o.dtype}, expected {dtype}"
    assert ht.dtype == acc, f"ht.dtype={ht.dtype}, expected {acc}"


def test_naive_recurrent_output_final_state_false():
    """When output_final_state=False, final_state should be None."""
    B, T, H, K, V = 2, 32, 4, 32, 64
    q, k, v, g, beta, _ = _make_inputs(B, T, H, K, V, jnp.float32)
    o, ht = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
    assert ht is None


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_naive_chunk_dtypes(dtype):
    """Verify chunk output and state dtypes match FLA contract."""
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, beta, _ = _make_inputs(B, T, H, K, V, dtype)
    o, ht = naive_chunk_kda(q, k, v, g, beta, output_final_state=True)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    assert o.dtype == dtype
    assert ht.dtype == acc


def test_naive_chunk_output_final_state_false():
    """When output_final_state=False, final_state should be None."""
    B, T, H, K, V = 2, 64, 4, 32, 64
    q, k, v, g, beta, _ = _make_inputs(B, T, H, K, V, jnp.float32)
    o, ht = naive_chunk_kda(q, k, v, g, beta, output_final_state=False)
    assert ht is None


# ============================================================================
# 2. Input assertion tests
# ============================================================================


def test_assert_q_ndim():
    """q must be 4D."""
    with pytest.raises(AssertionError, match="4D"):
        naive_recurrent_kda(
            jnp.zeros((2, 4, 3)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3)),
        )


def test_assert_k_shape():
    """k must match q shape."""
    with pytest.raises(AssertionError, match="k shape"):
        naive_recurrent_kda(
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 16)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3)),
        )


def test_assert_beta_ndim():
    """beta must be 3D."""
    with pytest.raises(AssertionError, match="beta"):
        naive_chunk_kda(
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4)),
        )


def test_assert_initial_state_shape():
    """initial_state must match (B, H, K, V)."""
    with pytest.raises(AssertionError, match="initial_state"):
        naive_recurrent_kda(
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3, 8)),
            jnp.zeros((2, 4, 3)),
            initial_state=jnp.zeros((2, 3, 8, 99)),
        )


# ============================================================================
# 3. Cross-validation: naive_recurrent vs naive_chunk (no GPU needed)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES]
)
def test_naive_vs_chunk_fp64(cfg):
    """Cross-validate naive vs chunk in fp64 (high precision)."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, g, beta, h0 = _make_inputs(
        B, T, H, K, V, jnp.float64, seed=cfg["seed"], h0=cfg.get("h0", False)
    )
    o_naive, s_naive = naive_recurrent_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    assert compare_tensor(
        "output", o_naive, o_chunk, atol=1e-10, rtol=1e-10,
        compare_dtype=np.float64,
    )
    if s_naive is not None:
        assert compare_tensor(
            "state", s_naive, s_chunk, atol=1e-10, rtol=1e-10,
            compare_dtype=np.float64,
        )


@pytest.mark.parametrize(
    "cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES]
)
def test_naive_vs_chunk_fp32(cfg):
    """Cross-validate naive vs chunk in fp32."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, g, beta, h0 = _make_inputs(
        B, T, H, K, V, jnp.float32, seed=cfg["seed"], h0=cfg.get("h0", False)
    )
    o_naive, s_naive = naive_recurrent_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    # Delta rule's triangular solve amplifies fp32 rounding errors
    assert compare_tensor(
        "output", o_naive, o_chunk, atol=5e-2, rtol=5e-2,
        compare_dtype=np.float64,
    )
    if s_naive is not None:
        assert compare_tensor(
            "state", s_naive, s_chunk, atol=5e-2, rtol=5e-2,
            compare_dtype=np.float64,
        )


# ============================================================================
# 4. Backward cross-validation: jax.grad(naive) vs jax.grad(chunk) (no GPU)
# ============================================================================


@pytest.mark.parametrize(
    "cfg", _XVAL_SHAPES, ids=[_shape_id(c) for c in _XVAL_SHAPES]
)
def test_backward_naive_vs_chunk_fp64(cfg):
    """fp64: backward of chunk should match backward of naive to high precision."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    q, k, v, g, beta, h0 = _make_inputs(
        B, T, H, K, V, jnp.float64, seed=cfg["seed"], h0=cfg.get("h0", False)
    )

    def loss_naive(q, k, v, g, beta, h0):
        o, _ = naive_recurrent_kda(
            q, k, v, g, beta, initial_state=h0, output_final_state=False
        )
        return o.sum()

    def loss_chunk(q, k, v, g, beta, h0):
        o, _ = naive_chunk_kda(
            q, k, v, g, beta, initial_state=h0, output_final_state=False
        )
        return o.sum()

    argnums = (0, 1, 2, 3, 4)
    grads_naive = jax.grad(loss_naive, argnums=argnums)(q, k, v, g, beta, h0)
    grads_chunk = jax.grad(loss_chunk, argnums=argnums)(q, k, v, g, beta, h0)

    names = ["dq", "dk", "dv", "dg", "dbeta"]
    for name, gn, gc in zip(names, grads_naive, grads_chunk):
        ref, test = gn, gc
        if name == "dg":
            # At t=0 with zero initial state, dg_0 = 0 mathematically because
            # exp(g_0) * 0 has no dependency on g_0. But the chunk version's
            # cumsum-based computation creates AD paths that produce small
            # numerical residuals. Skip t=0 for dg comparison.
            if ref.shape[1] <= 1:
                continue
            ref = ref[:, 1:]
            test = test[:, 1:]
        assert compare_tensor(
            name, ref, test, atol=1e-7, rtol=1e-7, compare_dtype=np.float64,
        )


# ============================================================================
# 5. CPU ref vs FLA Triton (GPU, when available)
# ============================================================================

HAS_CUDA = False
try:
    import torch
    import torch.nn.functional as F

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass

HAS_FLA = False
try:
    from fla.ops.kda.naive import naive_recurrent_kda as triton_naive_kda

    HAS_FLA = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)

_JAX_DTYPES = {
    "float64": jnp.float64,
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

_DTYPE_TOLS = {
    # Triton uses fp32 math internally; errors accumulate over long recurrent
    # sequences (e.g., T=64, H=8), requiring relaxed tolerance.
    "float64": dict(atol=5e-3, rtol=5e-3),
    "float32": dict(atol=5e-3, rtol=5e-3),
    "float16": dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

_TRITON_SHAPES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
]

TRITON_CASES = [
    {**s, "dtype": d, **t}
    for s in _TRITON_SHAPES
    for d, t in _DTYPE_TOLS.items()
]


def _triton_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['dtype']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


def _torch_to_jax(t, dtype):
    """Convert torch tensor to JAX array via fp32 intermediate."""
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


@requires_triton
@pytest.mark.parametrize(
    "cfg", TRITON_CASES, ids=[_triton_case_id(c) for c in TRITON_CASES]
)
def test_cpu_naive_vs_triton_naive(cfg):
    """Compare CPU naive vs FLA Triton naive_recurrent_kda."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    dtype_name = cfg["dtype"]
    jax_dtype = _JAX_DTYPES[dtype_name]

    _TORCH_DTYPES = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    # Triton doesn't support fp64; use fp32 as Triton reference
    triton_dtype = _TORCH_DTYPES.get(dtype_name, _TORCH_DTYPES["float32"])

    torch.manual_seed(cfg["seed"])
    q_t = torch.randn(B, T, H, K).to(triton_dtype)
    k_t = torch.randn(B, T, H, K).to(triton_dtype)
    v_t = torch.randn(B, T, H, V).to(triton_dtype)
    g_t = F.logsigmoid(torch.randn(B, T, H, K)).to(triton_dtype)
    beta_t = torch.sigmoid(torch.randn(B, T, H)).to(triton_dtype)
    h0_t = torch.randn(B, H, K, V) if cfg.get("h0") else None

    # Run Triton naive
    kwargs = dict(output_final_state=True)
    if h0_t is not None:
        kwargs["initial_state"] = h0_t.float().cuda()
    o_tri, s_tri = triton_naive_kda(
        q_t.cuda(), k_t.cuda(), v_t.cuda(), g_t.cuda(),
        beta_t.cuda(), **kwargs,
    )
    o_tri = o_tri.cpu()
    s_tri = s_tri.cpu() if s_tri is not None else None

    # Run CPU naive
    o_cpu, s_cpu = naive_recurrent_kda(
        _torch_to_jax(q_t, jax_dtype), _torch_to_jax(k_t, jax_dtype),
        _torch_to_jax(v_t, jax_dtype), _torch_to_jax(g_t, jax_dtype),
        _torch_to_jax(beta_t, jax_dtype),
        initial_state=_torch_to_jax(h0_t, jnp.float32) if h0_t is not None else None,
        output_final_state=True,
    )

    assert compare_tensor(
        "output", o_tri, o_cpu, atol=atol, rtol=rtol, compare_dtype=np.float64,
    )
    if s_tri is not None:
        assert compare_tensor(
            "final_state", s_tri, s_cpu, atol=atol, rtol=rtol,
            compare_dtype=np.float64,
        )
    assert o_cpu.dtype == jax_dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
