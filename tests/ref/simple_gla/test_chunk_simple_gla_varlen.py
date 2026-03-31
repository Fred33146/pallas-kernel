"""chunk_simple_gla varlen: verify cu_seqlens-based variable-length processing
matches independent per-sequence batch execution.

Tests:
  1. CPU-only: varlen forward vs independent batch (no GPU needed)
  2. CPU-only: varlen forward with initial state h0
  3. CPU-only: varlen backward vs per-sequence jax.grad(naive) (fp64)
  4. CPU-only: varlen backward with h0
  5. GPU: CPU ref varlen vs FLA Triton varlen (requires CUDA + FLA)
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
import jax.numpy as jnp
import pytest
from tops.cpu.ops.simple_gla import chunk_simple_gla, naive_simple_gla
from tops.cpu.ops.simple_gla.chunk import chunk_simple_gla_bwd
from tests.utils import compare_tensor

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass

try:
    from fla.ops.simple_gla import chunk_simple_gla as triton_chunk_simple_gla

    HAS_FLA = True
except ImportError:
    HAS_FLA = False

requires_cuda_and_fla = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)

# ============================================================================
# Configs
# ============================================================================

_VARLEN_SHAPES = [
    # ── equal segments, chunk-aligned ──
    dict(T=128, H=2, K=32, V=64, cu_seqlens=[0, 64, 128], C=64, seed=42),
    # ── unequal segments, chunk-aligned ──
    dict(T=192, H=4, K=32, V=64, cu_seqlens=[0, 64, 192], C=64, seed=7),
    # ── non-chunk-aligned (triggers per-segment padding) ──
    dict(T=100, H=2, K=32, V=64, cu_seqlens=[0, 30, 100], C=64, seed=11),
    # ── single token segment ──
    dict(T=65, H=2, K=32, V=64, cu_seqlens=[0, 1, 65], C=64, seed=400),
]

_DTYPE_TOLS = {
    "float32":  dict(atol=5e-5, rtol=5e-5),
    "float16":  dict(atol=5e-3, rtol=5e-3),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

GATE_MODES = ["g", "g_gamma", "none"]
GPU_GATE_MODES = ["g", "none"]  # g_gamma not in FLA's simple_gla API with cu_seqlens

GPU_VARLEN_CASES = [
    {**s, "dtype": d, "gate": gm, **t}
    for s in _VARLEN_SHAPES
    for d, t in _DTYPE_TOLS.items()
    for gm in GPU_GATE_MODES
]

# Backward GPU comparison: fp32 only, g mode only (g_gamma not in FLA varlen).
# bfloat16 excluded: FLA Triton backward kernel (chunk_bwd_kernel_dqkwg)
# triggers CUDA_ERROR_ILLEGAL_ADDRESS during autotuner with bfloat16.
_BWD_DTYPE_TOLS = {
    "float32": dict(atol=5e-4, rtol=5e-4),
    # "bfloat16": dict(atol=5e-1, rtol=5e-1),  # excluded: CUDA_ERROR_ILLEGAL_ADDRESS with bfloat16
}

GPU_BWD_VARLEN_CASES = [
    {**s, "dtype": d, **t}
    for s in _VARLEN_SHAPES
    for d, t in _BWD_DTYPE_TOLS.items()
]


def _varlen_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}"


def _gpu_case_id(c):
    n_segs = len(c["cu_seqlens"]) - 1
    return f"T{c['T']}_segs{n_segs}_{c['dtype']}_{c['gate']}"


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t, dtype):
    return jnp.array(t.detach().cpu().float().numpy(), dtype=dtype)


def _run_independent_batch(
    q, k, v, cu_seqlens, chunk_size,
    g=None, g_gamma=None, initial_state=None, output_final_state=False,
):
    """Run chunk_simple_gla independently per sequence and concatenate results."""
    N = len(cu_seqlens) - 1
    outputs = []
    final_states = []
    for n in range(N):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        h0_n = initial_state[n : n + 1] if initial_state is not None else None
        g_n = g[:, bos:eos] if g is not None else None
        o_n, ht_n = chunk_simple_gla(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g=g_n,
            g_gamma=g_gamma,
            initial_state=h0_n,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        outputs.append(o_n)
        if ht_n is not None:
            final_states.append(ht_n[0])
    o_cat = jnp.concatenate(outputs, axis=1)
    ht_cat = jnp.stack(final_states, axis=0) if final_states else None
    return o_cat, ht_cat


# ============================================================================
# 1. CPU-only: varlen vs independent batch (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("gate", GATE_MODES)
@pytest.mark.parametrize("cfg", _VARLEN_SHAPES, ids=[_varlen_id(c) for c in _VARLEN_SHAPES])
def test_simple_gla_varlen_fwd(cfg, gate):
    """Varlen forward should match independent per-sequence execution."""
    T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
    cu = jnp.array(cfg["cu_seqlens"])
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(keys[0], (1, T, H, K))
    k = jax.random.normal(keys[1], (1, T, H, K))
    v = jax.random.normal(keys[2], (1, T, H, V))

    g = jax.random.normal(keys[3], (1, T, H)) * 0.1 if gate == "g" else None
    g_gamma = jnp.full((H,), -0.05, dtype=jnp.float32) if gate == "g_gamma" else None

    o_var, _ = chunk_simple_gla(
        q, k, v, g=g, g_gamma=g_gamma, cu_seqlens=cu, chunk_size=C
    )
    o_ind, _ = _run_independent_batch(q, k, v, cu, C, g=g, g_gamma=g_gamma)

    assert compare_tensor("output", o_var, o_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)


# ============================================================================
# 2. CPU-only: varlen with initial state h0
# ============================================================================


def test_simple_gla_varlen_with_h0():
    """Varlen forward with initial state should match independent execution."""
    H, K, V, C = 2, 32, 64, 64
    cu = jnp.array([0, 64, 128])
    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q = jax.random.normal(keys[0], (1, 128, H, K))
    k = jax.random.normal(keys[1], (1, 128, H, K))
    v = jax.random.normal(keys[2], (1, 128, H, V))
    g = jax.random.normal(keys[3], (1, 128, H)) * 0.1
    h0 = jax.random.normal(keys[4], (2, H, K, V))

    o_var, ht_var = chunk_simple_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu, chunk_size=C
    )
    o_ind, ht_ind = _run_independent_batch(
        q, k, v, cu, C, g=g, initial_state=h0, output_final_state=True
    )

    assert compare_tensor("output", o_var, o_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)
    assert ht_var.shape == (2, H, K, V)
    assert compare_tensor("final_state", ht_var, ht_ind, atol=1e-5, rtol=1e-5, compare_dtype=np.float64)


# # ============================================================================
# # 3. CPU-only: varlen backward vs per-sequence jax.grad(naive) (fp64)
# # ============================================================================


# @pytest.mark.parametrize("gate", GATE_MODES)
# @pytest.mark.parametrize("cfg", _VARLEN_SHAPES, ids=[_varlen_id(c) for c in _VARLEN_SHAPES])
# def test_simple_gla_varlen_bwd(cfg, gate):
#     """Varlen chunk backward should match per-sequence jax.grad(naive) at fp64.

#     This is the varlen counterpart of test_chunk_bwd_vs_autograd_fp64 in
#     test_chunk_simple_gla.py.  Reference gradients are computed by running
#     jax.grad(naive_simple_gla) independently on each sequence, then
#     concatenating.  The chunk backward is run once with cu_seqlens.

#     Config 3 (T=100, segs=[30,70], C=64) exercises partial chunks, which is
#     critical for the dg_last placement fix.
#     """
#     T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
#     cu = jnp.array(cfg["cu_seqlens"])
#     N = len(cfg["cu_seqlens"]) - 1

#     keys = jax.random.split(jax.random.PRNGKey(seed), 5)
#     q = jax.random.normal(keys[0], (1, T, H, K), dtype=jnp.float64)
#     k = jax.random.normal(keys[1], (1, T, H, K), dtype=jnp.float64)
#     v = jax.random.normal(keys[2], (1, T, H, V), dtype=jnp.float64)

#     g = jax.nn.log_sigmoid(
#         jax.random.normal(keys[3], (1, T, H), dtype=jnp.float64)
#     ) if gate == "g" else None
#     g_gamma = -jax.nn.softplus(
#         jax.random.normal(keys[4], (H,), dtype=jnp.float64)
#     ) if gate == "g_gamma" else None

#     # Reference: per-sequence jax.grad(naive_simple_gla)
#     ref_dq_parts, ref_dk_parts, ref_dv_parts, ref_dg_parts = [], [], [], []
#     for n in range(N):
#         bos, eos = int(cu[n]), int(cu[n + 1])
#         q_n, k_n, v_n = q[:, bos:eos], k[:, bos:eos], v[:, bos:eos]
#         g_n = g[:, bos:eos] if g is not None else None

#         def loss_fn(q_, k_, v_, g_):
#             o, _ = naive_simple_gla(
#                 q_, k_, v_, g=g_, g_gamma=g_gamma, output_final_state=False,
#             )
#             return o.sum()

#         argnums = [0, 1, 2]
#         if g_n is not None:
#             argnums.append(3)
#         grads = jax.grad(loss_fn, argnums=argnums)(q_n, k_n, v_n, g_n)

#         ref_dq_parts.append(grads[0])
#         ref_dk_parts.append(grads[1])
#         ref_dv_parts.append(grads[2])
#         if g_n is not None:
#             ref_dg_parts.append(grads[3])

#     ref_dq = jnp.concatenate(ref_dq_parts, axis=1)
#     ref_dk = jnp.concatenate(ref_dk_parts, axis=1)
#     ref_dv = jnp.concatenate(ref_dv_parts, axis=1)
#     ref_dg = jnp.concatenate(ref_dg_parts, axis=1) if ref_dg_parts else None

#     # Chunk backward with cu_seqlens
#     scale = K ** -0.5
#     do = jnp.ones((1, T, H, V), dtype=jnp.float64)
#     dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
#         q, k, v, g, g_gamma, None, do=do, dht=None, scale=scale,
#         chunk_size=C, cu_seqlens=cu,
#     )

#     assert compare_tensor("dq", ref_dq, dq, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     assert compare_tensor("dk", ref_dk, dk, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     assert compare_tensor("dv", ref_dv, dv, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     if g is not None:
#         assert compare_tensor("dg", ref_dg, dg, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     else:
#         assert dg is None


# # ============================================================================
# # 4. CPU-only: varlen backward with initial state h0
# # ============================================================================


# def test_simple_gla_varlen_bwd_with_h0():
#     """Varlen backward with h0 should match per-sequence jax.grad(naive)."""
#     H, K, V, C = 2, 32, 64, 64
#     cu = jnp.array([0, 64, 128])
#     N = 2

#     keys = jax.random.split(jax.random.PRNGKey(99), 5)
#     q = jax.random.normal(keys[0], (1, 128, H, K), dtype=jnp.float64)
#     k = jax.random.normal(keys[1], (1, 128, H, K), dtype=jnp.float64)
#     v = jax.random.normal(keys[2], (1, 128, H, V), dtype=jnp.float64)
#     g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (1, 128, H), dtype=jnp.float64))
#     h0 = jax.random.normal(keys[4], (N, H, K, V), dtype=jnp.float64)

#     # Reference: per-sequence jax.grad(naive) including h0
#     ref_dq_parts, ref_dk_parts, ref_dv_parts, ref_dg_parts = [], [], [], []
#     for n in range(N):
#         bos, eos = int(cu[n]), int(cu[n + 1])
#         q_n, k_n, v_n = q[:, bos:eos], k[:, bos:eos], v[:, bos:eos]
#         g_n = g[:, bos:eos]
#         h0_n = h0[n : n + 1]

#         def loss_fn(q_, k_, v_, g_):
#             o, _ = naive_simple_gla(
#                 q_, k_, v_, g=g_, initial_state=h0_n, output_final_state=False,
#             )
#             return o.sum()

#         grads = jax.grad(loss_fn, argnums=[0, 1, 2, 3])(q_n, k_n, v_n, g_n)
#         ref_dq_parts.append(grads[0])
#         ref_dk_parts.append(grads[1])
#         ref_dv_parts.append(grads[2])
#         ref_dg_parts.append(grads[3])

#     ref_dq = jnp.concatenate(ref_dq_parts, axis=1)
#     ref_dk = jnp.concatenate(ref_dk_parts, axis=1)
#     ref_dv = jnp.concatenate(ref_dv_parts, axis=1)
#     ref_dg = jnp.concatenate(ref_dg_parts, axis=1)

#     # Chunk backward with cu_seqlens + h0
#     scale = K ** -0.5
#     do = jnp.ones((1, 128, H, V), dtype=jnp.float64)
#     dq, dk, dv, dg, dh0_out = chunk_simple_gla_bwd(
#         q, k, v, g, None, h0, do=do, dht=None, scale=scale,
#         chunk_size=C, cu_seqlens=cu,
#     )

#     assert compare_tensor("dq", ref_dq, dq, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     assert compare_tensor("dk", ref_dk, dk, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     assert compare_tensor("dv", ref_dv, dv, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
#     assert compare_tensor("dg", ref_dg, dg, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)

# ============================================================================
# 3. CPU-only: varlen backward vs per-sequence jax.grad(naive) (fp64)
# ============================================================================


@pytest.mark.parametrize("gate", GATE_MODES)
@pytest.mark.parametrize("cfg", _VARLEN_SHAPES, ids=[_varlen_id(c) for c in _VARLEN_SHAPES])
def test_simple_gla_varlen_bwd(cfg, gate):
    """Varlen chunk backward should match per-sequence jax.grad(naive) at fp64.

    This is the varlen counterpart of test_chunk_bwd_vs_autograd_fp64 in
    test_chunk_simple_gla.py.  Reference gradients are computed by running
    jax.grad(naive_simple_gla) independently on each sequence, then
    concatenating.  The chunk backward is run once with cu_seqlens.

    Config 3 (T=100, segs=[30,70], C=64) exercises partial chunks, which is
    critical for the dg_last placement fix.
    """
    T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
    cu = jnp.array(cfg["cu_seqlens"])
    N = len(cfg["cu_seqlens"]) - 1

    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(keys[0], (1, T, H, K), dtype=jnp.float64)
    k = jax.random.normal(keys[1], (1, T, H, K), dtype=jnp.float64)
    v = jax.random.normal(keys[2], (1, T, H, V), dtype=jnp.float64)

    g = jax.nn.log_sigmoid(
        jax.random.normal(keys[3], (1, T, H), dtype=jnp.float64)
    ) if gate == "g" else None
    g_gamma = -jax.nn.softplus(
        jax.random.normal(keys[4], (H,), dtype=jnp.float64)
    ) if gate == "g_gamma" else None

    # Reference: per-sequence jax.grad(naive_simple_gla)
    ref_dq_parts, ref_dk_parts, ref_dv_parts, ref_dg_parts = [], [], [], []
    for n in range(N):
        bos, eos = int(cu[n]), int(cu[n + 1])
        q_n, k_n, v_n = q[:, bos:eos], k[:, bos:eos], v[:, bos:eos]
        g_n = g[:, bos:eos] if g is not None else None

        def loss_fn(q_, k_, v_, g_):
            o, _ = naive_simple_gla(
                q_, k_, v_, g=g_, g_gamma=g_gamma, output_final_state=False,
            )
            return o.sum()

        argnums = [0, 1, 2]
        if g_n is not None:
            argnums.append(3)
        grads = jax.grad(loss_fn, argnums=argnums)(q_n, k_n, v_n, g_n)

        ref_dq_parts.append(grads[0])
        ref_dk_parts.append(grads[1])
        ref_dv_parts.append(grads[2])
        if g_n is not None:
            ref_dg_parts.append(grads[3])

    ref_dq = jnp.concatenate(ref_dq_parts, axis=1)
    ref_dk = jnp.concatenate(ref_dk_parts, axis=1)
    ref_dv = jnp.concatenate(ref_dv_parts, axis=1)
    ref_dg = jnp.concatenate(ref_dg_parts, axis=1) if ref_dg_parts else None

    # Chunk backward with cu_seqlens
    scale = K ** -0.5
    do = jnp.ones((1, T, H, V), dtype=jnp.float64)
    dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
        q, k, v, g, g_gamma, None, do=do, dht=None, scale=scale,
        chunk_size=C, cu_seqlens=cu,
    )

    assert compare_tensor("dq", ref_dq, dq, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    assert compare_tensor("dk", ref_dk, dk, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    assert compare_tensor("dv", ref_dv, dv, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    if g is not None:
        assert compare_tensor("dg", ref_dg, dg, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    else:
        assert dg is None


# ============================================================================
# 4. CPU-only: varlen backward with initial state h0
# ============================================================================


def test_simple_gla_varlen_bwd_with_h0():
    """Varlen backward with h0 should match per-sequence jax.grad(naive)."""
    H, K, V, C = 2, 32, 64, 64
    cu = jnp.array([0, 64, 128])
    N = 2

    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q = jax.random.normal(keys[0], (1, 128, H, K), dtype=jnp.float64)
    k = jax.random.normal(keys[1], (1, 128, H, K), dtype=jnp.float64)
    v = jax.random.normal(keys[2], (1, 128, H, V), dtype=jnp.float64)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (1, 128, H), dtype=jnp.float64))
    h0 = jax.random.normal(keys[4], (N, H, K, V), dtype=jnp.float64)

    # Reference: per-sequence jax.grad(naive) including h0
    ref_dq_parts, ref_dk_parts, ref_dv_parts, ref_dg_parts = [], [], [], []
    for n in range(N):
        bos, eos = int(cu[n]), int(cu[n + 1])
        q_n, k_n, v_n = q[:, bos:eos], k[:, bos:eos], v[:, bos:eos]
        g_n = g[:, bos:eos]
        h0_n = h0[n : n + 1]

        def loss_fn(q_, k_, v_, g_):
            o, _ = naive_simple_gla(
                q_, k_, v_, g=g_, initial_state=h0_n, output_final_state=False,
            )
            return o.sum()

        grads = jax.grad(loss_fn, argnums=[0, 1, 2, 3])(q_n, k_n, v_n, g_n)
        ref_dq_parts.append(grads[0])
        ref_dk_parts.append(grads[1])
        ref_dv_parts.append(grads[2])
        ref_dg_parts.append(grads[3])

    ref_dq = jnp.concatenate(ref_dq_parts, axis=1)
    ref_dk = jnp.concatenate(ref_dk_parts, axis=1)
    ref_dv = jnp.concatenate(ref_dv_parts, axis=1)
    ref_dg = jnp.concatenate(ref_dg_parts, axis=1)

    # Chunk backward with cu_seqlens + h0
    scale = K ** -0.5
    do = jnp.ones((1, 128, H, V), dtype=jnp.float64)
    dq, dk, dv, dg, dh0_out = chunk_simple_gla_bwd(
        q, k, v, g, None, h0, do=do, dht=None, scale=scale,
        chunk_size=C, cu_seqlens=cu,
    )

    assert compare_tensor("dq", ref_dq, dq, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    assert compare_tensor("dk", ref_dk, dk, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    assert compare_tensor("dv", ref_dv, dv, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)
    assert compare_tensor("dg", ref_dg, dg, atol=1e-8, rtol=1e-8, compare_dtype=np.float64)


# ============================================================================
# 5. GPU: CPU ref varlen vs FLA Triton varlen
# ============================================================================


@requires_cuda_and_fla
@pytest.mark.parametrize("cfg", GPU_VARLEN_CASES, ids=[_gpu_case_id(c) for c in GPU_VARLEN_CASES])
def test_simple_gla_varlen_cpu_vs_triton(cfg):
    """Compare CPU ref chunk_simple_gla(cu_seqlens=...) against FLA Triton."""
    T, H, K, V, C = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"]
    seed = cfg["seed"]
    gate = cfg["gate"]
    atol, rtol = cfg["atol"], cfg["rtol"]
    torch_dtype = getattr(torch, cfg["dtype"])
    jax_dtype = getattr(jnp, cfg["dtype"])
    cu_list = cfg["cu_seqlens"]

    torch.manual_seed(seed)
    q_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    k_t = torch.randn(1, T, H, K, dtype=torch_dtype)
    v_t = torch.randn(1, T, H, V, dtype=torch_dtype)
    g_t = torch.randn(1, T, H, dtype=torch_dtype) * 0.1 if gate == "g" else None

    cu_t = torch.tensor(cu_list, dtype=torch.long)

    # FLA Triton with cu_seqlens
    o_tri, ht_tri = triton_chunk_simple_gla(
        q_t.to("cuda"), k_t.to("cuda"), v_t.to("cuda"),
        g=g_t.to("cuda") if g_t is not None else None,
        cu_seqlens=cu_t.to("cuda"),
        output_final_state=True,
    )


# ============================================================================
# 6. GPU: CPU ref varlen backward vs FLA Triton varlen backward
# ============================================================================


def _gpu_bwd_case_id(c):
  n_segs = len(c["cu_seqlens"]) - 1
  return f"T{c['T']}_segs{n_segs}_{c['dtype']}"


@requires_cuda_and_fla
@pytest.mark.parametrize(
  "cfg", GPU_BWD_VARLEN_CASES, ids=[_gpu_bwd_case_id(c) for c in GPU_BWD_VARLEN_CASES]
)
def test_simple_gla_varlen_bwd_cpu_vs_triton(cfg):
  """Compare CPU ref varlen backward vs FLA Triton backward (via autograd).

  Only tests g mode (FLA's varlen API does not support g_gamma with cu_seqlens).
  Triton gradients are obtained via loss.backward(); CPU gradients come from
  chunk_simple_gla_bwd with cu_seqlens.
  """
  T, H, K, V, C = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"]
  seed = cfg["seed"]
  atol, rtol = cfg["atol"], cfg["rtol"]
  torch_dtype = getattr(torch, cfg["dtype"])
  jax_dtype = getattr(jnp, cfg["dtype"])
  cu_list = cfg["cu_seqlens"]

  # Free GPU memory from prior tests / Triton autotune cache
  torch.cuda.empty_cache()

  torch.manual_seed(seed)
  q_t = torch.randn(1, T, H, K, dtype=torch_dtype)
  k_t = torch.randn(1, T, H, K, dtype=torch_dtype)
  v_t = torch.randn(1, T, H, V, dtype=torch_dtype)
  do_t = torch.randn(1, T, H, V, dtype=torch_dtype)
  g_t = torch.randn(1, T, H, dtype=torch_dtype) * 0.1

  cu_t = torch.tensor(cu_list, dtype=torch.long)

  # ── Triton forward + backward via autograd ──
  q_g = q_t.clone().cuda().requires_grad_()
  k_g = k_t.clone().cuda().requires_grad_()
  v_g = v_t.clone().cuda().requires_grad_()
  g_g = g_t.clone().cuda().requires_grad_()
  do_g = do_t.clone().cuda()

  o_g, _ = triton_chunk_simple_gla(
    q_g, k_g, v_g, g=g_g,
    cu_seqlens=cu_t.cuda(), output_final_state=False,
  )
  loss = (o_g * do_g).sum()
  loss.backward()

  dq_tri = q_g.grad.cpu()
  dk_tri = k_g.grad.cpu()
  dv_tri = v_g.grad.cpu()
  dg_tri = g_g.grad.cpu()

  # ── CPU JAX backward ──
  q_j = _torch_to_jax(q_t, jax_dtype)
  k_j = _torch_to_jax(k_t, jax_dtype)
  v_j = _torch_to_jax(v_t, jax_dtype)
  do_j = _torch_to_jax(do_t, jax_dtype)
  g_j = _torch_to_jax(g_t, jax_dtype)
  cu_j = jnp.array(cu_list)

  scale = K**-0.5
  dq_j, dk_j, dv_j, dg_j, _ = chunk_simple_gla_bwd(
    q_j, k_j, v_j, g_j, None, None,
    do=do_j, dht=None, scale=scale, chunk_size=C, cu_seqlens=cu_j,
  )

  assert compare_tensor(
    "dq", dq_tri.float().numpy(), np.array(dq_j, dtype=np.float32),
    atol=atol, rtol=rtol,
  )
  assert compare_tensor(
    "dk", dk_tri.float().numpy(), np.array(dk_j, dtype=np.float32),
    atol=atol, rtol=rtol,
  )
  assert compare_tensor(
    "dv", dv_tri.float().numpy(), np.array(dv_j, dtype=np.float32),
    atol=atol, rtol=rtol,
  )
  # dg needs wider tolerance: at chunk/segment boundaries the gradient
  # value is near-zero, and Triton vs JAX CPU accumulation order for
  # dg_last (sum(h*dh) + reverse cumsum) causes rounding divergence.
  assert compare_tensor(
    "dg", dg_tri.float().numpy(), np.array(dg_j, dtype=np.float32),
    atol=max(atol, 1e-3), rtol=rtol,
  )
