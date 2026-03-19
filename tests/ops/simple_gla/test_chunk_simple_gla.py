"""chunk_simple_gla_fwd: FLA Triton GPU vs JAX chunk implementation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np

from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_chunk_available = False
try:
    from fla.ops.simple_gla import chunk_simple_gla as triton_chunk_simple_gla

    triton_chunk_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_chunk_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Test configs
#
# Constraints for JAX chunk path:
#   - K, V must be multiples of 128 (chunk_fwd_h_kernel Pallas requirement)
#   - T must be a multiple of chunk_size (default 64)
#   - gate: only "g_gamma" or "none" (JAX chunk_fwd_h does not support scalar g yet)
# ============================================================================

CHUNK_SIZE = 64

CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7),
    # ── with h0 ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, h0=True),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13, h0=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7, h0=True),
    # ── single head ──
    dict(B=2, T=64, H=1, K=128, V=128, seed=10),
    # ── K != V (both multiples of 128) ──
    dict(B=2, T=64, H=4, K=128, V=256, seed=20),
    dict(B=2, T=64, H=4, K=256, V=128, seed=21),
    # ── minimal T = chunk_size ──
    dict(B=2, T=64, H=2, K=128, V=128, seed=30),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=128, V=128, seed=50),
    # ── many heads ──
    dict(B=1, T=128, H=16, K=128, V=128, seed=60),
    # ── no gate ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=100, gate="none"),
    dict(B=2, T=128, H=4, K=128, V=128, seed=101, gate="none"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=102, gate="none", h0=True),
    # ── g_gamma only ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=150, gate="g_gamma"),
    dict(B=2, T=128, H=4, K=128, V=128, seed=151, gate="g_gamma"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=152, gate="g_gamma", h0=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=153, gate="g_gamma"),
    dict(B=1, T=64, H=2, K=128, V=256, seed=154, gate="g_gamma"),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=200, scale=0.1),
    dict(B=2, T=64, H=4, K=128, V=128, seed=201, scale=0.1, h0=True),
    # ── longer sequences ──
    dict(B=1, T=512, H=2, K=128, V=128, seed=300),
    dict(B=1, T=512, H=2, K=128, V=128, seed=301, h0=True),
    dict(B=1, T=512, H=2, K=128, V=128, seed=302, gate="g_gamma"),
    dict(B=1, T=512, H=2, K=128, V=128, seed=303, gate="g_gamma", h0=True),
    # ── multi-batch + long ──
    dict(B=4, T=256, H=2, K=128, V=128, seed=360),
    dict(B=2, T=256, H=4, K=128, V=128, seed=361, gate="g_gamma"),
]


def _case_id(c):
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


def _torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jnp.array(t.detach().cpu().float().numpy()), cpu_device)


def _run_triton(q, k, v, *, g=None, g_gamma=None, h0=None, scale=None):
    assert DEVICE == "cuda", "Triton implementation requires CUDA"
    kwargs = dict(output_final_state=True)
    if g is not None:
        kwargs["g"] = g.to(DEVICE)
    if g_gamma is not None:
        kwargs["g_gamma"] = g_gamma.to(DEVICE)
    if h0 is not None:
        kwargs["initial_state"] = h0.to(DEVICE)
    if scale is not None:
        kwargs["scale"] = scale
    o, ht = triton_chunk_simple_gla(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), **kwargs
    )
    return o.cpu(), ht.cpu() if ht is not None else None


def _run_jax_chunk(q, k, v, *, g_gamma=None, h0=None, scale=None,
                   chunk_size=CHUNK_SIZE):
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    g_gamma_j = _torch_to_jax(g_gamma) if g_gamma is not None else None
    h0_j = _torch_to_jax(h0) if h0 is not None else None

    # Use interpret=True on CPU (Pallas kernels require it)
    interpret = jax.default_backend() == "cpu"
    o, ht = chunk_simple_gla_fwd(
        q_j, k_j, v_j,
        g=None,
        g_gamma=g_gamma_j,
        scale=scale,
        h0=h0_j,
        use_ht=True,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Parametrized test — Triton chunk (gold) vs JAX chunk
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_triton_chunk_vs_jax_chunk(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 5e-2)
    rtol = cfg.get("rtol", 1e-2)
    gate = cfg.get("gate", "none")
    scale = cfg.get("scale", None)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)

    g_gamma = None
    if gate == "g_gamma":
        g_gamma = F.logsigmoid(torch.randn(H))

    N = B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None

    o_tri, ht_tri = _run_triton(q, k, v, g=None, g_gamma=g_gamma, h0=h0, scale=scale)
    o_jax, ht_jax = _run_jax_chunk(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale)

    assert compare_tensor("output", o_tri, o_jax, atol=atol, rtol=rtol)
    if ht_tri is not None and ht_jax is not None:
        assert compare_tensor("final_state", ht_tri, ht_jax, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
