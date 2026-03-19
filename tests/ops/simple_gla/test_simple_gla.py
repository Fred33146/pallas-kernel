"""simple_gla forward: JAX naive vs FLA Triton GPU."""

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

from tops.ops.simple_gla import simple_gla_naive
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.ops.simple_gla import fused_recurrent_simple_gla as triton_fused_recurrent

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Test configs
#
# Each dict can contain:
#   B, T, H, K, V       — shape (required)
#   seed                 — random seed (required)
#   atol, rtol           — tolerance (defaults: 1e-4, 1e-5)
#   gate                 — "g" | "g_gamma" | "g+g_gamma" | "none"  (default "g")
#   h0                   — True/False, whether to use initial_state (default False)
#   scale                — float or None (default None = K^{-0.5})
#   gate_logit_normalizer — float (default 1)
# ============================================================================

CASES = [
    # ── standard shapes ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── single head ──
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # ── very short T ──
    dict(B=1, T=1, H=2, K=32, V=64, seed=30),
    dict(B=1, T=3, H=2, K=32, V=64, seed=31),
    # ── odd T ──
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    dict(B=1, T=50, H=2, K=32, V=64, seed=41),
    dict(B=1, T=100, H=2, K=32, V=64, seed=42, h0=True),
    # ── large batch ──
    dict(B=8, T=32, H=4, K=32, V=64, seed=50),
    # ── many heads ──
    dict(B=1, T=64, H=16, K=32, V=64, seed=60),
    # ── small dims ──
    dict(B=2, T=32, H=2, K=8, V=16, seed=70),
    dict(B=2, T=32, H=2, K=8, V=16, seed=71, h0=True),
    # ── various ──
    dict(B=1, T=16, H=1, K=16, V=16, seed=99),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    dict(B=2, T=48, H=4, K=32, V=32, seed=99),
    # ── no gate ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=102, gate="none"),
    # ── g_gamma only ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=150, gate="g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=151, gate="g_gamma", h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=152, gate="g_gamma"),
    dict(B=2, T=37, H=4, K=16, V=32, seed=153, gate="g_gamma"),
    dict(B=1, T=1, H=2, K=32, V=64, seed=154, gate="g_gamma"),
    # ── g + g_gamma ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=160, gate="g+g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=161, gate="g+g_gamma", h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=162, gate="g+g_gamma"),
    dict(B=2, T=37, H=4, K=16, V=32, seed=163, gate="g+g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=164, gate="g+g_gamma", gate_logit_normalizer=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=165, gate="g+g_gamma", gate_logit_normalizer=10),
    # ── g_gamma + long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=170, gate="g_gamma"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=171, gate="g+g_gamma"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=172, gate="g+g_gamma", h0=True),
    # ── custom scale ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    # ── gate_logit_normalizer ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=210, gate_logit_normalizer=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=211, gate_logit_normalizer=10),
    # ── scale + h0 ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=140, scale=0.1, h0=True),
    # ── long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=1, T=512, H=2, K=32, V=64, seed=301),
    # ── long + h0 ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "g")
    if gate != "g":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    gln = c.get("gate_logit_normalizer", 1)
    if gln != 1:
        parts.append(f"gln={gln}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert torch tensor to JAX array on CPU."""
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
    o, s = triton_fused_recurrent(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), **kwargs
    )
    return o.cpu(), s.cpu() if s is not None else None


def _run_naive(q, k, v, *, g=None, g_gamma=None, h0=None, scale=None):
    """Run JAX naive simple_gla.

    Note: FLA Triton uses g of shape [B, T, H] (scalar per head),
    while our naive uses g of shape [B, T, H, K] (per element).
    We broadcast g to [B, T, H, K] for the naive implementation.
    Similarly, FLA uses g_gamma of shape [H], while naive uses [H, K].
    """
    q_jax = _torch_to_jax(q)
    k_jax = _torch_to_jax(k)
    v_jax = _torch_to_jax(v)
    K = q.shape[-1]

    g_jax = None
    if g is not None:
        # Broadcast [B, T, H] -> [B, T, H, K]
        g_expanded = g.unsqueeze(-1).expand(*g.shape, K)
        g_jax = _torch_to_jax(g_expanded)

    g_gamma_jax = None
    if g_gamma is not None:
        # g_gamma is [H] for both Triton and naive
        g_gamma_jax = _torch_to_jax(g_gamma)

    h0_jax = None
    if h0 is not None:
        h0_jax = _torch_to_jax(h0)

    o, s = simple_gla_naive(
        q_jax, k_jax, v_jax,
        g=g_jax,
        g_gamma=g_gamma_jax,
        scale=scale,
        initial_state=h0_jax,
        output_final_state=True,
    )
    return o, s


# ============================================================================
# Main parametrized test — Triton (gold) vs JAX naive
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_triton_vs_naive_fwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-5)
    gate = cfg.get("gate", "g")
    scale = cfg.get("scale", None)
    gln = cfg.get("gate_logit_normalizer", 1)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)

    g = None
    g_gamma = None
    if gate in ("g", "g+g_gamma"):
        g = F.logsigmoid(torch.randn(B, T, H)) / gln
    if gate in ("g_gamma", "g+g_gamma"):
        g_gamma = F.logsigmoid(torch.randn(H))

    N = B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None

    o_tri, s_tri = _run_triton(q, k, v, g=g, g_gamma=g_gamma, h0=h0, scale=scale)
    o_naive, s_naive = _run_naive(q, k, v, g=g, g_gamma=g_gamma, h0=h0, scale=scale)

    assert compare_tensor("output", o_tri, o_naive, atol=atol, rtol=rtol)
    assert compare_tensor("final_state", s_tri, s_naive, atol=atol, rtol=rtol)


if __name__ == "__main__":

    pytest.main([__file__, "-v"])
