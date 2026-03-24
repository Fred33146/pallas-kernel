"""simple_gla backward: Triton fused_recurrent vs JAX naive autodiff."""

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
# Test configs (keep T small — naive backward unrolls Python for-loops)
# ============================================================================

BWD_CASES = [
    # ── standard shapes ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    # ── single head ──
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # ── very short T ──
    dict(B=1, T=1, H=2, K=32, V=64, seed=30),
    dict(B=1, T=3, H=2, K=32, V=64, seed=31),
    # ── small dims ──
    dict(B=2, T=32, H=2, K=8, V=16, seed=70),
    dict(B=2, T=32, H=2, K=8, V=16, seed=71, h0=True),
    # ── no gate ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=102, gate="none"),
    # ── g_gamma only ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=150, gate="g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=151, gate="g_gamma", h0=True),
    # ── g + g_gamma ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=160, gate="g+g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=161, gate="g+g_gamma", h0=True),
    # ── odd T ──
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    # ── custom scale ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=140, scale=0.1, h0=True),
    # ── gate_logit_normalizer ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=210, gate_logit_normalizer=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=211, gate_logit_normalizer=10),
    # ── medium T ──
    dict(B=1, T=64, H=2, K=32, V=64, seed=7),
    dict(B=1, T=64, H=2, K=32, V=64, seed=8, gate="g_gamma"),
    dict(B=1, T=64, H=2, K=32, V=64, seed=9, gate="g+g_gamma", h0=True),
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
    """Convert torch tensor to JAX array on CPU (float32)."""
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jnp.array(t.detach().cpu().float().numpy()), cpu_device)


def _run_triton_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None, scale=None):
    """Run Triton forward + backward, return (dq, dk, dv)."""
    q_t = q.clone().to(DEVICE).requires_grad_(True)
    k_t = k.clone().to(DEVICE).requires_grad_(True)
    v_t = v.clone().to(DEVICE).requires_grad_(True)

    kwargs = dict(output_final_state=True)
    if g is not None:
        kwargs["g"] = g.to(DEVICE)
    if g_gamma is not None:
        kwargs["g_gamma"] = g_gamma.to(DEVICE)
    if h0 is not None:
        kwargs["initial_state"] = h0.to(DEVICE)
    if scale is not None:
        kwargs["scale"] = scale

    o, _ = triton_fused_recurrent(q_t, k_t, v_t, **kwargs)
    o.backward(do.to(DEVICE))

    return q_t.grad.cpu(), k_t.grad.cpu(), v_t.grad.cpu()


def _run_naive_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None, scale=None):
    """Run JAX naive forward + vjp backward, return (dq, dk, dv).

    Note: FLA Triton uses g of shape [B, T, H] (scalar per head),
    while our naive uses g of shape [B, T, H, K] (per element).
    We broadcast g to [B, T, H, K] for the naive implementation.
    g_gamma [H] is also broadcast to [B, T, H, K].
    """
    q_jax = _torch_to_jax(q)
    k_jax = _torch_to_jax(k)
    v_jax = _torch_to_jax(v)
    do_jax = _torch_to_jax(do)
    B, T, H, K = q.shape

    g_jax = None
    if g is not None:
        g_expanded = g.unsqueeze(-1).expand(B, T, H, K)
        g_jax = _torch_to_jax(g_expanded)

    g_gamma_jax = None
    if g_gamma is not None:
        g_gamma_1d = _torch_to_jax(g_gamma)  # [H]
        g_gamma_jax = jnp.broadcast_to(
            g_gamma_1d.reshape(1, 1, H, 1), (B, T, H, K)
        )

    h0_jax = _torch_to_jax(h0) if h0 is not None else None

    def fwd_fn(q, k, v):
        o, _ = simple_gla_naive(
            q, k, v,
            g=g_jax, g_gamma=g_gamma_jax,
            scale=scale, initial_state=h0_jax,
        )
        return o

    _, vjp_fn = jax.vjp(fwd_fn, q_jax, k_jax, v_jax)
    dq, dk, dv = vjp_fn(do_jax)

    return dq, dk, dv


# ============================================================================
# Parametrized test — Triton backward (gold) vs JAX naive autodiff
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_case_id(c) for c in BWD_CASES])
def test_triton_vs_naive_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    gate = cfg.get("gate", "g")
    scale = cfg.get("scale", None)
    gln = cfg.get("gate_logit_normalizer", 1)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do = torch.randn(B, T, H, V)

    g = None
    g_gamma = None
    if gate in ("g", "g+g_gamma"):
        g = F.logsigmoid(torch.randn(B, T, H)) / gln
    if gate in ("g_gamma", "g+g_gamma"):
        g_gamma = F.logsigmoid(torch.randn(H))

    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None

    dq_tri, dk_tri, dv_tri = _run_triton_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, scale=scale
    )
    dq_naive, dk_naive, dv_naive = _run_naive_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, scale=scale
    )

    assert compare_tensor("dq", dq_tri, dq_naive, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_tri, dk_naive, atol=atol, rtol=rtol)
    assert compare_tensor("dv", dv_tri, dv_naive, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
