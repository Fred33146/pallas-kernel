from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
import pytest

from tests.utils import compare_tensor
from tops.ops.common.chunk_o import chunk_fwd_o, chunk_fwd_o_ref



CASES = [
    dict(B=2, T=64, H=8, K=32, V=64, seed=0),
    dict(B=2, T=64, H=8, K=64, V=64, seed=2, gate="g_gamma"),
]


def _case_id(cfg):
    parts = [f"B{cfg['B']}_T{cfg['T']}_H{cfg['H']}_K{cfg['K']}_V{cfg['V']}"]
    if cfg.get("gate"):
        parts.append(f"gate={cfg['gate']}")
    if cfg.get("scale") is not None:
        parts.append(f"scale={cfg['scale']}")
    return "-".join(parts)


def _make_inputs(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    C = cfg.get("chunk_size", 64)
    NT = T // C
    key = jax.random.PRNGKey(cfg["seed"])
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    h = jax.random.normal(keys[3], (B, NT, H, K, V), dtype=jnp.bfloat16)

    gate = cfg.get("gate")
    g = None
    g_gamma = None
    if gate in {"g", "both"}:
        g = jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32) * 0.1
    if gate in {"g_gamma", "both"}:
        g_gamma = -jnp.abs(jax.random.normal(keys[4], (H,), dtype=jnp.float32)) * 0.1

    return q, k, v, h, g, g_gamma


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_chunk_fwd_o_pallas_vs_ref(cfg):
    q, k, v, h, g, g_gamma = _make_inputs(cfg)
    scale = cfg.get("scale")
    chunk_size = cfg.get("chunk_size", 64)

    o_ref = chunk_fwd_o_ref(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        chunk_size=chunk_size,
    )
    o_pl = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        chunk_size=chunk_size,
    )

    assert compare_tensor("output", o_ref, o_pl, atol=5e-2, rtol=5e-2, max_ulp=4)


def test_chunk_fwd_o_varlen_matches_ref():
    chunk_size = 64
    seqlens = jnp.array([0, 64, 192, 256], dtype=jnp.int32)
    B, T, H, K, V = 1, 256, 2, 32, 64
    NT = T // chunk_size
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 6)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    h = jax.random.normal(keys[3], (B, NT, H, K, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(keys[5], (H,), dtype=jnp.float32)) * 0.1

    o_ref = chunk_fwd_o_ref(
        q=q,
        k=k,
        v=v,
        h=h,
        g=None,
        g_gamma=g_gamma,
        cu_seqlens_cpu=seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=None,
        g_gamma=g_gamma,
        cu_seqlens_cpu=seqlens,
        chunk_size=chunk_size,
    )

    assert compare_tensor("output_varlen", o_ref, o, atol=5e-2, rtol=5e-2, max_ulp=4)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
