"""chunk_gla_bwd_with_pl: end-to-end Pallas orchestrator vs Jax reference orchestrator tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from src.ops.gla.chunk import chunk_gla_bwd, chunk_gla_bwd_with_pl
from tests.utils import compare_tensor

CASES = [
    dict(B=2, T=128, H=2, K=128, V=128, seed=42, chunk_size=64),
    # dict(B=1, T=256, H=4, K=64, V=64, seed=13, chunk_size=64),
    # dict(B=2, T=64, H=1, K=32, V=32, seed=14, chunk_size=32),
    dict(B=1, T=128, H=2, K=128, V=128, seed=15, h0=True, chunk_size=64),
    dict(B=1, T=128, H=2, K=128, V=128, seed=16, dht=True, chunk_size=64),
    dict(B=1, T=128, H=2, K=128, V=128, seed=17, h0=True, dht=True, chunk_size=64),
]

def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_chunk{c['chunk_size']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_chunk_gla_bwd_with_pl(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    chunk_size = cfg.get("chunk_size", 64)
    scale = cfg.get("scale", K**-0.5)
    
    key = jax.random.PRNGKey(cfg["seed"])
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    
    q = jax.random.normal(k1, (B, T, H, K))
    k = jax.random.normal(k2, (B, T, H, K))
    v = jax.random.normal(k3, (B, T, H, V))
    g = jax.nn.log_sigmoid(jax.random.normal(k4, (B, T, H, K)))
    do = jax.random.normal(k5, (B, T, H, V))
    
    h0 = jax.random.normal(k6, (B, H, K, V)) if cfg.get("h0") else None
    dht = jax.random.normal(k7, (B, H, K, V)) if cfg.get("dht") else None

    # Reference JAX version
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dh0 = chunk_gla_bwd(
        q, k, v, g, 
        g_cumsum=None, scale=scale, 
        initial_state=h0, h=None, A=None, 
        do=do, dht=dht, chunk_size=chunk_size
    )

    # Pallas version
    pl_dq, pl_dk, pl_dv, pl_dg, pl_dh0 = chunk_gla_bwd_with_pl(
        q, k, v, g, 
        g_cumsum=None, scale=scale, 
        initial_state=h0, h=None, A=None, 
        do=do, dht=dht, chunk_size=chunk_size
    )

    # Compare
    atol, rtol = 5e-2, 5e-2  # TPU/Pallas usually have some numerical differences vs pure JAX
    assert compare_tensor("dq", ref_dq, pl_dq, atol=atol, rtol=rtol)
    assert compare_tensor("dk", ref_dk, pl_dk, atol=atol, rtol=rtol)
    assert compare_tensor("dv", ref_dv, pl_dv, atol=atol, rtol=rtol)
    assert compare_tensor("dg", ref_dg, pl_dg, atol=atol, rtol=rtol)
    
    if h0 is not None or dht is not None:
        if ref_dh0 is not None and pl_dh0 is not None:
            assert compare_tensor("dh0", ref_dh0, pl_dh0, atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
