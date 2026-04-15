"""Chunk KDA backward: Pallas chunk_kda_bwd vs CPU ref chunk_kda_bwd.

Directly tests the backward orchestrator interface, without going through
custom_vjp or re-running the forward pass.

Usage:
    uv run python -m pytest tests/ops/kda/test_chunk_kda_bwd_tpu.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils import compare_tensor
from tops.ops.common.cumsum import chunk_local_cumsum_vector
from tops.ops.kda.chunk_bwd import chunk_kda_bwd as chunk_kda_bwd_pallas
from tops.cpu.ops.kda.chunk_bwd import chunk_kda_bwd as chunk_kda_bwd_cpu

_BWD_CONFIGS = [
    dict(B=1, T=64, H=2, K=64, V=64, chunk_size=64, seed=42, use_h0=False),
    dict(B=2, T=128, H=4, K=64, V=64, chunk_size=64, seed=7, use_h0=False),
    dict(B=1, T=256, H=2, K=128, V=128, chunk_size=64, seed=99, use_h0=False),
    dict(B=2, T=128, H=4, K=64, V=128, chunk_size=64, seed=13, use_h0=True),
    dict(B=1, T=128, H=2, K=64, V=64, chunk_size=64, seed=77, use_h0=True),
]


def _cfg_id(c):
    h0_tag = "_h0" if c.get("use_h0") else ""
    return f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}{h0_tag}"


def _prepare_bwd_inputs(B, T, H, K, V, chunk_size, seed, use_h0=False):
    """Generate all backward inputs including forward intermediates."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32) * 0.1
    g_raw = -jnp.abs(
        jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)
    ) * 0.3
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32)
    )
    do = jax.random.normal(keys[5], (B, T, H, V), dtype=jnp.float32) * 0.1

    h0 = None
    dht = None
    if use_h0:
        h0 = jax.random.normal(keys[6], (B, H, K, V), dtype=jnp.float32) * 0.01
        dht = jax.random.normal(keys[7], (B, H, K, V), dtype=jnp.float32) * 0.01

    # gate cumsum: g -> log2 space -> chunk-local cumsum
    _LN2 = float(jnp.log(jnp.array(2.0)))
    g_cumsum = chunk_local_cumsum_vector(
        g_raw / _LN2,
        chunk_size=chunk_size,
        head_first=False,
        output_dtype=jnp.float32,
    )

    # Aqk/Akk computed with pure JAX (no kernel dependency)
    NC = T // chunk_size
    C = chunk_size
    scale = K**-0.5

    q_c = q.reshape(B, NC, C, H, K)
    k_c = k.reshape(B, NC, C, H, K)
    g_c = g_cumsum.reshape(B, NC, C, H, K)
    beta_c = beta.reshape(B, NC, C, H)

    # exp2(g[i] - g[j]) for all pairs within chunk
    g_diff = g_c[:, :, :, None, :, :] - g_c[:, :, None, :, :, :]

    # Aqk: scale * sum_k q[i]*k[j]*exp2(g[i]-g[j]), lower triangular (i>=j)
    qg = q_c[:, :, :, None, :, :] * jnp.exp2(g_diff)
    Aqk_full = scale * jnp.sum(qg * k_c[:, :, None, :, :, :], axis=-1)
    causal = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    Aqk_full = jnp.where(causal[None, None, :, :, None], Aqk_full, 0.0)
    Aqk = Aqk_full.transpose(0, 1, 2, 4, 3).reshape(B, T, H, C)

    # Akk_raw: beta[i] * sum_k k[i]*k[j]*exp2(g[i]-g[j]), strict lower (i>j)
    kg = k_c[:, :, :, None, :, :] * jnp.exp2(g_diff)
    Akk_raw = jnp.sum(kg * k_c[:, :, None, :, :, :], axis=-1)
    Akk_raw = Akk_raw * beta_c[:, :, :, None, :]
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)
    Akk_raw = jnp.where(strict_lower[None, None, :, :, None], Akk_raw, 0.0)

    # Akk = (I - Akk_raw)^{-1}
    Akk_raw_t = Akk_raw.transpose(0, 1, 4, 2, 3)
    I_mat = jnp.eye(C, dtype=jnp.float32)
    M = I_mat[None, None, None, :, :] - Akk_raw_t
    Akk_inv = jnp.linalg.solve(M, jnp.broadcast_to(I_mat, M.shape))
    Akk = Akk_inv.transpose(0, 1, 3, 2, 4).reshape(B, T, H, C)

    return q, k, v, g_cumsum, beta, Aqk, Akk, do, h0, dht, scale


class TestChunkKDABwd:
    """chunk_kda_bwd (Pallas) vs chunk_kda_bwd (CPU ref)."""

    @pytest.mark.parametrize(
        "cfg", _BWD_CONFIGS, ids=[_cfg_id(c) for c in _BWD_CONFIGS]
    )
    def test_bwd_vs_cpu_ref(self, cfg):
        B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
        chunk_size = cfg["chunk_size"]
        use_h0 = cfg.get("use_h0", False)

        inputs = _prepare_bwd_inputs(
            B, T, H, K, V, chunk_size, cfg["seed"], use_h0
        )
        q, k, v, g_cumsum, beta, Aqk, Akk, do, h0, dht, scale = inputs

        # Pallas backward
        print("Running chunk_kda_bwd on Pallas...")
        dq_p, dk_p, dv_p, db_p, dg_p, dh0_p, _, _ = chunk_kda_bwd_pallas(
            q=q, k=k, v=v, beta=beta,
            Aqk=Aqk, Akk=Akk,
            scale=scale, initial_state=h0,
            do=do, dht=dht,
            g=g_cumsum, chunk_size=chunk_size,
        )

        # CPU ref backward (returns 8 values: dq, dk, dv, db, dg, dh0, dA_log, dbias)
        dq_r, dk_r, dv_r, db_r, dg_r, dh0_r, _, _ = chunk_kda_bwd_cpu(
            q=q, k=k, v=v, beta=beta,
            Aqk=Aqk, Akk=Akk,
            scale=scale, initial_state=h0,
            do=do, dht=dht,
            g=g_cumsum, chunk_size=chunk_size,
        )

        atol, rtol = 1e-2, 1e-2
        for name, ref, got in [
            ("dq", dq_r, dq_p),
            ("dk", dk_r, dk_p),
            ("dv", dv_r, dv_p),
            ("db", db_r, db_p),
            ("dg", dg_r, dg_p),
        ]:
            assert compare_tensor(
                name, ref, got, atol=atol, rtol=rtol, compare_dtype=np.float32
            ), f"{name} mismatch"

        if use_h0:
            assert dh0_p is not None, "dh0 should not be None when h0 is provided"
            assert compare_tensor(
                "dh0", dh0_r, dh0_p, atol=atol, rtol=rtol,
                compare_dtype=np.float32,
            ), "dh0 mismatch"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])