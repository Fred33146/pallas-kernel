from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
import numpy as np

from tops.cpu.ops.simple_gla import fused_recurrent_simple_gla as cpu_fused_recurrent_simple_gla
from tops.ops.simple_gla import fused_recurrent_simple_gla
from tops.ops.simple_gla.chunk import chunk_simple_gla


def _make_inputs(*, B, T, H, K, V, seed=0, dtype=jnp.float32):
    key = jax.random.PRNGKey(seed)
    qk, kk, vk, gk, hk = jax.random.split(key, 5)
    q = jax.random.normal(qk, (B, T, H, K), dtype=dtype)
    k = jax.random.normal(kk, (B, T, H, K), dtype=dtype)
    v = jax.random.normal(vk, (B, T, H, V), dtype=dtype)
    g = jax.nn.log_sigmoid(jax.random.normal(gk, (B, T, H), dtype=dtype))
    h0 = jax.random.normal(hk, (B, H, K, V), dtype=jnp.float32)
    return q, k, v, g, h0


def test_fused_recurrent_matches_cpu_reference_with_g():
    q, k, v, g, h0 = _make_inputs(B=2, T=5, H=3, K=16, V=8, seed=7)

    o_ref, ht_ref = cpu_fused_recurrent_simple_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True
    )
    o, ht = fused_recurrent_simple_gla(
        q, k, v, g=g, initial_state=h0, output_final_state=True
    )

    np.testing.assert_allclose(np.array(o), np.array(o_ref), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(ht), np.array(ht_ref), atol=1e-5, rtol=1e-5)


def test_fused_recurrent_matches_cpu_reference_with_g_gamma_and_varlen():
    B, T, H, K, V = 1, 7, 2, 8, 4
    q, k, v, _g, _h0 = _make_inputs(B=B, T=T, H=H, K=K, V=V, seed=11)
    g_gamma = jnp.array([-0.3, -0.7], dtype=jnp.float32)
    cu_seqlens = np.array([0, 3, 7], dtype=np.int32)
    h0 = jnp.arange(2 * H * K * V, dtype=jnp.float32).reshape(2, H, K, V) / 100

    o_ref, ht_ref = cpu_fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )
    o, ht = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )

    np.testing.assert_allclose(np.array(o), np.array(o_ref), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(ht), np.array(ht_ref), atol=1e-5, rtol=1e-5)


def test_fused_recurrent_varlen_jit_matches_eager():
    B, T, H, K, V = 1, 7, 2, 8, 4
    q, k, v, _g, _h0 = _make_inputs(B=B, T=T, H=H, K=K, V=V, seed=17)
    g_gamma = jnp.array([-0.25, -0.55], dtype=jnp.float32)
    cu_seqlens = jnp.array([0, 2, 5, 7], dtype=jnp.int32)
    h0 = jax.random.normal(jax.random.PRNGKey(99), (3, H, K, V), dtype=jnp.float32)

    eager_o, eager_ht = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )
    jit_fn = jax.jit(
        lambda q, k, v, g_gamma, h0, cu: fused_recurrent_simple_gla(
            q, k, v, g_gamma=g_gamma, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )
    )
    jit_o, jit_ht = jit_fn(q, k, v, g_gamma, h0, cu_seqlens)

    np.testing.assert_allclose(np.array(jit_o), np.array(eager_o), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(jit_ht), np.array(eager_ht), atol=1e-5, rtol=1e-5)


def test_fused_recurrent_decode_avoids_chunk_padding_decay():
    B, T, H, K, V = 1, 1, 2, 128, 128
    q, k, v, _g, h0 = _make_inputs(B=B, T=T, H=H, K=K, V=V, seed=23)
    q2, k2, v2, _g2, _ = _make_inputs(B=B, T=T, H=H, K=K, V=V, seed=24)
    g_gamma = jnp.array([-0.1, -0.3], dtype=jnp.float32)

    o_recurrent, ht_recurrent = fused_recurrent_simple_gla(
        q, k, v, g_gamma=g_gamma, initial_state=h0, output_final_state=True
    )
    _o_chunk, ht_chunk = chunk_simple_gla(
        q, k, v, g_gamma, initial_state=h0, output_final_state=True
    )

    np.testing.assert_allclose(
        np.array(o_recurrent),
        np.array(cpu_fused_recurrent_simple_gla(
            q, k, v, g_gamma=g_gamma, initial_state=h0, output_final_state=True
        )[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not np.allclose(np.array(ht_recurrent), np.array(ht_chunk), atol=1e-4, rtol=1e-4)

    o_next_good, _ = fused_recurrent_simple_gla(
        q2, k2, v2, g_gamma=g_gamma, initial_state=ht_recurrent, output_final_state=True
    )
    o_next_bad, _ = fused_recurrent_simple_gla(
        q2, k2, v2, g_gamma=g_gamma, initial_state=ht_chunk, output_final_state=True
    )
    assert not np.allclose(np.array(o_next_good), np.array(o_next_bad), atol=1e-4, rtol=1e-4)
