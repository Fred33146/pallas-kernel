"""Tests for chunk_fwd_h / chunk_bwd_dh with cu_seqlens."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tops.cpu.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from tops.cpu.ops.common.chunk_o import chunk_local_cumsum
from tops.cpu.ops.common.utils import gather_chunks
from tops.utils import prepare_chunk_indices, prepare_lens
from tops.utils import cdiv as _cdiv_top


def _make_inputs(key, B, T, H, K, V, dtype=jnp.float32):
    keys = jax.random.split(key, 4)
    k = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[1], (B, T, H, V), dtype=dtype)
    gk = jax.random.normal(keys[2], (B, T, H, K), dtype=dtype) * 0.1
    return k, v, gk


def _gather_flatten(tensors, cu, C):
    """Gather packed tensors into chunked layout, cumsum gates, flatten.

    Returns flattened tensors, flat_cu_seqlens, total_NT.
    The last tensor in `tensors` is treated as the gate (gk) and gets cumsummed.
    """
    chunk_indices = prepare_chunk_indices(cu, C)
    total_NT = chunk_indices.shape[0]

    results = []
    for t in tensors:
        t_c, _ = gather_chunks(t, cu, chunk_indices, C)
        results.append(t_c)

    # Build flat cu_seqlens
    lens = prepare_lens(cu)
    n_chunks_per_seq = jnp.array([int(_cdiv_top(int(l), C)) for l in lens])
    flat_cu_seqlens = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.int32),
        jnp.cumsum(n_chunks_per_seq * C),
    ])

    # cumsum the gate (last tensor) on chunked layout
    gk_c = results[-1]
    gk_cumsum_c = chunk_local_cumsum(gk_c, C, cu_seqlens=cu)

    # Flatten all to [1, total_NT*C, ...]
    flat = []
    for t_c in results[:-1]:
        flat.append(t_c.reshape(1, total_NT * C, *t_c.shape[2:]))
    flat.append(gk_cumsum_c.reshape(1, total_NT * C, *gk_cumsum_c.shape[2:]))

    return flat, flat_cu_seqlens, total_NT


def test_fwd_h_varlen_matches_independent():
    """Varlen chunk_fwd_h should match running each segment independently."""
    C = 16
    H, K, V = 2, 8, 16
    key = jax.random.PRNGKey(42)
    k, v, gk = _make_inputs(key, 1, 48, H, K, V)
    cu = jnp.array([0, 16, 48])

    (k_flat, v_flat, gk_cumsum_flat), flat_cu, _ = _gather_flatten(
        [k, v, gk], cu, C
    )

    h_var, ht_var = chunk_fwd_h(
        k_flat,
        v_flat,
        gk=gk_cumsum_flat,
        output_final_state=True,
        chunk_size=C,
        cu_seqlens=flat_cu,
    )

    # Independent: run each segment separately
    gk0 = chunk_local_cumsum(gk[:, :16], C)
    gk1 = chunk_local_cumsum(gk[:, 16:], C)
    h0_ind, ht0 = chunk_fwd_h(
        k[:, :16], v[:, :16], gk=gk0, output_final_state=True, chunk_size=C
    )
    h1_ind, ht1 = chunk_fwd_h(
        k[:, 16:], v[:, 16:], gk=gk1, output_final_state=True, chunk_size=C
    )

    h_expected = jnp.concatenate([h0_ind, h1_ind], axis=1)
    assert jnp.allclose(h_var, h_expected, atol=1e-6)

    assert ht_var.shape == (2, H, K, V)
    assert jnp.allclose(ht_var[0], ht0[0], atol=1e-6)
    assert jnp.allclose(ht_var[1], ht1[0], atol=1e-6)


def test_fwd_h_varlen_with_h0():
    """Per-sequence h0 should initialize each segment independently."""
    C = 16
    H, K, V = 2, 8, 16
    key = jax.random.PRNGKey(7)
    k, v, gk = _make_inputs(key, 1, 32, H, K, V)
    cu = jnp.array([0, 16, 32])
    h0 = jax.random.normal(jax.random.PRNGKey(99), (2, H, K, V))

    (k_flat, v_flat, gk_cumsum_flat), flat_cu, _ = _gather_flatten(
        [k, v, gk], cu, C
    )

    h_var, _ = chunk_fwd_h(
        k_flat,
        v_flat,
        gk=gk_cumsum_flat,
        h0=h0,
        chunk_size=C,
        cu_seqlens=flat_cu,
    )

    gk0 = chunk_local_cumsum(gk[:, :16], C)
    gk1 = chunk_local_cumsum(gk[:, 16:], C)
    h0_ind, _ = chunk_fwd_h(
        k[:, :16], v[:, :16], gk=gk0, h0=h0[0:1], chunk_size=C
    )
    h1_ind, _ = chunk_fwd_h(
        k[:, 16:], v[:, 16:], gk=gk1, h0=h0[1:2], chunk_size=C
    )

    h_expected = jnp.concatenate([h0_ind, h1_ind], axis=1)
    assert jnp.allclose(h_var, h_expected, atol=1e-6)


def test_fwd_h_no_cu_seqlens_unchanged():
    """Without cu_seqlens, behavior is identical to original."""
    C = 16
    H, K, V = 2, 8, 16
    key = jax.random.PRNGKey(13)
    k, v, gk = _make_inputs(key, 2, 32, H, K, V)

    gk_cumsum = chunk_local_cumsum(gk, C)

    h1, ht1 = chunk_fwd_h(
        k, v, gk=gk_cumsum, output_final_state=True, chunk_size=C, cu_seqlens=None
    )
    h2, ht2 = chunk_fwd_h(
        k, v, gk=gk_cumsum, output_final_state=True, chunk_size=C
    )

    assert jnp.allclose(h1, h2)
    assert jnp.allclose(ht1, ht2)


# === Task 4: chunk_bwd_dh tests ===


def test_bwd_dh_varlen_matches_independent():
    """Varlen chunk_bwd_dh should match running each segment independently."""
    C = 16
    H, K, V = 2, 8, 16
    key = jax.random.PRNGKey(55)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (1, 48, H, K))
    do = jax.random.normal(keys[1], (1, 48, H, V))
    gk = jax.random.normal(keys[2], (1, 48, H, K)) * 0.1
    cu = jnp.array([0, 16, 48])

    (q_flat, do_flat, gk_cumsum_flat), flat_cu, _ = _gather_flatten(
        [q, do, gk], cu, C
    )

    dh_var, _ = chunk_bwd_dh(q_flat, do_flat, gk=gk_cumsum_flat, chunk_size=C, cu_seqlens=flat_cu)

    gk0 = chunk_local_cumsum(gk[:, :16], C)
    gk1 = chunk_local_cumsum(gk[:, 16:], C)
    dh0_ind, _ = chunk_bwd_dh(q[:, :16], do[:, :16], gk=gk0, chunk_size=C)
    dh1_ind, _ = chunk_bwd_dh(q[:, 16:], do[:, 16:], gk=gk1, chunk_size=C)

    dh_expected = jnp.concatenate([dh0_ind, dh1_ind], axis=1)
    assert jnp.allclose(dh_var, dh_expected, atol=1e-6)


def test_bwd_dh_varlen_dh0():
    """Per-sequence dh0 from chunk_bwd_dh."""
    C = 16
    H, K, V = 2, 8, 16
    key = jax.random.PRNGKey(77)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (1, 32, H, K))
    do = jax.random.normal(keys[1], (1, 32, H, V))
    gk = jax.random.normal(keys[2], (1, 32, H, K)) * 0.1
    h0 = jax.random.normal(keys[3], (2, H, K, V))
    cu = jnp.array([0, 16, 32])

    (q_flat, do_flat, gk_cumsum_flat), flat_cu, _ = _gather_flatten(
        [q, do, gk], cu, C
    )

    _, dh0_var = chunk_bwd_dh(
        q_flat,
        do_flat,
        gk=gk_cumsum_flat,
        h0=h0,
        chunk_size=C,
        cu_seqlens=flat_cu,
    )

    assert dh0_var.shape == (2, H, K, V)

    gk0 = chunk_local_cumsum(gk[:, :16], C)
    gk1 = chunk_local_cumsum(gk[:, 16:], C)
    _, dh0_0 = chunk_bwd_dh(
        q[:, :16], do[:, :16], gk=gk0, h0=h0[0:1], chunk_size=C
    )
    _, dh0_1 = chunk_bwd_dh(
        q[:, 16:], do[:, 16:], gk=gk1, h0=h0[1:2], chunk_size=C
    )

    assert jnp.allclose(dh0_var[0], dh0_0[0], atol=1e-6)
    assert jnp.allclose(dh0_var[1], dh0_1[0], atol=1e-6)
