from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

def _scan_segment(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over one dense segment."""
    if reverse:
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        if g is not None:
            g = jnp.flip(g, axis=1)

    B, _T, H, K = q.shape
    V = v.shape[-1]
    h0 = (
        initial_state
        if initial_state is not None
        else jnp.zeros((B, H, K, V), dtype=q.dtype)
    )

    q_t = jnp.swapaxes(q, 0, 1)
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)
    g_t = (
        jnp.swapaxes(g, 0, 1)
        if g is not None
        else jnp.zeros((q_t.shape[0], B, H), dtype=q.dtype)
    )
    use_g = g is not None

    def step(h, xs):
        q_i, k_i, v_i, g_i = xs
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma[None, :]
        else:
            decay = jnp.broadcast_to(g_gamma[None, :], (B, H))

        h = h * jnp.exp(decay)[:, :, None, None]
        h = h + k_i[:, :, :, None] * v_i[:, :, None, :]
        o_i = jnp.sum(h * (q_i[:, :, :, None] * scale), axis=2)
        return h, o_i

    h_final, o_t = jax.lax.scan(step, h0, (q_t, k_t, v_t, g_t))
    o = jnp.swapaxes(o_t, 0, 1)

    if reverse:
        o = jnp.flip(o, axis=1)

    return o, h_final


def _scan_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
    cu_seqlens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over packed varlen data with one JAX scan."""
    _B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    seq_starts = cu_seqlens[:-1]
    seq_ends = cu_seqlens[1:]
    token_starts = seq_starts[seq_ids]
    token_ends = seq_ends[seq_ids]

    if reverse:
        scan_order = token_ends - 1 - (token_idx - token_starts)
        reset_mask = token_idx == (token_ends - 1)
    else:
        scan_order = token_idx
        reset_mask = token_idx == token_starts

    scan_seq_ids = seq_ids[scan_order]
    q_s = q[0, scan_order]
    k_s = k[0, scan_order]
    v_s = v[0, scan_order]
    g_s = g[0, scan_order] if g is not None else jnp.zeros((T, H), dtype=q.dtype)

    h0_all = (
        initial_state
        if initial_state is not None
        else jnp.zeros((N, H, K, V), dtype=q.dtype)
    )
    use_g = g is not None

    def step(carry, xs):
        h_prev, final_states = carry
        seq_id, do_reset, q_i, k_i, v_i, g_i = xs

        h = jnp.where(do_reset, h0_all[seq_id], h_prev)
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma
        else:
            decay = g_gamma

        h = h * jnp.exp(decay)[:, None, None]
        h = h + k_i[:, :, None] * v_i[:, None, :]
        o_i = jnp.sum(h * (q_i[:, :, None] * scale), axis=1)

        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_i

    init_carry = (
        jnp.zeros((H, K, V), dtype=q.dtype),
        h0_all,
    )
    (h_last, final_states), o_scan = jax.lax.scan(
        step,
        init_carry,
        (scan_seq_ids, reset_mask[scan_order], q_s, k_s, v_s, g_s),
    )
    del h_last

    inv_order = jnp.argsort(scan_order)
    o = o_scan[inv_order][None, ...]
    return o, final_states


def fused_recurrent_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Simple GLA fused recurrent forward for decode-friendly execution.

    Args:
        q: [B, T, H, K] queries.
        k: [B, T, H, K] keys.
        v: [B, T, H, V] values.
        g: [B, T, H] optional per-token log gate.
        g_gamma: [H] optional per-head constant log decay.
        scale: Optional query scaling factor. Defaults to K ** -0.5.
        initial_state: [N, H, K, V] optional recurrent state, where N=B for dense
            mode and N=len(cu_seqlens)-1 for varlen mode.
        output_final_state: Whether to return the final recurrent state.
        reverse: Whether to process each sequence in reverse time order.
        cu_seqlens: [N+1] cumulative sequence lengths for packed varlen inputs.

    Returns:
        Tuple of output [B, T, H, V] in q.dtype and optional final state
        [N, H, K, V] in the input dtype.
    """
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T,H,V], got {v.ndim}D"

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape[:3] == q.shape[:3], f"v shape {v.shape} incompatible with q"
    assert g is not None or g_gamma is not None, (
        "At least one of g or g_gamma must be provided"
    )
    if g is not None:
        assert g.ndim == 3 and g.shape == (B, T, H), (
            f"g shape {g.shape} != {(B, T, H)}"
        )
    if g_gamma is not None:
        assert g_gamma.ndim == 1 and g_gamma.shape[0] == H, (
            f"g_gamma shape {g_gamma.shape} != ({H},)"
        )
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    if initial_state is not None:
        assert initial_state.shape == (N, H, K, V), (
            f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"
        )

    if scale is None:
        scale = K ** -0.5
    scale = float(scale)

    q_f = q
    k_f = k
    v_f = v
    g_f = g
    g_gamma_f = g_gamma
    h0_f = initial_state

    if cu_seqlens is None:
        o, ht = _scan_segment(
            q_f,
            k_f,
            v_f,
            g=g_f,
            g_gamma=g_gamma_f,
            scale=scale,
            initial_state=h0_f,
            reverse=reverse,
        )
        return o, (ht if output_final_state else None)

    cu_f = jnp.asarray(cu_seqlens, dtype=jnp.int32)
    o, ht = _scan_varlen(
        q_f,
        k_f,
        v_f,
        g=g_f,
        g_gamma=g_gamma_f,
        scale=scale,
        initial_state=h0_f,
        reverse=reverse,
        cu_seqlens=cu_f,
    )
    return o, (ht if output_final_state else None)
