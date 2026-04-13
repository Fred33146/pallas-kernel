"""JAX CPU reference for KDA fused recurrent forward operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference
from tops.cpu.ops.common import acc_dtype as _acc_dtype

from tops.cpu.ops.kda.gate import fused_kda_gate


def _l2_normalize_last_dim(x: jax.Array) -> jax.Array:
    acc = _acc_dtype(x.dtype)
    x_f = x.astype(acc)
    denom = jnp.sqrt(jnp.sum(x_f * x_f, axis=-1, keepdims=True) + 1e-6)
    return x_f / denom


def _repeat_heads(x: jax.Array, target_heads: int, name: str) -> jax.Array:
    assert x.ndim == 4, f"{name} must be 4D, got shape {x.shape}"
    heads = x.shape[2]
    assert target_heads % heads == 0, (
        f"{name} heads {heads} must divide target heads {target_heads}"
    )
    if heads == target_heads:
        return x
    return jnp.repeat(x, target_heads // heads, axis=2)


def _repeat_head_scalar(x: jax.Array, target_heads: int, name: str) -> jax.Array:
    assert x.ndim == 3, f"{name} must be 3D, got shape {x.shape}"
    heads = x.shape[2]
    assert target_heads % heads == 0, (
        f"{name} heads {heads} must divide target heads {target_heads}"
    )
    if heads == target_heads:
        return x
    return jnp.repeat(x, target_heads // heads, axis=2)


def _to_internal_state(
    state: jax.Array | None,
    *,
    expected_shape: tuple[int, int, int, int],
    transpose_state_layout: bool,
) -> jax.Array | None:
    if state is None:
        return None
    if transpose_state_layout:
        expected = (expected_shape[0], expected_shape[1], expected_shape[3], expected_shape[2])
        assert state.shape == expected, f"initial_state shape {state.shape} != {expected}"
        return jnp.swapaxes(state, -1, -2)
    assert state.shape == expected_shape, f"initial_state shape {state.shape} != {expected_shape}"
    return state


def _from_internal_state(
    state: jax.Array | None,
    *,
    transpose_state_layout: bool,
) -> jax.Array | None:
    if state is None:
        return None
    if transpose_state_layout:
        return jnp.swapaxes(state, -1, -2)
    return state


def _fused_recurrent_kda_fwd_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: jax.Array | None = None,
    ssm_state_indices: jax.Array | None = None,
    num_accepted_tokens: jax.Array | None = None,
    inplace_final_state: bool = True,
    out: jax.Array | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[jax.Array, jax.Array | None]:
    """Internal forward implementation shared by the public wrappers."""
    del kwargs
    assert ssm_state_indices is None, "ssm_state_indices is not supported in the JAX CPU reference"
    assert num_accepted_tokens is None, "num_accepted_tokens is not supported in the JAX CPU reference"
    if inplace_final_state:
        assert initial_state is not None, "initial_state is required when inplace_final_state=True"
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got shape {q.shape}"
    assert k.ndim == 4 and k.shape[:2] == q.shape[:2] and k.shape[-1] == q.shape[-1], (
        f"k shape {k.shape} incompatible with q shape {q.shape}"
    )
    assert v.ndim == 4 and v.shape[:2] == q.shape[:2], (
        f"v shape {v.shape} incompatible with q shape {q.shape}"
    )
    assert g.ndim == 4 and g.shape[:2] == q.shape[:2] and g.shape[-1] == q.shape[-1], (
        f"g shape {g.shape} incompatible with q shape {q.shape}"
    )
    assert beta.ndim == 3 and beta.shape[:2] == q.shape[:2], (
        f"beta shape {beta.shape} incompatible with q shape {q.shape}"
    )

    acc = _acc_dtype(q.dtype)

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got {B}"

    q_h = _repeat_heads(q, HV, "q")
    k_h = _repeat_heads(k, HV, "k")
    g_h = _repeat_heads(g, HV, "g")
    beta_h = _repeat_head_scalar(beta, HV, "beta")

    if use_qk_l2norm_in_kernel:
        q_h = _l2_normalize_last_dim(q_h)
        k_h = _l2_normalize_last_dim(k_h)
    else:
        q_h = q_h.astype(acc)
        k_h = k_h.astype(acc)

    if use_gate_in_kernel:
        assert A_log is not None, "A_log is required when use_gate_in_kernel=True"
        g_h = fused_kda_gate(
            g_h,
            A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            output_dtype=acc,
        )
    else:
        g_h = g_h.astype(acc)

    v_f = v.astype(acc)
    beta_f = beta_h.astype(acc)

    if scale is None:
        scale = K ** -0.5

    state_shape = (N, HV, K, V)
    h0 = _to_internal_state(
        initial_state,
        expected_shape=state_shape,
        transpose_state_layout=transpose_state_layout,
    )

    if out is None:
        out_dtype = v.dtype
        out_buf = jnp.zeros((B, T, HV, V), dtype=acc)
    else:
        assert out.shape == (B, T, HV, V), (
            f"out shape {out.shape} != {(B, T, HV, V)}"
        )
        out_dtype = out.dtype
        out_buf = out.astype(acc)
    should_return_final_state = output_final_state or inplace_final_state
    final_states = [] if should_return_final_state else None

    for i_n in range(N):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
            b = 0
        else:
            bos, eos = 0, T
            b = i_n

        h = jnp.zeros((HV, K, V), dtype=acc)
        if h0 is not None:
            h = h + h0[i_n].astype(acc)

        for t in range(bos, eos):
            q_t = q_h[b, t] * scale
            k_t = k_h[b, t]
            v_t = v_f[b, t]
            g_t = g_h[b, t]
            beta_t = beta_f[b, t]

            h = h * jnp.exp(g_t)[..., None]
            v_pred = jnp.sum(k_t[..., None] * h, axis=-2)
            residual = v_t - v_pred
            h = h + jnp.einsum("hk,hv->hkv", beta_t[..., None] * k_t, residual)
            o_t = jnp.einsum("hk,hkv->hv", q_t, h)
            out_buf = out_buf.at[b, t].set(o_t)

        if should_return_final_state:
            final_states.append(h)

    ht = jnp.stack(final_states, axis=0) if should_return_final_state else None
    if inplace_final_state and initial_state is not None and ht is not None:
        ht = ht.astype(_to_internal_state(
            initial_state,
            expected_shape=state_shape,
            transpose_state_layout=transpose_state_layout,
        ).dtype)
    ht = _from_internal_state(ht, transpose_state_layout=transpose_state_layout)
    return out_buf.astype(out_dtype), ht


def _fused_recurrent_kda_bwd_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    do: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    dht: jax.Array | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Backward pass for fused recurrent KDA.

    Two-phase algorithm:
      Phase 1 (forward replay): replay the forward recurrence, store h'_t
              (decayed state before delta update) and h_t (state after update).
              Compute dq_t = scale * sum_v(h_t * do_t).
      Phase 2 (reverse pass): accumulate dh backward, compute dk, dv, dg, dbeta.

    Args:
        q:     [B, T, H, K] — Queries
        k:     [B, T, H, K] — Keys
        v:     [B, T, H, V] — Values
        g:     [B, T, H, K] — Per-element gate in log-space
        beta:  [B, T, H]    — Learning rate for delta rule
        do:    [B, T, H, V] — Gradient of output
        scale: Scalar query scale. Defaults to K ** -0.5.
        initial_state:  [B, H, K, V] — Initial hidden state. Optional.
        dht:   [B, H, K, V] — Gradient of final hidden state. Optional.

    Returns:
        dq:    [B, T, H, K]
        dk:    [B, T, H, K]
        dv:    [B, T, H, V]
        dg:    [B, T, H, K]
        dbeta: [B, T, H]
        dh0:   [B, H, K, V] or None
    """
    acc = _acc_dtype(q.dtype)

    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got shape {q.shape}"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], (
        f"v shape {v.shape} incompatible with q shape {q.shape}"
    )
    assert g.ndim == 4 and g.shape == q.shape, (
        f"g shape {g.shape} != q shape {q.shape}"
    )
    assert beta.ndim == 3 and beta.shape == q.shape[:3], (
        f"beta shape {beta.shape} != {q.shape[:3]}"
    )
    assert do.shape == v.shape, f"do shape {do.shape} != v shape {v.shape}"

    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Cast to accumulator dtype and transpose to [B, H, T, D]
    q_f = jnp.transpose(q, (0, 2, 1, 3)).astype(acc)   # [B, H, T, K]
    k_f = jnp.transpose(k, (0, 2, 1, 3)).astype(acc)   # [B, H, T, K]
    v_f = jnp.transpose(v, (0, 2, 1, 3)).astype(acc)    # [B, H, T, V]
    g_f = jnp.transpose(g, (0, 2, 1, 3)).astype(acc)    # [B, H, T, K]
    beta_f = jnp.transpose(beta, (0, 2, 1)).astype(acc)  # [B, H, T]
    do_f = jnp.transpose(do, (0, 2, 1, 3)).astype(acc)  # [B, H, T, V]

    if use_qk_l2norm_in_kernel:
        q_f = _l2_normalize_last_dim(q_f)
        k_f = _l2_normalize_last_dim(k_f)

    if use_gate_in_kernel:
        assert A_log is not None
        g_raw = jnp.transpose(g, (0, 2, 1, 3))
        g_f = fused_kda_gate(
            g_raw, A_log, dt_bias=dt_bias,
            lower_bound=lower_bound, output_dtype=acc,
        )

    h0 = None
    if initial_state is not None:
        h0 = _to_internal_state(
            initial_state,
            expected_shape=(B, H, K, V),
            transpose_state_layout=transpose_state_layout,
        )

    dht_internal = None
    if dht is not None:
        dht_internal = _to_internal_state(
            dht,
            expected_shape=(B, H, K, V),
            transpose_state_layout=transpose_state_layout,
        )

    # =========================================================================
    # Phase 1: Forward replay — store h'_t and h_t, compute dq
    # =========================================================================
    h_prime_all = jnp.zeros((B, H, T, K, V), dtype=acc)
    h_all = jnp.zeros((B, H, T, K, V), dtype=acc)
    dq_f = jnp.zeros((B, H, T, K), dtype=acc)

    for b in range(B):
        h = jnp.zeros((H, K, V), dtype=acc)
        if h0 is not None:
            h = h + h0[b].astype(acc)

        for t in range(T):
            q_t = q_f[b, :, t] * scale             # [H, K]
            k_t = k_f[b, :, t]                     # [H, K]
            v_t = v_f[b, :, t]                     # [H, V]
            g_t = g_f[b, :, t]                     # [H, K]
            beta_t = beta_f[b, :, t]               # [H]

            # ① decay
            h_prime = h * jnp.exp(g_t)[..., None]    # [H, K, V]
            h_prime_all = h_prime_all.at[b, :, t].set(h_prime)

            # ②③④ delta update
            v_pred = jnp.sum(k_t[..., None] * h_prime, axis=-2)  # [H, V]
            e_t = v_t - v_pred                                    # [H, V]
            h = h_prime + jnp.einsum("hk,hv->hkv", beta_t[..., None] * k_t, e_t)
            h_all = h_all.at[b, :, t].set(h)

            # ⑤ dq
            dq_t = jnp.einsum("hkv,hv->hk", h, do_f[b, :, t])   # [H, K]
            dq_f = dq_f.at[b, :, t].set(dq_t * scale)

    # =========================================================================
    # Phase 2: Reverse pass — compute dk, dv, dg, dbeta, dh0
    # =========================================================================
    dk_f = jnp.zeros((B, H, T, K), dtype=acc)
    dv_f = jnp.zeros((B, H, T, V), dtype=acc)
    dg_f = jnp.zeros((B, H, T, K), dtype=acc)
    dbeta_f = jnp.zeros((B, H, T), dtype=acc)
    dh0_out = jnp.zeros((B, H, K, V), dtype=acc) if initial_state is not None else None

    for b in range(B):
        dh = jnp.zeros((H, K, V), dtype=acc)
        if dht_internal is not None:
            dh = dh + dht_internal[b].astype(acc)

        for t in reversed(range(T)):
            q_t = q_f[b, :, t] * scale               # [H, K]
            k_t = k_f[b, :, t]                        # [H, K]
            v_t = v_f[b, :, t]                         # [H, V]
            g_t = g_f[b, :, t]                         # [H, K]
            beta_t = beta_f[b, :, t]                   # [H]
            do_t = do_f[b, :, t]                       # [H, V]
            h_prime = h_prime_all[b, :, t]             # [H, K, V]

            # ⑤ output gradient → state gradient
            dh = dh + q_t[..., None] * do_t[:, None, :]  # [H, K, V]

            # Recompute e_t from stored h'
            v_pred = jnp.sum(k_t[..., None] * h_prime, axis=-2)  # [H, V]
            e_t = v_t - v_pred                                     # [H, V]

            # ④ de = beta * (k^T @ dh)
            de = beta_t[..., None] * jnp.sum(dh * k_t[..., None], axis=-2)  # [H, V]

            # dk = beta * sum_v(dh * e) - sum_v(de * h')
            dk_t = (beta_t[..., None] * jnp.sum(dh * e_t[:, None, :], axis=-1)
                    - jnp.sum(de[:, None, :] * h_prime, axis=-1))  # [H, K]
            dk_f = dk_f.at[b, :, t].set(dk_t)

            # ③ dv = de
            dv_f = dv_f.at[b, :, t].set(de)

            # dbeta = sum_v(sum_k(dh * k) * e) = sum_v((k^T @ dh) * e)
            kt_dh = jnp.sum(dh * k_t[..., None], axis=-2)  # [H, V]
            dbeta_t = jnp.sum(kt_dh * e_t, axis=-1)         # [H]
            dbeta_f = dbeta_f.at[b, :, t].set(dbeta_t)

            # dh' = dh - k ⊗ de
            dh_prime = dh - k_t[..., None] * de[:, None, :]  # [H, K, V]

            # ① dg = sum_v(dh' * h')
            dg_t = jnp.sum(dh_prime * h_prime, axis=-1)      # [H, K]
            dg_f = dg_f.at[b, :, t].set(dg_t)

            # dh_prev = dh' * exp(g)
            dh = dh_prime * jnp.exp(g_t)[..., None]          # [H, K, V]

        if dh0_out is not None:
            dh0_out = dh0_out.at[b].set(dh)

    # Transpose back to [B, T, H, D] and cast to original dtypes
    dq = jnp.transpose(dq_f, (0, 2, 1, 3)).astype(q.dtype)
    dk = jnp.transpose(dk_f, (0, 2, 1, 3)).astype(k.dtype)
    dv = jnp.transpose(dv_f, (0, 2, 1, 3)).astype(v.dtype)
    dg = jnp.transpose(dg_f, (0, 2, 1, 3)).astype(g.dtype)
    dbeta = jnp.transpose(dbeta_f, (0, 2, 1)).astype(beta.dtype)

    if dh0_out is not None:
        dh0_out = _from_internal_state(
            dh0_out.astype(acc),
            transpose_state_layout=transpose_state_layout,
        )

    return dq, dk, dv, dg, dbeta, dh0_out


@cpu_reference
def fused_recurrent_kda_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    do: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Backward API aligned to ``fla.ops.kda.fused_recurrent_kda_bwd``.

    Args:
        q:     [B, T, H, K] — Queries
        k:     [B, T, H, K] — Keys
        v:     [B, T, H, V] — Values
        g:     [B, T, H, K] — Per-element gate in log-space
        beta:  [B, T, H]    — Learning rate for delta rule
        do:    [B, T, H, V] — Gradient of output
        initial_state: [B, H, K, V] — Initial hidden state. Optional.
        dht:   [B, H, K, V] — Gradient of final hidden state. Optional.
        scale: Scalar query scale. Defaults to K ** -0.5.

    Returns:
        dq:    [B, T, H, K]
        dk:    [B, T, H, K]
        dv:    [B, T, H, V]
        dg:    [B, T, H, K]
        dbeta: [B, T, H]
        dh0:   [B, H, K, V] or None
    """
    del kwargs
    return _fused_recurrent_kda_bwd_impl(
        q=q, k=k, v=v, g=g, beta=beta, do=do,
        scale=scale,
        initial_state=initial_state,
        dht=dht,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        transpose_state_layout=transpose_state_layout,
    )


@cpu_reference
def fused_recurrent_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = True,
    cu_seqlens: jax.Array | None = None,
    ssm_state_indices: jax.Array | None = None,
    num_accepted_tokens: jax.Array | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    out: jax.Array | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[jax.Array, jax.Array | None]:
    """Forward API aligned to ``fla.ops.kda.fused_recurrent_kda_fwd``."""
    return _fused_recurrent_kda_fwd_impl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        scale=scale,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        out=out,
        transpose_state_layout=transpose_state_layout,
        **kwargs,
    )


@cpu_reference
def fused_recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: jax.Array | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[jax.Array, jax.Array | None]:
    """Public API aligned to ``fla.ops.kda.fused_recurrent_kda``."""
    return fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        scale=scale,
        output_final_state=output_final_state,
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        transpose_state_layout=transpose_state_layout,
        **kwargs,
    )


__all__ = ["fused_recurrent_kda_fwd", "fused_recurrent_kda_bwd", "fused_recurrent_kda"]
