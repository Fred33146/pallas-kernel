import jax
import jax.numpy as jnp
import numpy as np


def naive_recurrent_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Naive recurrent GLA — Pure JAX implementation.

    Mathematically equivalent to chunk_gla / fused_recurrent_gla / fused_chunk_gla,
    but using naive step-by-step recurrence instead of Triton block or fused kernels.

    Core recurrence:
        h_t = h_{t-1} * exp(gk_t) + k_t^T v_t
        o_t = q_t * h_t  (then sum along K dimension)

    Args:
        q: [B, T, H, K] — Queries
        k: [B, T, H, K] — Keys
        v: [B, T, H, V] — Values
        gk: [B, T, H, K] — Gating (log-space, i.e., values after logsigmoid)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V] — Initial state
        output_final_state: Whether to output the final state
        cu_seqlens: [N+1] — Cumulative length of variable-length sequences (B must be 1 in this case)

    Returns:
        o: [B, T, H, V] — Output
        final_state: [N, H, K, V] or None
    """
    dtype = q.dtype
    # transpose: [B, T, H, K/V] -> [B, H, T, K/V], float32 calculation
    q, k, v, gk = (
        jnp.transpose(x, (0, 2, 1, 3)).astype(jnp.float32) for x in (q, k, v, gk)
    )
    B, H, T_total, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    if cu_seqlens is not None:
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = jnp.zeros_like(v)  # [1, H, T_total, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = int(cu_seqlens[i])
            eos = int(cu_seqlens[i + 1])
            seg_len = eos - bos

            q_seg = q[:, :, bos:eos, :]
            k_seg = k[:, :, bos:eos, :]
            v_seg = v[:, :, bos:eos, :]
            gk_seg = gk[:, :, bos:eos, :]

            h = jnp.zeros((1, H, K, V), dtype=jnp.float32)
            if initial_state is not None:
                h = h + initial_state[i : i + 1].astype(jnp.float32)

            for t in range(seg_len):
                q_t = q_seg[:, :, t] * scale
                k_t = k_seg[:, :, t]
                v_t = v_seg[:, :, t]
                gk_t = jnp.exp(gk_seg[:, :, t])
                kv_t = k_t[..., None] * v_t[..., None, :]
                h = h * gk_t[..., None] + kv_t
                o = o.at[:, :, bos + t].set((q_t[..., None] * h).sum(-2))

            if output_final_state:
                final_states.append(h[0])  # [H, K, V]

        final_state = jnp.stack(final_states, axis=0) if output_final_state else None
        return jnp.transpose(o, (0, 2, 1, 3)).astype(dtype), final_state
    else:
        o = jnp.zeros_like(v)  # [B, H, T, V]
        h = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        if initial_state is not None:
            h = h + initial_state.astype(jnp.float32)

        for t in range(T_total):
            q_t = q[:, :, t] * scale
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            gk_t = jnp.exp(gk[:, :, t])
            kv_t = k_t[..., None] * v_t[..., None, :]
            h = h * gk_t[..., None] + kv_t
            o = o.at[:, :, t].set((q_t[..., None] * h).sum(-2))

        final_state = h if output_final_state else None
        return jnp.transpose(o, (0, 2, 1, 3)).astype(dtype), final_state
