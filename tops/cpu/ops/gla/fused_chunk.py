from .chunk import chunk_gla


def fused_chunk_gla(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_size=16,
):
    """Fused chunk GLA — delegates to chunk_gla.

    In FLA this function is deprecated. On JAX CPU there is no difference
    between fused and non-fused chunk implementations.

    Args:
        q: [B, T, H, K] — Queries
        k: [B, T, H, K] — Keys
        v: [B, T, H, V] — Values
        g: [B, T, H, K] — Gating (log-space)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V] — Initial state
        output_final_state: Whether to output the final state
        cu_seqlens: [N+1] — Cumulative sequence lengths (not supported, placeholder)
        chunk_size: Chunk size for chunked computation

    Returns:
        o: [B, T, H, V] — Output
        final_state: [N, H, K, V] or None
    """
    return chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
