from __future__ import annotations

import enum
from typing import Literal

import jax

from tops.utils import assert_shape, assert_shape_or_none
from .chunk import chunk_simple_gla, chunk_simple_gla_bwd, chunk_simple_gla_fwd
from .fused_chunk import fused_chunk_simple_gla, fused_chunk_simple_gla_fwd, fused_chunk_simple_gla_bwd
from .fused_recurrent import fused_recurrent_simple_gla
from .naive import simple_gla_naive


def simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *args,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    reverse: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    mode: Literal["chunk", "fused_chunk", "fused_recurrent", "naive"] = "chunk",
    **kwargs,
) -> tuple[jax.Array, jax.Array | None]:
    """Unified Simple GLA interface dispatching to the chosen implementation mode.

    Computes the Simple GLA (Gated Linear Attention) recurrence:

        h_t = h_{t-1} * exp(gate_t) + k_t^T @ v_t
        o_t = q_t^T @ h_t

    where ``gate_t`` is derived from ``g`` (per-token) and/or ``g_gamma``
    (per-head constant decay).

    Args:
        q: Queries, shape ``[B, T, H, K]``.
        k: Keys, shape ``[B, T, H, K]``.
        v: Values, shape ``[B, T, H, V]``.
        g: Optional per-token log gate, shape ``[B, T, H]``.
            Supported by ``"fused_recurrent"`` and ``"naive"`` modes.
        g_gamma: Optional per-head constant log decay, shape ``[H]``.
            Required by ``"chunk"`` and ``"fused_chunk"`` modes;
            optional for ``"fused_recurrent"`` and ``"naive"``.
            When neither ``g`` nor ``g_gamma`` is provided, behaviour
            depends on the backend implementation.
        scale: Query scaling factor. Defaults to ``K ** -0.5``.
        initial_state: Initial recurrent state.
            Shape ``[B, H, K, V]`` for dense mode or ``[N, H, K, V]`` for
            varlen mode (where ``N = len(cu_seqlens) - 1``).
        output_final_state: Whether to return the final recurrent state.
        chunk_size: Chunk/block size for chunked modes. Ignored by
            ``"fused_recurrent"``.
        reverse: Process sequences in reverse time order. Only supported
            by ``"fused_recurrent"``.
        cu_seqlens_cpu: Cumulative sequence lengths for packed varlen inputs,
            shape ``[N+1]``. Only supported by ``"fused_recurrent"`` and
            ``"naive"``.
        mode: Implementation mode to use:
            - ``"chunk"``: Chunked Pallas TPU kernels (default). Supports
              backward pass. Requires ``g_gamma``.
            - ``"fused_chunk"``: Fused-chunk Pallas TPU kernels. Keeps hidden
              state in VMEM to avoid HBM round-trips. Requires ``g_gamma``.
            - ``"fused_recurrent"``: Element-wise recurrent via ``jax.lax.scan``,
              decode-friendly. Supports ``g``, ``g_gamma``, ``reverse``,
              and ``cu_seqlens``.
            - ``"naive"``: Pure JAX reference implementation for testing.
              Supports all parameters.

    Returns:
        Tuple of:
            - Output tensor, shape ``[B, T, H, V]``, in ``q.dtype``.
            - Final recurrent state ``[N, H, K, V]`` if
              ``output_final_state=True``, else ``None``.

    Raises:
        ValueError: If ``mode`` is not one of the supported values, or if
            unsupported parameters are passed for the chosen mode.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert_shape((q, k), (B, T, H, K), ("q", "k"))
    assert_shape(v, (B, T, H, V), "v")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(g_gamma, (H,), "g_gamma")

    if mode == "chunk":
        if g is not None:
            raise ValueError(
                "mode='chunk' does not support per-token gate `g`; "
                "use 'fused_recurrent' or 'naive' instead"
            )
        if reverse:
            raise ValueError("mode='chunk' does not support `reverse`")
        if cu_seqlens_cpu is not None:
            raise ValueError("mode='chunk' does not support `cu_seqlens`")
        assert g_gamma is not None, "mode='chunk' requires g_gamma"
        return chunk_simple_gla(
            q, k, v, g_gamma,
            initial_state=initial_state,
            scale=scale,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )

    elif mode == "fused_chunk":
        if g is not None:
            raise ValueError(
                "mode='fused_chunk' does not support per-token gate `g`; "
                "use 'fused_recurrent' or 'naive' instead"
            )
        if reverse:
            raise ValueError("mode='fused_chunk' does not support `reverse`")
        if cu_seqlens_cpu is not None:
            raise ValueError("mode='fused_chunk' does not support `cu_seqlens`")
        assert g_gamma is not None, "mode='fused_chunk' requires g_gamma"
        return fused_chunk_simple_gla(
            q, k, v, g_gamma,
            initial_state=initial_state,
            scale=scale,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )

    elif mode == "fused_recurrent":
        return fused_recurrent_simple_gla(
            q, k, v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens_cpu,
        )

    elif mode == "naive":
        return simple_gla_naive(
            q, k, v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of "
            f"'chunk', 'fused_chunk', 'fused_recurrent', 'naive'"
        )

class SimpleGLAKernelMode(enum.Enum):
    """Simple GLA kernel implementation mode."""

    CHUNK = "chunk"
    FUSED_CHUNK = "fused_chunk"


def simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK
):
    fn = None
    if mode == SimpleGLAKernelMode.CHUNK:
        fn = chunk_simple_gla_fwd
    elif mode == SimpleGLAKernelMode.FUSED_CHUNK:
        fn = fused_chunk_simple_gla_fwd
    else:
        raise Exception(f"mode {mode} not support")
    return fn(
            q, k, v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            h0=h0,
            use_ht=use_ht,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_seqlens_dev=cu_seqlens_dev,
            chunk_size=chunk_size
        )


def simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK
):
    fn = None
    if mode == SimpleGLAKernelMode.CHUNK:
        fn = chunk_simple_gla_bwd
    elif mode == SimpleGLAKernelMode.FUSED_CHUNK:
        fn = fused_chunk_simple_gla_bwd
    else:
        raise Exception(f"mode {mode} not support")

    return fn(
            q, k, v,
            do=do,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            h0=h0,
            dht=dht,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_seqlens_dev=cu_seqlens_dev,
            chunk_size=chunk_size
        )
