"""Benchmark for GLA (Gated Linear Attention) kernel variants.

Measures execution time of forward and backward passes for:
  - naive_recurrent_gla  : pure JAX loop-based reference
  - fused_recurrent_gla  : Pallas TPU/CPU fused recurrent kernel
  - chunk_gla            : chunked parallel GLA
  - fused_chunk_gla      : fused chunk GLA (alias of chunk_gla in JAX)
  - simple_gla_naive     : pure JAX loop-based Simple GLA reference
  - simple_gla_chunk     : chunked parallel Simple GLA (scalar per-head gate)

Reference:
  https://github.com/fla-org/flash-linear-attention/blob/main/benchmarks/ops/benchmark_gla.py

Usage:
  python benchmarks/ops/benchmark_gla.py
  python benchmarks/ops/benchmark_gla.py --B 4 --H 4 --D 64
  python benchmarks/ops/benchmark_gla.py --providers fused_recurrent chunk simple_gla_chunk
"""

from __future__ import annotations

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import os

from benchmarks.utils import add_common_args, bench_fn, run_benchmark
from tops.ops.gla import (
    chunk_gla,
    fused_chunk_gla,
    fused_recurrent_gla,
    naive_recurrent_gla,
)
from tops.ops.simple_gla import (
    chunk_simple_gla_bwd,
    chunk_simple_gla_fwd,
    simple_gla_naive,
    fused_chunk_simple_gla_fwd,
)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

ALL_PROVIDERS = [
    "naive_recurrent",
    "fused_recurrent",
    "chunk",
    "fused_chunk",
    "simple_gla_naive",
    "simple_gla_chunk",
    "fused_simple_gla_chunk",
    "fused_recurrent_bwd",
    "chunk_bwd",
    "fused_chunk_bwd",
    "simple_gla_chunk_bwd",
]

# Max sequence length for naive recurrent (O(T^2) memory)
NAIVE_MAX_T = 2048


def _make_inputs(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: jnp.dtype = jnp.bfloat16,
    *,
    key: jax.Array | None = None,
) -> dict:
    """Generate random q, k, v, gk, g_gamma tensors for GLA benchmarks.

    Args:
        B: Batch size.
        T: Sequence length.
        H: Number of heads.
        D: Head dimension (used for both K and V).
        dtype: Data type for the tensors.
        key: Optional PRNG key; uses jax.random.PRNGKey(42) if None.

    Returns:
        Dict with keys 'q', 'k', 'v', 'gk' ([B, T, H, D]) and 'g_gamma' ([H]).
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, T, H, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, D), dtype=dtype)
    # gk is log-space gate, clamp to avoid extreme values
    gk = jnp.clip(
        jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, D), dtype=dtype)),
        a_min=-5.0,
    )
    # g_gamma: scalar per-head gate for Simple GLA
    g_gamma = -jnp.abs(jax.random.normal(keys[4], (H,), dtype=jnp.float32))
    return {"q": q, "k": k, "v": v, "gk": gk, "g_gamma": g_gamma}


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------


def _run_provider(
    provider: str,
    inputs: dict,
    scale: float,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> float | None:
    """Run a single benchmark provider and return median time in ms.

    Args:
        provider: Name of the provider (e.g., 'fused_recurrent', 'chunk_bwd').
        inputs: Dict with keys 'q', 'k', 'v', 'gk', 'g_gamma'.
        scale: Scaling factor for queries.
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.

    Returns:
        Median execution time in milliseconds, or None if skipped.
    """
    q, k, v, gk = inputs["q"], inputs["k"], inputs["v"], inputs["gk"]
    g_gamma = inputs["g_gamma"]
    T = q.shape[1]
    D = q.shape[3]

    if provider == "naive_recurrent":
        if T > NAIVE_MAX_T:
            return None
        fn = partial(naive_recurrent_gla, q, k, v, gk, scale=scale)
    elif provider == "fused_recurrent":
        fn = partial(fused_recurrent_gla, q, k, v, gk=gk, scale=scale)
    elif provider == "chunk":
        fn = partial(chunk_gla, q, k, v, g=gk, scale=scale)
    elif provider == "fused_chunk":
        fn = partial(fused_chunk_gla, q, k, v, gk, scale=scale)
    elif provider == "fused_recurrent_bwd":

        @jax.jit
        def fwd_bwd():
            def loss_fn(q_, k_, v_, gk_):
                o, _ = fused_recurrent_gla(q_, k_, v_, gk=gk_, scale=scale)
                return o.sum()

            return jax.grad(loss_fn, argnums=(0, 1, 2, 3))(q, k, v, gk)

        fn = fwd_bwd
    elif provider == "chunk_bwd":

        @jax.jit
        def fwd_bwd():
            def loss_fn(q_, k_, v_, gk_):
                o, _ = chunk_gla(q_, k_, v_, g=gk_, scale=scale)
                return o.sum()

            return jax.grad(loss_fn, argnums=(0, 1, 2, 3))(q, k, v, gk)

        fn = fwd_bwd
    elif provider == "fused_chunk_bwd":

        @jax.jit
        def fwd_bwd():
            def loss_fn(q_, k_, v_, gk_):
                o, _ = fused_chunk_gla(q_, k_, v_, gk_, scale=scale)
                return o.sum()

            return jax.grad(loss_fn, argnums=(0, 1, 2, 3))(q, k, v, gk)

        fn = fwd_bwd
    elif provider == "simple_gla_naive":
        if T > NAIVE_MAX_T:
            return None
        fn = partial(
            simple_gla_naive, q, k, v,
            g_gamma=jnp.broadcast_to(g_gamma.reshape(1, 1, -1, 1), q.shape),
            scale=scale,
        )
    elif provider == "simple_gla_chunk":
        # chunk_simple_gla_fwd requires T % chunk_size == 0 and D % 128 == 0
        chunk_size = 64
        if T % chunk_size != 0 or D % 128 != 0:
            return None
        fn = partial(
            chunk_simple_gla_fwd, q, k, v,
            g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
    elif provider == "fused_simple_gla_chunk":
        # chunk_simple_gla_fwd requires T % chunk_size == 0 and D % 128 == 0
        chunk_size = 64
        if T % chunk_size != 0 or D % 128 != 0:
            return None
        fn = partial(
            fused_chunk_simple_gla_fwd, q, k, v,
            g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
    elif provider == "simple_gla_chunk_bwd":
        chunk_size = 64
        if T % chunk_size != 0 or D % 128 != 0:
            return None

        # Forward pass first to get output for backward
        o_fwd, _ = chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        do = jnp.ones_like(o_fwd)

        @jax.jit
        def run_bwd():
            return chunk_simple_gla_bwd(
                q, k, v, do,
                g_gamma=g_gamma, scale=scale,
                h0=None, dht=None,
                chunk_size=chunk_size,
            )

        fn = run_bwd
    else:
        raise ValueError(f"Unknown provider: {provider}")
    profile_dir = os.environ.get("PROFILE_DIR")
    return bench_fn(provider ,fn, warmup=warmup, iters=iters, profile_dir=profile_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GLA kernel variants")
    add_common_args(parser, all_providers=ALL_PROVIDERS)
    args = parser.parse_args()

    run_benchmark(
        name="GLA",
        make_inputs_fn=_make_inputs,
        run_provider_fn=_run_provider,
        all_providers=ALL_PROVIDERS,
        B=args.B,
        H=args.H,
        D=args.D,
        seq_lengths=args.seq_lengths,
        providers=args.providers,
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
    )
