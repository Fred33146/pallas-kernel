"""Common benchmark utilities for Pallas kernel benchmarks.

Provides:
  - bench_fn:         Time a callable with warmup and return median ms.
  - run_benchmark:    Generic benchmark loop across sequence lengths and providers.
  - add_common_args:  Add shared CLI arguments (B, H, D, dtype, etc.) to an argparse parser.

Usage in a kernel-specific benchmark file:

    from benchmarks.utils import bench_fn, run_benchmark, add_common_args

    def make_inputs(B, T, H, D, dtype, key):
        ...
        return {"q": q, "k": k, ...}

    def run_provider(provider, inputs, scale):
        ...
        return bench_fn(fn)

    ALL_PROVIDERS = ["naive", "chunk", ...]

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(...)
        add_common_args(parser, all_providers=ALL_PROVIDERS)
        args = parser.parse_args()
        run_benchmark(
            name="MyKernel",
            make_inputs_fn=make_inputs,
            run_provider_fn=run_provider,
            all_providers=ALL_PROVIDERS,
            B=args.B, H=args.H, D=args.D, ...
        )
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import jax
import os

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def bench_fn(
    name: str,
    fn: Callable,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    profile_dir=None,
) -> float:
    """Time *fn* using wall-clock after JAX compilation and warmup.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of warmup iterations (includes JIT compilation).
        iters: Number of timed iterations.

    Returns:
        Median execution time in milliseconds.
    """
    for _ in range(warmup):
        out = fn()
        jax.block_until_ready(out)

    times = []
    # Profiled run
    trace_dir = os.path.join(profile_dir, name) if profile_dir else None
    if trace_dir:
        jax.profiler.start_trace(trace_dir)

    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    if trace_dir:
        jax.profiler.stop_trace()
        print(f"  Profile saved to {trace_dir}")
    times.sort()
    return times[len(times) // 2]


def run_benchmark(
    name: str,
    make_inputs_fn: Callable,
    run_provider_fn: Callable,
    all_providers: list[str],
    B: int = 16,
    H: int = 8,
    D: int = 128,
    seq_lengths: list[int] | None = None,
    providers: list[str] | None = None,
    dtype: str = "bfloat16",
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> list[dict]:
    """Run benchmarks across sequence lengths and providers.

    Args:
        name: Benchmark name for the header (e.g. "GLA", "Simple GLA").
        make_inputs_fn: ``(B, T, H, D, dtype, key) -> dict`` that generates
            input tensors for one (B, T) configuration.
        run_provider_fn: ``(provider, inputs, scale, *, warmup, iters) -> float | None``
            that runs a single provider and returns median time in ms (or None
            to skip).
        all_providers: Full list of provider names (used when *providers* is None).
        B: Batch size.
        H: Number of heads.
        D: Head dimension.
        seq_lengths: Sequence lengths to sweep.  Defaults to [128, 256, ..., 16384].
        providers: Subset of providers to run.  Defaults to *all_providers*.
        dtype: ``"bfloat16"`` or ``"float32"``.
        warmup: Warmup iterations passed to *run_provider_fn*.
        iters: Benchmark iterations passed to *run_provider_fn*.

    Returns:
        List of dicts ``{"T": int, "provider": str, "time_ms": float}``.
    """
    import jax.numpy as jnp

    if seq_lengths is None:
        seq_lengths = [128 * 2**i for i in range(0, 8)]
    if providers is None:
        providers = all_providers

    jnp_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    scale = D**-0.5
    results: list[dict] = []

    print(f"\n{name} Benchmark  B={B}, H={H}, D={D}, dtype={dtype}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Warmup={warmup}, Iters={iters}")
    print("=" * 90)

    header = f"{'T':>8}"
    for p in providers:
        header += f" | {p:>22}"
    print(header)
    print("-" * 90)

    for T in seq_lengths:
        inputs = make_inputs_fn(B, T, H, D, jnp_dtype)
        row = f"{T:>8}"

        for provider in providers:
            try:
                t_ms = run_provider_fn(
                    provider, inputs, scale, warmup=warmup, iters=iters,
                )
            except Exception as e:
                print(f"\n  [ERROR] {provider} T={T}: {e}\n", end="")
                t_ms = None
            if t_ms is not None:
                row += f" | {t_ms:>19.3f} ms"
                results.append({"T": T, "provider": provider, "time_ms": t_ms})
            else:
                row += f" | {'N/A':>22}"

        print(row)

    print("=" * 90)
    return results


def add_common_args(
    parser: argparse.ArgumentParser,
    all_providers: list[str],
) -> None:
    """Add common benchmark CLI arguments to *parser*.

    Adds: --B, --H, --D, --dtype, --seq-lengths, --providers, --warmup, --iters.

    Args:
        parser: The argument parser to extend.
        all_providers: Valid provider names for the ``--providers`` choice list.
    """
    parser.add_argument("--B", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--H", type=int, default=8, help="Number of heads (default: 8)")
    parser.add_argument("--D", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths to benchmark (default: 128,256,...,16384)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=None,
        choices=all_providers,
        help="Providers to benchmark (default: all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"Benchmark iterations (default: {DEFAULT_ITERS})",
    )
