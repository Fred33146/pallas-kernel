#!/usr/bin/env python3
"""Performance benchmark for pallas_chunk_gla forward + backward.

Runs two variants:
  1. Direct call (no shard_map)
  2. Wrapped in shard_map (same as BailingMoeV2LinearAttention)

Usage:
  python3 tests/gla_perf_test.py
  PROFILE_DIR=/tmp/gla_perf_profile python3 tests/gla_perf_test.py
"""

import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from tops.ops.simple_gla.chunk import chunk_simple_gla

# ── Config ──────────────────────────────────────────────────────────────────
B, T, H, K = 2, 4096, 16, 128  # (2, 4096, 16, 128) → hidden_dim = 2048
CHUNK_SIZE = 64
DTYPE = jnp.bfloat16


def bench(name, fwd_bwd_fn, q, k_, v, g_gamma, profile_dir=None):
  """Warmup + timed run for a fwd+bwd function."""
  print(f"── {name} ──")

  # Warmup
  t0 = time.perf_counter()
  grads = fwd_bwd_fn(q, k_, v, g_gamma)
  jax.block_until_ready(grads)
  print(f"  Warmup (compile + execute): {time.perf_counter() - t0:.2f}s")

  # Profiled run
  trace_dir = os.path.join(profile_dir, name) if profile_dir else None
  if trace_dir:
    jax.profiler.start_trace(trace_dir)

  t0 = time.perf_counter()
  grads = fwd_bwd_fn(q, k_, v, g_gamma)
  jax.block_until_ready(grads)
  elapsed = time.perf_counter() - t0

  if trace_dir:
    jax.profiler.stop_trace()
    print(f"  Profile saved to {trace_dir}")

  print(f"  Fwd+Bwd: {elapsed * 1000:.2f}ms")
  print()


def main():
  print(f"Devices: {jax.devices()}")
  print(f"Backend: {jax.default_backend()}")
  print(f"Shape: q/k/v = ({B}, {T}, {H}, {K}), dtype={DTYPE}")
  print()

  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  q = jax.random.normal(k1, (B, T, H, K), dtype=DTYPE)
  k_ = jax.random.normal(k2, (B, T, H, K), dtype=DTYPE)
  v = jax.random.normal(k3, (B, T, H, K), dtype=DTYPE)
  g_gamma = jax.random.uniform(k4, (H,), dtype=jnp.float32) * -0.1

  profile_dir = os.environ.get("PROFILE_DIR")

  # ── 1. Direct call (no shard_map) ────────────────────────────────────────
  def fwd_bwd_direct(q, k, v, g_gamma):
    def fwd(q, k, v, g_gamma):
      o, _ = chunk_simple_gla(q, k, v, g_gamma, chunk_size=CHUNK_SIZE)
      return o.sum()
    return jax.grad(fwd, argnums=(0, 1, 2))(q, k, v, g_gamma)

  bench("direct", fwd_bwd_direct, q, k_, v, g_gamma, profile_dir)

  # ── 2. Wrapped in shard_map (same as attention_gla.py) ───────────────────
  mesh = Mesh(np.array(jax.devices()).reshape(-1), ("devices",))
  qkv_pspec = P(None, None, None, None)
  g_gamma_pspec = P(None,)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(qkv_pspec, qkv_pspec, qkv_pspec, g_gamma_pspec),
      out_specs=qkv_pspec,
      check_vma=False,
  )
  def _shard_map_chunk_gla(q, k, v, g_gamma):
    o, _ = chunk_simple_gla(q, k, v, g_gamma, chunk_size=CHUNK_SIZE)
    return o

  def fwd_bwd_shard_map(q, k, v, g_gamma):
    def fwd(q, k, v, g_gamma):
      with mesh:
        o = _shard_map_chunk_gla(q, k, v, g_gamma)
      return o.sum()
    return jax.grad(fwd, argnums=(0, 1, 2))(q, k, v, g_gamma)

  bench("shard_map", fwd_bwd_shard_map, q, k_, v, g_gamma, profile_dir)


if __name__ == "__main__":
  main()