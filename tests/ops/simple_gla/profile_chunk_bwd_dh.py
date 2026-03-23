"""Profile chunk_bwd_dh_kernel with xprof trace + LLO dump.

Usage (on TPU):
  export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=/tmp/mosaic_dumps --xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
  python tests/ops/simple_gla/profile_chunk_bwd_dh.py

Outputs:
  - xprof trace:  /tmp/xprof_chunk_bwd_dh/  (view with TensorBoard)
  - LLO dump:     /tmp/mosaic_dumps/
"""

from __future__ import annotations
import sys
from pathlib import Path
import jax.profiler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp

from tops.ops.common.chunk_h import chunk_bwd_dh_kernel
import os
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_mosaic_dump_to=/tmp/mosaic_dumps "
    "--xla_xprof_register_llo_debug_info=true "
    "--xla_enable_custom_call_region_trace=true"
)
print(f"TPU Args: {os.environ.get('LIBTPU_INIT_ARGS')}")

# ---------- cases ----------
CASES = {
    # grid=(1,1,1), 2 chunks — minimal trace
    "small_B1": dict(B=1, T=256, H=1, K=128, V=128, chunk_size=128),
    # grid=(1,1,1), 4 chunks — see varlen reset
    "small_B2": dict(B=2, T=256, H=1, K=128, V=128, chunk_size=128),
    # grid=(4,1,1) — parallel over H
    "grid_4H": dict(B=1, T=256, H=4, K=128, V=128, chunk_size=128),
    # grid=(1,4,1) — parallel over K blocks
    "grid_4K": dict(B=1, T=256, H=1, K=512, V=128, chunk_size=128),
    # grid=(1,1,4) — parallel over V blocks
    "grid_4V": dict(B=1, T=256, H=1, K=128, V=512, chunk_size=128),
    # grid=(1,2,2) — parallel over K and V blocks
    "grid_2K2V": dict(B=1, T=256, H=1, K=256, V=256, chunk_size=128),
    # "real_train": dict(B=2, T=4096, H=16, K=128, V=128, chunk_size=128),
}

SCALE_FN = lambda K: K**-0.5
PROFILE_DIR = "/tmp/xprof_chunk_bwd_dh"
WARMUP = 5
PROFILE_STEPS = 3


def make_inputs(rng_key, B, T, H, K, V):
    keys = jax.random.split(rng_key, 6)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    do = jax.random.normal(keys[3], (B, T, H, V), dtype=jnp.float32)
    gk = jax.random.normal(keys[4], (B, T, H, K), dtype=jnp.float32) * 0.1
    dht = jax.random.normal(keys[5], (B, H, K, V), dtype=jnp.float32)
    return q, k, v, gk, do, dht


def run_kernel(q, k, v, gk, do, dht, scale, chunk_size):
    return chunk_bwd_dh_kernel(
        q, k, v,
        gk=gk,
        do=do,
        dht=dht,
        output_dh0=True,
        scale=scale,
        chunk_size=chunk_size,
    )


def profile_case(name, cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    chunk_size = cfg["chunk_size"]
    scale = SCALE_FN(K)
    NT = T // chunk_size

    print(f"\n=== {name} ===")
    print(f"  B={B} T={T} H={H} K={K} V={V}  chunk_size={chunk_size}")
    print(f"  grid=({H}, {K//128}, {V//128})  chunks(NT)={B*NT}")

    rng = jax.random.key(42)
    q, k, v, gk, do, dht = make_inputs(rng, B, T, H, K, V)

    # warmup
    for i in range(WARMUP):
        dh, dh0 = run_kernel(q, k, v, gk, do, dht, scale, chunk_size)
        dh.block_until_ready()
        if i == 0:
            print(f"  dh shape:  {dh.shape}")
            print(f"  dh0 shape: {dh0.shape}")
    print(f"  Warmup done ({WARMUP} iters)")

    # profiled run
    trace_dir = f"{PROFILE_DIR}/{name}"
    jax.profiler.start_trace(trace_dir)
    for _ in range(PROFILE_STEPS):
        dh, dh0 = run_kernel(q, k, v, gk, do, dht, scale, chunk_size)
        dh.block_until_ready()
    jax.profiler.stop_trace()
    print(f"  Trace saved to {trace_dir}")


def main():
    print(f"Profile dir: {PROFILE_DIR}")
    jax.profiler.start_server(9999)

    for name, cfg in CASES.items():
        profile_case(name, cfg)

    print(f"\nView with:  tensorboard --logdir {PROFILE_DIR}")


if __name__ == "__main__":
    main()