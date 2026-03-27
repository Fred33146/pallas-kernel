"""Compare GPU Triton fp16 vs CPU ref fp32 for varlen sequences."""
import os
os.environ["TRITON_F32_DEFAULT"] = "ieee"

import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.utils import compute_ulp
from tops.cpu.ops.gla import chunk_gla
from fla.ops.gla import chunk_gla as triton_chunk_gla

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Varlen cases: same shapes as test_chunk_gla_varlen.py
CASES = [
    dict(label="T256_segs2_H2_K32_V64",  T=256,  H=2, K=32, V=64,  cu_seqlens=[0, 128, 256],             C=16, seed=42),
    dict(label="T512_segs2_H4_K32_V64",  T=512,  H=4, K=32, V=64,  cu_seqlens=[0, 256, 512],             C=16, seed=7),
    dict(label="T1024_segs2_H2_K32_V64", T=1024, H=2, K=32, V=64,  cu_seqlens=[0, 512, 1024],            C=16, seed=11),
    dict(label="T512_segs4_H2_K32_V64",  T=512,  H=2, K=32, V=64,  cu_seqlens=[0, 64, 192, 384, 512],   C=16, seed=400),
    dict(label="T1024_segs4_H2_K32_V64", T=1024, H=2, K=32, V=64,  cu_seqlens=[0, 128, 384, 768, 1024], C=16, seed=99),
]


def run(cfg):
    """GPU Triton varlen fp16 vs CPU ref varlen fp32."""
    T, H, K, V, C, seed = cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["C"], cfg["seed"]
    cu_list = cfg["cu_seqlens"]

    torch.manual_seed(seed)
    q32 = torch.randn(1, T, H, K, dtype=torch.float32)
    k32 = torch.randn(1, T, H, K, dtype=torch.float32)
    v32 = torch.randn(1, T, H, V, dtype=torch.float32)
    g32 = torch.randn(1, T, H, K, dtype=torch.float32) * 0.1
    cu_t = torch.tensor(cu_list, dtype=torch.long)

    # GPU Triton: fp16 + varlen
    q16, k16, v16, g16 = q32.half(), k32.half(), v32.half(), g32.half()
    o_tri, _ = triton_chunk_gla(
        q16.cuda(), k16.cuda(), v16.cuda(), g=g16.cuda(),
        cu_seqlens=cu_t.cuda(), output_final_state=False,
    )
    o_tri_f32 = o_tri.cpu().float()

    # CPU ref: fp32 + varlen
    def t2j(t): return jnp.array(t.detach().cpu().numpy(), dtype=jnp.float32)
    o_cpu, _ = chunk_gla(
        t2j(q32), t2j(k32), t2j(v32), g=t2j(g32),
        cu_seqlens=jnp.array(cu_list),
        output_final_state=False, chunk_size=C,
    )
    o_cpu_f32 = torch.from_numpy(np.array(o_cpu, dtype=np.float32))

    diff = (o_tri_f32 - o_cpu_f32).abs()
    diff_np = diff.numpy()

    denom = torch.maximum(o_tri_f32.abs(), o_cpu_f32.abs()).clamp(min=1e-38)
    rel_np = (diff / denom).numpy()

    ref = torch.maximum(o_tri_f32.abs(), o_cpu_f32.abs())
    ulp_fp32_diff = (diff / (compute_ulp(ref, dtype=torch.float32) + 1e-38)).numpy()
    ulp_fp16_diff = (diff / (compute_ulp(ref, dtype=torch.float16) + 1e-38)).numpy()

    # Find location of max relative error and print surrounding values
    max_rel_idx = np.unravel_index(np.argmax(rel_np), rel_np.shape)
    tri_val = float(o_tri_f32.numpy()[max_rel_idx])
    cpu_val = float(o_cpu_f32.numpy()[max_rel_idx])
    print(f"  MaxRel @ {max_rel_idx}: GPU={tri_val:.6e}  CPU={cpu_val:.6e}  diff={abs(tri_val-cpu_val):.6e}  rel={rel_np[max_rel_idx]:.4f}")

    # ULP16 and Rel breakdown by output magnitude
    ref_np = ref.numpy()
    for lo, hi in [(0, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, float("inf"))]:
        mask = (ref_np >= lo) & (ref_np < hi)
        if mask.sum() == 0:
            continue
        u = ulp_fp16_diff[mask]
        r = rel_np[mask]
        print(f"  |ref| in [{lo:.0e},{hi:.0e}): n={mask.sum():6d}"
              f"  RelMean={r.mean():.2e}  RelP99={np.percentile(r,99):.2e}"
              f"  ULP16 mean={u.mean():.2f}  p99={np.percentile(u,99):.2f}  max={u.max():.1f}")

    stats = {}
    for name, arr in [("abs", diff_np), ("rel", rel_np),
                      ("ulp_fp32", ulp_fp32_diff), ("ulp_fp16", ulp_fp16_diff)]:
        stats[name] = (float(arr.max()), float(arr.mean()), float(np.percentile(arr, 99)))
    return stats


hdr = (f"{'Case':<30}"
       f"  {'MaxAbs':>9} {'MeanAbs':>9} {'P99Abs':>9}"
       f"  {'MaxRel':>8} {'MeanRel':>8} {'P99Rel':>8}"
       f"  {'MaxULP32':>9} {'MeanULP32':>10} {'P99ULP32':>9}"
       f"  {'MaxULP16':>9} {'MeanULP16':>10} {'P99ULP16':>9}")
print("GPU fp16 varlen vs CPU fp32 varlen")
print(hdr)
print("-" * len(hdr))
for cfg in CASES:
    s = run(cfg)
    print(f"{cfg['label']:<30}"
          f"  {s['abs'][0]:>9.2e} {s['abs'][1]:>9.2e} {s['abs'][2]:>9.2e}"
          f"  {s['rel'][0]:>8.2e} {s['rel'][1]:>8.2e} {s['rel'][2]:>8.2e}"
          f"  {s['ulp_fp32'][0]:>9.1f} {s['ulp_fp32'][1]:>10.3f} {s['ulp_fp32'][2]:>9.2f}"
          f"  {s['ulp_fp16'][0]:>9.1f} {s['ulp_fp16'][1]:>10.3f} {s['ulp_fp16'][2]:>9.2f}")
