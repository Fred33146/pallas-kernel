import os

os.environ["TRITON_F32_DEFAULT"] = "ieee"

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp

from tests.utils import compare_tensor

from tops.cpu.ops.kda.chunk_bwd import (
    chunk_kda_bwd as jax_chunk_kda_bwd,
    chunk_kda_bwd_wy_dqkg_fused as jax_chunk_kda_bwd_wy_dqkg_fused,
)
SEED = 40

HAS_CUDA = torch.cuda.is_available()

HAS_FLA = False
try:
    from fla.ops.kda.chunk_bwd import (
        chunk_kda_bwd as triton_chunk_kda_bwd,
        chunk_kda_bwd_dAv as triton_chunk_kda_bwd_dAv,
        chunk_kda_bwd_wy_dqkg_fused as triton_chunk_kda_bwd_wy_dqkg_fused,
    )
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd as triton_chunk_kda_fwd
    from fla.utils import device

    HAS_FLA = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """PyTorch CUDA tensor -> JAX array (via CPU)."""
    return jnp.array(t.detach().cpu().float().numpy())

def _generate_fwd_inputs(B, T, H, K, V=None, chunk_size=64,
                         gate_logit_normalizer=1, dtype=torch.float32):
    """Generate random inputs and run Triton forward to get all intermediates."""
    if V is None:
        V = K
    torch.manual_seed(SEED)

    # Scale down inputs to prevent overflow in recurrent state accumulation,
    # especially for larger B*T*H*K or weak gate decay (high gate_logit_normalizer).
    input_scale = 0.1
    q = input_scale * torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = input_scale * torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = input_scale * torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32, device=device)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = 0.01 * torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    # Run Triton forward with disable_recompute=True to keep all intermediates
    (o, final_state, g_cumsum, Aqk, Akk,
     w, u, qg, kg, v_new, h, initial_state) = triton_chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
        disable_recompute=True,
    )

    do = input_scale * torch.randn_like(o)
    dht = 0.01 * torch.randn_like(h0)

    return dict(
        q=q, k=k, v=v, g=g, g_cumsum=g_cumsum, beta=beta,
        h0=h0, scale=scale,
        Aqk=Aqk, Akk=Akk,
        w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h,
        initial_state=initial_state,
        o=o, final_state=final_state,
        do=do, dht=dht,
        chunk_size=chunk_size,
    )

@requires_triton
@pytest.mark.skip(reason="Internal sub-function test; run e2e tests below instead")
class TestChunkKdaBwdWyDqkgFused:
    """chunk_kda_bwd_wy_dqkg_fused: JAX vs Triton, with real forward-pass data."""

    def _run_dhu_to_get_dh_dv(self, data):
        """Run the full Triton backward up to the dhu stage to get dh and dv
        that feed into the WY fused kernel."""
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

        # First get dAqk, dv from dAv stage
        _, dv = triton_chunk_kda_bwd_dAv(
            q=data["q"], k=data["k"], v=data["v_new"],
            do=data["do"], A=data["Aqk"],
            scale=data["scale"], chunk_size=data["chunk_size"],
        )

        # Then get dh, dv from dhu stage
        dh, _dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=data["qg"], k=data["kg"], w=data["w"],
            gk=data["g_cumsum"],
            h0=data["initial_state"],
            dht=data["dht"],
            do=data["do"], dv=dv,
            scale=data["scale"],
            use_exp2=True,
        )
        return dh, dv

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 64, 64),
        (2, 128, 2, 64, 64),
        (1, 192, 4, 32, 32),
        (2, 64, 2, 32, 64),
        (2, 2048, 8, 128, 128),
    ])
    def test_against_triton(self, B, T, H, K, V):
        data = _generate_fwd_inputs(B, T, H, K, V)
        dh, dv = self._run_dhu_to_get_dh_dv(data)

        # Triton ground truth
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dA_ref = \
            triton_chunk_kda_bwd_wy_dqkg_fused(
                q=data["q"], k=data["k"], v=data["v"], v_new=data["v_new"],
                g=data["g_cumsum"], beta=data["beta"], A=data["Akk"],
                h=data["h"], do=data["do"], dh=dh, dv=dv,
                scale=data["scale"], chunk_size=data["chunk_size"],
            )

        # JAX under test
        dq_jax, dk_jax, dv_jax, db_jax, dg_jax, dA_jax = \
            jax_chunk_kda_bwd_wy_dqkg_fused(
                q=torch_to_jax(data["q"]),
                k=torch_to_jax(data["k"]),
                v=torch_to_jax(data["v"]),
                v_new=torch_to_jax(data["v_new"]),
                g=torch_to_jax(data["g_cumsum"]),
                beta=torch_to_jax(data["beta"]),
                A=torch_to_jax(data["Akk"]),
                h=torch_to_jax(data["h"]),
                do=torch_to_jax(data["do"]),
                dh=torch_to_jax(dh),
                dv=torch_to_jax(dv),
                scale=data["scale"],
                chunk_size=data["chunk_size"],
            )

        compare_tensor("dq", dq_jax, dq_ref)
        compare_tensor("dk", dk_jax, dk_ref)
        compare_tensor("dv", dv_jax, dv_ref)
        compare_tensor("db", db_jax, db_ref)
        compare_tensor("dg", dg_jax, dg_ref)
        compare_tensor("dA", dA_jax, dA_ref)

    def test_shapes(self):
        B, T, H, K, V = 2, 128, 2, 64, 32
        data = _generate_fwd_inputs(B, T, H, K, V)
        dh, dv = self._run_dhu_to_get_dh_dv(data)
        BT = data["chunk_size"]

        dq, dk, dv2, db, dg, dA = jax_chunk_kda_bwd_wy_dqkg_fused(
            q=torch_to_jax(data["q"]),
            k=torch_to_jax(data["k"]),
            v=torch_to_jax(data["v"]),
            v_new=torch_to_jax(data["v_new"]),
            g=torch_to_jax(data["g_cumsum"]),
            beta=torch_to_jax(data["beta"]),
            A=torch_to_jax(data["Akk"]),
            h=torch_to_jax(data["h"]),
            do=torch_to_jax(data["do"]),
            dh=torch_to_jax(dh),
            dv=torch_to_jax(dv),
            scale=data["scale"],
            chunk_size=BT,
        )
        assert dq.shape == (B, T, H, K)
        assert dk.shape == (B, T, H, K)
        assert dv2.shape == (B, T, H, V)
        assert db.shape == (B, T, H)
        assert dg.shape == (B, T, H, K)
        assert dA.shape == (B, T, H, BT)

    def test_dA_strictly_lower_triangular(self):
        """dA from WY kernel should be strictly lower-triangular."""
        B, T, H, K = 1, 64, 1, 64
        data = _generate_fwd_inputs(B, T, H, K)
        dh, dv = self._run_dhu_to_get_dh_dv(data)
        BT = data["chunk_size"]

        _, _, _, _, _, dA_ref = triton_chunk_kda_bwd_wy_dqkg_fused(
            q=data["q"], k=data["k"], v=data["v"], v_new=data["v_new"],
            g=data["g_cumsum"], beta=data["beta"], A=data["Akk"],
            h=data["h"], do=data["do"], dh=dh, dv=dv,
            scale=data["scale"], chunk_size=BT,
        )
        _, _, _, _, _, dA_jax = jax_chunk_kda_bwd_wy_dqkg_fused(
            q=torch_to_jax(data["q"]),
            k=torch_to_jax(data["k"]),
            v=torch_to_jax(data["v"]),
            v_new=torch_to_jax(data["v_new"]),
            g=torch_to_jax(data["g_cumsum"]),
            beta=torch_to_jax(data["beta"]),
            A=torch_to_jax(data["Akk"]),
            h=torch_to_jax(data["h"]),
            do=torch_to_jax(data["do"]),
            dh=torch_to_jax(dh),
            dv=torch_to_jax(dv),
            scale=data["scale"],
            chunk_size=BT,
        )

        NT = (T + BT - 1) // BT
        for t_idx in range(NT):
            ts, te = t_idx * BT, min((t_idx + 1) * BT, T)
            bt = te - ts
            o = jnp.arange(bt)
            upper_diag = o[:, None] <= o[None, :]
            ref_vals = jnp.array(dA_ref[0, ts:te, 0, :bt].cpu().numpy())[upper_diag]
            jax_vals = dA_jax[0, ts:te, 0, :bt][upper_diag]
            assert jnp.allclose(ref_vals, 0.0, atol=1e-6)
            assert jnp.allclose(jax_vals, 0.0, atol=1e-6)


# ============================================================================
# End-to-end chunk_kda_bwd: JAX CPU ref vs Triton
# ============================================================================

_E2E_SHAPES = [
    dict(B=1, T=64, H=1, K=64, V=64, seed=40, with_h0=True),
    dict(B=2, T=128, H=2, K=64, V=64, seed=41, with_h0=True),
    dict(B=1, T=192, H=4, K=32, V=32, seed=42, with_h0=True),
    dict(B=1, T=4096, H=4, K=128, V=128, seed=43, with_h0=True),
    dict(B=1, T=4096, H=4, K=128, V=128, seed=44, with_h0=False),
    dict(B=1, T=4096, H=8, K=128, V=128, seed=45, with_h0=True),
]


def _e2e_id(c):
    return f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_s{c.get('seed', SEED)}"


def _generate_e2e_inputs(B, T, H, K, V=None, chunk_size=64, seed=SEED, with_h0=True, **kw):
    """Generate inputs and run Triton forward to get intermediates for e2e bwd."""
    if V is None:
        V = K
    # Ensure CUDA determinism across sequential test cases
    torch.cuda.synchronize()
    torch.manual_seed(seed)
    input_scale = 0.5
    q = input_scale * torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = input_scale * torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = input_scale * torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    h0 = 0.01 * torch.randn(B, H, K, V, dtype=torch.float32, device=device) if with_h0 else None
    scale = K ** -0.5

    (o, final_state, g_cumsum, Aqk, Akk,
     w, u, qg, kg, v_new, h, initial_state) = triton_chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        chunk_size=chunk_size, disable_recompute=True,
    )

    do = input_scale * torch.randn_like(o)
    dht = 0.01 * torch.randn_like(h0) if h0 is not None else None

    return dict(
        q=q, k=k, v=v, g_cumsum=g_cumsum, beta=beta,
        h0=h0, scale=scale, Aqk=Aqk, Akk=Akk,
        w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h,
        initial_state=initial_state,
        do=do, dht=dht, chunk_size=chunk_size,
    )


@requires_triton
class TestChunkKdaBwdE2E:
    """End-to-end chunk_kda_bwd: JAX vs Triton, comparing all gradient outputs."""

    def _run_and_compare(self, data, atol=2e-3, rtol=2e-3):
        BT = data["chunk_size"]

        # ---- Triton ----
        # Use disable_recompute=True with saved intermediates from forward.
        # Both sides use identical intermediates to avoid GPU/CPU recomputation drift.
        dq_tri, dk_tri, dv_tri, db_tri, dg_tri, dh0_tri, _, _ = triton_chunk_kda_bwd(
            q=data["q"], k=data["k"], v=data["v"],
            g=data["g_cumsum"], beta=data["beta"],
            Aqk=data["Aqk"], Akk=data["Akk"],
            scale=data["scale"], initial_state=data["initial_state"],
            do=data["do"], dht=data["dht"], chunk_size=BT,
            disable_recompute=True,
            w=data["w"], u=data["u"], qg=data["qg"],
            kg=data["kg"], v_new=data["v_new"], h=data["h"],
        )

        # Move Triton results to CPU and free GPU memory before JAX run
        def _to_np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().float().numpy()
            return np.asarray(t, dtype=np.float32)

        tri_np = {
            "dq": _to_np(dq_tri), "dk": _to_np(dk_tri), "dv": _to_np(dv_tri),
            "db": _to_np(db_tri), "dg": _to_np(dg_tri),
            "dh0": _to_np(dh0_tri) if dh0_tri is not None else None,
        }
        del dq_tri, dk_tri, dv_tri, db_tri, dg_tri, dh0_tri

        # Convert data to JAX (CPU) and release GPU tensors
        jax_data = {
            k: torch_to_jax(v) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        data.clear()
        torch.cuda.empty_cache()

        # ---- JAX ----
        # Also use disable_recompute=True with the same Triton-forward intermediates.
        dq_jax, dk_jax, dv_jax, db_jax, dg_jax, dh0_jax, _, _ = jax_chunk_kda_bwd(
            q=jax_data["q"], k=jax_data["k"],
            v=jax_data["v"], beta=jax_data["beta"],
            Aqk=jax_data["Aqk"], Akk=jax_data["Akk"],
            scale=jax_data["scale"],
            initial_state=jax_data["initial_state"],
            do=jax_data["do"],
            dht=jax_data["dht"],
            g=jax_data["g_cumsum"], chunk_size=BT,
            disable_recompute=True,
            w=jax_data["w"], u=jax_data["u"],
            qg=jax_data["qg"], kg=jax_data["kg"],
            v_new=jax_data["v_new"], h=jax_data["h"],
        )

        for name, jax_val in [("dq", dq_jax), ("dk", dk_jax), ("dv", dv_jax),
                               ("db", db_jax), ("dg", dg_jax)]:
            assert compare_tensor(name, _to_np(jax_val), tri_np[name], atol=atol, rtol=rtol)

        if tri_np["dh0"] is not None:
            assert dh0_jax is not None
            assert compare_tensor("dh0", _to_np(dh0_jax), tri_np["dh0"], atol=atol, rtol=rtol)

    @pytest.mark.parametrize("cfg", _E2E_SHAPES, ids=[_e2e_id(c) for c in _E2E_SHAPES])
    def test_e2e(self, cfg):
        torch.cuda.empty_cache()
        data = _generate_e2e_inputs(**cfg)
        self._run_and_compare(data)

if __name__ == "__main__":
  print("HAS_FLA = ", HAS_FLA)
  pytest.main([__file__, "-V", "-s", "-x"])
