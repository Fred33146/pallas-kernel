
import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.utils import compare_tensor

from tops.cpu.ops.kda.chunk_bwd import (
    chunk_kda_bwd_wy_dqkg_fused as jax_chunk_kda_bwd_wy_dqkg_fused,
)
SEED = 40

HAS_CUDA = torch.cuda.is_available()

HAS_FLA = False
try:
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd as triton_chunk_kda_fwd
    from fla.ops.kda.chunk_bwd import (
        chunk_kda_bwd as triton_chunk_kda_bwd,
        chunk_kda_bwd_dAv as triton_chunk_kda_bwd_dAv,
        chunk_kda_bwd_wy_dqkg_fused as triton_chunk_kda_bwd_wy_dqkg_fused,
    )
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

    # Run Triton forward to get Aqk, Akk
    (o, final_state, g_cumsum, Aqk, Akk,
     w, u, qg, kg, v_new, h, initial_state) = triton_chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
        disable_recompute=True
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
