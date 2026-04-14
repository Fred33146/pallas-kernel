"""Tests for chunk_kda_bwd_dAv and chunk_gated_delta_rule_bwd_dhu Pallas kernels.

Compares Pallas kernel outputs against FLA (Triton) reference with both
float32 and bfloat16 inputs to assess precision differences.
"""

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.utils import compare_tensor, torch_to_jax

from tops.ops.kda.chunk_bwd import (
    chunk_kda_bwd_dAv_kernel as pallas_dAv,
    chunk_gated_delta_rule_bwd_dhu_kernel as pallas_dhu,
)

SEED = 40

HAS_CUDA = torch.cuda.is_available()

HAS_FLA = False
try:
    from fla.ops.kda.chunk import chunk_kda_fwd as triton_chunk_kda_fwd
    from fla.ops.kda.chunk_bwd import (
        chunk_kda_bwd_dAv as triton_chunk_kda_bwd_dAv,
    )
    from fla.ops.kda.wy_fast import recompute_w_u_fwd as triton_recompute_w_u_fwd
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu as triton_chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h as triton_chunk_gated_delta_rule_fwd_h,
    )
    from fla.utils import device

    HAS_FLA = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)


def _generate_inputs(B, T, H, K, V=None, chunk_size=64,
                     gate_logit_normalizer=1, dtype=torch.float32):
    """Generate random inputs and run Triton forward + recompute intermediates.

    Returns a dict with all raw inputs and the intermediates needed by the
    backward kernels (Aqk, Akk, w, u, qg, kg, v_new, h, etc.).
    """
    if V is None:
        V = K
    torch.manual_seed(SEED)

    input_scale = 0.1
    q = input_scale * torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = input_scale * torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = input_scale * torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32, device=device)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = 0.01 * torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    # Run forward to get Aqk, Akk
    o, Aqk, Akk, final_state = triton_chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    # Recompute intermediates (same as chunk_kda_bwd does internally)
    w, u, qg, kg = triton_recompute_w_u_fwd(
        q=q, k=k, v=v, beta=beta, A=Akk, gk=g,
    )
    h, v_new, _ = triton_chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        initial_state=h0,
        output_final_state=False,
        use_exp2=True,
    )

    do = input_scale * torch.randn_like(o)
    dht = 0.01 * torch.randn_like(h0)

    return dict(
        q=q, k=k, v=v, g=g, beta=beta,
        h0=h0, scale=scale,
        Aqk=Aqk, Akk=Akk,
        w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h,
        initial_state=h0,
        o=o, final_state=final_state,
        do=do, dht=dht,
        chunk_size=chunk_size,
    )


# =====================================================================
# Tests for chunk_kda_bwd_dAv
# =====================================================================

@requires_triton
class TestChunkKdaBwdDAvPallas:
    """chunk_kda_bwd_dAv: Pallas vs Triton (FLA), float32 and bfloat16."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 64, 64),
        (2, 128, 2, 64, 64),
        (1, 128, 4, 32, 32),
        (2, 64, 2, 32, 64),
    ])
    def test_dAv_float32(self, B, T, H, K, V):
        """Float32 input: Pallas dAv vs Triton dAv."""
        data = _generate_inputs(B, T, H, K, V, dtype=torch.float32)

        # Triton ground truth
        dA_ref, dv_ref = triton_chunk_kda_bwd_dAv(
            q=data["q"], k=data["k"], v=data["v_new"],
            do=data["do"], A=data["Aqk"],
            scale=data["scale"], chunk_size=data["chunk_size"],
        )

        # Pallas under test
        dA_pallas, dv_pallas = pallas_dAv(
            q=torch_to_jax(data["q"]),
            k=torch_to_jax(data["k"]),
            v=torch_to_jax(data["v_new"]),
            do=torch_to_jax(data["do"]),
            A=torch_to_jax(data["Aqk"]),
            scale=data["scale"],
            chunk_size=data["chunk_size"],
        )

        assert compare_tensor("dA (fp32)", dA_pallas, dA_ref,
                              atol=5e-5, rtol=5e-5)
        assert compare_tensor("dv (fp32)", dv_pallas, dv_ref,
                              atol=5e-5, rtol=5e-5)

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 64, 64),
        (2, 128, 2, 64, 64),
        (1, 128, 4, 32, 32),
        (2, 64, 2, 32, 64),
    ])
    def test_dAv_bfloat16(self, B, T, H, K, V):
        """Bfloat16 input: Pallas dAv vs Triton dAv (relaxed tolerance)."""
        data = _generate_inputs(B, T, H, K, V, dtype=torch.bfloat16)

        dA_ref, dv_ref = triton_chunk_kda_bwd_dAv(
            q=data["q"], k=data["k"], v=data["v_new"],
            do=data["do"], A=data["Aqk"],
            scale=data["scale"], chunk_size=data["chunk_size"],
        )

        dA_pallas, dv_pallas = pallas_dAv(
            q=torch_to_jax(data["q"], dtype=jnp.bfloat16),
            k=torch_to_jax(data["k"], dtype=jnp.bfloat16),
            v=torch_to_jax(data["v_new"], dtype=jnp.bfloat16),
            do=torch_to_jax(data["do"], dtype=jnp.bfloat16),
            A=torch_to_jax(data["Aqk"], dtype=jnp.bfloat16),
            scale=data["scale"],
            chunk_size=data["chunk_size"],
        )

        # bfloat16 has ~7-bit mantissa, so tolerance is much larger
        assert compare_tensor("dA (bf16)", dA_pallas, dA_ref,
                              atol=1e-2, rtol=1e-2, dtype=torch.bfloat16)
        assert compare_tensor("dv (bf16)", dv_pallas, dv_ref,
                              atol=1e-2, rtol=1e-2, dtype=torch.bfloat16)

    def test_dAv_shapes_float32(self):
        B, T, H, K, V = 2, 128, 2, 64, 32
        BT = 64
        data = _generate_inputs(B, T, H, K, V, dtype=torch.float32)

        dA, dv = pallas_dAv(
            q=torch_to_jax(data["q"]),
            k=torch_to_jax(data["k"]),
            v=torch_to_jax(data["v_new"]),
            do=torch_to_jax(data["do"]),
            A=torch_to_jax(data["Aqk"]),
            scale=data["scale"],
            chunk_size=BT,
        )
        assert dA.shape == (B, T, H, BT)
        assert dv.shape == (B, T, H, V)
        assert dA.dtype == jnp.float32

    def test_dA_causal_masking(self):
        """dA should be zero in the strictly upper triangle (causal mask)."""
        B, T, H, K, V = 1, 64, 1, 64, 64
        BT = 64
        data = _generate_inputs(B, T, H, K, V)

        dA_ref, _ = triton_chunk_kda_bwd_dAv(
            q=data["q"], k=data["k"], v=data["v_new"],
            do=data["do"], A=data["Aqk"],
            scale=data["scale"], chunk_size=BT,
        )
        dA_pallas, _ = pallas_dAv(
            q=torch_to_jax(data["q"]),
            k=torch_to_jax(data["k"]),
            v=torch_to_jax(data["v_new"]),
            do=torch_to_jax(data["do"]),
            A=torch_to_jax(data["Aqk"]),
            scale=data["scale"],
            chunk_size=BT,
        )

        NT = T // BT
        for t_idx in range(NT):
            ts, te = t_idx * BT, (t_idx + 1) * BT
            o = jnp.arange(BT)
            upper_mask = o[:, None] < o[None, :]  # strictly upper
            ref_vals = jnp.array(dA_ref[0, ts:te, 0, :BT].cpu().numpy())[upper_mask]
            jax_vals = dA_pallas[0, ts:te, 0, :BT][upper_mask]
            assert jnp.allclose(ref_vals, 0.0, atol=1e-6)
            assert jnp.allclose(jax_vals, 0.0, atol=1e-6)

    def test_input_validation(self):
        """Shape assertions catch mismatched inputs."""
        B, T, H, K, V = 1, 64, 1, 32, 32
        data = _generate_inputs(B, T, H, K, V)

        with pytest.raises(AssertionError):
            pallas_dAv(
                q=torch_to_jax(data["q"]),
                k=torch_to_jax(data["k"]),
                v=jnp.ones((B, T, H, V + 1)),  # wrong V dim
                do=torch_to_jax(data["do"]),
                A=torch_to_jax(data["Aqk"]),
                scale=data["scale"],
                chunk_size=data["chunk_size"],
            )

        with pytest.raises(AssertionError):
            bad_T = 65
            pallas_dAv(
                q=jnp.ones((B, bad_T, H, K)),
                k=jnp.ones((B, bad_T, H, K)),
                v=jnp.ones((B, bad_T, H, V)),
                do=jnp.ones((B, bad_T, H, V)),
                A=jnp.ones((B, bad_T, H, 64)),
                scale=data["scale"],
                chunk_size=64,
            )


# =====================================================================
# Tests for chunk_gated_delta_rule_bwd_dhu
# =====================================================================

@requires_triton
class TestChunkGatedDeltaRuleBwdDhuPallas:
    """chunk_gated_delta_rule_bwd_dhu: Pallas vs Triton (FLA), float32 and bfloat16."""

    def _run_triton_dAv(self, data):
        """Get dv from the Triton dAv stage (needed as input to dhu)."""
        _, dv = triton_chunk_kda_bwd_dAv(
            q=data["q"], k=data["k"], v=data["v_new"],
            do=data["do"], A=data["Aqk"],
            scale=data["scale"], chunk_size=data["chunk_size"],
        )
        return dv

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 64, 64),
        (2, 128, 2, 64, 64),
        (1, 128, 4, 32, 32),
        (2, 64, 2, 32, 64),
    ])
    def test_dhu_float32(self, B, T, H, K, V):
        """Float32 input: Pallas dhu vs Triton dhu."""
        data = _generate_inputs(B, T, H, K, V, dtype=torch.float32)
        dv = self._run_triton_dAv(data)

        # Triton ground truth
        dh_ref, dh0_ref, dv2_ref = triton_chunk_gated_delta_rule_bwd_dhu(
            q=data["qg"], k=data["kg"], w=data["w"],
            gk=data["g"],
            h0=data["initial_state"],
            dht=data["dht"],
            do=data["do"], dv=dv,
            scale=data["scale"],
            use_exp2=True,
        )

        # Pallas under test
        dh_pallas, dh0_pallas, dv2_pallas = pallas_dhu(
            q=torch_to_jax(data["qg"]),
            k=torch_to_jax(data["kg"]),
            w=torch_to_jax(data["w"]),
            gk=torch_to_jax(data["g"]),
            h0=torch_to_jax(data["initial_state"]),
            dht=torch_to_jax(data["dht"]),
            do=torch_to_jax(data["do"]),
            dv=torch_to_jax(dv),
            scale=data["scale"],
            chunk_size=data["chunk_size"],
        )

        # dh: Triton returns [B*NT, H, K, V], Pallas returns [B, NT, H, K, V]
        NT = T // data["chunk_size"]
        assert compare_tensor("dh (fp32)", dh_pallas,
                              dh_ref.reshape(B, NT, H, K, V),
                              atol=1e-4, rtol=1e-4)
        assert compare_tensor("dh0 (fp32)", dh0_pallas, dh0_ref,
                              atol=1e-4, rtol=1e-4)
        assert compare_tensor("dv2 (fp32)", dv2_pallas, dv2_ref,
                              atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 64, 64),
        (2, 128, 2, 64, 64),
        (1, 128, 4, 32, 32),
        (2, 64, 2, 32, 64),
    ])
    def test_dhu_bfloat16(self, B, T, H, K, V):
        """Bfloat16 input: Pallas dhu vs Triton dhu (relaxed tolerance)."""
        data = _generate_inputs(B, T, H, K, V, dtype=torch.bfloat16)
        dv = self._run_triton_dAv(data)

        dh_ref, dh0_ref, dv2_ref = triton_chunk_gated_delta_rule_bwd_dhu(
            q=data["qg"], k=data["kg"], w=data["w"],
            gk=data["g"],
            h0=data["initial_state"],
            dht=data["dht"],
            do=data["do"], dv=dv,
            scale=data["scale"],
            use_exp2=True,
        )

        dh_pallas, dh0_pallas, dv2_pallas = pallas_dhu(
            q=torch_to_jax(data["qg"], dtype=jnp.bfloat16),
            k=torch_to_jax(data["kg"], dtype=jnp.bfloat16),
            w=torch_to_jax(data["w"], dtype=jnp.bfloat16),
            gk=torch_to_jax(data["g"]),  # gk stays float32
            h0=torch_to_jax(data["initial_state"]),
            dht=torch_to_jax(data["dht"]),
            do=torch_to_jax(data["do"], dtype=jnp.bfloat16),
            dv=torch_to_jax(dv, dtype=jnp.bfloat16),
            scale=data["scale"],
            chunk_size=data["chunk_size"],
        )

        NT = T // data["chunk_size"]
        assert compare_tensor("dh (bf16)", dh_pallas,
                              dh_ref.reshape(B, NT, H, K, V),
                              atol=5e-2, rtol=5e-2, dtype=torch.bfloat16)
        assert compare_tensor("dh0 (bf16)", dh0_pallas, dh0_ref,
                              atol=5e-2, rtol=5e-2, dtype=torch.bfloat16)
        assert compare_tensor("dv2 (bf16)", dv2_pallas, dv2_ref,
                              atol=5e-2, rtol=5e-2, dtype=torch.bfloat16)

    @pytest.mark.parametrize("with_dht", [True, False])
    def test_dhu_with_and_without_dht(self, with_dht):
        """Test with dht=None vs dht provided."""
        B, T, H, K, V = 1, 64, 1, 64, 64
        data = _generate_inputs(B, T, H, K, V)
        dv = self._run_triton_dAv(data)

        dht_triton = data["dht"] if with_dht else None
        dht_jax = torch_to_jax(data["dht"]) if with_dht else None

        dh_ref, dh0_ref, dv2_ref = triton_chunk_gated_delta_rule_bwd_dhu(
            q=data["qg"], k=data["kg"], w=data["w"],
            gk=data["g"],
            h0=data["initial_state"],
            dht=dht_triton,
            do=data["do"], dv=dv,
            scale=data["scale"],
            use_exp2=True,
        )

        dh_pallas, dh0_pallas, dv2_pallas = pallas_dhu(
            q=torch_to_jax(data["qg"]),
            k=torch_to_jax(data["kg"]),
            w=torch_to_jax(data["w"]),
            gk=torch_to_jax(data["g"]),
            h0=torch_to_jax(data["initial_state"]),
            dht=dht_jax,
            do=torch_to_jax(data["do"]),
            dv=torch_to_jax(dv),
            scale=data["scale"],
            chunk_size=data["chunk_size"],
        )

        BT = data["chunk_size"]
        NT = T // BT
        assert compare_tensor(f"dh (dht={'yes' if with_dht else 'no'})",
                              dh_pallas, dh_ref.reshape(B, NT, H, K, V),
                              atol=1e-4, rtol=1e-4)
        assert compare_tensor(f"dh0 (dht={'yes' if with_dht else 'no'})",
                              dh0_pallas, dh0_ref,
                              atol=1e-4, rtol=1e-4)
        assert compare_tensor(f"dv2 (dht={'yes' if with_dht else 'no'})",
                              dv2_pallas, dv2_ref,
                              atol=1e-4, rtol=1e-4)

    def test_dhu_shapes(self):
        B, T, H, K, V = 2, 128, 2, 64, 32
        BT = 64
        NT = T // BT
        data = _generate_inputs(B, T, H, K, V)
        dv = self._run_triton_dAv(data)

        dh, dh0, dv2 = pallas_dhu(
            q=torch_to_jax(data["qg"]),
            k=torch_to_jax(data["kg"]),
            w=torch_to_jax(data["w"]),
            gk=torch_to_jax(data["g"]),
            h0=torch_to_jax(data["initial_state"]),
            dht=torch_to_jax(data["dht"]),
            do=torch_to_jax(data["do"]),
            dv=torch_to_jax(dv),
            scale=data["scale"],
            chunk_size=BT,
        )
        assert dh.shape == (B, NT, H, K, V)
        assert dh0.shape == (B, H, K, V)
        assert dv2.shape == (B, T, H, V)
        assert dh.dtype == jnp.float32
        assert dh0.dtype == jnp.float32

    def test_input_validation(self):
        """Shape assertions catch mismatched inputs."""
        B, T, H, K, V = 1, 64, 1, 32, 32
        data = _generate_inputs(B, T, H, K, V)
        dv = self._run_triton_dAv(data)

        with pytest.raises(AssertionError):
            pallas_dhu(
                q=torch_to_jax(data["qg"]),
                k=torch_to_jax(data["kg"]),
                w=jnp.ones((B, T, H, K + 1)),  # wrong K dim
                gk=torch_to_jax(data["g"]),
                h0=torch_to_jax(data["initial_state"]),
                dht=torch_to_jax(data["dht"]),
                do=torch_to_jax(data["do"]),
                dv=torch_to_jax(dv),
                scale=data["scale"],
                chunk_size=data["chunk_size"],
            )

        with pytest.raises(AssertionError):
            pallas_dhu(
                q=torch_to_jax(data["qg"]),
                k=torch_to_jax(data["kg"]),
                w=torch_to_jax(data["w"]),
                gk=torch_to_jax(data["g"]),
                h0=torch_to_jax(data["initial_state"]),
                dht=jnp.ones((B, H, K + 1, V)),  # wrong dht shape
                do=torch_to_jax(data["do"]),
                dv=torch_to_jax(dv),
                scale=data["scale"],
                chunk_size=data["chunk_size"],
            )
