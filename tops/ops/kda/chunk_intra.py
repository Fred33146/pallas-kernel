"""KDA intra-chunk kernel: triangular system solve for delta-rule dependencies.

Within each chunk, the delta rule creates dependencies between positions
because each position's value correction depends on the current state,
which itself depends on all previous positions. This kernel solves the
resulting lower-triangular system using block-based forward substitution.

Outputs (forward):
  u: [B, H, T, V]           — delta-corrected values
  w: [B, H, T, K]           — correction weights for inter-chunk state
  qg: [B, H, T, K]          — q * exp2(g)
  kg: [B, H, T, K]          — k * exp2(g_last - g)
  Aqk: [B, H, NC, C, C]     — query-key attention matrix
  Akk_inv: [B, H, NC, C, C] — inverted key-key matrix
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tops.ops.utils import get_interpret
from tops.utils import assert_shape


def solve_unit_lower_triangular(A, b):
    """
    Solves (I + A) x = b for x, where A is strictly lower triangular.
    Uses block-based forward substitution for better performance on TPU.
    Args:
        A: (N, N) strictly lower triangular matrix in VMEM.
        b: (N, D) matrix in VMEM.
    Returns:
        x: (N, D) solution matrix.
    """
    N, D = b.shape
    B = 16
    num_blocks = N // B
    A = A.astype(jnp.float32)
    b = b.astype(jnp.float32)

    blocks = jnp.split(b, num_blocks, axis=0)

    for i in range(num_blocks):
        start = i * B
        end = (i + 1) * B

        A_ii = A[start:end, start:end]
        x_block = blocks[i]

        rows = [x_block[r] for r in range(B)]
        for j in range(B):
            if j > 0:
                vec = A_ii[j, :j][None, :]
                mat = jnp.stack(rows[:j])
                correction = jax.lax.dot_general(
                    vec, mat,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32
                ).squeeze(axis=0)
                rows[j] = rows[j] - correction

        x_block = jnp.stack(rows)
        blocks[i] = x_block

        if i < num_blocks - 1:
            rest_start = (i + 1) * B

            x_rest = jnp.concatenate(blocks[i+1:], axis=0)
            A_rest = A[rest_start:, start:end]

            update = jax.lax.dot_general(
                A_rest, x_block,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32
            )
            x_rest = x_rest - update

            remaining_blocks_count = num_blocks - 1 - i
            new_blocks = jnp.split(x_rest, remaining_blocks_count, axis=0)

            for k, nb in enumerate(new_blocks):
                blocks[i + 1 + k] = nb

    x = jnp.concatenate(blocks, axis=0)
    return x


def kda_intra_chunk_kernel(
    # Inputs (Ref)
    q_ref, k_ref, g_ref, beta_ref, v_ref, segment_ids_ref,
    # Outputs (Ref)
    u_out_ref, w_out_ref, qg_out_ref, kg_out_ref, Aqk_out_ref, Akk_inv_out_ref,
    # Config
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
):
    dtype = q_ref.dtype
    q = q_ref[0, 0, 0]          # (C, D)
    k = k_ref[0, 0, 0]          # (C, D)
    g = g_ref[0, 0, 0]          # (C, D)
    beta = beta_ref[0, 0, 0]    # (C, 1)
    v = v_ref[0, 0, 0]          # (C, V)
    segment_ids = segment_ids_ref[0, 0, 0, :, 0]

    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    causal_mask = idx[:, None] > idx[None, :]       # strictly lower triangular
    causal_mask_qk = idx[:, None] >= idx[None, :]   # lower triangular with diagonal

    # Segment mask: i and j must belong to the same segment
    segment_mask = segment_ids[:, None] == segment_ids[None, :]
    mask = causal_mask & segment_mask
    mask_qk = causal_mask_qk & segment_mask

    # Direct g_diff computation (numerically stable).
    # For causal entries (i >= j): g_cum is monotonically non-increasing,
    # so g_diff = g[i] - g[j] <= 0, thus exp(g_diff) in (0, 1]. No overflow.
    # For upper triangle (i < j): g_diff > 0 could overflow in exp.
    # Mask those to 0 before exp since they are zeroed out in Aqk/Akk anyway.
    g_f32 = g.astype(jnp.float32)
    g_diff = g_f32[:, None, :] - g_f32[None, :, :]                  # (C, C, D)
    g_diff_safe = jnp.where(causal_mask_qk[:, :, None], g_diff, 0.0)
    exp_g_diff = jnp.exp2(g_diff_safe)                               # (C, C, D)

    # Aqk[i,j] = scale * sum_d q[i,d] * k[j,d] * exp(g[i,d] - g[j,d])
    Aqk_raw = jnp.sum(
        q.astype(jnp.float32)[:, None, :] * k.astype(jnp.float32)[None, :, :] * exp_g_diff,
        axis=-1,
    ).astype(dtype)                                                  # (C, C)
    Aqk = jnp.where(mask_qk, Aqk_raw * scale, 0.0)

    # Akk[i,j] = beta[i] * sum_d k[i,d] * k[j,d] * exp(g[i,d] - g[j,d])
    Akk_raw = jnp.sum(
        k.astype(jnp.float32)[:, None, :] * k.astype(jnp.float32)[None, :, :] * exp_g_diff,
        axis=-1,
    ).astype(dtype)                                                  # (C, C)
    Akk = jnp.where(mask, Akk_raw * beta, 0.0)

    # Solve (I + Akk) x = b  for u, w, Akk_inv simultaneously
    v_scaled = v * beta                              # (C, V)
    target_w = k * jnp.exp2(g) * beta                # (C, D)
    identity = jnp.eye(chunk_size, dtype=v.dtype)     # (C, C)

    combined_b = jnp.concatenate([v_scaled, target_w, identity], axis=-1)  # (C, V+D+C)
    combined_x = solve_unit_lower_triangular(Akk, combined_b)

    u = combined_x[:, :value_dim]                     # (C, V)
    w = combined_x[:, value_dim:value_dim + head_dim] # (C, D)
    Akk_inv = combined_x[:, value_dim + head_dim:]    # (C, C)

    qg = q * jnp.exp2(g)                             # (C, D)

    g_last = g[chunk_size-1][None, :]                 # (1, D)
    kg = k * jnp.exp2(g_last - g)                     # (C, D)

    u_out_ref[0, 0, 0] = u.astype(u_out_ref.dtype)
    w_out_ref[0, 0, 0] = w.astype(w_out_ref.dtype)
    qg_out_ref[0, 0, 0] = qg
    kg_out_ref[0, 0, 0] = kg
    Aqk_out_ref[0, 0, 0] = Aqk
    Akk_inv_out_ref[0, 0, 0] = Akk_inv.astype(Akk_inv_out_ref.dtype)


@functools.partial(jax.jit, static_argnames=['chunk_size', 'scale'])
def kda_intra_chunk_fwd(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array = None,
    scale: float = 1.0,
    chunk_size: int = 128
):
    """Pallas TPU implementation of the KDA intra-chunk forward pass.

    Within each chunk the delta rule induces a lower-triangular system of
    dependencies.  This kernel solves that system via block forward
    substitution and simultaneously computes the attention matrices and
    gated projections needed by the inter-chunk pass.

    The gates g are assumed to already be in log2 space (chunk-local
    cumsum, scaled by 1/ln2), so all exponential operations inside the
    kernel use exp2 directly — consistent with FLA convention.

    Args:
        q:           [B, H, T, K] — query vectors.
        k:           [B, H, T, K] — key vectors.
        g:           [B, H, T, K] — chunk-local cumsum of gates in log2 space.
        beta:        [B, H, T]    — per-token scalar mixing coefficient.
        v:           [B, H, T, V] — value vectors (V may differ from K when
                                    expand_v > 1).
        segment_ids: [B, T]       — integer segment IDs for variable-length
                                    sequences.  Tokens in different segments
                                    do not attend to each other.  If None,
                                    all tokens are treated as one segment.
        scale:       float        — attention scale factor (default 1.0).
        chunk_size:  int          — tile size for the Pallas kernel grid.

    Returns:
        u:       [B, H, T, V]           — delta-corrected values.
        w:       [B, H, T, K]           — correction weights for inter-chunk
                                          state update.
        qg:      [B, H, T, K]           — q * exp2(g).
        kg:      [B, H, T, K]           — k * exp2(g_last - g).
        Aqk:     [B, H, NC, C, C]       — query-key attention matrix per chunk
                                          (NC = T // chunk_size, C = chunk_size).
        Akk_inv: [B, H, NC, C, C]       — solution of (I + Akk) X = I, i.e.
                                          the implicit inverse of the key-key
                                          Gram matrix per chunk.
    """
    B, H, T, K = k.shape
    V = v.shape[-1]  # value_dim, may differ from K when expand_v > 1
    assert T % chunk_size == 0, "Sequence length must be divisible by chunk_size"
    num_chunks = T // chunk_size

    assert_shape(q, (B, H, T, K), "q")
    assert_shape(k, (B, H, T, K), "k")
    assert_shape(g, (B, H, T, K), "g")
    assert_shape(beta, (B, H, T), "beta")
    assert_shape(v, (B, H, T, V), "v")

    if segment_ids is None:
        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

    # g is already in log2 space (cumsum scaled by 1/ln2 at _chunk_local_cumsum_kda)
    # so kernel uses exp2 directly, matching FLA convention.

    q_reshaped = q.reshape(B, H, num_chunks, chunk_size, K)
    k_reshaped = k.reshape(B, H, num_chunks, chunk_size, K)
    g_reshaped = g.reshape(B, H, num_chunks, chunk_size, K)
    beta_reshaped = beta.reshape(B, H, num_chunks, chunk_size, 1)
    v_reshaped = v.reshape(B, H, num_chunks, chunk_size, V)
    segment_ids_reshaped = segment_ids.reshape(B, 1, num_chunks, chunk_size, 1)

    grid = (B, H, num_chunks)

    u_reshaped, w_reshaped, qg_reshaped, kg_reshaped, Aqk_reshaped, Akk_inv_reshaped = pl.pallas_call(
        functools.partial(kda_intra_chunk_kernel, chunk_size=chunk_size, head_dim=K, value_dim=V, scale=scale),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, V), dtype=k.dtype), # u
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # w
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # qg
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # kg
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, chunk_size), dtype=k.dtype), # Aqk
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, chunk_size), dtype=k.dtype), # Akk_inv
        ],
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # q
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # k
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # g
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # beta
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, V)), # v
            pl.BlockSpec(index_map=lambda i, j, l: (i, 0, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # segment_ids
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, V)), # u
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # w
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # qg
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # kg
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # Aqk
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # Akk_inv
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
    )(q_reshaped, k_reshaped, g_reshaped, beta_reshaped, v_reshaped, segment_ids_reshaped)

    return (
        u_reshaped.reshape(B, H, T, V),
        w_reshaped.reshape(B, H, T, K),
        qg_reshaped.reshape(B, H, T, K),
        kg_reshaped.reshape(B, H, T, K),
        Aqk_reshaped,
        Akk_inv_reshaped
    )


def kda_intra_chunk_bwd_kernel(
    # Inputs (Ref)
    q_ref, k_ref, g_ref, beta_ref, segment_ids_ref,
    dAqk_ref, dAkk_ref,
    # Outputs (Ref)
    dq_ref, dk_ref, dg_ref, dbeta_ref,
    # Config
    chunk_size: int,
    head_dim: int,
    scale: float,
):
    dtype = q_ref.dtype
    q = q_ref[0, 0, 0]
    k = k_ref[0, 0, 0]
    g = g_ref[0, 0, 0]
    beta = beta_ref[0, 0, 0]
    segment_ids = segment_ids_ref[0, 0, 0, :, 0]

    dAqk = dAqk_ref[0, 0, 0]
    dAkk = dAkk_ref[0, 0, 0]

    # Recompute states
    g_ref_idx = chunk_size // 2
    g_ref_val = g[g_ref_idx][None, :]
    g_centered = g.astype(jnp.float32) - g_ref_val.astype(jnp.float32)

    q_state = q * jnp.exp2(g_centered).astype(q.dtype)
    k_state_q = k * jnp.exp2(g_centered).astype(k.dtype)
    k_state_k = k * jnp.exp2(-g_centered).astype(k.dtype)

    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    # Both Aqk and Akk use lower-triangular-with-diagonal mask (>=) in backward:
    # Aqk forward uses >= (includes diagonal), Akk forward uses > (strict lower),
    # but dAkk from upstream already has zeros on diagonal, so >= is safe here.
    causal_mask = idx[:, None] >= idx[None, :]
    segment_mask = segment_ids[:, None] == segment_ids[None, :]

    mask = causal_mask & segment_mask

    dAqk_masked = jnp.where(mask, dAqk, 0.0) * scale
    dAkk_masked = jnp.where(mask, dAkk, 0.0)

    Akk_raw = jax.lax.dot_general(
        k_state_q,
        k_state_k,
        (((1,), (1,)), ((), ())),
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    )
    dbeta = jnp.sum(dAkk_masked * Akk_raw, axis=1, keepdims=True).astype(beta.dtype)

    dAkk_raw = dAkk_masked * beta

    dq_state = jax.lax.dot_general(
        dAqk_masked, k_state_k,
        (((1,), (0,)), ((), ())),
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    )

    dk_state_k_1 = jax.lax.dot_general(
        dAqk_masked, q_state,
        (((0,), (0,)), ((), ())),
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    )

    dk_state_q = jax.lax.dot_general(
        dAkk_raw, k_state_k,
        (((1,), (0,)), ((), ())),
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    )
    exp_g = jnp.exp2(g_centered).astype(dtype)
    exp_neg_g = jnp.exp2(-g_centered).astype(dtype)

    dk_state_k_2 = jax.lax.dot_general(
        dAkk_raw, k_state_q,
        (((0,), (0,)), ((), ())),
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    )

    dk_state_k = dk_state_k_1 + dk_state_k_2
    dq = (dq_state * exp_g).astype(dtype)
    dk = (dk_state_q * exp_g + dk_state_k * exp_neg_g).astype(dtype)
    dg_c = (dq_state * q_state + dk_state_q * k_state_q - dk_state_k * k_state_k)

    # Handle g_ref gradient subtraction
    dg_ref_grad = -jnp.sum(dg_c, axis=0, keepdims=False) # (D,)
    dg = dg_c

    idx_range = jnp.arange(chunk_size, dtype=jnp.int32)
    mask_ref_bool = (idx_range == g_ref_idx)
    mask_ref = jnp.reshape(mask_ref_bool.astype(dg.dtype), (chunk_size, 1))
    dg = dg + mask_ref * dg_ref_grad[None, :].astype(dg.dtype)

    dq_ref[0, 0, 0] = dq
    dk_ref[0, 0, 0] = dk
    dg_ref[0, 0, 0] = dg.astype(dtype)
    dbeta_ref[0, 0, 0] = dbeta


@functools.partial(jax.jit, static_argnames=['chunk_size', 'scale'])
def _kda_intra_chunk_bwd_pallas(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    dAqk: jax.Array,
    dAkk: jax.Array,
    scale: float = 1.0,
    chunk_size: int = 128
):
    """Inner Pallas kernel call for intra-chunk backward.

    All tensors use the kernel-native [B, H, T, K] layout.

    Args:
        q:           [B, H, T, K] — query vectors.
        k:           [B, H, T, K] — key vectors.
        g:           [B, H, T, K] — chunk-local cumsum of gates in log2 space.
        beta:        [B, H, T]    — per-token scalar mixing coefficient.
        segment_ids: [B, T]       — integer segment IDs.
        dAqk:        [B, H, NC, C, C] — upstream gradient for Aqk.
        dAkk:        [B, H, NC, C, C] — upstream gradient for Akk.
        scale:       float        — attention scale factor.
        chunk_size:  int          — tile size.

    Returns:
        dq:    [B, H, T, K] — gradient w.r.t. q.
        dk:    [B, H, T, K] — gradient w.r.t. k.
        dg:    [B, H, T, K] — gradient w.r.t. g.
        dbeta: [B, H, T]    — gradient w.r.t. beta.
    """
    B, H, T, K = k.shape
    assert T % chunk_size == 0, "Sequence length must be divisible by chunk_size"
    num_chunks = T // chunk_size

    assert_shape(q, (B, H, T, K), "q")
    assert_shape(k, (B, H, T, K), "k")
    assert_shape(g, (B, H, T, K), "g")
    assert_shape(beta, (B, H, T), "beta")
    assert_shape(dAqk, (B, H, num_chunks, chunk_size, chunk_size), "dAqk")
    assert_shape(dAkk, (B, H, num_chunks, chunk_size, chunk_size), "dAkk")

    if segment_ids is None:
        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

    q_reshaped = q.reshape(B, H, num_chunks, chunk_size, K)
    k_reshaped = k.reshape(B, H, num_chunks, chunk_size, K)
    g_reshaped = g.reshape(B, H, num_chunks, chunk_size, K)
    beta_reshaped = beta.reshape(B, H, num_chunks, chunk_size, 1)
    segment_ids_reshaped = segment_ids.reshape(B, 1, num_chunks, chunk_size, 1)

    grid = (B, H, num_chunks)

    dq_reshaped, dk_reshaped, dg_reshaped, dbeta_reshaped = pl.pallas_call(
        functools.partial(kda_intra_chunk_bwd_kernel, chunk_size=chunk_size, head_dim=K, scale=scale),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # dq
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # dk
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, K), dtype=k.dtype), # dg
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, 1), dtype=k.dtype), # dbeta
        ],
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # q
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # k
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # g
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # beta
            pl.BlockSpec(index_map=lambda i, j, l: (i, 0, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # segment_ids
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # dAqk
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # dAkk
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # dq
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # dk
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, K)), # dg
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # dbeta
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
    )(q_reshaped, k_reshaped, g_reshaped, beta_reshaped, segment_ids_reshaped, dAqk, dAkk)

    return (
        dq_reshaped.reshape(B, H, T, K),
        dk_reshaped.reshape(B, H, T, K),
        dg_reshaped.reshape(B, H, T, K),
        dbeta_reshaped.reshape(B, H, T)
    )


def chunk_kda_bwd_intra(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    dAqk: jax.Array,
    dAkk: jax.Array,
    dq: jax.Array | None = None,
    dk: jax.Array | None = None,
    db: jax.Array | None = None,
    dg: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Stage 4: Intra-chunk backward through attention matrices.

    Propagates upstream gradients dAqk and dAkk back to q, k, g, beta,
    and accumulates into incoming dq, dk, db, dg from Stage 3.

    Interface aligned with CPU reference ``chunk_kda_bwd_intra``.
    External layout: [B, T, H, K]. Internal Pallas kernel uses [B, H, T, K].

    Args:
        q:     [B, T, H, K] — query vectors.
        k:     [B, T, H, K] — key vectors.
        g:     [B, T, H, K] — chunk-local cumsum of gates in log2 space.
        beta:  [B, T, H]    — per-token scalar mixing coefficient.
        dAqk:  [B, T, H, BT] — gradient of Aqk attention matrix.
        dAkk:  [B, T, H, BT] — gradient of Akk matrix.
        dq:    [B, T, H, K] — incoming dq from Stage 3 (or None).
        dk:    [B, T, H, K] — incoming dk from Stage 3 (or None).
        db:    [B, T, H]    — incoming db from Stage 3 (or None).
        dg:    [B, T, H, K] — incoming dg from Stage 3 (or None).
        cu_seqlens:    not implemented.
        chunk_indices: not implemented.
        chunk_size:    int — chunk size (default 64).
        safe_gate:     not implemented.

    Returns:
        dq:  [B, T, H, K] — accumulated query gradient.
        dk:  [B, T, H, K] — accumulated key gradient.
        db:  [B, T, H]    — accumulated beta gradient.
        dg:  [B, T, H, K] — accumulated gate gradient.
    """
    del cu_seqlens, chunk_indices, safe_gate

    B, T, H, K = q.shape
    BT = chunk_size
    NT = T // BT

    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(g, (B, T, H, K), "g")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape(dAqk, (B, T, H, BT), "dAqk")
    assert_shape(dAkk, (B, T, H, BT), "dAkk")
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

    # Default incoming gradients to zeros if not provided
    if dq is None:
        dq = jnp.zeros_like(q)
    if dk is None:
        dk = jnp.zeros_like(k)
    if db is None:
        db = jnp.zeros((B, T, H), dtype=q.dtype)
    if dg is None:
        dg = jnp.zeros_like(q)

    # --- Layout: [B, T, H, K] -> [B, H, T, K] for Pallas kernel ---
    q_bhtk = q.transpose(0, 2, 1, 3)
    k_bhtk = k.transpose(0, 2, 1, 3)
    g_bhtk = g.transpose(0, 2, 1, 3)
    beta_bht = beta.transpose(0, 2, 1)

    # dAqk/dAkk: [B, T, H, BT] -> [B, H, NC, BT, BT]
    dAqk_pallas = dAqk.reshape(B, NT, BT, H, BT).transpose(0, 3, 1, 2, 4)
    dAkk_pallas = dAkk.reshape(B, NT, BT, H, BT).transpose(0, 3, 1, 2, 4)

    segment_ids = jnp.zeros((B, T), dtype=jnp.int32)

    # scale=1.0 because dAqk from Stage 1 already incorporates scale
    dq_intra, dk_intra, dg_intra, dbeta_intra = _kda_intra_chunk_bwd_pallas(
        q=q_bhtk, k=k_bhtk, g=g_bhtk, beta=beta_bht,
        segment_ids=segment_ids,
        dAqk=dAqk_pallas, dAkk=dAkk_pallas,
        scale=1.0,
        chunk_size=chunk_size,
    )

    # --- Layout: [B, H, T, K] -> [B, T, H, K], accumulate ---
    dq = dq + dq_intra.transpose(0, 2, 1, 3)
    dk = dk + dk_intra.transpose(0, 2, 1, 3)
    dg = dg + dg_intra.transpose(0, 2, 1, 3)
    db = db + dbeta_intra.transpose(0, 2, 1)

    return dq, dk, db, dg
