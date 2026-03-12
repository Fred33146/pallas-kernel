import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools


def build_chunk_map(cu_seqlens, T_sum, BT):
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    return seq_idx


def _chunk_fwd_h_kernel(
    k_ref,  # [1, T_sum, BK]
    v_ref,  # [1, T_sum, BV]
    h0_ref,  # [N, 1, BK, BV]
    gk_ref,  # [1, T_sum, BK]
    cu_seqlens_ref,  # [num_seq+1]
    chunk_to_seq,  # [T_sum/BT]
    h_ref,  # [NS, 1, BK, BV]
    ht_ref,  # [N, 1, BK , BV]
    *,
    BT,
    BS,
):
    T_sum, BK = k_ref.shape[1], k_ref.shape[2]
    BV = v_ref.shape[2]
    N = h0_ref.shape[0]
    NT = pl.cdiv(T_sum, BT)
    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)
    b_h = jnp.zeros((BK, BV), dtype=jnp.float32)
    seq_idx = jnp.array(0, dtype=jnp.int32)

    def body(i_t, carry):
        b_h, seq_idx = carry
        t0 = i_t * BT

        seq_idx = chunk_to_seq[i_t]

        bos = cu_seqlens_ref[seq_idx]

        # reset h state
        def reset_state(_):
            if h0_ref is not None:
                return h0_ref[seq_idx, 0]
            else:
                return b_h_start

        b_h = lax.cond(
            t0 == bos,
            reset_state,
            lambda _: b_h,
            operand=None,
        )
        # store intermediate state
        i_s = i_t // NTS

        def store_fn(_):
            h_ref[i_s, 0] = b_h
            return None

        lax.cond((i_t % NTS) == 0, store_fn, lambda _: None, operand=None)

        k = k_ref[(0, pl.dslice(t0, BT), slice(None))]  # [BT,BK]
        v = v_ref[(0, pl.dslice(t0, BT), slice(None))]  # [BT,BV]
        if gk_ref is not None:
            gk = gk_ref[(0, pl.dslice(t0, BT), slice(None))]  # [BT,BK]
            g_last = gk[-1, :]
            decay = jnp.exp(g_last)
            b_h = b_h * decay[:, None]  # [BK, BV] * [BK,1]
            k = (k * jnp.exp(g_last[None, :] - gk)).astype(k.dtype)

        # state update
        b_h = b_h + jax.lax.dot(k.T, v)

        eos = cu_seqlens_ref[seq_idx + 1]

        is_last_chunk = t0 + BT >= eos

        def write_final(_):
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = b_h
            return None

        lax.cond(is_last_chunk, write_final, lambda _: None, operand=None)
        return (b_h, seq_idx)

    b_h, seq_idx = lax.fori_loop(0, NT, body, (b_h, seq_idx))


def check_chunk_fwd(x):
    assert x is None, "x should be None."


# note: The precision difference between this kernel on the TPU and FLA on the GPU is 5e-2.
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
        "interpret",
    ],
)
def chunk_fwd_h_kernel(
    k: jax.Array,  # [B,T,H,K]
    v: jax.Array,  # [B,T,H,V]
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    interpret: bool = False,
):
    check_chunk_fwd(g)
    check_chunk_fwd(gv)
    check_chunk_fwd(g_gamma)
    # todo: tune bk and bv for bast performance
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, (
        f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        cu_seqlens = jnp.arange(B * T + 1, step=T)
    T_sum = B * T
    chunk_to_seq = build_chunk_map(cu_seqlens=cu_seqlens, T_sum=T_sum, BT=BT)

    N, NS = (
        len(cu_seqlens) - 1,
        T_sum // BS,
    )  # split_offsets[-1] # NS number of chunk_size

    k = jnp.reshape(k, (T_sum, H, K))
    v = jnp.reshape(v, (T_sum, H, V))

    k = jnp.transpose(k, (1, 0, 2))  # (H,B*T,K)
    v = jnp.transpose(v, (1, 0, 2))  # (H,B*T,V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K))
        gk = jnp.transpose(gk, (1, 0, 2))  # (H,B*T,K)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV))

    def k_index_map(head_index, k_index, _):
        return head_index, 0, k_index

    def gk_index_map(head_index, k_index, _):
        return head_index, 0, k_index

    def v_index_map(head_index, _, v_index):
        return head_index, 0, v_index

    def h0_index_map(head_index, k_index, v_index):
        return 0, head_index, k_index, v_index

    def h_index_map(head_index, k_index, v_index):
        return 0, head_index, k_index, v_index

    def ht_index_map(head_index, k_index, v_index):
        return 0, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((NS, 1, BK, BV), ht_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=k.dtype))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), h_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, T_sum, BK), k_index_map),
        pl.BlockSpec((1, T_sum, BV), v_index_map),
    ]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, T_sum, BK), gk_index_map))
    else:
        in_specs.append(None)

    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    kernel = functools.partial(
        _chunk_fwd_h_kernel,
        BT=BT,
        BS=BS,
    )
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "arbitrary",
                "arbitrary",
            ),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
    )(k, v, h0, gk, cu_seqlens, chunk_to_seq)
    if output_final_state:
        return h, ht
    return h, None


def chunk_fwd_h_ref(
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    h0: jax.Array | None = None,
    output_final_state: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Inter-chunk hidden state propagation.

    Computes the hidden state at the start of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        gk: [B, T, H, K] — chunk-local cumsum of gates
        h0: [N, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        cu_seqlens_cpu: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_fwd_h"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )
    # seqlens = jnp.diff(cu_seqlens_cpu) if cu_seqlens_cpu is not None else None

    k = k.reshape(-1, H, K)
    v = v.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None
    h0 = h0.reshape(-1, H, K, V) if h0 is not None else None

    ht = jnp.zeros([N, H, K, V], dtype=jnp.float32)
    h_all = jnp.zeros([B, NT, H, K, V], dtype=k.dtype)
    for i_n in range(N):
        if cu_seqlens_cpu is None:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        h = jnp.zeros((H, K, V), dtype=jnp.float32)
        if h0 is not None:
            h = h + h0[i_n].astype(jnp.float32)

        NT = (eos - bos) // C
        for i_t in range(NT):
            if cu_seqlens_cpu is None:
                h_all = h_all.at[i_n, i_t].set(h.astype(h_all.dtype))
            else:
                h_all = h_all.at[0, bos // C + i_t].set(h.astype(h_all.dtype))
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]
            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= jnp.exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * jnp.exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            h = h + jnp.einsum("chk,chv->hkv", b_k, b_v)
        if output_final_state:
            ht = ht.at[i_n].set(h.astype(ht.dtype))

    return h_all, ht
