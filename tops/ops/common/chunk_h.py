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
    g_gamma,  # [H]
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
    NT = pl.cdiv(T_sum, BT)
    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)
    b_h = jnp.zeros((BK, BV), dtype=jnp.float32)
    seq_idx = jnp.array(0, dtype=jnp.int32)
    i_in_seq = jnp.array(0, dtype=jnp.int32)

    if g_gamma is not None:
        head_index = pl.program_id(0)
        b_g = g_gamma[head_index] * (jnp.arange(0, BT) + 1)

    def body(i_t, carry):
        b_h, seq_idx, i_in_seq = carry
        t0 = i_t * BT

        seq_idx = chunk_to_seq[i_t]

        bos = cu_seqlens_ref[seq_idx]
        eos = cu_seqlens_ref[seq_idx + 1]

        # reset h state
        def reset_state(_):
            nonlocal i_in_seq
            i_in_seq = 0
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

        if g_gamma is not None:
            b_g_last = g_gamma[head_index] * jnp.minimum(BT, eos - bos - i_in_seq * BT)
            b_h *= jnp.exp(b_g_last)
            v = (v * jnp.exp(b_g_last - b_g)[:, None]).astype(v.dtype)

        if gk_ref is not None:
            gk = gk_ref[(0, pl.dslice(t0, BT), slice(None))]  # [BT,BK]
            g_last = gk[BT - 1, :]
            decay = jnp.exp(g_last)
            b_h = b_h * decay[:, None]  # [BK, BV] * [BK,1]
            k = (k * jnp.exp(g_last[None, :] - gk)).astype(k.dtype)

        # state update
        b_h = b_h + jax.lax.dot(k.T, v, precision=lax.Precision.HIGHEST)

        is_last_chunk = t0 + BT >= eos

        def write_final(_):
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = b_h
            return None

        lax.cond(is_last_chunk, write_final, lambda _: None, operand=None)
        i_in_seq += 1
        return (b_h, seq_idx, i_in_seq)

    b_h, seq_idx, i_in_seq = lax.fori_loop(0, NT, body, (b_h, seq_idx, i_in_seq))


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

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
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
    )(k, v, h0, gk, g_gamma, cu_seqlens, chunk_to_seq)
    if output_final_state:
        return h, ht
    return h, None


def chunk_fwd_h_ref(
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array | None = None,
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

        if g_gamma is not None:
            b_g = jax.lax.dot(
                (jnp.arange(0, C) + 1)[:, None], g_gamma[None, :]
            )  # [C, H]

        NT = (eos - bos) // C
        for i_t in range(NT):
            if cu_seqlens_cpu is None:
                h_all = h_all.at[i_n, i_t].set(h.astype(h_all.dtype))
            else:
                h_all = h_all.at[0, bos // C + i_t].set(h.astype(h_all.dtype))
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if g_gamma is not None:
                b_g_last = g_gamma * jnp.minimum(C, (eos - bos) - i_t * C)  # [H]
                h *= jnp.exp(b_g_last[:, None, None])  # (H, K, V)
                b_v = (b_v * jnp.exp(b_g_last[None, :] - b_g)[:, :, None]).astype(
                    b_v.dtype
                )

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= jnp.exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * jnp.exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            h = h + lax.dot_general(
                b_k,
                b_v,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        if output_final_state:
            ht = ht.at[i_n].set(h.astype(ht.dtype))

    return h_all, ht


def chunk_bwd_dh_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float = 1.0,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Backward hidden state gradient propagation.

    Propagates gradients backward through chunks to compute dh at each
    chunk boundary and dh0.

    Args:
        q:   [B, T, H, K] — queries
        k:   [B, T, H, K] — keys
        v:   [B, T, H, V] — values
        gk:  [B, T, H, K] — chunk-local cumsum of gates
        do:  [B, T, H, V] — output gradient
        h0:  [N, H, K, V] — initial hidden state (optional)
        dht: [N, H, K, V] — terminal state gradient (optional)
        scale: scaling factor
        cu_seqlens_cpu: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V] — gradient at start of each chunk
        dh0: [N, H, K, V] or None — initial state gradient
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_bwd_dh"
    is_varlen = cu_seqlens_cpu is not None

    q = q.reshape(-1, H, K)
    do = do.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None

    dh_all = jnp.zeros([B, NT, H, K, V], dtype=jnp.float32)
    dh0_all = (
        jnp.zeros([N, H, K, V], dtype=jnp.float32)
        if (h0 is not None or dht is not None)
        else None
    )

    for i_n in range(N):
        if not is_varlen:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        NT_seq = (eos - bos) // C
        dh = jnp.zeros((H, K, V), dtype=jnp.float32)
        if dht is not None:
            dh = dh + dht[i_n].astype(jnp.float32)

        for i_t in range(NT_seq - 1, -1, -1):
            bi = 0 if is_varlen else i_n
            ti = bos // C + i_t if is_varlen else i_t
            dh_all = dh_all.at[bi, ti].set(dh)

            b_q = q[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_do = do[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                b_q_hat = b_q * jnp.exp(b_gk) * scale  # [C, H, K]
                dh = dh * jnp.exp(b_gk_last[:, :, None])
            else:
                b_q_hat = b_q * scale

            # contract over C (dim 0) and H (dim 1): [C,H,K]^T @ [C,H,V] -> [H,K,V]
            dh = dh + lax.dot_general(
                b_q_hat,
                b_do,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )

        if dh0_all is not None:
            dh0_all = dh0_all.at[i_n].set(dh)

    return dh_all, dh0_all


def _chunk_bwd_dh_kernel(
    q_ref,          # [1, T_sum, BK]
    do_ref,         # [1, T_sum, BV]
    dht_ref,        # [N, 1, BK, BV]
    gk_ref,         # [1, T_sum, BK]
    cu_seqlens_ref, # [num_seq + 1]
    chunk_to_seq,   # [T_sum // BT]
    dh_ref,         # [NS, 1, BK, BV]
    dh0_ref,        # [N, 1, BK, BV]
    *,
    BT: int,
    BS: int,
    scale: float,
):
    T_sum, BK = q_ref.shape[1], q_ref.shape[2]
    BV = do_ref.shape[2]
    NT = pl.cdiv(T_sum, BT)
    NTS = BS // BT
    
    b_dh_start = jnp.zeros((BK, BV), dtype=jnp.float32)
    b_dh = jnp.zeros((BK, BV), dtype=jnp.float32)
    seq_idx = jnp.array(0, dtype=jnp.int32)

    def body(step, carry):
        b_dh, seq_idx = carry
        # 核心：反向遍历时间维度
        i_t = NT - 1 - step
        t0 = i_t * BT
        
        seq_idx = chunk_to_seq[i_t]
        eos = cu_seqlens_ref[seq_idx + 1]
        
        # 1. 序列末尾处理：如果当前块碰到了 sequence 的末尾，重置 dh 为 dht 或者全0
        is_last_chunk = (t0 + BT >= eos)
        def reset_state(_):
            if dht_ref is not None:
                return dht_ref[seq_idx, 0]
            else:
                return b_dh_start
        b_dh = lax.cond(is_last_chunk, reset_state, lambda _: b_dh, operand=None)
        
        # 2. 存储传入当前 chunk 的梯度 dh (对应于 dh_all.at[..., i_t].set(dh))
        i_s = i_t // NTS
        def store_fn(_):
            dh_ref[i_s, 0] = b_dh
        lax.cond((i_t % NTS) == 0, store_fn, lambda _: None, operand=None)
        
        # 加载 HBM 数据到 SRAM
        b_q  = q_ref[0, pl.dslice(t0, BT), slice(None)]    # [BT, BK]
        b_do = do_ref[0, pl.dslice(t0, BT), slice(None)]   # [BT, BV]
        
        if gk_ref is not None:
            b_gk = gk_ref[0, pl.dslice(t0, BT), slice(None)] # [BT, BK]
            g_last = b_gk[BT - 1, :]
            
            # dh = dh * exp(g_last) (时序衰减的反向)
            b_dh = b_dh * jnp.exp(g_last)[:, None]  # [BK, BV] * [BK, 1]
            
            # 计算 q_hat = q * exp(gk) * scale
            b_q_hat = (b_q * jnp.exp(b_gk) * scale).astype(b_q.dtype)
        else:
            b_q_hat = (b_q * scale).astype(b_q.dtype)
            
        # 3. 计算并累积本 chunk 的隐状态梯度贡献
        # b_q_hat.T @ b_do -> [BK, BT] @ [BT, BV] -> [BK, BV]
        b_dh = b_dh + jax.lax.dot(b_q_hat.T, b_do,precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,)
        
        # 4. 序列起始处理：如果到了 sequence 的头部，写入 dh0
        bos = cu_seqlens_ref[seq_idx]
        is_first_chunk = (t0 == bos)
        def write_dh0(_):
            if dh0_ref is not None:
                dh0_ref[seq_idx, 0] = b_dh
        lax.cond(is_first_chunk, write_dh0, lambda _: None, operand=None)
        
        return (b_dh, seq_idx)

    lax.fori_loop(0, NT, body, (b_dh, seq_idx))

@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "chunk_size",
        "split_size",
        "states_in_fp32",
        "interpret",
    ],
)
def chunk_bwd_dh_kernel(
    q: jax.Array,                # [B, T, H, K]
    k: jax.Array,                # [B, T, H, K] (unused but kept for API compatibility)
    v: jax.Array,                # [B, T, H, V] (unused but kept for API compatibility)
    gk: jax.Array | None = None, # [B, T, H, K]
    do: jax.Array = None,        # [B, T, H, V]
    dht: jax.Array | None = None,# [N, H, K, V]
    scale: float = 1.0,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    interpret: bool = False,
):
    BK, BV = 128, 128
    B, T, H, K = q.shape
    V = do.shape[-1]
    
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    
    T_sum = B * T
    if cu_seqlens is None:
        cu_seqlens = jnp.arange(T_sum + 1, step=T)
    
    chunk_to_seq = build_chunk_map(cu_seqlens=cu_seqlens, T_sum=T_sum, BT=BT)
    N, NS = len(cu_seqlens) - 1, T_sum // BS

    # Reshape and transpose input tensors to [H, T_sum, D]
    q = jnp.reshape(q, (T_sum, H, K)).transpose(1, 0, 2)
    do = jnp.reshape(do, (T_sum, H, V)).transpose(1, 0, 2)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K)).transpose(1, 0, 2)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV))

    # Define BlockSpecs
    def idx_map_K(h, k, v): return h, 0, k
    def idx_map_V(h, k, v): return h, 0, v
    def idx_map_state(h, k, v): return 0, h, k, v

    dtype_out = q.dtype if not states_in_fp32 else jnp.float32
    out_shape = [
        jax.ShapeDtypeStruct(shape=(NS, H, K, V), dtype=dtype_out),
        jax.ShapeDtypeStruct(shape=(N,  H, K, V), dtype=dtype_out)
    ]
    out_specs = [
        pl.BlockSpec((NS, 1, BK, BV), idx_map_state),
        pl.BlockSpec((N,  1, BK, BV), idx_map_state)
    ]

    in_specs = [
        pl.BlockSpec((1, T_sum, BK), idx_map_K),
        pl.BlockSpec((1, T_sum, BV), idx_map_V),
    ]
    
    if dht is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), idx_map_state))
    else:
        in_specs.append(None)
        
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, T_sum, BK), idx_map_K))
    else:
        in_specs.append(None)

    # 引入 SRAM/VMEM 占位 (用于 cu_seqlens_ref 和 chunk_to_seq)
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))

    kernel = functools.partial(_chunk_bwd_dh_kernel, BT=BT, BS=BS, scale=scale)

    dh_all, dh0 = pl.pallas_call(
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
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=128 * 1024 * 1024, # 128 MB limit for VMEM
        ),
    )(q, do, dht, gk, cu_seqlens, chunk_to_seq)
    
    return dh_all, dh0
