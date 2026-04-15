# tops/ops/common/chunk_delta_h.py
"""Delta-rule inter-chunk hidden state propagation (Pallas TPU kernel + JAX reference).

Shared by KDA and Gated Delta Rule. Implements the delta-rule recurrence
where the value is corrected by subtracting the state's prediction before
accumulating into the hidden state:

    v_new_t = v_t - w_t @ h_{t-1}    (delta correction)
    h_t = h_{t-1} * decay_t + k_t^T @ v_new_t  (state update)

This differs from standard GLA (common/chunk_h.py) which uses:
    h_t = h_{t-1} * decay_t + k_t^T @ v_t       (no delta correction)

Gate types supported:
  - g:  [B, T, H] scalar per-head gate (applied as exp2(g) to state, and
        exp2(g_last - g) to v_new for accumulation)
  - gk: [B, T, H, K] per-element gate (applied as exp2(gk) to state only;
        k is assumed to be already gated, so no additional k gating is done)

Both gates operate in log2 space: decay = exp2(g) or exp2(gk).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tops.ops.utils import get_interpret
from tops.utils import (
    align_up,
    assert_shape,
    assert_shape_or_none,
    cdiv,
    export_public,
    pad_to_multiple,
)


def chunk_gated_delta_rule_fwd_h_ref(
    k: jax.Array,
    w: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Pure JAX reference for delta-rule inter-chunk state propagation.

    For each chunk c (with BT positions), stores the state BEFORE processing
    chunk c, then updates the state using all BT positions in chunk c.

    The algorithm for each chunk:
      1. Store h[c] = S (state before chunk)
      2. v_new = v - w @ S (delta correction using pre-decay state)
      3. Decay S:
         - If g (scalar): S *= exp2(g_last)
         - If gk (per-element): S *= exp2(gk_last[:, None])
      4. Gate v_new for accumulation (scalar g only):
         - If g: v_new *= exp2(g_last - g)[:, None]
         - If gk: no v_new gating (k is already position-gated)
      5. State update: S += k^T @ v_new

    IMPORTANT: k is expected to be already gated (e.g., kg from intra-chunk
    step). The gk gate is used ONLY for state decay, NOT for additionally
    gating k. This matches the Pallas kernel behavior.

    Args:
        k: [B, T, H, K] — Keys (gated, ready for outer product).
        w: [B, T, H, K] — Correction weights (w = beta * k * exp(g)).
        v: [B, T, H, V] — Values (after intra-chunk delta solve: u).
        g: [B, T, H] — Scalar per-head gate in log2 space. Optional.
        gk: [B, T, H, K] — Per-element gate in log2 space. Optional.
        initial_state: [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return the final hidden state.
        chunk_size: Chunk size. T must be divisible by chunk_size.

    Returns:
        h: [B, NT, H, K, V] — Hidden states (state before each chunk).
        v_new: [B, T, H, V] — Delta-corrected values (v - w @ h).
        final_state: [B, H, K, V] if output_final_state, else None.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT

    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (B, H, K, V), "initial_state")
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

    # Reshape to chunks: [B, T, H, D] → [B, H, NT, BT, D]
    k_c = jnp.transpose(k, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)
    w_c = jnp.transpose(w, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)
    v_c = jnp.transpose(v, (0, 2, 1, 3)).reshape(B, H, NT, BT, V).astype(jnp.float32)

    if g is not None:
        g_c = jnp.transpose(g, (0, 2, 1)).reshape(B, H, NT, BT).astype(jnp.float32)
    if gk is not None:
        gk_c = jnp.transpose(gk, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)

    # State: [B, H, K, V]
    S = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if initial_state is not None:
        S = S + initial_state.astype(jnp.float32)

    h_all = jnp.zeros((B, H, NT, K, V), dtype=jnp.float32)
    v_new_all = jnp.zeros((B, H, NT, BT, V), dtype=jnp.float32)

    for i in range(NT):
        # 1. Store state BEFORE processing this chunk
        h_all = h_all.at[:, :, i].set(S)

        # 2. Delta correction: v_new = v - w @ S
        w_i = w_c[:, :, i]  # [B, H, BT, K]
        v_i = v_c[:, :, i]  # [B, H, BT, V]
        v_new_i = v_i - jnp.einsum('bhck,bhkv->bhcv', w_i, S)
        v_new_all = v_new_all.at[:, :, i].set(v_new_i)

        # 3. Decay state by gate
        if g is not None:
            g_last = g_c[:, :, i, -1]  # [B, H] — scalar gate at last position
            S = S * jnp.exp2(g_last)[:, :, None, None]

        if gk is not None:
            gk_last = gk_c[:, :, i, -1]  # [B, H, K] — per-element gate at last position
            S = S * jnp.exp2(gk_last)[:, :, :, None]

        # 4. Gate v_new for accumulation (scalar g only)
        # When g (scalar per-head): distribute gate into v_new for k^T @ v_new
        # When gk (per-element): k is already gated, no v_new gating needed
        if g is not None:
            g_chunk = g_c[:, :, i]           # [B, H, BT]
            g_last_val = g_c[:, :, i, -1:]   # [B, H, 1]
            v_new_i = v_new_i * jnp.exp2(g_last_val - g_chunk)[:, :, :, None]

        # 5. State update: S += k^T @ v_new
        # k is already gated (kg from intra-chunk), used directly
        k_i = k_c[:, :, i]  # [B, H, BT, K]
        S = S + jnp.einsum('bhck,bhcv->bhkv', k_i, v_new_i)

    final_state = S if output_final_state else None

    # Reshape: [B, H, NT, K, V] → [B, NT, H, K, V]
    h_all = jnp.transpose(h_all, (0, 2, 1, 3, 4))
    # Reshape v_new: [B, H, NT, BT, V] → [B, T, H, V]
    v_new_all = v_new_all.reshape(B, H, T, V)
    v_new_all = jnp.transpose(v_new_all, (0, 2, 1, 3))

    return h_all, v_new_all, final_state


# ---------------------------------------------------------------------------
# Pallas TPU kernels for delta-rule inter-chunk state propagation
# ---------------------------------------------------------------------------


def _prepare_chunk_offsets(seqlens: jax.Array, chunk_size: int) -> jax.Array:
    """Compute cumulative chunk-count offsets for variable-length sequences.

    Given cumulative sequence lengths *seqlens* ``[0, s1, s2, ...]`` and a
    fixed *chunk_size*, returns a prefix-sum array ``[0, NT_0, NT_0+NT_1, ...]``
    where ``NT_i = ceil(len_i / chunk_size)``.  Used by the varlen kernel to
    map ``(sequence_id, chunk_id)`` to a flat chunk index in the output tensor.
    """
    return jnp.pad(
        cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32),
        (1, 0),
        constant_values=0,
    ).cumsum(-1)


# ── Varlen Pallas kernel ────────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_varlen_kernel(
    k_ref,      # [B, T, 1, K_PADSIZE // 128, 128]
    v_ref,      # [B, T, 1, V_PADSIZE//BV, BV * 2]
    w_ref,      # [B, T, 1, K_PADSIZE // 128, 128]
    g_ref,      # [B, T, H, 128]
    gk_ref,     # [B, T, H, K]
    h0_ref,     # [N, H, V, K]
    seqlens_ref,# [N + 1]
    chunk_offsets_ref, # [N + 1]

    # output
    h_ref,      # [B, NT, H, V, K]
    v_new_ref,  # [H, B, T, V_PADSIZE//BV, BV]
    ht_ref,     # [N, H, V, K]

    B,
    T,
    NT,
    H,
    K,
    V,
    BT,
    BV,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
    IS_VARLEN = True,
):
  assert IS_VARLEN == True
  # NOTE: Do NOT reshape refs here — on TPU, ref.reshape() changes the
  # logical view but the physical tile layout stays from the original
  # block spec, causing incorrect data reads.  Access refs directly
  # using indices that match the block spec shape.

  idx_v, idx_nh = pl.program_id(0), pl.program_id(1)
  idx_n, idx_h = idx_nh // H, idx_nh % H

  if IS_VARLEN:
    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    real_T = eos - bos
    real_NT = (real_T + BT - 1) // BT
    boh = chunk_offsets_ref[idx_n]
  else:
    bos = idx_n * T
    eos = bos + T
    real_NT = (T + BT - 1) // BT
    boh = idx_n * real_NT

  b_h1 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h2 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h3 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h4 = jnp.zeros([64, BV], dtype=jnp.float32)

  if USE_INITIAL_STATE:
    b_h1 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 0:64].astype(jnp.float32).transpose(1, 0)
    if K > 64:
      b_h2 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 64:128].astype(jnp.float32).transpose(1, 0)
    if K > 128:
      b_h3 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 128:192].astype(jnp.float32).transpose(1, 0)
    if K > 192:
      b_h4 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 192:256].astype(jnp.float32).transpose(1, 0)

  def loop_real_NT(idx_t, carry):
    b_h1, b_h2, b_h3, b_h4 = carry
    len_k1 = min(K, 64)
    h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 0:len_k1] = b_h1.astype(h_ref.dtype).transpose(1, 0)[:, :len_k1]
    if K > 64:
      len_k2 = min(K, 128) - 64
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 64:64+len_k2] = b_h2.astype(h_ref.dtype).transpose(1, 0)[:, :len_k2]
    if K > 128:
      len_k3 = min(K, 192) - 128
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 128:128+len_k3] = b_h3.astype(h_ref.dtype).transpose(1, 0)[:, :len_k3]
    if K > 192:
      len_k4 = min(K, 256) - 192
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 192:192+len_k4] = b_h4.astype(h_ref.dtype).transpose(1, 0)[:, :len_k4]

    valid_len = real_T - idx_t * BT
    mask = jnp.arange(BT)[:, None] < valid_len

    b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 0, :][:, 0:64]
    b_w = jnp.where(mask, b_w, 0)

    b_v = jnp.dot(b_w.astype(jnp.float32), b_h1, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 0, :][:, 64:128]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h2, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 128:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 1, :][:, 0:64]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h3, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 1, :][:, 64:128]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h4, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    b_v_raw = v_ref[0, pl.ds(bos + idx_t * BT, BT), 0, idx_v, 0:BV].astype(b_v.dtype)
    b_v_raw = jnp.where(mask, b_v_raw, 0)
    b_v = b_v_raw - b_v

    if SAVE_NEW_VALUE:
      v_new_slice = v_new_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), idx_v, 0:BV]
      v_new_val = b_v.astype(v_new_ref.dtype)
      v_new_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), idx_v, 0:BV] = jnp.where(mask, v_new_val, v_new_slice)

    last_idx = jnp.minimum((idx_t + 1) * BT, real_T) - 1

    if USE_G:
      m_t = (idx_t * BT + jnp.arange(0, BT)) < real_T
      b_g_last = g_ref[0, bos + last_idx, idx_h, 0]
      b_g = g_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, :]
      b_g = b_g[:BT, :1].reshape(BT)
      if USE_EXP2:
        b_v = b_v * jnp.where(m_t, jnp.exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = jnp.exp2(b_g_last)
      else:
        b_v = b_v * jnp.where(m_t, jnp.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = jnp.exp(b_g_last)
      b_h1 *= b_g_last
      if K > 64:
        b_h2 *= b_g_last
      if K > 128:
        b_h3 *= b_g_last
      if K > 192:
        b_h4 *= b_g_last

    if USE_GK:
      o_k1 = jnp.arange(0, 64)
      b_gk_last1 = jnp.where(o_k1 < K,
                      gk_ref[0, bos + last_idx, idx_h, 0, :],
                      0
                    ).astype(jnp.float32)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]

      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, bos + last_idx, idx_h, 1, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]

      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, bos + last_idx, idx_h, 2, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]

      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, bos + last_idx, idx_h, 3, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 0, :][:, 0:64]
    b_k = jnp.where(mask, b_k, 0)
    b_h1 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 0, :][:, 64:128]
      b_k = jnp.where(mask, b_k, 0)
      b_h2 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    if K > 128:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 1, :][:, 0:64]
      b_k = jnp.where(mask, b_k, 0)
      b_h3 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, 1, :][:, 64:128]
      b_k = jnp.where(mask, b_k, 0)
      b_h4 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    return b_h1, b_h2, b_h3, b_h4

  carry = (b_h1, b_h2, b_h3, b_h4)
  carry = jax.lax.fori_loop(0, real_NT, loop_real_NT, carry)
  b_h1, b_h2, b_h3, b_h4 = carry

  if STORE_FINAL_STATE:
    len_k1 = min(K, 64)
    ht_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 0:len_k1] = b_h1.astype(ht_ref.dtype)[:len_k1, :].T
    if K > 64:
      len_k2 = min(K, 128) - 64
      ht_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 64:64+len_k2] = b_h2.astype(ht_ref.dtype)[:len_k2, :].T
    if K > 128:
      len_k3 = min(K, 192) - 128
      ht_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 128:128+len_k3] = b_h3.astype(ht_ref.dtype)[:len_k3, :].T
    if K > 192:
      len_k4 = min(K, 256) - 192
      ht_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 192:192+len_k4] = b_h4.astype(ht_ref.dtype)[:len_k4, :].T


def _chunk_gated_delta_rule_fwd_varlen(
    k: jax.Array,
    w: jax.Array,
    v: jax.Array,
    seqlens: jax.Array,
    chunk_indices: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    BV: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = False,
):
  """Varlen launcher for delta-rule inter-chunk state forward pass.

  Pads and reshapes inputs for the TPU kernel, dispatches
  ``_chunk_gated_delta_rule_fwd_varlen_kernel``, then un-pads outputs.

  Args:
      k: [B, T, H, K] -- Keys.
      w: [B, T, H, K] -- Correction weights.
      v: [B, T, H, V] -- Delta-corrected values from intra-chunk.
      seqlens: [N+1] -- Cumulative sequence lengths.
      chunk_indices: [NT, 2] -- Precomputed chunk indices.
      g: [B, T, H] -- Scalar per-head gate (optional).
      gk: [B, T, H, K] -- Per-element gate (optional).
      initial_state: [N, H, K, V] -- Initial hidden state (optional).
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size.
      BV: V-dimension block size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation.

  Returns:
      h: [B, NT, H, K, V] -- Hidden states before each chunk.
      v_new: [B, T, H, V] or None -- Delta-corrected values.
      final_state: [N, H, K, V] or None.
  """
  B, T, H, K, V = *k.shape, v.shape[-1]
  BT = chunk_size
  K_BPE = k.dtype.itemsize
  W_BPE = w.dtype.itemsize
  V_BPE = v.dtype.itemsize
  K_PADSIZE = int(align_up(K, 512 // K_BPE))
  V_PADSIZE = int(align_up(V, 512 // V_BPE))
  # Pad V to a multiple of BV so kernel pl.ds(idx_v*BV, BV) stays in bounds.
  V_ALIGNED = int(align_up(V, BV))

  assert ((seqlens is None) or (seqlens is not None and chunk_indices is not None))
  if seqlens is None:
    N, NT, chunk_offsets = B, math.ceil(T / BT), None
  else:
    N, NT, chunk_offsets = len(seqlens) - 1, len(chunk_indices), _prepare_chunk_offsets(seqlens, BT)
  assert ((initial_state is None) or (initial_state.shape == (N, H, K, V)))

  if initial_state is not None:
    initial_state = initial_state.transpose(0, 1, 3, 2)
    # [N, H, K, V] -> [N, H, V, K]
    if V_ALIGNED > V:
      initial_state = jnp.pad(initial_state, ((0, 0), (0, 0), (0, V_ALIGNED - V), (0, 0)))
    initial_state = pad_to_multiple(initial_state, 512 // K_BPE, -1, 0)

  # [B, T, H, K] -> [B, T, H, K_PADSIZE]
  # -> [B, T, H, K_PADSIZE // 128, 128]
  k_paded = k
  k_paded = pad_to_multiple(k_paded, 512 // K_BPE, -1, 0)
  k_paded = k_paded.reshape(B, T, H, -1, 128)

  # [B, T, H, K] -> [B, T, H, K_PADSIZE]
  # -> [B, T, H, K_PADSIZE // 128, 128]
  w_paded = w
  w_paded = pad_to_multiple(w_paded, 512 // W_BPE, -1, 0)
  w_paded = w_paded.reshape(B, T, H, -1, 128)

  # [B, T, H, V] -> [B, T, H, V_PADSIZE]
  # -> [B, T, H, V_PADSIZE//BV, BV]
  # -> [B, T, H, V_PADSIZE//BV, BV * 2]
  v_paded = v
  v_paded = pad_to_multiple(v_paded, BV, -1, 0)
  v_paded = v_paded.reshape(B, T, H, -1, BV)
  v_paded = pad_to_multiple(v_paded, BV*2, -1, 0)

  h_shape = [B, NT, H, K, V_ALIGNED]
  v_new_shape = [B, T, H, V]
  final_state_shape = [N, H, K, V_ALIGNED]
  final_state_internal_shape = [N, H, V_ALIGNED, K]  # transposed for TPU sublane alignment
  h = jnp.zeros(h_shape, dtype=k.dtype)
  v_new = jnp.zeros(v_new_shape, dtype=jnp.float32)
  final_state = jnp.zeros(final_state_internal_shape, dtype=jnp.float32)

  # [B, NT, H, K, V] -> [B, NT, H, V, K]
  h = h.transpose(0, 1, 2, 4, 3)

  # [B, T, H, V] -> [H, B, T, V] -> [H, B, T, V_PADSIZE]
  # -> [H, B, T, V_PADSIZE//BV, BV]
  v_new = v_new.transpose(2, 0, 1, 3)
  v_new = pad_to_multiple(v_new, BV, -1, 0)
  v_new = v_new.reshape(H, B, T, -1, BV)

  if g is not None:
    g_fp32 = g.astype(jnp.float32)
    g_fp32 = g_fp32.reshape(B, T, H, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
  else:
    g_fp32 = None

  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    gk_fp32 = pad_to_multiple(gk_fp32, 128, -1, 0)
    gk_fp32 = gk_fp32.reshape(B, T, H, -1, 64)
  else:
    gk_fp32 = None

  h_spec = jax.ShapeDtypeStruct([B, NT, H, V_ALIGNED, K], h.dtype)
  v_new_spec = jax.ShapeDtypeStruct([H, B, T, V_PADSIZE//BV, BV], jnp.float32)
  final_state_spec = jax.ShapeDtypeStruct(final_state_internal_shape, jnp.float32)

  k_blockspec = pl.BlockSpec([B, T, 1, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  w_blockspec = pl.BlockSpec([B, T, 1, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  v_blockspec = pl.BlockSpec([B, T, 1, v_paded.shape[3], BV * 2], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  g_blockspec = pl.BlockSpec([B, T, H, 128], index_map = lambda v, bh: (0, 0, 0, 0))
  gk_blockspec = pl.BlockSpec([B, T, H, gk_fp32.shape[-2], 64], index_map = lambda v, bh: (0, 0, 0, 0, 0)) if gk_fp32 is not None else None
  init_blockspec = pl.BlockSpec([N, H, V_ALIGNED, K_PADSIZE], index_map = lambda v, bh: (0, 0, 0, 0))
  seqlens_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)
  chunk_offsets_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)

  h_blockspec = pl.BlockSpec([B, NT, H, V_ALIGNED, K], lambda v, bh : (0, 0, 0, 0, 0))
  v_new_blockspec = pl.BlockSpec([H, B, T, V_PADSIZE//BV, BV], lambda v, bh : (0, 0, 0, 0, 0))
  final_out_blockspec = pl.BlockSpec([N, H, V_ALIGNED, K], lambda v, bh : (0, 0, 0, 0))

  grid = (math.ceil(V / BV), N * H)
  interpret = get_interpret()
  h, v_out, final_out = pl.pallas_call(
    functools.partial(
        _chunk_gated_delta_rule_fwd_varlen_kernel,
        B=B,
        T=T,
        NT=NT,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=(g is not None),
        USE_GK=(gk is not None),
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=(final_state is not None),
        SAVE_NEW_VALUE=(v_new is not None),
        USE_EXP2=use_exp2,
        IS_VARLEN=(seqlens is not None),
    ),
    grid=grid,
    in_specs=[k_blockspec, v_blockspec, w_blockspec,
              g_blockspec if (g is not None) else None,
              gk_blockspec if (gk is not None) else None,
              init_blockspec if (initial_state is not None) else None,
              seqlens_blockspec, chunk_offsets_blockspec],
    out_shape=[h_spec, v_new_spec, final_state_spec],
    out_specs=[h_blockspec, v_new_blockspec, final_out_blockspec],
    interpret=interpret,
  )(k_paded, v_paded, w_paded, g_fp32, gk_fp32, initial_state, seqlens, chunk_offsets)

  if save_new_value:
    v_out = v_out.reshape(H, B, T, V_PADSIZE).transpose(1, 2, 0, 3)
    v_out = v_out[:,:,:,:V]
  h = h.transpose(0, 1, 2, 4, 3)
  # Trim V back to original if we padded to V_ALIGNED
  if V_ALIGNED > V:
    h = h[:, :, :, :, :V]
  # final_out is [N, H, V_ALIGNED, K], transpose to [N, H, K, V_ALIGNED]
  final_out = final_out.transpose(0, 1, 3, 2)
  if V_ALIGNED > V:
    final_out = final_out[:, :, :, :V]
  return h, (v_out if save_new_value else None), (final_out if output_final_state else None)


# ── Non-varlen Pallas kernel ────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_kernel(
    k_ref,     # [1, 1, T, K_PADSIZE]
    v_ref,     # [1, 1, T, V_ALIGNED]
    w_ref,     # [1, 1, T, K_PADSIZE]
    g_ref,     # [1, 1, T, G_PAD]
    gk_ref,    # [1, 1, T, K_PADSIZE]
    h0_ref,    # [1, 1, V_ALIGNED, K_PADSIZE]

    # outputs
    h_ref,      # [B, NT, H, V_ALIGNED, K_PADSIZE]
    v_new_ref,  # [H, B, T, NV, BV]
    ht_ref,     # [B, H, V_ALIGNED, K_PADSIZE]

    T, NT, H, K, V, BT, BV, NV,
    USE_G, USE_GK, USE_INITIAL_STATE, STORE_FINAL_STATE, SAVE_NEW_VALUE, USE_EXP2,
):
  """Non-varlen kernel for chunked gated delta rule forward pass.

  Grid: (NV=cdiv(V_ALIGNED, BV), B, H). Each grid point processes one
  (V-slice, batch, head) combination, looping over all NT chunks sequentially
  to maintain the recurrent hidden state b_h: [64, BV].

  Requires T % BT == 0 (caller-enforced), enabling:
   - Static NT -> Python for-loop unrolling (no fori_loop needed)
   - No partial-last-chunk masks
   - Simplified last_idx = (i_t + 1) * BT - 1

  Block spec design follows _chunk_gla_fwd_o_gk_kernel:
   - Inputs transposed to (H, B, T, dim) outside; block [1, 1, T, dim] loads
     one head's full T-length sequence per grid point.
   - Dynamic V-slicing inside via pl.ds(idx_v * BV, BV).
  """
  idx_v = pl.program_id(0)
  idx_b = pl.program_id(1)
  idx_h = pl.program_id(2)

  b_h1 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 64:
    b_h2 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 128:
    b_h3 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 192:
    b_h4 = jnp.zeros([64, BV], dtype=jnp.float32)

  if USE_INITIAL_STATE:
    # h0_ref[0, 0]: [V_ALIGNED, K_PADSIZE]; select [BV, 64] then .T -> [64, BV]
    b_h1 += h0_ref[0, 0, pl.ds(idx_v * BV, BV), 0:64].astype(jnp.float32).T
    if K > 64:
      b_h2 += h0_ref[0, 0, pl.ds(idx_v * BV, BV), 64:128].astype(jnp.float32).T  # type: ignore
    if K > 128:
      b_h3 += h0_ref[0, 0, pl.ds(idx_v * BV, BV), 128:192].astype(jnp.float32).T  # type: ignore
    if K > 192:
      b_h4 += h0_ref[0, 0, pl.ds(idx_v * BV, BV), 192:256].astype(jnp.float32).T  # type: ignore

  for i_t in range(NT):
    # Store current h state.  h layout: [B, NT, H, V_ALIGNED, K_PADSIZE] with V before K
    # (same transposed layout as varlen kernel).  b_h is [64, BV]; .T gives [BV, 64].
    len_k1 = min(K, 64)
    h_ref[idx_b, i_t, idx_h, pl.ds(idx_v * BV, BV), 0:len_k1] = b_h1.T[:, :len_k1].astype(h_ref.dtype)
    if K > 64:
      len_k2 = min(K, 128) - 64
      h_ref[idx_b, i_t, idx_h, pl.ds(idx_v * BV, BV), 64:64+len_k2] = b_h2.T[:, :len_k2].astype(h_ref.dtype)  # type: ignore
    if K > 128:
      len_k3 = min(K, 192) - 128
      h_ref[idx_b, i_t, idx_h, pl.ds(idx_v * BV, BV), 128:128+len_k3] = b_h3.T[:, :len_k3].astype(h_ref.dtype)  # type: ignore
    if K > 192:
      len_k4 = min(K, 256) - 192
      h_ref[idx_b, i_t, idx_h, pl.ds(idx_v * BV, BV), 192:192+len_k4] = b_h4.T[:, :len_k4].astype(h_ref.dtype)  # type: ignore

    # Compute w @ h  (result shape: [BT, BV])
    b_w = w_ref[0, 0, i_t * BT : i_t * BT + BT, 0:64]
    b_v = jnp.dot(b_w.astype(jnp.float32), b_h1, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_w = w_ref[0, 0, i_t * BT : i_t * BT + BT, 64:128]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h2, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 128:
      b_w = w_ref[0, 0, i_t * BT : i_t * BT + BT, 128:192]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h3, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_w = w_ref[0, 0, i_t * BT : i_t * BT + BT, 192:256]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h4, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # v_new = u - w @ h  (u = original values at this chunk's V-slice)
    # v_ref[0, 0]: [T, V_ALIGNED]; pl.ds selects the BV columns for this grid point.
    b_u = v_ref[0, 0, i_t * BT : i_t * BT + BT, pl.ds(idx_v * BV, BV)]
    b_v = b_u.astype(b_v.dtype) - b_v

    if SAVE_NEW_VALUE:
      # v_new_ref layout: [H, B, T, NV, BV]; write BT rows for chunk i_t, V-slice idx_v.
      v_new_ref[idx_h, idx_b, pl.ds(i_t * BT, BT), idx_v, 0:BV] = b_v.astype(v_new_ref.dtype)

    # T % BT == 0 guaranteed (caller-enforced), so last token = (i_t+1)*BT - 1 always.
    last_idx = (i_t + 1) * BT - 1

    if USE_G:
      # g_ref[0, 0]: [T, G_PAD]; scalar gate in column 0.  No mask needed (full chunks only).
      b_g_last = g_ref[0, 0, last_idx, 0].astype(jnp.float32)
      b_g = g_ref[0, 0, i_t * BT : i_t * BT + BT, 0].astype(jnp.float32)
      if USE_EXP2:
        b_v = b_v * jnp.exp2(b_g_last - b_g)[:, None]
        b_g_last = jnp.exp2(b_g_last)
      else:
        b_v = b_v * jnp.exp(b_g_last - b_g)[:, None]
        b_g_last = jnp.exp(b_g_last)
      b_h1 *= b_g_last
      if K > 64:
        b_h2 *= b_g_last
      if K > 128:
        b_h3 *= b_g_last
      if K > 192:
        b_h4 *= b_g_last

    if USE_GK:
      # gk_ref[0, 0]: [T, K_PADSIZE]; per-dim gate.
      o_k1 = jnp.arange(0, 64)
      b_gk_last1 = jnp.where(o_k1 < K, gk_ref[0, 0, last_idx, 0:64], 0).astype(jnp.float32)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]
      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, 0, last_idx, 64:128], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]
      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, 0, last_idx, 128:192], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]
      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, 0, last_idx, 192:256], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    # h += k.T @ v_new
    b_k = k_ref[0, 0, i_t * BT : i_t * BT + BT, 0:64]
    b_h1 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_k = k_ref[0, 0, i_t * BT : i_t * BT + BT, 64:128]
      b_h2 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 128:
      b_k = k_ref[0, 0, i_t * BT : i_t * BT + BT, 128:192]
      b_h3 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_k = k_ref[0, 0, i_t * BT : i_t * BT + BT, 192:256]
      b_h4 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

  if STORE_FINAL_STATE:
    # ht layout: [B, H, V_ALIGNED, K_PADSIZE] (V before K, same as varlen).
    len_k1 = min(K, 64)
    ht_ref[idx_b, idx_h, pl.ds(idx_v * BV, BV), 0:len_k1] = b_h1.T[:, :len_k1].astype(ht_ref.dtype)
    if K > 64:
      len_k2 = min(K, 128) - 64
      ht_ref[idx_b, idx_h, pl.ds(idx_v * BV, BV), 64:64+len_k2] = b_h2.T[:, :len_k2].astype(ht_ref.dtype)
    if K > 128:
      len_k3 = min(K, 192) - 128
      ht_ref[idx_b, idx_h, pl.ds(idx_v * BV, BV), 128:128+len_k3] = b_h3.T[:, :len_k3].astype(ht_ref.dtype)
    if K > 192:
      len_k4 = min(K, 256) - 192
      ht_ref[idx_b, idx_h, pl.ds(idx_v * BV, BV), 192:192+len_k4] = b_h4.T[:, :len_k4].astype(ht_ref.dtype)


def _chunk_gated_delta_rule_fwd(
    k: jax.Array,
    w: jax.Array,
    u: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = False,
):
  """Non-varlen launcher for the chunked gated delta rule forward pass.

  Requires T % chunk_size == 0 (enforced by ``chunk_gated_delta_rule_fwd_h``).
  Uses the dedicated ``_chunk_gated_delta_rule_fwd_kernel`` -- NOT the varlen
  path.  Follows _chunk_gla_fwd_o_gk's block-spec style: inputs are transposed
  to (H, B, T, dim) so each grid point loads one head's full T-length sequence.

  Grid: (NV=cdiv(V_ALIGNED, BV), B, H)

  Args:
      k: [B, T, H, K] -- Keys.
      w: [B, T, H, K] -- Correction weights.
      u: [B, T, H, V] -- Delta-corrected values from intra-chunk.
      g: [B, T, H] -- Scalar per-head gate (optional).
      gk: [B, T, H, K] -- Per-element gate (optional).
      initial_state: [B, H, K, V] -- Initial hidden state (optional).
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation.

  Returns:
      h: [B, NT, H, K, V] -- Hidden states before each chunk.
      v_new: [B, T, H, V] or None -- Delta-corrected values.
      final_state: [B, H, K, V] or None.
  """
  B, T, H, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = T // BT   # exact -- T % BT == 0 enforced by caller

  BV = 128  # must be >=128 so pl.ds(idx_v*BV, BV) on last dim is provably 128-element-aligned on TPU
  K_PADSIZE = int(align_up(K, 64))      # pad K to 64-element blocks
  V_ALIGNED = int(align_up(V, BV))      # pad V to BV-element blocks (multiple of 128)
  NV = V_ALIGNED // BV

  # -- Pad and transpose inputs to (H, B, T, dim) layout ---
  # k, w: [B, T, H, K] -> pad K -> transpose -> (H, B, T, K_PADSIZE)
  k_pad = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else k
  w_pad = jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else w
  k_t = jnp.transpose(k_pad, (2, 0, 1, 3))   # (H, B, T, K_PADSIZE)
  w_t = jnp.transpose(w_pad, (2, 0, 1, 3))   # (H, B, T, K_PADSIZE)

  # u (values): [B, T, H, V] -> pad V -> transpose -> (H, B, T, V_ALIGNED)
  u_pad = jnp.pad(u, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V))) if V_ALIGNED > V else u
  v_t = jnp.transpose(u_pad, (2, 0, 1, 3))   # (H, B, T, V_ALIGNED)

  # g (scalar gate): [B, T, H] -> float32 -> [B, T, H, 1] -> pad last -> [H, B, T, G_PAD]
  if g is not None:
    g_fp32 = g.astype(jnp.float32).reshape(B, T, H, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)   # (B, T, H, 128)
    g_t = jnp.transpose(g_fp32, (2, 0, 1, 3))       # (H, B, T, 128)
  else:
    g_t = None

  # gk (per-dim gate): [B, T, H, K] -> float32 -> pad K -> transpose -> (H, B, T, K_PADSIZE)
  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    if K_PADSIZE > K:
      gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    gk_t = jnp.transpose(gk_fp32, (2, 0, 1, 3))     # (H, B, T, K_PADSIZE)
  else:
    gk_t = None

  # h0 (initial state): [N=B, H, K, V] -> [B, H, V, K] -> pad V, K -> transpose -> (H, B, V_ALIGNED, K_PADSIZE)
  if initial_state is not None:
    h0 = jnp.transpose(initial_state, (0, 1, 3, 2))  # [B, H, V, K]
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, V_ALIGNED - V), (0, 0)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    h0 = jnp.transpose(h0, (1, 0, 2, 3))             # (H, B, V_ALIGNED, K_PADSIZE)
  else:
    h0 = None

  # -- Output shapes ---
  # h stored as [B, NT, H, V_ALIGNED, K_PADSIZE] with V before K (varlen convention).
  h_spec     = jax.ShapeDtypeStruct([B, NT, H, V_ALIGNED, K_PADSIZE], k.dtype)
  v_new_spec = jax.ShapeDtypeStruct([H, B, T, NV, BV], jnp.float32)
  ht_spec    = jax.ShapeDtypeStruct([B, H, V_ALIGNED, K_PADSIZE], jnp.float32)

  # -- Block specs ---
  g_pad_size = g_t.shape[-1] if g_t is not None else 128  # always 128 after pad_to_multiple

  k_blockspec  = pl.BlockSpec([1, 1, T, K_PADSIZE], index_map=lambda v, b, h: (h, b, 0, 0))
  v_blockspec  = pl.BlockSpec([1, 1, T, V_ALIGNED], index_map=lambda v, b, h: (h, b, 0, 0))
  w_blockspec  = pl.BlockSpec([1, 1, T, K_PADSIZE], index_map=lambda v, b, h: (h, b, 0, 0))
  g_blockspec  = pl.BlockSpec([1, 1, T, g_pad_size], index_map=lambda v, b, h: (h, b, 0, 0)) if g is not None else None
  gk_blockspec = pl.BlockSpec([1, 1, T, K_PADSIZE], index_map=lambda v, b, h: (h, b, 0, 0)) if gk is not None else None
  h0_blockspec = pl.BlockSpec([1, 1, V_ALIGNED, K_PADSIZE], index_map=lambda v, b, h: (h, b, 0, 0)) if initial_state is not None else None

  h_blockspec_out     = pl.BlockSpec([B, NT, H, V_ALIGNED, K_PADSIZE], lambda v, b, h: (0, 0, 0, 0, 0))
  v_new_blockspec_out = pl.BlockSpec([H, B, T, NV, BV],                lambda v, b, h: (0, 0, 0, 0, 0))
  ht_blockspec_out    = pl.BlockSpec([B, H, V_ALIGNED, K_PADSIZE],     lambda v, b, h: (0, 0, 0, 0))

  grid = (NV, B, H)
  interpret = get_interpret()
  h_out, v_new_out, ht_out = pl.pallas_call(
    functools.partial(
        _chunk_gated_delta_rule_fwd_kernel,
        T=T, NT=NT, H=H, K=K, V=V, BT=BT, BV=BV, NV=NV,
        USE_G=(g is not None),
        USE_GK=(gk is not None),
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state,
        SAVE_NEW_VALUE=save_new_value,
        USE_EXP2=use_exp2,
    ),
    grid=grid,
    in_specs=[k_blockspec, v_blockspec, w_blockspec,
              g_blockspec, gk_blockspec, h0_blockspec],
    out_shape=[h_spec, v_new_spec, ht_spec],
    out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
    interpret=interpret,
  )(k_t, v_t, w_t, g_t, gk_t, h0)

  # -- Post-process outputs ---
  # h: [B, NT, H, V_ALIGNED, K_PADSIZE] -> trim K -> transpose (V<->K) -> trim V -> [B, NT, H, K, V]
  h_out = h_out[:, :, :, :, :K]              # trim padded K
  h_out = h_out.transpose(0, 1, 2, 4, 3)     # [B, NT, H, K, V_ALIGNED]
  if V_ALIGNED > V:
    h_out = h_out[:, :, :, :, :V]            # trim padded V

  # v_new: [H, B, T, NV, BV] -> [H, B, T, V_ALIGNED] -> [B, T, H, V]
  if save_new_value:
    v_new_out = v_new_out.reshape(H, B, T, V_ALIGNED)
    v_new_out = v_new_out.transpose(1, 2, 0, 3)   # [B, T, H, V_ALIGNED]
    if V_ALIGNED > V:
      v_new_out = v_new_out[:, :, :, :V]
  else:
    v_new_out = None

  # ht: [B, H, V_ALIGNED, K_PADSIZE] -> trim K -> transpose (V<->K) -> trim V -> [B, H, K, V]
  if output_final_state:
    ht_out = ht_out[:, :, :, :K]             # trim padded K
    ht_out = ht_out.transpose(0, 1, 3, 2)    # [B, H, K, V_ALIGNED]
    if V_ALIGNED > V:
      ht_out = ht_out[:, :, :, :V]
  else:
    ht_out = None

  return h_out, v_new_out, ht_out


# ── Public dispatch function ────────────────────────────────────────────────


def chunk_gated_delta_rule_fwd_h(
    k: jax.Array,          # [B, T, H, K]
    w: jax.Array,          # [B, T, H, K]
    u: jax.Array,          # [B, T, H, V]
    g: jax.Array | None = None,   # [B, T, H] scalar gate
    gk: jax.Array | None = None,  # [B, T, H, K] per-element gate
    initial_state: jax.Array | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = True,
    cu_seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Dispatch delta-rule inter-chunk state forward to varlen or non-varlen kernel.

    Computes the chunked recurrence where the value is delta-corrected before
    accumulation:

        v_new_t = u_t - w_t @ h_{t-1}    (delta correction)
        h_t = h_{t-1} * decay_t + k_t^T @ v_new_t  (state update)

    Args:
        k: [B, T, H, K] -- Keys (gated, ready for outer product).
        w: [B, T, H, K] -- Correction weights.
        u: [B, T, H, V] -- Delta-corrected values from intra-chunk.
        g: [B, T, H] -- Scalar per-head gate in log2 space. Optional.
        gk: [B, T, H, K] -- Per-element gate in log2 space. Optional.
        initial_state: [N, H, K, V] -- Initial hidden state. N=B for non-varlen.
        output_final_state: Whether to return final hidden state.
        chunk_size: Chunk size. T must be divisible by chunk_size.
        save_new_value: Whether to compute and return v_new.
        use_exp2: Use exp2 for gate computation (True for log2 space).
        cu_seqlens: [N+1] -- Cumulative sequence lengths for varlen. Optional.
        chunk_indices: Precomputed chunk indices for varlen. Optional.

    Returns:
        h: [B, NT, H, K, V] -- Hidden states before each chunk.
        v_new: [B, T, H, V] -- Delta-corrected values, or None.
        final_state: [N, H, K, V] or None.
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = cdiv(T, BT)
    N = B if cu_seqlens is None else cu_seqlens.shape[-1] - 1

    # -- Input validation ---
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(u, (B, T, H, V), "u")
    assert_shape_or_none(g, (B, T, H), "g")
    assert_shape_or_none(gk, (B, T, H, K), "gk")
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    assert (cu_seqlens is None) or (B == 1), f"varlen mode requires B==1, got B={B}"

    if cu_seqlens is None:
        assert T % chunk_size == 0, "For non-varlen input, T must be divisible by chunk_size"
        return _chunk_gated_delta_rule_fwd(
            k, w, u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            use_exp2=use_exp2,
        )
    else:
        return _chunk_gated_delta_rule_fwd_varlen(
            k, w, u,
            seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            use_exp2=use_exp2,
        )


__all__ = export_public(globals())
