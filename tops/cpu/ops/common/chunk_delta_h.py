"""KDA-specific inter-chunk hidden state propagation (delta rule).
Unlike GLA's chunk_h.py which uses simple k^T @ v accumulation, the delta
rule variant includes a WY correction: v_corrected = u - w @ S before
accumulation. This matches FLA's chunk_gated_delta_rule_fwd_kernel_h.
Gate behavior (gk only, per-element K-dim gate, log2 space):
  - k is already gated as kg = k * exp2(gn - g)
  - h *= exp2(gk_last) at each chunk boundary
  - v is NOT separately gated (gating is baked into w, u via WY transform)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from tops.cpu.ops.common.utils import acc_dtype, cdiv, dot, pad_to_multiple
from tops.ops.utils import exp2
from tops.utils import assert_shape


def chunk_gated_delta_rule_fwd_h(
  k: jax.Array,
  w: jax.Array,
  u: jax.Array,
  *,
  g: jax.Array | None = None,
  gk: jax.Array | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  save_new_value: bool = True,
  cu_seqlens: jax.Array | None = None,
  use_exp2: bool = False,
  transpose_state_layout: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
  """Forward hidden state recurrence for gated delta rule.
  For each chunk t:
    1. Store h_t (hidden state entering this chunk)
    2. v_corrected = u_t - w_t @ h_t  (delta rule correction)
    3. h_{t+1} = h_t * exp(gk_last) + kg_t^T @ v_corrected
  Args:
      k:  [B, T, H, K] — gated keys (kg = k * exp(gn - g))
      w:  [B, T, H, K] — effective keys from WY transform
      u:  [B, T, H, V] — effective values from WY transform
      g:  unused (kept for API compat with FLA)
      gk: [B, T, H, K] — chunk-local cumsummed K-dim gates (log2 space)
      initial_state: [B, H, K, V] or [N, H, K, V] — initial hidden state
      output_final_state: whether to return final state
      chunk_size: block size
      save_new_value: whether to return v_new (corrected values)
      cu_seqlens: variable-length support (not implemented in v1)
      transpose_state_layout: state layout transposition (not implemented in v1)
  Returns:
      h:           [B, NT, H, K, V] — per-chunk hidden states
      v_new:       [B, T, H, V] — corrected values, or None
      final_state: [B, H, K, V] or None
  """
  del g, cu_seqlens, transpose_state_layout  # not used in v1

  B, T, H, K = k.shape
  V = u.shape[-1]
  C = chunk_size
  NT = T // C
  acc = acc_dtype(k.dtype)

  # Reshape to chunks: [B, NT, C, H, D]
  k_c = k.reshape(B, NT, C, H, K)
  w_c = w.reshape(B, NT, C, H, K)
  u_c = u.reshape(B, NT, C, H, V)
  if gk is not None:
    gk_c = gk.reshape(B, NT, C, H, K)

  h = jnp.zeros((B, H, K, V), dtype=acc)
  if initial_state is not None:
    h = h + initial_state.astype(acc)

  h_list = []
  v_new_list = [] if save_new_value else None

  for i in range(NT):
    h_list.append(h)

    b_k = k_c[:, i]  # [B, C, H, K]
    b_w = w_c[:, i]  # [B, C, H, K]
    b_u = u_c[:, i]  # [B, C, H, V]

    # Delta rule correction: v_corrected = u - w @ h
    # w: [B, C, H, K], h: [B, H, K, V] -> w @ h: [B, C, H, V]
    v_corr = b_u - jnp.einsum("bchk,bhkv->bchv", b_w, h)

    if save_new_value:
      v_new_list.append(v_corr)

    # Gate: h *= exp2(gk_last)
    if gk is not None:
      b_gk = gk_c[:, i]  # [B, C, H, K]
      gk_last = b_gk[:, -1]  # [B, H, K]
      h = h * jnp.exp2(gk_last[:, :, :, None])

    # Accumulate: h += k^T @ v_corrected
    h = h + dot("bchk,bchv->bhkv", b_k, v_corr, acc)

  h_all = jnp.stack(h_list, axis=1)  # [B, NT, H, K, V]
  ht = h if output_final_state else None

  if save_new_value:
    v_new = jnp.concatenate(v_new_list, axis=1).reshape(B, T, H, V)
  else:
    v_new = None

  return h_all, v_new, ht


def chunk_gated_delta_rule_bwd_dhu(
    q: jax.Array,
    k: jax.Array,
    w: jax.Array,
    do: jax.Array,
    dv: jax.Array,
    *,
    gk: jax.Array | None = None,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float = 1.0,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    transpose_state_layout: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
  """Backward recurrence through chunks computing dh, dh0, dv2.

  Mirrors chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 from Triton.

  Args:
      q:    [B, T, H, K]   gated query tensor.
      k:    [B, T, H, K]   gated key tensor.
      w:    [B, T, H, K]   delta-rule erase weight.
      do:   [B, T, H, V]   output gradient.
      dv:   [B, T, H, V]   value gradient from dAv stage.
      gk:   [B, T, H, K]   per-key gate (cumsum'd, log2-space).
      h0:   [B, H, K, V]   initial hidden state or None.
      dht:  [B, H, K, V]   gradient of final state or None.
      scale: float          softmax scaling factor.
      chunk_size: int       chunk block size.

  Returns:
      dh:  [B, NT, H, K, V]  per-chunk hidden state gradient.
      dh0: [B, H, K, V]     gradient of initial state or None.
      dv2: [B, T, H, V]     updated value gradient.
  """
  del cu_seqlens, chunk_indices, transpose_state_layout  # not used in v1
  B, T, H, K = q.shape
  V = do.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)

  # =================== input shape assertions ===================
  assert_shape(q, (B, T, H, K), "q")
  assert_shape(k, (B, T, H, K), "k")
  assert_shape(w, (B, T, H, K), "w")
  if gk is not None:
      assert_shape(gk, (B, T, H, K), "gk")
  assert_shape(do, (B, T, H, V), "do")
  assert_shape(dv, (B, T, H, V), "dv")
  if h0 is not None:
      assert_shape(h0, (B, H, K, V), "h0")
  if dht is not None:
      assert_shape(dht, (B, H, K, V), "dht")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  # ==============================================================

  def _per_batch_head(q_bh, k_bh, w_bh, gk_bh, do_bh, dv_bh, dht_bh, h0_bh):
      """Process one (batch, head). All shapes: q,k,w,gk=[T,K], do,dv=[T,V],
      dht_bh=[K,V], h0_bh=[K,V]."""
      T_actual = q_bh.shape[0]
      T_padded = NT * BT

      q_bh = pad_to_multiple(q_bh, T_padded, 0)
      k_bh = pad_to_multiple(k_bh, T_padded, 0)
      w_bh = pad_to_multiple(w_bh, T_padded, 0)
      gk_bh = pad_to_multiple(gk_bh, T_padded, 0)
      do_bh = pad_to_multiple(do_bh, T_padded, 0)
      dv_bh = pad_to_multiple(dv_bh, T_padded, 0)

      # Reshape into chunks [NT, BT, D]
      q_c = q_bh.reshape(NT, BT, K)
      k_c = k_bh.reshape(NT, BT, K)
      w_c = w_bh.reshape(NT, BT, K)
      gk_c = gk_bh.reshape(NT, BT, K)
      do_c = do_bh.reshape(NT, BT, V)
      dv_c = dv_bh.reshape(NT, BT, V)

      # Reverse scan through chunks
      def scan_fn(carry, t_rev):
          b_dh = carry  # [K, V]
          i_t = NT - 1 - t_rev

          b_q = q_c[i_t]    # [BT, K]
          b_k = k_c[i_t]    # [BT, K]
          b_w = w_c[i_t]    # [BT, K]
          b_gk = gk_c[i_t]  # [BT, K]
          b_do = do_c[i_t]  # [BT, V]
          b_dv = dv_c[i_t]  # [BT, V]

          # Store current dh for this chunk
          dh_out = b_dh  # [K, V]

          # Last valid position gate value
          last_idx = jnp.minimum((i_t + 1) * BT, T_actual) - 1
          chunk_start = i_t * BT
          local_last = last_idx - chunk_start  # local index within chunk

          # gk at last position: [K]
          b_gk_last = b_gk[local_last]

          # Compute dv2: inter-chunk contribution
          # dv2 = k @ dh + dv   (k is [BT,K], dh is [K,V] → [BT,V])
          b_dv2 = jnp.dot(b_k.astype(jnp.float32), b_dh.astype(jnp.float32))  # [BT, V]

          # Apply gating: dv2 *= exp2(gk_last - gk) per position per K-dim
          # But dv2 is [BT,V] and gk_last/gk are [K]-dim. The gating was already baked
          # into k (k is kg = k * exp2(gk_last - gk)), so we only need scalar gate.
          # Actually in the Triton kernel, k is kg (gated key), and gk here is the
          # cumsum gate. The kernel applies exp2(gk_last - gk) as a per-position scalar
          # from the scalar gate g (not gk). But in KDA's case, there's no scalar g,
          # only gk. Looking at the Triton code: USE_GK path applies
          # exp2(gk_last[:]) to dh, not to dv2. The USE_G path (scalar gate) applies
          # exp2(g_last - g[i]) to dv2. In KDA, the caller passes gk as gk, g is None.
          # So USE_G=False, USE_GK=True.
          # This means: no gating on dv2, only gk gating on dh.

          b_dv2 = b_dv2 + b_dv  # [BT, V]

          # Update dh: apply gk gating first
          b_dh = b_dh * exp2(b_gk_last[:, None])  # [K, V] gated per K-dim

          # Then accumulate: dh += q^T @ do * scale - w^T @ dv2
          # q,w are loaded transposed in Triton: (K,T) with (1, H*K) strides
          # So dot is q^T[K,BT] @ do[BT,V] = [K,V]
          b_dh = b_dh + (
              jnp.dot(b_q.astype(jnp.float32).T, b_do.astype(jnp.float32)) * scale
              - jnp.dot(b_w.astype(jnp.float32).T, b_dv2.astype(jnp.float32))
          )

          return b_dh, (dh_out, b_dv2)

      # Initialize from dht
      init_dh = dht_bh.astype(jnp.float32) if dht_bh is not None else jnp.zeros((K, V), dtype=jnp.float32)

      final_dh, (dh_all, dv2_all) = lax.scan(scan_fn, init_dh, jnp.arange(NT))
      # dh_all is [NT, K, V] in reverse order (t_rev=0 → chunk NT-1)
      # Reverse to get natural order
      dh_all = dh_all[::-1]
      dv2_all = dv2_all[::-1]

      dv2_out = dv2_all.reshape(T_padded, V)[:T_actual]
      dh0_out = final_dh  # [K, V] — gradient of initial state

      return dh_all, dh0_out, dv2_out

  # Reshape [B,T,H,D] → [B*H, T, D]
  def _to_bh(x, d):
      return x.transpose(0, 2, 1, 3).reshape(B * H, T, d)

  q_flat = _to_bh(q, K)
  k_flat = _to_bh(k, K)
  w_flat = _to_bh(w, K)
  gk_flat = _to_bh(gk, K)
  do_flat = _to_bh(do, V)
  dv_flat = _to_bh(dv, V)

  # dht: [B, H, K, V] → [B*H, K, V]
  if dht is not None:
      dht_flat = dht.reshape(B * H, K, V)
  else:
      dht_flat = jnp.zeros((B * H, K, V), dtype=jnp.float32)

  # h0: [B, H, K, V] → [B*H, K, V]
  if h0 is not None:
      h0_flat = h0.reshape(B * H, K, V)
  else:
      h0_flat = jnp.zeros((B * H, K, V), dtype=jnp.float32)

  dh_flat, dh0_flat, dv2_flat = jax.vmap(_per_batch_head)(
      q_flat, k_flat, w_flat, gk_flat, do_flat, dv_flat, dht_flat, h0_flat
  )

  # dh_flat: [B*H, NT, K, V] → [B, H, NT, K, V] → [B, NT, H, K, V]
  dh = dh_flat.reshape(B, H, NT, K, V).transpose(0, 2, 1, 3, 4)

  # dh0: [B*H, K, V] → [B, H, K, V]
  dh0 = dh0_flat.reshape(B, H, K, V) if h0 is not None else None

  # dv2: [B*H, T, V] → [B, H, T, V] → [B, T, H, V]
  dv2 = dv2_flat.reshape(B, H, T, V).transpose(0, 2, 1, 3)

  return dh, dh0, dv2



__all__ = [
  "chunk_gated_delta_rule_fwd_h",
  "chunk_gated_delta_rule_bwd_dhu",
]
