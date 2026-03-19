import jax

from tops.ops.common.chunk_h import chunk_fwd_h_kernel as chunk_fwd_h
from tops.ops.common.chunk_o import chunk_fwd_o


def chunk_simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
  B, T, H, K, V = *q.shape, v.shape[-1]

  assert (B, T, H, K) == k.shape
  assert (B, T, H, V) == v.shape
  assert (g is None) or ((B, T, H) == g.shape)
  assert (g_gamma is None) or ((H,) == g_gamma.shape)
  assert (h0 is None) or ((B, H, K, V) == h0.shape)
  assert (cu_seqlens is None) or ((B + 1,) == cu_seqlens.shape)
  assert T % chunk_size == 0
  assert (cu_seqlens is None) or (cu_seqlens % chunk_size == 0).all()
  assert (K % 128 == 0) and (V % 128 == 0)

  h, ht = chunk_fwd_h(
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      gk=None,
      gv=None,
      h0=h0,
      output_final_state=use_ht,
      states_in_fp32=False,
      cu_seqlens=cu_seqlens,
      chunk_size=chunk_size,
  )
  o = chunk_fwd_o(
      q=q,
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      h=h,
      scale=scale,
      cu_seqlens_cpu=cu_seqlens,
      chunk_size=chunk_size,
  )
  return o, ht

