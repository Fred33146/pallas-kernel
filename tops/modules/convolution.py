import jax
import jax.numpy as jnp
from flax import nnx


class ShortConvolution(nnx.Module):
    """Causal depthwise 1D convolution (JAX/Flax NNX version).

    - groups = hidden_size (depthwise separable)
    - causal padding = kernel_size - 1 (left padding)
    - optional SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation

        # JAX Conv: kernel shape [K, in_features/groups, out_features]
        # For depthwise: feature_group_count=hidden_size, so in_features/groups = 1
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(kernel_size,),
            feature_group_count=hidden_size,
            use_bias=bias,
            padding="VALID",
            rngs=rngs,
        )

    def _apply_activation(self, x: jax.Array) -> jax.Array:
        if self.activation in ("silu", "swish"):
            return jax.nn.silu(x)
        return x

    def _causal_conv1d(
        self,
        x: jax.Array,
        cu_seqlens=None,
    ) -> jax.Array:
        """Apply causal conv1d.
        x: [B, T, D] -> [B, T, D]  (depthwise causal convolution + optional SiLU)
        """
        W = self.kernel_size

        if cu_seqlens is not None:
            B, T_total, D = x.shape
            assert B == 1, "cu_seqlens requires B=1"
            N = len(cu_seqlens) - 1
            segments = []
            for i in range(N):
                bos = int(cu_seqlens[i])
                eos = int(cu_seqlens[i + 1])
                seg = x[:, bos:eos, :]  # [1, seg_len, D]
                # Manual left padding: pad on the time axis
                seg_padded = jnp.pad(seg, ((0, 0), (W - 1, 0), (0, 0)))
                seg_out = self.conv(seg_padded)  # [1, seg_len, D]
                segments.append(seg_out)
            y = jnp.concatenate(segments, axis=1)
        else:
            # Manual left padding for causal convolution
            x_padded = jnp.pad(x, ((0, 0), (W - 1, 0), (0, 0)))
            y = self.conv(x_padded)

        return self._apply_activation(y)

    def step(
        self,
        x: jax.Array,
        cache: jax.Array | None,
        output_final_state: bool = False,
        cu_seqlens=None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Single-step decoding with cache update.

        x: [B, 1, D] or [1, N, D] with cu_seqlens
        cache: [N, W, D] (JAX channels-last, unlike PyTorch [N, D, W])
        """
        W = self.kernel_size
        B = x.shape[0]
        D = self.hidden_size
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if output_final_state and cache is None:
            cache = jnp.zeros((N, W, D))

        # Get current token
        if cu_seqlens is not None:
            x_step = x[0]  # [N, D]
        else:
            x_step = x[:, 0, :]  # [B, D]

        if cache is not None:
            # Roll cache and put new token at the last position
            cache = jnp.roll(cache, shift=-1, axis=1)
            cache = cache.at[:, -1, :].set(x_step)
            # Dot product with convolution weights
            # conv.kernel shape: [K, 1, C] -> squeeze to [K, C]
            w = self.conv.kernel.value[:, 0, :]  # [K, C]
            y = (cache * w).sum(axis=1)  # [N, D]
            if self.conv.bias is not None:
                y = y + self.conv.bias.value
        else:
            w = self.conv.kernel.value[:, 0, :]  # [K, C]
            y = x_step * w[-1]
            if self.conv.bias is not None:
                y = y + self.conv.bias.value

        y = self._apply_activation(y)
        y = y.reshape(x.shape)
        return y, cache

    def __call__(
        self,
        x: jax.Array,
        cache: jax.Array | None = None,
        output_final_state: bool = False,
        cu_seqlens=None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass.

        Args:
            x: [B, T, D]
            cache: [N, W, D] or None
            output_final_state: whether to return final conv state
            cu_seqlens: [N+1] or None

        Returns:
            y: [B, T, D]
            final_state: [N, W, D] or None
        """
        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        W = self.kernel_size

        # Decoding mode: single token per sequence
        if B * T == N:
            return self.step(x, cache, output_final_state, cu_seqlens)

        # Prefill / training mode
        y = self._causal_conv1d(x, cu_seqlens)

        # Calculate final state (window of the last W tokens)
        final_state = None
        if output_final_state:
            if cu_seqlens is not None:
                final_states = []
                for i in range(N):
                    bos = int(cu_seqlens[i])
                    eos = int(cu_seqlens[i + 1])
                    seg = x[0, bos:eos, :]  # [seg_len, D]
                    # Left pad to ensure at least W tokens
                    if seg.shape[0] < W:
                        seg = jnp.pad(seg, ((W - seg.shape[0], 0), (0, 0)))
                    final_states.append(seg[-W:, :])  # [W, D]
                final_state = jnp.stack(final_states, axis=0)  # [N, W, D]
            else:
                if T < W:
                    x_padded = jnp.pad(x, ((0, 0), (W - T, 0), (0, 0)))
                else:
                    x_padded = x
                final_state = x_padded[:, -W:, :]  # [B, W, D]

        return y, final_state
