import math

import torch
import jax
import jax.numpy as jnp
import numpy as np


def torch_to_jax(t: torch.Tensor, dtype=None) -> jax.Array:
    """Convert a torch tensor to a JAX array.

    Args:
        t: Input torch tensor.
        dtype: Target JAX dtype. If None, preserves bfloat16 when the input
            is bfloat16, otherwise uses float32.
    """
    np_arr = t.detach().cpu().float().numpy()
    jax_arr = jnp.array(np_arr)
    if dtype is not None:
        return jax_arr.astype(dtype)
    if t.dtype == torch.bfloat16:
        return jax_arr.astype(jnp.bfloat16)
    return jax_arr


def build_alibi_slopes(n_attention_heads: int) -> np.ndarray:
    """Build ALiBi slopes identical to MaxText BailingMoeV2LinearAttention.

    Args:
        n_attention_heads: Number of attention heads.

    Returns:
        np.ndarray of shape (n_attention_heads,) with float32 slopes.
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    return np.array(get_slopes(n_attention_heads), dtype=np.float32)


def make_alibi_g_gamma(
    H: int, num_layers: int, layer_idx: int
) -> jnp.ndarray:
    """Build g_gamma exactly as MaxText does for a given layer.

    Args:
        H: Number of attention heads.
        num_layers: Total number of decoder layers.
        layer_idx: Index of the current layer (0-based).

    Returns:
        jnp.ndarray of shape (H,) with per-head negative decay rates.
    """
    slope_base = build_alibi_slopes(H)
    denom = max(num_layers - 1, 1)
    slope_scale = 1.0 - layer_idx / denom + 1e-5
    return jnp.array(-slope_base * slope_scale)


def compute_ulp(x, dtype=None):
    """Compute 1 ULP (Unit in the Last Place) at each value in *x*.

    Works with both numpy arrays and torch tensors.

    The general formula for a normal floating-point number is::

        ULP(x) = 2^(frexp_exp(x) - nmant - 1)

    where *nmant* is the number of explicit mantissa bits of the target
    *dtype*.  For subnormals / zero the ULP is clamped to the smallest
    representable positive value of the dtype (``tiny * eps``).

    Args:
        x: numpy ndarray or torch Tensor.
        dtype: floating-point dtype for ULP computation.  Accepts both
            numpy dtypes (``np.float16``, ``np.float32``, …) and torch
            dtypes (``torch.bfloat16``, ``torch.float16``, …).
            If *None*, ``x.dtype`` is used.

    Returns:
        ULP values (float32), same container type and shape as *x*.
    """
    is_torch = isinstance(x, torch.Tensor)

    if dtype is None:
        dtype = x.dtype

    # --- extract dtype parameters ---
    if isinstance(dtype, torch.dtype):
        fi = torch.finfo(dtype)
    else:
        fi = np.finfo(dtype)
    nmant = round(-math.log2(float(fi.eps)))
    min_ulp = float(fi.tiny) * float(fi.eps)

    # --- compute ---
    if is_torch:
        abs_x = x.detach().abs().float()
        _, exp = torch.frexp(abs_x)
        ulp = torch.ldexp(
            torch.ones_like(abs_x), (exp.int() - nmant - 1)
        )
        ulp = ulp.clamp_min(min_ulp)
        ulp = torch.where(abs_x == 0, min_ulp, ulp)
        return ulp
    else:
        abs_x = np.abs(x).astype(np.float64)
        _, exp = np.frexp(abs_x)
        ulp = np.ldexp(1.0, exp.astype(np.int32) - nmant - 1)
        ulp = np.maximum(ulp, min_ulp)
        ulp = np.where(abs_x == 0, min_ulp, ulp)
        return ulp


def compare_tensor(
    name: str,
    gold: np.ndarray | jax.Array | torch.Tensor,
    tensor: np.ndarray | jax.Array | torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    max_ulp: int = 1,
    dtype=torch.bfloat16,
    compare_dtype=np.float64,
) -> bool:
    if gold is None and tensor is None:
        print(f"[{name}] Both are None. MATCH.")
        return False
    if gold is None or tensor is None:
        print(f"[{name}] One is None! MISMATCH.")
        return False

    if isinstance(gold, torch.Tensor):
        gold = gold.detach().cpu().to(torch.float64).numpy().astype(compare_dtype)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().to(torch.float64).numpy().astype(compare_dtype)
    if isinstance(gold, jax.Array):
        gold = np.array(gold).astype(compare_dtype)
    if isinstance(tensor, jax.Array):
        tensor = np.array(tensor).astype(compare_dtype)
    if isinstance(gold, np.ndarray):
        gold = gold.astype(compare_dtype)
    if isinstance(tensor, np.ndarray):
        tensor = tensor.astype(compare_dtype)

    if gold.shape != tensor.shape:
        print(
            f"[{name}] Shape mismatch: Left {gold.shape} vs Right {tensor.shape}. FAIL."
        )
        if gold.squeeze().shape == tensor.squeeze().shape:
            print(
                f"  Attempting comparison with squeezed shapes: {gold.squeeze().shape}"
            )
            gold = gold.squeeze()
            tensor = tensor.squeeze()
        else:
            return False

    diff = np.abs(gold - tensor)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(tensor))
    max_rel_diff = np.max(diff / (np.abs(tensor) + 1e-12))

    is_close = np.allclose(gold, tensor, atol=atol, rtol=rtol, equal_nan=True)

    # ULP check: if diff <= max_ulp ULPs, also consider it passing
    if not is_close:
        ulp = compute_ulp(np.maximum(np.abs(gold), np.abs(tensor)), dtype=dtype)
        tolerance = np.maximum(atol + rtol * np.abs(tensor), max_ulp * ulp)
        is_close = bool(np.all(diff <= tolerance))

    status = "PASS" if is_close else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Max Value        : {max_val:.6e}")
    print(f"  Max Abs Diff     : {max_diff:.6e}")
    print(f"  Max Rel Diff     : {max_rel_diff:.6e}")

    if not is_close:
        ulp = compute_ulp(np.maximum(np.abs(gold), np.abs(tensor)), dtype=dtype)
        tolerance = np.maximum(atol + rtol * np.abs(tensor), max_ulp * ulp)
        error_ratio = diff / (tolerance + 1e-12)
        idx = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
        ulp_diff = diff[idx] / ulp[idx]
        print(f"  Max Mismatch details at index {idx}:")
        print(f"    Left (Triton)  = {gold[idx]}")
        print(f"    Right (Pallas) = {tensor[idx]}")
        print(f"    Diff           = {diff[idx]}")
        print(
            f"    Tolerance      = {tolerance[idx]} (max of atol+rtol*|Right|, {max_ulp}*ulp)"
        )
        print(f"    ULP diff       = {ulp_diff:.2f}")
        print(f"    Ratio          = {error_ratio[idx]}")

    return is_close
