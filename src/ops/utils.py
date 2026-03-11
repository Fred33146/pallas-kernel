import jax

_IS_TPU_RUNTIME_CACHED: bool | None = None


def is_tpu_runtime() -> bool:
    """Return True if the current JAX runtime is on TPU devices.

    Prefer checking actual devices; fall back to default backend if necessary.
    """
    global _IS_TPU_RUNTIME_CACHED
    if _IS_TPU_RUNTIME_CACHED is not None:
        return _IS_TPU_RUNTIME_CACHED
    try:
        devs = jax.devices()
        _IS_TPU_RUNTIME_CACHED = len(devs) > 0 and all(
            d.platform == "tpu" for d in devs
        )
    except Exception:
        _IS_TPU_RUNTIME_CACHED = jax.default_backend() == "tpu"
    return _IS_TPU_RUNTIME_CACHED
