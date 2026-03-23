import functools

import jax


def cpu_reference(fn):
    """Decorator ensuring CPU reference implementations always run on CPU.

    Moves all JAX array inputs to CPU and sets the default device to CPU,
    so that all intermediate computations also stay on CPU. This guarantees
    deterministic, hardware-independent results for reference implementations.

    Usage:
        @cpu_reference
        def my_cpu_kernel(q, k, v, ...):
            ...
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            args = tuple(
                jax.device_put(a, cpu) if isinstance(a, jax.Array) else a
                for a in args
            )
            kwargs = {
                k: jax.device_put(v, cpu) if isinstance(v, jax.Array) else v
                for k, v in kwargs.items()
            }
            return fn(*args, **kwargs)

    return wrapper
