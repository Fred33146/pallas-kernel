# Switch: Native Function Dispatch

## Overview

The `switch` module (`tops/switch.py`) provides a decorator-based mechanism to dynamically switch between accelerator tpu kernel implementations and their jax-native counterparts at runtime, controlled entirely by environment variables.

This enables:
- Running the same codebase on jax native for debugging and testing without code changes.
- Selectively switching specific functions or modules to jax native implementations.
- Globally enabling jax-native mode for the entire project.

## Architecture

```
tops/
├── ops/                    # Accelerator (TPU) implementations
│   ├── gla/
│   │   └── fused_recurrent.py   ──┐
│   └── simple_gla/               │  @switch_func maps
│       └── chunk_h.py         ──┐ │  tops.X -> tops.cpu.X
├── cpu/                         │ │
│   └── ops/                     │ │
│       ├── gla/                 │ │
│       │   └── fused_recurrent.py ◄┘
│       └── simple_gla/          │
│           └── chunk_h.py     ◄─┘
├── env.py                  # Environment variable names
└── switch.py               # Dispatch logic
```

## Usage

Apply `@switch_func` as a decorator on any accelerator kernel function:

```python
from tops.switch import switch_func

@switch_func
def chunk_fwd_h(q, k, v, gk):
    # TPU Pallas kernel implementation
    ...
```

The decorator looks up the corresponding jax native function at `tops.cpu.<same.module.path>.chunk_fwd_h`. If the environment is configured to use native mode and the jax native function exists with a matching signature, the decorated function is replaced by the jax native version transparently.

## Environment Variables

| Variable | Type | Description |
|---|---|---|
| `TOPS_NATIVE` | `bool` | Master switch. Must be `true`/`1` to enable any native dispatch. |
| `TOPS_USE_NATIVE_MODULES` | `csv` | Comma-separated list of full module paths to switch (e.g. `tops.ops.gla.chunk_h`). |
| `TOPS_USE_NATIVE_FUNCS` | `csv` | Comma-separated list of function names to switch. |

### Decision Logic

```
TOPS_NATIVE=false ?
  └── Yes ──> Use original (accelerator) implementation.
  └── No (true) ──> Are TOPS_USE_NATIVE_MODULES and TOPS_USE_NATIVE_FUNCS both unset/empty?
                      ├── Yes ──> Globally enable: switch ALL functions to jax native.
                      └── No  ──> Switch only if the function name or full module path
                                  matches the filter list.
```

### Examples

```bash
# 1. Global: switch all functions to CPU
export TOPS_NATIVE=true

# 2. Switch only specific functions
export TOPS_NATIVE=true
export TOPS_USE_NATIVE_FUNCS=chunk_fwd_h,fused_recurrent_fwd

# 3. Switch all functions in specific modules
export TOPS_NATIVE=true
export TOPS_USE_NATIVE_MODULES=tops.ops.simple_gla

# 4. Combine module and function filters
export TOPS_NATIVE=true
export TOPS_USE_NATIVE_MODULES=tops.ops.simple_gla
export TOPS_USE_NATIVE_FUNCS=chunk_fwd_h

# 5. Disabled (default) - all functions use accelerator implementations
# (TOPS_NATIVE unset or set to false/0)
```

## Safety Guarantees

`switch_func` performs three checks before switching. If any fails, the original function is returned unchanged:

1. **Prefix check** - The function's module path must start with `tops.`. Non-tops modules are returned immediately.
2. **Existence check** - The jax native alternative function must exist in the corresponding `tops.cpu.*` module.
3. **Signature check** - The jax native function must have the exact same signature (parameter names, defaults, annotations) as the original.

```
# Warning examples:
# "alter func is None, import module path tops.cpu.ops.gla.chunk_h, switch func failed."
# "function signature between chunk_fwd_h and chunk_fwd_h is different, switch func failed."
```

## Module Path Convention

The decorator transforms module paths by replacing the `tops.` prefix with `tops.cpu.` (only when the module starts with `tops.`):

| Original module | CPU module |
|---|---|
| `tops.ops.gla.fused_recurrent` | `tops.cpu.ops.gla.fused_recurrent` |
| `tops.ops.simple_gla.chunk_h` | `tops.cpu.ops.simple_gla.chunk_h` |

CPU implementations must mirror the accelerator module structure under `tops/cpu/` and export functions with identical names and signatures.

## Adding a New Switchable Function

1. Implement the CPU version under `tops/cpu/` mirroring the original module path.
2. Ensure the function name and signature match exactly.
3. Add `@switch_func` to the accelerator function:

```python
# tops/ops/gla/fused_recurrent.py
from tops.switch import switch_func

@switch_func
def fused_recurrent_fwd(q, k, v, gk, gv, h0, ...):
    # Pallas kernel implementation
    ...
```

```python
# tops/cpu/ops/gla/fused_recurrent.py
def fused_recurrent_fwd(q, k, v, gk, gv, h0, ...):
    # Pure JAX CPU reference implementation
    ...
```

## Testing

Tests are located at `tests/test_switch.py` and cover:

- `same_signature` - function signature comparison
- `is_env_true` / `get_env_val` - environment variable parsing
- `use_native` - dispatch decision logic (global, filtered, disabled)
- `switch_func` - end-to-end decorator behavior (switching, fallback, signature mismatch, module path transform)

Run tests:

```bash
pytest tests/test_switch.py -v
```
