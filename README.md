# tops

JAX/Flax Pallas kernels for Gated Linear Attention (GLA).

## Installation

Install directly from the repository:

```bash
pip install git+https://github.com/primatrix/pallas-kernel.git
```

Or build and install locally:

```bash
git clone https://github.com/primatrix/pallas-kernel.git
cd pallas-kernel
pip install .
```

### Optional dependencies

For GPU support (CUDA 12):

```bash
pip install "tops[gpu] @ git+https://github.com/primatrix/pallas-kernel.git"
```

For TPU support:

```bash
pip install "tops[tpu] @ git+https://github.com/primatrix/pallas-kernel.git"
```

For development:

```bash
pip install -e ".[dev]"
```

## Building packages

Use the provided build script to create distributable packages:

```bash
./scripts/build.sh        # Build sdist and wheel into dist/
./scripts/build.sh clean  # Remove build artifacts
```

## Usage

```python
from tops.ops.gla import chunk_gla, fused_recurrent_gla, fused_chunk_gla
from tops.layers.gla import GatedLinearAttention
from tops.modules.layernorm import RMSNorm
```

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) ([中文版](CONTRIBUTING.zh.md)) to get started.

## License

Apache License 2.0
