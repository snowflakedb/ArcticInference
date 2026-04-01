# Version Compatibility Guide

This document describes the version compatibility between Arctic Inference and vLLM.

## Quick Reference

| Arctic Inference | vLLM (LLM Inference) | vLLM (Embedding) | PyTorch | Release Date |
|-----------------|---------------------|------------------|---------|--------------|
| 0.1.1 (latest)  | 0.11.0              | 0.9.2            | 2.8.0   | Dec 2025     |
| 0.1.0           | 0.10.1              | 0.9.2            | 2.8.0   | Nov 2025     |
| 0.0.9           | 0.9.2               | 0.9.2            | 2.5.1   | Oct 2025     |
| 0.0.8           | 0.9.0.1             | 0.9.0.1          | 2.5.1   | Sep 2025     |
| 0.0.7           | 0.8.4               | 0.8.4            | 2.4.0   | Aug 2025     |
| 0.0.6           | 0.8.4               | -                | 2.4.0   | Jul 2025     |
| 0.0.4           | 0.8.1               | -                | 2.4.0   | Jun 2025     |

## Installation

### Standard Installation (LLM Inference)

For the latest version with LLM inference support:

```bash
pip install arctic-inference[vllm]
```

This installs Arctic Inference with the compatible vLLM version automatically.

### Embedding Installation

For embedding inference (uses a different vLLM version):

```bash
pip install arctic-inference[embedding]
```

### Specific Version Installation

To install a specific Arctic Inference version:

```bash
# Install specific version
pip install arctic-inference[vllm]==0.1.1

# Or with embedding support
pip install arctic-inference[embedding]==0.1.1
```

## Version Detection

Arctic Inference automatically checks for vLLM version compatibility at startup. If there's a mismatch, you'll see an error like:

```
RuntimeError: Arctic Inference plugin requires vllm==0.11.0 but found vllm==0.10.0!
```

### Bypassing Version Check

If you need to bypass the version check (not recommended for production):

```bash
export ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1
```

## Feature Compatibility by Version

### Arctic Inference 0.1.x (vLLM 0.10.x - 0.11.x)

- Shift Parallelism
- Ulysses Sequence Parallelism
- SwiftKV
- Arctic Speculator (MLP/LSTM)
- Suffix Decoding
- Embedding Inference (via separate vLLM version)
- Dynasor (CoT Reasoning)

### Arctic Inference 0.0.9 (vLLM 0.9.2)

- Ulysses Sequence Parallelism
- SwiftKV
- Arctic Speculator
- Suffix Decoding
- Embedding Inference

### Arctic Inference 0.0.7-0.0.8 (vLLM 0.8.4 - 0.9.0.1)

- SwiftKV
- Arctic Speculator
- Basic Suffix Decoding

## Upgrade Guide

### Upgrading from 0.0.x to 0.1.x

1. **Update vLLM first** (if not using `pip install arctic-inference[vllm]`):
   ```bash
   pip install vllm==0.11.0
   ```

2. **Update Arctic Inference**:
   ```bash
   pip install arctic-inference[vllm]==0.1.1
   ```

3. **Note**: Shift Parallelism is a new feature in 0.1.x that provides additional performance benefits.

### Downgrading

If you need to downgrade to match an older vLLM version:

```bash
# For vLLM 0.9.2
pip install arctic-inference[vllm]==0.0.9

# For vLLM 0.8.4
pip install arctic-inference[vllm]==0.0.7
```

## Troubleshooting

### Version Mismatch Error

If you see a version mismatch error:

1. Check your installed vLLM version:
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

2. Check your Arctic Inference version:
   ```bash
   python -c "import arctic_inference; from importlib.metadata import version; print(version('arctic_inference'))"
   ```

3. Install the matching versions using the table above.

### Multiple vLLM Versions

If you need both LLM inference and embedding with different vLLM versions, use separate virtual environments:

```bash
# Environment for LLM inference
python -m venv llm_env
source llm_env/bin/activate
pip install arctic-inference[vllm]

# Environment for embedding
python -m venv embed_env
source embed_env/bin/activate
pip install arctic-inference[embedding]
```

## Related Links

- [Arctic Inference GitHub](https://github.com/snowflakedb/ArcticInference)
- [Arctic Inference PyPI](https://pypi.org/project/arctic-inference/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
