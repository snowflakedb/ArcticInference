# Contributing to Arctic Inference

Thank you for your interest in contributing to Arctic Inference! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for full functionality)
- PyTorch 2.8.0+
- vLLM 0.9.1+ or 0.11.0+ (depending on feature)

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/snowflakedb/ArcticInference.git
   cd ArcticInference
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below.

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Run linting:**
   ```bash
   ruff check arctic_inference/
   ruff format --check arctic_inference/
   ```

5. **Commit your changes:**
   ```bash
   git commit -m "Brief description of changes"
   ```

### Coding Standards

- **Code Style:** We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- **Type Hints:** Use type hints for all public functions and methods
- **Docstrings:** Use Google-style docstrings for public APIs
- **Testing:** Add tests for new functionality

#### Exception Handling

- Avoid bare `except:` clauses; always specify exception types
- Use specific exceptions (e.g., `ValueError`, `TypeError`) rather than generic `Exception` where possible
- Never use `assert` for error handling in production code; use proper exceptions

#### Logging

- Use the `logging` module instead of `print()` statements
- Use appropriate log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

### Testing

- Unit tests go in `tests/unit_tests/`
- Benchmark tests go in `tests/benchmarks/`
- Run tests with: `pytest tests/ -v`

### Pull Request Process

1. **Ensure your PR:**
   - Has a clear, descriptive title
   - Includes a description of changes and motivation
   - References any related issues (e.g., "Fixes #123")
   - Passes all CI checks

2. **PR Title Format:**
   ```
   [Component] Brief description
   ```
   Examples:
   - `[SwiftKV] Fix attention mask handling for long sequences`
   - `[Ulysses] Add support for non-divisible batch sizes`
   - `[Docs] Update installation instructions`

3. **Review Process:**
   - At least one maintainer approval is required
   - Address all review comments
   - Keep the PR focused; split large changes into smaller PRs

## Project Structure

```
arctic_inference/
├── vllm/                  # vLLM integration and patches
│   ├── swiftkv/           # SwiftKV implementation
│   ├── spec_dec/          # Speculative decoding
│   └── ...
├── suffix_decoding/       # Suffix decoding cache
├── embedding/             # Embedding service (gRPC)
├── dynasor/               # CoT reasoning module
└── common/                # Shared utilities
```

## Reporting Issues

When reporting issues, please include:

1. **Environment information:**
   - Python version
   - PyTorch version
   - vLLM version
   - GPU model and CUDA version
   - Arctic Inference version

2. **Steps to reproduce** the issue

3. **Expected vs. actual behavior**

4. **Error messages and stack traces** (if applicable)

## License

By contributing to Arctic Inference, you agree that your contributions will be licensed under the Apache License 2.0.
