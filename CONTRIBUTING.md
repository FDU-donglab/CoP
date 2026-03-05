# Contributing to Noise Genome Estimator

We appreciate your interest in contributing! Here are some guidelines to help you get started.

## Getting Started

1. Fork the repository
2. Create a new branch for your feature or bugfix (`feature/your-feature` or `fix/your-bugfix`)
3. Install the package in editable mode: `pip install -e .`
4. Make your changes
5. Verify your changes by running a quick training dry-run (e.g., 1–2 epochs on a small dataset)
6. Submit a pull request with a clear description of what was changed and why

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Commit Messages

Use clear, descriptive commit messages:
- Use the imperative mood ("add feature" not "adds feature")
- Limit the first line to 72 characters
- Reference issues when applicable (fixes #123)

## Testing

There is currently no automated test suite. Before submitting a PR, please manually verify your changes by running a short training or evaluation pass to confirm nothing is broken:

```bash
python train.py --mode train \
    --train-dataset-path ./datasets/train \
    --validation-dataset-path ./datasets/val \
    --num-epochs 2 \
    --batch-size 4
```

## Documentation

- Update README.md if adding new features
- Add docstrings to code
- Update setup.py if adding new dependencies

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Any error messages or stack traces

## Questions?

Feel free to open an issue or contact the maintainers.

Thank you for contributing!
