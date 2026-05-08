# Contributing to RNAFold-Net

Thank you for your interest in contributing to RNAFold-Net. This document provides guidelines for contributing code, reporting issues, and suggesting improvements.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/rnafold-net.git
   cd rnafold-net
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to verify setup:
   ```bash
   pytest tests/
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

## Testing

All new features must include tests. Run the test suite before submitting:

```bash
pytest tests/ -v --cov=rnafold_net
```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, focused commits
3. Add tests for new functionality
4. Update documentation if needed
5. Run the full test suite
6. Submit pull request with clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- Operating system
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error traceback

## Feature Requests

For feature suggestions, please describe:
- Use case and motivation
- Proposed API or interface
- Potential implementation approach
- Relevant research or prior art

## Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Welcome newcomers and help them learn
- Credit original authors and sources
