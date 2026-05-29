# Contributing to IDPR

Thank you for your interest in contributing to the IDPR project! This document provides guidelines for contributing code, documentation, and other improvements.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/IDPR-Intelligent-Drone-Positioning-and-Routing.git
   cd IDPR-Intelligent-Drone-Positioning-and-Routing
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

2. Run tests before making changes:
   ```bash
   pytest tests/
   ```

## Code Style

We follow PEP 8 style guidelines. Please ensure your code adheres to these standards:

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable and function names
- Add docstrings to all functions and classes

Format your code using Black:
```bash
black src/ --line-length 100
```

Check code quality with Flake8:
```bash
flake8 src/ --max-line-length 100
```

## Commit Messages

Write clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally after the first line

Example:
```
Add zone avoidance to drone environment

- Implement forbidden zone generation
- Add zone crossing penalty to reward function
- Update documentation with zone parameters

Fixes #42
```

## Pull Request Process

1. Update the README.md with any new features or changes
2. Update ARCHITECTURE.md if you modify the code structure
3. Ensure all tests pass:
   ```bash
   pytest tests/
   ```
4. Submit a pull request with a clear description of the changes

## Reporting Issues

When reporting bugs, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, CUDA version if applicable)
- Relevant error messages or logs

## Feature Requests

For feature requests, please:

- Describe the desired functionality
- Explain the use case and motivation
- Provide examples if applicable
- Discuss potential implementation approaches

## Testing

We encourage writing tests for new features:

```python
# tests/test_drone_env.py
import pytest
from src.drone_env import DroneEnv

def test_drone_env_initialization():
    env = DroneEnv()
    obs, info = env.reset()
    assert obs is not None
    assert env.num_drones > 0
```

Run tests with:
```bash
pytest tests/ -v
```

## Documentation

- Update docstrings for modified functions
- Add comments for complex logic
- Update relevant markdown files in the `docs/` directory
- Include examples for new features

## Performance Considerations

When contributing code:

- Avoid unnecessary computations in hot loops
- Use NumPy vectorization where possible
- Consider memory usage for large networks
- Profile code for performance-critical sections

## Questions?

Feel free to open an issue or discussion for questions about contributing.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to IDPR!
