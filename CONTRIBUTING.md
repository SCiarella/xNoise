# Contributing to DDPM-CLIP

Thank you for your interest in contributing to DDPM-CLIP! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the [Issues](https://github.com/SCiarella/xNoise/issues) section
2. If not, create a new issue with a clear title and description
3. Include:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU/CPU)
   - Relevant code snippets or error messages

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions and classes
   - Update documentation if needed

3. **Test your changes**
   - Ensure existing functionality still works
   - Add tests for new features if applicable

4. **Commit your changes**
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Wait for review and address any feedback

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Keep functions focused on a single task
- Add type hints where helpful

### Documentation

- Use NumPy-style docstrings for functions and classes
- Update README.md if adding new features
- Include code examples in docstrings when helpful

Example docstring format:
```python
def example_function(param1, param2):
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
        
    Returns
    -------
    return_type
        Description of return value.
    """
    pass
```

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xNoise.git
   cd xNoise
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   
   This installs the package in editable mode along with development tools like pytest, black, flake8, and isort.

## Areas for Contribution

We welcome contributions in these areas:

- **Bug fixes**: Help us squash bugs!
- **Documentation**: Improve README, add tutorials, fix typos
- **Performance**: Optimize code, reduce memory usage
- **Features**: New architectures, sampling methods, datasets
- **Tests**: Add unit tests and integration tests
- **Examples**: Jupyter notebooks demonstrating features

## Questions?

Feel free to open an issue with the label "question" if you have any questions about contributing.

## Code of Conduct

- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DDPM-CLIP! 🎉
