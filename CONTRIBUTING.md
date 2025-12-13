# Contributing to Smart Palletizer

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to Smart Palletizer. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed and what you expected to see
* Include screenshots if applicable
* Include your environment details (OS, Python version, library versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Create an issue and provide the following:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain the expected behavior
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Follow the Python style guidelines (PEP 8)
* Include docstrings for new functions/classes
* Update documentation if needed
* Add tests for new features
* Ensure the test suite passes
* Make sure your code lints

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/smart_palletizer.git
cd smart_palletizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Style Guidelines

### Python Style Guide

* Follow [PEP 8](https://peps.python.org/pep-0008/)
* Use 4 spaces for indentation
* Maximum line length of 100 characters
* Use descriptive variable names
* Add type hints where appropriate
* Write docstrings for all public functions/classes

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐛 `:bug:` when fixing a bug
    * ✨ `:sparkles:` when adding a new feature
    * 📝 `:memo:` when writing docs
    * 🚀 `:rocket:` when improving performance
    * ✅ `:white_check_mark:` when adding tests

### Documentation

* Use clear and simple language
* Include code examples where applicable
* Keep documentation up to date with code changes
* Add comments for complex algorithms

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=smart_palletizier

# Lint code
flake8 src/
black --check src/
```

## Project Structure

```
smart_palletizer/
├── src/smart_palletizier/  # Main source code
├── examples/               # Usage examples
├── data/                   # Test data
├── docs/                   # Documentation
└── tests/                  # Unit tests (to be added)
```

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
