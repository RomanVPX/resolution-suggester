# Contributing to ResolutionSuggester

Thank you for your interest in contributing to the ResolutionSuggester project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/resolution-suggester.git
   cd resolution-suggester
   ```
3. Set up the development environment:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes, following the code style guidelines
2. Add tests that verify your changes
3. Ensure all tests pass:
   ```bash
   pytest
   ```
4. Verify code quality:
   ```bash
   pylint src/resolution_suggester
   mypy src/
   ```
5. Commit your changes with a clear, concise commit message
6. Push to your fork and submit a pull request to the `develop` branch

## Code Style

- Line length: 127 characters maximum
- Use snake_case for variables and functions, PascalCase for classes
- Always include type annotations
- Follow the existing code style in the codebase
- Use docstrings for public functions and classes

## Testing

- Write unit tests for all new functionality
- Test both normal cases and edge cases/error conditions
- Tests should be placed in the `tests/` directory, mirroring the structure of `src/`

## Internationalization

- Use the `_()` function from the i18n module for all user-facing strings
- When adding new strings, update the translation template:
  ```bash
  python scripts/update_translations.py
  ```

## Pull Request Guidelines

- Keep pull requests focused on a single topic
- Provide a clear description of what your PR addresses
- Reference any related issues in your PR description
- Make sure CI tests pass for your PR

## Questions?

If you have any questions about contributing, feel free to open an issue for discussion.

Thank you for contributing to ResolutionSuggester!