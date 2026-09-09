# Contributing to Snowflake MLOps Template

Thank you for your interest in contributing! This project follows the standard Snowflake open-source contribution workflow.

## How to Contribute

1. **Fork the repository** and create a feature branch from `main`
2. **Make your changes** following the existing code style
3. **Run checks locally** before submitting:
   ```bash
   uv sync
   uv run ruff check source/ scripts/
   uv run ruff format --check source/ scripts/
   uv run pytest tests/ --ignore=tests/test_endpoint.py
   ```
4. **Submit a Pull Request** targeting the `main` branch

## Code Style

- Python 3.12+
- Formatted with [ruff](https://docs.astral.sh/ruff/) (line length 120)
- No trailing whitespace, no unused imports

## Pull Request Requirements

- All status checks must pass (Code Quality: lint + format + tests)
- At least 1 approving review from a code owner
- Clear description of what changed and why

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
