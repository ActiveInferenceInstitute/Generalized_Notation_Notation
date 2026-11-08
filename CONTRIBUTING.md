# Contributing to Generalized Notation Notation (GNN)

Thank you for your interest in contributing to GNN! This project benefits from contributions of all kinds — from bug reports and documentation improvements to new pipeline modules and framework integrations.

## Getting Started

### Prerequisites

- **Python 3.11–3.14** for the package. Use Python 3.14 for the default development profile.
- **uv 0.13.6** to reproduce CI's package-manager version. See the [setup guide](SETUP_GUIDE.md) for installation.
- **Git** for version control
- **just** for the recipe examples. The [manual commands](#ci-and-automation) use uv directly.

### Setup

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation

# Install dependencies
uv sync --frozen --extra dev --python 3.14

# Install pre-commit hooks
uv run --frozen --no-sync pre-commit install

# Run the fast development test selection
just test
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions including optional dependency groups.

## How to Contribute

### Reporting Bugs

1. Search [existing issues](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues) first.
2. If no existing issue matches, open a new one with:
   - A clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - Python version, OS, and relevant dependency versions

### Suggesting Features

Open a [GitHub Discussion](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions) or issue describing:
- The problem the feature would solve
- Your proposed approach
- Any relevant examples or references

### Submitting Code Changes

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the conventions below.
3. **Run the tests** to ensure nothing is broken:

   ```bash
   just test
   ```

4. **Commit** with clear, descriptive messages:

   ```bash
   git commit -m "Add support for new matrix type in GNN parser"
   ```

5. **Push** to your fork and **open a Pull Request** against `main`.

## Code Conventions

### Architecture

All pipeline modules follow the **thin orchestrator pattern**:
- **Numbered scripts** (`src/gnn/N_*.py`): Handle CLI args, logging, and delegation only (<150 lines)
- **Module directories** (`src/gnn/<module>/`): Contain all domain logic in `processor.py`, public API in `__init__.py`

### Code Standards

- **Type hints** for all public functions
- **Exit codes**: 0 = success, 1 = error, 2 = warnings
- Tests exercise real code paths and real data dependencies
- Follow existing patterns — read neighboring modules before writing new ones

### Module Structure

Every module must include:

```text
src/gnn/<module>/
  __init__.py      # Public API
  processor.py     # Core logic
  AGENTS.md        # Module documentation
  mcp.py           # MCP tool registration (if applicable)
```

### Testing

- Tests go in `tests/<module>/test_*.py` (one subdirectory per module, e.g. `tests/gnn/`, `tests/render/`)
- Run module-specific tests: `uv run pytest tests/gnn/ -v`
- Check coverage: `uv run pytest --cov=src/gnn --cov-report=term-missing`
- Aim for >80% test coverage on new code

### Documentation

- Update `AGENTS.md` in the relevant module directory
- If adding a pipeline step, update `AGENTS.md`, `DOCS.md`, and `README.md`
- Use concrete examples and real outputs over promotional language

### CI and automation

Pull requests against `main` run the primary CI and documentation audit even
when the diff contains only documentation. See the [workflow roster](.github/workflows/README.md)
for the separate dependency review, CodeQL, and workflow lint triggers.

Before opening a PR, align locally where possible:

```bash
# Using just (recommended)
just lint                  # Ruff lint check
just test                  # Fast test suite
just test-mod MODULE       # Test specific module

# Or manually
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync pytest -m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain" --tb=short -q
uv run --frozen --no-sync ruff check src/
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
```

For workflow YAML edits, run `actionlint .github/workflows/*.yml`. When your
change affects a pipeline, MCP, or native backend path, run the relevant
selection with its required runtime. The [testing guide](docs/testing/README.md)
lists the test profiles.

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include a description of what changed and why
- Reference any related issues (e.g., "Fixes #42")
- Ensure all tests pass before requesting review
- Update documentation if your change affects user-facing behavior

## Style Guide

Follow the project [Style Guide](docs/style_guide.md) for formatting and naming conventions. Key points:
- Python code follows PEP 8
- GNN files use Markdown with structured sections (see [GNN Syntax](docs/gnn/reference/gnn_syntax.md))
- Commit messages should be imperative ("Add feature" not "Added feature")

## Security

### Security considerations

If you discover a security vulnerability, please follow the reporting process in [SECURITY.md](SECURITY.md). Do **not** open a public issue for security vulnerabilities.

## Code of Conduct

Please review and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive experience for everyone.

## Recognition

Contributors are recognized in [release notes](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases), the [CHANGELOG](CHANGELOG.md), and the [contributors graph](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/graphs/contributors).

## Questions?

- Check the [documentation](docs/)
- Open a [GitHub Discussion](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)
- See [SUPPORT.md](SUPPORT.md) for additional help channels
