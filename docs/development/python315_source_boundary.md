# Python 3.15 source-integration boundary

Installed-package support remains `>=3.11,<3.15` in
[pyproject.toml](../../pyproject.toml). Python 3.15 is excluded from ordinary
wheel installation until the complete wheel profiles and suite qualify it.
Passing a source check does not expand that range.

A downstream consumer can pin an exact GNN source revision and admit a named
module subset under CPython 3.15. Record that revision, interpreter, dependency
versions, selected modules and numerical witnesses. Admission belongs to that
source pair; it does not certify the GNN distribution or all optional extras.
NumPy-dependent numerical modules need their actual NumPy runtime and numerical
witnesses separately; syntax acceptance alone makes no numerical claim.

The deterministic regression is:

```bash
uv run --frozen --no-sync python -m pytest tests/setup/test_python315_source_boundary.py -q --no-cov
```

With `python3.15` on PATH, it parses and compiles every `src/gnn/**/*.py` file
(including numerical modules), then imports the step registry and shared type
definitions directly from source using `-I -B -S`. It checks registry behavior,
type behavior and that imports add only standard-library modules. It neither
installs GNN nor adds checkout paths to the interpreter search path. Without
that interpreter, the source probe explicitly skips; the installed-range
contract still runs. This is a lightweight source boundary, not full Python
3.15 package, NumPy, or numerical-runtime qualification.

The GEO/GNN paired workflow independently keeps its existing Python 3.11/3.12
matrix and frozen package selections. It reclaims unused hosted toolchains
before installing GEO, disables setup-uv archive caching, uses `uv sync
--no-cache` for both runtimes and prunes between installations. The workflow
contract prevents archive double-buffering from returning; actual hosted free
space and paired scientific acceptance require the hosted run.
