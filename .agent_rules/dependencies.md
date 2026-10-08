# Dependencies and readiness

[pyproject.toml](../pyproject.toml) declares core packages, extras, development
groups and Python constraints; [uv.lock](../uv.lock) records resolved splits.
Derive inventories from these files. PyMDP, JAX, Flax, DisCoPy and the declared
scientific stack are core dependencies; extras are the actual named groups.

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --no-sync gnn health
```

Read the reported diagnoses. Exit 2 reports degraded readiness; importable
generators alone do not establish native backend execution.

Install a requested extra explicitly, such as `uv sync --frozen --extra torch`
for PyTorch. Julia and CmdStan require native toolchain provisioning; Julia
adapters use committed projects. See [setup](../SETUP_GUIDE.md). Installing an
extra does not select an experimental backend.

Use [framework metadata](../src/gnn/frameworks.py),
[renderer registry](../src/gnn/render/framework_registry.py) and
[readiness](../src/gnn/utils/runtime_safety/framework_availability.py).
Keep missing distributions, unsupported runtime/version, broken installed
imports, missing toolchains, timed-out probes and missing executors distinct.
An import alone cannot establish model-family or numerical acceptance.

Do not repair locks with floating side installations, guessed disable flags
or duplicate declarations. For justified changes, resolve normally and verify
each affected split, installed/native evidence and tracked-file bookends.
New platform support follows ordinary installation and genuine acceptance in
the [installed-platform guide](../docs/development/installed_platform_acceptance.md).
