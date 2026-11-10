# GNN Configuration Guide

This guide describes the configuration surfaces that are implemented by the current
GNN package. The repository has two related configuration paths:

- `input/config.yaml` is the pipeline's project configuration file. `src/gnn/main.py`
  loads it automatically when it exists.
- Command-line arguments are parsed by `src/gnn/utils/arguments/arg_parsing.py`. Explicit CLI
  values take precedence over setup and test defaults read from `input/config.yaml`.

There is no supported project-root `config.yaml`, user-level `~/.gnn/config.yaml`,
profile loader, or generic `--config` override for the main pipeline.

## Quick start

```bash
# Use the checked-in configuration and run a focused path.
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5,11,12" \
  --verbose

# Skip selected steps.
uv run --frozen --no-sync python src/gnn/main.py --skip-steps "2,13" --verbose

# Skip the LLM step without editing the YAML file.
uv run --frozen --no-sync python src/gnn/main.py --skip-llm --verbose
```

For a single-file validation envelope, use the unified CLI:

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict --json
```

## `input/config.yaml`

The checked-in file is the only automatically discovered pipeline YAML file. Keep
paths relative to the repository root unless a module explicitly documents another
base directory.

### Pipeline and setup defaults

```yaml
pipeline:
  enabled: true
  steps: []             # empty means use the configured/default step plan
  skip_steps: []        # numeric step IDs, for example [13]
  fast_only: true       # Step 2 default
  # comprehensive: false # enable the full Step 2 suite when needed

setup:
  dev: true             # applies uv sync --frozen --extra dev --python 3.14 when Step 1 runs
  recreate_venv: false  # maps to --recreate-uv-env
  # install_all_extras: false

llm:
  model: "smollm2:135m-instruct-q4_K_S"
  timeout_seconds: null
  max_files: null
  prompt_timeout: 45
```

The same file also contains `uv`, `testing_matrix`, `io`, `logging`, `validation`,
`performance`, and `security` sections. Those sections are consumed by the modules
that own them; they are not a universal schema for every pipeline step. When adding a
new key, update the consuming module and its documentation together.

### Testing matrix

`testing_matrix` controls folder routing and global steps. The canonical step names
and order remain in `src/gnn/pipeline/step_registry.py`.

```yaml
testing_matrix:
  enabled: true
  global_steps:
    0_template: true
    1_setup: true
    2_tests: true
  default_steps: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  folders: {}
```

Step numbers are integers from 0 through 24. Step 13 (LLM) is normally controlled
by `pipeline.skip_steps` or the `--skip-llm` flag.

## Supported main-pipeline options

Run `uv run --frozen --no-sync python src/gnn/main.py --help` for the complete list. The most common
options are:

| Option | Purpose |
|---|---|
| `--target-dir PATH` | Directory containing GNN input files |
| `--output-dir PATH` | Base output directory |
| `--only-steps LIST` | Run only comma-separated step numbers |
| `--skip-steps LIST` | Skip comma-separated step numbers |
| `--skip-llm` | Add Step 13 to the skipped steps |
| `--frameworks VALUE` | `all`, `lite`, or a comma-separated framework list |
| `--strict` | Enable strict validation where supported |
| `--estimate-resources` | Enable resource estimation for applicable steps |
| `--dev` | Install development dependencies during Step 1 |
| `--install-all-extras` | Install all optional dependency groups during Step 1 |
| `--install-optional --optional-groups GROUPS` | Install selected optional groups during Step 1 |
| `--recreate-uv-env` | Recreate the UV-managed environment during Step 1 |
| `--verbose` | Enable verbose output |
| `--log-format human\|json` | Select pipeline log format |
| `--no-animations` | Disable Step 16 GridWorld GIF artifacts |

Lists passed to `--only-steps` and `--skip-steps` are comma-separated strings, for
example `"3,5,11,12"`.

## Step-specific examples

```bash
# Setup and optional dependencies.
uv run --frozen --no-sync python src/gnn/1_setup.py --dev --verbose
uv run --frozen --no-sync python src/gnn/1_setup.py --install-optional --optional-groups "audio,gui"
uv run --frozen --no-sync python src/gnn/1_setup.py --install-all-extras
uv run --frozen --no-sync python src/gnn/1_setup.py --recreate-uv-env --dev

# Type checking.
uv run --frozen --no-sync python src/gnn/5_type_checker.py \
  --target-dir input/gnn_files \
  --strict --estimate-resources --verbose

# Render selected backends.
uv run --frozen --no-sync python src/gnn/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks "pymdp,jax" \
  --strict-framework-success

# Execute rendered scripts. Use --render-output-dir to avoid stale artifacts.
uv run --frozen --no-sync python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --timeout 600
```

The Step 12 executor does not provide a dry-run flag. To inspect availability before
execution, use:

```bash
uv run --frozen --no-sync gnn health
```

## Environment variables

Environment variables are used by specific modules rather than by a generic config
interpolation layer. Common examples include:

- `OPENAI_API_KEY` for cloud LLM access.
- `OLLAMA_MODEL` and `OLLAMA_TEST_MODEL` for local Ollama model selection.
- `GNN_JAX_PLATFORM` to select the JAX device for PyMDP subprocesses.
- `GNN_ALLOW_UNSAFE_EXEC` and `GNN_SANDBOX` for the Step 12 execution safety gate.
- `GNN_TORCHLENS_CAPTURE` (`1`, `true`, `yes` or `on`) to have discrete PyTorch
  scripts trace their rollout with TorchLens during Step 12 (`torchlens`
  extra), and `GNN_TORCHLENS_ARTIFACTS` (`summary`, the default, or `full`)
  to also write the portable trace bundle and op table. Unset, nothing is
  captured; see the
  [PyTorch renderer README](../../src/gnn/render/pytorch/README.md#torchlens-capture-opt-in).

Do not put credentials in `input/config.yaml` or commit them. See the security
and LLM documentation for provider-specific behavior.

## Configuration validation

There is no `main.py --validate-config`, `--show-config`, `--profile`, or generic
dry-run command. To check YAML syntax and the top-level sections that the runtime
consumes, load the checked-in file directly:

```bash
uv run --frozen --no-sync python - <<'PY'
import yaml

config = yaml.safe_load(open("input/config.yaml"))
print("Configuration sections:", sorted(config))
print("pipeline:", config.get("pipeline"))
PY
```

The package loader in `gnn.utils.config_io.config_loader` additionally validates
section values; note that its relative-path checks currently resolve against
`src/` rather than the repository root, so run it only from a checkout where
that matches your layout.

For a non-mutating environment check, use:

```bash
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync gnn health
```

When documentation or code changes the configuration contract, update this page,
`input/config.yaml`, and the relevant tests in the same change.

See [supported configuration examples](examples.md) for complete task-oriented commands.

## Related references

- [Setup guide](../SETUP.md)
- [Pipeline guide](../pipeline/README.md)
- [GNN syntax reference](../gnn/reference/gnn_syntax.md)
- [Unified CLI](../../src/gnn/cli/README.md)
- [Configuration loader](../../src/gnn/utils/config_io/config_loader.py)
- [Main parser](../../src/gnn/utils/arguments/arg_parsing.py)
