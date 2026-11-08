# GNN Deployment Guide

GNN is primarily a local, repository-root pipeline. Deployment should preserve that
boundary: run the pipeline in an isolated environment, keep model inputs read-only
when possible, write artifacts to a dedicated output directory, and provide secrets
through the environment rather than YAML.

## Security

Deployment operators are responsible for host hardening, authentication, secret
management, network boundaries, and review of rendered-script execution.

## Local or batch deployment

```bash
uv sync --frozen --python 3.14
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync gnn health

uv run --frozen --no-sync gnn run \
  --target-dir input/gnn_files \
  --output-dir output/run-$(date +%Y%m%d-%H%M%S) \
  --only-steps 3 5 11 12 16 23
```

The pipeline loads `input/config.yaml` automatically. It does not load
`config.production.yaml`, `config.security.yaml`, arbitrary `--config` overrides, or
profile files through `src/gnn/main.py`. See [Configuration](../configuration/README.md).

## Container deployment

The repository Dockerfile builds a test image that requires separate runtime
qualification. This example runs a small parsing and visualization workflow.
Create the host output directory and ensure the container user (UID 1000)
can write to it before starting the container.

```bash
docker build -t gnn:local .
mkdir -p output
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,mode=1777 \
  --env XDG_CACHE_HOME=/tmp/.cache \
  -v "$PWD/input/gnn_files:/workspace/input/gnn_files:ro" \
  -v "$PWD/output:/workspace/output" \
  gnn:local \
  gnn run --target-dir input/gnn_files/basics --output-dir output/container-example \
  --only-steps 3 5 8 --skip-steps 0 1 2
```

Do not expose the MCP/API surface publicly without adding authentication, network
segmentation, rate limiting, and secret management. The repository's local tools are
not an internet-facing production service by default.

## API surface

The optional FastAPI service is owned by the API module. Inspect its live help and
module documentation before deployment rather than assuming a generic health or
configuration endpoint:

```bash
uv sync --frozen --extra api --python 3.12
uv run --frozen --no-sync --python 3.12 gnn serve --help
```

Bind services to a private interface and place them behind the deployment's own
authentication and TLS controls.

## Julia runtimes

If a deployment executes Julia renderings, instantiate both committed environments as
needed and pass the matching project through the executor:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'
```

## Secrets and filesystem policy

- Set `OPENAI_API_KEY`, Ollama variables, and other provider credentials in the
  deployment secret store or environment.
- Never place credentials in `input/config.yaml`, container images, or tracked files.
- Mount inputs read-only where possible.
- Keep generated artifacts in a run-specific output directory.
- Treat rendered scripts as executable code and retain the Step 12 safety gate unless
  a reviewed local exception is required.

## Verification

Before promoting a deployment configuration:

```bash
uv run gnn preflight
uv run gnn health
uv run python src/gnn/main.py --help
uv run --extra dev python -m pytest tests/ -q \
  --ignore=tests/llm/test_llm_ollama.py \
  --ignore=tests/llm/test_llm_ollama_integration.py
```

Run a small representative model first and inspect the generated summary before
processing a large corpus. For backup and recovery of manifests and outputs, retain
both the exact input revision and the run summary; generated output is not source.

## Related references

- [Setup](../SETUP.md)
- [Configuration](../configuration/README.md)
- [Security](../security/README.md)
- [Pipeline](../pipeline/README.md)
- [Troubleshooting](../troubleshooting/README.md)
