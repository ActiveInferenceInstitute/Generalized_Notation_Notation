# GNN Pipeline — Developer Command Reference
# See: AGENTS.md, pyproject.toml, .github/workflows/ci.yml
#
# Install: brew install just  (or: cargo install just)
# Usage:   just <recipe>       (or: just --list)

# Default recipe: show available commands
default:
    @just --list --unsorted

# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

# Run tests with the same marker filter as CI, so local and CI
# exercise the same surface; use test-stopfast or test-full for other scopes.
test:
    uv run pytest tests/ -q --tb=short -m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain"

# Run fast test suite, stop at first failure
test-stopfast:
    uv run pytest tests/ -q --tb=short -x -m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain"

# Run full test suite (excludes ollama + env-heavy + toolchain via markers)
test-full:
    uv run pytest tests/ -q --tb=no \
        -m "not ollama and not env_heavy and not toolchain"

# Run tests for a specific module (e.g., just test-mod render)
test-mod MODULE:
    uv run pytest tests/{{ MODULE }}/ -v

# Run tests with coverage report (same marker filter as the default suite:
# SC-44 replaced the per-file Ollama ignores with the `ollama` marker and
# excludes env-heavy uv-sync tests).
test-cov:
    uv run pytest tests/ -m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain" \
        --cov=gnn --cov-report=term-missing

# Run the extras-unlock tests (same files the ci.yml `extras` job gates on);
# needs the ml-ai/torch extras in the environment (uv sync --extra ml-ai --extra torch).
test-extras:
    uv run --extra dev --extra ml-ai --extra torch python -m pytest \
        tests/ml_integration/test_ml_integration_inference.py \
        tests/render/test_continuous_renderers.py \
        -q --tb=short

# ─────────────────────────────────────────────
# Linting & Formatting
# ─────────────────────────────────────────────

# Run ruff linter
lint:
    uv run ruff check src/gnn scripts

# Run ruff linter with auto-fix
lint-fix:
    uv run ruff check src/gnn scripts --fix

# Format code with ruff
format:
    uv run ruff format src scripts
    uv run ruff check src scripts --select I --fix

# Check formatting without modifying files
format-check:
    uv run ruff format --check src scripts

# Run mypy type checking
typecheck:
    uv run mypy src/gnn --show-error-codes

# Run bandit security scan (same thresholds as CI)
security:
    uv run bandit -r src/gnn -c pyproject.toml -q --severity-level medium --confidence-level medium

# Check import-linter boundary contracts (three contracts in pyproject.toml)
import-linter:
    uv run lint-imports

# Enforce the numbered thin-orchestrator line caps (150 hard + ratchet)
thin-orchestrators:
    uv run python scripts/check_thin_orchestrators.py

# Check argparse-registered flags against maintained-doc mentions (ratcheted)
flag-parity:
    uv run python scripts/check_flag_parity.py

# Check optional-dependency consumers and stale mypy overrides (ratcheted)
dep-hygiene:
    uv run python scripts/check_dep_hygiene.py

# Audit the validate_gnn* public surface against its manifest
validate-surface:
    uv run python scripts/audit_validate_surface.py

# Run MCP + skills resolvability health gate
skills-health:
    uv run --extra dev python scripts/check_mcp_skills_health.py --strict

# Run capability contract audit
capability:
    uv run python scripts/check_capability_contracts.py --strict

# Run manuscript token audit
tokens:
    uv run python scripts/check_manuscript_tokens.py --strict

# Fail when committed hydrated prose is stale for the token map (run the SC-22 ritual)
hydrated-prose:
    uv run python scripts/check_hydrated_prose.py

# Run the POMDP GridWorld pipeline end-to-end, then validate the output contract
# (regenerates the volatile run tree from input/gnn_files/pomdp_gridworld;
#  requires local Julia toolchains for the RxInfer/ActiveInference.jl proof)
gridworld:
    #!/usr/bin/env bash
    set -euo pipefail
    # Headless GR/Plots for the Julia scripts: without GKSwstype the GR
    # backend may spawn a Qt window (gksqt) and block the run.
    export GKSwstype=100
    status=0
    uv run python src/gnn/main.py --target-dir input/gnn_files/pomdp_gridworld --frameworks all --verbose || status=$?
    # Thin-orchestrator exit contract: 0 = success, 2 = success with warnings.
    if [[ $status -ne 0 && $status -ne 2 ]]; then
        exit "$status"
    fi
    # The checker is the hard gate: exit 0 pass / 1 contract violation.
    uv run python scripts/check_pomdp_gridworld_outputs.py output

# Run the v3 orchestration acceptance gate (same command as CI)
v3-acceptance:
    PYTHONPATH=src uv run --extra dev python scripts/run_v3_orchestration_acceptance.py --strict

# Assert the live MCP tool count meets the CI floor (MCP_TOOL_FLOOR lives in tests/mcp/test_mcp_audit.py)
mcp-count:
    PYTHONPATH=src uv run --extra dev python -c "from tests.mcp.test_mcp_audit import count_mcp_tools, MCP_TOOL_FLOOR; assert count_mcp_tools() >= MCP_TOOL_FLOOR, 'MCP tool count below CI floor'"

# Emit durable v3 run manifests for a completed run (e.g. just manifest output)
manifest OUT:
    uv run python scripts/emit_run_manifest.py {{ OUT }}

# Run Step 22 GUI processing (default headless artifacts; e.g. just gui --interactive --gui-types gui_1)
gui ARGS='':
    uv run gnn gui {{ ARGS }}

# Run maintained-doc terminology audit
doc-terms:
    uv run python scripts/check_maintained_doc_terms.py --strict

# Run documentation contract audit
doc-contracts:
    uv run python scripts/check_doc_contracts.py --strict

# Run GNN documentation pattern audit
doc-patterns:
    uv run python scripts/check_gnn_doc_patterns.py --strict

# Run fast quality gates without the full pytest suite
quality: format-check lint import-linter thin-orchestrators flag-parity dep-hygiene validate-surface terminology doc-terms audit doc-contracts doc-patterns typecheck security capability skills-health tokens hydrated-prose v3-acceptance mcp-count

# Run focused PyMDP/POMDP behavior checks
test-pymdp-focused:
    uv run pytest \
        tests/execute/test_pymdp_contracts.py \
        tests/execute/test_discrete_models_pymdp.py \
        tests/visualization/test_visualization_matrices.py \
        -q --tb=short

# Collect pytest inventory without executing tests
test-collect:
    uv run pytest --collect-only tests/ -q --tb=no \
        -m "not ollama"

# ─────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────

# Run full pipeline
pipeline:
    uv run python src/gnn/main.py --target-dir input/gnn_files --verbose

# Run specific pipeline steps (e.g., just pipeline-steps "3,5,7,8")
pipeline-steps STEPS:
    uv run python src/gnn/main.py --only-steps "{{ STEPS }}" --target-dir input/gnn_files --verbose

# Run a single pipeline step (e.g., just step 3)
step N:
    uv run python src/gnn/{{ N }}_*.py --target-dir input/gnn_files --output-dir output --verbose

# ─────────────────────────────────────────────
# Renderer Operations
# ─────────────────────────────────────────────

# Check renderer availability
render-health:
    PYTHONPATH=src uv run python -c "from gnn.render.health import check_renderers; \
        statuses = check_renderers(); \
        [print(f'  {\"✅\" if s.available else \"❌\"} {s.name}') for s in statuses.values()]"

# Render and execute for specific frameworks (e.g., just render-exec "pymdp,jax")
render-exec FRAMEWORKS:
    uv run python src/gnn/main.py --only-steps "11,12" \
        --frameworks "{{ FRAMEWORKS }}" \
        --target-dir input/gnn_files --verbose

# ─────────────────────────────────────────────
# Documentation & Audit
# ─────────────────────────────────────────────

# Run documentation audit without mutating reports
audit:
    uv run python docs/development/docs_audit.py --strict --check-anchors --no-write

# Run maintained-tree terminology audit
terminology:
    uv run python scripts/check_repo_terminology.py --strict


# Count test files and items
test-count:
    @echo "Test files:"
    @find tests -name 'test_*.py' | wc -l
    @echo "Collected test items:"
    @uv run pytest --collect-only tests/ -q --tb=no \
        -m "not ollama" 2>/dev/null | tail -1

# ─────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────

# Install all dependencies (including dev extras)
setup:
    uv sync --extra dev

# Recreate the virtual environment from scratch
setup-clean:
    rm -rf .venv
    uv sync --extra dev

# Validate the JAX + PyMDP stack
validate-stack:
    PYTHONPATH=src uv run python -c "from gnn.utils.runtime_safety.jax_stack_validation import verify_jax_pymdp_stack; \
        verify_jax_pymdp_stack(); print('✅ JAX + PyMDP stack OK')"

# ─────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────

# Run performance benchmark tests (pipeline performance group)
bench:
    uv run pytest tests/pipeline/test_pipeline_performance.py \
        -v --tb=short \
        -m "performance" \
        --benchmark-save=baseline

# Run performance tests and compare against saved baseline
bench-compare:
    uv run pytest tests/pipeline/test_pipeline_performance.py \
        -v --tb=short \
        -m "performance" \
        --benchmark-compare=baseline

# ─────────────────────────────────────────────
# Step Registry
# ─────────────────────────────────────────────

# List all 25 pipeline steps from the canonical registry
steps:
    #!/usr/bin/env bash
    export PYTHONPATH=src
    uv run python - <<'PYEOF'
    from gnn.pipeline.step_registry import STEPS
    for s in STEPS:
        tags = ','.join(sorted(s.tags))
        print(f'{s.script_name:30s} {tags}')
    print()
    print(f'Total: {len(STEPS)} steps')
    print(f"Core:  {len([s for s in STEPS if 'core' in s.tags])}")
    print(f"LLM:   {len([s for s in STEPS if 'llm' in s.tags])}")
    PYEOF

# Export step registry as JSON (useful for tooling)
steps-json:
    #!/usr/bin/env bash
    export PYTHONPATH=src
    uv run python - <<'PYEOF'
    import json
    from gnn.pipeline.step_registry import STEPS
    data = [{'script_name': s.script_name, 'description': s.description,
             'module_function': s.module_function, 'tags': sorted(s.tags)}
            for s in STEPS]
    print(json.dumps(data, indent=2))
    PYEOF
