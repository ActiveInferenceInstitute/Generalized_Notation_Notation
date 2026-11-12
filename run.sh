#!/usr/bin/env bash
# run.sh — front door for GeneralizedNotationNotation (GNN).
#
# Modes:
#   (no arg)   alias for `gates`
#   gates      the canonical gate: ruff format --check, ruff check, mypy,
#              pytest under the coverage floor, docs/terminology/
#              doc-contract audits, the capability-contract audit, the MCP
#              functional surface plus a full-tree collection check, and the
#              v3 orchestration acceptance gate. Mirrors the ci.yml `quality`
#              job plus the unit-suite selection and coverage floor from `test`.
#   sync       uv sync --frozen --extra dev
#   lint       ruff format --check + ruff check (src, scripts, tests, docs)
#   type       mypy src/gnn and the release-gate script
#   test       pytest with the repository's default selection
#   coverage   pytest with the coverage report and the configured floor
#   docs       documentation, terminology and capability-contract audits
#   acceptance v3 orchestration acceptance gate
#   help       this text
#
# The coverage floor lives once, in pyproject.toml [tool.coverage.report]
# fail_under (coverage.py ignores the key under [tool.coverage.run]).
set -euo pipefail

cd "$(dirname "$0")"

# Match the `quality` job env in .github/workflows/ci.yml so a local gate run
# and the CI gate are the same evidence class: a fixed hash seed keeps dict and
# set iteration order stable, which JSON outputs and test order can observe,
# and `src` on the path matches the editable-package layout CI imports from.
export PYTHONHASHSEED=0
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

need_uv() {
    command -v uv >/dev/null || {
        echo "uv required (https://docs.astral.sh/uv/)" >&2
        exit 127
    }
}

# One frozen install, then `--no-sync` for every step: re-resolving per step is
# how a gate silently drifts from the lock the CI lane installed.
sync_env() {
    need_uv
    uv sync --frozen --extra dev
}

# Selection CI uses for the parallel suite: runtime-gate integration groups
# (pipeline contracts, MCP module discovery) stay out of the bounded gate, along
# with the Ollama daemon, venv-mutating and external-toolchain groups.
PYTEST_SELECT=(-m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain")

# pytest-xdist ships in the dev group; the worksteal distribution is what CI uses.
PYTEST_PARALLEL=(-n auto --dist worksteal)

gate_lint() {
    sync_env
    uv run --frozen --no-sync ruff format --check src scripts tests docs
    uv run --frozen --no-sync ruff check src/gnn scripts tests docs
}

gate_type() {
    sync_env
    uv run --frozen --no-sync mypy src/gnn .github/release_gate.py --show-error-codes
}

gate_test() {
    sync_env
    uv run --frozen --no-sync python -m pytest tests/ "${PYTEST_SELECT[@]}" \
        "${PYTEST_PARALLEL[@]}" -q --tb=short
}

gate_coverage() {
    sync_env
    uv run --frozen --no-sync python -m pytest tests/ "${PYTEST_SELECT[@]}" \
        "${PYTEST_PARALLEL[@]}" -q --tb=short \
        --cov=gnn --cov-report=term-missing
}

gate_docs() {
    sync_env
    uv run --frozen --no-sync python docs/development/docs_audit.py \
        --strict --check-anchors --no-write
    uv run --frozen --no-sync python scripts/check_repo_terminology.py --strict
    uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
    uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
    uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
    uv run --frozen --no-sync python scripts/check_capability_contracts.py --strict
}

gate_contracts() {
    sync_env
    # Same files the ci.yml `quality` job's "Functional MCP and capability PR
    # gate" step runs; `mcp` is marker-excluded from the unit selection above,
    # so these are the only place the MCP surface is exercised in the gate.
    uv run --frozen --no-sync python -m pytest \
        tests/mcp/test_mcp_functional.py \
        tests/mcp/test_mcp_http_auth.py \
        tests/mcp/test_transport_reliability.py \
        tests/render/test_framework_availability.py \
        tests/docs/test_capability_contracts.py \
        -q --tb=short -rs
    # Collection must stay importable across the whole tree (the `quality`
    # job's collect-only step); a collection error fails here, not in CI.
    uv run --frozen --no-sync python -m pytest --collect-only tests/ -q --tb=no \
        -m "not ollama" >/dev/null
}

gate_acceptance() {
    sync_env
    uv run --frozen --no-sync python scripts/run_v3_orchestration_acceptance.py --strict
}

gate_all() {
    echo "== gate: ruff format --check + ruff check =="
    gate_lint
    echo "== gate: mypy =="
    gate_type
    echo "== gate: pytest + coverage floor =="
    gate_coverage
    echo "== gate: docs / terminology / capability contracts =="
    gate_docs
    echo "== gate: MCP functional surface + collection =="
    gate_contracts
    echo "== gate: v3 orchestration acceptance =="
    gate_acceptance
    echo "== gate: green =="
}

case "${1:-gates}" in
  gates)      gate_all ;;
  sync)       sync_env ;;
  lint)       gate_lint ;;
  type)       gate_type ;;
  test)       gate_test ;;
  coverage)   gate_coverage ;;
  docs)       gate_docs ;;
  acceptance) gate_acceptance ;;
  help|-h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' ;;
  *) echo "Usage: $0 {gates|sync|lint|type|test|coverage|docs|acceptance|help}" >&2; exit 2 ;;
esac
