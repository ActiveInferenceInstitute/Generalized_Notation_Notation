"""Tests for the capability-contract ladder in scripts/check_capability_contracts.py.

Public functions: test_capability_contracts_are_current,
test_capability_contracts_fail_strict_by_default, test_v350_version_pair_contract_fires,
test_autonomy_claim_before_v4_fails, test_v350_executor_registry_contract_fires
"""

from __future__ import annotations

import re

import pytest

from scripts import check_capability_contracts
from scripts.check_capability_contracts import run_audit


def _patched_read(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> None:
    """Route _read through the real files, applying per-path text overrides."""
    real_read = check_capability_contracts._read

    def fake_read(path: str) -> str:
        text = real_read(path)
        return overrides.get(path, text)

    monkeypatch.setattr(check_capability_contracts, "_read", fake_read)


def test_capability_contracts_are_current() -> None:
    assert run_audit() == []


def test_v350_version_pair_contract_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 3.5.0 release without a v4.0.0 next-target must fail the ladder."""
    # This historical contract needs an explicit historical fixture. The live
    # roadmap need not retain these fields once the release program advances.
    todo_text = "**Current Version**: 3.5.0\n**Next Target**: v5.0.0\n"
    _patched_read(monkeypatch, {"TO-DO.md": todo_text})

    assert "TO-DO.md: v3.5.0 release must set v4.0.0 as next target" in run_audit()


def test_autonomy_claim_before_v4_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autonomy/self-editing wording in any pre-4.0.0 roadmap section fails."""
    todo_text = check_capability_contracts._read("TO-DO.md") + (
        "\n## Roadmap items for v3.6.0\n\n"
        "- [ ] Allow agents to begin self-editing their GNN files.\n"
    )
    _patched_read(monkeypatch, {"TO-DO.md": todo_text})

    failures = run_audit()
    assert any(
        "autonomy/self-editing claim appears before v4.0.0 in v3.6.0" in failure
        for failure in failures
    )


def test_v350_executor_registry_contract_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a backend from FRAMEWORK_DIR_NAMES must fail the v3.5.0 pin."""
    executor_text = check_capability_contracts._read(
        "src/gnn/execute/executor_specs.py"
    )
    match = re.search(
        r"FRAMEWORK_DIR_NAMES: tuple\[str, \.\.\.\] = \((?P<names>[^)]*)\)",
        executor_text,
    )
    assert match is not None
    names = re.findall(r'"([^"]+)"', match.group("names"))
    assert "stan" in names
    # Rebuild the declaration without the "stan" backend.
    remaining = tuple(name for name in names if name != "stan")
    rebuilt = "FRAMEWORK_DIR_NAMES: tuple[str, ...] = (\n"
    for name in remaining:
        rebuilt += f'    "{name}",\n'
    rebuilt += ")"
    narrowed = executor_text.replace(match.group(0), rebuilt, 1)
    _patched_read(monkeypatch, {"src/gnn/execute/executor_specs.py": narrowed})

    failures = run_audit()
    assert any(
        "FRAMEWORK_DIR_NAMES must match maintained registry capabilities" in failure
        for failure in failures
    )
    assert any(
        "FRAMEWORK_DIR_NAMES missing v3.5.0 backend stan" in failure
        for failure in failures
    )


def test_capability_contracts_fail_strict_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_capability_contracts, "run_audit", lambda: ["synthetic failure"]
    )

    assert check_capability_contracts.main([]) == 1
    assert check_capability_contracts.main(["--warn-only"]) == 0
