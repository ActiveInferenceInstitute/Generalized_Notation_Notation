"""Tests for the executor's lazy runner-import registry.

Importing ``gnn.execute`` must not bloom heavy optional backends (jax, torch,
discopy, matplotlib, networkx, numpyro); runner modules load only when the
framework registry is consulted, and ``_runner_state`` is the single seam
tests patch to force a backend unavailable.
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

import gnn.execute.executor as executor_module
from gnn.execute.executor import GNNExecutor, _RunnerState, list_frameworks

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backend modules that must stay out of ``sys.modules`` until a framework
# registry lookup actually needs them.
_HEAVY_BACKEND_MODULES = (
    "jax",
    "discopy",
    "matplotlib",
    "networkx",
    "torch",
    "numpyro",
)

_COLD_IMPORT_SNIPPET = (
    "import sys; "
    "import gnn.execute; "
    "import gnn.execute.executor; "
    f"mods = {list(_HEAVY_BACKEND_MODULES)!r}; "
    'print(",".join(m for m in mods if m in sys.modules))'
)


def test_cold_import_blooms_no_heavy_backends() -> None:
    """Importing the executor in a fresh interpreter imports no heavy backend."""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _COLD_IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.stdout.strip() == ""


def test_runner_state_refreshes_standalone_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.utils.runtime_safety import framework_availability

    verdicts = iter((True, False))
    monkeypatch.setattr(
        framework_availability,
        "check_framework",
        lambda name: framework_availability.FrameworkStatus(name, next(verdicts)),
    )
    first = executor_module._runner_state("pymdp")
    second = executor_module._runner_state("pymdp")
    assert isinstance(first, _RunnerState)
    assert first.available and not second.available


def test_list_frameworks_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """``list_frameworks`` reports every registered backend in canonical order."""
    from gnn.utils.runtime_safety import framework_availability

    monkeypatch.setattr(
        framework_availability,
        "check_framework",
        lambda name: framework_availability.FrameworkStatus(name, True),
    )
    records = list_frameworks()
    assert len(records) == len(executor_module.FRAMEWORK_DIR_NAMES)
    for record in records:
        assert set(record) == {
            "framework",
            "result_key",
            "available",
            "operation",
            "readiness",
        }
        assert isinstance(record["available"], bool)
        assert record["readiness"]["available"] == record["available"]
    assert [record["framework"] for record in records] == list(
        executor_module.FRAMEWORK_DIR_NAMES
    )


def test_list_frameworks_loads_runners_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry use — not module import — is what pulls runner modules in."""
    from gnn.utils.runtime_safety import framework_availability

    monkeypatch.setattr(
        framework_availability,
        "check_framework",
        lambda name: framework_availability.FrameworkStatus(name, True),
    )
    executor_module.list_frameworks()
    assert "gnn.execute.discopy.discopy_executor" in sys.modules


def test_unavailable_runner_state_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """A patched unavailable ``_runner_state`` makes lean dispatch fail closed."""
    monkeypatch.setattr(
        executor_module,
        "_runner_state",
        lambda key: _RunnerState(available=False, runner=None),
    )
    result = GNNExecutor()._execute_lean_verification("model.md")
    assert result == {"success": False, "error": "fep_lean unavailable"}


def test_hardware_probe_reports_actual_devices_without_parent_import(
    tmp_path, monkeypatch
):
    import sys

    from gnn.execute.executor import get_available_hardware

    shim = tmp_path / "jax.py"
    shim.write_text(
        "from types import SimpleNamespace\ndef devices(): return [SimpleNamespace(platform='cpu')]\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    before = sys.modules.get("jax")
    assert get_available_hardware() == ["cpu"]
    assert sys.modules.get("jax") is before


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_named_julia_runner_requires_shared_readiness(monkeypatch, framework):
    from gnn.utils.runtime_safety import framework_availability

    seen = []

    def unavailable(name):
        seen.append(name)
        return framework_availability.FrameworkStatus(
            name, False, reason_code="missing_module"
        )

    monkeypatch.setattr(framework_availability, "check_framework", unavailable)
    state = executor_module._runner_state(framework)
    assert state.available is False
    assert seen == [framework]
