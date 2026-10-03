#!/usr/bin/env python3
"""Capability doctor (``execute.doctor``): report shape and composition.

All tests are offline — the runtime probes (``check_framework``,
``check_julia_availability``, ``plan_execute``) are monkeypatched, so no
framework runtime or Julia toolchain is required. The report's JSON
serializability (the MCP payload contract) is pinned directly.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute import doctor  # noqa: E402
from gnn.utils.runtime_safety.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK,
    FRAMEWORK_PROBE_STATEMENT,
    FrameworkStatus,
)


def _plan(status: str = "ready") -> Dict[str, Any]:
    """Return a minimal ExecutionPlan-shaped dict for patching plan_execute."""
    return {
        "requested_frameworks": ["pymdp"],
        "target_directory": "output",
        "output_directory": "output/12_execute_output",
        "render_output_dir": "output/11_render_output",
        "render_contract_found": False,
        "status": status,
        "total_scripts": 1 if status == "ready" else 0,
        "would_execute": [{"script_name": "runnable.py"}] if status == "ready" else [],
        "would_skip_dependency": [],
        "unknown_framework_scripts": [],
        "missing_render_scripts": [],
        "render_failures": [],
    }


def _patch_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool = True,
    julia: bool = True,
    plan: Dict[str, Any] | None = None,
) -> Dict[str, FrameworkStatus]:
    """Patch all runtime probes; return the statuses handed to the doctor."""
    statuses = {
        name: FrameworkStatus(
            name=name,
            available=available,
            missing_module=None if available else FRAMEWORK_IMPORT_CHECK[name][0],
            install_hint=None if available else FRAMEWORK_IMPORT_CHECK[name][1],
        )
        for name in FRAMEWORK_IMPORT_CHECK
    }
    for name in ("rxinfer", "activeinference_jl"):
        statuses[name] = FrameworkStatus(
            name=name,
            available=julia,
            reason_code=None if julia else "executor_unavailable",
            reason=None if julia else "Julia not found",
        )
    monkeypatch.setattr(doctor, "check_framework", lambda name, **_: statuses[name])
    monkeypatch.setattr(
        doctor,
        "check_julia_availability",
        lambda: (julia, "/usr/local/bin/julia" if julia else None),
    )
    if plan is not None:
        monkeypatch.setattr(
            doctor,
            "plan_execute",
            lambda *a, **k: plan,  # type: ignore[arg-type,return-value]
        )
    return statuses


@pytest.mark.parametrize("runnable", [False, True])
def test_discovered_plan_readiness_requires_a_runnable_script(
    monkeypatch, runnable
) -> None:
    plan = _plan("ready")
    plan["would_execute"] = [{"script_name": "runnable.py"}] if runnable else []
    plan["would_skip_dependency"] = [
        {"script_name": "missing.jl", "reason_code": "missing_module"}
    ]
    _patch_probes(monkeypatch, plan=plan)
    report = doctor.collect_doctor_report(
        target_dir="output", output_dir="output/12_execute_output"
    )
    assert report["execution"]["status"] == "ready"
    assert report["execution_ready"] is runnable


@pytest.mark.unit
class TestFrameworkSection:
    """Per-framework records mirror the canonical registry."""

    def test_every_registry_framework_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)
        report = doctor.collect_doctor_report()
        assert set(report["frameworks"]) == set(FRAMEWORK_IMPORT_CHECK) | {
            "rxinfer",
            "activeinference_jl",
        }

    def test_python_entry_shape_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch, available=True)
        entry = doctor.collect_doctor_report()["frameworks"]["jax"]
        assert entry["kind"] == "python_import"
        assert entry["probe_module"] == "jax"
        assert entry["available"] is True
        assert "missing_module" not in entry
        assert "install_hint" not in entry

    def test_unavailable_entry_carries_missing_module_and_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch, available=False)
        entry = doctor.collect_doctor_report()["frameworks"]["bnlearn"]
        assert entry["available"] is False
        assert entry["missing_module"] == "bnlearn"
        assert entry["install_hint"] == "uv sync --extra bnlearn"

    def test_stan_carries_the_toolchain_probe_statement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)
        frameworks = doctor.collect_doctor_report()["frameworks"]
        assert frameworks["stan"]["requires_toolchain_probe"] is True
        assert (
            frameworks["stan"]["toolchain_probe"] == FRAMEWORK_PROBE_STATEMENT["stan"]
        )
        assert frameworks["pymdp"]["requires_toolchain_probe"] is False
        assert "toolchain_probe" not in frameworks["pymdp"]

    def test_julia_frameworks_mirror_the_julia_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch, julia=False)
        report = doctor.collect_doctor_report()
        assert report["julia"] == {"available": False, "path": None}
        for name in ("rxinfer", "activeinference_jl"):
            entry = report["frameworks"][name]
            assert entry["kind"] == "julia_project"
            assert entry["available"] is False
        assert report["frameworks_missing"] == ["activeinference_jl", "rxinfer"]

    def test_availability_name_lists_partition_the_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)
        report = doctor.collect_doctor_report()
        names = set(report["frameworks"])
        assert (
            set(report["frameworks_available"]) | set(report["frameworks_missing"])
            == names
        )
        assert not set(report["frameworks_available"]) & set(
            report["frameworks_missing"]
        )


@pytest.mark.unit
class TestExecutionSection:
    """The readiness section forwards the directory pair to plan_execute."""

    def test_not_probed_when_no_directories(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)

        def _fail(*a: Any, **k: Any) -> Dict[str, Any]:
            raise AssertionError("plan_execute must not run without directories")

        monkeypatch.setattr(doctor, "plan_execute", _fail)
        report = doctor.collect_doctor_report()
        assert report["execution"] == {
            "status": "not_probed",
            "reason": "no target/output directories supplied",
        }
        assert report["execution_ready"] is None

    def test_directory_pair_requires_both_sides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)
        with pytest.raises(ValueError, match="supplied together"):
            doctor.collect_doctor_report(target_dir="output")
        with pytest.raises(ValueError, match="supplied together"):
            doctor.collect_doctor_report(output_dir="output/12_execute_output")

    def test_ready_plan_sets_execution_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: Dict[str, Any] = {}

        def _plan_capture(
            target_dir: Path, output_dir: Path, **kwargs: Any
        ) -> Dict[str, Any]:
            seen["args"] = (target_dir, output_dir)
            seen["kwargs"] = kwargs
            return _plan("ready")

        monkeypatch.setattr(
            doctor,
            "check_framework",
            lambda name, **_: FrameworkStatus(name=name, available=True),
        )
        monkeypatch.setattr(doctor, "check_julia_availability", lambda: (True, None))
        monkeypatch.setattr(doctor, "plan_execute", _plan_capture)
        report = doctor.collect_doctor_report(
            target_dir="output",
            output_dir="output/12_execute_output",
            frameworks="pymdp",
        )
        assert report["execution_ready"] is True
        assert report["execution"]["status"] == "ready"
        assert seen["args"] == (Path("output"), Path("output/12_execute_output"))
        assert seen["kwargs"] == {"frameworks": "pymdp"}

    def test_no_render_output_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch, plan=_plan("no_render_output"))
        report = doctor.collect_doctor_report(
            target_dir="output", output_dir="output/12"
        )
        assert report["execution_ready"] is False
        assert report["execution"]["status"] == "no_render_output"

    def test_invalid_frameworks_raises_the_planner_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_probes(monkeypatch)

        def _reject(*a: Any, **k: Any) -> Dict[str, Any]:
            raise ValueError("invalid frameworks argument")

        monkeypatch.setattr(doctor, "plan_execute", _reject)
        with pytest.raises(ValueError, match="invalid frameworks"):
            doctor.collect_doctor_report(target_dir="output", output_dir="output/12")


@pytest.mark.unit
class TestPayloadContract:
    """The report must survive a JSON round-trip (the MCP payload contract)."""

    def test_report_is_json_serializable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_probes(monkeypatch, available=False, julia=False, plan=_plan("ready"))
        report = doctor.collect_doctor_report(
            target_dir="output", output_dir="output/12"
        )
        restored = json.loads(json.dumps(report))
        assert restored == report


@pytest.mark.unit
class TestMCPExposure:
    """The thin envelope wrapper catches builder failures like every tool."""

    def test_success_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gnn.execute import mcp as execute_mcp

        monkeypatch.setattr(
            execute_mcp,
            "collect_doctor_report",
            lambda **_: {"success": True, "frameworks": {}, "execution_ready": None},
        )
        payload = execute_mcp.get_doctor_report_mcp()
        assert payload == {"success": True, "frameworks": {}, "execution_ready": None}

    def test_error_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gnn.execute import mcp as execute_mcp

        def _raise(**_: Any) -> Dict[str, Any]:
            raise ValueError("target_dir and output_dir must be supplied together")

        monkeypatch.setattr(execute_mcp, "collect_doctor_report", _raise)
        payload = execute_mcp.get_doctor_report_mcp(target_directory="output")
        assert payload["success"] is False
        assert "supplied together" in payload["error"]

    def test_tool_is_registered(self) -> None:
        from gnn.execute import mcp as execute_mcp

        registered: Dict[str, Any] = {}

        class _Registry:
            def register_tool(
                self, name: str, func: Any, schema: Any, description: str, **kw: Any
            ) -> None:
                registered[name] = {"schema": schema, "description": description, **kw}

        execute_mcp.register_tools(_Registry())
        assert "get_doctor_report" in registered
        assert registered["get_doctor_report"]["category"] == "execute"
        assert registered["get_doctor_report"]["schema"]["required"] == []
