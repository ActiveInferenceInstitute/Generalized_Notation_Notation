"""Known Julia backends share bounded committed-project readiness."""

import logging
from pathlib import Path

import pytest

from gnn.utils.runtime_safety import framework_availability as availability


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_absent_julia_launcher_is_not_ready(framework, monkeypatch) -> None:
    from gnn.execute import julia_setup

    monkeypatch.setattr(julia_setup, "julia_executable", lambda: None)
    status = availability.check_framework(framework)
    assert not status.available and status.reason_code == "executor_unavailable"
    assert status.missing_module is None


def test_missing_committed_project_is_a_toolchain_gap(tmp_path, monkeypatch) -> None:
    from gnn.execute import julia_env

    monkeypatch.setattr(
        julia_env, "_julia_project_for_framework", lambda name: tmp_path
    )
    status = availability.check_framework("rxinfer", executor="/test/julia")
    assert not status.available and status.reason_code == "missing_toolchain"
    assert status.missing_module is None


def test_same_julia_interpreter_alias_reuses_invocation_probe(monkeypatch) -> None:
    from types import SimpleNamespace

    from gnn.execute import julia_setup
    from gnn.pipeline import run_context

    context = SimpleNamespace(
        run_id="julia-alias-run", bounded_timeout=lambda seconds: seconds
    )
    monkeypatch.setattr(run_context, "current_run_context", lambda: context)
    monkeypatch.setattr(julia_setup, "julia_executable", lambda: "/test/julia")
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return {
            "success": True,
            "return_code": 0,
            "stdout": 'GNN_FRAMEWORK_STATUS:{"available":true}',
        }

    monkeypatch.setattr(availability, "_run_probe_envelope", probe)
    assert availability.check_framework("rxinfer").available
    assert availability.check_framework("rxinfer", executor="julia").available
    assert len(calls) == 1 and calls[0][0] == "/test/julia"


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_missing_requested_julia_executor_has_typed_diagnosis(framework) -> None:
    status = availability.check_framework(framework, executor="/no/such/julia")
    assert status.reason_code == "executor_unavailable"
    assert status.execution_error_type == "FileNotFoundError"


@pytest.mark.needs_julia
@pytest.mark.parametrize("broken_load", [False, True])
def test_real_julia_distinguishes_missing_package_and_failed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken_load: bool
) -> None:
    from gnn.execute import julia_env

    name = "GNNProbeBrokenPackage" if broken_load else "GNNProbeAbsentPackage"
    (tmp_path / "Project.toml").write_text(
        f'name = "{name}"\nuuid = "a92a0f32-cf92-4077-8994-cdf93dce0f01"\nversion = "0.1.0"\n'
    )
    (tmp_path / "Manifest.toml").write_text('manifest_format = "2.0"\n')
    if broken_load:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / f"{name}.jl").write_text(
            f'module {name}\nthrow(ArgumentError("deliberate load failure"))\nend\n'
        )
    monkeypatch.setattr(
        julia_env, "_julia_project_for_framework", lambda framework: tmp_path
    )
    monkeypatch.setitem(availability.FRAMEWORK_JULIA_PACKAGES, "rxinfer", (name,))
    status = availability.check_framework("rxinfer")
    assert not status.available
    assert status.reason_code == ("probe_failed" if broken_load else "missing_module")
    assert status.missing_module == (None if broken_load else name)
    assert status.cleanup_verified is True and status.streams_drained is True


def test_doctor_planner_and_runtime_preserve_same_julia_cause(
    tmp_path, monkeypatch
) -> None:
    from gnn.execute import doctor, planning
    from gnn.execute import processor as facade
    from gnn.execute.processor import single

    status = availability.FrameworkStatus(
        "rxinfer",
        False,
        missing_module="RxInfer",
        reason_code="missing_module",
        reason="Dependency not installed in committed Julia project: RxInfer",
        runtime_version="1.12.7",
        cleanup_verified=True,
        streams_drained=True,
    )
    monkeypatch.setattr(doctor, "check_framework", lambda *args, **kwargs: status)
    monkeypatch.setattr(planning, "check_framework", lambda *args, **kwargs: status)
    monkeypatch.setattr(
        facade, "_check_framework_by_name", lambda *args, **kwargs: status
    )
    script = tmp_path / "11_render_output" / "model" / "rxinfer" / "model_rxinfer.jl"
    script.parent.mkdir(parents=True)
    script.write_text('error("This rendered model must never start")\n')
    entry = doctor._julia_framework_entry("rxinfer")
    assert entry["kind"] == "julia_project" and entry["available"] is False
    assert entry["reason_code"] == "missing_module"
    plan = planning.plan_execute(
        script.parents[2],
        tmp_path,
        frameworks="rxinfer",
        render_output_dir=script.parents[2],
    )
    assert plan["would_execute"] == [] and len(plan["would_skip_dependency"]) == 1
    assert plan["would_skip_dependency"][0]["reason"] == status.reason
    receipt = single.execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "rxinfer",
            "executor": "julia",
        },
        tmp_path / "execution",
        False,
        logging.getLogger(__name__),
    )
    assert receipt["skipped"] and receipt["attempts_started"] == 0
    assert receipt["reason_code"] == entry["reason_code"] == "missing_module"
    assert receipt["error"] == status.reason and receipt["cleanup_verified"] is True
