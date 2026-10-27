"""Bounds for generated GNN size estimates in scripts/experiments/run_pymdp_gnn_scaling_analysis.py."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import load_module_from_path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "run_pymdp_gnn_scaling_analysis.py"
)


@pytest.fixture(scope="module")
def scaling_mod() -> Any:
    # The generator lives in scripts/; the relocated runner anchors that
    # directory from the repository root. The helper adds experiments/ only
    # for the duration of this module load.
    return load_module_from_path("pymdp_scaling", _SCRIPT, sys_path=_SCRIPT.parent)


def test_estimate_upper_bounds_generated_size(scaling_mod: Any) -> None:
    for n, t in ((2, 10), (8, 100), (16, 500)):
        body = scaling_mod.generate_gnn_file(n, t, 0.85, 0.8)
        actual = len(body.encode("utf-8"))
        est = scaling_mod.estimate_gnn_file_bytes(n, t, 0.85, 0.8)
        assert est >= actual, (n, t, est, actual)


def test_skip_reason_matches_max_n(scaling_mod: Any) -> None:
    r = scaling_mod._skip_reason(
        512, 10, max_n=256, max_file_bytes=None, a_signal=0.85, b_signal=0.8
    )
    assert r == "max_n"


def test_skip_reason_timeout_pair(scaling_mod: Any) -> None:
    r = scaling_mod._skip_reason(
        16, 3000, max_n=1024, max_file_bytes=None, a_signal=0.85, b_signal=0.8
    )
    assert r == "timeout_bounds"


def test_relative_output_dir_resolves_from_project_root(scaling_mod: Any) -> None:
    out_dir = scaling_mod._resolve_output_dir("input/gnn_files/pymdp_scaling_study")

    project_root = Path(__file__).resolve().parents[2]
    assert scaling_mod.PROJECT_ROOT == project_root
    assert out_dir == project_root / "input/gnn_files/pymdp_scaling_study"


def test_pipeline_invocation_uses_project_root_python_and_frameworks(
    scaling_mod: Any, tmp_path: Any
) -> None:
    cmd, cwd = scaling_mod._build_pipeline_invocation(
        tmp_path,
        tmp_path / "pipeline_output",
        timeout=123,
        frameworks="pymdp,jax",
    )

    assert cwd == scaling_mod.PROJECT_ROOT
    assert cmd[:4] == ["uv", "run", "python", "src/gnn/main.py"]
    assert (cwd / cmd[3]).is_file()
    assert cmd[cmd.index("--target-dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path / "pipeline_output")
    assert cmd[cmd.index("--render-output-dir") + 1] == str(
        tmp_path / "pipeline_output" / "11_render_output"
    )
    assert cmd[cmd.index("--frameworks") + 1] == "pymdp,jax"
    assert "--strict-framework-success" in cmd


def test_pipeline_invocation_forwards_local_execution_workers(
    scaling_mod: Any, tmp_path: Any
) -> None:
    cmd, cwd = scaling_mod._build_pipeline_invocation(
        tmp_path,
        tmp_path / "pipeline_output",
        timeout=123,
        frameworks="pymdp",
        execution_workers=3,
    )

    assert cwd == scaling_mod.PROJECT_ROOT
    assert cmd[cmd.index("--execution-workers") + 1] == "3"
    assert "--distributed" not in cmd


def test_pipeline_invocation_forwards_distributed_backend(
    scaling_mod: Any, tmp_path: Any
) -> None:
    cmd, cwd = scaling_mod._build_pipeline_invocation(
        tmp_path,
        tmp_path / "pipeline_output",
        timeout=123,
        frameworks="pymdp",
        execution_workers=4,
        distributed=True,
        backend="dask",
    )

    assert cwd == scaling_mod.PROJECT_ROOT
    assert "--distributed" in cmd
    assert cmd[cmd.index("--backend") + 1] == "dask"
    assert cmd[cmd.index("--execution-workers") + 1] == "4"


def test_pipeline_invocations_split_integration_after_execution_by_default(
    scaling_mod: Any, tmp_path: Any
) -> None:
    phases = scaling_mod._build_pipeline_invocations(
        tmp_path,
        tmp_path / "pipeline_output",
        timeout=123,
        frameworks="pymdp",
        pipeline_steps="3,11,12,17",
    )

    assert [phase.label for phase in phases] == ["main", "integration"]
    assert phases[0].cmd[phases[0].cmd.index("--only-steps") + 1] == "3,11,12"
    assert phases[1].cmd[phases[1].cmd.index("--only-steps") + 1] == "17"
    assert all(phase.cwd == scaling_mod.PROJECT_ROOT for phase in phases)


def test_pipeline_invocations_can_keep_single_partial_pipeline_mode(
    scaling_mod: Any, tmp_path: Any
) -> None:
    phases = scaling_mod._build_pipeline_invocations(
        tmp_path,
        tmp_path / "pipeline_output",
        timeout=123,
        frameworks="pymdp",
        pipeline_steps="3,11,12,17",
        run_integration_on_failure=True,
    )

    assert [phase.label for phase in phases] == ["main"]
    assert phases[0].cmd[phases[0].cmd.index("--only-steps") + 1] == "3,11,12,17"


def test_run_manifest_records_config_and_planned_pairs(
    scaling_mod: Any, tmp_path: Any
) -> None:
    plan = scaling_mod.build_sweep_plan(
        [2, 4],
        [10],
        max_n=4,
        max_file_bytes=None,
        a_signal=0.85,
        b_signal=0.8,
    )

    manifest = scaling_mod._build_run_manifest(
        status="planned",
        plan=plan,
        out_dir=tmp_path / "specs",
        pipeline_output_dir=tmp_path / "pipeline",
        config={
            "n_values": [2, 4],
            "t_values": [10],
            "a_signal": 0.85,
            "b_signal": 0.8,
            "frameworks": "pymdp",
            "strict_framework_success": True,
            "execution_workers": 2,
            "distributed": False,
            "backend": "ray",
            "pipeline_steps": "3,11,12,17",
            "run_integration_on_failure": False,
        },
    )

    assert manifest["status"] == "planned"
    assert manifest["planned_pairs"] == [[2, 10], [4, 10]]
    assert manifest["configuration"]["execution_workers"] == 2
    assert manifest["configuration"]["distributed"] is False
    assert manifest["configuration"]["backend"] == "ray"
    assert manifest["configuration"]["pipeline_steps"] == "3,11,12,17"
    assert manifest["configuration"]["run_integration_on_failure"] is False


def test_resource_gate_write_returns_none_when_volume_full(
    scaling_mod: Any, monkeypatch: Any
) -> None:
    def fail_with_enospc(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(scaling_mod.Path, "write_text", fail_with_enospc)

    assert scaling_mod._write_resource_gate_file({"kind": "test"}) is None


def test_non_mapping_yaml_config_recovers_to_empty_dict(
    scaling_mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "pymdp_scaling_config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    monkeypatch.setattr(scaling_mod, "CONFIG_FILE", config_path)

    assert scaling_mod._load_and_validate_config() == {}


def test_usage_snapshot_returns_stable_disk_fields(
    scaling_mod: Any, tmp_path: Path
) -> None:
    snapshot = scaling_mod._usage_snapshot(tmp_path)

    assert snapshot["path"] == str(tmp_path.resolve())
    assert snapshot["total_bytes"] >= snapshot["free_bytes"]
    assert snapshot["used_bytes"] >= 0
    assert 0.0 <= snapshot["used_percent"] <= 100.0
