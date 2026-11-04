"""Sweep-level integration test: the scaling script's ``--factorized`` sweep
runs the Step 12 Kronecker executor end-to-end.

The sweep must route through the public executor entry
``gnn.execute.jax.kronecker_executor.execute_kronecker_factorized`` (not the
bare ``run_factorized_active_inference``), so every run lands the standard
executor artifacts — ``simulation_data/simulation_results.json`` carrying the
``jax_kronecker_factorized_v1`` schema plus ``kronecker_execution_summary.json``
— under ``<pipeline_output_dir>/pymdp_kronecker_pipeline/T<t>``, and the
manifest rows record those artifact paths together with ``joint_materialized``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACTOR_SIZES = [2, 2, 2, 2]
TIMESTEPS = 10

SCALING_SCRIPT = (
    PROJECT_ROOT / "scripts" / "experiments" / "run_pymdp_gnn_scaling_analysis.py"
)


def _load_module(name: str, relative_path: Path) -> Any:
    scripts_dir = str(relative_path.parent)
    need_cleanup = scripts_dir not in sys.path
    if need_cleanup:
        sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, relative_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if need_cleanup and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


from gnn.analysis.framework_extractors import extract_jax_data  # noqa: E402


class TestFactorizedSweepExecutor:
    """The factorized sweep produces executor artifacts Step 16 can consume."""

    @pytest.fixture(scope="class")
    @classmethod
    def sweep_manifest(cls, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        """Run the sweep once (factors 2,2,2,2 / T=10) and return the manifest."""
        scaling = _load_module("pymdp_scaling_sweep_executor", SCALING_SCRIPT)
        tmp = tmp_path_factory.mktemp("factorized_sweep")
        args = SimpleNamespace(
            factors=",".join(str(n) for n in FACTOR_SIZES),
            factor_timesteps=str(TIMESTEPS),
            factor_output_dir=str(tmp / "specs"),
            pipeline_output_dir=str(tmp / "output"),
            a_signal=0.85,
        )
        assert scaling._run_factorized_sweep(args) == 0
        return json.loads(
            (tmp / "output" / "pymdp_kronecker_scaling_manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_records_existing_artifact_paths(
        self, sweep_manifest: dict[str, Any]
    ) -> None:
        assert sweep_manifest["schema_version"] == "pymdp_kronecker_scaling_manifest_v1"
        assert sweep_manifest["factor_sizes"] == FACTOR_SIZES
        assert sweep_manifest["joint_state_space_size"] == 16
        row = sweep_manifest["runs"][0]
        assert row["timesteps"] == TIMESTEPS
        assert row["joint_state_space_size"] == 16
        assert row["joint_materialized"] is False
        assert row["validation_all_valid"] is True
        assert row["spec_file"].endswith("kronecker_N16_T10.md")

        # Artifact paths recorded in the manifest must exist on disk. Paths are
        # PROJECT_ROOT-relative; pathlib keeps absolute tmp paths intact when
        # joined onto PROJECT_ROOT.
        results_file = PROJECT_ROOT / row["simulation_results_file"]
        summary_file = PROJECT_ROOT / row["execution_summary_file"]
        assert results_file.is_file()
        assert summary_file.is_file()
        assert (PROJECT_ROOT / row["spec_file"]).is_file()

    def test_executor_artifacts_carry_kronecker_schema(
        self, sweep_manifest: dict[str, Any]
    ) -> None:
        row = sweep_manifest["runs"][0]
        results = json.loads(
            (PROJECT_ROOT / row["simulation_results_file"]).read_text(encoding="utf-8")
        )
        assert results["schema_version"] == "jax_kronecker_factorized_v1"
        assert results["success"] is True
        model_parameters = results["model_parameters"]
        assert model_parameters["joint_state_space_size"] == 16
        assert model_parameters["joint_materialized"] is False
        assert results["validation"]["all_valid"] is True

        summary = json.loads(
            (PROJECT_ROOT / row["execution_summary_file"]).read_text(encoding="utf-8")
        )
        assert summary["schema_version"] == "jax_kronecker_factorized_v1"
        assert summary["success"] is True
        assert summary["all_valid"] is True
        assert summary["joint_state_space_size"] == 16
        assert summary["joint_materialized"] is False

    def test_extract_jax_data_returns_per_factor_fields(
        self, sweep_manifest: dict[str, Any]
    ) -> None:
        row = sweep_manifest["runs"][0]
        payload = json.loads(
            (PROJECT_ROOT / row["simulation_results_file"]).read_text(encoding="utf-8")
        )
        extracted = extract_jax_data(payload)
        assert extracted["schema_version"] == "jax_kronecker_factorized_v1"
        assert extracted["num_factors"] == len(FACTOR_SIZES)
        assert extracted["num_timesteps"] == TIMESTEPS
        assert extracted["factors"] == ["factor0", "factor1", "factor2", "factor3"]
        for name in extracted["factors"]:
            assert len(extracted["beliefs_by_factor"][name]) == TIMESTEPS
            assert len(extracted["actions_by_factor"][name]) == TIMESTEPS
            assert extracted["efe_per_factor"][name]
            assert extracted["policy_by_factor"][name]
            for belief in extracted["beliefs_by_factor"][name]:
                assert abs(sum(belief) - 1.0) < 1e-6
        assert extracted["model_parameters"]["joint_state_space_size"] == 16
        assert extracted["model_parameters"]["joint_materialized"] is False
