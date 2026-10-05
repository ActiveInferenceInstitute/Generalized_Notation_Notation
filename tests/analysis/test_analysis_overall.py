"""Test suite for Analysis module.

Public classes: TestAnalysisOverall, TestPostSimulationVisualization, TestAnalysisModuleImports, TestActiveInferenceJLAnalyzer, TestDisCoPyAnalyzer, TestJAXAnalyzer, TestPyMDPAnalyzer, TestRxInferAnalyzer, TestAnalyzerSimulationMetrics

Public functions: _gridworld_payload, test_gridworld_animation_suite_and_manifest
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.processor import process_analysis


def _gridworld_payload(schema_version: str, framework: str) -> dict[str, Any]:
    states = [0, 1, 2, 5, 8]
    beliefs = [
        [1.0 if index == state else 0.0 for index in range(9)] for state in states
    ]
    return {
        "schema_version": schema_version,
        "success": True,
        "framework": framework,
        "num_timesteps": len(states),
        "observations": states,
        "observations_by_modality": {"joint_observation": states},
        "hidden_states_by_factor": {"joint_state": states},
        "actions": [3, 3, 1, 1, 4],
        "actions_by_control_factor": {"joint_action": [3, 3, 1, 1, 4]},
        "beliefs": beliefs,
        "beliefs_by_factor": {"joint_state": beliefs},
        "expected_free_energy": [1.0, 0.8, 0.6, 0.4, 0.2],
        "variational_free_energy": [0.5, 0.4, 0.3, 0.2, 0.1],
        "validation": {"all_valid": True},
        "model_parameters": {
            "num_states": 9,
            "num_observations": 9,
            "num_actions": 5,
            "B_shape": [9, 9, 5],
        },
        "matrix_provenance": {
            "B": {"canonical_order": "next_state_previous_state_action"}
        },
    }


def test_gridworld_animation_suite_and_manifest(tmp_path: Any) -> None:
    """Current GridWorld schemas produce GIFs and a unified manifest."""
    from gnn.analysis.visualizations import (
        _current_schema_visualization_data,
        generate_gridworld_animation_suite,
        write_gridworld_analysis_manifest,
    )

    execution_dir = tmp_path / "12_execute_output"
    analysis_dir = tmp_path / "16_analysis_output"
    cross_dir = analysis_dir / "cross_framework"
    cross_dir.mkdir(parents=True)
    (analysis_dir / "analysis_results.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "cross_model_comparison_report.md").write_text(
        "current report", encoding="utf-8"
    )
    (cross_dir / "cross_framework_comparison.png").write_bytes(b"png")
    (cross_dir / "pomdp_gridworld_3x3_post_simulation_analysis.json").write_text(
        "{}", encoding="utf-8"
    )
    (cross_dir / "old_model_post_simulation_analysis.json").write_text(
        "{}", encoding="utf-8"
    )
    dashboard_dir = cross_dir / "unified_dashboard"
    dashboard_dir.mkdir()
    (dashboard_dir / "unified_belief_comparison.png").write_bytes(b"png")
    stale_framework_dir = analysis_dir / "jax"
    stale_framework_dir.mkdir()
    (stale_framework_dir / "old_model_jax_dashboard.png").write_bytes(b"png")

    schemas = {
        "pymdp": "pymdp_simulation_v1",
        "rxinfer": "rxinfer_simulation_v1",
        "activeinference_jl": "activeinference_jl_simulation_v1",
    }
    framework_data: dict[str, dict[str, Any]] = {}
    for framework, schema_version in schemas.items():
        payload = _gridworld_payload(schema_version, framework)
        sim_dir = execution_dir / "pomdp_gridworld_3x3" / framework / "simulation_data"
        sim_dir.mkdir(parents=True)
        sim_file = sim_dir / "simulation_results.json"
        sim_file.write_text(json.dumps(payload), encoding="utf-8")
        framework_data[f"{framework}_pomdp_gridworld_3x3"] = {
            "framework": framework,
            "model_name": "pomdp_gridworld_3x3",
            "simulation_data": _current_schema_visualization_data(payload),
            "raw_simulation_data": payload,
            "source_file": str(sim_file),
        }

    gifs = generate_gridworld_animation_suite(framework_data, cross_dir)
    assert len(gifs) == 7
    assert all(Path(path).exists() and Path(path).stat().st_size > 0 for path in gifs)

    manifest_path = write_gridworld_analysis_manifest(
        execution_dir,
        analysis_dir,
        allowed_frameworks=set(schemas),
        allowed_model_names={"pomdp_gridworld_3x3"},
    )
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "gridworld_analysis_manifest_v1"
    assert manifest["frameworks"] == sorted(schemas)
    assert manifest["matrix_provenance_equal"] is True
    assert len(manifest["outputs"]["gif"]) == 7
    assert manifest["outputs"]["dashboard"] == [
        "cross_framework/unified_dashboard/unified_belief_comparison.png"
    ]
    assert manifest["outputs"]["png"]
    assert manifest["outputs"]["statistics"]
    manifest_outputs = "\n".join(
        output for outputs in manifest["outputs"].values() for output in outputs
    )
    assert "old_model" not in manifest_outputs
    assert "jax_dashboard" not in manifest_outputs


def test_gridworld_manifest_flags_missing_matrix_provenance(tmp_path: Any) -> None:
    """A payload lacking matrix_provenance fails the strict equality fold."""
    from gnn.analysis.visualizations import write_gridworld_analysis_manifest

    execution_dir = tmp_path / "12_execute_output"
    analysis_dir = tmp_path / "16_analysis_output"
    for framework, schema_version in (
        ("pymdp", "pymdp_simulation_v1"),
        ("numpyro", "numpyro_simulation_v1"),
    ):
        payload = _gridworld_payload(schema_version, framework)
        if framework == "numpyro":
            del payload["matrix_provenance"]
        sim_dir = execution_dir / "pomdp_gridworld_3x3" / framework / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    manifest_path = write_gridworld_analysis_manifest(
        execution_dir,
        analysis_dir,
        allowed_frameworks={"pymdp", "numpyro"},
        allowed_model_names={"pomdp_gridworld_3x3"},
    )
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["matrix_provenance_equal"] is False


class TestAnalysisOverall:
    """Test suite for Analysis module."""

    @pytest.fixture
    def sample_gnn_for_analysis(self, safe_filesystem: Any) -> Any:
        """Create a sample GNN file to analyze."""
        content = "\n# Analysis Target\n\n## StateSpaceBlock\ns[10, type=float]\n\n## Connections\ns->s\n\n## Time\nDynamic\n"
        return safe_filesystem.create_file("model_analysis.md", content)

    def test_process_analysis_flow(
        self, safe_filesystem: Any, sample_gnn_for_analysis: Any
    ) -> None:
        """Test the analysis processing workflow."""
        target_dir = sample_gnn_for_analysis.parent
        output_dir = safe_filesystem.create_dir("analysis_output")
        try:
            success = process_analysis(target_dir, output_dir, verbose=True)
            assert success is True
            results_dir = output_dir
            assert results_dir.exists()
            assert (results_dir / "analysis_results.json").exists()
            assert (results_dir / "analysis_summary.md").exists()
            with open(results_dir / "analysis_results.json", "r") as f:
                data = json.load(f)
            assert data["processed_files"] == 1
            assert len(data["statistical_analysis"]) == 1
        except ImportError:
            raise AssertionError(
                "Skipping analysis test due to missing dependencies (numpy/matplotlib)"
            )
        except Exception as e:
            pytest.fail(f"Analysis processing failed: {e}")

    def test_process_analysis_no_files(self, safe_filesystem: Any) -> None:
        """Test behavior with no files.

        Phase 1.1 contract: "no input" is a warning (exit-code 2), NOT a hard
        error. Previously this returned False, which the pipeline template then
        translated to exit-code 1 (error). Now it returns 2 so operators see
        "nothing to do" as a distinct signal from "ran and failed".
        """
        empty_dir = safe_filesystem.create_dir("empty")
        output_dir = safe_filesystem.create_dir("output")
        result = process_analysis(empty_dir, output_dir)
        assert result == 2, f"expected exit-code 2 for no-input, got {result!r}"

    def test_generate_animations_normalization_contract(self) -> None:
        """Step 16 uses generate_animations as canonical and supports compatibility inverse."""
        from gnn.analysis.processor import _normalize_generate_animations

        logger = logging.getLogger("test_analysis_animation_contract")

        assert _normalize_generate_animations({}, logger) is True
        assert (
            _normalize_generate_animations({"generate_animations": False}, logger)
            is False
        )
        assert _normalize_generate_animations({"no_animations": True}, logger) is False
        assert (
            _normalize_generate_animations(
                {"generate_animations": True, "no_animations": False},
                logger,
            )
            is True
        )
        with pytest.raises(ValueError, match="Ambiguous animation flags"):
            _normalize_generate_animations(
                {"generate_animations": True, "no_animations": True},
                logger,
            )

    def test_execution_scope_filters_stale_frameworks(
        self, safe_filesystem: Any
    ) -> None:
        """Current Step 12 summary constrains Step 16 framework analysis."""
        from gnn.analysis.analyzer import analyze_framework_outputs
        from gnn.analysis.processor import (
            _filter_execution_summary,
            _scope_from_execution_summary,
        )

        execution_dir = safe_filesystem.create_dir("12_execute_output")
        summaries_dir = execution_dir / "summaries"
        summaries_dir.mkdir(parents=True)
        summary: dict[str, Any] = {
            "requested_frameworks": ["pymdp"],
            "execution_details": [
                {
                    "framework": "pymdp",
                    "model_name": "current_model",
                    "success": True,
                    "execution_time": 0.1,
                    "implementation_directory": str(
                        execution_dir / "current_model" / "pymdp"
                    ),
                },
                {
                    "framework": "rxinfer",
                    "model_name": "stale_model",
                    "success": True,
                    "execution_time": 0.2,
                    "implementation_directory": str(
                        execution_dir / "stale_model" / "rxinfer"
                    ),
                },
            ],
            "framework_status": {
                "pymdp": {"status": "success"},
                "rxinfer": {"status": "success"},
            },
        }
        with open(summaries_dir / "execution_summary.json", "w") as f:
            json.dump(summary, f)

        scope = _scope_from_execution_summary(
            summary, target_model_names={"current_model"}
        )
        assert scope["frameworks"] == {"pymdp"}

        scoped_summary = _filter_execution_summary(summary, {"pymdp"})
        assert [
            detail["framework"] for detail in scoped_summary["execution_details"]
        ] == ["pymdp"]
        assert list(scoped_summary["framework_status"]) == ["pymdp"]

        framework_results = analyze_framework_outputs(
            execution_dir,
            allowed_frameworks={"pymdp"},
        )
        assert set(framework_results["frameworks"]) == {"pymdp"}


class TestPostSimulationVisualization:
    """Test suite for post-simulation visualization functions."""

    def test_generate_belief_heatmaps(self, safe_filesystem: Any) -> None:
        """Test belief heatmap generation."""
        from gnn.analysis.post_simulation import generate_belief_heatmaps

        beliefs: list[Any] = [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.5, 0.4],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
        output_dir = safe_filesystem.create_dir("viz_output")
        output_file = output_dir / "belief_heatmap.png"
        result = generate_belief_heatmaps(beliefs, output_file, "Test Belief Heatmap")
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_action_analysis(self, safe_filesystem: Any) -> None:
        """Test action analysis visualization."""
        from gnn.analysis.post_simulation import generate_action_analysis

        actions: list[Any] = [
            0,
            1,
            2,
            0,
            1,
            1,
            2,
            2,
            0,
            0,
            1,
            2,
            2,
            1,
            0,
            1,
            2,
            0,
            1,
            2,
        ]
        output_dir = safe_filesystem.create_dir("viz_output")
        output_file = output_dir / "action_analysis.png"
        result = generate_action_analysis(actions, output_file, "Test Action Analysis")
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_free_energy_plots(self, safe_filesystem: Any) -> None:
        """Test free energy plot generation."""
        from gnn.analysis.post_simulation import generate_free_energy_plots

        free_energy: list[Any] = [
            10.0,
            9.5,
            9.0,
            8.5,
            8.2,
            7.8,
            7.5,
            7.2,
            7.0,
            6.8,
            6.5,
            6.3,
            6.1,
            6.0,
            5.9,
            5.8,
            5.7,
            5.6,
            5.5,
            5.5,
        ]
        output_dir = safe_filesystem.create_dir("viz_output")
        output_file = output_dir / "free_energy.png"
        result = generate_free_energy_plots(
            free_energy, output_file, "Test Free Energy"
        )
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Iteration VFE and timestep EFE are independent sample domains.
        from unittest.mock import patch

        from gnn.analysis import viz_plots

        vfe = [float(i) for i in range(20)]
        efe = [[float(i), float(i) + 1.0] for i in range(15)]
        dual_output = output_dir / "vfe_efe.png"
        with patch.object(viz_plots, "safe_savefig", return_value=str(dual_output)):
            assert viz_plots.generate_vfe_vs_efe_plot(
                vfe, efe, dual_output, vfe_per_iteration=True
            ) == str(dual_output)
            fig = viz_plots.plt.gcf()
            try:
                vfe_ax, efe_ax = fig.axes
                assert not vfe_ax.get_shared_x_axes().joined(vfe_ax, efe_ax)
                assert vfe_ax.get_xlabel() == "Inference Iteration (1-based)"
                assert efe_ax.get_xlabel() == "Time Step"
                assert list(vfe_ax.lines[0].get_xdata()) == list(range(1, 21))
                assert list(efe_ax.lines[0].get_xdata()) == list(range(15))
                assert list(vfe_ax.lines[0].get_ydata()) == vfe
                assert list(efe_ax.lines[0].get_ydata()) == [row[0] for row in efe]
            finally:
                viz_plots.plt.close(fig)

            # Existing callers still receive the original shared timestep plot.
            viz_plots.generate_vfe_vs_efe_plot(vfe, efe, dual_output)
            fig = viz_plots.plt.gcf()
            try:
                assert fig.axes[0].get_xlabel() == "Time Step"
                assert list(fig.axes[0].lines[0].get_ydata()) == vfe
                assert list(fig.axes[1].lines[0].get_ydata()) == [row[0] for row in efe]
            finally:
                viz_plots.plt.close(fig)

    def test_generate_observation_analysis(self, safe_filesystem: Any) -> None:
        """Test observation analysis visualization."""
        from gnn.analysis.post_simulation import generate_observation_analysis

        observations: list[Any] = [0, 1, 0, 2, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1]
        output_dir = safe_filesystem.create_dir("viz_output")
        output_file = output_dir / "observations.png"
        result = generate_observation_analysis(
            observations, output_file, "Test Observations"
        )
        assert result == str(output_file)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_analyze_free_energy(self) -> None:
        """Test free energy analysis function."""
        from gnn.analysis.post_simulation import analyze_free_energy

        fe_values: list[Any] = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.1, 2.05, 2.02]
        result = analyze_free_energy(fe_values, "pymdp", "test_model")
        assert result["framework"] == "pymdp"
        assert result["model_name"] == "test_model"
        assert result["free_energy_count"] == 10
        assert "mean_free_energy" in result
        assert "std_free_energy" in result
        assert result["free_energy_decreasing"]

    def test_analyze_simulation_traces(self) -> None:
        """Test simulation trace analysis function."""
        from gnn.analysis.post_simulation import analyze_simulation_traces

        traces: list[Any] = [[0, 1, 2, 1, 0], [1, 2, 2, 0, 1, 2], [0, 0, 1, 2]]
        result = analyze_simulation_traces(traces, "rxinfer", "test_model")
        assert result["framework"] == "rxinfer"
        assert result["trace_count"] == 3
        assert result["trace_lengths"] == [5, 6, 4]
        assert result["avg_trace_length"] == 5.0

    def test_analyze_policy_convergence(self) -> None:
        """Test policy convergence analysis."""
        from gnn.analysis.post_simulation import analyze_policy_convergence

        policy_traces: list[Any] = [
            [0.33, 0.33, 0.34],
            [0.4, 0.3, 0.3],
            [0.5, 0.25, 0.25],
            [0.7, 0.15, 0.15],
            [0.9, 0.05, 0.05],
        ]
        result = analyze_policy_convergence(policy_traces, "jax", "test_model")
        assert result["framework"] == "jax"
        assert result["policy_count"] == 5
        assert len(result["policy_entropy"]) == 5
        assert result["policy_entropy"][0] > result["policy_entropy"][-1]

    def test_compare_framework_results(self) -> None:
        """Test cross-framework comparison."""
        from gnn.analysis.post_simulation import compare_framework_results

        framework_results: dict[str, Any] = {
            "pymdp": {
                "success": True,
                "execution_time": 1.5,
                "simulation_data": {"free_energy": [10.0, 8.0, 6.0]},
            },
            "rxinfer": {
                "success": True,
                "execution_time": 0.8,
                "simulation_data": {"free_energy": [12.0, 9.0, 7.0]},
            },
        }
        result = compare_framework_results(framework_results, "test_model")
        assert result["framework_count"] == 2
        assert "pymdp" in result["frameworks_compared"]
        assert "rxinfer" in result["frameworks_compared"]
        assert "fastest_execution" not in result["comparisons"]
        assert (
            "missing declared model_id"
            in result["comparisons"]["unavailable_metrics"]["fastest_execution"]
        )


class TestAnalysisModuleImports:
    """Test that all new visualization functions are properly exported."""

    def test_visualization_function_exports(self) -> None:
        """Test that new visualization functions are exported from analysis module."""
        from gnn.analysis import (
            animate_belief_evolution,
            generate_action_analysis,
            generate_belief_heatmaps,
            generate_cross_framework_comparison,
            generate_free_energy_plots,
            generate_observation_analysis,
            plot_belief_evolution,
            visualize_all_framework_outputs,
        )

        assert callable(visualize_all_framework_outputs)
        assert callable(generate_belief_heatmaps)
        assert callable(generate_action_analysis)
        assert callable(generate_free_energy_plots)
        assert callable(generate_observation_analysis)
        assert callable(generate_cross_framework_comparison)
        assert callable(plot_belief_evolution)
        assert callable(animate_belief_evolution)


class TestActiveInferenceJLAnalyzer:
    def test_module_importable(self) -> Any:
        from gnn.analysis.activeinference_jl import analyzer

    def test_generate_analysis_from_logs_missing_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.activeinference_jl.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path / "nonexistent", tmp_path / "out")
        assert isinstance(result, list)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.activeinference_jl.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path, tmp_path / "out")
        assert isinstance(result, list)


class TestDisCoPyAnalyzer:
    def test_module_importable(self) -> Any:
        from gnn.analysis.discopy import analyzer

    def test_extract_circuit_data_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.discopy.analyzer import extract_circuit_data

        result = extract_circuit_data(tmp_path)
        assert isinstance(result, dict)

    def test_analyze_diagram_structure_empty(self) -> Any:
        from gnn.analysis.discopy.analyzer import analyze_diagram_structure

        result = analyze_diagram_structure([])
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.discopy.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path, tmp_path / "out")
        assert isinstance(result, list)


class TestJAXAnalyzer:
    def test_module_importable(self) -> Any:
        from gnn.analysis.jax import analyzer

    def test_parse_raw_output_empty_string(self) -> Any:
        from gnn.analysis.jax.analyzer import parse_raw_output

        result = parse_raw_output("")
        assert isinstance(result, dict)

    def test_extract_simulation_data_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.jax.analyzer import extract_simulation_data

        result = extract_simulation_data(tmp_path)
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.jax.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path, tmp_path / "out")
        assert isinstance(result, list)


class TestPyMDPAnalyzer:
    def test_module_importable(self) -> Any:
        from gnn.analysis.pymdp import analyzer

    def test_generate_analysis_from_logs_missing_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.pymdp.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path / "nonexistent", tmp_path / "out")
        assert isinstance(result, list)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.pymdp.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path, tmp_path / "out")
        assert isinstance(result, list)

    def test_generate_analysis_prefers_current_named_results(
        self, tmp_path: Any, monkeypatch: Any, caplog: Any
    ) -> Any:
        from gnn.analysis.pymdp import analyzer

        exec_dir = tmp_path / "execute"
        sim_dir = exec_dir / "model_a" / "pymdp" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps({"framework": "PyMDP", "model_name": "historical"}),
            encoding="utf-8",
        )
        (sim_dir / "Model A_simulation_results.json").write_text(
            json.dumps(
                {
                    "schema_version": "pymdp_simulation_v1",
                    "framework": "PyMDP",
                    "model_name": "Model A",
                    "beliefs_by_factor": {"joint_state": [[0.7, 0.3], [0.2, 0.8]]},
                    "hidden_states_by_factor": {"joint_state": [0, 1]},
                    "observations_by_modality": {"joint_observation": [0, 1]},
                    "actions_by_control_factor": {"joint_action": [0, 0]},
                    "expected_free_energy": [0.1, 0.2],
                    "variational_free_energy": [0.3, 0.4],
                    "policy_posterior": [[1.0], [1.0]],
                    "model_parameters": {"num_states": 2},
                    "metrics": {"belief_confidence": [0.7, 0.8]},
                }
            ),
            encoding="utf-8",
        )

        def _save_all_visualizations(**kwargs: Any) -> Any:
            return {"beliefs": kwargs["output_dir"] / "beliefs.png"}

        monkeypatch.setattr(
            analyzer, "save_all_visualizations", _save_all_visualizations
        )
        with caplog.at_level(logging.ERROR, logger="gnn.analysis.pymdp"):
            result = analyzer.generate_analysis_from_logs(exec_dir, tmp_path / "out")

        assert len(result) == 1
        assert "unsupported PyMDP schema" not in caplog.text


class TestRxInferAnalyzer:
    def test_module_importable(self) -> Any:
        from gnn.analysis.rxinfer import analyzer

    def test_extract_simulation_data_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.rxinfer.analyzer import extract_simulation_data

        result = extract_simulation_data(tmp_path)
        assert isinstance(result, dict)

    def test_generate_analysis_from_logs_empty_dir(self, tmp_path: Any) -> Any:
        from gnn.analysis.rxinfer.analyzer import generate_analysis_from_logs

        result = generate_analysis_from_logs(tmp_path, tmp_path / "out")
        assert isinstance(result, list)


class TestAnalyzerSimulationMetrics:
    """Behavioral tests for analyzer.py private simulation metric functions."""

    def _make_logger(self) -> Any:
        import logging

        return logging.getLogger("test_analyzer")

    def test_extract_simulation_metrics_returns_dict(self, tmp_path: Any) -> Any:
        """_extract_simulation_metrics returns a dict with expected keys."""
        from gnn.analysis.analyzer import _extract_simulation_metrics

        logger = self._make_logger()
        result = _extract_simulation_metrics("pymdp", [], tmp_path, logger)
        assert isinstance(result, dict)
        assert "beliefs" in result
        assert "actions" in result
        assert "observations" in result
        assert "free_energy" in result
        assert "execution_times" in result

    def test_extract_simulation_metrics_reads_pymdp_v1_json(self, tmp_path: Any) -> Any:
        """_extract_simulation_metrics loads current PyMDP schema when present."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        sim_dir = tmp_path / "sim_data"
        sim_dir.mkdir()
        sim_results: dict[str, Any] = {
            "schema_version": "pymdp_simulation_v1",
            "framework": "PyMDP",
            "beliefs_by_factor": {"joint_state": [[0.9, 0.1], [0.8, 0.2]]},
            "actions_by_control_factor": {"joint_action": [0, 1]},
            "observations_by_modality": {"joint_observation": [2, 3]},
            "expected_free_energy": [-1.5, -1.3],
        }
        (sim_dir / "simulation_results.json").write_text(json.dumps(sim_results))
        detail: dict[str, Any] = {
            "implementation_directory": str(sim_dir),
            "execution_time": 0.5,
        }
        logger = self._make_logger()
        result = _extract_simulation_metrics("pymdp", [detail], tmp_path, logger)
        assert result["beliefs"] == sim_results["beliefs_by_factor"]["joint_state"]
        assert (
            result["actions"]
            == sim_results["actions_by_control_factor"]["joint_action"]
        )
        assert result["free_energy"] == sim_results["expected_free_energy"]
        assert result["execution_times"] == [0.5]

    def test_extract_simulation_metrics_rejects_non_v1_pymdp_json(
        self, tmp_path: Any
    ) -> Any:
        """PyMDP cross-framework metrics should not consume older JSON shapes."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        sim_dir = tmp_path / "sim_data"
        sim_dir.mkdir()
        (sim_dir / "simulation_results.json").write_text(
            json.dumps({"framework": "PyMDP", "beliefs": [[0.9, 0.1]]})
        )
        detail: dict[str, Any] = {
            "implementation_directory": str(sim_dir),
            "execution_time": 0.5,
        }
        result = _extract_simulation_metrics(
            "pymdp", [detail], tmp_path, self._make_logger()
        )
        assert result["beliefs"] == []
        assert result["data_source"] is None

    def test_extract_simulation_metrics_missing_dir(self, tmp_path: Any) -> Any:
        """_extract_simulation_metrics handles nonexistent impl_dir gracefully."""
        from gnn.analysis.analyzer import _extract_simulation_metrics

        logger = self._make_logger()
        detail: dict[str, Any] = {
            "implementation_directory": str(tmp_path / "nonexistent"),
            "execution_time": 1.0,
        }
        result = _extract_simulation_metrics("rxinfer", [detail], tmp_path, logger)
        assert isinstance(result, dict)
        assert result["execution_times"] == [1.0]

    def test_extract_simulation_metrics_rejects_stale_checkout_output(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skipped details must not scan repo-root output from an isolated run."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        checkout = tmp_path / "checkout"
        stale_dir = checkout / "output" / "11_render_output" / "simple_mdp" / "rxinfer"
        stale_dir.mkdir(parents=True)
        (stale_dir / "simulation_results.json").write_text(
            json.dumps({"beliefs": [[1.0]], "free_energy": [0.0]}),
            encoding="utf-8",
        )
        execution_dir = tmp_path / "run" / "12_execute_output"
        execution_dir.mkdir(parents=True)
        monkeypatch.chdir(checkout)

        detail: dict[str, Any] = {
            "framework": "numpyro",
            "success": False,
            "skipped": True,
            "error": "Dependency not installed: numpyro",
        }
        result = _extract_simulation_metrics(
            "numpyro", [detail], execution_dir, self._make_logger()
        )
        assert result["beliefs"] == []
        assert result["free_energy"] == []
        assert result["data_source"] is None

    def test_extract_simulation_metrics_bnlearn_execution_logs(
        self, tmp_path: Any
    ) -> Any:
        """bnlearn writes execution_logs/*_results.json; metrics should still record completion."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        impl = tmp_path / "markov_chain" / "bnlearn"
        el = impl / "execution_logs"
        el.mkdir(parents=True)
        structured: dict[str, Any] = {
            "framework": "bnlearn",
            "model_name": "markov_chain",
            "success": True,
            "simulation_data": {"beliefs": [], "actions": [], "observations": []},
        }
        (el / "Simple_bnlearn.py_results.json").write_text(json.dumps(structured))
        detail: dict[str, Any] = {
            "implementation_directory": str(impl),
            "execution_time": 1.0,
        }
        logger = self._make_logger()
        result = _extract_simulation_metrics("bnlearn", [detail], tmp_path, logger)
        assert result["model_parameters"].get("bnlearn_completed") is True
        assert result["model_parameters"].get("model_name") == "markov_chain"
        assert result["data_source"]

    def test_extract_simulation_metrics_rxinfer_prefers_simulation_data(
        self, tmp_path: Any
    ) -> Any:
        """simulation_data/simulation_results.json must win over sparse execution_logs."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        impl = tmp_path / "rx" / "rxinfer"
        el = impl / "execution_logs"
        sd = impl / "simulation_data"
        el.mkdir(parents=True)
        sd.mkdir(parents=True)
        sparse: dict[str, Any] = {
            "framework": "rxinfer",
            "model_name": "markov_chain",
            "success": True,
            "simulation_data": {"beliefs": [], "actions": [], "observations": []},
        }
        (el / "Model_rxinfer.jl_results.json").write_text(json.dumps(sparse))
        rich: dict[str, Any] = {
            "beliefs": [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]],
            "actions": [1, 1],
            "observations": [2, 3],
            "efe_history": [0.1, 0.2],
        }
        (sd / "simulation_results.json").write_text(json.dumps(rich))
        detail: dict[str, Any] = {
            "implementation_directory": str(impl),
            "execution_time": 1.0,
        }
        logger = self._make_logger()
        result = _extract_simulation_metrics("rxinfer", [detail], tmp_path, logger)
        assert result["beliefs"] == rich["beliefs"]
        assert "simulation_data" in result["data_source"].replace("\\", "/")
        assert "simulation_results.json" in result["data_source"].replace("\\", "/")

    def test_extract_simulation_metrics_discopy_supplements_circuit_info(
        self, tmp_path: Any
    ) -> Any:
        """DisCoPy: execution log plus circuit_info.json yields circuit metrics (no empty extract)."""
        import json

        from gnn.analysis.analyzer import _extract_simulation_metrics

        impl = tmp_path / "m" / "discopy"
        el = impl / "execution_logs"
        sd = impl / "simulation_data"
        el.mkdir(parents=True)
        sd.mkdir(parents=True)
        result_summary: dict[str, Any] = {
            "framework": "discopy",
            "model_name": "markov_chain",
            "success": True,
            "simulation_data": {
                "traces": [],
                "beliefs": [],
                "actions": [],
                "observations": [],
            },
        }
        (el / "Model_discopy.py_results.json").write_text(json.dumps(result_summary))
        circuit_data: dict[str, Any] = {
            "model_name": "markov_chain",
            "components": ["A_matrix", "B_matrix"],
            "analysis": {"num_components": 8},
            "parameters": {"num_states": 3, "num_observations": 3, "num_actions": 1},
        }
        (sd / "circuit_info.json").write_text(json.dumps(circuit_data))
        detail: dict[str, Any] = {
            "implementation_directory": str(impl),
            "execution_time": 0.5,
        }
        logger = self._make_logger()
        result = _extract_simulation_metrics("discopy", [detail], tmp_path, logger)
        assert result.get("circuit_info") is not None
        assert result["circuit_info"].get("num_components") == 8
        assert result.get("model_parameters", {}).get("num_states") == 3

    def test_compare_framework_results_empty_input(self) -> Any:
        """_compare_framework_results returns dict with expected keys for empty input."""
        from gnn.analysis.analyzer import _compare_framework_results

        logger = self._make_logger()
        result = _compare_framework_results({}, logger)
        assert isinstance(result, dict)
        assert "success_rates" in result
        assert "performance_comparison" in result
        assert "data_coverage" in result
        assert "simulation_statistics" in result

    def test_compare_framework_results_success_rates(self) -> Any:
        """_compare_framework_results computes success rates correctly."""
        from gnn.analysis.analyzer import _compare_framework_results

        logger = self._make_logger()
        framework_data: dict[str, Any] = {
            "pymdp": {"success_count": 3, "total_count": 4, "execution_times": []},
            "jax": {"success_count": 4, "total_count": 4, "execution_times": []},
        }
        result = _compare_framework_results(framework_data, logger)
        assert abs(result["success_rates"]["pymdp"] - 0.75) < 1e-06
        assert abs(result["success_rates"]["jax"] - 1.0) < 1e-06

    def test_compare_framework_results_execution_times(self) -> Any:
        """_compare_framework_results computes perf stats when times present."""
        from gnn.analysis.analyzer import _compare_framework_results

        logger = self._make_logger()
        framework_data: dict[str, Any] = {
            "pymdp": {
                "success_count": 2,
                "total_count": 2,
                "execution_times": [1.0, 2.0],
            }
        }
        result = _compare_framework_results(framework_data, logger)
        perf = result["performance_comparison"]["pymdp"]
        assert abs(perf["mean"] - 1.5) < 1e-06
        assert perf["min"] == 1.0
        assert perf["max"] == 2.0

    def test_visualize_simulation_results_no_details(self, tmp_path: Any) -> Any:
        """visualize_simulation_results returns list (empty) when no details."""
        from gnn.analysis.analyzer import visualize_simulation_results

        result = visualize_simulation_results({"execution_details": []}, tmp_path)
        assert isinstance(result, list)
