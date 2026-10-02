"""Scientific comparison requires identity, input, inference and native semantics."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from gnn.analysis.comparison_contract import (
    compare_scientific_pair,
    performance_admission,
    prepare_scientific_record,
    revalidate_scientific_record,
)
from gnn.analysis.framework_comparison import (
    analyze_framework_outputs,
    generate_framework_comparison_report,
)
from gnn.analysis.trace_analysis import (
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_state_distributions,
    compare_framework_results,
)


def _categorical() -> dict[str, Any]:
    return {
        "model_name": "shared display name",
        "model_id": "nested/model-identity",
        "source_sha256": "a" * 64,
        "source_relative_path": "nested/model.md",
        "model_kind": "discrete",
        "inference_mode": "filtering",
        "dtype": "float64",
        "belief_axes": ["timestep", "state"],
        "beliefs": [[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]],
        "observations": [0, 1, 0],
        "actions": [0, 1, 1],
        "num_timesteps": 3,
        "success": True,
        "execution_time": 1.0,
        "execution_configuration": {"timesteps": 3, "seed": 42},
        "execution_environment": {"platform": "test-cpu", "threads": 1},
    }


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return compare_scientific_pair(
        prepare_scientific_record(left), prepare_scientific_record(right)
    )


def test_same_model_has_value_bound_witness_and_public_parity() -> None:
    left, right = _categorical(), _categorical()
    right["beliefs"][0] = [0.60000000001, 0.39999999999]
    right["execution_time"] = 0.5
    pair = _pair(left, right)
    assert pair["status"] == "comparable"
    assert pair["source_sha256"] == "a" * 64
    assert pair["views"]["joint_state"]["within_tolerance"] is True
    assert pair["views"]["joint_state"]["max_absolute_difference"] > 0
    public = compare_framework_results({"jax": left, "pytorch": right}, "label")
    assert public["comparisons"]["metric_agreement"]["jax_vs_pytorch"] == pair
    assert public["comparisons"]["fastest_execution"]["framework"] == "pytorch"


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("model_id", "another/model-identity", "incompatible model_id"),
        ("source_sha256", "b" * 64, "incompatible source_sha256"),
        (
            "source_relative_path",
            "another/model.md",
            "incompatible source_relative_path",
        ),
        ("inference_mode", "smoothing", "incompatible inference_mode"),
        ("belief_axes", ["state", "timestep"], "unaligned belief_axes"),
        ("observations", [1, 0, 0], "unaligned observations"),
        ("actions", [1, 1, 1], "unaligned actions"),
        ("posterior_semantics", "full_sequence", "incompatible posterior_semantics"),
        ("input_sha256", "c" * 64, "incompatible input_sha256"),
    ],
)
def test_same_shapes_or_display_stems_do_not_admit_different_models(
    key: str, value: Any, reason: str
) -> None:
    left, right = _categorical(), _categorical()
    right[key] = value
    pair = _pair(left, right)
    assert pair["status"] == "not_comparable"
    assert reason in pair["reasons"]
    assert "views" not in pair


@pytest.mark.parametrize(
    "key", ["model_id", "source_sha256", "model_kind", "inference_mode", "dtype"]
)
def test_missing_metadata_is_unavailable_not_inferred_from_a_name(key: str) -> None:
    left, right = _categorical(), _categorical()
    del right[key]
    pair = _pair(left, right)
    assert pair["status"] == "not_comparable"
    assert f"missing declared {key}" in pair["reasons"]


@pytest.mark.parametrize(
    "location", ["runtime_metadata", "model_parameters", "source_identity"]
)
def test_conflicting_payload_custody_is_not_overwritten_by_step12(
    location: str,
) -> None:
    data = _categorical()
    data[location] = {"source_sha256": "b" * 64}
    record = prepare_scientific_record(data, {"source_sha256": "a" * 64})
    assert "conflicting source_sha256 declarations" in record["reasons"]


@pytest.mark.parametrize("dtype,expected", [("float32", True), ("float64", False)])
def test_declared_dtype_controls_numeric_tolerance(dtype: str, expected: bool) -> None:
    left, right = _categorical(), _categorical()
    left["dtype"] = right["dtype"] = dtype
    right["beliefs"][0] = [0.6000005, 0.3999995]
    pair = _pair(left, right)
    assert pair["status"] == "comparable"
    assert pair["views"]["joint_state"]["within_tolerance"] is expected


def test_mixed_declared_float_precision_uses_least_precision() -> None:
    left, right = _categorical(), _categorical()
    left["dtype"] = "float32"
    pair = _pair(left, right)
    assert pair["tolerance"] == {"rtol": 1e-5, "atol": 1e-6}


@pytest.mark.parametrize(
    "beliefs",
    [
        [[-0.1, 1.1]],
        [[float("nan"), 0.5]],
        [[float("inf"), 0]],
        [[0, 0]],
        [[0.1, 0.2]],
        [["not numeric", 1]],
        [[0.5, 0.5], [1]],
    ],
)
def test_malformed_probability_results_are_not_normalized(beliefs: Any) -> None:
    left = _categorical()
    left["beliefs"] = beliefs
    before = deepcopy(left)
    pair = _pair(left, _categorical())
    assert pair["status"] == "not_comparable"
    assert any("invalid scientific result" in reason for reason in pair["reasons"])
    np.testing.assert_equal(left["beliefs"], before["beliefs"])


@pytest.mark.parametrize(
    "analyzer", [analyze_policy_convergence, analyze_state_distributions]
)
@pytest.mark.parametrize(
    "values",
    [[[-0.1, 1.1]], [[0, 0]], [[0.1, 0.2]], [[float("nan"), 0.5]], [[float("inf"), 0]]],
)
def test_public_probability_analysis_returns_failure_without_repair(
    analyzer, values
) -> None:
    result = analyzer(values, "jax", "model")
    assert result["status"] == "failed"
    assert "error" in result
    assert "state_entropy" not in result and "policy_entropy" not in result


def _gaussian() -> dict[str, Any]:
    data = _categorical()
    data.update(
        model_kind="continuous",
        inference_mode="kalman_filter",
        beliefs=[[-5.0, -3.0], [-4.0, -2.0]],
        num_timesteps=2,
        posterior_cov=[[[0.4, 0.1], [0.1, 0.3]], [[0.2, 0], [0, 0.1]]],
        observations_continuous=[[0.5], [0.4]],
    )
    data.pop("actions")
    data.pop("observations")
    return data


def test_gaussian_uncertainty_is_covariance_bound_and_has_no_confidence() -> None:
    data = _gaussian()
    record = prepare_scientific_record(data)
    assert record["reasons"] == []
    stats = record["statistics"]
    assert "mean_confidence" not in stats
    assert stats["mean_posterior_std"] > 0
    assert stats["final_posterior_mean"] == [-4, -2]
    pair = _pair(data, deepcopy(data))
    assert pair["status"] == "comparable"
    assert "confidence_correlation" not in pair["views"]["continuous"]
    assert pair["views"]["continuous"]["covariance_within_tolerance"] is True


@pytest.mark.parametrize(
    "covariance",
    [
        [[[-1, 0], [0, 1]], [[1, 0], [0, 1]]],
        [[[1, 0.5], [0, 1]], [[1, 0], [0, 1]]],
        [[[1]], [[1]]],
    ],
)
def test_invalid_gaussian_covariance_cannot_produce_numeric_agreement(
    covariance,
) -> None:
    data = _gaussian()
    data["posterior_cov"] = covariance
    record = prepare_scientific_record(data)
    assert not record["views"]
    assert "mean_confidence" not in record["statistics"]
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


@pytest.mark.parametrize("family", ["multi_agent", "factored"])
def test_native_view_identities_survive_and_renames_prevent_comparison(
    family: str,
) -> None:
    data = _categorical()
    data["model_kind"] = family
    data.pop("beliefs")
    suffix = "agent" if family == "multi_agent" else "factor"
    data[f"beliefs_by_{suffix}"] = {
        "alpha": [[0.6, 0.4]] * 3,
        "beta": [[0.2, 0.3, 0.5]] * 3,
    }
    data[f"observations_by_{suffix}"] = {"alpha": [0, 0, 0], "beta": [1, 1, 1]}
    if family == "multi_agent":
        data["agents"] = ["alpha", "beta"]
    pair = _pair(data, deepcopy(data))
    assert pair["status"] == "comparable"
    assert set(pair["views"]) == {"alpha", "beta"}
    changed = deepcopy(data)
    changed[f"beliefs_by_{suffix}"]["renamed"] = changed[f"beliefs_by_{suffix}"].pop(
        "beta"
    )
    if family == "multi_agent":
        changed["agents"] = ["alpha", "renamed"]
    assert _pair(data, changed)["status"] == "not_comparable"


@pytest.mark.parametrize(
    "alteration",
    ["kind", "false_validation", "source_values", "missing_inputs", "missing_axes"],
)
def test_equal_but_unverified_or_invalid_claims_are_refused(alteration: str) -> None:
    data = _categorical()
    if alteration == "kind":
        data["model_kind"] = "unimplemented_family"
    elif alteration == "false_validation":
        data["validation"] = {"source_probabilities_valid": False}
    elif alteration == "source_values":
        data["source_model_parameters"] = {"A": [[-1, 2], [2, -1]], "D": [0.5, 0.5]}
    elif alteration == "missing_inputs":
        data.pop("observations")
    else:
        data.pop("belief_axes")
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_source_parameter_value_corruption_and_optional_inputs_are_checked() -> None:
    left, right = _categorical(), _categorical()
    left["source_model_parameters"] = {"A": [[0.8, 0.1], [0.2, 0.9]], "D": [0.4, 0.6]}
    right["source_model_parameters"] = {"A": [[0.7, 0.1], [0.3, 0.9]], "D": [0.4, 0.6]}
    pair = _pair(left, right)
    assert pair["status"] == "not_comparable"
    assert "unaligned source_model_parameters" in pair["reasons"]
    right = _categorical()
    right.pop("actions")
    assert "unaligned actions" in _pair(_categorical(), right)["reasons"]


def test_unbound_and_different_environment_timings_do_not_rank() -> None:
    left, right = _categorical(), _categorical()
    right["execution_environment"]["threads"] = 2
    result = compare_framework_results({"jax": left, "pytorch": right}, "model")
    assert result["comparisons"]["execution_time"] == {"jax": 1.0, "pytorch": 1.0}
    assert "fastest_execution" not in result["comparisons"]
    assert (
        "different execution_environment"
        in result["comparisons"]["unavailable_metrics"]["fastest_execution"]
    )


def _write_result(
    root: Path, framework: str, payload: dict[str, Any]
) -> dict[str, Any]:
    path = (
        root
        / payload["model_id"]
        / framework
        / "simulation_data"
        / "simulation_results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return {
        "framework": framework,
        "model_name": payload["model_name"],
        "model_id": payload["model_id"],
        "source_sha256": payload["source_sha256"],
        "success": True,
        "skipped": False,
        "execution_time": 1.0,
        "implementation_directory": str(path.parent.parent),
        "output_file": str(path),
    }


def test_pipeline_corpus_records_are_not_overwritten_or_cross_named(
    tmp_path: Path,
) -> None:
    a, b = _categorical(), _categorical()
    b.update(
        model_id="different/model-identity",
        source_sha256="b" * 64,
        source_relative_path="different/model.md",
    )
    details = [
        _write_result(tmp_path, "jax", a),
        _write_result(tmp_path, "pytorch", b),
        _write_result(tmp_path, "jax", b),
    ]
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "execution_summary.json").write_text(
        json.dumps({"execution_details": details})
    )
    result = analyze_framework_outputs(tmp_path)
    assert result["metrics"]["total_executions"] == 3
    assert result["frameworks"]["jax"]["execution_times"] == [1.0, 1.0]
    assert len(result["frameworks"]["jax"]["scientific_records"]) == 2
    assert result["frameworks"]["jax"]["beliefs"] == []
    agreement = result["comparisons"]["metric_agreement"]
    assert len(agreement) == 1
    assert next(iter(agreement.values()))["model_id"] == "different/model-identity"
    report = Path(
        generate_framework_comparison_report(result, tmp_path / "reports")
    ).read_text()
    assert (
        "different/model-identity" in report and "no source-bound counterpart" in report
    )
    assert "Operational Timing Aggregates" in report


def test_pipeline_gaussian_statistics_and_report_are_not_categorical(
    tmp_path: Path,
) -> None:
    details = [_write_result(tmp_path, "jax", _gaussian())]
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "execution_summary.json").write_text(
        json.dumps({"execution_details": details})
    )
    result = analyze_framework_outputs(tmp_path)
    stats = result["comparisons"]["simulation_statistics"]["jax"]
    assert "mean_confidence" not in stats and stats["mean_posterior_std"] > 0
    report = Path(
        generate_framework_comparison_report(result, tmp_path / "reports")
    ).read_text()
    assert "continuous" in report and "Posterior Std" in report
    assert "-2.5000" not in report


def test_asymmetric_rank_three_likelihood_preserves_leading_observation_axis() -> None:
    data = _categorical()
    a = np.asarray(
        [[[0.8, 0.3, 0.6], [0.1, 0.4, 0.2]], [[0.2, 0.7, 0.4], [0.9, 0.6, 0.8]]]
    )
    data["source_model_parameters"] = {"A": a.tolist()}
    assert _pair(data, deepcopy(data))["status"] == "comparable"
    data["source_model_parameters"]["A"][0][1][2] = 0.5
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


@pytest.mark.parametrize("location", ["model_parameters", "matrix_provenance"])
def test_external_source_transition_orientation_is_respected(location: str) -> None:
    data = _categorical()
    b = [[[0.9, 0.3], [0.1, 0.7]], [[0.2, 0.6], [0.8, 0.4]], [[0.4, 0.8], [0.6, 0.2]]]
    data["source_model_parameters"] = {"B": b}
    if location == "model_parameters":
        data[location] = {"b_tensor_order": "action_next_state_previous_state"}
    else:
        data[location] = {"B": {"source_order": "action_next_state_previous_state"}}
    assert _pair(data, deepcopy(data))["status"] == "comparable"
    data[location] = {}
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_descriptive_false_flags_are_not_invalid_probability_claims() -> None:
    data = _categorical()
    data["validation"] = {
        "joint_materialized": False,
        "converged": False,
        "beliefs_valid": True,
    }
    assert _pair(data, deepcopy(data))["status"] == "comparable"


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_parameters", {"num_states": 3}),
        ("source_model_parameters", {"D": [0.2, 0.3, 0.5]}),
    ],
)
def test_same_posterior_shapes_must_match_declared_source_dimensions(
    key, value
) -> None:
    data = _categorical()
    data[key] = value
    pair = _pair(data, deepcopy(data))
    assert pair["status"] == "not_comparable"
    assert any("posterior width" in reason for reason in pair["reasons"])


def test_source_coupling_changes_refuse_and_dictionary_order_is_irrelevant() -> None:
    data = _categorical()
    data["model_kind"] = "multi_agent"
    data.pop("beliefs")
    data["beliefs_by_agent"] = {"one": [[0.8, 0.2]] * 3, "two": [[0.3, 0.3, 0.4]] * 3}
    data["agent_coupling"] = {"one": {"two": 0.3}}
    swapped = deepcopy(data)
    swapped["beliefs_by_agent"] = dict(
        reversed(list(swapped["beliefs_by_agent"].items()))
    )
    assert _pair(data, swapped)["status"] == "comparable"
    swapped["agent_coupling"]["one"]["two"] = 0.9
    assert "unaligned agent_coupling" in _pair(data, swapped)["reasons"]


@pytest.mark.parametrize(
    "values", [[[1, 2], [3]], [[1, 2], [3, 4]], [float("nan")], [float("inf")]]
)
def test_free_energy_is_not_silently_reduced_or_denetworked(values) -> None:
    result = analyze_free_energy(values, "jax", "model")
    assert result["status"] == "failed" and "mean_free_energy" not in result


def test_scalar_free_energy_requires_declared_convention_before_comparison() -> None:
    left, right = _categorical(), _categorical()
    left["free_energy"] = right["free_energy"] = [3.0, 2.0, 1.0]
    pair = _pair(left, right)
    assert "free_energy" not in pair
    assert "convention" in pair["unavailable_metrics"]["free_energy"]
    for data in (left, right):
        data.update(
            free_energy_kind="variational", free_energy_convention="negative_elbo"
        )
    pair = _pair(left, right)
    assert pair["free_energy"]["within_tolerance"] is True


def test_malformed_input_identity_and_runtime_metadata_refuse_without_exception() -> (
    None
):
    data = _categorical()
    data.pop("observations")
    data["input_sha256"] = "invalid"
    assert (
        "input_sha256 must be a SHA-256 hex digest"
        in _pair(data, deepcopy(data))["reasons"]
    )
    data = _categorical()
    data.pop("model_kind")
    data["runtime_metadata"] = "malformed"
    pair = _pair(data, deepcopy(data))
    assert pair["status"] == "not_comparable"
    assert "invalid runtime_metadata: metadata must be a mapping" in pair["reasons"]


def test_failed_and_duplicate_attempts_cannot_win_a_performance_rank() -> None:
    good = prepare_scientific_record(_categorical())
    bad_result = _categorical()
    bad_result.update(success=False, execution_time=0.1)
    bad = prepare_scientific_record(bad_result)
    assert "unsuccessful or unfinished attempted result" in performance_admission(
        {"jax": [good], "pytorch": [bad]}
    )
    assert "different attempted model identity sets" in performance_admission(
        {"jax": [good], "pytorch": [good, good]}
    )


def test_transition_action_semantics_allow_t_minus_one_without_padding() -> None:
    data = _categorical()
    data["actions"] = [0, 1]
    data["action_semantics"] = "transition_into_next_state"
    pair = _pair(data, deepcopy(data))
    assert pair["status"] == "comparable"
    assert data["actions"] == [0, 1]


def test_scalar_free_energy_length_requires_bound_timestep_or_index() -> None:
    data = _categorical()
    data.update(
        free_energy=[3, 2],
        free_energy_kind="variational",
        free_energy_convention="negative_elbo",
    )
    pair = _pair(data, deepcopy(data))
    assert "free_energy" not in pair
    assert "posterior timesteps" in pair["unavailable_metrics"]["free_energy"]
    data.update(free_energy_index=[0, 1], free_energy_axes=["iteration"])
    assert _pair(data, deepcopy(data))["free_energy"]["within_tolerance"] is True


def _plot_data(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "jax": {"framework": "jax", "simulation_data": left},
        "pytorch": {"framework": "pytorch", "simulation_data": right},
    }


def _assert_pngs(paths: list[str]) -> None:
    assert paths
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        assert Path(path).stat().st_size > 1000


def test_real_gaussian_png_artists_use_means_and_covariance_without_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.analysis import viz_dashboard

    observed: list[dict[str, Any]] = []
    real_save = viz_dashboard.safe_savefig

    def inspect_and_save(path, **kwargs):
        figure = viz_dashboard.plt.gcf()
        observed.extend(
            {
                "ylabel": axis.get_ylabel(),
                "lines": [line.get_ydata().tolist() for line in axis.lines],
                "bands": len(axis.collections),
            }
            for axis in figure.axes
        )
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_dashboard, "safe_savefig", inspect_and_save)
    data = _plot_data(_gaussian(), _gaussian())
    pngs = viz_dashboard.generate_unified_framework_dashboard(
        data, tmp_path / "gaussian"
    )
    _assert_pngs(pngs)
    assert any(
        "Posterior mean" in axis["ylabel"] and axis["bands"] == 4 for axis in observed
    )
    assert any([-5.0, -4.0] in axis["lines"] for axis in observed)
    assert all(
        "Probability" not in axis["ylabel"]
        and "confidence" not in axis["ylabel"].lower()
        for axis in observed
    )
    assert (
        viz_dashboard.generate_confidence_comparison(data, tmp_path / "confidence.png")
        == []
    )
    receipt = json.loads((tmp_path / "confidence.admission.json").read_text())
    assert "Gaussian means" in receipt["excluded"]["nested/model-identity"][0]


def test_categorical_confidence_png_uses_validated_probabilities(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.analysis import viz_dashboard

    labels: list[str] = []
    real_save = viz_dashboard.safe_savefig

    def inspect_and_save(path, **kwargs):
        labels.extend(axis.get_ylabel() for axis in viz_dashboard.plt.gcf().axes)
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_dashboard, "safe_savefig", inspect_and_save)
    pngs = viz_dashboard.generate_confidence_comparison(
        _plot_data(_categorical(), _categorical()), tmp_path / "confidence.png"
    )
    _assert_pngs(pngs)
    assert labels == ["Maximum categorical probability", "Categorical entropy (nats)"]


@pytest.mark.parametrize("alteration", ["unbound", "different_model", "duplicate"])
def test_numeric_comparison_plots_refuse_unbound_or_overwritten_models(
    tmp_path: Path, alteration: str
) -> None:
    from gnn.analysis import viz_dashboard

    left, right = _categorical(), _categorical()
    data = _plot_data(left, right)
    if alteration == "unbound":
        left.pop("model_id")
        right.pop("model_id")
    elif alteration == "different_model":
        right.update(model_id="another/model", source_sha256="b" * 64)
    else:
        data["duplicate"] = {"framework": "jax", "simulation_data": deepcopy(left)}
    assert (
        viz_dashboard.generate_confidence_comparison(data, tmp_path / "confidence.png")
        == []
    )
    assert not (tmp_path / "confidence.png").exists()
    receipt = json.loads((tmp_path / "confidence.admission.json").read_text())
    assert receipt["excluded"]
    if alteration != "different_model":
        assert (
            viz_dashboard.generate_unified_framework_dashboard(
                data, tmp_path / "unified"
            )
            == []
        )


def test_current_schema_preserves_gaussian_and_source_semantics() -> None:
    from gnn.analysis.viz_schema import _current_schema_visualization_data

    data = _gaussian()
    data["schema_version"] = "rxinfer_simulation_v1"
    projected = _current_schema_visualization_data(data)
    assert projected["source_sha256"] == data["source_sha256"]
    assert projected["model_id"] == data["model_id"]
    assert projected["posterior_cov"] == data["posterior_cov"]
    assert projected["model_kind"] == "continuous"
    assert projected["inference_mode"] == "kalman_filter"


@pytest.mark.parametrize("efe", [[0.3, 0.2], [[0.3, 0.2], [0.2, 0.1], [0.1, 0.0]]])
def test_efe_png_does_not_average_or_truncate_unaligned_series(
    tmp_path: Path, efe
) -> None:
    from gnn.analysis.viz_dashboard import generate_efe_convergence_comparison

    data = _categorical()
    data.update(
        expected_free_energy=efe, expected_free_energy_convention="negative_value"
    )
    assert (
        generate_efe_convergence_comparison(
            _plot_data(data, deepcopy(data)), tmp_path / "efe.png"
        )
        == []
    )
    assert not (tmp_path / "efe.png").exists()
    receipt = json.loads((tmp_path / "efe.admission.json").read_text())
    assert receipt["excluded"]


def test_efe_png_positive_and_radar_claim_unavailable(tmp_path: Path) -> None:
    from gnn.analysis.viz_dashboard import (
        generate_efe_convergence_comparison,
        generate_framework_radar,
    )

    data = _categorical()
    data.update(
        expected_free_energy=[0.3, 0.2, 0.1],
        expected_free_energy_convention="negative_value",
    )
    _assert_pngs(
        generate_efe_convergence_comparison(
            _plot_data(data, deepcopy(data)), tmp_path / "efe.png"
        )
    )
    assert (
        generate_framework_radar(
            tmp_path / "missing_summary.json",
            _plot_data(data, deepcopy(data)),
            tmp_path / "radar.png",
        )
        == []
    )
    receipt = json.loads((tmp_path / "radar.admission.json").read_text())
    assert receipt["status"] == "unavailable" and "calibrated" in receipt["reason"]


def test_cached_mutable_views_cannot_bypass_public_comparison_or_plot_validation(
    tmp_path: Path,
) -> None:
    from gnn.analysis.viz_dashboard import generate_confidence_comparison

    data = _categorical()
    record = prepare_scientific_record(data)
    record["views"]["joint_state"]["beliefs"] = [[-1, 2]] * 3
    fresh = revalidate_scientific_record(record)
    assert "cached native views differ from validated payload" in fresh["reasons"]
    assert (
        compare_scientific_pair(record, prepare_scientific_record(_categorical()))[
            "status"
        ]
        == "not_comparable"
    )
    rows = {
        "jax": {"framework": "jax", "scientific_records": [record]},
        "pytorch": {
            "framework": "pytorch",
            "scientific_records": [prepare_scientific_record(_categorical())],
        },
    }
    assert generate_confidence_comparison(rows, tmp_path / "invalid.png") == []
    assert not (tmp_path / "invalid.png").exists()


def test_revalidation_never_erases_an_original_execution_binding_conflict() -> None:
    record = prepare_scientific_record(_categorical(), {"source_sha256": "b" * 64})
    assert "conflicting source_sha256 declarations" in record["reasons"]
    fresh = revalidate_scientific_record(record)
    assert "conflicting source_sha256 declarations" in fresh["reasons"]
    assert (
        compare_scientific_pair(record, prepare_scientific_record(_categorical()))[
            "status"
        ]
        == "not_comparable"
    )


@pytest.mark.parametrize("escape", ["model_name", "symlink_file", "symlink_directory"])
def test_analysis_never_imports_execution_results_outside_current_output_tree(
    tmp_path: Path, escape: str
) -> None:
    root, outside = tmp_path / "execute", tmp_path / "outside"
    result_file = outside / "jax" / "simulation_data" / "simulation_results.json"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(json.dumps(_categorical()))
    impl = root / "model" / "jax"
    impl.mkdir(parents=True)
    detail: dict[str, Any] = {
        "framework": "jax",
        "success": True,
        "model_name": "model",
        "execution_time": 1.0,
    }
    if escape == "model_name":
        detail["model_name"] = "../outside"
    elif escape == "symlink_file":
        simulation_dir = impl / "simulation_data"
        simulation_dir.mkdir()
        (simulation_dir / "simulation_results.json").symlink_to(result_file)
        detail["implementation_directory"] = str(impl)
    else:
        link = root / "alias"
        link.symlink_to(outside / "jax", target_is_directory=True)
        detail["implementation_directory"] = str(link)
    summaries = root / "summaries"
    summaries.mkdir()
    (summaries / "execution_summary.json").write_text(
        json.dumps({"execution_details": [detail]})
    )
    result = analyze_framework_outputs(root)
    assert result["metrics"]["total_executions"] == 1
    assert result["frameworks"]["jax"]["data_source"] is None
    assert result["frameworks"]["jax"]["scientific_records"] == []


@pytest.mark.parametrize(
    "summary_location", ["summaries", "root", "dangling_preferred"]
)
def test_execution_summary_symlinks_cannot_import_foreign_attempts(
    tmp_path: Path, summary_location: str
) -> None:
    root, outside = tmp_path / "execute", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    foreign = outside / "foreign_summary.json"
    foreign.write_text(
        json.dumps(
            {
                "execution_details": [
                    {
                        "framework": "outside_marker",
                        "success": True,
                        "execution_time": 2.0,
                    }
                ]
            }
        )
    )
    summary = root / "execution_summary.json"
    if summary_location in {"summaries", "dangling_preferred"}:
        # A valid root fallback must never hide a rejected preferred summary.
        summary.write_text(
            json.dumps(
                {
                    "execution_details": [
                        {
                            "framework": "fallback_marker",
                            "success": True,
                            "execution_time": 1.0,
                        }
                    ]
                }
            )
        )
        summary = root / "summaries" / "execution_summary.json"
        summary.parent.mkdir()
    summary.symlink_to(foreign)
    if summary_location == "dangling_preferred":
        foreign.unlink()
    result = analyze_framework_outputs(root)
    assert result["frameworks"] == {}
    assert result["metrics"] == {}
    assert (
        "within the current output tree"
        in result["unavailable_metrics"]["execution_summary"]
    )


def test_execution_summary_internal_symlink_retains_actual_attempts(
    tmp_path: Path,
) -> None:
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    current = summaries / "current.json"
    current.write_text(
        json.dumps(
            {
                "execution_details": [
                    {"framework": "jax", "success": True, "execution_time": 0.0}
                ]
            }
        )
    )
    (summaries / "execution_summary.json").symlink_to(current)
    result = analyze_framework_outputs(tmp_path)
    assert result["metrics"]["total_executions"] == 1
    assert result["frameworks"]["jax"]["success_count"] == 1


@pytest.mark.parametrize("axes", [["state", "timestep"], ["arbitrary"], "bad axes", []])
def test_equal_but_invalid_posterior_axis_declarations_never_admit(axes) -> None:
    data = _categorical()
    data["belief_axes"] = axes
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_unknown_schema_suffix_does_not_supply_canonical_axes() -> None:
    data = _categorical()
    data.pop("belief_axes")
    data["schema_version"] = "unknown_simulation_v1"
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_nonfinite_boolean_negative_timing_is_excluded_with_diagnostic(
    tmp_path: Path,
) -> None:
    details = [
        {"framework": "jax", "success": True, "execution_time": value}
        for value in (float("inf"), float("nan"), True, -1, 1.25)
    ]
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "execution_summary.json").write_text(
        json.dumps({"execution_details": details})
    )
    result = analyze_framework_outputs(tmp_path)
    assert result["metrics"]["total_successful"] == 5
    assert result["frameworks"]["jax"]["execution_times"] == [1.25]
    assert result["comparisons"]["performance_comparison"]["jax"]["mean"] == 1.25
    assert (
        result["comparisons"]["unavailable_metrics"]["jax"]["execution_time"][
            "invalid_count"
        ]
        == 4
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("index", [True, 3, [0, 1], [0, float("nan"), 2]])
def test_supplied_posterior_index_is_a_finite_aligned_per_row_series(index) -> None:
    data = _categorical()
    data["time_index"] = index
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_supplied_scalar_quantity_index_is_always_validated() -> None:
    data = _categorical()
    data.update(
        free_energy=[3, 2, 1],
        free_energy_kind="variational",
        free_energy_convention="negative_elbo",
        free_energy_index=[0, 1],
    )
    pair = _pair(data, deepcopy(data))
    assert "free_energy" not in pair
    assert "quantity-specific index" in pair["unavailable_metrics"]["free_energy"]


@pytest.mark.parametrize(
    "parameters",
    [{"F": [[float("inf")]]}, {"C": [float("inf"), 0]}, {"Q": [[-1, 0], [0, 1]]}],
)
def test_supplied_source_values_are_finite_and_covariances_psd(parameters) -> None:
    data = _gaussian()
    data["source_model_parameters"] = parameters
    assert _pair(data, deepcopy(data))["status"] == "not_comparable"


def test_operational_png_counts_all_attempts_and_preserves_missing_quantities(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.analysis import viz_dashboard

    observed: list[dict[str, Any]] = []
    real_save = viz_dashboard.safe_savefig

    def inspect_and_save(path, **kwargs):
        observed.extend(
            {
                "ylabel": axis.get_ylabel(),
                "heights": [patch.get_height() for patch in axis.patches],
                "texts": [text.get_text() for text in axis.texts],
            }
            for axis in viz_dashboard.plt.gcf().axes
        )
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_dashboard, "safe_savefig", inspect_and_save)
    data = {
        "jax_model": {
            "framework": "jax",
            "results": [
                {
                    "success": True,
                    "execution_time": 0.0,
                    "simulation_data": {"beliefs": [[0.5, 0.5]] * 2},
                },
                {
                    "success": False,
                    "execution_time": 4.0,
                    "simulation_data": {"beliefs": [[0.5, 0.5]] * 6},
                },
                {"success": True, "skipped": True, "execution_time": True},
                {"success": True, "status": "skipped", "execution_time": float("inf")},
            ],
        },
        "jax_sibling": {
            "framework": "jax",
            "results": [
                {
                    "success": True,
                    "execution_time": 2.0,
                    "simulation_data": {"beliefs": [[0.5, 0.5]] * 4},
                }
            ],
        },
        "pytorch_model": {"framework": "pytorch", "results": [{"success": False}]},
    }
    output = tmp_path / "operational.png"
    _assert_pngs([viz_dashboard.generate_cross_framework_comparison(data, output)])
    assert observed[0]["heights"] == [2.0]
    assert observed[1]["heights"] == [4.0]
    assert observed[0]["texts"] == ["Unavailable"]
    assert observed[1]["texts"] == ["Unavailable"]
    assert observed[2]["heights"] == [2 / 5, 0.0]
    assert observed[2]["texts"] == ["2/5\nskipped: 2", "0/1\nskipped: 0"]
    receipt = json.loads(output.with_suffix(".operational.json").read_text())
    jax = receipt["frameworks"]["jax"]
    assert (
        jax["total_count"],
        jax["success_count"],
        jax["skipped_count"],
        jax["failed_count"],
    ) == (5, 2, 2, 1)
    assert jax["execution_times"] == [0.0, 4.0, 2.0]
    assert jax["invalid_timing_count"] == 2
    assert jax["missing_steps_count"] == 2
    assert receipt["frameworks"]["pytorch"]["unavailable_metrics"] == {
        "execution_times": "No valid measured quantities supplied",
        "steps_completed": "No valid measured quantities supplied",
    }
    json.dumps(receipt, allow_nan=False)
