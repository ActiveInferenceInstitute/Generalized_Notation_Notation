"""Scientific semantics and readable artifacts without native inference claims."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.thrml import (
    adapt_result,
    analyze_payload,
    generate_analysis_from_logs,
)
from tests.helpers.thrml_results import categorical_result


def test_sampling_metrics_bind_empirical_witness_and_preserve_identity() -> None:
    payload = categorical_result()
    summary = adapt_result(payload)
    assert summary["metrics"]["retained_samples"] == 4
    assert summary["source_identity"] == payload["source_identity"]
    assert summary["runtime_metadata"] == payload["runtime_metadata"]
    assert summary["inference_mode"] == "gibbs_monte_carlo"
    assert "mean_empirical_entropy_nats" in summary["metrics"]
    assert "variational_free_energy" not in summary["metrics"]
    assert "calibration" not in summary["metrics"]
    assert "convergence" in summary["unavailable_metrics"]


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("model_kind", "continuous", "unsupported"),
        ("schema_version", True, "integer"),
        ("inference_mode", "exact", "contract"),
        ("action_semantics", "policy_optimization", "contract"),
        ("sampling_dtype", "float32", "contract"),
        ("success", False, "contract"),
        ("num_timesteps", 3, "length"),
        ("beliefs", [[0.6, 0.4], [0.25, 0.75]], "replicate|sample witness"),
        ("beliefs", [[0.5, float("nan")], [0.25, 0.75]], "finite"),
        ("beliefs", [[-0.1, 1.1], [0.25, 0.75]], "probability"),
        ("sample_states", [[0, 2], [0, 1], [1, 1], [1, 1]], "integer indices"),
        ("sample_states", [[True, 0], [0, 1], [1, 1], [1, 1]], "Boolean"),
        (
            "sample_states",
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            "integer indices",
        ),
        ("predictive_observations", [[0.6, 0.4], [0.4, 0.6]], "replicate"),
        ("observations", [0], "shape"),
        ("transition_actions", [0, 1], "shape"),
        ("variational_free_energy", [0.0, 0.0], "verified mapping"),
        ("vfe_per_iteration", [0.0, 0.0], "verified mapping"),
        ("efe_per_action", [[0.0, 0.0], [0.0, 0.0]], "verified mapping"),
        ("posterior_cov", [[[1.0]]], "verified mapping"),
        (
            "validation",
            {"mapping_supported": False, "all_conditionals_valid": True},
            "validity",
        ),
        ("dependencies", {"thrml": "0.1.3"}, "dependency identity"),
    ],
)
def test_malformed_or_unverified_scientific_claims_fail(
    key: str, value: Any, error: str
) -> None:
    payload = categorical_result()
    payload[key] = value
    with pytest.raises(ValueError, match=error):
        adapt_result(payload)


@pytest.mark.parametrize("key", ["A", "B", "D"])
def test_structural_zeros_are_not_clipped_or_repaired(key: str) -> None:
    payload = categorical_result()
    if key == "A":
        payload["parameters"][key][0][0] = 0
    elif key == "B":
        payload["parameters"][key][0][0][0] = 0
    else:
        payload["parameters"][key][0] = 0
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="structural zeros unsupported"):
        adapt_result(payload)
    assert payload == before


@pytest.mark.parametrize(
    "field,value", [("num_samples", True), ("burn_in", -1), ("thin", 0), ("seed", 1.5)]
)
def test_sampling_provenance_requires_integral_valid_options(
    field: str, value: Any
) -> None:
    payload = categorical_result()
    payload["sampling"][field] = value
    with pytest.raises(ValueError, match="integer"):
        adapt_result(payload)


@pytest.mark.parametrize("digest", [None, "", "C" * 64, "c" * 63, "g" * 64])
def test_semantic_identity_requires_a_real_sha256_shape(digest: Any) -> None:
    payload = categorical_result()
    payload["source_identity"]["semantic_sha256"] = digest
    with pytest.raises(ValueError, match="lowercase SHA256"):
        adapt_result(payload)


def test_hash_identity_cannot_be_promoted_to_an_inference_proof() -> None:
    payload = categorical_result()
    payload["source_identity"]["claim"] = "exact inference proved"
    with pytest.raises(ValueError, match="bounded evidence claim"):
        adapt_result(payload)


def test_missing_float64_runtime_policy_cannot_verify_sampling() -> None:
    payload = categorical_result()
    payload["runtime_metadata"].pop("jax_enable_x64")
    with pytest.raises(ValueError, match="float64/x64"):
        adapt_result(payload)


def test_single_timestep_requires_no_transition_action() -> None:
    payload = categorical_result()
    payload["num_timesteps"] = 1
    payload["sampling"]["num_timesteps"] = 1
    for field in ("beliefs", "predictive_observations", "observations"):
        payload[field] = payload[field][:1]
    payload["sample_states"] = [[row[0]] for row in payload["sample_states"]]
    payload["transition_actions"] = []
    assert adapt_result(payload)["metrics"]["num_timesteps"] == 1


def test_same_shape_corruption_cannot_pass_through_matching_predictions() -> None:
    payload = categorical_result()
    payload["beliefs"][0] = [0.6, 0.4]
    payload["predictive_observations"][0] = [0.54, 0.46]
    with pytest.raises(ValueError, match="sample witness"):
        adapt_result(payload)


def test_optional_source_habits_are_not_fabricated() -> None:
    payload = categorical_result()
    payload["parameters"].pop("E")
    assert adapt_result(payload)["metrics"]["retained_samples"] == 4
    assert "E" not in payload["parameters"]


def _components() -> dict[str, Any]:
    payload = categorical_result()
    first, second = copy.deepcopy(payload), copy.deepcopy(payload)
    first["component_id"], second["component_id"] = "agent-a", "agent-b"
    for key in (
        "parameters",
        "beliefs",
        "predictive_observations",
        "observations",
        "transition_actions",
        "sample_states",
    ):
        payload.pop(key)
    payload.update(
        composition="independent_components",
        components={"agent-a": first, "agent-b": second},
    )
    return payload


def test_independent_components_keep_native_marginal_identity() -> None:
    payload = _components()
    summary = adapt_result(payload)
    assert set(summary["components"]) == {"agent-a", "agent-b"}
    assert summary["metrics"]["num_components"] == 2
    assert "beliefs" not in summary
    payload["components"]["agent-b"]["component_id"] = "agent-a"
    with pytest.raises(ValueError, match="identities"):
        adapt_result(payload)


@pytest.mark.parametrize(
    "claim", ["expected_free_energy", "posterior_cov", "converged"]
)
def test_component_dispatch_cannot_hide_unsupported_global_claims(claim: str) -> None:
    payload = _components()
    payload[claim] = [0.0]
    with pytest.raises(ValueError, match="verified mapping"):
        adapt_result(payload)


def _joint_result() -> dict[str, Any]:
    payload = categorical_result()
    payload["parameters"].update(
        A=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
        B=[[[0.25, 0.25] for _ in range(4)] for _ in range(4)],
        D=[0.25] * 4,
    )
    payload.update(
        composition="joint_categorical",
        sample_states=[[0, 0], [0, 1], [2, 3], [3, 3]],
        beliefs=[[0.5, 0.0, 0.25, 0.25], [0.25, 0.25, 0.0, 0.5]],
        predictive_observations=[[0.55, 0.45], [0.45, 0.55]],
        composition_metadata={
            "state_factors": [
                {"name": "first", "size": 2},
                {"name": "second", "size": 2},
            ],
            "observation_modalities": [{"name": "sensor", "size": 2}],
        },
        state_factor_marginals={
            "first": [[0.5, 0.5], [0.5, 0.5]],
            "second": [[0.75, 0.25], [0.25, 0.75]],
        },
        observation_modality_marginals={"sensor": [[0.55, 0.45], [0.45, 0.55]]},
        resource_admission={"estimated_bytes": 1024, "retained_matrix_entries": 46},
        source_model_parameters={"num_factors": 2, "num_states": 2},
        composed_model_parameters={"num_states": 4, "num_obs": 2, "num_actions": 2},
    )
    return payload


def test_joint_marginals_and_admission_remain_bound_and_readable(
    tmp_path: Path,
) -> None:
    payload = _joint_result()
    summary = analyze_payload(payload, tmp_path)
    assert summary["state_factor_marginals"] == payload["state_factor_marginals"]
    assert (
        summary["observation_modality_marginals"]
        == payload["observation_modality_marginals"]
    )
    assert summary["resource_admission"] == payload["resource_admission"]
    assert summary["source_model_parameters"] == payload["source_model_parameters"]
    assert summary["composed_model_parameters"] == payload["composed_model_parameters"]
    assert summary["metrics"]["num_states"] == 4
    stored = json.loads((tmp_path / "thrml_analysis.json").read_text())
    assert stored["source_model_parameters"]["num_states"] == 2
    assert stored["composed_model_parameters"]["num_states"] == 4
    assert len(summary["artifacts"]) == 5
    from PIL import Image

    for path in summary["artifacts"]:
        with Image.open(path) as image:
            image.verify()


@pytest.mark.parametrize(
    "field", ["state_factor_marginals", "observation_modality_marginals"]
)
def test_same_shape_marginal_corruption_cannot_be_relabelled_as_verified(
    field: str,
) -> None:
    payload = _joint_result()
    name = next(iter(payload[field]))
    payload[field][name][0] = [0.6, 0.4]
    with pytest.raises(ValueError, match="joint projection"):
        adapt_result(payload)


def test_named_marginal_cannot_be_verified_without_its_source_descriptor() -> None:
    payload = _joint_result()
    payload["composition_metadata"]["state_factors"] = []
    with pytest.raises(ValueError, match="source descriptors"):
        adapt_result(payload)


def test_readable_flat_and_composed_plots_preserve_named_components(
    tmp_path: Path,
) -> None:
    from PIL import Image

    for index, payload in enumerate((categorical_result(), _components())):
        summary = analyze_payload(payload, tmp_path / str(index))
        assert len(summary["artifacts"]) == 2 * (index + 1)
        for path in summary["artifacts"]:
            with Image.open(path) as image:
                image.verify()
        stored = json.loads((tmp_path / str(index) / "thrml_analysis.json").read_text())
        assert stored["source_identity"] == payload["source_identity"]


def test_analyzer_discovers_each_native_identity_once(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for name in ("first", "second"):
        payload = categorical_result()
        payload["source_identity"]["artifact_stem"] = name
        directory = root / name / "thrml" / "simulation_data"
        directory.mkdir(parents=True)
        (directory / "simulation_results.json").write_text(json.dumps(payload))
    paths = generate_analysis_from_logs(root, tmp_path / "analysis")
    assert len(paths) == len(set(paths)) == 2
    assert all(Path(path).is_file() for path in paths)


def test_analyzer_rejects_external_results_through_a_parent_link(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "simulation_results.json").write_text(json.dumps(categorical_result()))
    (model_dir / "simulation_data").symlink_to(external, target_is_directory=True)
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="unsafe"):
        generate_analysis_from_logs(root, output)
    assert not output.exists()


def test_ambiguous_historical_identity_fails_before_overwriting_analysis(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    payload = categorical_result()
    for folder in ("current", "historical"):
        directory = root / folder / "thrml" / "simulation_data"
        directory.mkdir(parents=True)
        (directory / "simulation_results.json").write_text(json.dumps(payload))
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="duplicate model artifact identity"):
        generate_analysis_from_logs(root, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value,error",
    [("num_timesteps", 999, "posterior length"), ("seed", 2**32, "uint32")],
)
def test_native_sampling_schedule_cannot_claim_a_different_run(
    field: str, value: int, error: str
) -> None:
    payload = categorical_result()
    payload["sampling"][field] = value
    with pytest.raises(ValueError, match=error):
        adapt_result(payload)


def test_independent_component_schedule_must_match_its_enclosing_receipt() -> None:
    payload = _components()
    payload["components"]["agent-b"]["sampling"]["seed"] += 1
    with pytest.raises(ValueError, match="metadata differs"):
        adapt_result(payload)


def test_direct_output_is_analyzed_without_importing_foreign_backend_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    directory = root / "direct_case" / "simulation_data"
    directory.mkdir(parents=True)
    (directory / "simulation_results.json").write_text(json.dumps(categorical_result()))
    foreign = root / "other_model" / "jax" / "simulation_data"
    foreign.mkdir(parents=True)
    (foreign / "simulation_results.json").write_text("not THRML JSON")
    reports = generate_analysis_from_logs(root, tmp_path / "analysis")
    assert len(reports) == 1
