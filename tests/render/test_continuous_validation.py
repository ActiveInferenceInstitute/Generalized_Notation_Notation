"""Invalid scientific Gaussian parameters fail before backend code generation."""

from copy import deepcopy

import pytest

from gnn.render.continuous_common import extract_continuous_spec

BASE = {
    "initialparameterization": {
        "F": [[1.0, 0.0], [0.0, 1.0]],
        "H": [[1.0, 0.0]],
        "Q": [[0.1, 0.0], [0.0, 0.1]],
        "R": [[0.2]],
        "prior_mean": [0.0, 1.0],
        "prior_cov": [[1.0, 0.0], [0.0, 1.0]],
    },
    "model_parameters": {"num_timesteps": 2, "dt": 0.1},
}


@pytest.mark.parametrize(
    "key,value",
    [
        ("F", []),
        ("F", [1.0, 2.0]),
        ("H", []),
        ("Q", [[0.1, 0.2], [0.0, 0.1]]),
        ("Q", [[-0.1, 0.0], [0.0, 0.1]]),
        ("R", [[0.0]]),
        ("prior_cov", [[1.0, 1.0], [1.0, 1.0]]),
        ("prior_mean", [0.0, float("inf")]),
        ("goal_mean", [1.0, 2.0]),
    ],
)
def test_invalid_parameter_block_rejected(key: str, value: object) -> None:
    spec = deepcopy(BASE)
    spec["initialparameterization"][key] = value
    with pytest.raises(ValueError):
        extract_continuous_spec(spec)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_timesteps", 0),
        ("num_timesteps", 2.2),
        ("dt", float("nan")),
        ("dt", 0.0),
        ("random_seed", -1),
    ],
)
def test_invalid_runtime_parameters_rejected(key: str, value: object) -> None:
    spec = deepcopy(BASE)
    spec["model_parameters"][key] = value
    with pytest.raises(ValueError):
        extract_continuous_spec(spec)


@pytest.mark.parametrize("key", ["num_timesteps", "dt", "random_seed"])
def test_boolean_scientific_runtime_scalar_rejected(key: str) -> None:
    spec = deepcopy(BASE)
    spec["model_parameters"][key] = True
    with pytest.raises(ValueError, match="boolean"):
        extract_continuous_spec(spec)


def test_boolean_control_gain_rejected() -> None:
    spec = deepcopy(BASE)
    spec["initialparameterization"].update(goal_mean=[1.0, 1.0], control_gain=[True])
    with pytest.raises(ValueError, match="boolean"):
        extract_continuous_spec(spec)


@pytest.mark.parametrize("value", ["1.5", "0", "-1"])
def test_extraction_rejects_runtime_scalar_before_lossy_integer_projection(
    tmp_path, value: str
) -> None:
    from pathlib import Path

    from gnn.extract.pomdp_extractor import extract_pomdp_from_file

    source = Path("input/gnn_files/continuous/continuous_navigation.md").read_text()
    invalid = tmp_path / "invalid_continuous.md"
    invalid.write_text(source.replace("num_timesteps: 15", f"num_timesteps: {value}"))
    result, errors = extract_pomdp_from_file(
        invalid, strict_validation=True, on_error="collect"
    )
    assert result is None
    assert any("positive integer" in error.message for error in errors)
