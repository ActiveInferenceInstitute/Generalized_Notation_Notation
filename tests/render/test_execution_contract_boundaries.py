"""Public admission and lossless run-parameter regression controls (F1/F2)."""

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

import gnn.render as render
import gnn.render.jax as jax_render
from gnn.execute.jax.semantic_runner import run_contract
from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.execution_contracts import CONTRACTS, contract_payload
from gnn.render.pomdp_processor import pomdp_to_gnn_spec

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "hierarchical/hierarchical_pomdp",
    "hierarchical/temporal_hierarchy",
    "discrete/tmaze_epistemic",
)
GENERATORS = tuple(name for name in render.__all__ if name.startswith("generate_"))
JAX_VARIANTS = tuple(jax_render.__all__)
INVALID_COUNTS = (
    5.9,
    1.9,
    5.0,
    True,
    False,
    float("nan"),
    float("inf"),
    -1,
    0,
    "5",
    None,
)
INVALID_SEEDS = (1.9, 1.0, True, False, float("nan"), float("inf"), -1, "1", None)


@pytest.fixture(scope="module")
def source_specs():
    return [
        pomdp_to_gnn_spec(
            extract_pomdp_from_file(
                ROOT / "input/gnn_files" / (model + ".md"), on_error="raise"
            )
        )
        for model in MODELS
    ]


def factor_spec(contract=None):
    matrices = {
        f"{key}_f{i}": value
        for i in range(2)
        for key, value in {
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "B": [[1.0, 0.0], [0.0, 1.0]],
            "C": [0.0, 0.0],
            "D": [0.5, 0.5],
        }.items()
    }
    parameters = {"b_tensor_order": "next_state_previous_state_action"}
    if contract is not None:
        parameters["execution_contract"] = contract
    return {
        "name": "boundary_probe",
        "model_parameters": parameters,
        "structured_pomdp": {"matrices": matrices},
    }


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("index", range(3))
def test_exported_generators_refuse_exact_semantic_sources(
    name, index, source_specs, tmp_path
):
    with pytest.raises(ValueError, match="unsupported-execution-contract"):
        getattr(render, name)(deepcopy(source_specs[index]), tmp_path / "refused.py")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", GENERATORS)
def test_exported_generators_refuse_unknown_before_write(name, tmp_path):
    target = tmp_path / "existing.py"
    target.write_text("existing artifact")
    with pytest.raises(ValueError, match="unsupported-execution-contract"):
        getattr(render, name)(factor_spec("future_contract_v99"), target)
    assert target.read_text() == "existing artifact"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", JAX_VARIANTS)
@pytest.mark.parametrize("contract", (*CONTRACTS, "future_contract_v99"))
def test_every_jax_variant_rejects_unknown_or_incompatible_factor_contract(
    name, contract, tmp_path
):
    ok, reason, files = getattr(jax_render, name)(
        factor_spec(contract), tmp_path / "nested/model.py"
    )
    assert not ok and not files
    assert (
        "unsupported-execution-contract"
        if contract not in CONTRACTS
        else "Missing execution contract parameter"
    ) in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", JAX_VARIANTS)
@pytest.mark.parametrize("index", range(3))
def test_every_jax_variant_delegates_valid_sources(name, index, source_specs, tmp_path):
    target = tmp_path / "model.py"
    ok, reason, files = getattr(jax_render, name)(deepcopy(source_specs[index]), target)
    assert ok, reason
    assert files == [str(target)]
    # Inspect the serialized data without executing a model on import.
    tree = ast.parse(target.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign))
    payload = json.loads(ast.literal_eval(assignment.value.args[0]))
    assert payload == contract_payload(source_specs[index])
    assert "semantic_runner import run_contract" in target.read_text()


def test_ordinary_factorized_behavior_remains(tmp_path):
    target = tmp_path / "factor.py"
    ok, reason, files = jax_render.render_gnn_to_jax_factorized(factor_spec(), target)
    assert ok, reason
    assert files == [str(target)]
    assert "run_factorized_active_inference" in target.read_text()
    assert "semantic_runner" not in target.read_text()


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("mode", ("pipeline", "standalone"))
def test_exported_pymdp_class_refuses_semantic_sources(
    index, mode, source_specs, tmp_path
):
    ok, reason, files = render.PyMDPRenderer({"mode": mode}).render_spec(
        deepcopy(source_specs[index]), tmp_path / "nested/model.py"
    )
    assert not ok and not files and "unsupported-execution-contract" in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "index,key",
    (
        (0, "num_timesteps"),
        (1, "num_fast_transitions"),
        (1, "num_timesteps"),
        (2, "num_timesteps"),
    ),
)
@pytest.mark.parametrize("value", INVALID_COUNTS)
def test_source_payload_and_override_counts_fail_closed(
    index, key, value, source_specs, tmp_path
):
    s = deepcopy(source_specs[index])
    payload = contract_payload(s)
    s["model_parameters"][key] = value
    ok, reason, files = jax_render.render_gnn_to_jax(s, tmp_path / "source.py")
    assert not ok and not files and key in reason
    with pytest.raises(ValueError, match=key):
        contract_payload(s)
    payload["parameters"][key] = value
    with pytest.raises(ValueError, match=key):
        run_contract(json.loads(json.dumps(payload)))
    ok, reason, files = jax_render.render_gnn_to_jax(
        source_specs[index], tmp_path / "override.py", {key: value}
    )
    assert not ok and not files and key in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("value", INVALID_SEEDS)
def test_source_payload_and_override_seeds_fail_closed(
    index, value, source_specs, tmp_path
):
    s = deepcopy(source_specs[index])
    payload = contract_payload(s)
    s["model_parameters"]["seed"] = value
    ok, reason, files = jax_render.render_gnn_to_jax(s, tmp_path / "source.py")
    assert not ok and not files and "seed" in reason
    payload["parameters"]["seed"] = value
    with pytest.raises(ValueError, match="seed"):
        run_contract(json.loads(json.dumps(payload)))
    ok, reason, files = jax_render.render_gnn_to_jax(
        source_specs[index], tmp_path / "override.py", {"seed": value}
    )
    assert not ok and not files and "seed" in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", (1, 4, 6, 19))
def test_incomplete_blocks_refused_at_source_execution_and_after_override(
    value, source_specs, tmp_path
):
    s = deepcopy(source_specs[0])
    payload = contract_payload(s)
    s["model_parameters"]["num_timesteps"] = value
    ok, reason, files = jax_render.render_gnn_to_jax(s, tmp_path / "source.py")
    assert not ok and not files and "complete five-observation blocks" in reason
    payload["parameters"]["num_timesteps"] = value
    with pytest.raises(ValueError, match="complete five-observation blocks"):
        run_contract(json.loads(json.dumps(payload)))
    ok, reason, files = jax_render.render_gnn_to_jax(
        source_specs[0], tmp_path / "override.py", {"num_timesteps": value}
    )
    assert not ok and not files and "complete five-observation blocks" in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "index,key,count",
    ((0, "num_timesteps", 5), (1, "num_fast_transitions", 1), (2, "num_timesteps", 3)),
)
def test_valid_minimal_counts_and_zero_seed_execute_without_coercion(
    index, key, count, source_specs
):
    payload = contract_payload(source_specs[index])
    payload["parameters"].update({key: count, "seed": 0})
    result = run_contract(json.loads(json.dumps(payload)))
    assert result["parameters"][key] == count
    assert result["parameters"]["seed"] == 0
    if index == 0:
        assert result["num_observations"] == count
    elif index == 1:
        assert result["num_fast_transitions"] == count


@pytest.mark.parametrize("index", range(3))
def test_exported_pymdp_file_entry_refuses_exact_sources(index, tmp_path):
    source = ROOT / "input/gnn_files" / (MODELS[index] + ".md")
    ok, reason = render.PyMDPRenderer().render_file(
        source, tmp_path / "nested/model.py"
    )
    assert not ok and "unsupported-execution-contract" in reason
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("contract", (*CONTRACTS, "future_contract_v99"))
def test_exported_pymdp_class_cannot_hide_contract_in_flat_spec(contract, tmp_path):
    from tests.render.test_jax_renderer import _jax_spec

    s = _jax_spec()
    s["model_parameters"]["execution_contract"] = contract
    ok, reason, files = render.PyMDPRenderer().render_spec(
        s, tmp_path / "nested/model.py"
    )
    assert not ok and not files and "unsupported-execution-contract" in reason
    assert list(tmp_path.iterdir()) == []
