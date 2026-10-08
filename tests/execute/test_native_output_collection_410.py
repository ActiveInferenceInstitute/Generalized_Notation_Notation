"""Public saved-output consumers preserve authored native files and trace values.

These are file-consumption witnesses, not native Julia/optional-framework inference.
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import pytest
from PIL import Image

from gnn.execute.data_extractors import extract_simulation_data_from_files
from gnn.execute.processor import collect_execution_outputs, execute_single_script

LOGGER = logging.getLogger(__name__)


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


@pytest.mark.parametrize(
    ("framework", "folder"),
    [
        ("pymdp", "output/pymdp_simulations"),
        ("discopy", "discopy_diagrams"),
        ("jax", "jax_outputs"),
        ("numpyro", "numpyro_outputs"),
        ("pytorch", "pytorch_outputs"),
        ("ngclearn", "ngclearn_outputs"),
    ],
)
def test_saved_collection_keeps_distinct_sibling_outputs_and_replaces_stale_data(
    tmp_path, framework, folder
):
    script = save(tmp_path / "render" / framework / "authored.py", "print('saved')")
    source = script.parent / folder
    expected = [
        save(source / "left" / "result.json", '{"source":"left","value":0.25}'),
        save(source / "right" / "result.json", '{"source":"right","value":0.875}'),
        save(source / "posterior_trace.json", '{"beliefs":[[0.8,0.2],[0.4,0.6]]}'),
    ]
    Image.new("RGB", (12, 12), "navy").save(save(source / "plot.png", b""))
    if framework == "pymdp":
        expected.append(save(source / "sample.pkl", pickle.dumps({"action": [1, 0]})))
    if framework in ["numpyro", "pytorch", "ngclearn"]:
        expected.append(save(source / "energy_trace.csv", "t,value\n0,-2.0\n1,0.5\n"))
    output = tmp_path / "execute"
    stale = save(output / "simulation_data" / "old" / "stale.json", '{"old":true}')
    stale_trace = save(output / "traces" / "stale.csv", "old")
    before = {path: path.read_bytes() for path in expected}

    collected = collect_execution_outputs(script, output, framework, LOGGER)

    actual = [Path(path) for paths in collected.values() for path in paths]
    assert len(actual) == len(expected) and len(set(actual)) == len(expected)
    assert sorted(path.read_bytes() for path in actual) == sorted(before.values())
    assert {path: path.read_bytes() for path in before} == before
    assert not stale.exists() and not stale_trace.exists()
    assert len(collected["simulation_data"]) == 2 + (framework == "pymdp")
    assert len(collected["traces"]) == 1 + (
        framework in ["numpyro", "pytorch", "ngclearn"]
    )
    assert collected["visualizations"] == []
    assert (source / "plot.png").exists()
    assert all(path.is_relative_to(output) for path in actual)


def test_activeinference_collection_preserves_nested_data_and_energy_trace_bytes(
    tmp_path,
):
    script = save(tmp_path / "render" / "activeinference_jl" / "model.jl", "# saved")
    native = script.parent / "activeinference_outputs_saved"
    files = [
        save(script.parent / "simulation_results.json", '{"source":"root"}'),
        save(native / "simulation_data" / "beliefs.json", '{"beliefs":[[0.2,0.8]]}'),
        save(native / "simulation_data" / "observations.csv", "t,obs\n0,2\n"),
        save(native / "free_energy_traces" / "energy_trace.csv", "t,F\n0,-0.75\n"),
    ]
    Image.new("RGB", (12, 12), "navy").save(
        save(native / "visualizations" / "belief.png", b"")
    )
    result = collect_execution_outputs(
        script, tmp_path / "execute", "activeinference_jl", LOGGER
    )
    copies = [Path(path) for paths in result.values() for path in paths]
    assert sorted(path.read_bytes() for path in copies) == sorted(
        path.read_bytes() for path in files
    )
    assert len(result["simulation_data"]) == 3 and len(result["traces"]) == 1
    assert result["visualizations"] == []


def test_saved_activeinference_results_keep_latest_csv_axes_and_shared_science(
    tmp_path,
):
    output = tmp_path / "execute"
    old = output / "activeinference_outputs_old"
    latest = output / "activeinference_outputs_latest"
    old.mkdir(parents=True)
    save(old / "model_parameters.json", '{"model_name":"obsolete"}')
    params = {
        "model_name": "authored",
        "n_states": 2,
        "n_observations": 3,
        "n_actions": 4,
        "timestamp": "saved-source-time",
    }
    save(latest / "model_parameters.json", json.dumps(params))
    save(
        latest / "simulation_results.csv",
        "timestep,observation,action,belief_0,belief_1\n0,2,3,0.8,0.2\n1,0,1,0.4,0.6\n",
    )
    save(latest / "summary.txt", "Probability and action validation: PASSED\n")
    Image.new("RGB", (12, 12), "navy").save(latest / "belief.png")
    os.utime(old, (100, 100))
    os.utime(latest, (200, 200))
    science = {
        "schema_version": "1.0",
        "beliefs": [[0.8, 0.2], [0.4, 0.6]],
        "observations": [2, 0],
        "actions": [3, 1],
        "true_states": [0, 1],
        "expected_free_energy": [-0.75, 0.25],
        "expected_free_energy_convention": "minimize_cost",
        "policy_posterior": [[0.1, 0.9]],
        "validation": {"all_beliefs_valid": True},
        "model_parameters": {"num_states": 2},
    }
    shared = save(
        output / "simulation_data" / "simulation_results.json", json.dumps(science)
    )
    trace = save(output / "free_energy_traces" / "saved.csv", "t,F\n0,-0.75\n")
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}

    result = extract_simulation_data_from_files(output, "activeinference_jl", LOGGER)

    assert {path: path.read_bytes() for path in before} == before
    assert all(result[key] == value for key, value in params.items())
    assert all(result[key] == value for key, value in science.items())
    assert result["free_energy"] == [-0.75, 0.25]
    assert result["timesteps"] == 2 and result["validation_passed"] is True
    assert result["visualization_count"] == 1 and result["visualization_files"] == [
        "belief.png"
    ]
    assert result["free_energy_trace_files"] == [trace.name]
    assert shared.read_bytes() == before[shared]


def test_saved_activeinference_bad_parameters_do_not_erase_valid_csv_and_diagnostics(
    tmp_path, caplog
):
    native = tmp_path / "activeinference_outputs_saved"
    save(native / "model_parameters.json", "{broken")
    save(
        native / "simulation_results.csv",
        "observation,action,belief_0,belief_1\n2,1,0.25,0.75\n",
    )
    save(native / "summary.txt", "Validation FAILED\n")
    with caplog.at_level(logging.WARNING):
        result = extract_simulation_data_from_files(
            tmp_path, "activeinference_jl", LOGGER
        )
    assert result["beliefs"] == [[0.25, 0.75]] and result["actions"] == [1]
    assert result["observations"] == [2] and result["validation_passed"] is False
    assert "model_name" not in result
    assert "Error reading model_parameters.json" in caplog.text


def test_saved_discopy_analysis_preserves_authored_circuit_components_and_parameters(
    tmp_path,
):
    payload = {
        "circuit": "sensor >> inference",
        "components": ["sensor", "inference"],
        "analysis": {"rank": 2},
        "parameters": {"gain": -0.25},
    }
    source = save(
        tmp_path / "simulation_data" / "circuit_analysis.json", json.dumps(payload)
    )
    Image.new("RGB", (12, 12), "navy").save(
        save(tmp_path / "diagram_outputs" / "circuit.png", b"")
    )
    result = extract_simulation_data_from_files(tmp_path, "discopy", LOGGER)
    assert all(result[key] == value for key, value in payload.items())
    assert result["diagram_count"] == 1 and result["diagram_files"] == ["circuit.png"]
    assert json.loads(source.read_text()) == payload


def test_public_single_script_execution_collects_real_child_output_and_trace_values(
    tmp_path,
):
    script = save(
        tmp_path / "render" / "sensor" / "python" / "native.py",
        "from pathlib import Path\nimport json\n"
        "Path('simulation_results.json').write_text(json.dumps({'observations':[2,0],'actions':[1,0],'beliefs':[[0.8,0.2],[0.4,0.6]]}))\n"
        "Path('energy_trace.csv').write_text('t,F\\n0,-0.75\\n1,0.25\\n')\n"
        "print('authored finite trace [0.25, 0.75]')\n",
    )
    result = execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "python",
            "executor": sys.executable,
        },
        tmp_path / "execute",
        False,
        LOGGER,
        timeout=10,
    )
    assert result["success"] is True and result["return_code"] == 0, result
    assert result["cleanup_verified"] is True
    collected = result["collected_outputs"]
    assert len(collected["simulation_data"]) == len(collected["traces"]) == 1
    assert json.loads(Path(collected["simulation_data"][0]).read_text()) == {
        "observations": [2, 0],
        "actions": [1, 0],
        "beliefs": [[0.8, 0.2], [0.4, 0.6]],
    }
    assert Path(collected["traces"][0]).read_text() == "t,F\n0,-0.75\n1,0.25\n"
    assert result["simulation_data"]["arrays"] == ["0.25, 0.75"]
    persisted = json.loads(Path(result["structured_result_file"]).read_text())
    assert persisted["collected_outputs"] == collected
