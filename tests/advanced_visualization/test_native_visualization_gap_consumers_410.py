"""Native saved-model plots and public D2 source generation controls.

D2 controls cover returned source, saved verbatim by the caller. They do not
invoke a D2 compiler or claim that a rendered SVG exists.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from gnn.advanced_visualization import D2Visualizer
from tests.advanced_visualization.test_native_network_consumers_410 import (
    assert_native_artifact,
    run_saved_model,
    saved_model,
)
from tests.advanced_visualization.test_native_network_consumers_410 import (
    figures as figures,
)
from tests.visualization.test_native_matrix_views_410 import native_png


@pytest.fixture(autouse=True)
def close_native_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("count", [2, 5])
def test_saved_policy_variable_displays_its_declared_uniform_prior(
    tmp_path, figures, count
):
    _, parsed, inventory = saved_model(
        tmp_path,
        [],
        variables=[{"name": "pi", "var_type": "policy", "dimensions": [count]}],
    )
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "pomdp")
    assert result is True
    assert (summary["successful"], summary["failed"], summary["skipped"]) == (1, 0, 1)
    policy, absent = sorted(summary["attempts"], key=lambda row: row["viz_type"])
    assert policy["viz_type"] == "policy" and policy["fallback_used"] is False
    assert absent["output_files"] == []
    path = output / "authored_network_policy_visualization.png"
    assert policy["output_files"] == [str(path)]
    native_png(path)
    axis = figures[path.name].axes[0]
    assert [bar.get_height() for bar in axis.patches] == pytest.approx(
        [1 / count] * count
    )
    assert axis.get_title() == "Policy Distribution: pi"
    assert axis.get_ylabel() == "Probability"


def test_saved_single_node_self_loop_retains_real_graph_counts(tmp_path, figures):
    _, parsed, inventory = saved_model(
        tmp_path,
        [("one", "one")],
        variables=[{"name": "one", "var_type": "hidden_state", "dimensions": [1]}],
    )
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "network")
    assert result is True
    path = assert_native_artifact(summary, output)
    graph, metrics = figures[path.name].axes
    assert [text.get_text() for text in graph.texts] == ["one"]
    values = dict(
        zip(
            [label.get_text() for label in metrics.get_xticklabels()],
            [bar.get_height() for bar in metrics.patches],
            strict=True,
        )
    )
    assert {
        key: values[key] for key in ["nodes", "edges", "density", "avg_clustering"]
    } == {
        "nodes": 1,
        "edges": 1,
        "density": 0,
        "avg_clustering": 0,
    }


def test_public_d2_structure_source_preserves_authored_ontology_and_arrows(tmp_path):
    model = {
        "model_name": "Fixture: semantic graph",
        "state_space": {
            "A": {"dimensions": [2, 3]},
            "C": {"dimensions": [3]},
            "s": {"dimensions": [2]},
            "o": {"dimensions": []},
            "u": {"dimensions": [1]},
            "pi": {"dimensions": [4]},
            "plain": {"dimensions": []},
        },
        "actinf_annotations": {
            "A": "LikelihoodMatrix",
            "C": "PreferenceVector",
            "s": "HiddenState",
            "o": "Observation",
            "u": "Action",
            "pi": "Policy",
        },
        "connections": [
            {
                "source": "o",
                "target": "s",
                "type": "->",
                "label": 'datum [0] {"quoted"}',
            },
            {"source": "s", "target": "u", "type": "<-"},
            {"source": "u", "target": "pi", "type": "<->"},
            {"source": "C", "target": "A", "type": "-"},
            {"source": "", "target": "plain", "type": "->"},
        ],
    }
    before = deepcopy(model)
    spec = D2Visualizer().generate_model_structure_diagram(model)
    path = tmp_path / f"{spec.name}.d2"
    path.write_text(spec.d2_content, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    assert model == before
    assert spec.metadata == {"model_name": model["model_name"], "type": "structure"}
    expected_nodes = {
        "A": ('"A [2×3]\\nLikelihoodMatrix"', "hexagon"),
        "C": ('"C [3]\\nPreferenceVector"', "diamond"),
        "s": ('"s [2]\\nHiddenState"', "cylinder"),
        "o": ("o\\nObservation", "circle"),
        "u": ('"u [1]\\nAction"', "square"),
        "pi": ('"pi [4]\\nPolicy"', "parallelogram"),
        "plain": ("plain", "rectangle"),
    }
    for name, (label, shape) in expected_nodes.items():
        assert f"    {name}: {label} {{\n      shape: {shape}\n    }}" in content
    assert 'state_space.o -> state_space.s: "datum [0] {\\"quoted\\"}"' in content
    assert "state_space.s <- state_space.u" in content
    assert "state_space.u <-> state_space.pi" in content
    assert "state_space.C -- state_space.A" in content
    assert "state_space. ->" not in content
    assert "- **A**: LikelihoodMatrix" in content


def test_public_d2_pomdp_source_preserves_matrix_and_tensor_dimensions(tmp_path):
    model = {
        "model_name": "Fixture POMDP",
        "state_space": {
            "A": {"dimensions": [3, 2], "description": 'sensor {"literal"}'},
            "B": {"dimensions": [2, 2, 4]},
            "D": {"dimensions": [2]},
            "E": {"dimensions": [1, 4]},
        },
    }
    before = deepcopy(model)
    spec = D2Visualizer().generate_pomdp_diagram(model, "fixture_action_axes")
    path = tmp_path / f"{spec.name}.d2"
    path.write_text(spec.d2_content, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    assert model == before and spec.metadata["type"] == "pomdp"
    assert 'A: "A [3×2]"' in content
    assert 'B: "B [2×2×4]"' in content
    assert 'tooltip: "sensor {\\"literal\\"}"' in content
    assert "D: " not in content and "E: " not in content
    assert "label: infer_states()" in content
    assert "label: infer_policies()" in content
    assert "label: sample_action()" in content
    assert not list(tmp_path.glob("*.svg"))


@pytest.mark.parametrize("include_frameworks", [False, True])
def test_public_d2_pipeline_source_has_requested_framework_scope(
    tmp_path, include_frameworks
):
    spec = D2Visualizer().generate_pipeline_flow_diagram(include_frameworks)
    path = tmp_path / "pipeline.d2"
    path.write_text(spec.d2_content, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    assert "parse -> validate -> export -> visualize" in content
    assert ("render -> execute" in content) is include_frameworks
    assert ("PyMDP, RxInfer.jl" in content) is include_frameworks
    assert (
        "generation -> analysis: Simulation results" in content
    ) is include_frameworks
    assert not list(tmp_path.glob("*.svg"))


def test_public_d2_framework_source_contains_only_requested_known_frameworks(tmp_path):
    frameworks = ["rxinfer", "pymdp"]
    spec = D2Visualizer().generate_framework_mapping_diagram(frameworks)
    path = tmp_path / "frameworks.d2"
    path.write_text(spec.d2_content, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    assert spec.metadata["frameworks"] == frameworks
    assert "label: Julia Reactive Inference" in content
    assert "label: Python Active Inference" in content
    assert "rxinfer_exec: RXINFER Simulation" in content
    assert "pymdp_exec: PYMDP Simulation" in content
    assert "jax_exec:" not in content and "discopy_exec:" not in content
    assert frameworks == ["rxinfer", "pymdp"]
