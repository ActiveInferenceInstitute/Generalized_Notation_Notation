"""Public partial-model render contracts reach saved native parameter consumers.

These tests construct parameters and diagrams only: no generated simulation,
inference, training or provider code is invoked. Dimension-based defaults are
compatibility behavior, distinct from the independently checked authored values.
"""

import ast
import importlib.util
import pickle

import numpy as np
import pytest

from gnn.render.jax import render_gnn_to_jax, render_gnn_to_jax_pomdp

RENDERERS = [render_gnn_to_jax, render_gnn_to_jax_pomdp]


def authored_values():
    return {
        "A": np.array([[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]]),
        "B": np.array([[[0.9, 0.4], [0.2, 0.7]], [[0.1, 0.6], [0.8, 0.3]]]),
        "C": np.array([-2.0, 0.75, 1.25]),
        "D": np.array([0.8, 0.2]),
    }


def compatibility_defaults():
    # Independent specified identity/zero/uniform defaults for 3 observations,
    # 2 states and 2 actions, written out rather than calling extractor helpers.
    return {
        "A": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "B": np.array([[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]),
        "C": np.array([0.0, 0.0, 0.0]),
        "D": np.array([0.5, 0.5]),
    }


def partial_model(omitted):
    return {
        "model_name": "Partial channel Ω",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 3,
            "num_actions": 2,
        },
        "initialparameterization": {
            key: value.tolist()
            for key, value in authored_values().items()
            if key not in omitted
        },
    }


def saved_parameter_literals(tree, renderer):
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == ("create_params" if renderer is render_gnn_to_jax else "create_pomdp_solver")
    )
    if renderer is render_gnn_to_jax:
        result = next(
            node.value for node in function.body if isinstance(node, ast.Return)
        )
        assert isinstance(result, ast.Dict)
        return {
            ast.literal_eval(key)[0]: np.array(ast.literal_eval(value.args[0]))
            for key, value in zip(result.keys, result.values)
        }
    return {
        node.targets[0].id: np.array(ast.literal_eval(node.value.args[0]))
        for node in function.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in "ABCD"
    }


def consume_saved_parameters(tmp_path, renderer, model, expected):
    before = pickle.dumps(model, protocol=5)
    output = tmp_path / "new" / "saved_model.py"
    ok, message, files = renderer(model, output)
    assert ok, message
    assert files == [str(output)]
    saved = output.read_bytes()
    tree = ast.parse(saved.decode("utf-8"))
    assert "Partial_channel_Ω" in ast.get_docstring(tree)
    literals = saved_parameter_literals(tree, renderer)
    assert set(literals) == set(expected)
    for key, values in expected.items():
        np.testing.assert_allclose(literals[key], values, rtol=0, atol=1e-12)

    descriptor = importlib.util.spec_from_file_location(
        "partial_native_parameters", output
    )
    assert descriptor is not None and descriptor.loader is not None
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    if renderer is render_gnn_to_jax:
        parameters = module.create_params()
        actual = {
            key: np.asarray(
                parameters[f"{key}_{'matrix' if key in 'AB' else 'vector'}"]
            )
            for key in expected
        }
        dimensions = (module.NUM_OBSERVATIONS, module.NUM_STATES, module.NUM_ACTIONS)
    else:
        solver = module.create_pomdp_solver()
        actual = {key: np.asarray(getattr(solver.models, key)) for key in expected}
        dimensions = (solver.num_observations, solver.num_states, solver.num_actions)
    assert dimensions == (3, 2, expected["B"].shape[2])
    for key, values in expected.items():
        assert actual[key].shape == values.shape
        np.testing.assert_allclose(actual[key], values, rtol=0, atol=1e-7)
    # Explicit axis oracles distinguish every state's conditional channel and
    # action slice; no inference calculation is needed to detect transposition.
    np.testing.assert_allclose(actual["A"].sum(axis=0), [1, 1], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        actual["B"].sum(axis=0), np.ones((2, expected["B"].shape[2])), rtol=0, atol=1e-7
    )
    assert pickle.dumps(model, protocol=5) == before
    assert output.read_bytes() == saved
    assert list(output.parent.glob("*.py")) == [output]
    assert not list(output.parent.glob("*.json"))


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize("omitted", ["A", "B", "C", "D", "ABCD"])
def test_partial_parameters_keep_each_authored_table_and_only_missing_defaults(
    tmp_path, renderer, omitted
):
    expected = authored_values()
    defaults = compatibility_defaults()
    for key in omitted:
        expected[key] = defaults[key]
    consume_saved_parameters(tmp_path, renderer, partial_model(omitted), expected)


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize("representation", ["list", "ndarray"])
def test_partial_passive_transition_adds_only_one_action_axis(
    tmp_path, renderer, representation
):
    model = partial_model("C")
    passive = np.array([[0.9, 0.2], [0.1, 0.8]])
    model["initialparameterization"]["B"] = (
        passive.tolist() if representation == "list" else passive.copy()
    )
    expected = authored_values()
    expected["B"] = passive[:, :, np.newaxis]
    expected["C"] = compatibility_defaults()["C"]
    consume_saved_parameters(tmp_path, renderer, model, expected)


def assert_refusal_preserves_output(tmp_path, renderer, model, diagnosis):
    before = pickle.dumps(model, protocol=5)
    sentinel = tmp_path / "existing.py"
    sentinel.write_bytes(b"# existing caller artifact\n")
    missing = tmp_path / "absent" / "never.py"
    for output in (sentinel, missing):
        ok, message, files = renderer(model, output)
        assert not ok and files == []
        assert diagnosis in message, message
    assert sentinel.read_bytes() == b"# existing caller artifact\n"
    assert not missing.parent.exists()
    assert pickle.dumps(model, protocol=5) == before


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize(
    "key,value,diagnosis",
    [
        ("A", [[0.7], [0.2], [0.1]], "B state axes"),
        ("B", [[[1.0], [0.0], [0.0]]] * 3, "B state axes"),
        ("C", [-2.0, 0.75], "C length"),
        ("D", [0.8, 0.1, 0.1], "D length"),
        ("C", [-2.0, float("inf"), 1.25], "finite"),
        ("C", [-2.0, 0.75j, 1.25], "real"),
        ("A", [[-0.7, 0.1], [0.2, 0.3], [1.5, 0.6]], "nonnegative"),
        ("D", [0.2, 0.2], "probability mass must be one"),
    ],
)
def test_partial_authored_incompatibility_refuses_before_writes(
    tmp_path, renderer, key, value, diagnosis
):
    # Omit a different table to exercise the documented partial compatibility
    # route, rather than duplicating complete canonical-model admission tests.
    model = partial_model("C" if key == "D" else "D")
    model["initialparameterization"][key] = value
    assert_refusal_preserves_output(tmp_path, renderer, model, diagnosis)


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize(
    "raw,diagnosis",
    [
        ("", "missing A, B, C, D"),
        (
            "A = [0.7,0.1; 0.2,0.3; 0.1,0.6]\nC = [-2,0.75,1.25]\nD = [0.8,0.2]",
            "missing B",
        ),
        (
            "A = [0.7,0.1; 0.2,0.3; 0.1,0.6]\nB = [0.9,0.2; 0.1,0.8]\nC = [-2,0.75,1.25]\nD = [0.8,0.2]",
            "B must be 3-D",
        ),
        (
            "A = [0.7,bad; 0.2; 0.1,0.6]\nB = [ ]\nC = [bad, ]\nD = [ ]",
            "B must be 3-D",
        ),
    ],
)
def test_raw_bracket_inputs_cannot_publish_an_incomplete_transition_model(
    tmp_path, renderer, raw, diagnosis
):
    # The existing square-bracket route cannot supply a 3-D B tensor. Its
    # parsing/recovery never licenses successful publication of that model.
    model = {"ModelName": "Raw source refusal", "InitialParameterization": raw}
    assert_refusal_preserves_output(tmp_path, renderer, model, diagnosis)


@pytest.mark.parametrize(
    "states,connections,domain,codomain,boxes,offsets",
    [
        (
            "s[2]\no[3]\nz[4]",
            "s > o\no -> z",
            (2,),
            (4,),
            [("s_to_o", (2,), (3,)), ("o_to_z", (3,), (4,))],
            [0, 0],
        ),
        (
            "s[2]\nt[3]\no[5]",
            "(s, t) > o",
            (2, 3),
            (5,),
            [("s_t_to_o", (2, 3), (5,))],
            [0],
        ),
        (
            "s[2]\no[3]\np[5]\nq[7]",
            "s > o\np - q",
            (2, 5),
            (3, 7),
            [("s_to_o", (2,), (3,)), ("p_to_q", (5,), (7,))],
            [0, 1],
        ),
        (
            "s[2,3,type=float]\no[5]",
            "# authored comment\ng = ExpectedFreeEnergy\ninvalid ~ edge\ns > o # valid edge",
            (2, 3),
            (5,),
            [("s_to_o", (2, 3), (5,))],
            [0],
        ),
    ],
    ids=["sequential", "tensor-domain", "independent-parallel", "annotations"],
)
def test_public_file_translation_builds_actual_tensor_diagrams_without_evaluation(
    tmp_path, states, connections, domain, codomain, boxes, offsets
):
    from discopy.tensor import Box, Diagram, Dim

    from gnn.render.discopy.translator import gnn_file_to_discopy_diagram

    source = tmp_path / "authored_diagram.md"
    source.write_text(
        f"## ModelName\nAuthored categorical Ω\n\n## StateSpaceBlock\n{states}\n\n## Connections\n{connections}\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    diagram = gnn_file_to_discopy_diagram(source)
    assert isinstance(diagram, Diagram)
    assert (diagram.dom, diagram.cod) == (Dim(*domain), Dim(*codomain))
    assert len(diagram.boxes) == len(boxes)
    for box, (name, box_dom, box_cod) in zip(diagram.boxes, boxes):
        assert isinstance(box, Box)
        assert (box.name, box.dom, box.cod) == (name, Dim(*box_dom), Dim(*box_cod))
        assert (
            box.data is None
        )  # Abstract categorical structure, no numeric evaluation.
    assert list(diagram.offsets) == offsets
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize(
    "case,content,diagnosis",
    [
        ("missing", None, "GNN file not found"),
        ("directory", None, "Error converting GNN file"),
        (
            "unstructured",
            "authored content without section headers",
            "No sections found",
        ),
        (
            "unknown-wire",
            "## StateSpaceBlock\ns[2]\n## Connections\ns > absent\n",
            "Unknown variable 'absent'",
        ),
        (
            "annotations-only",
            "## StateSpaceBlock\ns[2]\n## Connections\ng = ExpectedFreeEnergy\n",
            "No valid connections",
        ),
        (
            "invalid-dimension",
            "## StateSpaceBlock\ns[-2]\no[3]\n## Connections\ns > o\n",
            "Error creating DisCoPy Dim",
        ),
    ],
)
def test_public_file_translation_refusal_retains_authored_input_and_no_artifacts(
    tmp_path, caplog, case, content, diagnosis
):
    from gnn.render.discopy.translator import gnn_file_to_discopy_diagram

    source = tmp_path / "authored.md"
    if case == "directory":
        source.mkdir()
    elif content is not None:
        source.write_text(content, encoding="utf-8")
    before = source.read_bytes() if source.is_file() else None
    listing = set(tmp_path.iterdir())
    assert gnn_file_to_discopy_diagram(source) is None
    assert diagnosis in caplog.text
    assert set(tmp_path.iterdir()) == listing
    if before is not None:
        assert source.read_bytes() == before


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("A", [[0.7, 0.1], [0.2, 0.3], [0.1]]),
        ("A", [0.7, 0.2, 0.1]),
        ("A", "[[0.7,0.1],[0.2,0.3],[0.1,0.6]]"),
        ("B", [0.9, 0.1]),
        ("B", {"state": [0.9, 0.1]}),
        ("C", [[-2.0, 0.75, 1.25]]),
        ("C", None),
        ("D", [[0.8, 0.2]]),
        ("D", {"state": 0.8}),
    ],
    ids=[
        "ragged-A",
        "rank-A",
        "text-A",
        "rank-B",
        "mapping-B",
        "rank-C",
        "null-C",
        "rank-D",
        "mapping-D",
    ],
)
def test_present_malformed_partial_table_cannot_publish_dimension_defaults(
    tmp_path, renderer, key, value
):
    model = partial_model("C" if key == "D" else "D")
    model["initialparameterization"][key] = value
    assert_refusal_preserves_output(
        tmp_path, renderer, model, f"Invalid authored {key} parameter"
    )


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize("representation", ["tuple", "ndarray"])
@pytest.mark.parametrize("omitted", ["C", "D"])
def test_ordered_partial_numeric_containers_preserve_values_without_caller_mutation(
    tmp_path, renderer, representation, omitted
):
    model = partial_model(omitted)
    expected = authored_values()
    expected[omitted] = compatibility_defaults()[omitted]

    def ordered_tuple(value):
        return (
            tuple(ordered_tuple(item) for item in value)
            if isinstance(value, list)
            else value
        )

    for key in model["initialparameterization"]:
        values = expected[key]
        if representation == "tuple":
            model["initialparameterization"][key] = ordered_tuple(values.tolist())
        else:
            array = values.copy()
            array.flags.writeable = False
            model["initialparameterization"][key] = array
    consume_saved_parameters(tmp_path, renderer, model, expected)
    if representation == "ndarray":
        assert all(
            not value.flags.writeable
            for value in model["initialparameterization"].values()
        )


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize("collection", [None, ["A", "B"], "malformed"])
def test_present_nonmapping_partial_collection_cannot_become_all_defaults(
    tmp_path, renderer, collection
):
    model = partial_model("ABCD")
    model["initialparameterization"] = collection
    assert_refusal_preserves_output(
        tmp_path, renderer, model, "initialparameterization must be a dictionary"
    )


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
@pytest.mark.parametrize("key", ["A", "B", "C", "D"])
@pytest.mark.parametrize("invalid", ["nonfinite", "complex"])
def test_partial_array_admission_preserves_real_and_finite_refusals(
    tmp_path, renderer, key, invalid
):
    model = partial_model("C" if key == "D" else "D")
    value = authored_values()[key].astype(complex if invalid == "complex" else float)
    value.flat[0] = 1j if invalid == "complex" else np.inf
    value.flags.writeable = False
    model["initialparameterization"][key] = value
    assert_refusal_preserves_output(
        tmp_path, renderer, model, "real" if invalid == "complex" else "finite"
    )
    assert not value.flags.writeable


@pytest.mark.parametrize("renderer", RENDERERS, ids=["general", "pomdp"])
def test_partial_object_container_cannot_hide_complex_payoffs(tmp_path, renderer):
    model = partial_model("D")
    model["initialparameterization"]["C"] = np.array(
        [-2.0 + 1j, 0.75, 1.25], dtype=object
    )
    assert_refusal_preserves_output(tmp_path, renderer, model, "complex")
