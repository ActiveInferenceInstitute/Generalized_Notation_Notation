"""Real optional Gradio routes, stateful edits and independently reopened exports.

The explicit GUI lane provisions the frozen gui extra. Optional imports stay
inside consumers so ordinary core collection requires no Gradio distribution.
No browser, listening socket, provider or model execution is used.
"""

import asyncio
import importlib
import json
import logging
from contextlib import asynccontextmanager

import pytest

pytestmark = pytest.mark.env_heavy

MODEL = """# Authored native GUI model
## GNNSection
ActInfPOMDP
## ModelName
Native signed preference and structural zero model
## StateSpaceBlock
A[2,2,type=float] # likelihood
B[2,2,2,type=float] # transition tensor
C[2,type=float] # signed preferences
D[2,type=float] # prior
s[2,type=float] # hidden state
o[2,type=int] # observation
## Connections
D>s
s-A
A-o
s-B
## InitialParameterization
A={(1.0,0.0),(0.0,1.0)}
B={((1.0,0.25),(0.0,0.5)),((0.0,0.75),(1.0,0.5))}
C={(-1.5,2.0)}
D={(1.0,0.0)}
sensor_added=[-0.25,0.0,1.5]
## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
o=Observation
## ModelParameters
num_hidden_states: 2
num_obs: 2
num_actions: 2
## Footer
Authored native GUI consumer
"""

BUILDERS = {
    "gui1": ("gnn.gui.gui_1.ui", "build_gui"),
    "gui2": ("gnn.gui.gui_2.ui", "build_visual_gui"),
    "gui3": ("gnn.gui.gui_3.ui_designer", "build_design_studio"),
}


@pytest.fixture(autouse=True)
def offline_gui_environment(monkeypatch, tmp_path):
    """Disable optional telemetry before the first lazy import."""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-temp"))


class NativeEditor:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.session = "native-saved-consumer"

    async def call(self, name, data):
        matches = [d for d in self.config["dependencies"] if d["api_name"] == name]
        assert len(matches) == 1, (name, self.config["dependencies"])
        assert len(matches[0]["inputs"]) == len(data)
        response = await self.client.post(
            f"/editor/gradio_api/run/{name}",
            json={"data": data, "session_hash": self.session},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["is_generating"] is False
        assert len(result["data"]) == len(matches[0]["outputs"])
        return result["data"]


@asynccontextmanager
async def native_editor(kind, export, source=MODEL):
    import gradio as gr
    import httpx
    from fastapi import FastAPI

    module, builder = BUILDERS[kind]
    demo = getattr(importlib.import_module(module), builder)(
        source, export, logging.getLogger("native-gradio-consumer")
    )
    app = gr.mount_gradio_app(
        FastAPI(), demo, path="/editor", ssr_mode=False, mcp_server=False
    )
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://localhost"
            ) as client:
                response = await client.get("/editor/config")
                assert response.status_code == 200, response.text
                config = response.json()
                # A saved snapshot is reopened through a distinct JSON reader.
                snapshot = export.with_suffix(".config.json")
                snapshot.write_text(json.dumps(config), encoding="utf-8")
                assert json.loads(snapshot.read_text(encoding="utf-8")) == config
                # Gradio starts its in-process queue during ASGI lifespan, but
                # only launch() would create a listening server/local URL.
                assert demo.local_url is None
                assert not hasattr(demo, "server")
                yield NativeEditor(client, config)
    finally:
        demo.close()


def read_saved_model(path):
    from gnn.parsers import GNNParsingSystem
    from gnn.schema_validator import GNNParser

    assert path.is_file()
    syntax = GNNParser().parse_file(path)
    registry = GNNParsingSystem().parse_file(path)
    assert registry.success, registry.errors
    return syntax, registry.model


def table(rows, headers=None):
    return {
        "headers": headers or [f"column_{i}" for i in range(len(rows[0]))],
        "data": rows,
    }


@pytest.mark.parametrize(
    "kind,apis",
    [
        ("gui1", {"add_state", "update_state", "save_md"}),
        ("gui2", {"manual_update", "save_gnn", "switch_b_slice"}),
        ("gui3", {"preview_design", "export_design"}),
    ],
)
def test_real_builders_publish_native_controls_without_launch_or_export(
    tmp_path, kind, apis
):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor(kind, export) as editor:
            assert editor.config["components"]
            assert apis <= {d["api_name"] for d in editor.config["dependencies"]}
            assert any(c["type"] == "code" for c in editor.config["components"])
            assert not export.exists()

    asyncio.run(scenario())


def test_gui1_registered_edit_refresh_rename_save_reopens_in_two_readers(tmp_path):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor("gui1", export) as editor:
            added = await editor.call(
                "add_state",
                [MODEL, "pending_sensor", "1", "float", "saved café sensor"],
            )
            assert added[1] == ""
            refreshed = await editor.call("refresh_states", [added[0]])
            assert "pending_sensor" in [c[0] for c in refreshed[0]["choices"]]
            renamed = await editor.call(
                "update_state",
                [
                    added[0],
                    "pending_sensor",
                    "sensor_added",
                    "3",
                    "float",
                    "saved café sensor",
                ],
            )
            assert renamed[1] == ""
            saved = await editor.call("save_md", [renamed[0]])
            assert "✅" in saved[0]
            assert export.read_text(encoding="utf-8") == renamed[0]
            syntax, model = read_saved_model(export)
            assert "pending_sensor" not in syntax.variables
            assert syntax.variables["sensor_added"].dimensions == [3]
            assert syntax.variables["sensor_added"].description == "saved café sensor"
            assert syntax.parameters["sensor_added"] == [-0.25, 0.0, 1.5]
            sensors = [v for v in model.variables if v.name == "sensor_added"]
            assert len(sensors) == 1
            assert sensors[0].dimensions == [3]
            assert sensors[0].description == "saved café sensor"
            assert {p.name: p.value for p in model.parameters}["sensor_added"] == [
                -0.25,
                0.0,
                1.5,
            ]

    asyncio.run(scenario())


def test_gui1_native_invalid_dimension_and_oversize_save_preserve_prior_file(tmp_path):
    async def scenario():
        from gnn.gui.gui_1.markdown import MAX_MARKDOWN_CHARS

        export = tmp_path / "saved.md"
        async with native_editor("gui1", export) as editor:
            await editor.call("save_md", [MODEL])
            before = export.read_bytes()
            rejected = await editor.call(
                "add_state", [MODEL, "bad", "2,wat", "float", ""]
            )
            # Native Code preprocessing trims the submitted trailing newline.
            assert rejected[0] == MODEL.strip()
            assert "❌" in rejected[1]
            refused = await editor.call("save_md", ["x" * (MAX_MARKDOWN_CHARS + 1)])
            assert "❌" in refused[-1]
            assert export.read_bytes() == before
            read_saved_model(export)
            assert not list(tmp_path.glob("tmp*"))

    asyncio.run(scenario())


def matrix_inputs(*, invalid=False, slice_index=1):
    return [
        None,
        table([["bad", 0] if invalid else [1, 0], [0, 1]]),
        table([[0, 1], [1, 0]]),
        table([[-2.5], [0.0]]),
        table([[1.0], [0.0]]),
        slice_index,
    ]


def test_gui2_native_tensor_slice_edit_plots_and_save_preserve_other_actions(tmp_path):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor("gui2", export) as editor:
            edited = await editor.call("manual_update", matrix_inputs())
            validation = await editor.call("validate_editor_matrices", matrix_inputs())
            assert "Validation Passed" in validation[0]
            assert edited[7] == ""
            for plot in edited[1:5]:
                assert plot["type"] == "plotly"
                assert json.loads(plot["plot"])["data"]
            saved = await editor.call("save_gnn", [edited[6]])
            assert "✅" in saved[0]
            assert export.read_text(encoding="utf-8") == edited[6]
            syntax, model = read_saved_model(export)
            expected = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
            assert list(syntax.parameters["C"]) == [-2.5, 0.0]
            assert list(syntax.parameters["D"]) == [1.0, 0.0]
            assert syntax.variables["B"].dimensions == [2, 2, 2]
            parameters = {p.name: p.value for p in model.parameters}
            assert parameters["B"] == expected
            assert parameters["C"] == [[-2.5, 0.0]]
            assert parameters["D"] == [[1.0, 0.0]]

    asyncio.run(scenario())


def test_gui2_native_saved_signed_and_zero_vectors_have_valid_gnn_syntax(tmp_path):
    async def scenario():
        from gnn.extract.pomdp_extractor import extract_pomdp_from_file

        export = tmp_path / "saved.md"
        async with native_editor("gui2", export) as editor:
            edited = await editor.call("manual_update", matrix_inputs())
            await editor.call("save_gnn", [edited[6]])
            syntax, model = read_saved_model(export)
            assert list(syntax.parameters["C"]) == [-2.5, 0.0]
            assert list(syntax.parameters["D"]) == [1.0, 0.0]
            parameters = {p.name: p.value for p in model.parameters}
            assert parameters["C"] == [[-2.5, 0.0]]
            assert parameters["D"] == [[1.0, 0.0]]
            reopened = extract_pomdp_from_file(export, on_error="raise")
            assert reopened.C_vector == [-2.5, 0.0]
            assert reopened.D_vector == [1.0, 0.0]

    asyncio.run(scenario())


@pytest.mark.parametrize("edit_action", [False, True])
def test_gui2_native_noncubic_transition_planes_survive_independent_consumers(
    tmp_path, edit_action
):
    async def scenario():
        from gnn.extract.pomdp_extractor import extract_pomdp_from_file

        source = (
            MODEL.replace("B[2,2,2", "B[2,2,3")
            .replace(
                "B={((1.0,0.25),(0.0,0.5)),((0.0,0.75),(1.0,0.5))}",
                "B={((1.0,0.25,0.0),(0.0,0.5,1.0)),((0.0,0.75,1.0),(1.0,0.5,0.0))}",
            )
            .replace("num_actions: 2", "num_actions: 3")
        )
        untouched = "extra_tensor=[[[1,-2,3]],[[0,5,6]]]"
        source = source.replace(
            "## InitialParameterization\n",
            "## InitialParameterization\n" + untouched + "\n",
        ).replace(
            "## Connections\n", "extra_tensor[2,1,3,type=float]\n## Connections\n"
        )
        authored = tmp_path / "authored.md"
        authored.write_text(source, encoding="utf-8")
        before = authored.read_bytes()
        original = extract_pomdp_from_file(authored, on_error="raise")
        # Extractor containers may be tuples; the published numerical axes
        # and values are compared after ordinary JSON container conversion.
        original_b = json.loads(json.dumps(original.B_matrix))
        assert original_b == [
            [[1.0, 0.25, 0.0], [0.0, 0.5, 1.0]],
            [[0.0, 0.75, 1.0], [1.0, 0.5, 0.0]],
        ]
        export = tmp_path / "saved.md"
        async with native_editor(
            "gui2", export, authored.read_text(encoding="utf-8")
        ) as editor:
            values = {
                c["props"].get("label"): c["props"].get("value")
                for c in editor.config["components"]
            }
            initial = values["B Matrix Values - Current Action Slice"]
            assert initial["data"] == [[1.0, 0.0], [0.0, 1.0]]
            # The native slice selector persists the visible action 0 table.
            switched = await editor.call("switch_b_slice", [None, initial, 1])
            selected = switched[1]["value"]
            assert selected["data"] == [[0.25, 0.5], [0.75, 0.5]]
            if edit_action:
                selected = table([[0.0, 1.0], [1.0, 0.0]])
            edited = await editor.call(
                "manual_update",
                [
                    None,
                    values["A Matrix Values - Edit cells directly"],
                    selected,
                    values["C Values"],
                    values["D Values"],
                    1,
                ],
            )
            assert edited[7] == ""
            await editor.call("save_gnn", [edited[6]])
            syntax, model = read_saved_model(export)
            assert syntax.variables["B"].dimensions == [2, 2, 3]
            expected = (
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]]
                if edit_action
                else original_b
            )
            assert {p.name: p.value for p in model.parameters}["B"] == expected
            reopened = extract_pomdp_from_file(export, on_error="raise")
            assert json.loads(json.dumps(reopened.B_matrix)) == expected
            assert reopened.C_vector == [-1.5, 2.0]
            assert reopened.D_vector == [1.0, 0.0]
            assert reopened.A_matrix == original.A_matrix
            assert reopened.num_actions == 3
            assert untouched in export.read_text(encoding="utf-8")
            assert syntax.variables["extra_tensor"].dimensions == [2, 1, 3]
            assert authored.read_bytes() == before

    asyncio.run(scenario())


def test_gui2_native_invalid_table_retains_session_and_prior_saved_model(tmp_path):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor("gui2", export) as editor:
            edited = await editor.call("manual_update", matrix_inputs())
            await editor.call("save_gnn", [edited[6]])
            before = export.read_bytes()
            refused = await editor.call("manual_update", matrix_inputs(invalid=True))
            assert "non-numeric" in refused[7]
            assert "Validation failed" in refused[7]
            assert export.read_bytes() == before
            # A following genuine update uses the retained native session state.
            recovered = await editor.call("manual_update", matrix_inputs(slice_index=0))
            assert recovered[7] == ""
            assert "## InitialParameterization" in recovered[6]
            read_saved_model(export)

    asyncio.run(scenario())


def design_inputs():
    return [
        table(
            [
                ["s", "2", "hidden café state"],
                ["A", "2,2", "likelihood"],
                ["o", "2", "observation"],
            ],
            ["Variable", "Dimensions", "Description"],
        ),
        table(
            [
                ["s", "HiddenState", "state ontology"],
                ["A", "LikelihoodMatrix", "likelihood ontology"],
            ],
            ["Variable", "OntologyTerm", "Description"],
        ),
        "s-A\nA-o",
        2,
        2,
        2,
        2,
        "Bounded",
    ]


def test_gui3_native_design_preview_export_reopens_semantic_model(tmp_path):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor("gui3", export) as editor:
            preview = await editor.call("preview_design", design_inputs())
            assert "## StateSpaceBlock" in preview[0]
            assert not export.exists()
            saved = await editor.call("export_design", design_inputs())
            assert "✅" in saved[0]
            assert export.read_text(encoding="utf-8") == preview[0]
            syntax, model = read_saved_model(export)
            assert syntax.variables["A"].dimensions == [2, 2]
            assert syntax.variables["s"].description == "hidden café state"
            assert {(v.name, tuple(v.dimensions)) for v in model.variables} == {
                ("s", (2,)),
                ("A", (2, 2)),
                ("o", (2,)),
            }
            assert "s=HiddenState" in export.read_text(encoding="utf-8")
            assert "A=LikelihoodMatrix" in export.read_text(encoding="utf-8")
            assert "s-A\nA-o" in export.read_text(encoding="utf-8")
            assert {
                (m.variable_name, m.ontology_term) for m in model.ontology_mappings
            } == {("s", "HiddenState"), ("A", "LikelihoodMatrix")}
            assert {
                (tuple(c.source_variables), tuple(c.target_variables))
                for c in model.connections
            } == {(("s",), ("A",)), (("A",), ("o",))}

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["duplicate", "undefined_connection", "dimension"])
def test_gui3_native_rejected_design_preserves_previous_export(tmp_path, invalid):
    async def scenario():
        export = tmp_path / "saved.md"
        async with native_editor("gui3", export) as editor:
            await editor.call("export_design", design_inputs())
            before = export.read_bytes()
            inputs = design_inputs()
            expected = {
                "duplicate": "Duplicate",
                "undefined_connection": "undefined",
                "dimension": "dimension",
            }[invalid]
            if invalid == "duplicate":
                inputs[0]["data"].append(["s", "3", "duplicate"])
            elif invalid == "undefined_connection":
                inputs[2] = "s>missing"
            else:
                inputs[0]["data"][0][1] = "2,[3]"
            preview = await editor.call("preview_design", inputs)
            refused = await editor.call("export_design", inputs)
            assert "Error generating preview" in preview[0]
            assert expected.lower() in preview[0].lower()
            assert "❌ Export failed" in refused[0]
            assert expected.lower() in refused[0].lower()
            assert export.read_bytes() == before
            read_saved_model(export)
            assert not list(tmp_path.glob("tmp*"))

    asyncio.run(scenario())
