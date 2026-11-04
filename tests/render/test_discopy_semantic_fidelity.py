"""DisCoPy diagrams must encode the specification's declared wiring.

A diagram whose boxes do not match the GNN ``Connections`` section is a
semantic failure even when DisCoPy constructs it successfully. These tests
pin the supported generator to one box per declared connection, with the
box domain/codomain taken from the connection endpoints.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from discopy.monoidal import Ty

from gnn.render.discopy import render_gnn_to_discopy


def _load_module(path: Path) -> Any:
    descriptor = importlib.util.spec_from_file_location("discopy_wiring_consumer", path)
    assert descriptor is not None and descriptor.loader is not None
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return module


def _synthetic_spec() -> dict[str, Any]:
    return {
        "model_name": "Distinctive wiring",
        "variables": [
            {"name": "s", "dimensions": [2, 1]},
            {"name": "s_prime", "dimensions": [2, 1]},
            {"name": "o", "dimensions": [3, 1]},
            {"name": "u", "dimensions": [1, 1]},
        ],
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 3,
            "num_actions": 1,
        },
        "initial_parameterization": {
            "A": [[0.7, 0.2], [0.2, 0.5], [0.1, 0.3]],
        },
        "connections": [
            {"source": "D", "relation": ">", "target": "s"},
            {"source": "s", "relation": "-", "target": "A"},
            {"source": "A", "relation": ">", "target": "o"},
            {"source": "u", "relation": ">", "target": "s_prime"},
        ],
    }


def _expected_signatures(spec: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    labels = {">": "generates", "-": "couples", "=": "defines"}
    rows: list[tuple[str, str, str, str]] = []
    for connection in spec["connections"]:
        source = connection["source"]
        relation = connection["relation"]
        target = connection["target"]
        rows.append(
            (
                f"{source}__{labels[relation]}__{target}",
                source,
                target,
                relation,
            )
        )
    return rows


def test_connection_boxes_match_declared_wiring(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    path = tmp_path / "wiring.py"
    ok, message, warnings = render_gnn_to_discopy(spec, path)
    assert ok, message
    assert warnings == []
    module = _load_module(path)

    diagram, boxes, signatures = module.build_diagram()
    expected = _expected_signatures(spec)
    assert len(boxes) == len(spec["connections"]) == 4
    assert [box.name for box in boxes] == [row[0] for row in expected]
    assert [(sig["source"], sig["target"], sig["relation"]) for sig in signatures] == [
        ([row[1]], [row[2]], row[3]) for row in expected
    ]
    assert (boxes[0].dom, boxes[0].cod) == (Ty("D"), Ty("s"))
    assert (boxes[1].dom, boxes[1].cod) == (Ty("s"), Ty("A"))
    assert (boxes[2].dom, boxes[2].cod) == (Ty("A"), Ty("o"))
    assert (boxes[3].dom, boxes[3].cod) == (Ty("u"), Ty("s_prime"))
    assert {box.name for box in boxes}.isdisjoint(
        {"StateInf", "PolicyInf", "ActionSel"}
    )

    analysis = module.analyze_diagram(diagram, boxes, signatures)
    assert analysis["num_connections"] == 4
    assert analysis["num_boxes"] == 4
    assert analysis["connections"] == module.CONNECTIONS
    assert analysis["box_signatures"] == signatures
    assert analysis["diagram_domain"] == str(diagram.dom)
    assert analysis["diagram_codomain"] == str(diagram.cod)


def test_different_connection_sets_produce_different_diagrams(tmp_path: Path) -> None:
    first = _synthetic_spec()
    second = {
        "model_name": "Other wiring",
        "variables": [{"name": "x", "dimensions": [2, 1]}],
        "connections": [
            {"source": "x", "relation": ">", "target": "y"},
            {"source": "y", "relation": "-", "target": "Z"},
        ],
    }
    names: list[list[str]] = []
    for index, spec in enumerate((first, second)):
        path = tmp_path / f"diagram_{index}.py"
        ok, message, _ = render_gnn_to_discopy(spec, path)
        assert ok, message
        module = _load_module(path)
        _, boxes, _ = module.build_diagram()
        names.append([box.name for box in boxes])
    assert names[0] != names[1]
    assert len(names[0]) == len(first["connections"])
    assert len(names[1]) == len(second["connections"])


def test_actinf_pomdp_agent_has_one_box_per_declared_connection(tmp_path: Path) -> None:
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    spec = pomdp_to_gnn_spec(
        extract_pomdp_from_file(Path("input/gnn_files/discrete/actinf_pomdp_agent.md"))
    )
    assert len(spec["connections"]) == 11
    path = tmp_path / "actinf_discopy.py"
    ok, message, warnings = render_gnn_to_discopy(spec, path)
    assert ok, message
    assert warnings == []
    module = _load_module(path)
    _, boxes, signatures = module.build_diagram()
    assert len(boxes) == len(spec["connections"]) == 11

    # Every declared connection is represented exactly once, with its endpoints.
    def _wires(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        text = str(value).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return [part.strip() for part in text.split(",") if part.strip()]

    observed = [
        (signature["source"], signature["relation"], signature["target"])
        for signature in signatures
    ]
    expected = [
        (_wires(row["source"]), row["relation"], _wires(row["target"]))
        for row in spec["connections"]
    ]
    assert observed == expected
    assert len({box.name for box in boxes}) == 11


def test_missing_connections_refuses_and_preserves_existing_artifact(
    tmp_path: Path,
) -> None:
    spec = _synthetic_spec()
    spec["connections"] = []
    sentinel = tmp_path / "existing.py"
    sentinel.write_bytes(b"retained previous categorical diagram\n")
    for path in (sentinel, tmp_path / "absent" / "new.py"):
        ok, message, warnings = render_gnn_to_discopy(spec, path)
        assert not ok
        assert message.startswith("unsupported-discopy-connections:")
        assert "Connections" in message
        assert warnings == []
    assert sentinel.read_bytes() == b"retained previous categorical diagram\n"
    assert not (tmp_path / "absent").exists()


def test_malformed_connection_refuses_before_writing(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    spec["connections"] = [{"source": "", "relation": ">", "target": "s"}]
    path = tmp_path / "absent" / "new.py"
    ok, message, warnings = render_gnn_to_discopy(spec, path)
    assert not ok
    assert message.startswith("unsupported-discopy-connections:")
    assert warnings == []
    assert not path.parent.exists()


def test_generated_script_exports_connections_and_box_signatures(
    tmp_path: Path,
) -> None:
    spec = _synthetic_spec()
    path = tmp_path / "wiring.py"
    ok, message, _ = render_gnn_to_discopy(spec, path)
    assert ok, message
    result = subprocess.run(
        [sys.executable, path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads((tmp_path / "discopy_diagrams" / "circuit_info.json").read_text())
    assert info["connections"] == spec["connections"]
    assert info["analysis"]["num_connections"] == 4
    assert info["analysis"]["num_boxes"] == 4
    assert len(info["box_signatures"]) == 4
    assert [row["name"] for row in info["box_signatures"]] == [
        row[0] for row in _expected_signatures(spec)
    ]


def test_discopy_declares_its_explicit_connection_graph_contract() -> None:
    from gnn.render.framework_registry import (
        RENDER_DIMENSION_VALUES,
        supports_render_dimension,
    )

    assert "connection_graph" in RENDER_DIMENSION_VALUES
    assert supports_render_dimension("discopy", "connection_graph", "explicit")
    assert not supports_render_dimension("pymdp", "connection_graph", "explicit")
