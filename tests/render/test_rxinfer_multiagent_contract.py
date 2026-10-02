"""Provides helper functions: test_rxinfer_compact_multiagent_keys_drive_agent_count, test_rxinfer_nr_agents_does_not_silently_fallback_to_four_defaults, test_rxinfer_clustered_topology_is_preserved_in_config, test_rxinfer_topology_rejects_unknown_edge_endpoint, and 16 more.

Public functions: test_rxinfer_compact_multiagent_keys_drive_agent_count, test_rxinfer_nr_agents_does_not_silently_fallback_to_four_defaults, test_rxinfer_clustered_topology_is_preserved_in_config, test_rxinfer_topology_rejects_unknown_edge_endpoint, test_rxinfer_topology_rejects_unknown_cluster_member, test_rxinfer_topology_rejects_partial_edge_record, test_rxinfer_topology_rejects_non_list_cluster_members, test_rxinfer_accepts_underscored_initial_parameterization, test_rxinfer_string_topology_edges_render_valid_toml, test_rxinfer_execution_metadata_reads_toml_topology, test_rxinfer_execution_metadata_ignores_unmatched_sidecar, test_rxinfer_parser_preserves_compact_multiagent_keys, test_rxinfer_renderer_writes_execution_metadata_sidecar, test_rxinfer_execution_metadata_supports_indexed_agents_without_fallback, test_rxinfer_execution_metadata_ignores_schemaless_sidecar, test_rxinfer_execution_metadata_ignores_stale_sidecar_hash, test_rxinfer_execution_metadata_ignores_unmatched_toml_sidecar, test_rxinfer_step12_result_records_agent_metadata_on_success, test_rxinfer_step12_result_records_agent_metadata_on_failure, _rxinfer_metadata
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import time
import tomllib
from pathlib import Path

import pytest

from gnn.execute.processor import (
    _load_rxinfer_execution_metadata_from_script,
    execute_single_script,
)
from gnn.extract.pomdp_extractor import POMDPExtractor
from gnn.render.pomdp_processor import POMDPRenderProcessor
from gnn.render.rxinfer.model_contracts import (
    _create_toml_config_structure,
    _write_toml_with_exact_formatting,
)
from gnn.render.rxinfer.rxinfer_renderer import (
    build_rxinfer_execution_metadata,
    render_gnn_to_rxinfer,
)
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus


@pytest.fixture
def ready_python_rxinfer(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Admit the Python metadata fixture through the bounded readiness seam.

    These tests execute real Python children to test sidecar propagation. They
    do not claim a Python interpreter provides a working Julia/RxInfer backend.
    """
    probes = []

    def ready(framework: str, **kwargs) -> FrameworkStatus:
        assert framework == "rxinfer"
        assert kwargs["executor"] == sys.executable
        assert kwargs["deadline_monotonic"] > time.monotonic()
        probes.append(kwargs)
        return FrameworkStatus(framework, True)

    monkeypatch.setattr("gnn.execute.processor._check_framework_by_name", ready)
    return probes


def test_rxinfer_compact_multiagent_keys_drive_agent_count() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 3,
            "agent_ids": [10, 20, 30],
            "agent_initial_positions": [[0, 0], [1, 0], [0, 1]],
            "agent_target_positions": [[2, 2], [3, 2], [2, 3]],
            "agent_radii": [0.5, 0.6, 0.7],
        }
    }
    config = _create_toml_config_structure(spec, {})
    assert config["model"]["nr_agents"] == 3
    assert [agent["id"] for agent in config["agents"]] == [10, 20, 30]
    assert config["agents"][2]["target_position"] == [2, 3]


def test_rxinfer_nr_agents_does_not_silently_fallback_to_four_defaults() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 3,
            "agent_ids": [1, 2, 3],
            "agent_initial_positions": [[0, 0], [1, 0], [0, 1]],
        }
    }
    with pytest.raises(ValueError, match="agent_target_positions"):
        _create_toml_config_structure(spec, {})


def test_rxinfer_clustered_topology_is_preserved_in_config() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 3,
            "agent_ids": [10, 20, 30],
            "agent_initial_positions": [[0, 0], [1, 0], [0, 1]],
            "agent_target_positions": [[2, 2], [3, 2], [2, 3]],
            "agent_edges": [[10, 20], {"source": 20, "target": 30}],
            "agent_clusters": {"left": [10, 20], "right": [30]},
            "message_passing": "clustered_mean_field",
        }
    }
    config = _create_toml_config_structure(spec, {})

    assert config["topology"]["type"] == "clustered"
    assert config["topology"]["edges"] == [
        {"source": 10, "target": 20},
        {"source": 20, "target": 30},
    ]
    assert config["topology"]["clusters"][0] == {
        "name": "left",
        "agent_ids": [10, 20],
    }
    assert config["topology"]["message_passing"] == "clustered_mean_field"


def test_rxinfer_topology_rejects_unknown_edge_endpoint() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 2,
            "agent_ids": ["a", "b"],
            "agent_initial_positions": [[0, 0], [1, 0]],
            "agent_target_positions": [[2, 2], [3, 2]],
            "agent_edges": [["a", "missing"]],
        }
    }

    with pytest.raises(ValueError, match="undeclared agent"):
        _create_toml_config_structure(spec, {})


def test_rxinfer_topology_rejects_unknown_cluster_member() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 2,
            "agent_ids": ["a", "b"],
            "agent_initial_positions": [[0, 0], [1, 0]],
            "agent_target_positions": [[2, 2], [3, 2]],
            "agent_clusters": {"bad": ["a", "missing"]},
        }
    }

    with pytest.raises(ValueError, match="undeclared agent"):
        _create_toml_config_structure(spec, {})


def test_rxinfer_topology_rejects_partial_edge_record() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 2,
            "agent_ids": ["a", "b"],
            "agent_initial_positions": [[0, 0], [1, 0]],
            "agent_target_positions": [[2, 2], [3, 2]],
            "agent_edges": [{"source": "a"}],
        }
    }

    with pytest.raises(ValueError, match="source and target"):
        _create_toml_config_structure(spec, {})


def test_rxinfer_topology_rejects_non_list_cluster_members() -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 2,
            "agent_ids": ["a", "b"],
            "agent_initial_positions": [[0, 0], [1, 0]],
            "agent_target_positions": [[2, 2], [3, 2]],
            "agent_clusters": {"bad": "a,b"},
        }
    }

    with pytest.raises(ValueError, match="members must be a list"):
        _create_toml_config_structure(spec, {})


def test_rxinfer_accepts_underscored_initial_parameterization() -> None:
    config = _create_toml_config_structure(
        {
            "initial_parameterization": {
                "nr_agents": 2,
                "agent_ids": ["agent_a", "agent_b"],
                "agent_initial_positions": [[0, 0], [1, 0]],
                "agent_target_positions": [[2, 2], [3, 2]],
            }
        },
        {},
    )

    assert config["model"]["nr_agents"] == 2
    assert [agent["id"] for agent in config["agents"]] == ["agent_a", "agent_b"]


def test_rxinfer_string_topology_edges_render_valid_toml() -> None:
    config = _create_toml_config_structure(
        {
            "initial_parameterization": {
                "nr_agents": 2,
                "agent_ids": ["a", "b"],
                "agent_initial_positions": [[0, 0], [1, 0]],
                "agent_target_positions": [[2, 2], [3, 2]],
                "agent_edges": [["a", "b"]],
            }
        },
        {},
    )
    buffer = io.StringIO()
    _write_toml_with_exact_formatting(buffer, config)
    parsed = tomllib.loads(buffer.getvalue())

    assert parsed["agents"][0]["id"] == "a"
    assert parsed["topology"]["edges"][0] == {"source": "a", "target": "b"}


def test_rxinfer_execution_metadata_reads_toml_topology(tmp_path: Path) -> None:
    spec = {
        "initialparameterization": {
            "nr_agents": 2,
            "agent_ids": [1, 2],
            "agent_initial_positions": [[0, 0], [1, 0]],
            "agent_target_positions": [[2, 2], [3, 2]],
            "agent_edges": [[1, 2]],
        }
    }
    config = _create_toml_config_structure(spec, {})
    script_path = tmp_path / "model.jl"
    script_path.write_text("# rendered julia\n", encoding="utf-8")
    with (tmp_path / "model.toml").open("w", encoding="utf-8") as handle:
        _write_toml_with_exact_formatting(handle, config)

    metadata = _load_rxinfer_execution_metadata_from_script(script_path)

    assert metadata["agent_count"] == 2
    assert metadata["topology"]["type"] == "network"
    assert metadata["topology"]["edges"] == [{"source": 1, "target": 2}]


def test_rxinfer_execution_metadata_ignores_unmatched_sidecar(tmp_path: Path) -> None:
    script_path = tmp_path / "a_rxinfer.jl"
    script_path.write_text("# rendered julia\n", encoding="utf-8")
    (tmp_path / "b.metadata.json").write_text(
        json.dumps({"agent_count": 99, "topology": {"type": "wrong"}}),
        encoding="utf-8",
    )

    assert _load_rxinfer_execution_metadata_from_script(script_path) == {}


def test_rxinfer_parser_preserves_compact_multiagent_keys(tmp_path: Path) -> None:
    content = """
## ModelName
Compact Agents

## StateSpaceBlock
s[2,1,type=categorical]
o[2,1,type=categorical]
u[2,1,type=categorical]

## Connections
s > o
s > s
u > s

## InitialParameterization
nr_agents=3
agent_ids=[1,2,3]
agent_initial_positions=[[0.0,0.0],[1.0,0.0],[0.0,1.0]]
agent_target_positions=[[2.0,2.0],[3.0,2.0],[2.0,3.0]]
agent_edges=[[1,2],[2,3]]
agent_clusters=[{"name":"left","agent_ids":[1,2]},{"name":"right","agent_ids":[3]}]
message_passing=clustered_mean_field
A={(0.9,0.1),(0.1,0.9)}
B={((0.9,0.1),(0.1,0.9)),((0.1,0.9),(0.9,0.1))}
C={(1.0,0.0)}
D={(0.5,0.5)}
"""
    pomdp_space = POMDPExtractor(strict_validation=True).extract_from_gnn_content(
        content
    )
    assert pomdp_space is not None

    initial = pomdp_space.initial_parameterization
    assert initial is not None
    assert initial["nr_agents"] == 3
    assert initial["agent_ids"] == [1, 2, 3]
    assert initial["agent_edges"] == [[1, 2], [2, 3]]
    assert initial["agent_clusters"][0]["name"] == "left"

    gnn_spec = POMDPRenderProcessor(tmp_path)._pomdp_to_gnn_spec(pomdp_space)
    assert gnn_spec["initialparameterization"]["nr_agents"] == 3
    assert gnn_spec["initialparameterization"]["agent_clusters"][1]["agent_ids"] == [3]


def test_rxinfer_renderer_writes_execution_metadata_sidecar(tmp_path: Path) -> None:
    spec = {
        "name": "Compact Agents",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
        },
        "initialparameterization": {
            "nr_agents": 3,
            "agent_ids": [1, 2, 3],
            "agent_initial_positions": [[0, 0], [1, 0], [0, 1]],
            "agent_target_positions": [[2, 2], [3, 2], [2, 3]],
            "agent_edges": [[1, 2], [2, 3]],
            "agent_clusters": {"left": [1, 2], "right": [3]},
            "message_passing": "clustered_mean_field",
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.1, 0.9], [0.9, 0.1]],
            ],
            "C": [1.0, 0.0],
            "D": [0.5, 0.5],
        },
    }
    output_path = tmp_path / "compact_agents_rxinfer.jl"

    success, message, warnings = render_gnn_to_rxinfer(spec, output_path)

    assert success, message
    assert warnings == []
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["agent_count"] == 3
    assert metadata["topology"]["type"] == "clustered"
    assert metadata["schema"] == "gnn_rxinfer_execution_metadata_v1"
    assert len(metadata["script_sha256"]) == 64
    assert metadata["topology"]["edges"] == [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
    ]
    assert metadata["topology"]["message_passing"] == "clustered_mean_field"

    loaded = _load_rxinfer_execution_metadata_from_script(output_path)
    assert loaded["agent_count"] == 3
    assert loaded["topology"]["source"] == str(metadata_path)
    assert loaded["metadata_verification"] == "script_sha256_match"


def test_rxinfer_execution_metadata_supports_indexed_agents_without_fallback() -> None:
    metadata = build_rxinfer_execution_metadata(
        {
            "initialparameterization": {
                "agent1_id": 10,
                "agent1_initial_position": [0, 0],
                "agent1_target_position": [1, 1],
                "agent2_id": 20,
                "agent2_initial_position": [1, 0],
                "agent2_target_position": [2, 1],
            }
        }
    )

    assert metadata["agent_count"] == 2
    assert metadata["topology"]["agent_ids"] == [10, 20]


def test_rxinfer_execution_metadata_ignores_schemaless_sidecar(tmp_path: Path) -> None:
    script_path = tmp_path / "demo_rxinfer.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    script_path.with_suffix(".metadata.json").write_text(
        json.dumps({"agent_count": 99, "topology": {"type": "wrong"}}),
        encoding="utf-8",
    )

    assert _load_rxinfer_execution_metadata_from_script(script_path) == {}


def test_rxinfer_execution_metadata_ignores_stale_sidecar_hash(tmp_path: Path) -> None:
    script_path = tmp_path / "demo_rxinfer.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    script_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema": "gnn_rxinfer_execution_metadata_v1",
                "script_sha256": "0" * 64,
                "agent_count": 99,
                "topology": {"type": "wrong"},
            }
        ),
        encoding="utf-8",
    )

    assert _load_rxinfer_execution_metadata_from_script(script_path) == {}


def test_rxinfer_execution_metadata_ignores_unmatched_toml_sidecar(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "demo_rxinfer.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "other.toml").write_text(
        """
        [[agents]]
        id = 99
        radius = 1.0
        initial_position = [0, 0]
        target_position = [1, 1]
        """,
        encoding="utf-8",
    )

    assert _load_rxinfer_execution_metadata_from_script(script_path) == {}


def test_rxinfer_step12_result_records_agent_metadata_on_success(
    tmp_path: Path,
    ready_python_rxinfer: list[dict],
) -> None:
    script_path = tmp_path / "demo" / "rxinfer" / "demo_rxinfer.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("print('ok')\n", encoding="utf-8")
    script_path.with_suffix(".metadata.json").write_text(
        json.dumps(_rxinfer_metadata(script_path, 3)),
        encoding="utf-8",
    )

    result = execute_single_script(
        {
            "path": script_path,
            "name": script_path.name,
            "framework": "rxinfer",
            "executor": sys.executable,
        },
        tmp_path / "12_execute_output",
        False,
        logging.getLogger("test"),
        timeout=10,
    )

    assert result["success"] is True
    assert result["return_code"] == 0 and "ok" in result["stdout"]
    assert len(ready_python_rxinfer) == 1
    assert result["execution_metadata"]["agent_count"] == 3
    structured = json.loads(
        (tmp_path / "12_execute_output")
        .joinpath(
            "demo", "rxinfer", "execution_logs", f"{script_path.name}_results.json"
        )
        .read_text(encoding="utf-8")
    )
    assert structured["success"] is True
    assert structured["execution_metadata"]["agent_count"] == 3
    assert structured["execution_metadata"]["topology"]["type"] == "clustered"


def test_rxinfer_step12_result_records_agent_metadata_on_failure(
    tmp_path: Path,
    ready_python_rxinfer: list[dict],
) -> None:
    script_path = tmp_path / "demo" / "rxinfer" / "demo_rxinfer.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
    script_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            _rxinfer_metadata(
                script_path,
                2,
                {"type": "agent_population", "agent_ids": [10, 20]},
            )
        ),
        encoding="utf-8",
    )

    result = execute_single_script(
        {
            "path": script_path,
            "name": script_path.name,
            "framework": "rxinfer",
            "executor": sys.executable,
        },
        tmp_path / "12_execute_output",
        False,
        logging.getLogger("test"),
        timeout=10,
    )

    assert result["success"] is False
    assert result["return_code"] == 2
    assert len(ready_python_rxinfer) == 1
    assert result["execution_metadata"]["agent_count"] == 2
    structured = json.loads(
        (tmp_path / "12_execute_output")
        .joinpath(
            "demo", "rxinfer", "execution_logs", f"{script_path.name}_results.json"
        )
        .read_text(encoding="utf-8")
    )
    assert structured["success"] is False
    assert structured["execution_metadata"]["agent_count"] == 2


def _rxinfer_metadata(
    script_path: Path,
    agent_count: int,
    topology: dict[str, object] | None = None,
) -> dict[str, object]:
    topology = topology or {
        "type": "clustered",
        "agent_ids": [1, 2, 3],
        "edges": [{"source": 1, "target": 2}],
        "message_passing": "clustered_mean_field",
    }
    return {
        "schema": "gnn_rxinfer_execution_metadata_v1",
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "agent_count": agent_count,
        "topology": topology,
    }
