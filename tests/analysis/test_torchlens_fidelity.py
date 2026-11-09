"""Declared-connections-vs-traced-dataflow comparison (pure JSON, no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gnn.analysis.pytorch.torchlens_fidelity import (
    FIDELITY_JSON,
    FIDELITY_MARKDOWN,
    evaluate_capture,
    site_owners,
    write_fidelity_report,
)

# A trimmed site graph with the shape the emitted rollout produces.
SITE_EDGES: List[List[Any]] = [
    ["agent.A", "agent.state_inference", 4],
    ["agent.A", "agent.efe.ambiguity", 8],
    ["agent.B", "agent.transition", 3],
    ["agent.B", "agent.efe", 4],
    ["agent.C", "agent.efe.risk", 4],
    ["agent.D", "agent.state_inference", 1],
    ["agent.E", "agent.policy", 4],
    ["agent.efe", "agent.efe.risk", 4],
    ["agent.efe.ambiguity", "agent.efe", 4],
    ["agent.efe.risk", "agent.efe", 4],
    ["agent.efe", "agent.policy", 4],
    ["agent.policy", "agent.action", 4],
    ["agent.action", "agent.transition", 3],
    ["agent.state_inference", "agent.transition", 3],
    ["agent.state_inference", "agent.efe", 4],
    ["agent.transition", "agent.state_inference", 3],
    ["process.A", "process.observe", 4],
    ["process.observe", "agent.state_inference", 4],
]

VARIABLE_SITES: Dict[str, List[str]] = {
    "A": ["agent.A", "process.A"],
    "B": ["agent.B", "process.B"],
    "C": ["agent.C"],
    "D": ["agent.D", "process.D"],
    "E": ["agent.E"],
    "G": ["agent.efe"],
    "o": ["process.observe"],
    "s": ["agent.state_inference"],
    "s_prime": ["agent.transition"],
    "u": ["agent.action"],
    "π": ["agent.policy"],
}


def _capture(**overrides: Any) -> Dict[str, Any]:
    capture: Dict[str, Any] = {
        "status": "captured",
        "torchlens_version": "3.0.2",
        "num_ops": 99,
        "trace_reproduces_simulation": True,
        "validated": True,
        "agent_step_validation": {"first_step": True, "last_step": True},
        "declared_connections": [
            {"source": "D", "relation": ">", "target": "s"},
            {"source": "s", "relation": "-", "target": "A"},
            {"source": "B", "relation": ">", "target": "u"},
            {"source": "o", "relation": ">", "target": "A"},
            {"source": "(s, u)", "relation": ">", "target": "s_prime"},
            {"source": "o", "relation": ">", "target": "C"},
            {"source": "F", "relation": "-", "target": "s"},
            {"source": "π", "relation": "<", "target": "G"},
        ],
        "variable_sites": VARIABLE_SITES,
        "unmapped_variables": {"F": "variational free energy is not computed"},
        "site_edges": SITE_EDGES,
        "intervention_witnesses": {
            "timestep": 3,
            "policy": [0.6, 0.4],
            "argmax_action": 0,
            "lesions": {
                "no_risk": {
                    "site": "agent.efe.risk",
                    "edit": "zero_ablate",
                    "policy": [0.5, 0.5],
                    "argmax_action": 0,
                    "kl_full_to_lesioned": 0.02,
                }
            },
        },
    }
    capture.update(overrides)
    return capture


def _status(
    report: Dict[str, Any], source: str, relation: str, target: str
) -> Dict[str, Any]:
    return next(
        edge
        for edge in report["declared_edges"]
        if (edge["source"], edge["relation"], edge["target"])
        == (source, relation, target)
    )


def test_sites_belong_to_the_longest_mapped_prefix() -> None:
    owners = site_owners(
        {
            "G": ["agent.efe"],
            "G_ins": ["agent.efe.risk"],
            "A": ["agent.A", "process.A"],
        },
        ["agent.efe", "agent.efe.risk", "agent.efe.ambiguity", "process.A", "rollout"],
    )
    assert owners["agent.efe"] == {"G"}
    assert owners["agent.efe.ambiguity"] == {"G"}
    assert owners["agent.efe.risk"] == {"G_ins"}
    assert owners["process.A"] == {"A"}
    assert "rollout" not in owners


def test_declared_edges_are_classified_from_the_trace() -> None:
    report = evaluate_capture(_capture())
    assert report["status"] == "evaluated"
    direct = _status(report, "D", ">", "s")
    assert direct["status"] == "direct"
    assert direct["pairs"][0]["path"] == ["agent.D", "agent.state_inference"]

    undirected = _status(report, "s", "-", "A")
    assert undirected["status"] == "direct"
    assert undirected["pairs"][0]["direction"] == "backward"

    indirect = _status(report, "B", ">", "u")
    assert indirect["status"] == "indirect"
    assert indirect["pairs"][0]["path"] == [
        "agent.B",
        "agent.efe",
        "agent.policy",
        "agent.action",
    ]

    # Buffers have no traced inputs: the only path runs A -> o.
    reverse = _status(report, "o", ">", "A")
    assert reverse["status"] == "reverse_only"
    assert reverse["pairs"][0]["path"] == ["process.A", "process.observe"]
    assert _status(report, "o", ">", "C")["status"] == "not_observed"
    assert _status(report, "π", "<", "G")["status"] == "direct"

    grouped = _status(report, "(s, u)", ">", "s_prime")
    assert grouped["status"] == "direct"
    assert {pair["source"] for pair in grouped["pairs"]} == {"s", "u"}

    unmapped = _status(report, "F", "-", "s")
    assert unmapped["status"] == "unmapped"
    assert unmapped["pairs"][0]["reasons"]["F"].startswith("variational free energy")


def test_summary_counts_and_undeclared_dependencies() -> None:
    report = evaluate_capture(_capture())
    assert report["status_counts"] == {
        "direct": 4,
        "indirect": 1,
        "not_observed": 1,
        "reverse_only": 1,
        "unmapped": 1,
    }
    assert report["mapped_declared_edges"] == 7
    assert report["supported_declared_edges"] == 5
    assert report["structural_agreement"] == pytest.approx(5 / 7)
    undeclared = {(d["source"], d["target"]) for d in report["undeclared_dependencies"]}
    assert ("A", "G") in undeclared
    assert ("o", "s") in undeclared
    assert ("s", "G") in undeclared
    # Declared pairs (either direction for "-") never appear as undeclared.
    assert ("A", "s") not in undeclared and ("D", "s") not in undeclared
    # Internal flow between sites of one variable is not a dependency.
    assert ("G", "G") not in undeclared


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"status": "failed", "reason": "boom"}, "status is 'failed'"),
        (
            {"status": "unavailable", "reason": "no torchlens"},
            "status is 'unavailable'",
        ),
        ({"trace_reproduces_simulation": False}, "did not reproduce"),
        ({"validated": False}, "validation of the agent step did not pass"),
    ],
)
def test_unverified_captures_are_refused(
    overrides: Dict[str, Any], reason: str
) -> None:
    report = evaluate_capture(_capture(**overrides))
    assert report["status"] == "unverified"
    assert reason in report["reason"]
    assert "declared_edges" not in report


def test_report_writer_skips_disabled_and_writes_evidence(tmp_path: Path) -> None:
    assert (
        write_fidelity_report({"torchlens": {"status": "disabled"}}, "m", tmp_path)
        == []
    )
    assert write_fidelity_report({}, "m", tmp_path) == []
    written = write_fidelity_report({"torchlens": _capture()}, "m", tmp_path / "m")
    assert str(tmp_path / "m" / FIDELITY_JSON) in written
    assert str(tmp_path / "m" / FIDELITY_MARKDOWN) in written
    report = json.loads((tmp_path / "m" / FIDELITY_JSON).read_text())
    assert report["model_name"] == "m" and report["status"] == "evaluated"
    markdown = (tmp_path / "m" / FIDELITY_MARKDOWN).read_text()
    assert "| `B > u` | indirect |" in markdown
    assert "no_risk" in markdown


def test_unverified_capture_still_writes_an_explicit_report(tmp_path: Path) -> None:
    written = write_fidelity_report(
        {"torchlens": _capture(status="unavailable", reason="missing")}, "m", tmp_path
    )
    assert len(written) == 2  # JSON + Markdown; no figure for an unverified capture
    assert "Not evaluated" in (tmp_path / FIDELITY_MARKDOWN).read_text()


def _execute_tree(root: Path, torchlens: Dict[str, Any]) -> Path:
    result_dir = root / "smoke" / "pytorch" / "simulation_data"
    result_dir.mkdir(parents=True)
    payload = {
        "model_name": "smoke",
        "beliefs": [[0.5, 0.5], [0.6, 0.4]],
        "actions": [0, 1],
        "observations": [0, 1],
        "efe_history": [[1.0, 2.0], [1.5, 1.0]],
        "validation": {"all_valid": True},
        "torchlens": torchlens,
    }
    path = result_dir / "simulation_results.json"
    path.write_text(json.dumps(payload))
    return path


def test_step16_pytorch_analyzer_writes_the_fidelity_report(tmp_path: Path) -> None:
    from gnn.analysis.pytorch.analyzer import generate_analysis_from_logs

    _execute_tree(tmp_path / "exec", _capture())
    written = generate_analysis_from_logs(tmp_path / "exec", tmp_path / "out")
    assert str(tmp_path / "out" / "smoke" / "pytorch_analysis.json") in written
    assert str(tmp_path / "out" / "smoke" / FIDELITY_JSON) in written


def test_step16_pytorch_analyzer_writes_nothing_extra_without_capture(
    tmp_path: Path,
) -> None:
    from gnn.analysis.pytorch.analyzer import generate_analysis_from_logs

    _execute_tree(tmp_path / "exec", {"status": "disabled", "reason": "not set"})
    written = generate_analysis_from_logs(tmp_path / "exec", tmp_path / "out")
    assert not any(FIDELITY_JSON in path for path in written)
    assert not (tmp_path / "out" / "smoke" / FIDELITY_JSON).exists()


def test_mcp_tool_evaluates_files_and_directories_read_only(tmp_path: Path) -> None:
    from gnn.analysis.mcp import evaluate_torchlens_fidelity_mcp

    path = _execute_tree(tmp_path / "exec", _capture())
    before = sorted(p.name for p in tmp_path.rglob("*"))
    single = evaluate_torchlens_fidelity_mcp(str(path))
    tree = evaluate_torchlens_fidelity_mcp(str(tmp_path / "exec"))
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    assert single["success"] and single["reports_count"] == 1
    assert single["reports"][0]["model_name"] == "smoke"
    assert single["reports"][0]["report"]["status"] == "evaluated"
    assert tree["reports"] == single["reports"]

    missing = evaluate_torchlens_fidelity_mcp(str(tmp_path / "nowhere"))
    assert missing == {
        "success": False,
        "error": f"Path not found: {tmp_path / 'nowhere'}",
    }


def test_mcp_tool_reports_runs_without_capture(tmp_path: Path) -> None:
    from gnn.analysis.mcp import evaluate_torchlens_fidelity_mcp

    path = _execute_tree(tmp_path / "exec", {"status": "disabled", "reason": "not set"})
    result = evaluate_torchlens_fidelity_mcp(str(path))
    assert result["reports"][0]["report"] == {
        "status": "not_captured",
        "reason": "not set",
    }
