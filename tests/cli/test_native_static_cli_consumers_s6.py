"""Actual CLI consumers of authored static models and saved receipts.

Static complexity bounds and parse summaries are informational artifacts;
these witnesses do not execute a backend or certify current-run completion.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SOURCES = (
    REPO / "src/gnn/cli/__init__.py",
    REPO / "src/gnn/cli/parser.py",
    REPO / "src/gnn/cli/commands.py",
    REPO / "src/gnn/cli/handlers_complexity.py",
    REPO / "src/gnn/cli/handlers_pipeline.py",
)
MODEL = """---
author: Caller Scientist
version: caller-v1
tags: [native, saved]
---
## GNNSection
ActInfPOMDP
## GNNVersionAndFlags
GNN v1
## ModelName
Caller Static Model
## StateSpaceBlock
A[2,2,type=float]
B[2,2,2,type=float]
C[2,type=float]
D[2,1,type=float]
s[2,1,type=float]
o[2,1,type=int]
u[1,type=int]
## Connections
D>s
s-A
A-o
s-B
B>u
u>s
## InitialParameterization
A={(0.9,0.2),(0.1,0.8)}
B={((0.95,0.05),(0.05,0.95)),((0.05,0.95),(0.95,0.05))}
C={(0.0,0.0)}
D={(0.5,0.5)}
## Time
Static
## ModelParameters
b_tensor_order: next_state_previous_state_action
num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 5
"""


def _invoke(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    before_env = dict(os.environ)
    source_bytes = {path: path.read_bytes() for path in SOURCES}
    env = dict(before_env, PYTHONPATH=str(REPO / "src"))
    command = [sys.executable, "-P", "-m", "gnn.cli", *arguments]
    result = subprocess.run(
        command, cwd=workspace, env=env, text=True, capture_output=True, timeout=60
    )
    receipts = workspace / "native-receipts"
    receipts.mkdir(exist_ok=True)
    path = receipts / f"call-{len(list(receipts.glob('*.json')))}.json"
    path.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(workspace),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "source_sha256": {
                    str(path): hashlib.sha256(content).hexdigest()
                    for path, content in source_bytes.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert dict(os.environ) == before_env
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    return result


def _model(workspace: Path, *, relative: str = "selected/caller.md") -> Path:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODEL, encoding="utf-8")
    return path


def _assert_static_receipt(receipt: dict, source: Path, name: str) -> None:
    assert receipt["receipt_type"] == "gnn.complexity_estimate/v1"
    assert receipt["model"] == {
        "name": name,
        "path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert receipt["model_kinds"] == ["flat"]
    assert receipt["structure"]["variable_count"] == 7
    assert receipt["structure"]["edge_count"] == 6
    assert {row["framework"] for row in receipt["per_backend"]} == {
        "pymdp",
        "rxinfer",
        "discopy",
        "activeinference_jl",
        "jax",
        "numpyro",
        "pytorch",
        "ngclearn",
        "lean",
        "stan",
        "bnlearn",
    }
    for row in receipt["per_backend"]:
        if row["framework"] == "lean":
            assert row["complexity_class"] == "verification"
            assert (
                row["asymptotic"]
                == "class-only: proof cost is not numerically estimated"
            )
            assert row["drivers"] == {}
        else:
            assert "ESTIMATE" in row["asymptotic"], row
        assert not {"execution_time", "wall_median_seconds", "peak_rss_mb"} & row.keys()
    assert not {"execution_time", "environment", "run_id"} & receipt.keys()


@pytest.mark.parametrize("directory", [False, True])
def test_static_complexity_cli_saves_byte_bound_file_or_recursive_corpus_receipts(
    tmp_path: Path, directory: bool
) -> None:
    source = _model(tmp_path)
    sources = [source]
    names = ["Caller Static Model"]
    if directory:
        other = _model(tmp_path, relative="selected/nested/other.md")
        other.write_text(MODEL.replace("Caller Static Model", "Nested Caller Model"))
        sources.append(other)
        names.append("Nested Caller Model")
    inputs = {path: path.read_bytes() for path in sources}
    output = tmp_path / "receipts" / "static.json"
    result = _invoke(
        tmp_path,
        "complexity",
        str(source.parent if directory else source),
        "--json",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result
    payload = json.loads(result.stdout)
    assert payload["status"] == "success" and payload["error"] is None
    assert payload["meta"]["command"] == "complexity"
    assert payload["data"]["output_path"] == str(output)
    saved = json.loads(output.read_text())
    receipts = saved["receipts"] if directory else [saved]
    assert receipts == (
        payload["data"]["receipts"] if directory else [payload["data"]["receipt"]]
    )
    assert len(receipts) == len(sources)
    by_path = {receipt["model"]["path"]: receipt for receipt in receipts}
    assert set(by_path) == {str(path) for path in sources}
    for path, name in zip(sources, names, strict=True):
        _assert_static_receipt(by_path[str(path)], path, name)
    assert {path: path.read_bytes() for path in inputs} == inputs
    assert not list(tmp_path.rglob("pipeline_execution_summary.json"))
    assert not list(tmp_path.rglob("*_results.json"))


def test_plain_complexity_table_agrees_with_saved_static_receipt(
    tmp_path: Path,
) -> None:
    source = _model(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "plain-receipt.json"
    result = _invoke(tmp_path, "complexity", str(source), "--output", str(output))
    assert result.returncode == 0, result
    saved = json.loads(output.read_text())
    _assert_static_receipt(saved, source, "Caller Static Model")
    assert "Static complexity bounds" in result.stdout
    assert "framework" in result.stdout and "applicable" in result.stdout
    assert "Caller Static Model" in result.stdout
    for row in saved["per_backend"]:
        assert row["framework"] in result.stdout
        assert " ".join(row["asymptotic"].split()) in result.stdout
    assert str(output) in result.stdout
    assert source.read_bytes() == before


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_missing_or_empty_static_input_refuses_and_preserves_prior_output(
    tmp_path: Path, empty: bool, json_output: bool
) -> None:
    source = tmp_path / ("empty-corpus" if empty else "missing.md")
    if empty:
        source.mkdir()
        (source / "caller.txt").write_bytes(b"not a GNN source\n")
    output = tmp_path / "prior-static.json"
    before = b'{"caller_owned_prior_receipt": true}\n'
    output.write_bytes(before)
    args = ["complexity", str(source), "--output", str(output)]
    if json_output:
        args.append("--json")
    result = _invoke(tmp_path, *args)
    assert result.returncode == 1, result
    diagnosis = "no GNN models found" if empty else "GNN model path not found"
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["status"] == "error" and payload["data"] == {}
        assert payload["meta"]["command"] == "complexity"
        assert diagnosis in payload["error"] and str(source) in payload["error"]
    else:
        assert diagnosis in result.stdout and str(source) in result.stdout
    assert output.read_bytes() == before
    assert not list(tmp_path.rglob("pipeline_execution_summary.json"))
    if empty:
        assert (source / "caller.txt").read_bytes() == b"not a GNN source\n"


def test_parse_summary_matches_independently_saved_json_and_yaml(
    tmp_path: Path,
) -> None:
    source = _model(tmp_path)
    before = source.read_bytes()
    structured = _invoke(tmp_path, "parse", str(source), "--json")
    assert structured.returncode == 0, structured
    saved = tmp_path / "parsed.json"
    saved.write_text(structured.stdout, encoding="utf-8")
    envelope = json.loads(saved.read_text())
    assert envelope["status"] == "success" and envelope["error"] is None
    assert envelope["meta"]["command"] == "parse"
    data = envelope["data"]
    assert data["file"] == str(source) and data["errors"] == []
    assert data["metadata"] == {
        "author": "Caller Scientist",
        "version": "caller-v1",
        "tags": ["native", "saved"],
    }
    assert {item["name"] for item in data["variables"]} == {
        "A",
        "B",
        "C",
        "D",
        "s",
        "o",
        "u",
    }
    assert len(data["variables"]) == 7 and len(data["connections"]) == 6
    summary = _invoke(tmp_path, "parse", str(source), "--format", "summary")
    assert summary.returncode == 0, summary
    assert summary.stdout.splitlines() == [
        "File: caller.md",
        "Variables: 7",
        "Connections: 6",
        "Metadata: author, version, tags",
    ]
    summary_file = tmp_path / "parsed-summary.txt"
    summary_file.write_text(summary.stdout, encoding="utf-8")
    yaml_result = _invoke(tmp_path, "parse", str(source), "--format", "yaml")
    assert yaml_result.returncode == 0, yaml_result
    yaml_file = tmp_path / "parsed.yaml"
    yaml_file.write_text(yaml_result.stdout, encoding="utf-8")
    assert yaml.safe_load(yaml_file.read_text()) == data
    assert source.read_bytes() == before
    assert not list(tmp_path.rglob("pipeline_execution_summary.json"))


def test_unknown_parse_format_refuses_before_replacing_saved_artifacts(
    tmp_path: Path,
) -> None:
    source = _model(tmp_path)
    saved = tmp_path / "parsed.json"
    saved.write_bytes(b'{"prior_parse": "caller-owned"}\n')
    before = {path: path.read_bytes() for path in (source, saved)}
    result = _invoke(tmp_path, "parse", str(source), "--format", "invented-format")
    assert result.returncode == 2, result
    assert "invalid choice: 'invented-format'" in result.stderr
    assert result.stdout == ""
    assert {path: path.read_bytes() for path in before} == before
