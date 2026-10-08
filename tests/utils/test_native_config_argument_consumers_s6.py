"""Native consumers of the frozen, prior configuration utility export.

This public ``parse_arguments`` API projects saved options; it does not launch
the pipeline or replace its canonical current-run admission contract.
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
OWNER = REPO / "src/gnn/utils/arguments/arg_parsing.py"
PUBLIC_CONSUMER = """
import json
import os
from pathlib import Path
from gnn.utils import parse_arguments
from gnn.utils.arguments import parse_arguments as concern_export
import gnn.utils.arguments.arg_parsing as owner
before = dict(os.environ)
arguments = parse_arguments()
assert dict(os.environ) == before
print(json.dumps({"arguments": arguments.to_dict(),
                  "owner": str(Path(owner.__file__).resolve()),
                  "same_export": parse_arguments is concern_export}))
"""


def _invoke(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    before_env = dict(os.environ)
    before_source = OWNER.read_bytes()
    env = dict(before_env, PYTHONPATH=str(REPO / "src"))
    command = [sys.executable, "-P", "-c", PUBLIC_CONSUMER, *arguments]
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
                "owner_sha256": hashlib.sha256(before_source).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert dict(os.environ) == before_env
    assert OWNER.read_bytes() == before_source
    return result


def _saved_configuration(workspace: Path) -> tuple[Path, dict]:
    target = workspace / "caller-models"
    target.mkdir()
    (target / "caller.md").write_text("caller-owned model bytes\n")
    ontology = workspace / "terms.json"
    ontology.write_text('{"caller_term": "preserved"}\n')
    authored = {
        "pipeline": {
            "target_dir": str(target),
            "output_dir": str(workspace / "configured-output"),
            "pipeline_summary_file": str(workspace / "configured-summary.json"),
            "recursive": True,
            "verbose": True,
            "enable_round_trip": False,
            "enable_cross_format": False,
            "skip_steps": [13, 14],
            "only_steps": [3, 5],
            "fast_only": False,
            "comprehensive": True,
        },
        "type_checker": {"strict": False, "estimate_resources": True},
        "ontology": {"terms_file": str(ontology)},
        "llm": {"tasks": "validation", "timeout": 47},
        "setup": {"recreate_venv": False, "dev": False},
        "sapf": {"duration": 2.5},
    }
    path = workspace / "saved-caller.yaml"
    path.write_text(yaml.safe_dump(authored), encoding="utf-8")
    return path, authored


def _assert_saved_inputs_unchanged(workspace: Path, before: dict[Path, bytes]) -> None:
    assert {path: path.read_bytes() for path in before} == before
    assert not list(workspace.rglob("pipeline_execution_summary.json"))
    assert not list(workspace.rglob("run_context.json"))
    assert not (workspace / "configured-output").exists()
    assert not (workspace / "configured-summary.json").exists()


def test_public_export_reads_saved_yaml_without_launching_any_step(
    tmp_path: Path,
) -> None:
    config, authored = _saved_configuration(tmp_path)
    inputs = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = _invoke(tmp_path, "--config-file", str(config))
    assert result.returncode == 0, result
    payload = json.loads(result.stdout)
    assert Path(payload["owner"]) == OWNER.resolve()
    assert payload["same_export"] is True
    projected = payload["arguments"]
    expected = {
        **{
            name: authored["pipeline"][name]
            for name in (
                "target_dir",
                "output_dir",
                "pipeline_summary_file",
                "recursive",
                "verbose",
                "enable_round_trip",
                "enable_cross_format",
                "fast_only",
                "comprehensive",
            )
        },
        "skip_steps": "13,14",
        "only_steps": "3,5",
        "strict": False,
        "estimate_resources": True,
        "ontology_terms_file": authored["ontology"]["terms_file"],
        "llm_tasks": "validation",
        "llm_timeout": 47,
        "duration": 2.5,
        "recreate_venv": False,
        "dev": False,
    }
    assert {name: projected[name] for name in expected} == expected
    _assert_saved_inputs_unchanged(tmp_path, inputs)


@pytest.mark.parametrize("skip_steps", ["14", "14,13"])
def test_explicit_false_paths_and_skip_llm_override_saved_configuration(
    tmp_path: Path, skip_steps: str
) -> None:
    config, _ = _saved_configuration(tmp_path)
    target = tmp_path / "override-models"
    target.mkdir()
    ontology = tmp_path / "override-terms.json"
    ontology.write_text("{}\n")
    output = tmp_path / "override-output"
    summary = tmp_path / "override-summary.json"
    inputs = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = _invoke(
        tmp_path,
        "--config-file",
        str(config),
        "--target-dir",
        str(target),
        "--output-dir",
        str(output),
        "--pipeline-summary-file",
        str(summary),
        "--ontology-terms-file",
        str(ontology),
        "--no-recursive",
        "--no-verbose",
        "--no-estimate-resources",
        "--skip-steps",
        skip_steps,
        "--skip-llm",
        "--only-steps",
        "3",
        "--enable-round-trip",
        "--enable-cross-format",
        "--strict",
        "--llm-tasks",
        "summarization",
        "--llm-timeout",
        "31",
        "--duration",
        "1.25",
        "--dev",
        "--recreate-venv",
    )
    assert result.returncode == 0, result
    payload = json.loads(result.stdout)
    assert Path(payload["owner"]) == OWNER.resolve()
    projected = payload["arguments"]
    expected = {
        "target_dir": str(target),
        "output_dir": str(output),
        "pipeline_summary_file": str(summary),
        "ontology_terms_file": str(ontology),
        "recursive": False,
        "verbose": False,
        "estimate_resources": False,
        "skip_steps": "14,13",
        "only_steps": "3",
        "enable_round_trip": True,
        "enable_cross_format": True,
        "strict": True,
        "llm_tasks": "summarization",
        "llm_timeout": 31,
        "duration": 1.25,
        "dev": True,
        "recreate_venv": True,
        "fast_only": False,
        "comprehensive": True,
    }
    assert {name: projected[name] for name in expected} == expected
    assert projected["skip_steps"].split(",").count("13") == 1
    _assert_saved_inputs_unchanged(tmp_path, inputs)
    assert not output.exists() and not summary.exists()


def test_unknown_public_option_refuses_without_replacing_prior_files(
    tmp_path: Path,
) -> None:
    config, _ = _saved_configuration(tmp_path)
    sentinel = tmp_path / "configured-summary.json"
    sentinel.write_bytes(b'{"prior_run": "caller-owned"}\n')
    inputs = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = _invoke(
        tmp_path, "--config-file", str(config), "--unknown-consumer-option"
    )
    assert result.returncode == 2, result
    assert "unrecognized arguments: --unknown-consumer-option" in result.stderr
    assert result.stdout == ""
    assert {path: path.read_bytes() for path in inputs} == inputs
    assert not (tmp_path / "configured-output").exists()
