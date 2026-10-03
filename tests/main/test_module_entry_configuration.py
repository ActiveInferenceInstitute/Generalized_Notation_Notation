"""Real entrypoints bind caller configuration without changing package roots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import yaml

from gnn.execute.subprocess_envelope import run_subprocess_envelope

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "entrypoint", ["module", "distribution_file", "repository_file"]
)
def test_entrypoint_configuration_and_selection(
    tmp_path: Path, entrypoint: str
) -> None:
    """Only a direct source-checkout script anchors to its repository config.

    The distribution-file case copies the actual package into a directory
    without a project manifest. Ordinary installed-wheel acceptance is a
    separate release gate; this case exercises its real filesystem layout.
    """
    caller = tmp_path / "caller"
    target = caller / "input" / "gnn_files"
    source_bytes = (
        REPO / "input/gnn_files/thrml/categorical_smoothing.md"
    ).read_bytes()
    for folder in ("selected", "excluded"):
        source = target / folder / "model.md"
        source.parent.mkdir(parents=True)
        source.write_bytes(source_bytes)
    (target / "README.md").write_text("Documentation only.\n", encoding="utf-8")
    config = {
        "pipeline": {"timeout": {"step": 45, "total": 60}},
        "testing_matrix": {
            "enabled": True,
            "default_steps": [],
            "folders": {"selected": [3], "excluded": []},
        },
        "render": {"backend_options": {"thrml": {"seed": 42, "num_samples": 4096}}},
    }
    config_path = caller / "input" / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "GNN_RUN_CONTEXT_FILE",
            "GNN_RUN_ID",
        }
    }
    env["GNN_SANDBOX"] = "off"
    if entrypoint == "module":
        # Explicit source import for this local regression; -P excludes cwd
        # imports. The release gate separately verifies python -I and a wheel.
        env["PYTHONPATH"] = str(REPO / "src")
        command = [sys.executable, "-P", "-m", "gnn.main"]
    elif entrypoint == "distribution_file":
        package = tmp_path / "distribution" / "site-packages" / "gnn"
        shutil.copytree(
            REPO / "src" / "gnn",
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        assert not (package.parent.parent / "pyproject.toml").exists()
        command = [sys.executable, "-I", str(package / "main.py")]
    else:
        assert (REPO / "pyproject.toml").is_file()
        command = [sys.executable, "-I", str(REPO / "src/gnn/main.py")]
    output = caller / "output"
    command.extend(
        [
            "--target-dir",
            str(target),
            "--output-dir",
            str(output),
            "--only-steps",
            "3",
            "--skip-steps",
            "0,1,2,13",
            "--skip-llm",
            "--serialize-preset",
            "minimal",
            "--timeout",
            "60",
        ]
    )
    launched = time.monotonic()
    result = run_subprocess_envelope(
        command,
        cwd=caller,
        env=env,
        timeout=65,
        deadline_monotonic=launched + 65,
        sandbox=False,
    )
    (caller / "entrypoint-result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    assert result["success"], result
    assert result["cleanup_verified"] and result["streams_drained"], result
    summary = json.loads(
        (output / "00_pipeline_summary/pipeline_execution_summary.json").read_text()
    )
    context = json.loads((output / "00_pipeline_summary/run_context.json").read_text())
    resolved = json.loads(context["config_json"])
    assert context["input_root"] == str(target.resolve())
    assert context["selected_steps"] == [3]
    assert summary["run_id"] == context["run_id"]
    assert [step["script_name"] for step in summary["steps"]] == ["3_gnn.py"]
    assert summary["overall_status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    assert sorted(model["relative_path"] for model in context["models"]) == [
        "excluded/model.md",
        "selected/model.md",
    ]
    assert all(
        model["sha256"] == hashlib.sha256(source_bytes).hexdigest()
        for model in context["models"]
    )
    assert ["README.md", "documentation"] in context["exclusions"]
    selected = [model["relative_path"] for model in context["models"] if model["steps"]]
    if entrypoint == "repository_file":
        expected = yaml.safe_load((REPO / "input/config.yaml").read_text())
        assert resolved == expected
        assert selected == ["excluded/model.md", "selected/model.md"]
    else:
        assert resolved == config
        assert selected == ["selected/model.md"]
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_digest
