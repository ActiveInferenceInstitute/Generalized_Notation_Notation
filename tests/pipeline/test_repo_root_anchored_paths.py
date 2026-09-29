#!/usr/bin/env python3
"""Module-level paths anchored on ``__file__`` must reach the repository root.

After the package moved under ``src/gnn/``, three ``parents[2]``-style anchors
kept pointing at ``src/``: the LLM step ignored every ``llm:`` setting in
``input/config.yaml`` (a full run analysed 41 non-model files instead of the
configured 8), the run hash recorded the runtime config as absent, and the
executor's default output dir landed in ``src/output``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "input" / "config.yaml"


def test_run_hash_binds_the_real_runtime_config() -> None:
    from gnn.pipeline.hasher import RUNTIME_CONFIG_PATH, runtime_config_identity

    assert RUNTIME_CONFIG_PATH == CONFIG
    assert runtime_config_identity()["present"] is True


def test_llm_step_reads_the_llm_section_of_input_config(monkeypatch) -> None:
    from gnn.llm.processor import _get_llm_config

    monkeypatch.delenv("GNN_TESTING_NO_LLM_CONFIG", raising=False)
    expected = (yaml.safe_load(CONFIG.read_text()) or {}).get("llm", {})
    assert expected, "input/config.yaml must carry an llm: section"
    assert _get_llm_config() == expected


def test_executor_default_output_dir_is_under_repo_output(
    tmp_path, monkeypatch
) -> None:
    from gnn.execute.executor import GNNExecutor

    executor = GNNExecutor()
    assert executor.output_dir == REPO_ROOT / "output" / "12_execute_output"
