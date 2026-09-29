#!/usr/bin/env python3
"""Every step's model discovery must skip documentation markdown.

``input/gnn_files`` ships README.md, AGENTS.md and INDEX.md beside the models.
A full run showed Steps 10, 13, 14 and 19 treating them as models (the LLM
step counted 41 "models" for 36 real ones). Each discovery path is exercised
here against one real model plus the documentation files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
DOC_NAMES = ("README.md", "AGENTS.md", "INDEX.md")


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "gnn_files"
    (root / "family").mkdir(parents=True)
    shutil.copy(MODEL, root / "family" / MODEL.name)
    for name in DOC_NAMES:
        (root / name).write_text("# docs\n", encoding="utf-8")
        (root / "family" / name).write_text("# docs\n", encoding="utf-8")
    return root


def test_research_discovery_skips_docs(tmp_path: Path) -> None:
    from gnn.research.processor import discover_gnn_files

    found = discover_gnn_files(_tree(tmp_path), recursive=True)
    assert [p.name for p in found] == [MODEL.name]


def test_ml_integration_discovery_skips_docs(tmp_path: Path) -> None:
    from gnn.ml_integration.processor import _discover_gnn_files

    found = _discover_gnn_files(_tree(tmp_path), recursive=True)
    assert [p.name for p in found] == [MODEL.name]


def test_ontology_step_processes_only_models(tmp_path: Path) -> None:
    from gnn.ontology.processor import process_ontology

    out = tmp_path / "out"
    out.mkdir()
    process_ontology(_tree(tmp_path), out, recursive=True)
    results = json.loads((out / "ontology_results.json").read_text())
    assert results["processed_files"] == 1
