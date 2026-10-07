"""Julia project hygiene and model-family manifest pins.

Pins the Julia project contracts for the two executed Julia backends and the
continuous model-family row:

- Both ``Project.toml`` files parse as TOML, carry real project headers
  (``name`` + ``uuid``), and agree on a single Julia version floor
  (``julia = "1.10"``); the UUIDs must be freshly generated, never the
  former placeholder value.
- Both committed ``Manifest.toml`` files parse as TOML and carry a
  ``project_hash`` (Pkg keeps them in sync with their project).
- The shipped ``input/model_family_manifest.json`` parses via
  ``load_model_family_manifest`` and the continuous family covers the
  ngc-learn LGSSM exemplar with the ``ngclearn`` backend listed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from gnn.pipeline.model_family_acceptance import load_model_family_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
RXINFER_PROJECT = REPO_ROOT / "src/gnn/execute/rxinfer/Project.toml"
RXINFER_MANIFEST = REPO_ROOT / "src/gnn/execute/rxinfer/Manifest.toml"
ACTIVEINFERENCE_PROJECT = REPO_ROOT / "src/gnn/execute/activeinference_jl/Project.toml"
ACTIVEINFERENCE_MANIFEST = (
    REPO_ROOT / "src/gnn/execute/activeinference_jl/Manifest.toml"
)
PLACEHOLDER_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def test_project_tomls_have_real_headers_and_one_julia_floor() -> None:
    rxinfer = tomllib.loads(RXINFER_PROJECT.read_text(encoding="utf-8"))
    activeinference = tomllib.loads(ACTIVEINFERENCE_PROJECT.read_text(encoding="utf-8"))

    for project, expected_name in (
        (rxinfer, "GnnRxInferModels"),
        (activeinference, "GnnActiveInferenceModels"),
    ):
        assert project["name"] == expected_name
        uuid = project["uuid"]
        assert uuid != PLACEHOLDER_UUID
        # Valid UUID shape: 8-4-4-4-12 hex segments.
        segments = uuid.split("-")
        assert [len(s) for s in segments] == [8, 4, 4, 4, 12]
        int(segments[0], 16)  # raises on non-hex placeholder-shaped values
        assert project["authors"] == ["GNN Pipeline"]
        assert project["version"] == "1.0.0"
        assert project["compat"]["julia"] == "1.10"

    assert rxinfer["compat"]["julia"] == activeinference["compat"]["julia"]


def test_committed_manifests_parse_with_project_hash() -> None:
    for manifest_path in (RXINFER_MANIFEST, ACTIVEINFERENCE_MANIFEST):
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["manifest_format"] == "2.0"
        assert manifest.get("project_hash")
        assert manifest["julia_version"].startswith("1.12.")


def test_continuous_family_lists_ngclearn_and_lgssm_exemplar() -> None:
    families = load_model_family_manifest(Path("input/model_family_manifest.json"))
    continuous = {family.name: family for family in families}["continuous"]

    assert continuous.frameworks is not None
    assert "ngclearn" in continuous.frameworks.split(",")
    assert "ngclearn_lgssm.md" in continuous.representative_files
    assert (REPO_ROOT / "input/gnn_files/continuous/ngclearn_lgssm.md").is_file()
