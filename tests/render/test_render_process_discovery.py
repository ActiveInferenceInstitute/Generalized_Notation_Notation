"""Regression tests for process_render recursive discovery of nested exemplar GNN files.

Verifies recursive discovery in ``gnn.render.processor``:

1. ``process_render(..., recursive=True)`` walks every nested model folder.
   All 38 model sources receive current receipts: 30 render and eight unsupported
   model contracts receive explicit receipts. Private malformed fixtures retain
   strict scientific rejection coverage. Documentation is excluded.
2. Passing ``recursive=False`` via kwargs reverts to a top-level-only glob, so
   no nested files are found and ``process_render`` returns exit code ``2``.

Kept fast: no Julia is executed, only code generation and summary JSON checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gnn.render.processor import process_render

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEMPLAR_DIR = REPO_ROOT / "input" / "gnn_files"
EXPECTED_EXEMPLAR_COUNT = 38
EXPECTED_UNSUPPORTED_SOURCES = {
    "hierarchical/hierarchical_pomdp.md": "unsupported-execution-contract",
    "hierarchical/temporal_hierarchy.md": "unsupported-execution-contract",
    "discrete/tmaze_epistemic.md": "unsupported-execution-contract",
    "continuous/multi_agent_lgssm.md": "unsupported-composition",
    "continuous/hybrid_discrete_continuous.md": "unsupported-composition",
    "continuous/factored_continuous_lgssm.md": "unsupported-factored-continuous",
    "discrete/time_varying_dynamics.md": "unsupported-nonstationary",
    "discrete/regime_switched_dynamics.md": "unsupported-nonstationary",
}
EXPECTED_RENDERED_COUNT = EXPECTED_EXEMPLAR_COUNT - len(EXPECTED_UNSUPPORTED_SOURCES)


def _count_exemplar_md_files() -> int:
    """Count GNN exemplar model files, matching the processor's discovery policy."""
    from gnn.processing.discovery import is_model_source_path

    return sum(1 for path in EXEMPLAR_DIR.rglob("*.md") if is_model_source_path(path))


def test_process_render_recursive_discovers_and_renders_all_exemplars(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "render_out"

    result = process_render(
        target_dir=EXEMPLAR_DIR,
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
    )

    # Unsupported contracts are explicitly receipted without fabricated artifacts.
    assert result is True

    summary_path = output_dir / "render_processing_summary.json"
    assert summary_path.exists(), (
        f"render_processing_summary.json not written to {output_dir}"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # (1) Recursive discovery is the fix under test: all exemplars found.
    assert summary["total_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["total_files"] == _count_exemplar_md_files()

    from gnn.processing.discovery import is_model_source_path

    selected = {
        str(path.resolve())
        for path in EXEMPLAR_DIR.rglob("*.md")
        if is_model_source_path(path)
    }
    assert set(summary["file_results"]) == selected
    assert not any(Path(source).name == "INDEX.md" for source in selected)
    assert summary["successful_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["failed_files"] == 0
    assert summary["total_framework_attempts"] == EXPECTED_RENDERED_COUNT
    assert summary["successful_framework_renderings"] == EXPECTED_RENDERED_COUNT
    assert summary["failed_framework_renderings"] == []
    assert len(summary["unsupported_framework_renderings"]) == len(
        EXPECTED_UNSUPPORTED_SOURCES
    )

    for source, record in summary["file_results"].items():
        path = Path(source)
        relative = str(path.relative_to(EXEMPLAR_DIR))
        assert record["source_identity"] == {
            "path": source,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        rendering = record["framework_results"]["rxinfer"]
        if relative in EXPECTED_UNSUPPORTED_SOURCES:
            assert record["overall_success"] is True
            assert rendering["status"] == "unsupported"
            assert rendering["unsupported"] is True
            assert EXPECTED_UNSUPPORTED_SOURCES[relative] in rendering["message"]
            assert not rendering.get("output_files")
            assert not rendering["artifact_identities"]
        else:
            assert record["overall_success"] is True
            assert rendering["success"] is True
            assert rendering["artifact_identities"]
            for artifact in rendering["artifact_identities"]:
                artifact_path = Path(artifact["path"])
                assert artifact_path.is_relative_to(output_dir)
                assert (
                    artifact["sha256"]
                    == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                )

    # Each plain rendered exemplar produces exactly one RxInfer.jl artifact.
    rendered_jl = list(output_dir.rglob("*.jl"))
    assert len(rendered_jl) == EXPECTED_RENDERED_COUNT
    assert not any(
        Path(source).stem in path.name
        for source in EXPECTED_UNSUPPORTED_SOURCES
        for path in rendered_jl
    )


def test_process_render_private_invalid_source_has_failed_receipt(
    tmp_path: Path,
) -> None:
    """A malformed source fails the required run without a repaired artifact."""
    source = EXEMPLAR_DIR / "basics/static_perception.md"
    source_bytes = source.read_bytes()
    content = source_bytes.decode()
    valid = "A={\n  (0.9, 0.2),\n  (0.1, 0.8)\n}"
    assert content.count(valid) == 1
    selected = tmp_path / "input"
    selected.mkdir()
    malformed = selected / "invalid_likelihood.md"
    malformed.write_text(content.replace(valid, "A={\n  (0.9, 0.1),\n  (0.2, 0.8)\n}"))
    malformed_bytes = malformed.read_bytes()
    output_dir = tmp_path / "output"
    assert process_render(selected, output_dir, frameworks=["rxinfer"]) is False
    summary = json.loads((output_dir / "render_processing_summary.json").read_text())
    assert summary["total_files"] == summary["failed_files"] == 1
    assert (
        summary["successful_files"] == summary["successful_framework_renderings"] == 0
    )
    assert summary["total_framework_attempts"] == 1
    assert len(summary["failed_framework_renderings"]) == 1
    assert summary["unsupported_framework_renderings"] == []
    assert set(summary["file_results"]) == {str(malformed.resolve())}
    record = summary["file_results"][str(malformed.resolve())]
    assert record["source_identity"] == {
        "path": str(malformed.resolve()),
        "sha256": hashlib.sha256(malformed_bytes).hexdigest(),
    }
    assert record["overall_success"] is False
    receipt = record["framework_results"]["rxinfer"]
    assert receipt["success"] is False and receipt["status"] == "failed"
    assert receipt.get("unsupported") is not True
    assert "probability mass must be one" in receipt["message"]
    assert not receipt["output_files"] and not receipt["artifact_identities"]
    assert not list(output_dir.rglob("*.jl"))
    assert malformed.read_bytes() == malformed_bytes
    assert source.read_bytes() == source_bytes


def test_process_render_recursive_false_skips_nested_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "render_out"

    result = process_render(
        target_dir=EXEMPLAR_DIR,
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        recursive=False,
    )

    # There are no top-level "*.md" files, so recursion disabled finds nothing
    # and the processor returns exit code 2 (no input).
    assert result == 2

    summary_path = output_dir / "render_processing_summary.json"
    assert not summary_path.exists()
    assert not list(output_dir.rglob("*.jl"))


def test_process_render_aggregates_summary_across_invocations(tmp_path: Path) -> None:
    """Only explicitly same-run folder calls may carry byte-bound results.

    The pipeline freezes a corpus once. Standalone callers can explicitly bind
    multiple folder calls to one run; a new run must exclude prior evidence.
    """
    output_dir = tmp_path / "render_out"

    first = process_render(
        target_dir=EXEMPLAR_DIR / "precision",
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        run_id="folder-aggregation-test",
    )
    assert first is True

    summary = json.loads(
        (output_dir / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    first_total = summary["total_files"]
    first_keys = set(summary["file_results"])
    assert first_total == len(first_keys) == 2

    second = process_render(
        target_dir=EXEMPLAR_DIR / "discrete",
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        run_id="folder-aggregation-test",
    )
    assert second is True  # Semantic and nonstationary models are unsupported.

    summary = json.loads(
        (output_dir / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    merged_keys = set(summary["file_results"])

    # The second invocation carried forward the first folder's results and
    # added its own: nothing was dropped, and the aggregate count matches.
    assert first_keys <= merged_keys
    assert len(merged_keys) > first_total
    assert summary["total_files"] == len(merged_keys)
    assert any("precision/" in key for key in merged_keys)
    assert any("discrete/" in key for key in merged_keys)
    expected_discrete = {
        str(path.resolve()) for path in (EXEMPLAR_DIR / "discrete").glob("*.md")
    }
    assert merged_keys == first_keys | expected_discrete
    assert summary["failed_files"] == 0
    assert summary["successful_files"] == len(merged_keys)
    assert summary["receipt_identity"]["run_id"] == "folder-aggregation-test"

    third = process_render(
        target_dir=EXEMPLAR_DIR / "precision",
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        run_id="fresh-folder-run",
    )
    assert third is True
    fresh = json.loads(
        (output_dir / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    assert fresh["receipt_identity"]["run_id"] == "fresh-folder-run"
    assert set(fresh["file_results"]) == first_keys
    assert fresh["total_files"] == fresh["successful_files"] == 2
    assert fresh["failed_files"] == 0
    assert not set(fresh["file_results"]) & expected_discrete
    assert list((output_dir / "history").glob("render-*.json"))
