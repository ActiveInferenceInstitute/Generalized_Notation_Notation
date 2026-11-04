"""The non-POMDP render fallback must refuse, not emit a template program.

Before this contract, `_process_single_gnn_file_basic` called the legacy
``generate_pymdp_code`` / ``generate_bnlearn_code`` templates with an empty model
(``{"model_name": stem, "variables": [], "connections": []}``) and reported success,
so a spec that produced no POMDP state space still yielded a ``*_pymdp.py`` /
``*_bnlearn.py`` artifact that represented no specification at all. That is the same
"two entry points, two programs" shape the DisCoPy fallback was fixed for.

The maintained corpus never reaches this branch (all 38 sources extract), so these
tests make it reachable deliberately. Without that, the refusal would be untested and
could silently regress to template emission.

DisCoPy is deliberately left alone: it renders from declared ``Connections`` through
the canonical renderer, so a spec can legitimately render on DisCoPy while PyMDP and
bnlearn refuse. These tests assert that distinction rather than a blanket block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gnn.render.processor import process_render

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "input" / "gnn_files" / "basics" / "static_perception.md"
REFUSAL_PREFIX = "unsupported-pomdp-extraction:"


@pytest.fixture
def no_extraction(monkeypatch):
    """process_render imports the extractor lazily, so patch the source module."""
    monkeypatch.setattr(
        "gnn.extract.pomdp_extractor.extract_pomdp_from_file",
        lambda *a, **k: None,
    )


def _selected_copy(tmp_path: Path, content: str | None = None) -> Path:
    selected = tmp_path / "input"
    selected.mkdir()
    text = SPEC.read_text() if content is None else content
    (selected / SPEC.name).write_text(text)
    return selected


def _summary(output_dir: Path) -> dict:
    return json.loads((output_dir / "render_processing_summary.json").read_text())


def _record(summary: dict, source: Path) -> dict:
    return summary["file_results"][str(source.resolve())]


def test_fallback_refuses_pymdp_and_bnlearn_but_keeps_discopy(
    tmp_path: Path, no_extraction
) -> None:
    selected = _selected_copy(tmp_path)
    output_dir = tmp_path / "output"

    process_render(selected, output_dir, frameworks=["pymdp", "bnlearn", "discopy"])

    written = [p.name for p in output_dir.rglob("*.py")]
    assert not [n for n in written if n.endswith(("_pymdp.py", "_bnlearn.py"))], written
    assert [n for n in written if n.endswith("_discopy.py")], written

    summary = _summary(output_dir)
    record = _record(summary, selected / SPEC.name)
    assert record["status"] == "rendered"

    for framework in ("pymdp", "bnlearn"):
        receipt = record["framework_results"][framework]
        assert receipt["unsupported"] is True
        assert receipt["status"] == "unsupported"
        assert receipt["generated_files"] == []
        assert receipt["message"].startswith(REFUSAL_PREFIX)

    assert record["framework_results"]["discopy"]["success"] is True
    refused = {
        item["framework"] for item in summary["unsupported_framework_renderings"]
    }
    assert {"pymdp", "bnlearn"} <= refused
    assert summary["successful_framework_renderings"] == 1
    assert summary["failed_files"] == 0
    assert summary["unsupported_files"] == 0


def test_refused_file_is_counted_unsupported_not_failed(tmp_path: Path) -> None:
    """A refused file must not be reported as a failed file.

    Exercised against the receipt writer directly: the maintained renderers refuse at
    framework level and DisCoPy still renders an unconnected spec, so a fully-refused
    *file* is not constructible from the corpus today. The accounting rule still has to
    hold, because "failed" must keep meaning "attempted and did not work".
    """
    from gnn.render.processor.receipts import _write_render_receipt

    # The receipt writer hashes the recorded source, so it must exist on disk.
    source_path = tmp_path / "spec.md"
    source_path.write_bytes(SPEC.read_bytes())
    source = str(source_path.resolve())
    record = {
        "overall_success": False,
        "success": False,
        "status": "unsupported",
        "unsupported": True,
        "reason": f"{REFUSAL_PREFIX} nothing extractable",
        "framework_results": {
            framework: {
                "success": False,
                "unsupported": True,
                "status": "unsupported",
                "message": f"{REFUSAL_PREFIX} nothing extractable",
            }
            for framework in ("pymdp", "bnlearn")
        },
    }

    summary = _write_render_receipt(
        tmp_path / "output",
        tmp_path,
        {source: record},
        {},
        "run-fallback-refusal",
        "Basic rendering",
    )

    assert summary["successful_files"] == 0
    assert summary["unsupported_files"] == 1
    assert summary["failed_files"] == 0
    assert summary["unsupported_framework_count"] == 2
    assert summary["successful_framework_renderings"] == 0


def test_canonical_pymdp_program_is_identical_across_entry_points(
    tmp_path: Path,
) -> None:
    """Step 11's directory flow and the processor API write the same program bytes."""
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import POMDPRenderProcessor

    selected = _selected_copy(tmp_path)
    directory_output = tmp_path / "out_dir"
    process_render(selected, directory_output, frameworks=["pymdp"])

    space = extract_pomdp_from_file(SPEC, strict_validation=True)
    assert space is not None
    processor_output = tmp_path / "out_processor"
    processor_output.mkdir()
    POMDPRenderProcessor(processor_output / SPEC.stem).process_pomdp_for_all_frameworks(
        space, gnn_file_path=SPEC, frameworks=["pymdp"]
    )

    from_directory = list(directory_output.rglob("*_pymdp.py"))
    from_processor = list(processor_output.rglob("*_pymdp.py"))
    assert len(from_directory) == len(from_processor) == 1
    assert from_directory[0].read_bytes() == from_processor[0].read_bytes()
