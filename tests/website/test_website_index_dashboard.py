"""Dashboard-fold tests for the generated index page.

Pins the rich dashboard data folded into ``index.html`` by
``WebsiteGenerator._page_index``: the canonical execution summary's overall
badge / end time / duration / peak-memory meta, the registry+filesystem
artifact browser (independent of the summary's ``output_dir`` records),
per-step memory receipts with graceful ``—`` fallbacks, the truthful
pending state when no summary exists, and the absence of external
http(s) resource references.
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


_SUMMARY: dict[str, Any] = {
    "overall_status": "FAILED",
    "end_time": "2026-09-25T12:34:56",
    "total_duration_seconds": 83.4,
    "performance_summary": {"peak_memory_mb": 512.0},
    "steps": [
        {
            "step_number": 1,
            "description": "Environment setup",
            "status": "SUCCESS",
            "memory_usage_mb": 96.5,
            "peak_memory_mb": 128.0,
            "memory_delta_mb": 8.0,
        },
        {
            "step_number": 3,
            "description": "GNN file processing",
            "status": "FAILED",
            "memory_usage_mb": None,
            "peak_memory_mb": 256.0,
            "memory_delta_mb": None,
        },
    ],
}


def _write_summary(root: Path, payload: dict[str, Any]) -> None:
    """Write the canonical execution summary under ``root``."""
    summary_dir = root / "00_pipeline_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _numbered_output_dirs(root: Path) -> None:
    """Create two registry output directories with files, incl. a subdir.

    Directory names come from the site catalogue itself
    (``StepInfo.output_dir_name``) so the test never hardcodes the
    numbering convention.
    """
    from gnn.website import PIPELINE_STEPS

    by_number = {step.number: step for step in PIPELINE_STEPS}
    first = root / by_number[1].output_dir_name
    first.mkdir(parents=True)
    (first / "env_report.txt").write_text("ok", encoding="utf-8")
    subdir = first / "logs"
    subdir.mkdir()
    (subdir / "nested.log").write_text("log", encoding="utf-8")
    third = root / by_number[3].output_dir_name
    third.mkdir(parents=True)
    (third / "gnn_report.json").write_text("{}", encoding="utf-8")


def _generate(tmp_path: Any) -> tuple[dict[str, Any], Path]:
    """Generate the site for ``tmp_path`` and return ``(result, index_path)``."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    site = tmp_path / "site"
    generator = WebsiteGenerator()
    result = generator.generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(site),
            "pipeline_output_root": str(tmp_path),
        }
    )
    return result, site / "index.html"


class TestIndexDashboardFold:
    """The index page folds the canonical dashboard data."""

    def test_summary_data_renders_on_index(self, tmp_path: Any) -> None:
        _write_summary(tmp_path, _SUMMARY)
        _numbered_output_dirs(tmp_path)
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        assert "FAILED" in html  # overall badge label (error tier)
        assert "2026-09-25T12:34:56" in html  # end_time string
        assert "1m 23.4s" in html  # formatted total_duration_seconds
        assert "512.0 MB" in html  # aggregate peak memory receipt
        assert "env_report.txt" in html  # artifact preview entry
        assert "logs/nested.log" in html  # subdirectory file, relative path
        assert "gnn_report.json" in html  # artifact preview entry
        assert "96.5 MB" in html  # per-step memory receipt value
        assert "—</td>" in html  # null memory renders as an em-dash cell

    def test_artifact_found_without_summary_output_dir_key(self, tmp_path: Any) -> None:
        payload = dict(_SUMMARY)
        _write_summary(tmp_path, payload)
        _numbered_output_dirs(tmp_path)
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        # Discovered from the registry + filesystem only; the summary step
        # records carry no output_dir keys at all.
        assert "gnn_report.json" in html
        assert "env_report.txt" in html

    def test_missing_summary_keeps_truthful_pending_state(self, tmp_path: Any) -> None:
        _numbered_output_dirs(tmp_path)
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        assert "PENDING" in html
        assert "No memory receipts recorded." in html
        assert "2026-09-25T12:34:56" not in html
        # The registry+filesystem artifact view is summary-independent.
        assert "env_report.txt" in html

    def test_artifact_preview_caps_at_ten_names(self, tmp_path: Any) -> None:
        from gnn.website import PIPELINE_STEPS

        by_number = {step.number: step for step in PIPELINE_STEPS}
        target = tmp_path / by_number[1].output_dir_name
        target.mkdir(parents=True)
        for i in range(14):
            (target / f"artifact_{i:02d}.txt").write_text("x", encoding="utf-8")
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        assert html.count("artifact_") == 10  # capped preview
        assert "…+4 more" in html  # explicit truncation marker

    def test_no_external_http_references(self, tmp_path: Any) -> None:
        _write_summary(tmp_path, _SUMMARY)
        _numbered_output_dirs(tmp_path)
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        # ld+json @context is a vocabulary IDENTIFIER (never fetched), not a
        # remote resource; assert no fetched external refs by stripping the
        # JSON-LD blocks before the raw-URL grep.
        import re as _re

        fetched_surface = _re.sub(
            r'<script type="application/ld\+json">.*?</script>',
            "",
            html,
            flags=_re.DOTALL,
        )
        assert "http://" not in fetched_surface
        assert "https://" not in fetched_surface
