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
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

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

    def test_worker_receipts_distinguish_limits_from_recorded_concurrency(
        self, tmp_path: Path
    ) -> None:
        from gnn.website import WebsiteGenerator

        summary = {
            **_SUMMARY,
            "execution_resources": {
                "logical_cpus_available": 12,
                "file_workers_requested": 4,
                "step_workers_requested": None,
                "execution_workers_requested": 1,
            },
            "parallel_tiers": [
                {
                    "tier_index": 2,
                    "step_names": ["3_gnn", "5_type_checker <recorded>"],
                    "workers_used": 2,
                    "memory_estimate_mb": 48.5,
                },
                None,
            ],
        }
        result = WebsiteGenerator().generate_website(
            {"output_dir": tmp_path / "site", "pipeline_summary": summary},
            filesystem=False,
        )
        assert result["success"]
        html = (tmp_path / "site/index.html").read_text()
        assert "CPU and Worker Receipts" in html
        assert "File worker limit (Steps 3/5)" in html
        assert "Execution worker limit (Step 12)" in html
        assert 'class="dash-meta__value">Automatic</span>' in html
        assert "Recorded concurrent step tiers" in html
        assert "5_type_checker &lt;recorded&gt;" in html
        assert "<td>2</td><td>48.5 MB</td>" in html
        assert "neither establishes actual per-file worker use or speedup" in html

        # Missing receipts cannot inherit this machine's CPU count or limits.
        result = WebsiteGenerator().generate_website(
            {"output_dir": tmp_path / "empty"}, filesystem=False
        )
        assert result["success"]
        assert (
            "CPU and Worker Receipts" not in (tmp_path / "empty/index.html").read_text()
        )

        summary["execution_resources"] = {
            "file_workers_requested": True,
            "step_workers_requested": "<invalid>",
        }
        summary["parallel_tiers"] = "invalid"
        result = WebsiteGenerator().generate_website(
            {"output_dir": tmp_path / "malformed", "pipeline_summary": summary},
            filesystem=False,
        )
        assert result["success"]
        html = (tmp_path / "malformed/index.html").read_text()
        assert "<invalid>" not in html and ">True<" not in html
        assert "Recorded concurrent step tiers" not in html

    def test_artifact_found_without_summary_output_dir_key(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
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
        assert "filesystem census" in html

        # A bound run must not promote inherited files into its artifact view.
        from gnn.pipeline.run_context import build_run_context
        from gnn.website import WebsiteGenerator
        from gnn.website.collection import website_data_from_dict

        context = build_run_context(tmp_path / "input", tmp_path, "site-run", [20], {})
        context_path = tmp_path / "run_context.json"
        context.write(context_path)
        monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_path))
        monkeypatch.setenv("GNN_RUN_ID", context.run_id)
        current = tmp_path / "00_pipeline_summary/current_summary.json"
        current.write_text(
            json.dumps(
                {
                    "run_id": context.run_id,
                    "steps": [
                        {"artifacts": [{"path": "3_gnn_output/gnn_report.json"}]}
                    ],
                }
            ),
            encoding="utf-8",
        )
        data = website_data_from_dict(
            {"pipeline_output_root": Path(os.path.relpath(tmp_path, Path.cwd()))},
            output_dir=tmp_path / "site",
        )
        data["step_statuses"] = {1: "ok", 3: "warning"}
        current_html = WebsiteGenerator()._page_index(data)
        assert "Current run artifacts" in current_html
        assert "gnn_report.json" in current_html
        assert "env_report.txt" not in current_html
        assert "nested.log" not in current_html
        assert "No current run artifacts recorded." in current_html
        assert re.search(
            r"Steps Complete</div>\s*<div class=\"value\">2</div>", current_html
        )

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
        special = target / "part #1 ? é.json"
        special.write_text("{}", encoding="utf-8")
        outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
        outside.write_text("outside", encoding="utf-8")
        (target / "external-link.txt").symlink_to(outside)
        result, index = _generate(tmp_path)

        assert result["success"] is True
        html = index.read_text(encoding="utf-8")
        assert "Show 5 more files" in html
        assert '<details class="dash-more">' in html
        assert "external-link.txt" not in html
        links = re.findall(r'<li><a href="([^"]+)">', html)
        assert len(links) == 15
        resolved = {(index.parent / unquote(href)).resolve() for href in links}
        assert resolved == {
            path.resolve()
            for path in target.iterdir()
            if path != target / "external-link.txt"
        }
        assert any(
            "%23" in href and "%3F" in href and "%C3%A9" in href for href in links
        )

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

    def test_memory_receipts_join_by_script_name_to_registry_step(
        self, tmp_path: Any
    ) -> None:
        """A real summary numbers steps by execution position, not by registry
        step; the receipts table must join them by ``script_name``."""
        from gnn.website import WebsiteGenerator

        summary = {
            "overall_status": "SUCCESS_WITH_WARNINGS",
            "steps": [
                {
                    "step_number": 1,
                    "script_name": "3_gnn.py",
                    "description": "GNN file processing",
                    "status": "SUCCESS",
                    "memory_usage_mb": 111.0,
                    "peak_memory_mb": 222.0,
                    "memory_delta_mb": 1.0,
                },
            ],
        }
        result = WebsiteGenerator().generate_website(
            {"output_dir": tmp_path / "site", "pipeline_summary": summary},
            filesystem=False,
        )
        assert result["success"]
        html = (tmp_path / "site/index.html").read_text(encoding="utf-8")
        assert re.search(
            r"<td><code>03</code></td>\s*<td>GNN file processing</td>\s*"
            r"<td>111\.0 MB</td>",
            html,
        )
        assert not re.search(
            r"<td><code>01</code></td>\s*<td>GNN file processing</td>", html
        )

    def test_preliminary_summary_labels_snapshot_not_finished(
        self, tmp_path: Any
    ) -> None:
        """A preliminary summary is a live snapshot, not a finished run."""
        from gnn.website import WebsiteGenerator

        summary = {
            "overall_status": "RUNNING",
            "preliminary": True,
            "evidence_phase": "current_run_snapshot",
            "end_time": "2026-11-02T12:09:54",
            "steps": [],
        }
        result = WebsiteGenerator().generate_website(
            {"output_dir": tmp_path / "site", "pipeline_summary": summary},
            filesystem=False,
        )
        assert result["success"]
        html = (tmp_path / "site/index.html").read_text(encoding="utf-8")
        assert ">Snapshot<" in html
        assert "Preliminary snapshot" in html
        assert ">Finished<" not in html
