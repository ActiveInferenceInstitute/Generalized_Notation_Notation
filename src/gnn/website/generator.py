#!/usr/bin/env python3
"""
Website generator module for GNN pipeline.

Generates a full-featured, premium, dark-mode static HTML website
from GNN pipeline artifacts. Produces the seven site pages, one page per
parsed GNN model, and a client-side search index:
  - index.html         — Pipeline dashboard with step cards
  - pipeline.html      — Full 25-step pipeline status table
  - gnn_files.html     — GNN source file browser
  - analysis.html      — Statistical analysis results
  - visualization.html — Gallery of all generated visualizations
  - reports.html       — JSON/text report viewer
  - mcp.html           — MCP tools registry across all modules
  - model/<slug>.html  — one page per parsed GNN source model
  - search-index.json  — client-side search index (also inlined on the
                         gnn_files page because fetch() fails on file://)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gnn.website.collection import collect_website_data, website_data_from_dict
from gnn.website.model_navigation import (
    _artifact_matches_model as _artifact_matches_model,
)
from gnn.website.model_navigation import (
    _attach_model_viz_assets as _attach_model_viz_assets,
)
from gnn.website.model_navigation import (
    _build_search_data as _build_search_data,
)
from gnn.website.model_navigation import (
    _build_search_pages as _build_search_pages,
)
from gnn.website.model_navigation import (
    _ensure_model_slugs as _ensure_model_slugs,
)
from gnn.website.model_navigation import (
    _model_slug as _model_slug,
)
from gnn.website.model_navigation import (
    _model_snippet as _model_snippet,
)
from gnn.website.pages import SITE_PAGES
from gnn.website.steps import (  # noqa: F401  (re-export)
    PIPELINE_STEPS,
    StepInfo,
    get_pipeline_steps,
)
from gnn.website.templates import (
    _CSS as _CSS,
)
from gnn.website.templates import (
    _SEARCH_JS as _SEARCH_JS,
)
from gnn.website.templates import (
    _esc as _esc,
)
from gnn.website.templates import (
    _page as _page,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Shared design system — dark-mode CSS
# ─────────────────────────────────────────────────────────────────────────────


# Shared status-badge styling: single source of truth for the index and
# pipeline pages (previously duplicated inline in each page builder).
_BADGE_CLASS: dict[str, str] = {
    "ok": "badge-ok",
    "error": "badge-error",
    "skip": "badge-skip",
}
_STEP_BADGE_LABEL: dict[str, str] = {
    "ok": "✓ Complete",
    "error": "✗ Error",
    "skip": "⊘ Skipped",
}
_PIPELINE_BADGE_LABEL: dict[str, str] = {
    "ok": "Complete",
    "error": "Error",
    "skip": "Skipped",
}


_OVERALL_BADGE_CLASS: dict[str, str] = {
    "SUCCESS": "badge-ok",
    "SUCCESS_WITH_WARNINGS": "badge-warning",
    "PARTIAL_SUCCESS": "badge-warning",
    "FAILED": "badge-error",
}


def _stat_int(value: Any, default: int = 0) -> int:
    """Best-effort int for artifact-recorded stat values, falling back to ``default``.

    Guard against ``null``/non-numeric JSON fields: an unusable stat must
    render as the fallback instead of failing the whole page.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars with an explicit marker when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n… [truncated]"


def _fmt_secs(value: Any) -> str:
    """Format a duration in seconds (``—`` when absent or unusable)."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m"


def _fmt_mb(value: Any) -> str:
    """Format a megabyte figure (``—`` when absent or unusable)."""
    try:
        megabytes = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{megabytes:.1f} MB"


def _overall_badge(status: Any) -> str:
    """Badge fragment for the canonical summary's overall status.

    Case-tolerant normalization: SUCCESS → ok tier, SUCCESS_WITH_WARNINGS /
    PARTIAL_SUCCESS → warning tier, FAILED → error tier, and anything
    unknown (including an absent summary) → the truthful pending tier.
    """
    normalized = str(status or "").strip().upper() or "PENDING"
    badge_cls = _OVERALL_BADGE_CLASS.get(normalized, "badge-pending")
    return f'<span class="step-badge {badge_cls}">{_esc(normalized)}</span>'


def _step_dir_stats(
    step: StepInfo, p_root: Path, cap: int = 10
) -> tuple[bool, int, int, list[str], bool]:
    """Filesystem facts for one registry step's output directory.

    Registry + filesystem only — per-step ``output_dir`` summary fields are
    never consulted. Returns ``(exists, file_count, total_bytes, preview,
    truncated)`` where ``preview`` holds up to ``cap`` sorted relative names.
    """
    directory = p_root / step.output_dir_name
    if not directory.is_dir():
        return False, 0, 0, [], False
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    preview = [f.relative_to(directory).as_posix() for f in files[:cap]]
    return (
        True,
        len(files),
        sum(f.stat().st_size for f in files),
        preview,
        len(files) > cap,
    )


def _write_atomic(dest: Path, content: str) -> None:
    """Write ``content`` to ``dest`` via a temp file and atomic rename."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dest.parent, delete=False
    )
    try:
        with tmp:
            tmp.write(content)
        os.replace(tmp.name, dest)
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


class WebsiteGenerator:
    """Generates a premium multi-page static HTML website from pipeline artifacts."""

    def __init__(self) -> None:
        """Initialize the instance.

        MCP page data is sourced from the step-21 artifacts in the pipeline
        output root at generation time; no constructor configuration needed.
        """

    # ── Public API ──────────────────────────────────────────────────────────
    def _page_builders(self) -> dict[str, Callable[[dict], str]]:
        """Filename → page-builder mapping, derived from ``SITE_PAGES``."""
        return {page.filename: getattr(self, page.builder) for page in SITE_PAGES}

    def generate_website(self, website_data: dict, *, filesystem: bool = True) -> dict:
        """Generate the complete static website.

        Each page is rendered and written independently: a failure on one
        page records an error and leaves the remaining pages intact, and the
        overall ``success`` flag is ``True`` only when no errors occurred.

        Pure-dict mode (``filesystem=False``): no disk collectors run — the
        data dict is built from ``website_data`` itself via
        ``website_data_from_dict`` (missing datasets take the collectors'
        empty defaults; the only filesystem work is writing the site).
        """
        result: dict[str, Any] = {
            "success": True,
            "pages_created": 0,
            "pages": [],
            "model_pages_created": 0,
            "model_pages": [],
            "errors": [],
            "warnings": [],
        }
        try:
            output_dir = Path(
                website_data.get("output_dir", "output/20_website_output")
            )

            if filesystem:
                input_dir = Path(website_data.get("input_dir", "output"))
                p_root = Path(website_data.get("pipeline_output_root", str(input_dir)))
                output_dir.mkdir(parents=True, exist_ok=True)
                assets_dir = output_dir / "assets"
                assets_dir.mkdir(exist_ok=True)
                data = self._collect_all_data(
                    p_root, input_dir, output_dir, assets_dir, website_data
                )
            else:
                # Pure-dict mode writes only where the caller explicitly
                # aims it: without an explicit ``output_dir`` the default
                # would silently target the repository's ``output/`` tree.
                if "output_dir" not in website_data:
                    raise ValueError(
                        "filesystem=False requires an explicit 'output_dir' "
                        "in website_data"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                assets_dir = output_dir / "assets"
                assets_dir.mkdir(exist_ok=True)
                data = website_data_from_dict(website_data, output_dir=str(output_dir))

            # One shared slug map for the per-model pages, the listing deep
            # links, and the search index: assigned once, in collection
            # order (sorted GNN filenames), before any page renders.
            models = data.get("models") or []
            _ensure_model_slugs(models)
            _attach_model_viz_assets(data)
            data["search_data"] = _build_search_data(data)
            # Collection warnings (e.g. visualization artifacts skipped
            # because their copy failed) surface in the result manifest
            # instead of dying inside the collectors.
            _collected_warnings = data.get("warnings")
            if isinstance(_collected_warnings, list):
                result["warnings"].extend(str(w) for w in _collected_warnings)

            builders = self._page_builders()
            for filename, build_page in builders.items():
                try:
                    page_html = build_page(data)
                except (ValueError, TypeError, KeyError, IndexError) as e:
                    result["errors"].append(f"Failed to render {filename}: {e}")
                    continue
                try:
                    _write_atomic(output_dir / filename, page_html)
                    result["pages_created"] += 1
                    result["pages"].append(filename)
                except OSError as e:
                    result["errors"].append(f"Failed to write {filename}: {e}")

            # Per-model pages (C2): written under model/ and bookkept under
            # the model_* keys only — pages/pages_created stay pinned to the
            # seven site pages.
            if models:
                model_dir = output_dir / "model"
                model_dir.mkdir(exist_ok=True)
                for model in models:
                    slug = str(model["slug"])
                    model_filename = f"model/{slug}.html"
                    try:
                        page_html = self._page_model(model)
                    except (ValueError, TypeError, KeyError, IndexError) as e:
                        result["errors"].append(
                            f"Failed to render {model_filename}: {e}"
                        )
                        continue
                    try:
                        _write_atomic(model_dir / f"{slug}.html", page_html)
                        result["model_pages_created"] += 1
                        result["model_pages"].append(model_filename)
                    except OSError as e:
                        result["errors"].append(
                            f"Failed to write {model_filename}: {e}"
                        )

            # Client-side search index (C3): standalone JSON at the site
            # root; the listing page inlines the same payload (see
            # _search_html) because fetch() fails on file://.
            try:
                _write_atomic(
                    output_dir / "search-index.json",
                    json.dumps(data["search_data"], indent=2),
                )
            except OSError as e:
                result["errors"].append(f"Failed to write search-index.json: {e}")

        except Exception as e:
            result["errors"].append(str(e))

        result["success"] = not result["errors"]
        return result

    def create_pages(self, output_dir: Path, data: dict) -> dict:
        """Create individual website pages (compatibility API)."""
        return self.generate_website({**data, "output_dir": str(output_dir)})

    # ── Data collection ─────────────────────────────────────────────────────

    def _collect_all_data(
        self,
        p_root: Path,
        input_dir: Path,
        output_dir: Path,
        assets_dir: Path,
        user_data: dict,
    ) -> dict:
        """Collect all data (delegates to the pure ``collect_website_data``)."""

        return collect_website_data(
            p_root,
            input_dir,
            assets_dir,
            output_dir=output_dir,
            user_data=user_data,
        )

    # ── Page generators ─────────────────────────────────────────────────────

    def _page_index(self, data: dict) -> str:
        """Render the dashboard landing page."""
        n_ok = sum(1 for s in data["step_statuses"].values() if s == "ok")
        n_steps = len(PIPELINE_STEPS)
        n_files = data["processed_files"]
        mcp_summary = data.get("mcp_summary")
        n_tools = _stat_int(
            mcp_summary.get("tools_registered", 0)
            if isinstance(mcp_summary, dict)
            else 0
        )

        stats = f"""
<div class="stats-row">
  <div class="stat-card success">
    <div class="label">Steps Complete</div>
    <div class="value">{n_ok}</div>
    <div class="sub">of {n_steps} pipeline steps</div>
  </div>
  <div class="stat-card accent">
    <div class="label">GNN Files</div>
    <div class="value">{n_files}</div>
    <div class="sub">source models</div>
  </div>
  <div class="stat-card accent2">
    <div class="label">MCP Tools</div>
    <div class="value">{n_tools}</div>
    <div class="sub">registered tools</div>
  </div>
  <div class="stat-card accent">
    <div class="label">Visualizations</div>
    <div class="value">{len(data["visualizations"])}</div>
    <div class="sub">generated artifacts</div>
  </div>
</div>"""

        # Step grid
        cards = ""
        for step in PIPELINE_STEPS:
            status = data["step_statuses"].get(step.number, "pending")
            badge_cls = _BADGE_CLASS.get(status, "badge-pending")
            badge_label = _STEP_BADGE_LABEL.get(status, "○ Pending")
            cards += f"""
<div class="step-card">
  <div class="step-num">STEP {step.number:02d}</div>
  <div class="step-name">{_esc(step.name)}</div>
  <div class="step-desc">{_esc(step.description)}</div>
  <span class="step-badge {badge_cls}">{badge_label}</span>
</div>"""

        # Pipeline run header — canonical execution summary only; every value
        # degrades to the truthful pending/"—" state when the summary is absent.
        summary = data.get("pipeline_summary")
        if not isinstance(summary, dict):
            summary = {}
        performance = summary.get("performance_summary")
        if not isinstance(performance, dict):
            performance = {}
        run_header = f"""
<div class="section">
  <div class="section-title">Pipeline Run</div>
  <div class="dash-meta">
    <div class="dash-meta__item">
      <span class="dash-meta__label">Overall status</span>
      {_overall_badge(summary.get("overall_status"))}
    </div>
    <div class="dash-meta__item">
      <span class="dash-meta__label">Finished</span>
      <span class="dash-meta__value">{_esc(summary.get("end_time") or "—")}</span>
    </div>
    <div class="dash-meta__item">
      <span class="dash-meta__label">Duration</span>
      <span class="dash-meta__value">{_fmt_secs(summary.get("total_duration_seconds"))}</span>
    </div>
    <div class="dash-meta__item">
      <span class="dash-meta__label">Peak memory</span>
      <span class="dash-meta__value">{_fmt_mb(performance.get("peak_memory_mb"))}</span>
    </div>
  </div>
</div>"""

        # Memory receipts — rows joined to the step grid by step number.
        summary_steps: dict[int, dict[str, Any]] = {}
        raw_steps = summary.get("steps")
        if isinstance(raw_steps, list):
            for raw in raw_steps:
                if isinstance(raw, dict) and isinstance(raw.get("step_number"), int):
                    summary_steps[raw["step_number"]] = raw
        if summary_steps:
            registry_names = {step.number: step.name for step in PIPELINE_STEPS}
            mem_rows = ""
            for number in sorted(set(registry_names) | set(summary_steps)):
                record = summary_steps.get(number)
                name = (
                    (record.get("description") if record else "")
                    or registry_names.get(number)
                    or f"step {number:02d}"
                )
                mem_rows += f"""<tr>
  <td><code>{number:02d}</code></td>
  <td>{_esc(name)}</td>
  <td>{_fmt_mb(record.get("memory_usage_mb") if record else None)}</td>
  <td>{_fmt_mb(record.get("peak_memory_mb") if record else None)}</td>
  <td>{_fmt_mb(record.get("memory_delta_mb") if record else None)}</td>
</tr>"""
            memory = f"""
<div class="section">
  <div class="section-title">Memory Receipts</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Step</th><th>Usage</th><th>Peak</th><th>Δ</th></tr></thead>
      <tbody>{mem_rows}</tbody>
    </table>
  </div>
</div>"""
        else:
            memory = """
<div class="section">
  <div class="section-title">Memory Receipts</div>
  <div class="card"><p>No memory receipts recorded.</p>
  <p>The pipeline execution summary did not run or recorded no per-step memory data for this pipeline run.</p></div>
</div>"""

        # Artifact browser — registry + filesystem only; the summary's
        # per-step ``output_dir`` fields are never consulted. Missing
        # directories render as an explicit empty state instead of being
        # dropped.
        p_root = data.get("p_root")
        if p_root is None:
            artifacts = """
<div class="section">
  <div class="section-title">Artifacts</div>
  <div class="card"><p>No pipeline output root available.</p></div>
</div>"""
        else:
            art_cards = ""
            for step in PIPELINE_STEPS:
                exists, count, size, names, truncated = _step_dir_stats(step, p_root)
                if not exists:
                    count_label, body_html = (
                        "not produced",
                        '<p class="dash-empty">Not produced — no output directory on disk.</p>',
                    )
                elif not names:
                    count_label, body_html = (
                        f"{count} files · {size} bytes",
                        '<p class="dash-empty">Output directory exists but holds no files.</p>',
                    )
                else:
                    count_label = (
                        f"{count} file{'s' if count != 1 else ''} · {size} bytes"
                    )
                    items = "".join(f"<li>{_esc(name)}</li>" for name in names)
                    more = (
                        f'<p class="dash-more">…+{count - len(names)} more</p>'
                        if truncated
                        else ""
                    )
                    body_html = f'<ul class="dash-files">{items}</ul>{more}'
                art_cards += f"""
<div class="dash-art">
  <div class="dash-art__head">
    <span class="pill badge-pending">STEP {step.number:02d}</span>
    <span class="dash-art__name">{_esc(step.name)}</span>
    <span class="dash-art__count">{_esc(count_label)}</span>
  </div>
  <div class="dash-art__body">{body_html}</div>
</div>"""
            artifacts = f"""
<div class="section">
  <div class="section-title">Artifacts</div>
  {art_cards}
</div>"""

        body = f"""
<div class="page-header">
  <h1>GNN Pipeline Dashboard</h1>
  <p class="subtitle">Results overview — generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
{stats}
{run_header}
<div class="section">
  <div class="section-title">Pipeline Steps</div>
  <div class="step-grid">{cards}</div>
</div>
{memory}
{artifacts}"""
        return _page("Dashboard", "index", body)

    def _page_pipeline(self, data: dict) -> str:
        """Render the full 25-step pipeline status table."""
        rows = ""
        for step in PIPELINE_STEPS:
            status = data["step_statuses"].get(step.number, "pending")
            badge_cls = _BADGE_CLASS.get(status, "badge-pending")
            badge_label = _PIPELINE_BADGE_LABEL.get(status, "Pending")
            rows += f"""<tr>
  <td><code>{step.number:02d}</code></td>
  <td>{_esc(step.name)}</td>
  <td>{_esc(step.description)}</td>
  <td><span class="step-badge {badge_cls}">{badge_label}</span></td>
  <td><code style="font-size:11px;color:var(--text-3)">{_esc(step.script_name)}</code></td>
</tr>"""
        body = f"""
<div class="page-header">
  <h1>⚡ Pipeline Steps</h1>
  <p class="subtitle">Full 25-step GNN processing pipeline status</p>
</div>
<div class="table-wrap">
  <table>
    <thead><tr><th>#</th><th>Step</th><th>Description</th><th>Status</th><th>Script</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
        return _page("Pipeline", "pipeline", body)

    def _page_gnn_files(self, data: dict) -> str:
        """Render the GNN source file browser.

        Aggregate rows keep the 3000-char preview cap; the model rows, the
        per-model deep links, and the client-side search box are backed by
        the ``model/<slug>.html`` pages and the inline search payload.
        """
        if not data["gnn_files"]:
            content = '<div class="card"><p>No GNN source files found.</p></div>'
        else:
            content = ""
            for gf in data["gnn_files"]:
                try:
                    src = _truncate(
                        gf.read_text(encoding="utf-8", errors="replace"), 3000
                    )
                    size = gf.stat().st_size
                except Exception:
                    src, size = "(could not read)", 0
                content += f"""
<details>
  <summary>{_esc(gf.name)} <span class="pill badge-pending" style="margin-left:8px">{size} bytes</span></summary>
  <div class="details-body"><pre>{_esc(src)}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📂 GNN Source Files</h1>
  <p class="subtitle">{len(data["gnn_files"])} models discovered</p>
</div>
{self._search_html(data)}
{self._model_rows_html(data)}
<div class="section">{content}</div>
<script>{_SEARCH_JS}</script>"""
        return _page("GNN Files", "gnn_files", body)

    # ── Per-model pages and client-side search ──────────────────────────────

    def _search_html(self, data: dict) -> str:
        """Search box fed by the inline search-index payload (file://-safe).

        The inline payload is the standalone ``search-index.json`` content
        with ``</`` escaped as ``<\\/``; fetch() fails on file://, so the
        vanilla filter below reads this inline copy instead.
        """
        search_data = data.get("search_data")
        if search_data is None:
            search_data = _build_search_data(data)
        payload = json.dumps(search_data, indent=2).replace("</", "<\\/")
        return f"""
<div class="section site-search">
  <input id="gnn-site-search" type="search">
  <ul id="gnn-search-results" hidden></ul>
  <script type="application/json" id="gnn-search-data">{payload}</script>
</div>"""

    def _model_rows_html(self, data: dict) -> str:
        """Model rows deep-linking every parsed model page (``model/<slug>.html``)."""
        models = data.get("models") or []
        if not models:
            return ""
        _ensure_model_slugs(models)  # no-op when generate_website pre-assigned
        rows = ""
        for model in models:
            slug = str(model["slug"])
            rows += f"""<tr id="model-{_esc(slug)}">
  <td><a href="model/{_esc(slug)}.html" style="color:var(--accent-2)">{_esc(model.get("name") or slug)}</a></td>
  <td><code>{_esc(model.get("source_name") or "")}</code></td>
  <td>{len(model.get("variables") or [])}</td>
  <td>{len(model.get("edges") or [])}</td>
</tr>"""
        return f"""
<div class="section">
  <div class="section-title">Models</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Model</th><th>Source</th><th>Variables</th><th>Edges</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    def _page_model(self, model: dict) -> str:
        """Render one per-model page (written at ``model/<slug>.html``, depth 1)."""
        slug = str(model.get("slug") or _model_slug(str(model.get("name") or "")))
        name = str(model.get("name") or slug)
        body = f"""
<div class="page-header">
  <h1>{_esc(name)}</h1>
  <p class="subtitle">Source: <a href="../gnn_files.html#model-{_esc(slug)}" style="color:var(--accent-2)">{_esc(model.get("source_name") or slug)}</a> · {len(model.get("variables") or [])} variables · {len(model.get("edges") or [])} edges</p>
</div>
{self._model_variables_table(model)}
{self._model_edges_table(model)}
{self._model_images_html(model)}
<div class="section">
  <div class="section-title">Full GNN Source</div>
  {self._model_source_html(model)}
</div>"""
        return _page(
            _esc(name),
            "gnn_files",
            body,
            depth=1,
            breadcrumbs=[
                ("index.html", "Home"),
                ("gnn_files.html", "GNN Files"),
                (None, name),
            ],
        )

    @staticmethod
    def _model_source_html(model: dict) -> str:
        """Full (never truncated) source text in the listing's details/pre style."""
        source = model.get("source")
        try:
            text = Path(str(source)).read_text(encoding="utf-8", errors="replace")
            size = Path(str(source)).stat().st_size
        except Exception:
            text, size = "(could not read source file)", 0
        return f"""
<details>
  <summary>{_esc(model.get("source_name") or "Full GNN source")} <span class="pill badge-pending" style="margin-left:8px">{size} bytes</span></summary>
  <div class="details-body"><pre>{_esc(text)}</pre></div>
</details>"""

    @staticmethod
    def _model_variables_table(model: dict) -> str:
        """Variables table from the parsed model (explicit None row when empty)."""
        rows = ""
        for var in model.get("variables") or []:
            dims = ", ".join(str(d) for d in (var.get("dimensions") or [])) or "—"
            rows += f"""<tr>
  <td><code>{_esc(var.get("name") or "")}</code></td>
  <td>{_esc(var.get("type") or "")}</td>
  <td><code>{_esc(dims)}</code></td>
  <td>{_esc(var.get("data_type") or "")}</td>
  <td>{_esc(var.get("description") or "")}</td>
</tr>"""
        if not rows:
            rows = '<tr><td colspan="5">None</td></tr>'
        return f"""
<div class="section">
  <div class="section-title">Variables</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Name</th><th>Type</th><th>Dimensions</th><th>Data type</th><th>Description</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    @staticmethod
    def _model_edges_table(model: dict) -> str:
        """Edges table from the parsed model (explicit None row when empty)."""
        rows = ""
        for edge in model.get("edges") or []:
            rows += f"""<tr>
  <td>{_esc(", ".join(edge.get("sources") or []) or "—")}</td>
  <td>{_esc(", ".join(edge.get("targets") or []) or "—")}</td>
  <td>{_esc(edge.get("type") or "")}</td>
  <td>{_esc(edge.get("annotation") or "")}</td>
</tr>"""
        if not rows:
            rows = '<tr><td colspan="4">None</td></tr>'
        return f"""
<div class="section">
  <div class="section-title">Edges</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Source</th><th>Target</th><th>Type</th><th>Annotation</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    @staticmethod
    def _model_images_html(model: dict) -> str:
        """Collected image assets for this model, at the depth-correct assets/ path."""
        images = model.get("images") or []
        if not images:
            return ""
        cards = ""
        for image in images:
            stem = Path(str(image)).stem
            cards += f"""
<div class="viz-card">
  <img src="../assets/{_esc(image)}" alt="{_esc(stem)}" loading="lazy">
  <div class="viz-info">
    <div class="viz-title">{_esc(stem)}</div>
    <div class="viz-desc">Visualization artifact</div>
  </div>
</div>"""
        return f"""
<div class="section">
  <div class="section-title">Visualizations</div>
  <div class="viz-grid">{cards}</div>
</div>"""

    def _page_analysis(self, data: dict) -> str:
        """Render the analysis metrics page."""
        if not data["analysis"]:
            inner = '<div class="card"><p>No analysis results found. Run step 16 (Analysis) to generate results.</p></div>'
        else:
            inner = ""
            for item in data["analysis"]:
                name = item.get("file_name") or item.get("name", "Result")
                stats_html = ""
                for k, v in item.items():
                    if k in ("file_name", "name"):
                        continue
                    stats_html += (
                        f"<tr><td><code>{_esc(k)}</code></td><td>{_esc(v)}</td></tr>"
                    )
                inner += f"""
<div class="card">
  <h3>{_esc(name)}</h3>
  <div class="table-wrap" style="margin-top:8px">
    <table><tbody>{stats_html}</tbody></table>
  </div>
</div>"""
        body = f"""
<div class="page-header">
  <h1>📊 Analysis</h1>
  <p class="subtitle">Statistical analysis results</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Analysis", "analysis", body)

    def _page_visualization(self, data: dict) -> str:
        """Render the visualization gallery."""
        gui_card = ""
        if data.get("gui_navigation"):
            gui_card = """
<div class="card" style="margin-bottom:24px">
  <div class="viz-title" style="color:var(--accent-2)">Interactive GUI navigation</div>
  <p style="margin:8px 0 0">Interactive model editors and artifacts generated by Step 22.</p>
  <p style="margin:12px 0 0"><a href="../22_gui_output/navigation.html" target="_blank" style="color:var(--accent-2)">Open GUI navigation →</a></p>
</div>"""
        if not data["visualizations"]:
            inner = '<div class="card"><p>No visualizations found. Run steps 8–9 to generate visualizations.</p></div>'
        else:
            cards = ""
            for v in data["visualizations"]:
                title = _esc(v["title"])
                path = _esc(v["path"])
                kind = _esc(str(v.get("type", "image")).title())
                if v["type"] == "image":
                    cards += f"""
<div class="viz-card">
  <img src="assets/{path}" alt="{title}" loading="lazy">
  <div class="viz-info">
    <div class="viz-title">{title}</div>
    <div class="viz-desc">{kind} artifact</div>
  </div>
</div>"""
                else:
                    cards += f"""
<div class="viz-card">
  <div style="padding:16px;background:var(--bg-surface);text-align:center">
    <a href="assets/{path}" target="_blank" style="color:var(--accent-2);font-size:13px">🔗 Open interactive: {title}</a>
  </div>
  <div class="viz-info">
    <div class="viz-title">{title}</div>
    <div class="viz-desc">Interactive HTML visualization</div>
  </div>
</div>"""
            inner = f'<div class="viz-grid">{cards}</div>'
        body = f"""
<div class="page-header">
  <h1>🖼️ Visualizations</h1>
  <p class="subtitle">{len(data["visualizations"])} artifacts generated</p>
</div>
<div class="section">{gui_card}{inner}</div>"""
        return _page("Visualizations", "visualization", body)

    def _page_reports(self, data: dict) -> str:
        """Render the report artifact viewer."""
        if not data["reports"]:
            inner = '<div class="card"><p>No report artifacts found in pipeline output directories.</p></div>'
        else:
            inner = ""
            for rep in data["reports"]:
                try:
                    parsed = json.loads(rep["content"])
                    pretty = _truncate(json.dumps(parsed, indent=2), 1500)
                except Exception:
                    pretty = _truncate(rep["content"], 1500)
                name = _esc(rep["name"])
                origin = _esc(rep["dir"])
                size = _esc(rep["size"])
                inner += f"""
<details>
  <summary>{name} <span style="color:var(--text-3);font-size:11px;margin-left:8px">{origin} · {size} bytes</span></summary>
  <div class="details-body"><pre>{_esc(pretty)}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📋 Reports</h1>
  <p class="subtitle">{len(data["reports"])} report artifacts collected from pipeline output</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Reports", "reports", body)

    def _page_mcp(self, data: dict) -> str:
        """Render the MCP registry from the step-21 processing artifacts."""
        summary: dict[str, Any] = data.get("mcp_summary") or {}
        if not summary:
            # Truthful empty state: step 21 produced nothing we can report.
            inner = (
                '<div class="card"><p>No MCP data recorded.</p>'
                "<p>Step 21 (MCP Processing) did not run or wrote no MCP output "
                "for this pipeline run.</p></div>"
            )
            n_tools = 0
        else:
            n_tools = int(summary.get("tools_registered", 0) or 0)
            status = str(summary.get("processing_status", "unknown"))
            failed = status == "failed"
            badge_cls = "badge-error" if failed else "badge-ok"
            message = str(summary.get("message", ""))
            timestamp = str(summary.get("timestamp", ""))
            mcp_version = str(summary.get("mcp_version", ""))
            n_modules = int(summary.get("registered_modules_count", 0) or 0)
            n_resources = int(summary.get("resources_count", 0) or 0)

            rows = (
                f"<tr><td>Status</td><td>"
                f'<span class="pill {badge_cls}">{_esc(status)}</span></td></tr>'
                f"<tr><td>Timestamp</td><td>{_esc(timestamp)}</td></tr>"
                f"<tr><td>MCP version</td><td>{_esc(mcp_version)}</td></tr>"
                f"<tr><td>Registered modules</td><td>{n_modules}</td></tr>"
                f"<tr><td>Resources</td><td>{n_resources}</td></tr>"
            )
            if failed and summary.get("error"):
                rows += (
                    "<tr><td>Error</td>"
                    '<td style="color:var(--error)">'
                    f"{_esc(str(summary['error']))}</td></tr>"
                )

            chips = "".join(
                f'<span class="pill badge-pending" style="margin-left:6px">'
                f"{_esc(str(m))}</span>"
                for m in (summary.get("registered_modules") or [])
            )
            chips_html = f'<div style="margin:10px 0">{chips}</div>' if chips else ""

            inner = f"""
<div class="card">
  <h3>MCP Processing Summary</h3>
  <p>{_esc(message)}</p>
  <div class="table-wrap" style="margin-top:8px">
    <table><tbody>{rows}</tbody></table>
  </div>
  {chips_html}
</div>"""

            tools: list[dict[str, Any]] = data["mcp_tools"]
            if tools:
                by_mod: dict[str, list[dict[str, Any]]] = {}
                for t in sorted(tools, key=lambda x: (x.get("module", ""), x["name"])):
                    mod = t.get("module") or "core"
                    by_mod.setdefault(mod, []).append(t)

                for mod, mod_tools in sorted(by_mod.items()):
                    cards_html = ""
                    for t in mod_tools:
                        desc = t.get("desc") or ""
                        cat = t.get("category") or ""
                        cat_html = f" · {_esc(cat)}" if cat else ""
                        desc_html = (
                            f'<div class="tool-desc">{_esc(desc)}</div>' if desc else ""
                        )
                        cards_html += f"""
<div class="tool-card">
  <div class="tool-name">{_esc(t["name"])}</div>
  <div class="tool-mod">{_esc(mod)}{cat_html}</div>
  {desc_html}
</div>"""
                    inner += f"""
<div class="section">
  <div class="section-title">{_esc(mod)} <span class="pill badge-pending" style="margin-left:6px">{len(mod_tools)}</span></div>
  {cards_html}
</div>"""

        body = f"""
<div class="page-header">
  <h1>🔧 MCP Tools Registry</h1>
  <p class="subtitle">{n_tools} tools registered across all modules via the Model Context Protocol</p>
</div>
{inner}"""
        return _page("MCP Tools", "mcp", body)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────


def generate_website(
    logger: logging.Logger,
    input_dir: Path,
    output_dir: Path,
    *,
    pipeline_output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a premium website from GNN pipeline artifacts."""
    try:
        generator = WebsiteGenerator()
        p_root = pipeline_output_root if pipeline_output_root else output_dir.parent
        website_data: dict[str, Any] = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "pipeline_output_root": str(p_root),
        }
        if not input_dir.exists():
            return {
                "success": False,
                "pages_created": 0,
                "errors": [f"Input directory not found: {input_dir}"],
                "warnings": [],
            }
        result = generator.generate_website(website_data)
        if result["success"]:
            logger.info(
                f"Website generated: {result['pages_created']} pages → {output_dir}"
            )
        else:
            for e in result["errors"]:
                logger.error(f"Website error: {e}")
        return result
    except Exception as e:
        logger.error(f"Website generation failed: {e}")
        return {
            "success": False,
            "pages_created": 0,
            "errors": [str(e)],
            "warnings": [],
        }
