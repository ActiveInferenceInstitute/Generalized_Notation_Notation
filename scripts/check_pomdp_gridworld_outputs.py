#!/usr/bin/env python3
"""Validate the published POMDP GridWorld root output contract.

This checker is intentionally focused on the public run produced from
``input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md``. It verifies that
the committed ``output/`` tree is not stale broad-run data and that the
GridWorld model flowed through parse, render, execute, analysis, report, and
website stages with explicit framework/accounting artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALL_RENDER_TARGETS: tuple[str, ...] = (
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "stan",
    "bnlearn",
)

STRICT_EXECUTION_TARGETS: tuple[str, ...] = (
    "pymdp",
    "rxinfer",
    "activeinference_jl",
)

GRIDWORLD_STEM = "pomdp_gridworld_3x3"
GRIDWORLD_MARKERS = ("pomdp_gridworld", "gridworld", "POMDP GridWorld")


@dataclass
class ContractReport:
    """Validation result for the POMDP GridWorld output contract."""

    output_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no hard contract errors were found."""
        return not self.errors

    def require(self, condition: bool, message: str) -> None:
        """Record a hard error when condition is false."""
        if not condition:
            self.errors.append(message)

    def note(self, message: str) -> None:
        """Record a successful check note."""
        self.checked.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "output_dir": str(self.output_dir),
            "errors": self.errors,
            "warnings": self.warnings,
            "checked": self.checked,
        }


def _load_json(path: Path, report: ContractReport) -> Any:
    if not path.exists():
        report.errors.append(f"Missing JSON artifact: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON exception not important
        report.errors.append(f"Invalid JSON artifact {path}: {exc}")
        return None


def _text_contains_marker(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in GRIDWORLD_MARKERS)


def _first_mapping_value(mapping: dict[str, Any]) -> dict[str, Any]:
    if not mapping:
        return {}
    first = next(iter(mapping.values()))
    return first if isinstance(first, dict) else {}


def _check_pipeline_summary(output_dir: Path, report: ContractReport) -> None:
    summary_path = (
        output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    )
    summary = _load_json(summary_path, report)
    if not isinstance(summary, dict):
        return

    encoded = json.dumps(summary, sort_keys=True).lower()
    report.require(
        "pomdp_gridworld" in encoded or GRIDWORLD_STEM in encoded,
        "Pipeline summary does not reference the POMDP GridWorld target.",
    )

    steps = summary.get("steps", [])
    report.require(
        isinstance(steps, list) and len(steps) >= 25,
        "Pipeline summary must include all 25 numbered steps.",
    )
    if isinstance(steps, list):
        bad_steps: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_name = str(
                step.get("script_name")
                or step.get("step")
                or step.get("name")
                or "unknown"
            )
            exit_code = step.get("exit_code")
            status = str(step.get("status", "")).lower()
            if exit_code not in (0, 2, None) or status in {"failed", "error"}:
                bad_steps.append(
                    f"{step_name}: status={status!r} exit_code={exit_code!r}"
                )
        report.require(
            not bad_steps,
            "Pipeline summary contains non-success/non-warning steps: "
            + "; ".join(bad_steps),
        )
    report.note("pipeline summary")


def _check_parse_outputs(output_dir: Path, report: ContractReport) -> None:
    summary_path = output_dir / "3_gnn_output" / "gnn_processing_summary.json"
    summary = _load_json(summary_path, report)
    if isinstance(summary, dict):
        report.require(
            summary.get("total_files") == 1,
            "Step 3 should process exactly one GridWorld model.",
        )
        report.require(
            summary.get("successful_parses") == 1,
            "Step 3 should parse the GridWorld model successfully.",
        )
    parsed_candidates = sorted(
        (output_dir / "3_gnn_output").glob(f"**/{GRIDWORLD_STEM}*_parsed.json")
    )
    report.require(
        bool(parsed_candidates), "Step 3 parsed GridWorld JSON artifact is missing."
    )
    report.note("parse outputs")


def _check_render_outputs(output_dir: Path, report: ContractReport) -> None:
    render_dir = output_dir / "11_render_output"
    summary = _load_json(render_dir / "render_processing_summary.json", report)
    if not isinstance(summary, dict):
        return

    report.require(
        summary.get("total_files") == 1,
        "Step 11 should render exactly one GridWorld model.",
    )
    report.require(
        summary.get("successful_files") == 1,
        "Step 11 should mark the GridWorld file successful.",
    )
    report.require(
        not summary.get("failed_framework_renderings"),
        "Step 11 has failed framework renderings: "
        + json.dumps(summary.get("failed_framework_renderings"), sort_keys=True),
    )

    file_result = _first_mapping_value(
        summary.get("file_results", {})
        if isinstance(summary.get("file_results"), dict)
        else {}
    )
    framework_results = file_result.get("framework_results", {})
    report.require(
        isinstance(framework_results, dict),
        "Step 11 summary lacks framework_results mapping.",
    )
    if isinstance(framework_results, dict):
        missing = [name for name in ALL_RENDER_TARGETS if name not in framework_results]
        report.require(
            not missing, f"Step 11 did not attempt all render targets: {missing}"
        )
        failed = [
            f"{name}: {result.get('message', '')}"
            for name, result in framework_results.items()
            if name in ALL_RENDER_TARGETS
            and isinstance(result, dict)
            and not result.get("success")
        ]
        report.require(
            not failed, "Step 11 render target failures: " + "; ".join(failed)
        )

    for framework in ALL_RENDER_TARGETS:
        framework_dir = render_dir / GRIDWORLD_STEM / framework
        report.require(
            framework_dir.exists(),
            f"Missing Step 11 framework directory: {framework_dir}",
        )
        report.require(
            any(framework_dir.glob("*")),
            f"Step 11 framework directory is empty: {framework_dir}",
        )
    report.note("render outputs")


def _simulation_payloads(output_dir: Path, framework: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(
        (output_dir / "12_execute_output").glob(
            f"**/{framework}/simulation_data/*simulation_results.json"
        )
    ):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict):
            payloads.append(loaded)
    return payloads


def _check_execute_outputs(output_dir: Path, report: ContractReport) -> None:
    summary_path = (
        output_dir / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    summary = _load_json(summary_path, report)
    if isinstance(summary, dict):
        encoded = json.dumps(summary, sort_keys=True).lower()
        report.require(
            any(
                token in encoded
                for token in (
                    "succeeded",
                    "failed",
                    "skipped",
                    "success_count",
                    "failure_count",
                )
            ),
            "Step 12 summary should include explicit success/skipped/error accounting.",
        )

    for framework in STRICT_EXECUTION_TARGETS:
        payloads = _simulation_payloads(output_dir, framework)
        report.require(
            payloads, f"Missing Step 12 collected simulation results for {framework}."
        )
        if not payloads:
            continue
        payload = payloads[0]
        report.require(
            payload.get("success") is True,
            f"{framework} simulation payload is not successful.",
        )
        report.require(
            payload.get("num_timesteps") == 15,
            f"{framework} should report 15 timesteps.",
        )
        report.require(
            payload.get("model_parameters", {}).get("B_shape") == [9, 9, 5],
            f"{framework} should preserve GridWorld B shape [9, 9, 5].",
        )
        report.require(
            payload.get("validation", {}).get("all_valid") is True,
            f"{framework} simulation validation should be all_valid.",
        )
    report.note("execute outputs")


def _check_analysis_outputs(output_dir: Path, report: ContractReport) -> None:
    analysis_dir = output_dir / "16_analysis_output"
    manifest_path = (
        analysis_dir / "cross_framework" / "gridworld_analysis_manifest.json"
    )
    manifest = _load_json(manifest_path, report)
    if isinstance(manifest, dict):
        frameworks = sorted(manifest.get("frameworks", []))
        strict_targets = sorted(STRICT_EXECUTION_TARGETS)
        missing = [target for target in strict_targets if target not in frameworks]
        report.require(
            not missing,
            f"GridWorld analysis manifest should list strict execution targets "
            f"{strict_targets} as a subset, missing {missing} (got {frameworks}).",
        )
        report.require(
            manifest.get("matrix_provenance_equal") is True,
            "GridWorld analysis manifest should confirm matching matrix provenance.",
        )
        outputs = manifest.get("outputs", {})
        report.require(
            bool(outputs.get("png")),
            "GridWorld analysis manifest should list PNG outputs.",
        )
        report.require(
            len(outputs.get("gif", [])) >= 7,
            "GridWorld analysis manifest should list GridWorld GIF outputs.",
        )
        report.require(
            bool(outputs.get("statistics")),
            "GridWorld analysis manifest should list statistics outputs.",
        )

    pngs = sorted(analysis_dir.glob("**/*.png"))
    gifs = sorted(analysis_dir.glob("**/*.gif"))
    report.require(bool(pngs), "Step 16 PNG analysis outputs are missing.")
    report.require(bool(gifs), "Step 16 GIF animation outputs are missing.")
    report.require(
        all(path.stat().st_size > 0 for path in pngs[:10]),
        "One or more Step 16 PNG outputs are empty.",
    )
    report.require(
        all(path.stat().st_size > 0 for path in gifs),
        "One or more Step 16 GIF outputs are empty.",
    )
    report.note("analysis outputs")


def _check_report_and_website_outputs(output_dir: Path, report: ContractReport) -> None:
    report_dir = output_dir / "23_report_output"
    report.require(report_dir.exists(), "Step 23 report output directory is missing.")
    report_files = [
        path
        for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".html", ".json"}
    ]
    report.require(bool(report_files), "Step 23 report files are missing.")
    report.require(
        any(_text_contains_marker(path) for path in report_files),
        "Step 23 report files do not reference the GridWorld POMDP run.",
    )

    website_dir = output_dir / "20_website_output"
    required_pages = (
        "index.html",
        "pipeline.html",
        "reports.html",
        "analysis.html",
        "visualization.html",
    )
    for page in required_pages:
        report.require(
            (website_dir / page).exists(), f"Step 20 website page is missing: {page}"
        )
    website_pages = [website_dir / page for page in required_pages]
    report.require(
        any(_text_contains_marker(path) for path in website_pages),
        "Step 20 website pages do not reference the GridWorld POMDP run.",
    )
    report.note("report and website outputs")


def validate_output_tree(output_dir: Path | str = "output") -> ContractReport:
    """Validate a root output tree for the POMDP GridWorld publication contract."""
    resolved = Path(output_dir)
    report = ContractReport(output_dir=resolved)
    report.require(resolved.exists(), f"Output directory does not exist: {resolved}")
    if not resolved.exists():
        return report

    _check_pipeline_summary(resolved, report)
    _check_parse_outputs(resolved, report)
    _check_render_outputs(resolved, report)
    _check_execute_outputs(resolved, report)
    _check_analysis_outputs(resolved, report)
    _check_report_and_website_outputs(resolved, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the POMDP GridWorld public output contract."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output",
        help="Root output directory to validate (default: output)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    args = parser.parse_args(argv)

    report = validate_output_tree(args.output_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"POMDP GridWorld output contract: {status}")
        print(f"Output directory: {report.output_dir}")
        for item in report.checked:
            print(f"  checked: {item}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        for error in report.errors:
            print(f"  error: {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
