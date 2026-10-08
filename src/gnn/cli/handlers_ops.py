#!/usr/bin/env python3
"""
Operational subcommand handlers: report, reproduce, preflight, and health.

Each handler receives the parsed argparse namespace and returns a process
exit code per the CLI contract (0 success, 1 error, 2 warnings). Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .commands import preflight_severities, publish_pipeline_report
from .helpers import EXIT_ERROR, EXIT_SUCCESS, EXIT_WARNING, _print_envelope

logger = logging.getLogger(__name__)


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate pipeline report from existing outputs."""
    is_json = getattr(args, "json", False)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error("Output directory not found: %s", output_dir)
        if is_json:
            _print_envelope(
                "error",
                error=f"Output directory not found: {output_dir}",
                command="report",
            )
        return EXIT_ERROR

    report, report_path = publish_pipeline_report(output_dir)
    if is_json:
        _print_envelope(
            "success",
            data={"report_path": str(report_path), "report_chars": len(report)},
            command="report",
        )
    else:
        print(f"📄 Report written to: {report_path}")
    return EXIT_SUCCESS


def _cmd_reproduce(args: argparse.Namespace) -> int:
    """Re-run from a previous run hash."""
    from gnn.pipeline.hasher import lookup_run, verify_indexed_run

    history_dir = args.history_dir

    run_entry = lookup_run(args.run_hash, history_dir)
    if not run_entry:
        print(f"❌ Run hash not found: {args.run_hash}")
        return EXIT_ERROR

    problems = verify_indexed_run(run_entry)
    if problems:
        for problem in problems:
            logger.error("Cannot reproduce run: %s", problem)
        return EXIT_ERROR

    print(f"🔄 Reproducing run: {args.run_hash}")
    config = run_entry.get("config", {})
    run_args_dict = dict(config.get("args", {}))

    try:
        from gnn.main import main as pipeline_main
        from gnn.main import resolve_steps_to_execute
        from gnn.utils.arguments.pipeline_arguments import PipelineArguments

        # Reconstruct PipelineArguments
        # Some paths might need to be converted back to Path objects
        if "target_dir" in run_args_dict:
            run_args_dict["target_dir"] = Path(run_args_dict["target_dir"])
        if "output_dir" in run_args_dict:
            run_args_dict["output_dir"] = Path(run_args_dict["output_dir"])

        reproduced_args = PipelineArguments(**run_args_dict)
        selected = resolve_steps_to_execute(reproduced_args, config["pipeline"], logger)
        if [step[0] for step in selected] != config["identity_config"][
            "selected_steps"
        ]:
            logger.error("Cannot reproduce run: resolved step selection changed")
            return EXIT_ERROR

        # Trigger execution bypassing normal CLI arg parsing
        print("🚀 Bypassing CLI parser, running with reconstructed config")

        # Pass the full config structure that main() expects back in override_config
        full_config_override: dict[str, Any] = config["identity_config"]["input_config"]

        return pipeline_main(
            override_args=reproduced_args, override_config=full_config_override
        )

    except ImportError as e:
        logger.error(f"Could not import pipeline for reproduction: {e}")
        return EXIT_ERROR
    except Exception as e:
        logger.error(f"Failed to reproduce run: {e}")
        return EXIT_ERROR


def _cmd_preflight(args: argparse.Namespace) -> int:
    """Run environment & config checks."""
    is_json = getattr(args, "json", False)
    from gnn.pipeline.preflight import run_preflight

    report = run_preflight(config_path=args.config)
    has_errors, has_warnings = preflight_severities(report)
    if is_json:
        data = {
            "optional_groups": getattr(report, "optional_groups", {}),
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
            "is_ok": report.is_ok,
            "issues": [
                {
                    "category": getattr(i, "category", "general"),
                    "severity": getattr(i, "severity", "warning"),
                    "message": getattr(i, "message", str(i)),
                }
                for i in report.issues
            ],
        }
        _print_envelope(
            "error" if has_errors else "warning" if has_warnings else "success",
            data=data,
            error="Preflight checks reported errors"
            if has_errors
            else "Preflight checks reported warnings"
            if has_warnings
            else None,
            command="preflight",
        )
    else:
        print(report.to_markdown())
        groups = getattr(report, "optional_groups", {})
        if groups:
            print("\nOptional dependency groups: " + ", ".join(groups))
    if has_errors:
        return EXIT_ERROR
    if has_warnings:
        return EXIT_WARNING
    return EXIT_SUCCESS


def _cmd_health(args: argparse.Namespace) -> int:
    """Show renderer & dependency status."""
    is_json = getattr(args, "json", False)
    from gnn.pipeline.preflight import check_environment
    from gnn.render.health import check_renderers

    renderers = check_renderers()
    env = check_environment()
    has_errors = any(issue.severity == "error" for issue in env.issues)
    has_warnings = any(issue.severity == "warning" for issue in env.issues)
    has_degraded_state = has_errors or has_warnings

    if is_json:
        data = {
            "renderers": {name: status.to_dict() for name, status in renderers.items()},
            "environment": {
                "checks_passed": env.checks_passed,
                "checks_failed": env.checks_failed,
                "is_ok": env.is_ok,
                "issues": [
                    {
                        "category": getattr(i, "category", "general"),
                        "severity": getattr(i, "severity", "warning"),
                        "message": getattr(i, "message", str(i)),
                    }
                    for i in env.issues
                ],
            },
        }
        _print_envelope(
            "error"
            if args.strict and has_errors
            else "warning"
            if has_degraded_state
            else "success",
            data=data,
            error="Environment health checks reported errors"
            if has_errors
            else "Environment health checks reported warnings"
            if has_warnings
            else None,
            command="health",
        )
    else:
        print("🔧 Renderer generator modules:")
        for name, status in sorted(renderers.items()):
            emoji = "🟢" if status.available else "🔴"
            print(f"  {emoji} {name}")

        available = sum(1 for r in renderers.values() if r.available)
        print(f"\n  {available}/{len(renderers)} generator modules importable")
        print(
            f"\n🏗️ Environment: {env.checks_passed} passed, {env.checks_failed} failed"
        )
        for issue in env.issues:
            sev = "⚠️" if issue.severity != "error" else "❌"
            print(f"  {sev} {issue.message}")

        if env.checks_failed and not args.strict:
            print("\nDefault health is informational; pass --strict to fail on errors.")

    if args.strict and has_errors:
        return EXIT_ERROR
    if has_degraded_state:
        return EXIT_WARNING
    return EXIT_SUCCESS
