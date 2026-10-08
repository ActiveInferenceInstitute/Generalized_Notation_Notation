#!/usr/bin/env python3
"""
Config Validation & Environment Checks — Pre-flight diagnostics.

Provides:
  - validate_config(): validates input/config.yaml structure and values
  - check_environment(): verifies required tools and dependencies
  - PreflightReport: structured result with issues and recommendations
"""

import logging
import shutil
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _import_dependency(module_name: str) -> tuple[Any | None, str | None]:
    """Import a dependency and distinguish missing modules from broken imports."""
    try:
        return import_module(module_name), None
    except ModuleNotFoundError as e:
        if e.name == module_name:
            return None, "missing"
        return None, f"import failed: {e}"
    except ImportError as e:
        return None, f"import failed: {e}"


@dataclass
class PreflightIssue:
    """A single preflight check result."""

    category: str  # config, dependency, environment, permission
    severity: str  # error, warning, info
    message: str
    fix: Optional[str] = None


@dataclass
class PreflightReport:
    """Complete preflight check result."""

    issues: List[PreflightIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    optional_groups: dict[str, str] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        """Return whether ok."""
        return self.checks_failed == 0

    def add_pass(self, msg: str) -> None:
        """Provide add pass behavior."""
        self.checks_passed += 1
        logger.debug(f"✅ {msg}")

    def add_issue(
        self, category: str, severity: str, msg: str, fix: Optional[str] = None
    ) -> None:
        """Provide add issue behavior."""
        self.issues.append(
            PreflightIssue(category=category, severity=severity, message=msg, fix=fix)
        )
        if severity == "error":
            self.checks_failed += 1
        logger.warning(f"⚠️ [{category}] {msg}")

    def to_markdown(self) -> str:
        """Provide to markdown behavior."""
        lines: list[str] = ["# Preflight Check Report", ""]
        emoji = "🟢" if self.is_ok else "🔴"
        lines.append(
            f"{emoji} **{self.checks_passed} passed**, **{self.checks_failed} failed**"
        )
        lines.append("")

        if self.issues:
            lines.append("## Issues")
            for issue in self.issues:
                sev = (
                    "❌"
                    if issue.severity == "error"
                    else "⚠️"
                    if issue.severity == "warning"
                    else "ℹ️"
                )
                lines.append(f"- {sev} **[{issue.category}]** {issue.message}")
                if issue.fix:
                    lines.append(f"  - Fix: `{issue.fix}`")

        return "\n".join(lines)


def validate_config(config_path: Optional[Path] = None) -> PreflightReport:
    """
    Validate pipeline config file.

    Args:
        config_path: Path to config.yaml. Defaults to input/config.yaml.

    Returns:
        PreflightReport with any issues found.
    """
    report = PreflightReport()
    config_path = config_path or Path("input/config.yaml")

    # Check file exists
    if not config_path.exists():
        report.add_issue(
            "config",
            "warning",
            f"Config file not found: {config_path}",
            fix="cp input/config.yaml.example input/config.yaml",
        )
        return report

    report.add_pass(f"Config file exists: {config_path}")

    # Parse YAML
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Manual parse
        config = {}
        report.add_issue(
            "dependency", "info", "PyYAML not installed — limited config validation"
        )
        return report
    except Exception as e:
        report.add_issue("config", "error", f"Config parse error: {e}")
        return report

    if not isinstance(config, dict):
        report.add_issue("config", "error", "Config file must be a YAML mapping")
        return report

    report.add_pass("Config is valid YAML")

    # Validate known sections
    if "llm" in config and not isinstance(config["llm"], dict):
        report.add_issue("config", "error", "llm must be a YAML mapping")
    elif "llm" in config:
        llm = config["llm"]
        if "model" in llm:
            report.add_pass(f"LLM model configured: {llm['model']}")
        if "timeout_seconds" in llm:
            timeout = llm["timeout_seconds"]
            import math

            if timeout is None:
                report.add_pass("LLM timeout: automatic (600s per selected model)")
            elif (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                report.add_issue(
                    "config", "error", f"Invalid llm.timeout_seconds: {timeout}"
                )
            elif timeout > 3600:
                report.add_issue(
                    "config", "warning", f"Very large LLM timeout: {timeout}s"
                )
            else:
                report.add_pass(f"LLM timeout: {timeout}s")

    # Validate pipeline.skip_steps with the canonical parser (values must be
    # exact integers in 0..24). Catching a bad list here — before a container
    # plan or a pipeline run consumes it — is the whole point of preflight.
    pipeline_section = config.get("pipeline")
    if isinstance(pipeline_section, dict) and "skip_steps" in pipeline_section:
        from gnn.pipeline.pipeline_container_plan import read_skip_steps

        try:
            skipped = read_skip_steps(config_path)
        except ValueError as e:
            report.add_issue(
                "config",
                "error",
                f"Invalid pipeline.skip_steps: {e}",
                fix="Use exact integers 0-24, e.g. skip_steps: [15, 16]",
            )
        else:
            report.add_pass(f"pipeline.skip_steps: {skipped if skipped else 'none'}")

    return report


def check_environment() -> PreflightReport:
    """
    Check that required tools and dependencies are available.

    Returns:
        PreflightReport with environment status.
    """
    report = PreflightReport()

    # Python version
    import sys

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info >= (3, 11):
        report.add_pass(f"Python {py_version}")
    else:
        report.add_issue(
            "environment",
            "error",
            f"Python {py_version} — requires ≥3.11",
            fix="pyenv install 3.11",
        )

    # Required packages
    for pkg in ["numpy", "pytest", "yaml"]:
        module, error = _import_dependency(pkg)
        if module is not None:
            report.add_pass(f"Package: {pkg}")
        elif error == "missing":
            report.add_issue(
                "dependency",
                "warning",
                f"Package not found: {pkg}",
                fix=f"pip install {pkg}",
            )
        else:
            report.add_issue(
                "dependency",
                "warning",
                f"Package import failed: {pkg} ({error})",
                fix=f"pip install --upgrade {pkg}",
            )

    # Step 12 execution stack (core ``pyproject.toml`` dependencies)
    for mod, label in [
        ("jax", "jax"),
        ("jaxlib", "jaxlib"),
        ("torch", "torch"),
        ("numpyro", "numpyro"),
        ("discopy", "discopy"),
    ]:
        module, error = _import_dependency(mod)
        if module is not None:
            ver = getattr(module, "__version__", "?")
            report.add_pass(f"Package: {label} {ver}")
        elif error == "missing":
            report.add_issue(
                "dependency",
                "error",
                f"Package not found: {label} (step 12 backend)",
                fix="uv sync",
            )
        else:
            report.add_issue(
                "dependency",
                "error",
                f"Package import failed: {label} (step 12 backend): {error}",
                fix="uv sync --reinstall-package "
                f"{label} or adjust compatible dependency versions",
            )

    # Optional packages
    for pkg, purpose in [("pydantic", "schemas"), ("fastapi", "API"), ("pygls", "LSP")]:
        module, error = _import_dependency(pkg)
        if module is not None:
            report.add_pass(f"Optional: {pkg} ({purpose})")
        elif error == "missing":
            report.add_issue(
                "dependency",
                "info",
                f"Optional package: {pkg} ({purpose})",
                fix=f"pip install {pkg}",
            )
        else:
            report.add_issue(
                "dependency",
                "info",
                f"Optional package import failed: {pkg} ({purpose}): {error}",
                fix=f"pip install --upgrade {pkg}",
            )

    # Tools
    for tool in ["ollama", "ruff", "pkl", "d2"]:
        if shutil.which(tool):
            report.add_pass(f"Tool: {tool}")
        else:
            sev = "info" if tool in ("ollama", "pkl", "d2") else "warning"
            report.add_issue(
                "environment",
                sev,
                f"Tool not found: {tool}",
                fix=f"See documentation for {tool}",
            )

    # Directories
    for d in [Path("input/gnn_files"), Path("output")]:
        if d.exists():
            report.add_pass(f"Directory: {d}")
        else:
            report.add_issue(
                "environment",
                "warning",
                f"Directory not found: {d}",
                fix=f"mkdir -p {d}",
            )

    return report


def run_preflight(config_path: Optional[Path] = None) -> PreflightReport:
    """Run all preflight checks and combine results."""
    config_report = validate_config(config_path)
    env_report = check_environment()

    combined = PreflightReport()
    from gnn.setup.constants import OPTIONAL_GROUPS

    combined.optional_groups = dict(OPTIONAL_GROUPS)
    combined.issues = config_report.issues + env_report.issues
    combined.checks_passed = config_report.checks_passed + env_report.checks_passed
    combined.checks_failed = config_report.checks_failed + env_report.checks_failed

    return combined
