"""Test-result accessors and report generation (HTML/Markdown/JSON).

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1)."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

from gnn.utils.testing.runner import _run_all_test_results


def get_test_results(output_dir: Path) -> Dict[str, Any]:
    """Read an unambiguous saved report; this does not verify a new run."""
    supplied = Path(output_dir)
    candidates = (
        [supplied]
        if supplied.is_file()
        else [
            candidate
            for candidate in (
                supplied / "test_results" / "test_report.json",
                supplied / "test_report.json",
                supplied / "test_results.json",
            )
            if candidate.is_file()
        ]
    )
    if len(candidates) > 1:
        return {
            "status": "ambiguous_results",
            "success": False,
            "error": "Multiple saved result views; supply an explicit report file",
            "candidates": [str(path) for path in candidates],
        }
    if candidates:
        results_file = candidates[0]
        with results_file.open(encoding="utf-8") as stream:
            data = cast("dict[str, Any]", json.load(stream))
        report = _canonical_category_report(data)
        report.update(source_report=str(results_file), readback_only=True)
        return report
    return {"status": "no_results"}


def generate_test_report(results: Dict[str, Any], output_dir: Path) -> bool:
    """Generate test report."""
    try:
        report_file = output_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(_canonical_category_report(results), f, indent=2)
        return True
    except Exception:
        return False


def get_test_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test summary."""
    return {"status": "basic_summary"}


def get_test_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test statistics."""
    return {"total": 0, "passed": 0, "failed": 0, "skipped": 0}


def get_test_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test performance metrics."""
    return {"execution_time": 0.0}


def get_test_logs(output_dir: Path) -> List[str]:
    """Get test logs."""
    return []


def get_test_artifacts(output_dir: Path) -> List[Path]:
    """Get test artifacts."""
    return []


def get_test_metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test metadata."""
    return {"timestamp": time.time()}


def get_test_timestamps(results: Dict[str, Any]) -> Dict[str, float]:
    """Get test timestamps."""
    return {"start": time.time(), "end": time.time()}


def get_test_duration(results: Dict[str, Any]) -> float:
    """Get test duration."""
    return 0.0


def get_test_status(results: Dict[str, Any]) -> str:
    """Get test status."""
    return "unknown"


def get_test_progress(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test progress."""
    return {"completed": 0, "total": 0}


def validate_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate report data and return validation results."""
    validation_results: dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "missing_fields": [],
        "extra_fields": [],
    }

    # Check required fields
    required_fields: list[Any] = ["timestamp", "step_name", "status"]
    for field in required_fields:
        if field not in data:
            validation_results["is_valid"] = False
            validation_results["missing_fields"].append(field)

    # Check data types
    if "timestamp" in data and not isinstance(data["timestamp"], str):
        validation_results["is_valid"] = False
        validation_results["errors"].append("timestamp must be a string")

    if "step_name" in data and not isinstance(data["step_name"], str):
        validation_results["is_valid"] = False
        validation_results["errors"].append("step_name must be a string")

    if "status" in data and data["status"] not in ["success", "failure", "warning"]:
        validation_results["is_valid"] = False
        validation_results["errors"].append(
            "status must be one of: success, failure, warning"
        )

    # Check for extra fields
    allowed_fields = required_fields + [
        "duration",
        "files_processed",
        "errors",
        "warnings",
    ]
    for field in data:
        if field not in allowed_fields:
            validation_results["warnings"].append(f"Unexpected field: {field}")

    return validation_results


def run_all_tests_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """Run all tests via MCP and return results."""
    try:
        target_dir = Path(target_directory)
        output_dir = Path(output_directory)

        report = _run_all_test_results(target_dir, output_dir, verbose)
        return {
            **report,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "verbose": verbose,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "target_directory": target_directory,
            "output_directory": output_directory,
            "verbose": verbose,
            "timestamp": datetime.now().isoformat(),
        }


def _category_passed(result: Any) -> bool:
    """Read explicit category status, retaining boolean saved-report support."""
    return result is True or (
        isinstance(result, dict) and result.get("success") is True
    )


def _canonical_category_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep public status booleans and move saved structured outcomes to details."""
    results = data.get("results")
    if not isinstance(results, dict):
        return data
    normalized = dict(data)
    details = data.get("category_results", {})
    category_results = {
        category: details[category]
        for category in results
        if isinstance(details, dict) and category in details
    }
    statuses: dict[str, bool] = {}
    for category, result in results.items():
        if not isinstance(result, bool) and not (
            isinstance(result, dict) and isinstance(result.get("success"), bool)
        ):
            raise ValueError("Saved category results must declare boolean success")
        statuses[category] = _category_passed(result)
        if isinstance(result, dict):
            category_results[category] = dict(result)
    normalized["results"] = statuses
    normalized["declared_success"] = data.get("declared_success", data.get("success"))
    normalized["success"] = (
        data.get("success") is True and bool(statuses) and all(statuses.values())
    )
    if category_results:
        normalized["category_results"] = category_results
    return normalized


def generate_html_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate an HTML test report file."""
    try:
        data = _canonical_category_report(data)
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GNN Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .section {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GNN Test Report</h1>
        <p>Generated: {data.get("timestamp", "Unknown")}</p>
    </div>
    
    <div class="section">
        <h2>Test Summary</h2>
        <p>Status: <span class="{"success" if data.get("success", False) else "failure"}">{"Success" if data.get("success", False) else "Failure"}</span></p>
        <p>Target Directory: {data.get("target_directory", "Unknown")}</p>
        <p>Output Directory: {data.get("output_directory", "Unknown")}</p>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <ul>
"""

        if "results" in data:
            for category, result in data["results"].items():
                status_class = "success" if _category_passed(result) else "failure"
                status_text = "Passed" if _category_passed(result) else "Failed"
                html_content += f'            <li><span class="{status_class}">{category}: {status_text}</span></li>\n'

        html_content += """
        </ul>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)

        return True

    except Exception as e:
        print(f"Failed to generate HTML report: {e}")
        return False


def generate_markdown_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate a Markdown test report file."""
    try:
        data = _canonical_category_report(data)
        markdown_content = f"""# GNN Test Report

**Generated**: {data.get("timestamp", "Unknown")}

## Test Summary

- **Status**: {"✅ Success" if data.get("success", False) else "❌ Failure"}
- **Target Directory**: {data.get("target_directory", "Unknown")}
- **Output Directory**: {data.get("output_directory", "Unknown")}

## Test Results

"""

        if "results" in data:
            for category, result in data["results"].items():
                status_icon = "✅" if _category_passed(result) else "❌"
                status_text = "Passed" if _category_passed(result) else "Failed"
                markdown_content += f"- **{category}**: {status_icon} {status_text}\n"

        markdown_content += """

## Details

This report was generated by the GNN test suite.
"""

        with open(output_path, "w") as f:
            f.write(markdown_content)

        return True

    except Exception as e:
        print(f"Failed to generate Markdown report: {e}")
        return False


def generate_json_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate a JSON test report file."""
    try:
        with open(output_path, "w") as f:
            json.dump(_canonical_category_report(data), f, indent=2)

        return True

    except Exception as e:
        print(f"Failed to generate JSON report: {e}")
        return False


def generate_comprehensive_report(
    pipeline_dir: Path, output_dir: Path, logger: logging.Logger
) -> bool:
    """Render a saved test report; the return value describes artifact generation."""
    try:
        # Create report directory
        report_dir = output_dir / "test_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Render only the explicitly supplied saved execution report. This
        # operation does not execute tests or certify a new run's completion.
        test_data = get_test_results(pipeline_dir)
        if test_data.get("status") in {"ambiguous_results", "no_results"}:
            raise ValueError(test_data.get("error", "Saved test report is unavailable"))
        if not isinstance(test_data.get("results"), dict):
            raise ValueError("Saved test report must contain category results")

        # Generate different report formats
        success = True
        success &= generate_html_report_file(test_data, report_dir / "test_report.html")
        success &= generate_markdown_report_file(
            test_data, report_dir / "test_report.md"
        )
        success &= generate_json_report_file(test_data, report_dir / "test_report.json")

        if success:
            logger.info("Comprehensive test report generated successfully")
        else:
            logger.warning("Some report formats failed to generate")

        return success

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        return False
