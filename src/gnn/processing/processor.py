#!/usr/bin/env python3
"""
GNN processor module for GNN pipeline.
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.processing.discovery import is_model_source_path


def _process_single_gnn_file(file_path: Path) -> Dict[str, Any]:
    """Process single gnn file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        parsed = parse_gnn_file(file_path, content=content)
        if not parsed["success"]:
            return {
                "parsed_result": None,
                "validation_result": None,
                "error": {
                    "file": str(file_path),
                    "error": parsed["error"],
                    "error_type": "ParseError",
                },
            }
        return {
            "parsed_result": parsed,
            "validation_result": _check_content_structure(
                file_path, content, parsed["sections"], parsed["variables"]
            ),
            "error": None,
        }
    except Exception as e:
        return {
            "parsed_result": None,
            "validation_result": None,
            "error": {
                "file": str(file_path),
                "error": str(e),
                "error_type": type(e).__name__,
            },
        }


def process_gnn_directory_lightweight(
    target_dir: Path,
    output_dir: Optional[Path] = None,
    recursive: bool = True,
    parallel: bool = False,
    *,
    workers: int | None = None,
    executor: str | None = None,
) -> Dict[str, Any]:
    """
    Lightweight GNN directory processing without heavy dependencies.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory to save results (optional)
        recursive: Whether to process subdirectories
        parallel: Whether to process discovered files concurrently
        workers: Explicit CPU worker limit (1..256); defaults to serial.
        executor: ``process`` (explicit workers) or ``thread`` (existing parallel option).

    Returns:
        Dictionary with processing results
    """
    try:
        from gnn.utils.file_workers import ordered_file_map, resolve_file_workers

        requested_workers, executor_mode = resolve_file_workers(
            workers, executor, parallel=parallel
        )
        target_dir = Path(target_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Input path does not exist: {target_dir}")
        if not target_dir.is_dir() and not target_dir.is_file():
            raise ValueError(f"Input must be a directory or model file: {target_dir}")
        # Discover GNN files
        gnn_files = discover_gnn_files(target_dir, recursive)

        if target_dir.is_file() and not gnn_files:
            raise ValueError(f"Not a supported GNN model source: {target_dir}")

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "target_directory": str(target_dir),
            "files_found": len(gnn_files),
            "files_processed": 0,
            "files_failed": 0,
            "files": [str(path) for path in gnn_files],
            "success": True,
            "errors": [],
            "parsed_files": [],
            "validation_results": [],
        }

        def record_file_result(file_result: Dict[str, Any]) -> None:
            """Provide record file result behavior."""
            error_info = file_result.get("error")
            if error_info:
                results["success"] = False
                results["files_failed"] += 1
                results["errors"].append(error_info)
                return

            parsed_result = file_result.get("parsed_result")
            if parsed_result:
                results["parsed_files"].append(parsed_result)
                results["files_processed"] += 1

            validation_result = file_result.get("validation_result")
            if validation_result:
                results["validation_results"].append(validation_result)
                if not validation_result["valid"]:
                    results["success"] = False
                    results["files_failed"] += 1
                    results["errors"].append(
                        {
                            "file": validation_result["file_path"],
                            "error": "; ".join(validation_result["errors"]),
                            "error_type": "ValidationError",
                        }
                    )

        worker_metadata: dict[str, Any] = {
            "requested_workers": requested_workers,
            "effective_workers": min(requested_workers, len(gnn_files)),
        }
        if workers is not None or executor is not None:
            results["worker_execution"] = worker_metadata
        for file_result in ordered_file_map(
            _process_single_gnn_file,
            gnn_files,
            workers=max(1, min(requested_workers, len(gnn_files))),
            executor=executor_mode,
            metadata=worker_metadata,
        ):
            record_file_result(file_result)

        if output_dir:
            try:
                _atomic_write_json(
                    Path(output_dir) / "gnn_processing_results.json", results
                )
            except OSError as exc:
                results["success"] = False
                results["errors"].append(
                    {
                        "file": str(output_dir),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        return results

    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "target_directory": str(target_dir),
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "files": [],
            "success": False,
            "errors": [{"error": str(e), "error_type": type(e).__name__}],
            "parsed_files": [],
            "validation_results": [],
        }


def _extract_sections_lightweight(content: str) -> List[str]:
    """
    Extract section headers from GNN content using lightweight regex parsing.

    Searches for markdown-style headers (lines starting with #) and extracts
    the header text. This provides a quick overview of document structure
    without full markdown parsing.

    Args:
        content: Raw text content of the GNN file.

    Returns:
        List of section header strings found in the content.
    """
    sections: list[Any] = []

    # Look for markdown headers
    header_pattern = r"^#+\s+(.+)$"
    matches = re.finditer(header_pattern, content, re.MULTILINE)

    for match in matches:
        section_title = match.group(1).strip()
        sections.append(section_title)

    return sections


def _extract_variables_lightweight(content: str) -> List[str]:
    """
    Extract variable names from GNN content using lightweight regex parsing.

    Searches for common variable definition patterns including:
    - Type annotations (name: type)
    - Assignments (name = value)
    - Array/matrix definitions (name[dimensions])

    Args:
        content: Raw text content of the GNN file.

    Returns:
        List of unique variable names found in the content.
    """
    variables: list[Any] = []
    seen: set[str] = set()

    # Names start at a word boundary and consume a complete word. Delimiters
    # cannot be word characters, so suffix retries and name backtracking add no
    # matches. Possessive quantifiers are available on every supported Python.
    var_patterns: list[Any] = [
        r"(?<!\w)(\w++)\s*:\s*(\w+)",  # name: type
        r"(?<!\w)(\w++)\s*=\s*([^;\n]+)",  # name = value
        r"(?<!\w)(\w++)\s*\[([^\]]+)\]",  # name[dimensions]
    ]

    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            var_name = match.group(1)
            if var_name not in seen:
                seen.add(var_name)
                variables.append(var_name)

    return variables


def discover_gnn_files(
    directory: Union[str, Path], recursive: bool = True
) -> List[Path]:
    """
    Discover GNN files in a directory (narrow globs: *.md, *.gnn, *.txt).

    Pipeline Step 3 uses ``multi_format_processor.process_gnn_multi_format`` with a
    broader extension list for full multi-format discovery; see ``src/gnn/SPEC.md``.

    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories

    Returns:
        List of discovered GNN file paths
    """
    directory = Path(directory)
    gnn_files: list[Any] = []

    if not directory.exists():
        return gnn_files

    if directory.is_file():
        return (
            [directory]
            if directory.suffix.lower() in {".md", ".gnn", ".txt"}
            and is_model_source_path(directory)
            else []
        )

    # Define GNN file patterns
    gnn_patterns: list[Any] = ["*.md", "*.gnn", "*.txt"]

    for pattern in gnn_patterns:
        if recursive:
            gnn_files.extend(directory.rglob(pattern))
        else:
            gnn_files.extend(directory.glob(pattern))

    return [path for path in gnn_files if is_model_source_path(path)]


def parse_gnn_file(
    file_path: Union[str, Path], content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a GNN file and extract basic information.

    Args:
        file_path: Path to the GNN file
        content: Pre-read file content; if None the file is opened and read.

    Returns:
        Dictionary with parsed information
    """
    file_path = Path(file_path)

    try:
        if content is None:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

        # Extract basic information
        sections = _extract_sections_lightweight(content)
        variables = _extract_variables_lightweight(content)

        # Count lines and characters
        line_count = len(content.splitlines())
        char_count = len(content)

        # Basic structure analysis
        structure_info: dict[str, Any] = {
            "has_variables": len(variables) > 0,
            "has_sections": len(sections) > 0,
            "variable_count": len(variables),
            "section_count": len(sections),
            "line_count": line_count,
            "char_count": char_count,
        }

        return {
            "success": True,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
            if file_path.exists()
            else len(content.encode("utf-8")),
            "sections": sections,
            "variables": variables,
            "structure_info": structure_info,
            "parse_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "success": False,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "error": str(e),
            "errors": [str(e)],
            "parse_timestamp": datetime.now().isoformat(),
        }


def check_gnn_file_structure(
    file_path: Union[str, Path], content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate the structure of a GNN file.

    Args:
        file_path: Path to the GNN file
        content: Pre-read file content; if None the file is opened and read.

    Returns:
        Dictionary with validation results
    """
    file_path = Path(file_path)

    try:
        if content is None:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

        return _check_content_structure(
            file_path,
            content,
            _extract_sections_lightweight(content),
            _extract_variables_lightweight(content),
        )

    except Exception as e:
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "validation_timestamp": datetime.now().isoformat(),
        }


def _check_content_structure(
    file_path: Path, content: str, sections: List[str], variables: List[str]
) -> Dict[str, Any]:
    """Validate one content read, reusing its existing parse metadata."""
    validation_result: dict[str, Any] = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "valid": True,
        "errors": [],
        "warnings": [],
        "validation_timestamp": datetime.now().isoformat(),
    }

    # Basic validation checks
    if len(content.strip()) == 0:
        validation_result["valid"] = False
        validation_result["errors"].append("File is empty")

    # Check for minimum content
    if len(content) < 10:
        validation_result["warnings"].append("File content is very short")

    # Check for basic GNN structure
    if len(sections) == 0 and len(variables) == 0:
        validation_result["warnings"].append("No clear GNN structure detected")

    # Check for common issues
    if content.count("{") != content.count("}"):
        validation_result["warnings"].append("Unmatched braces detected")

    if content.count("[") != content.count("]"):
        validation_result["warnings"].append("Unmatched brackets detected")

    return validation_result


def process_gnn_directory(
    directory: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    recursive: bool = True,
    parallel: bool = False,
    *,
    workers: int | None = None,
    executor: str | None = None,
) -> Dict[str, Any]:
    """
    Process all GNN files in a directory.

    Discovers GNN files in the specified directory, parses each file,
    and returns aggregated results. Optionally saves results to an output
    directory as JSON.

    Args:
        directory: Directory containing GNN files to process.
        output_dir: Optional directory to save processing results as JSON.
            If provided, creates 'gnn_processing_results.json' in this location.
        recursive: Whether to search subdirectories for GNN files.
        parallel: Whether to process discovered files concurrently.
        workers: Explicit worker limit, choosing processes by default.
        executor: Select ``process`` or ``thread`` for file computation.

    Returns:
        Dictionary containing:
            - status: "SUCCESS" / "FAILED" / "ERROR"
            - files: List of discovered file paths
            - processed_files: List of successfully processed file paths
            - error: Present on "ERROR" / "FAILED" status with diagnostic message
    """
    # Phase 1.3: validate input path exists before glob-walking. Previously the
    # function silently returned status=SUCCESS/files=[] for missing paths,
    # masking pipeline configuration errors. We accept both directories AND
    # single .md files because callers (tests, ad-hoc scripts) legitimately
    # use both — ``discover_gnn_files`` downstream handles each case.
    dir_arg: Optional[Union[str, Path]] = directory
    if dir_arg is None:
        return {
            "status": "FAILED",
            "files": [],
            "processed_files": [],
            "error": "directory argument is None",
        }
    _dir_path = Path(dir_arg)
    if not _dir_path.exists():
        return {
            "status": "FAILED",
            "files": [],
            "processed_files": [],
            "error": f"path does not exist: {_dir_path}",
        }
    # Use lightweight processing and wrap into status dict expected by tests
    lightweight_result = process_gnn_directory_lightweight(
        _dir_path,
        recursive=recursive,
        parallel=parallel,
        workers=workers,
        executor=executor,
    )
    # Extract actual file paths from parsed file results
    parsed_files = lightweight_result.get("parsed_files", [])
    file_paths = lightweight_result.get("files", [])
    failed_files = {error.get("file") for error in lightweight_result.get("errors", [])}
    processed_paths = [
        pf["file_path"]
        for pf in parsed_files
        if pf.get("success") and pf["file_path"] not in failed_files
    ]
    result: Dict[str, Any] = {
        "status": "SUCCESS" if lightweight_result.get("success", False) else "FAILED",
        "files": file_paths,
        "processed_files": processed_paths,
    }
    if "worker_execution" in lightweight_result:
        result["worker_execution"] = lightweight_result["worker_execution"]
    if lightweight_result.get("errors"):
        result["errors"] = lightweight_result["errors"]
        result["error"] = "; ".join(error["error"] for error in result["errors"])
    if output_dir is not None:
        try:
            _atomic_write_json(Path(output_dir) / "gnn_processing_results.json", result)
        except OSError as exc:
            result.update(status="ERROR", error=str(exc), error_type=type(exc).__name__)

    return result


def generate_gnn_report(
    processing_results: Dict[str, Any], output_path: (Union[str, Path]) | None = None
) -> str:
    """
    Generate a report from GNN processing results.

    Args:
        processing_results: Results from GNN processing
        output_path: Optional path to save the report

    Returns:
        Report content as string
    """
    report = f"""
# GNN Processing Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Summary
- **Target Directory**: {processing_results.get("target_directory", "Unknown")}
- **Files Found**: {processing_results.get("files_found", 0)}
- **Files Processed**: {processing_results.get("files_processed", 0)}
- **Success**: {processing_results.get("success", False)}
- **Errors**: {len(processing_results.get("errors", []))}

## File Analysis
"""

    parsed_files = processing_results.get("parsed_files", [])
    if parsed_files:
        report += f"\n### Parsed Files ({len(parsed_files)})\n"
        for file_info in parsed_files:
            report += f"- **{file_info.get('file_name', 'Unknown')}**\n"
            report += f"  - Variables: {file_info.get('structure_info', {}).get('variable_count', 0)}\n"
            report += f"  - Sections: {file_info.get('structure_info', {}).get('section_count', 0)}\n"
            report += f"  - Lines: {file_info.get('structure_info', {}).get('line_count', 0)}\n"

    validation_results = processing_results.get("validation_results", [])
    if validation_results:
        valid_count = sum(
            1 for result in validation_results if result.get("valid", False)
        )
        report += "\n### Validation Results\n"
        report += f"- Valid Files: {valid_count}/{len(validation_results)}\n"

        invalid_files = [r for r in validation_results if not r.get("valid", False)]
        if invalid_files:
            report += f"- Invalid Files: {len(invalid_files)}\n"
            for result in invalid_files[:5]:  # Show first 5
                report += f"  - {result.get('file_name', 'Unknown')}: {', '.join(result.get('errors', []))}\n"

    errors = processing_results.get("errors", [])
    if errors:
        report += "\n### Errors\n"
        for error in errors[:10]:  # Show first 10
            if isinstance(error, dict):
                report += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
            else:
                report += f"- {error}\n"

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

    return report


def get_module_info() -> Dict[str, Any]:
    """Return metadata and capability information about the GNN module."""
    return {
        "name": "GNN Module",
        "version": "1.0.0",
        "description": "GNN file discovery, parsing, and validation",
        "features": [
            "GNN file discovery",
            "Lightweight parsing",
            "Structure validation",
            "Report generation",
        ],
        "available_validators": ["structure", "syntax"],
        "available_parsers": ["markdown", "json"],
        "schema_formats": ["markdown-schema", "json-schema"],
        "supported_formats": ["Markdown", "GNN", "Text"],
        "capabilities": {
            "file_discovery": True,
            "content_parsing": True,
            "structure_validation": True,
            "report_generation": True,
        },
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    """Publish one complete JSON result; remove only our unfinished temporary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
