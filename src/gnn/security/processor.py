#!/usr/bin/env python3
"""
Security processor module for GNN pipeline.

Layered structure (composability doctrine):

- Pure policy layer: ``resolve_security_policy`` / ``ResolvedSecurityPolicy``
  resolve and validate a policy request without touching the filesystem.
- Pure scan layer: ``check_vulnerabilities`` / ``_check_python_ast`` /
  ``scan_source`` turn content into findings; no policy decisions.
- Verdict layer: ``scan_script_for_execution`` / ``findings_at_or_above``
  apply a severity threshold to findings (fail-closed on unknown severity).
- Orchestration layer: ``process_security`` glues the layers and writes the
  durable receipts (``security_results.json`` / ``security_summary.md``).
"""

import ast
import hashlib
import json
import logging
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
)

logger = logging.getLogger(__name__)

#: Timeout for the ``julia -e Meta.parseall`` syntax probe used by the
#: pre-execution gate. Julia startup can take a second or two; 30s is ample
#: headroom without stalling the gate on a hung interpreter.
_JULIA_PARSE_TIMEOUT_S = 30.0

#: Advisory Julia pattern sweep. These patterns flag *suspicious* constructs
#: (medium severity — informational at the default ``block_on="high"`` gate).
#: They are advisory even when Julia is available: the blocking signal for
#: Julia is ``Meta.parseall`` failing (malformed code → high).
_JULIA_SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brun\s*\(\s*`", "Julia backtick command execution"),
    (r"\bCmd\s*\(\s*\[", "Julia Cmd construction"),
]

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3}

# Security levels select analysis depth; an explicit ``block_on`` can make any
# scanning level enforcement-capable. Strict mode cannot disable vulnerability
# scanning because doing so would silently weaken the requested policy.
_SECURITY_LEVELS: dict[str, dict[str, Any]] = {
    "basic": {"scan_vulnerabilities": False, "default_block_on": None},
    "standard": {"scan_vulnerabilities": True, "default_block_on": None},
    "strict": {"scan_vulnerabilities": True, "default_block_on": "high"},
}

#: Sensitive credential-adjacent key patterns flagged by
#: ``perform_security_check``. Matches are reported with redacted context.
_SENSITIVE_PATTERNS: tuple[str, ...] = (
    r"password\s*[:=]",
    r"secret\s*[:=]",
    r"api_key\s*[:=]",
    r"token\s*[:=]",
    r"private_key\s*[:=]",
)

#: Regex vulnerability patterns applied to every scanned file (GNN markdown
#: included). ``(pattern, description)`` pairs; findings are ``medium`` unless
#: a more specific detector says otherwise.
_VULN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\beval\s*\(", "Code injection vulnerability"),
    (r"\bexec\s*\(", "Code execution vulnerability"),
    (r"\bimport\s+os\b", "OS command injection risk"),
    (
        r"\bsubprocess\s*\.\s*call\s*\(",
        "Subprocess call -- potential command injection",
    ),
    (
        r"\bsubprocess\s*\.\s*Popen\s*\(",
        "Subprocess Popen -- potential command injection",
    ),
    (
        r"\bsubprocess\s*\.\s*(?:run|check_call|check_output)\s*\(",
        "Subprocess execution risk",
    ),
    (r"\bfile\s*\(", "Previous file() call"),
)

#: Hardcoded-credential patterns; findings are ``high`` severity and the
#: matched value is never included in the receipt.
_CREDENTIAL_PATTERNS: tuple[str, ...] = (
    r'password\s*[:=]\s*["\'][^"\']{4,}["\']',
    r'secret\s*[:=]\s*["\'][^"\']{4,}["\']',
    r'api_key\s*[:=]\s*["\'][^"\']{8,}["\']',
)

#: Single human-readable reason emitted when ``process_security`` receives an
#: unusable policy. One message keeps the receipt stable for consumers.
_INVALID_POLICY_MESSAGE = (
    "Invalid security policy: security_level must be basic, standard, "
    "or strict; block_on must be low, medium, or high; and "
    "check_vulnerabilities must be a boolean and remain enabled for "
    "strict or explicitly enforced policies"
)


class SecurityScanError(Exception):
    """Raised when a per-file security check cannot be completed.

    Subclasses ``Exception`` so existing ``except Exception`` consumers
    (orchestrator, MCP wrappers) keep working unchanged.
    """


def _make_finding(
    file_path: Path,
    vulnerability_type: str,
    detection_method: str,
    severity: str,
    *,
    pattern: Optional[str] = None,
    line: Optional[int] = None,
    context: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    """Build a finding dict with the canonical key set and ordering.

    Keys absent from a given detector's semantics (``pattern``, ``line``) are
    omitted rather than filled with sentinel values, matching the
    per-detector shapes that tests pin.
    """
    finding: Dict[str, Any] = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "vulnerability_type": vulnerability_type,
        "detection_method": detection_method,
    }
    if pattern is not None:
        finding["pattern"] = pattern
    if line is not None:
        finding["line"] = line
    if context:
        finding["context"] = context
    finding["severity"] = severity
    finding.update(extra)
    return finding


def _severity_rank_of(finding: Dict[str, Any]) -> int:
    """Rank a finding's severity; unknown/missing ranks fail closed to high."""
    return _SEVERITY_RANK.get(
        str(finding.get("severity", "high")).lower(),
        _SEVERITY_RANK["high"],
    )


def findings_at_or_above(
    findings: Iterable[Dict[str, Any]], block_on: str
) -> List[Dict[str, Any]]:
    """Return the findings whose severity ranks at or above ``block_on``.

    Args:
        findings: Finding dicts as produced by ``check_vulnerabilities`` or
            ``scan_source``.
        block_on: Threshold name (``low``/``medium``/``high``); must already
            be validated by the caller.

    Returns:
        Findings at or above the threshold, in input order. The default gate
        posture is fail-closed: a finding with an unknown severity ranks as
        ``high``.
    """
    threshold = _SEVERITY_RANK[block_on.strip().lower()]
    return [finding for finding in findings if _severity_rank_of(finding) >= threshold]


def count_by_severity(findings: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count findings per (lowercased) severity label.

    Labels absent from the input are omitted from the result rather than
    reported as zero, so callers can distinguish "no high findings" from
    "no findings at all".
    """
    counts: Dict[str, int] = {}
    for finding in findings:
        label = str(finding.get("severity", "medium")).lower()
        counts[label] = counts.get(label, 0) + 1
    return counts


@dataclass(frozen=True)
class ResolvedSecurityPolicy:
    """Immutable result of validating a security policy request.

    ``is_valid=False`` always pairs with a human-readable ``error``; the
    remaining fields are still populated so callers can echo the request
    back in a receipt without re-deriving normalization rules.
    """

    security_level: str
    scan_vulnerabilities: bool
    block_on: Optional[str]
    enforced: bool
    requested_scan_vulnerabilities: Optional[Any]
    requested_block_on: Optional[str]
    is_valid: bool
    error: Optional[str]

    def to_receipt(self) -> Dict[str, Any]:
        """Return the static ``policy`` block for ``security_results.json``.

        ``decision`` and ``blocked_findings`` are runtime outcomes and are
        added by the caller.
        """
        return {
            "security_level": self.security_level,
            "enforced": self.enforced,
            "scan_vulnerabilities": self.scan_vulnerabilities,
            "requested_scan_vulnerabilities": self.requested_scan_vulnerabilities,
            "requested_block_on": self.requested_block_on,
            "block_on": self.block_on,
        }


def resolve_security_policy(
    security_level: Any = "standard",
    block_on: Any = None,
    check_vulnerabilities: Any = None,
) -> ResolvedSecurityPolicy:
    """Resolve and validate a security policy request. Pure; never raises.

    This is the single source of truth for the policy semantics used by
    ``process_security`` (and available to callers that want to validate a
    policy before scanning anything):

    - ``security_level`` selects analysis depth (``basic`` / ``standard`` /
      ``strict``).
    - An explicit ``block_on`` makes any level enforcement-capable.
    - ``check_vulnerabilities`` may force scanning on/off, except that strict
      (or explicitly enforced) policies cannot disable it — doing so would
      silently weaken the requested policy.

    Args:
        security_level: Requested level (any input; normalized to lowercase).
        block_on: Requested blocking threshold, or None.
        check_vulnerabilities: Requested scan override (bool), or None.

    Returns:
        The resolved policy. ``block_on`` is the effective threshold
        (validated request, else the level default, else None).
    """
    level = str(security_level).strip().lower()
    requested_threshold = (
        str(block_on).strip().lower() if block_on is not None else None
    )
    valid_thresholds = set(_SEVERITY_RANK) - {"info"}
    level_policy = _SECURITY_LEVELS.get(level)
    scan_vulnerabilities = (
        bool(
            (level_policy and level_policy["scan_vulnerabilities"])
            or block_on is not None
        )
        if check_vulnerabilities is None
        else check_vulnerabilities is True
    )
    receipt_requested_scan = (
        check_vulnerabilities
        if check_vulnerabilities is None or isinstance(check_vulnerabilities, bool)
        else str(check_vulnerabilities)
    )
    effective_block_on = (
        requested_threshold
        if requested_threshold in valid_thresholds
        else (
            str(level_policy["default_block_on"])
            if level_policy and level_policy["default_block_on"] is not None
            else None
        )
    )
    invalid_scan_policy = check_vulnerabilities is not None and not isinstance(
        check_vulnerabilities, bool
    )
    strict_scan_disabled = level == "strict" and not scan_vulnerabilities
    enforced_scan_disabled = block_on is not None and not scan_vulnerabilities
    is_valid = not (
        level_policy is None
        or (
            requested_threshold is not None
            and requested_threshold not in valid_thresholds
        )
        or invalid_scan_policy
        or strict_scan_disabled
        or enforced_scan_disabled
    )
    return ResolvedSecurityPolicy(
        security_level=level,
        scan_vulnerabilities=scan_vulnerabilities,
        block_on=effective_block_on,
        enforced=level == "strict" or block_on is not None,
        requested_scan_vulnerabilities=receipt_requested_scan,
        requested_block_on=requested_threshold,
        is_valid=is_valid,
        error=None if is_valid else _INVALID_POLICY_MESSAGE,
    )


def process_security(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process security validation for GNN files.

    Policy semantics live in ``resolve_security_policy``; this function
    resolves the policy, scans files, applies the blocking threshold, and
    writes the durable receipts.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: ``security_level`` (str), ``block_on`` (str), and
            ``check_vulnerabilities`` (bool) select the policy; see
            ``resolve_security_policy``

    Returns:
        True if processing successful, False otherwise
    """
    step_logger = logging.getLogger("security")
    policy = resolve_security_policy(
        kwargs.get("security_level", "standard"),
        kwargs.get("block_on"),
        kwargs.get("check_vulnerabilities"),
    )

    try:
        log_step_start(step_logger, "Processing security")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "security_checks": [],
            "vulnerabilities": [],
            "recommendations": [],
            "policy": {
                **policy.to_receipt(),
                "decision": "allow",
                "blocked_findings": 0,
            },
        }

        if not policy.is_valid:
            results["success"] = False
            results["policy"]["decision"] = "deny_invalid_policy"
            results["errors"].append(policy.error)

        # Find GNN files
        gnn_files = sorted(target_dir.glob("*.md"))
        if results["errors"]:
            gnn_files = []
        elif not gnn_files:
            step_logger.warning("No GNN files found for security processing")
            results["success"] = False
            results["policy"]["decision"] = "deny_no_input"
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform security checks
                    security_check = perform_security_check(gnn_file, verbose)
                    results["security_checks"].append(security_check)

                    # Check for vulnerabilities
                    if policy.scan_vulnerabilities:
                        vulnerabilities = check_vulnerabilities(gnn_file, verbose)
                        results["vulnerabilities"].extend(vulnerabilities)

                    # Generate security recommendations
                    recommendations = generate_security_recommendations(
                        gnn_file, verbose
                    )
                    results["recommendations"].extend(recommendations)

                except Exception as e:
                    error_info: dict[str, Any] = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results["errors"].append(error_info)
                    results["success"] = False
                    results["policy"]["decision"] = "deny_processing_error"
                    step_logger.error(f"Error processing {gnn_file}: {e}")

        threshold_name = results["policy"]["block_on"]
        if threshold_name is not None:
            blocked = findings_at_or_above(
                results["vulnerabilities"], str(threshold_name)
            )
            results["policy"]["blocked_findings"] = len(blocked)
            if blocked:
                results["success"] = False
                results["policy"]["decision"] = "deny"

        # Save detailed results
        results_file = results_dir / "security_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Generate security summary
        summary = generate_security_summary(results)
        summary_file = results_dir / "security_summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        if results["success"]:
            log_step_success(step_logger, "Security processing completed successfully")
        else:
            log_step_error(step_logger, "Security processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(step_logger, "Security processing failed", error=str(e))
        return False


def perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Perform a sensitive-data and integrity check on a single file.

    Scans for credential-adjacent patterns (``_SENSITIVE_PATTERNS``), hashes
    the exact bytes inspected, records the POSIX permission mode, and scores
    the result. The matched *values* are never included in the receipt —
    contexts are redacted.

    Raises:
        SecurityScanError: If the file cannot be read.
    """
    try:
        raw_content = file_path.read_bytes()
        content = raw_content.decode("utf-8", errors="replace")

        # Hash the exact bytes inspected. Text-mode hashing normalized CRLF on
        # some platforms, so it was not a reliable file-integrity receipt.
        file_hash = hashlib.sha256(raw_content).hexdigest()

        try:
            file_permissions = oct(stat.S_IMODE(file_path.stat().st_mode))
        except OSError:
            file_permissions = "unknown"

        found_patterns: list[Dict[str, Any]] = []
        for pattern in _SENSITIVE_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                found_patterns.append(
                    {
                        "pattern": pattern,
                        "line": content[: match.start()].count("\n") + 1,
                        "context": f"{match.group(0)} [REDACTED]",
                    }
                )

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size": len(raw_content),
            "sensitive_patterns": found_patterns,
            "file_permissions": file_permissions,
            "security_score": calculate_security_score(found_patterns),
            "check_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise SecurityScanError(
            f"Failed to perform security check on {file_path}: {e}"
        ) from e


def check_vulnerabilities(
    file_path: Path, verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Check for security vulnerabilities in a GNN file.

    Uses three complementary techniques:
    1. Regex pattern matching for GNN markdown files (``_VULN_PATTERNS``)
    2. Python AST analysis for generated .py files (eval, exec, os.system
       detection, ``shell=True``)
    3. World-writable permission checks for .py files (``stat.S_IWOTH``)

    Returns:
        Findings sorted by (line, vulnerability_type, detection_method,
        pattern) for deterministic receipts.
    """
    vulnerabilities: List[Dict[str, Any]] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # -- Technique 1: Regex patterns (for GNN markdown and all files) --
        for pattern, description in _VULN_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                vulnerabilities.append(
                    _make_finding(
                        file_path,
                        description,
                        "regex",
                        "medium",
                        pattern=pattern,
                        line=content[: match.start()].count("\n") + 1,
                        context=match.group(0)[:80],
                    )
                )

        # Hardcoded credentials: high severity, value never reported.
        for pattern in _CREDENTIAL_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                vulnerabilities.append(
                    _make_finding(
                        file_path,
                        "Hardcoded credentials",
                        "regex",
                        "high",
                        pattern=pattern,
                        line=content[: match.start()].count("\n") + 1,
                        context="[REDACTED]",
                    )
                )

        # -- Technique 2: AST analysis for Python files --
        if file_path.suffix == ".py":
            vulnerabilities.extend(_check_python_ast(file_path, content))

        # -- Technique 3: World-writable permission check for .py files --
        # Generated .py files shouldn't be world-writable in shared
        # environments. The mode is checked directly: gating on
        # ``os.access(W_OK)`` would reflect the *current process's* write
        # permission and can mask a world-writable mode (read-only mounts,
        # files owned by another user).
        if file_path.suffix == ".py":
            try:
                mode = file_path.stat().st_mode
                if mode & stat.S_IWOTH:
                    vulnerabilities.append(
                        _make_finding(
                            file_path,
                            "World-writable file permissions",
                            "permission_check",
                            "low",
                            pattern="stat.S_IWOTH",
                            line=0,
                            context=f"Mode: {oct(mode)}",
                        )
                    )
            except OSError:
                logger.debug(
                    "Permission check failed on %s (platform limitation)",
                    file_path.name,
                )

    except Exception as e:
        vulnerabilities.append(
            _make_finding(
                file_path,
                "File access error",
                "file_read",
                "low",
                error=str(e),
            )
        )

    return sorted(
        vulnerabilities,
        key=lambda finding: (
            int(finding.get("line", 0)),
            str(finding.get("vulnerability_type", "")),
            str(finding.get("detection_method", "")),
            str(finding.get("pattern", "")),
        ),
    )


#: Dangerous builtin calls: ``name -> (description, severity)``.
_DANGEROUS_CALLS: dict[str, Tuple[str, str]] = {
    "eval": ("Code injection via eval()", "high"),
    "exec": ("Code injection via exec()", "high"),
    "compile": ("Dynamic code compilation", "medium"),
    "__import__": ("Dynamic import -- verify input is trusted", "medium"),
}

#: Dangerous attribute calls ``obj.method()``: ``(module, method) ->
#: (description, severity)``. Module aliases (``import os as o``) resolve
#: through ``module_aliases``; ``from subprocess import run`` resolves
#: through call-alias tracking.
_DANGEROUS_METHODS: dict[Tuple[str, str], Tuple[str, str]] = {
    ("os", "system"): ("OS command injection via os.system()", "high"),
    ("os", "popen"): ("OS command injection via os.popen()", "high"),
    ("subprocess", "call"): ("Subprocess execution", "medium"),
    ("subprocess", "Popen"): ("Subprocess execution", "medium"),
    ("subprocess", "run"): ("Subprocess execution -- verify shell=False", "low"),
    ("subprocess", "check_call"): ("Subprocess execution", "medium"),
    ("subprocess", "check_output"): ("Subprocess execution", "medium"),
    ("subprocess", "getoutput"): ("Subprocess shell execution", "high"),
    ("subprocess", "getstatusoutput"): ("Subprocess shell execution", "high"),
    ("pickle", "loads"): ("Arbitrary code execution via pickle.loads()", "high"),
    ("pickle", "load"): ("Arbitrary code execution via pickle.load()", "high"),
    ("marshal", "loads"): ("Arbitrary code execution via marshal.loads()", "high"),
}


def _check_python_ast(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """
    Perform AST-level security analysis on Python source text.

    Detects dangerous function calls at the AST node level,
    which is more reliable than regex (handles multiline calls, string formatting).

    Dangerous patterns detected:
    - eval() / exec(): code injection vectors
    - os.system(): command injection
    - compile() with user-controlled input: code injection
    - __import__() dynamic import: potentially dangerous
    - import-alias and from-import call-alias tracking for the method tables

    Args:
        file_path: Path label for the scanned source (for error context;
            synthetic names such as ``<memory>.py`` are allowed)
        content: File content as string

    Returns:
        List of vulnerability dicts found via AST analysis, sorted by
        (line, vulnerability_type, detection_method)
    """
    vulns: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        # The pre-execution gate cannot prove an unparseable script is
        # safe, so this is a fail-closed finding rather than advisory.
        return [
            _make_finding(
                file_path,
                "Syntax error (cannot AST scan)",
                "ast_parse",
                "high",
                line=e.lineno or 0,
                context=str(e),
            )
        ]

    module_aliases: dict[str, str] = {}
    call_aliases: dict[str, Tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                module_aliases[name.asname or name.name.split(".", 1)[0]] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                call_aliases[name.asname or name.name] = (node.module, name.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Attribute) or not isinstance(
                value.value, ast.Name
            ):
                continue
            module_name = module_aliases.get(value.value.id, value.value.id)
            target_names: list[str] = []
            if isinstance(node, ast.Assign):
                target_names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node.target, ast.Name):
                target_names = [node.target.id]
            for target_name in target_names:
                call_aliases[target_name] = (module_name, value.attr)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        line = getattr(node, "lineno", 0)

        # Direct function calls: eval(), exec(), etc.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in _DANGEROUS_CALLS:
                desc, severity = _DANGEROUS_CALLS[func_name]
                vulns.append(
                    _make_finding(
                        file_path,
                        desc,
                        "ast_analysis",
                        severity,
                        pattern=f"{func_name}()",
                        line=line,
                        context=f"{func_name}() call at line {line}",
                    )
                )
            elif (
                func_name in call_aliases
                and call_aliases[func_name] in _DANGEROUS_METHODS
            ):
                obj_name, method_name = call_aliases[func_name]
                desc, severity = _DANGEROUS_METHODS[(obj_name, method_name)]
                if _uses_shell_true(node) and obj_name == "subprocess":
                    desc = "Subprocess execution with shell=True"
                    severity = "high"
                vulns.append(
                    _make_finding(
                        file_path,
                        desc,
                        "ast_analysis",
                        severity,
                        pattern=f"{obj_name}.{method_name}()",
                        line=line,
                        context=f"{func_name}() at line {line}",
                    )
                )

        # Attribute calls: os.system(), pickle.loads(), etc.
        elif isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                local_name = node.func.value.id
                obj_name = module_aliases.get(local_name, local_name)
                key = (obj_name, method_name)
                if key in _DANGEROUS_METHODS:
                    desc, severity = _DANGEROUS_METHODS[key]
                    if _uses_shell_true(node) and obj_name == "subprocess":
                        desc = "Subprocess execution with shell=True"
                        severity = "high"
                    vulns.append(
                        _make_finding(
                            file_path,
                            desc,
                            "ast_analysis",
                            severity,
                            pattern=f"{obj_name}.{method_name}()",
                            line=line,
                            context=f"{obj_name}.{method_name}() at line {line}",
                        )
                    )

    return sorted(
        vulns,
        key=lambda finding: (
            int(finding.get("line", 0)),
            str(finding.get("vulnerability_type", "")),
            str(finding.get("detection_method", "")),
        ),
    )


def _uses_shell_true(node: Any) -> bool:
    """Return whether a call's ``shell`` argument may enable a command shell.

    Only a literal ``shell=False`` is statically safe. Truthy literals and
    dynamic expressions are conservatively treated as enabled so the security
    gate cannot be bypassed with ``shell=flag`` or ``shell=1``.
    """
    return any(
        keyword.arg == "shell"
        and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        )
        for keyword in node.keywords
    )


def scan_source(
    source: str,
    *,
    file_name: str = "<memory>.py",
    block_on: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan Python source text (not a file) for dangerous constructs.

    Companion to ``check_vulnerabilities`` for callers that hold code in
    memory: the render step can validate generated Python *before* writing
    it to disk, and tests / MCP callers can scan snippets without fixtures.

    Args:
        source: Python source text to analyze.
        file_name: Display name used in findings (defaults to a synthetic
            ``<memory>.py`` label).
        block_on: Optional severity threshold; when given, the result carries
            ``blocked``/``decision``/``ok`` fields exactly like the
            ``scan_script_for_execution`` verdict (including the ``block_on``
            echo), so callers can reuse one handling path for both file and
            text scans. Asymmetry vs the file gate: an invalid threshold here
            is a bare ``deny_invalid_policy`` receipt without the
            ``policy_validation`` finding the file gate emits — there is no
            ``Unreadable script`` case to piggyback on.

    Returns:
        ``{"file_name", "findings"}`` plus, when ``block_on`` is set, the
        verdict fields ``ok`` / ``blocked`` / ``decision`` / ``block_on``.
    """
    label = Path(file_name)
    findings = _check_python_ast(label, source)
    result: Dict[str, Any] = {"file_name": label.name, "findings": findings}
    if block_on is not None:
        normalized = block_on.strip().lower() if isinstance(block_on, str) else ""
        if normalized not in _SEVERITY_RANK:
            result["ok"] = False
            result["blocked"] = findings
            result["decision"] = "deny_invalid_policy"
        else:
            blocked = findings_at_or_above(findings, normalized)
            result["ok"] = not blocked
            result["blocked"] = blocked
            result["decision"] = "deny" if blocked else "allow"
        result["block_on"] = normalized
    return result


def generate_security_recommendations(
    file_path: Path, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Generate security improvement recommendations for a file.

    Heuristic checks over the file text: security documentation section,
    input validation, error handling, and security logging. Failures to read
    the file are reported as a low-priority recommendation, not raised.
    """
    recommendations: List[Dict[str, Any]] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Check for basic security practices
        if not re.search(r"#\s*Security", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add security documentation section",
                    "priority": "medium",
                    "description": "Consider adding a security section to document security considerations",
                }
            )

        # Check for input validation
        if re.search(r"input\s*[:=]", content, re.IGNORECASE):
            if not re.search(r"validate|check|verify", content, re.IGNORECASE):
                recommendations.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "recommendation": "Add input validation",
                        "priority": "high",
                        "description": "Input validation should be implemented for all user inputs",
                    }
                )

        # Check for error handling
        if not re.search(r"try\s*:|except\s*:", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add error handling",
                    "priority": "medium",
                    "description": "Implement proper error handling for robust security",
                }
            )

        # Check for logging
        if not re.search(r"log|logging", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add security logging",
                    "priority": "medium",
                    "description": "Implement security event logging for monitoring",
                }
            )

    except Exception as e:
        recommendations.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "recommendation": "File access error",
                "priority": "low",
                "description": f"Could not analyze file: {e}",
            }
        )

    return recommendations


def calculate_security_score(vulnerabilities: List[Dict[str, Any]]) -> float:
    """Calculate a severity-weighted security score on a 0-100 scale.

    Higher is better: 100.0 for no findings, decreasing as findings accumulate
    (high weighs 10, medium 5, low 1, unknown 5). The score is deterministic
    in the input list only — timestamps and file paths do not participate.
    """
    if not vulnerabilities:
        return 100.0

    # Weight vulnerabilities by severity
    severity_weights: dict[str, float] = {"high": 10.0, "medium": 5.0, "low": 1.0}

    total_score = 0.0
    for vuln in vulnerabilities:
        severity = vuln.get("severity", "medium")
        total_score += severity_weights.get(severity, 5.0)

    # Convert to 0-100 scale (higher is better)
    max_possible_score = len(vulnerabilities) * 10.0
    if max_possible_score == 0:
        return 100.0

    score = max(0.0, 100.0 - (total_score / max_possible_score) * 100.0)
    return score


def generate_security_summary(results: Dict[str, Any]) -> str:
    """Generate a security summary report."""
    summary = f"""
# Security Analysis Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}

## Security Results
- **Security Checks**: {len(results.get("security_checks", []))}
- **Vulnerabilities Found**: {len(results.get("vulnerabilities", []))}
- **Recommendations**: {len(results.get("recommendations", []))}

## Vulnerability Summary
"""

    vulnerabilities = results.get("vulnerabilities", [])
    if vulnerabilities:
        severity_counts = count_by_severity(vulnerabilities)
        high_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]

        summary += f"- **High Severity**: {severity_counts.get('high', 0)}\n"
        summary += f"- **Medium Severity**: {severity_counts.get('medium', 0)}\n"
        summary += f"- **Low Severity**: {severity_counts.get('low', 0)}\n"

        if high_vulns:
            summary += "\n### High Severity Vulnerabilities\n"
            for vuln in high_vulns[:5]:  # Show first 5
                summary += f"- **{vuln.get('file_name', 'Unknown')}**: {vuln.get('vulnerability_type', 'Unknown')}\n"
    else:
        summary += "- No vulnerabilities found\n"

    summary += "\n## Recommendations\n"

    recommendations = results.get("recommendations", [])
    if recommendations:
        high_recs = [r for r in recommendations if r.get("priority") == "high"]
        medium_recs = [r for r in recommendations if r.get("priority") == "medium"]

        if high_recs:
            summary += "\n### High Priority Recommendations\n"
            for rec in high_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"

        if medium_recs:
            summary += "\n### Medium Priority Recommendations\n"
            for rec in medium_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"
    else:
        summary += "- No recommendations generated\n"

    return summary


def _julia_meta_parseall(content: str) -> Optional[tuple[bool, str]]:
    """Validate Julia source with ``Meta.parseall`` via a ``julia`` subprocess.

    Parsing does **not** execute the script — ``Meta.parseall`` only builds the
    AST, so the probe itself is safe to run on untrusted rendered code.

    Returns:
        ``(True, "")`` when the source parses cleanly.
        ``(False, message)`` when parsing failed (malformed script).
        ``None`` when Julia is not available on PATH (caller should fall back
        to the advisory regex sweep).
    """
    from gnn.utils.runtime_safety.julia_probe import julia_executable

    if julia_executable() is None:
        return None

    probe = (
        "function _gnn_parsecheck(s)\n"
        "    ex = Base.Meta.parseall(s)\n"
        "    function _has_incomplete(e)\n"
        "        if e isa Expr\n"
        "            if e.head === :incomplete\n"
        "                return true\n"
        "            end\n"
        "            for a in e.args\n"
        "                _has_incomplete(a) && return true\n"
        "            end\n"
        "        end\n"
        "        return false\n"
        "    end\n"
        "    if _has_incomplete(ex)\n"
        "        # Extract the first error message embedded in the AST\n"
        "        msg = sprint(print, ex)\n"
        '        println("GNN_PARSE_FAIL: ", msg[1:min(end,200)])\n'
        "        exit(1)\n"
        "    end\n"
        '    println("GNN_PARSE_OK")\n'
        "end\n"
        "s = read(stdin, String)\n"
        "try\n"
        "    _gnn_parsecheck(s)\n"
        "catch e\n"
        '    println("GNN_PARSE_FAIL: ", sprint(showerror, e))\n'
        "    exit(1)\n"
        "end\n"
    )
    try:
        proc = subprocess.run(
            ["julia", "--startup-file=no", "-e", probe],
            input=content,
            capture_output=True,
            text=True,
            timeout=_JULIA_PARSE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Julia installed but unusable (hang/misconfig) — degrade to advisory.
        logger.debug("Julia parse probe unavailable: %s", exc)
        return None

    stdout = (proc.stdout or "").strip()
    if proc.returncode == 0 and "GNN_PARSE_OK" in stdout:
        return True, ""
    if "GNN_PARSE_FAIL" not in stdout:
        # Julia is on PATH but the probe itself did not run (e.g. a juliaup
        # launcher with no installed toolchain, a broken depot). That is an
        # environment problem, not malformed rendered code — degrade to the
        # advisory regex sweep instead of blocking every Julia script.
        logger.debug(
            "Julia parse probe did not run (rc=%s): %s",
            proc.returncode,
            ((proc.stderr or "") + stdout)[-300:],
        )
        return None
    message = stdout.strip()
    # Keep the message bounded for findings context.
    return False, message[-400:]


def _julia_regex_sweep(script_path: Path, content: str) -> List[Dict[str, Any]]:
    """Advisory Julia textual sweep (backtick ``run`` / ``Cmd`` construction).

    These findings are classified ``medium`` — informational at the default
    ``block_on="high"`` gate, but they still block when the operator lowers the
    threshold. The blocking signal for Julia is ``Meta.parseall`` failing.
    """
    return [
        _make_finding(
            script_path,
            description,
            "regex",
            "medium",
            pattern=pattern,
            line=content[: match.start()].count("\n") + 1,
            context=match.group(0)[:80],
        )
        for pattern, description in _JULIA_SUSPICIOUS_PATTERNS
        for match in re.finditer(pattern, content)
    ]


def _deny_verdict(
    script_path: Path, decision: str, finding: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the uniform deny receipt for gate pre-scan failures.

    The finding is both the only finding and the only blocked item; the
    ``scanned`` flag is False because no AST/parse analysis ran.
    """
    return {
        "ok": False,
        "blocked": [finding],
        "findings": [finding],
        "scanned": False,
        "decision": decision,
    }


def scan_script_for_execution(
    script_path: Path,
    *,
    block_on: str = "high",
) -> Dict[str, Any]:
    """Pre-execution gate: scan a rendered script before Step 12 runs it.

    The pipeline renders GNN text specifications into executable Python/Julia
    scripts. Step 18 (``process_security``) runs *after* Step 12, so by itself
    it is forensic, not preventive. This function closes that gap by applying
    the AST scanner to a rendered ``.py`` script *before* execution, returning
    a structured verdict the executor can act on.

    Args:
        script_path: Path to the rendered script (``.py`` or ``.jl``).
        block_on: Severity threshold that blocks execution. Findings at or
            above this severity set ``ok=False``. Defaults to ``"high"``.

    Returns:
        Dict with keys:
            - ``ok`` (bool): True if execution may proceed.
            - ``blocked`` (list): findings that triggered the block.
            - ``findings`` (list): all findings (blocked + informational).
            - ``scanned`` (bool): whether AST/parse analysis was performed.
            - ``block_on`` (str): the normalized threshold in force.
            - ``decision`` (str): ``allow`` / ``deny`` / ``deny_*`` reason.

    Julia (``.jl``) scripts are validated with ``Meta.parseall`` via a Julia
    subprocess when Julia is on PATH; malformed code is a high-severity block.
    When Julia is unavailable the scan degrades to an advisory textual sweep
    (``scanned=False``, findings are medium/informational only).
    Unknown severities fail closed (ranked as high).
    """
    script_path = Path(script_path)
    normalized_block_on = block_on.strip().lower() if isinstance(block_on, str) else ""
    if normalized_block_on not in _SEVERITY_RANK:
        return {
            **_deny_verdict(
                script_path,
                "deny_invalid_policy",
                _make_finding(
                    script_path,
                    "Invalid security threshold",
                    "policy_validation",
                    "high",
                    context=f"Unsupported block_on value: {block_on}",
                ),
            ),
            "block_on": normalized_block_on,
        }

    try:
        content = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            **_deny_verdict(
                script_path,
                "deny_unreadable",
                _make_finding(
                    script_path,
                    "Unreadable script",
                    "file_read",
                    "high",
                    context=str(exc),
                ),
            ),
            "block_on": normalized_block_on,
        }

    findings: List[Dict[str, Any]] = []
    suffix = script_path.suffix.lower()
    if suffix == ".py":
        findings = _check_python_ast(script_path, content)
        scanned = True
    elif suffix == ".jl":
        # Julia scripts: validate with Meta.parseall (blocking) when Julia is
        # available; fall back to an advisory textual sweep when it is not.
        parse_result = _julia_meta_parseall(content)
        if parse_result is None:
            # Julia unavailable (or probe failed) — advisory sweep only, the
            # same posture as the previous textual-only scan.
            scanned = False
            findings = _julia_regex_sweep(script_path, content)
        else:
            parsed_ok, parse_message = parse_result
            scanned = True
            if not parsed_ok:
                # Malformed Julia is a hard block: the script cannot run as-is
                # and a syntax error is the strongest signal of tampering.
                findings.append(
                    _make_finding(
                        script_path,
                        "Malformed Julia code (Meta.parseall failed)",
                        "julia_meta_parseall",
                        "high",
                        line=1,
                        context=parse_message,
                    )
                )
            # Suspicious patterns remain medium (advisory at the default
            # block_on="high" gate) even when the code parses cleanly.
            findings.extend(_julia_regex_sweep(script_path, content))
    else:
        scanned = False
        findings = [
            _make_finding(
                script_path,
                "Unsupported executable script type",
                "file_type_policy",
                "high",
                line=0,
                context=script_path.suffix or "no suffix",
            )
        ]

    blocked = findings_at_or_above(findings, normalized_block_on)
    return {
        "ok": not blocked,
        "blocked": blocked,
        "findings": findings,
        "scanned": scanned,
        "block_on": normalized_block_on,
        "decision": "deny" if blocked else "allow",
    }
