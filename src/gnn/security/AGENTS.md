# Security Module - Agent Scaffolding

## Module Overview

**Purpose**: Security scanning and validation for GNN pipeline files — injection-pattern and Python AST vulnerability detection, severity scoring, recommendations, and a pre-execution gate for rendered scripts.

**Pipeline Step**: Step 18: Security validation (18_security.py)

**Category**: Security / Vulnerability Scanning

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

1. Scan GNN files for injection patterns and suspicious constructs (pattern matching + Python AST analysis)
2. Score files by vulnerability severity (0-100 security score)
3. Generate actionable security recommendations per file
4. Gate rendered scripts before Step 12 executes them (`scan_script_for_execution`)
5. Write `security_results.json` and `security_summary.md`
6. Resolve and validate security policies up front (`resolve_security_policy`)
7. Scan Python source text in memory before writing files (`scan_source`)

---

## API Reference

The processor is layered (pure policy → pure scan → verdict → orchestration);
each layer is importable on its own.

### Policy Layer (pure, no filesystem)

#### `resolve_security_policy(security_level="standard", block_on=None, check_vulnerabilities=None) -> ResolvedSecurityPolicy`
**Description**: Single source of truth for policy semantics. Normalizes and
validates a policy request and returns a frozen `ResolvedSecurityPolicy`
dataclass (`security_level`, `scan_vulnerabilities`, `block_on`, `enforced`,
`requested_scan_vulnerabilities`, `requested_block_on`, `is_valid`, `error`).
Never raises; invalid requests fail closed with a human-readable `error`.
`process_security` delegates here; callers may pre-validate a policy before
scanning anything.

Rules: `security_level` selects depth (`basic`/`standard`/`strict`); an
explicit `block_on` makes any level enforcement-capable; strict (or
explicitly enforced) policies cannot disable vulnerability scanning.
`ResolvedSecurityPolicy.to_receipt()` emits the static `policy` block of
`security_results.json` (`decision`/`blocked_findings` are runtime outcomes
added by the caller).

#### `findings_at_or_above(findings, block_on) -> List[Dict]`
Filter findings to those ranking at/above a validated threshold. Unknown or
missing severities fail closed (ranked as `high`).

#### `count_by_severity(findings) -> Dict[str, int]`
Severity histogram; labels absent from the input are omitted (not zero).

### Scan Layer

#### `perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
Sensitive-data and integrity check on a single file: credential-pattern
scanning (`password`, `secret`, `api_key`, `token`, `private_key`), SHA-256 of
the exact bytes inspected, real POSIX permission mode (octal; `"unknown"` when
`stat` fails), and a security score. Raises `SecurityScanError` when the file
cannot be read.

**Returns**: `Dict[str, Any]` with keys `file_path`, `file_name`, `file_hash`,
`file_size`, `sensitive_patterns` (redacted contexts), `file_permissions`,
`security_score`, `check_timestamp`.

#### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
Scan a file with three techniques: regex pattern tables (all files), Python
AST analysis (`.py`: dangerous calls, aliases, `shell=True`), and world-writable
mode checks via `stat.S_IWOTH` (`.py` only). Findings sort by
(line, vulnerability_type, detection_method, pattern) for deterministic
receipts. Unreadable files yield a `low` "File access error" finding, not an
exception.

#### `scan_source(source: str, *, file_name: str = "<memory>.py", block_on: str | None = None) -> Dict[str, Any]`
**New in 1.7.0**: scan Python source *text* (no file). Lets the render step
validate generated code before writing it and lets tests/MCP callers scan
snippets. Returns `{"file_name", "findings"}`; with `block_on` set, adds the
same verdict fields as `scan_script_for_execution`
(`ok`/`blocked`/`decision`/`block_on`).

### Verdict Layer

#### `scan_script_for_execution(script_path: Path, *, block_on: str = "high") -> Dict[str, Any]`
Pre-execution security gate for rendered scripts (RED_TEAM V-01/V-06).
Applies the Python AST scanner to a rendered `.py` script *before* Step 12
runs it, returning `{ok, blocked, findings, scanned, block_on, decision}`.
Findings at/above `block_on` severity set `ok=False`; unknown severities fail
closed. `.jl` scripts get an advisory textual sweep plus a `Meta.parseall`
syntax probe (parse failure = high severity). Wired into
`execute.processor.execute_single_script` (escape hatch:
`GNN_ALLOW_UNSAFE_EXEC=1`). Exported from the package root since 1.7.0.

### Errors

#### `SecurityScanError(Exception)`
Typed failure of a per-file security check. Subclasses `Exception`, so
existing `except Exception` consumers (orchestrator, MCP wrappers) are
unaffected.

### Orchestration

#### `process_security(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main security processing function called by orchestrator (18_security.py).
Resolves the policy via `resolve_security_policy`, scans all GNN files in
`target_dir`, applies the blocking threshold, and writes the receipts.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to scan
- `output_dir` (Path): Output directory for security reports
- `verbose` (bool): Enable verbose logging (default: False)
- `security_level` (str, via kwargs): `"basic"`, `"standard"`, `"strict"` (default: `"standard"`)
- `block_on` (str, via kwargs): Explicit blocking threshold (`low`/`medium`/`high`)
- `check_vulnerabilities` (bool, via kwargs): Force scanning on/off

**Returns**: `bool` - True if security processing succeeded, False otherwise

**Example**:
```python
from gnn.security import process_security
from pathlib import Path

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    verbose=True,
    security_level="strict",
)
```

#### `generate_security_recommendations(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
Heuristic recommendations per file (security section, input validation, error
handling, logging). Read failures become a low-priority recommendation.

#### `calculate_security_score(vulnerabilities: List[Dict[str, Any]]) -> float`
Severity-weighted score, 0–100 (100 = no findings; high weighs 10, medium 5,
low 1).

#### `generate_security_summary(results: Dict[str, Any]) -> str`
Markdown summary of a results receipt.

---

## Dependencies

### Required Dependencies
- `pathlib`, `json`, `hashlib`, `re` - Standard library

### Optional Dependencies
- Julia interpreter - `Meta.parseall` syntax probe for `.jl` scripts in the pre-execution gate (advisory regex sweep without it)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Security Levels (actual `_SECURITY_LEVELS` in `processor.py`)
```python
_SECURITY_LEVELS = {
    "basic": {"scan_vulnerabilities": False, "default_block_on": None},
    "standard": {"scan_vulnerabilities": True, "default_block_on": None},
    "strict": {"scan_vulnerabilities": True, "default_block_on": "high"},
}
```
An explicit `block_on` kwarg makes any scanning level enforcement-capable.
Strict mode cannot disable vulnerability scanning, since doing so would
silently weaken the requested policy.

---

## Usage Examples

### Basic Security Validation
```python
from gnn.security.processor import process_security

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    security_level="standard",
)
```

### Pre-Validate a Policy Before Scanning
```python
from gnn.security import resolve_security_policy

policy = resolve_security_policy("strict")
assert policy.is_valid
# Use policy.block_on ("high") with the pre-execution gate:
from gnn.security import scan_script_for_execution

verdict = scan_script_for_execution(Path("output/12_execute_output/model.py"), block_on=policy.block_on)
if not verdict["ok"]:
    print(verdict["blocked"])
```

### Scan Generated Code Before Writing It
```python
from gnn.security import scan_source

result = scan_source(rendered_python, file_name="rendered_model.py", block_on="high")
if result["ok"]:
    out_path.write_text(rendered_code)
else:
    raise RuntimeError(f"render blocked: {result['blocked']}")
```

---

## Output Specification

### Output Products
- `security_results.json` - Processing results
- `security_summary.md` - Human-readable security summary

### Output Directory Structure
```
output/18_security_output/
├── security_results.json
└── security_summary.md
```

---

## Security Features

### Threat Detection
1. **Injection Pattern Scanning**: Regex pattern tables for OS command injection, suspicious imports, script-injection constructs, and hardcoded credentials
2. **Python AST Analysis**: Detects `shell=True`, dangerous calls, import-alias indirection, and dynamic execution
3. **Julia Script Analysis**: Advisory regex sweep plus `Meta.parseall` syntax probe (30 s timeout)
4. **Permission Checks**: World-writable mode detection (`stat.S_IWOTH`) for `.py` files; `perform_security_check` reports the real octal mode

### Pre-Execution Gate
`scan_script_for_execution()` is wired into Step 12's `execute_single_script()`:
rendered scripts are scanned before execution and blocked on findings at or
above the configured severity. Escape hatch: `GNN_ALLOW_UNSAFE_EXEC=1`.

---

## Error Handling

### Error Categories
1. **Scan Errors**: Per-file scan failures are logged and skipped
2. **Threat Findings**: High-severity findings block the pre-execution gate (configurable via `block_on`)
3. **Report Errors**: Write failures cause a False return

---

## Integration Points

### Orchestrated By
- **Script**: `18_security.py` (Step 18)
- **Function**: `process_security()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `src/gnn/execute/processor/` (via `execute/processor/single.py`) - Imports `scan_script_for_execution` for the Step 12 pre-execution gate
- `src/gnn/18_security.py` - Thin orchestrator (Step 18)
- `tests/security/*` - Security tests

### Data Flow
```
GNN Files → Pattern + AST Scanning → Severity Scoring → security_results.json + security_summary.md
Rendered Scripts → scan_script_for_execution → Block or Allow → Step 12 Execution
```

---

## Testing

### Test Files
- `tests/security/test_security_overall.py` - Module-level tests
- `tests/security/test_security_functional.py` - Functional tests
- `tests/security/test_pre_exec_gate.py` - Pre-execution gate tests
- `tests/security/test_security_mcp_tools.py` - MCP tool tests
- `tests/security/test_security_policy_and_source.py` - Policy resolver, severity helpers, `scan_source`, typed errors, permission semantics (added 1.7.0)
- `tests/security/test_sandbox.py`, `test_pygments_archetype_redos.py` - Auxiliary security tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/security/ \
    --cov=src/gnn/security --cov-report=term-missing
```

### Key Test Scenarios
1. Injection-pattern and AST vulnerability detection
2. Pre-execution gate blocking behavior and escape hatch
3. Security scoring and recommendations
4. Error handling with unscannable files
5. Policy resolution semantics and receipt compatibility (1.7.0)
6. In-memory source scanning with and without a threshold (1.7.0)
7. Fail-closed severity ranking for unknown/missing severities (1.7.0)
8. Real POSIX permission reporting and world-writable detection (1.7.0)

---

## MCP Integration

### Tools Registered
- `process_security` - Run security scanning and compliance checks on pipeline files
- `scan_gnn_file` - Lightweight security scan of a single GNN file
- `get_security_report` - Read saved reports from a previous security run
- `list_security_checks` - List the security checks performed (CVE scan, injection detection, path traversal, etc.)

### MCP File Location
- `src/gnn/security/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Security validation reports false positives
**Symptom**: Valid models reported as having vulnerabilities
**Cause**: Pattern rules may flag benign constructs
**Solution**:
- Use `security_level="basic"` to disable scanning (report-only pipeline still runs)
- Set an explicit `block_on` threshold to control what blocks
- Use `--verbose` for detailed scan logs

#### Issue 2: Rendered script blocked before execution
**Symptom**: Step 12 refuses to run a rendered script
**Cause**: `scan_script_for_execution` found findings at/above the blocking severity
**Solution**:
- Review the findings in the block report; fix the rendered script if genuinely unsafe
- Lower the threshold via `block_on` if the finding is advisory
- Last-resort escape hatch: set `GNN_ALLOW_UNSAFE_EXEC=1` (use only when the script is trusted)

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Injection-pattern and Python AST vulnerability scanning
- Severity-based security scoring
- Pre-execution gate for rendered scripts (Python + Julia)
- Security recommendations
- Pure policy resolution (`resolve_security_policy` + `ResolvedSecurityPolicy`)
- In-memory source scanning (`scan_source`)
- Shared severity helpers (`findings_at_or_above`, `count_by_severity`)
- Typed scan failure (`SecurityScanError`)

**Known Issues**:
- None currently

### 1.6.0 (previous)

**Features**:
- Injection-pattern and Python AST vulnerability scanning
- Severity-based security scoring
- Pre-execution gate for rendered scripts (Python + Julia)
- Security recommendations

### Roadmap
- **Next Version**: Enhanced threat detection
- **Future**: Real-time security monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Common Weakness Enumeration](https://cwe.mitre.org/)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
