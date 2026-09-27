# Step 18: Security

## Architectural Mapping

**Orchestrator**: `src/gnn/18_security.py` (52 lines)
**Implementation Layer**: `src/gnn/security/`

## Module Description

This module provides comprehensive security validation and access control capabilities for GNN models and pipeline components, including vulnerability assessment, security analysis, and compliance checking.


```
src/gnn/security/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── processor.py                   # Scanners, process_security, pre-execution script gate
└── mcp.py                         # Model Context Protocol integration
```



Main function for processing security-related tasks.

## Agent Identity & Capabilities

# Security Module - Agent Scaffolding

## Module Overview

**Purpose**: Security validation, access control, and threat detection for the GNN processing pipeline

**Pipeline Step**: Step 18: Security validation (18_security.py)

**Category**: Security / Access Control

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Security validation of GNN models and pipeline components
2. Access control and authorization management
3. Threat detection and vulnerability assessment
4. Secure data handling and encryption
5. Security policy enforcement
6. Audit logging and compliance reporting

### Key Capabilities
- Model security validation and risk assessment
- Access control for sensitive operations
- Threat detection and mitigation
- Data encryption and secure storage
- Security policy configuration
- Audit trail maintenance
- Compliance reporting

---

## API Reference

### Public Functions

#### `process_security(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main security processing function called by orchestrator (18_security.py). Validates security, assesses vulnerabilities, and checks compliance.

**Parameters**:
- `target_dir` (Path): Directory containing files to validate
- `output_dir` (Path): Output directory for security reports
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `security_level` (str, optional): Security validation level ("basic", "standard", "strict") (default: "standard")
- `check_vulnerabilities` (bool, optional): Enable vulnerability scanning (default: True)
- `check_compliance` (bool, optional): Enable compliance checking (default: True)
- `compliance_standards` (List[str], optional): Standards to check against (default: ["OWASP Top 10"])
- `**kwargs`: Additional security options

**Returns**: `bool` - True if security validation passed, False otherwise

**Example**:
```python
from security import process_security
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    logger=logger,
    verbose=True,
    security_level="strict",
    compliance_standards=["OWASP Top 10", "CWE"],
)
```

#### `perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform a security check on a single GNN file (sensitive-pattern scanning and hashing).

**Parameters**:
- `file_path` (Path): Path to GNN file to check
- `verbose` (bool): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Security check results with:
- `file_hash` (str): SHA-256 hash of the file contents
- `sensitive_patterns` (List[Dict]): Detected sensitive patterns (e.g. `password`, `api_key`, `secret`)
- `file_permissions` (str): Simplified file permission summary
- `security_score` (float): Security score computed from detected patterns
- `check_timestamp` (str): ISO timestamp of the check

#### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
**Description**: Check a GNN (or generated Python) file for security vulnerabilities and unsafe file permissions. Combines regex pattern matching (e.g. `eval(`, `exec(`, `subprocess.*`) with Python AST analysis of generated `.py` files, and flags world-writable files via `os.access`.

**Parameters**:
- `file_path` (Path): File path to check
- `verbose` (bool): Enable verbose output (default: False)

**Returns**: `List[Dict[str, Any]]` - Detected vulnerabilities, each with `vulnerability_type`, `detection_method` (`"regex"`, `"ast"`, or `"permission_check"`), and the matched `line`/`pattern`.

---

## Dependencies

### Required Dependencies
- `cryptography` - Encryption and hashing
- `pathlib` - Path manipulation
- `json` - Data serialization

### Optional Dependencies
- `PyYAML` - Configuration file parsing
- `requests` - External security service integration

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Security Levels
```python
SECURITY_LEVELS = {
    "basic": {
        "validate_file_integrity": True,
        "check_basic_permissions": True,
        "log_access": True,
    },
    "standard": {
        "validate_file_integrity": True,
        "check_basic_permissions": True,
        "log_access": True,
        "scan_for_malicious_content": True,
        "validate_model_structure": True,
    },
    "strict": {
        "validate_file_integrity": True,
        "check_basic_permissions": True,
        "log_access": True,
        "scan_for_malicious_content": True,
        "validate_model_structure": True,
        "encrypt_sensitive_data": True,
        "require_authorization": True,
    },
}
```

### Security Policies
```python
SECURITY_POLICIES = {
    "allowed_file_types": [".md", ".json", ".yaml"],
    "max_file_size_mb": 100,
    "require_encryption": False,
    "audit_all_operations": True,
    "block_suspicious_content": True,
}
```

---

## Usage Examples

### Basic Security Validation
```python
from security.processor import process_security

success = process_security(
    target_dir="input/gnn_files",
    output_dir="output/18_security_output",
    security_level="standard",
)
```

### Model Security Check
```python
from security.processor import perform_security_check

security_result = perform_security_check(
    file_path="models/sensitive_model.md", verbose=True
)

print(f"Security score: {security_result['security_score']}")
if security_result["sensitive_patterns"]:
    print("Sensitive patterns found:")
    for pattern in security_result["sensitive_patterns"]:
        print(f"  - {pattern['context']} (line {pattern['line']})")
```

### Vulnerability Check
```python
from security.processor import check_vulnerabilities

vulnerabilities = check_vulnerabilities("models/confidential.md")

if vulnerabilities:
    print("Vulnerabilities found:")
    for vuln in vulnerabilities:
        print(f"  - {vuln['vulnerability_type']} (line {vuln['line']})")
else:
    print("No vulnerabilities detected")
```

---

## Pre-execution script gate

Step 18 runs after Step 12, so on its own it is forensic. `scan_script_for_execution(script_path, *, block_on="high")` in `src/gnn/security/processor.py` closes that gap: Step 12 (`src/gnn/execute/processor/`) calls it on every rendered script before launching it and refuses to run the script when the verdict is `ok=False` (error type `SecurityGateBlocked`).

- **Python scripts** are analysed with the AST scanner; findings at or above `block_on` block execution.
- **Julia scripts** are validated with `Meta.parseall` through a `julia` subprocess (`_julia_meta_parseall`, 30 s timeout). Parsing builds the AST without executing the script. Only an explicit parse failure (`GNN_PARSE_FAIL`) blocks; when Julia is absent, times out, or the probe itself does not run (e.g. a launcher with no installed toolchain — the v3.2.0 fix), the gate degrades to the advisory regex sweep `_julia_regex_sweep` with `scanned=False`.
- Setting `GNN_ALLOW_UNSAFE_EXEC` in the environment bypasses the gate (handled in `src/gnn/execute/processor/single.py::execute_single_script`).

## Output Specification

### Output Products
- `security_results.json` - Structured findings and per-file results
- `security_summary.md` - Human-readable security summary

### Output Directory Structure
```
output/18_security_output/
├── security_results.json
└── security_summary.md
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds per model
- **Memory**: ~20-50MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Validation**: < 1 second
- **Standard Validation**: 1-2 seconds
- **Strict Validation**: 2-5 seconds
- **Threat Detection**: Variable based on content

---

## Security Features

### Threat Detection
1. **Malicious Content Detection**: Pattern-based threat detection
2. **Suspicious Script Detection**: Script injection detection
3. **Data Exfiltration Detection**: Unauthorized data access patterns
4. **Cryptographic Validation**: Digital signature verification

### Access Control
1. **File Permission Validation**: OS-level permission checks
2. **Operation Authorization**: Role-based access control
3. **Audit Logging**: Comprehensive operation logging
4. **Security Context**: Security-aware operation context

### Data Protection
1. **Encryption Support**: Sensitive data encryption
2. **Secure Storage**: Protected data storage
3. **Key Management**: Encryption key lifecycle management
4. **Data Sanitization**: Secure data cleanup

---

## Error Handling

### Security Errors
1. **Access Denied**: Insufficient permissions
2. **Threat Detected**: Malicious content found
3. **Validation Failed**: Security requirements not met
4. **Encryption Error**: Cryptographic operation failure

### Recovery Strategies
- **Access Issues**: Request elevated permissions
- **Threats**: Isolate and report suspicious content
- **Validation**: Provide remediation guidance
- **Encryption**: Use alternative encryption methods

---

## Integration Points

### Orchestrated By
- **Script**: `18_security.py` (Step 18)
- **Function**: `process_security()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- All pipeline steps requiring security validation
- `tests.test_security_*` - Security tests

### Data Flow
```
File Input → Security Validation → Threat Detection → Access Control → Security Report → Pipeline Continuation
```

---

## Testing

### Test Files
- `tests/security/test_security_overall.py` - Module-level tests
- `tests/security/test_security_functional.py` - Functional tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/security/ --cov=security --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Security validation with various threat types
2. Access control enforcement
3. Encryption and data protection
4. Audit logging functionality
5. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `process_security` - Run the Step 18 scan over a directory
- `scan_gnn_file` - Scan one GNN file
- `get_security_report` - Read the latest security results
- `list_security_checks` - List the checks the scanner applies

### MCP File Location
- `src/gnn/security/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Security validation reports false positives
**Symptom**: Valid models reported as having vulnerabilities  
**Cause**: Security rules too strict or outdated  
**Solution**: 
- Use `process_security(..., security_level="basic")` for lenient validation (Python API kwarg; there is no `--security-level` CLI flag)
- Review security rules and update if needed
- Check compliance standards are appropriate
- Use `--verbose` flag for detailed validation logs

#### Issue 2: Access control checks fail
**Symptom**: Valid operations blocked by access control  
**Cause**: Permission configuration incorrect or overly restrictive  
**Solution**:
- Verify file permissions are correct
- Check access control configuration
- Review security policy settings
- Ensure user has required permissions

---

## Version History

### Current module status

**Features**:
- Security validation
- Access control
- Threat detection
- Vulnerability assessment
- Compliance reporting

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced threat detection
- **Future**: Real-time security monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Security Guide](../../../docs/security/)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Common Weakness Enumeration](https://cwe.mitre.org/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

---
## Documentation
- **[README](../../../src/gnn/security/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/security/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/security/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/security/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/security](../../../src/gnn/security)
