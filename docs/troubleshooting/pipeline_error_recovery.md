# Pipeline Error Recovery Guide

> **📋 Document Metadata**  
> **Type**: Troubleshooting Guide | **Audience**: Developers & Operators | **Complexity**: Intermediate  
> **Cross-References**: [Pipeline Architecture](../gnn/operations/gnn_tools.md) | [Testing Guide](../testing/README.md)

## Overview

This guide provides comprehensive error recovery procedures for the GNN pipeline, focusing on common failure modes and their resolutions.

## Critical Error Patterns

### 1. NumPy Recursion Error (Step 11 - Render)

#### Symptoms
- Error message: `RecursionError: maximum recursion depth exceeded`
- Occurs during type imports in NumPy
- Typically in `numpy._typing`

#### Resolution
```python
import sys

sys.setrecursionlimit(3000)  # Increase from default 1000

# If using in code:
try:
    import numpy as np
except RecursionError:
    import sys

    sys.setrecursionlimit(3000)
    import numpy as np
```

#### Prevention
- Add recursion limit configuration to `1_setup.py`
- Monitor memory usage during type checking
- Consider using lightweight type checking for large models

### 2. Async/Await Issues (Step 13 - LLM)

#### Symptoms
- Warning: `RuntimeWarning: coroutine 'OpenAIProvider.generate_response' was never awaited`
- Incomplete LLM analysis results
- Hanging API calls

#### Resolution
```python
# Correct implementation:
async def analyze_gnn_file(file_path: Path) -> Dict[str, Any]:
    async with OpenAIProvider() as provider:
        response = await provider.analyze(file_path.read_text())
        return {"analysis": response.strip()}


# In synchronous code:
import asyncio

result = asyncio.run(analyze_gnn_file(file_path))
```

#### Prevention
- Use async/await consistently throughout LLM module
- Implement proper cleanup in async context managers
- Add timeout handling for API calls

### 3. Missing GNN Lightweight Processing

#### Symptoms
- Error: `cannot import name 'process_gnn_directory_lightweight'`
- GNN processing fallback fails
- Step 3 warnings

#### Resolution
The lightweight processor exists in the package — import it rather than
reimplementing it:

```python
from gnn import process_gnn_directory_lightweight

results = process_gnn_directory_lightweight("input/gnn_files")
```

If that import fails, your environment is out of sync: run `uv sync --extra dev`
and check `uv run gnn health` before debugging further.

#### Prevention
- Keep the `gnn` export list in `src/gnn/__init__.py` authoritative for the symbols above
- Run the focused Step 3 tests after touching processing imports
- Document any new fallback surface in the module AGENTS/README pair

### 4. JAX/TPU Initialization (Step 12 - Execute)

#### Symptoms
- Error: `INTERNAL: Failed to open libtpu.so`
- JAX device initialization failures
- Missing accelerator support

#### Resolution
```python
# The executor sets the JAX platform per run (see src/gnn/execute/processor/single.py,
# which maps GNN_JAX_PLATFORM onto JAX_PLATFORM_NAME);
# for ad-hoc work, fall back to CPU explicitly:
def initialize_jax_devices():
    """Initialize JAX with graceful fallback."""
    try:
        import jax

        devices = jax.devices()
    except:
        # Fallback to CPU-only
        import os

        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import jax

        devices = jax.devices()
    return devices
```

#### Prevention
- Check hardware capabilities during setup
- Provide CPU fallback configurations
- Document platform-specific requirements

## Step-Specific Recovery Procedures

### Step 11 (Render)
1. Check Python recursion limit
2. Verify NumPy installation
3. Monitor memory usage
4. Use incremental rendering for large models

### Step 12 (Execute)
1. Verify framework availability
2. Check hardware requirements
3. Validate simulation configurations
4. Monitor resource usage

### Step 13 (LLM)
1. Verify API credentials
2. Check network connectivity
3. Monitor rate limits
4. Implement retry mechanisms

## General Recovery Guidelines

### 1. Logging and Diagnostics
- Enable verbose logging: `--verbose`
- Check run logs in `output/00_pipeline_logs/`
- Monitor system resources
- Review pipeline execution summary

### 2. Resource Management
- Monitor memory usage
- Track disk space
- Check CPU utilization
- Manage network connections

### 3. Data Integrity
- Validate input files
- Check output consistency
- Verify file permissions
- Monitor file system operations

### 4. Error Reporting
- Collect error details
- Generate diagnostic reports
- Track error patterns
- Update error documentation

## Automated Recovery and Diagnostic Tools

### 1. Preflight and Environment Checker
```bash
# Check environment health & dependencies
uv run gnn health --strict
uv run python src/gnn/1_setup.py --verbose
```

### 2. Preflight Health Checker
```bash
uv run gnn preflight
```

## Contributing to Error Recovery

When encountering new error patterns:

1. Document the error details
2. Identify root causes
3. Develop recovery procedures
4. Update this guide
5. Add regression tests

## References

- [Pipeline Architecture](../gnn/operations/gnn_tools.md)
- [Testing Guide](../testing/README.md)
- [Configuration Guide](../configuration/README.md)
- [Development Guide](../development/README.md) 