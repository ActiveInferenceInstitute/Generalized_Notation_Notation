# MCP Module - Agent Scaffolding

## Module Overview

**Purpose**: Model Context Protocol implementation for standardized tool discovery, registration, and execution across all GNN modules

**Pipeline Step**: Step 21: Model Context Protocol processing (21_mcp.py)

**Category**: Protocol Integration / Tool Management

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

### Primary Responsibilities
1. Implement Model Context Protocol (MCP) for tool registration and discovery
2. Provide standardized interface for tool execution across modules
3. Enable inter-module communication and resource sharing
4. Manage MCP server lifecycle and client connections
5. Support multiple MCP transport protocols (stdio, HTTP)

### Key Capabilities
- Tool registration and discovery system
- Resource access and management
- JSON-RPC protocol implementation
- Server and client implementations
- Enhanced error handling and validation
- Performance monitoring and caching
- Concurrent execution control

---

## API Reference

### Public Functions

#### `process_mcp(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main MCP processing function called by orchestrator (21_mcp.py). Discovers and registers MCP tools from all modules.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for MCP results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `mcp_mode` (str, optional): Accepted via `**kwargs`; not consumed by `process_mcp` (default: `"tool_discovery"`)
- `enable_tools` (bool, optional): Accepted via `**kwargs`; not consumed (default: True)
- `transport` (str, optional): Accepted via `**kwargs`; not consumed (default: `"stdio"`)
- `**kwargs`: Forwarded to `initialize()` — recognized keys include `performance_mode`, `enable_caching`, `enable_rate_limiting`, `strict_validation`, `cache_ttl`, `tool_timeout` (aka `mcp_tool_timeout`), `modules_allowlist` (aka `mcp_modules_allowlist`), `per_module_timeout` (aka `mcp_per_module_timeout`), `overall_timeout` (aka `mcp_overall_timeout`). See the Default Settings table.

**Returns**: `bool` - True if MCP processing succeeded, False otherwise

**Example**:
```python
from gnn.mcp import process_mcp
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_mcp(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/21_mcp_output"),
    logger=logger,
    verbose=True,
    mcp_mode="tool_discovery",
    enable_tools=True,
    transport="stdio",
)
```

#### `register_module_tools(module_name: Optional[str] = None) -> bool | List[Dict[str, Any]]`
**Description**: Discover one (or all) pipeline modules and call their
`register_tools(mcp_instance)` function against the global singleton. Each
target module owns its tool definitions via `src/gnn/<module>/mcp.py`.

**Parameters**:
- `module_name` (Optional[str]): Single module to register (e.g. `"gnn"`).
  `None` triggers full discovery via `MCP.discover_modules()`.

**Returns**: `bool` when `module_name` is provided; a list of registered tool
dictionaries when discovering all modules.

#### `register_tools(mcp: Optional[MCP] = None) -> bool`
**Description**: Module-level helper equivalent to
`mcp.discover_modules()`. Registers every module's tools against the supplied
`MCP` instance (default: the global singleton). Exposed from
`mcp.server_core` and re-exported from the package.

**Returns**: `bool` — True when discovery runs to completion.

#### `get_available_tools() -> List[Dict[str, Any]]`
**Description**: Get list of all available MCP tools across all modules.

**Returns**: `List[Dict[str, Any]]` - List of tool information dictionaries with:
- `name` (str): Tool name
- `module` (str): Source module
- `description` (str): Tool description
- `schema` (Dict): Parameter schema
- `category` (str): Tool category

---

## MCP Protocol Implementation

### Core Classes

#### `MCP` - Main Protocol Class
**Description**: Core Model Context Protocol implementation

Tool execution validates object parameters against the registered JSON schema
before calling the handler. Type, required-field, authentication, and unknown
tool failures are surfaced as explicit MCP/JSON-RPC errors; structured domain
results that contain an `error` field remain normal tool results for clients to
inspect. Request, success, failure, concurrency, and timing metrics are updated
on the live execution path.

**Key Methods**:
- `initialize()` - Initialize MCP server
- `register_tools()` - Register available tools
- `handle_request()` - Process incoming requests
- `list_available_tools()` - List available tools

#### `MCPTool` - Tool Definition Class
**Description**: Represents a registered MCP tool

**Attributes**:
- `name` - Tool name
- `description` - Tool description
- `input_schema` - JSON schema for inputs
- `handler` - Function to execute tool

#### `MCPResource` - Resource Definition Class
**Description**: Represents accessible MCP resources

**Attributes** (see `src/gnn/mcp/models.py` for the dataclass):
- `uri_template` - URI template (e.g. `gnn://documentation/{doc_name}`)
- `retriever` - Callable returning the resource content
- `description` - Resource description
- `module` / `category` / `version` - Provenance metadata
- `mime_type` - Resource MIME type (default `application/json`)
- `cacheable` - Whether the resource may be cached
- `tags` - Free-form tags

### Transport Protocols

#### STDIO Transport
- **Purpose**: Standard input/output communication
- **Use Case**: Local tool execution and testing
- **Implementation**: `mcp.server_stdio`
- **Framing**: UTF-8 JSON-RPC lines have a 1 MiB per-line cap. Each server
  retains coalesced-read suffix bytes; fragmented lines remain one request.
- **EOF**: Already admitted requests and their responses drain before exit,
  including a final line without a newline. Oversized lines fail closed.
- **Cancellation**: The maintained stdio dispatcher does not implement
  request cancellation; an explicit `notifications/cancelled` request reports
  method-not-found. POSIX operator SIGINT stops the server's transport loop.

#### HTTP Transport
- **Purpose**: Web-based MCP server
- **Use Case**: Remote tool access and web integration
- **Implementation**: `mcp.server_http`

---

## Dependencies

### Required Dependencies
- `json` - JSON-RPC protocol implementation
- `pathlib` - Path and URI handling
- `typing` - Type annotations and validation
- `logging` - Request/response logging

### Optional Dependencies
- `aiohttp` - HTTP server implementation (recovery: basic HTTP)
- `websockets` - currently unused by the maintained server transports (stdio/HTTP)
- `fastapi` - REST API framework (recovery: basic HTTP)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing (processor.py)

---

## Configuration

### Environment Variables

Security and hardening knobs (`GNN_MCP_TOKEN`, `GNN_MCP_ALLOW_INSECURE_LOCAL`,
`GNN_MCP_ALLOW_UNSAFE_TOOLS`, `GNN_MCP_RATE_LIMIT_PER_MINUTE`,
`GNN_MCP_SAFE_TOOLS`, `GNN_MCP_SAFE_RESOURCES`, `MCP_SDK_PATH`) are documented
in `src/gnn/mcp/MCP_DOCUMENTATION.md`.

### Configuration Files

No dedicated `mcp_config.yaml` is read. Runtime configuration is set via the
`initialize(...)` factory (see `mcp.mcp`) and the `GNN_MCP_*` env vars — **not**
via step-21 CLI flags (`21_mcp.py` exposes only `--target-dir`, `--output-dir`,
`--verbose`).

### Default Settings

The live defaults are embedded in `mcp.mcp.MCP.__init__` and exposed via
`initialize(...)`. Authoritative table (`initialize()` defaults; `MCP.__init__`
sets `enable_caching`/`enable_rate_limiting` to `True` until a `performance_mode`
override applies):

| Knob | Default | Set via |
|------|---------|---------|
| `performance_mode` | `"low"` | `initialize(performance_mode=...)` |
| `strict_validation` | `False` (from mode) | `initialize(strict_validation=...)` / `MCP(strict_validation=...)` |
| `enable_caching` | `False` (from mode) | `initialize(enable_caching=...)` / `MCP(enable_caching=...)` |
| `enable_rate_limiting` | `False` (from mode) | `initialize(enable_rate_limiting=...)` |
| `cache_ttl` | `300.0` s | `initialize(cache_ttl=...)` — default TTL for entries stored by result-cache-eligible tools |
| `tool_timeout` | `None` (un-timed) | `initialize(tool_timeout=...)` — default timeout (seconds) for tools that did not register one; `None` leaves such tools un-timed |
| `per_module_timeout` | `30.0` s | `initialize(per_module_timeout=...)` |
| `overall_timeout` | `120.0` s | `initialize(overall_timeout=...)` |
| `modules_allowlist` | `None` (all modules) | `initialize(modules_allowlist=...)` |

See `src/gnn/mcp/MCP_DOCUMENTATION.md` → *Configuration* for the complete surface
and `src/gnn/mcp/SKILL.md` for the capabilities-API summary.

---

## Usage Examples

### Basic MCP Processing
```python
from gnn.mcp.processor import process_mcp

success = process_mcp(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/21_mcp_output"),
    logger=logger,
    mcp_mode="tool_discovery",
)
```

### Tool Registration
```python
from gnn.mcp import register_module_tools

success = register_module_tools("gnn")
all_tools = register_module_tools()
```

### Tool Discovery
```python
from gnn.mcp import get_available_tools

tools = get_available_tools()
for tool in tools:
    print(f"Tool: {tool['name']} - {tool['description']}")
```

---

## Output Specification

### Output Products
- `mcp_results.json` - Detailed MCP processing report
- `mcp_processing_summary.json` - MCP processing summary
- `registered_tools.json` - All registered tools information (written when tools are available)

Directory round-trip tools write `round_trip_results.json`. A file-read error
remains a failed file in its saved `passed`/`failed` counts and in the public
tool result. An empty input directory remains successful no-work with zero
files passed or failed; parser mode and per-file error details are retained.

### Output Directory Structure
```
output/21_mcp_output/
├── mcp_results.json
├── mcp_processing_summary.json
└── registered_tools.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds (tool registration)
- **Memory**: ~10-20MB for tool registry
- **Status**: Production Ready
### Expected Performance
- **Fast Path**: <1s for tool discovery
- **Slow Path**: ~5s for comprehensive tool validation
- **Memory**: ~5-15MB for typical tool sets

---

## Error Handling

### Graceful Degradation
- **No transport libraries**: Recovery to stdio-only mode
- **Tool registration failures**: Continue with available tools
- **Server startup failures**: Recovery to client-only mode

### Error Categories
1. **Protocol Errors**: Invalid JSON-RPC requests/responses
2. **Tool Errors**: Tool execution failures or timeouts
3. **Transport Errors**: Network or I/O communication failures
4. **Validation Errors**: Invalid tool schemas or parameters

---

## Integration Points

### Orchestrated By
- **Script**: `21_mcp.py` (Step 21)
- **Function**: `process_mcp()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns (processor.py)

### Imported By
- `tests/mcp/test_mcp_overall.py`, `test_mcp_tools.py`, `test_mcp_functional.py`, `test_mcp_audit.py`, `test_mcp_http_auth.py`, `test_mcp_performance.py`, `test_mcp_standalone_modules.py`, `test_mcp_configurability.py` - MCP tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Module Tools → MCP Registration → Tool Discovery → Execution Requests → Response Handling
```

---

## Testing

### Test Files
- `tests/mcp/test_mcp_overall.py`, `test_mcp_tools.py`, `test_mcp_functional.py`, `test_mcp_audit.py`, `test_mcp_http_auth.py`, `test_mcp_performance.py`, `test_mcp_standalone_modules.py`, `test_mcp_configurability.py`

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/mcp/ \
    --cov=src/gnn/mcp --cov-report=term-missing
```
### Key Test Scenarios
1. Tool registration and discovery across modules
2. JSON-RPC protocol compliance
3. Multiple transport protocol operation
4. Error handling with malformed requests
5. Performance under concurrent tool execution

---

## MCP Integration

### Tools Registered (Across All Modules)
- `gnn_*` - GNN file processing tools
- `analysis_*` - Statistical analysis tools
- `visualization_*` - Visualization generation tools
- `render_*` - Code generation tools
- `gui_*` - GUI interaction tools

### Tool Categories
- **File Processing**: GNN parsing, validation, transformation
- **Analysis**: Statistical analysis, complexity metrics
- **Visualization**: Chart generation, interactive displays
- **Code Generation**: Multi-framework code rendering
- **Model Management**: Registry operations, version control

---

## Troubleshooting

### Common Issues

#### Issue 1: Tool registration fails
**Symptom**: Tools not discovered or registered  
**Cause**: Module discovery issues or tool definition errors  
**Solution**: 
- Verify all modules have `mcp.py` files with tool definitions
- Check tool function signatures match expected format
- Use `--verbose` flag for detailed discovery logs
- Review MCP tool registration patterns

#### Issue 2: Tool execution errors
**Symptom**: Tools registered but execution fails  
**Cause**: Parameter validation errors or function implementation issues  
**Solution**:
- Verify tool parameter schemas are correct
- Check tool function implementations handle errors
- Review tool execution logs for specific errors
- Validate tool inputs match schema

---

## Version History

Module `__version__` is `3.2.0` (`__init__.py`), matching the unified pipeline version. No formal changelog is maintained in this file.

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [MCP Implementation Spec](mcp_implementation_spec.md)
- [MCP Integration Guide](../../../docs/mcp/)

### External Resources
- [Model Context Protocol Specification](https://modelcontextprotocol.io)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Architecture Compliance**: Thin Orchestrator Pattern (delegates to `mcp.processor.process_mcp` → `MCP.discover_modules`)


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
