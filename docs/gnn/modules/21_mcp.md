# Step 21: Mcp

## Architectural Mapping

**Orchestrator**: `src/gnn/21_mcp.py` (53 lines)
**Implementation Layer**: `src/gnn/mcp/`

## Module Description

This directory contains the comprehensive Model Context Protocol (MCP) implementation for the GeneralizedNotationNotation (GNN) project. The MCP server exposes all GNN functionalities as standardized tools that can be accessed by MCP-compatible clients.


The GNN MCP implementation provides:
- **Core MCP Server**: JSON-RPC 2.0 compliant server with tool and resource management
- **Multiple Transport Layers**: stdio and HTTP transport support
- **Comprehensive Tool Ecosystem**: Tools from all GNN modules (gnn, type_checker, export, visualization, etc.)
- **Meta-Tools**: Server introspection and diagnostic capabilities
- **CLI Interface**: Command-line access to all MCP functionality
- **Extensible Architecture**: Easy addition of new tools and resources



```mermaid
graph TD
    Pipeline[21_mcp.py] --> Processor[process_mcp]
    Processor --> Discovery[Tool Discovery]
    Processor --> Registration[Tool Registration]
    Processor --> Server[MCP Server]
```

## Agent Identity & Capabilities

# MCP Module - Agent Scaffolding

## Module Overview

**Purpose**: Model Context Protocol implementation for standardized tool discovery, registration, and execution across all GNN modules

**Pipeline Step**: Step 21: Model Context Protocol processing (21_mcp.py)

**Category**: Protocol Integration / Tool Management

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

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
- `mcp_mode` (str, optional): MCP mode ("tool_discovery", "server", "client") (default: "tool_discovery")
- `enable_tools` (bool, optional): Enable MCP tools functionality (default: True)
- `transport` (str, optional): Transport protocol ("stdio", "http") (default: "stdio")
- `**kwargs`: Additional MCP options

**Returns**: `bool` - True if MCP processing succeeded, False otherwise

**Example**:
```python
from mcp import process_mcp
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

#### `register_module_tools(module_name: str | None = None) -> Any`
**Description**: Register tools for a specific module, or all modules if no name given (`src/gnn/mcp/processor.py`). Tools are discovered from each module's `mcp.py` `register_tools`, not passed in by the caller.

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

**Key Methods**:
- `initialize()` - Initialize MCP server
- `register_tools()` - Register available tools
- `handle_request()` - Process incoming requests
- `list_tools()` - List available tools

#### `MCPTool` - Tool Definition Class
**Description**: Represents a registered MCP tool

**Attributes**:
- `name` - Tool name
- `description` - Tool description
- `input_schema` - JSON schema for inputs
- `handler` - Function to execute tool

#### `MCPResource` - Resource Definition Class
**Description**: Represents accessible MCP resources

**Attributes**:
- `uri` - Resource URI
- `name` - Resource name
- `description` - Resource description
- `mime_type` - Resource MIME type

### Transport Protocols

#### STDIO Transport
- **Purpose**: Standard input/output communication
- **Use Case**: Local tool execution and testing
- **Implementation**: `mcp.server_stdio`

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
- `fastapi` - REST API framework (recovery: basic HTTP)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables

No dedicated `mcp_config.yaml` is read and there is no `DEFAULT_MCP_SETTINGS`
constant. Runtime configuration is set via the `initialize(...)` factory
(`src/gnn/mcp/mcp.py`) and the `GNN_MCP_*` environment variables — not via
step-21 CLI flags (`src/gnn/21_mcp.py` exposes only `--target-dir`,
`--output-dir`, `--verbose`).

---

## Usage Examples

### Basic MCP Processing
```python
from mcp.processor import process_mcp

success = process_mcp(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/21_mcp_output"),
    logger=logger,
    mcp_mode="tool_discovery",
)
```

### Tool Registration
```python
from mcp import register_module_tools

# Register tools for one module (or all modules when called with no argument)
register_module_tools("gnn")
register_module_tools()
```


---

## Output Specification

### Output Products
- `mcp_processing_summary.json` - MCP processing summary
- `registered_tools.json` - All registered tools information
- `mcp_server_status.json` - Server status and configuration
- `tool_execution_log.json` - Tool execution history

### Output Directory Structure
```
output/21_mcp_output/
├── mcp_processing_summary.json
├── registered_tools.json
├── mcp_server_status.json
└── tool_execution_log.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds (tool registration)
- **Memory**: ~10-20MB for tool registry
- **Status**: ✅ Production Ready

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
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests/mcp/test_mcp_overall.py` - MCP module tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Module Tools → MCP Registration → Tool Discovery → Execution Requests → Response Handling
```

---

## Testing

### Test Files
- `tests/mcp/test_mcp_tools.py` - Tool registration tests
- `tests/mcp/test_mcp_functional.py` - Functional tests
- `tests/mcp/test_mcp_audit.py` - Audit tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/mcp/ --cov=gnn.mcp --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

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

### Current package version

See [pyproject.toml](../../../pyproject.toml).

**Features**:
- Tool registration and discovery
- Resource access and management
- JSON-RPC protocol implementation
- Server and client implementations
- Enhanced error handling
- Performance monitoring

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced transport protocols
- **Future**: Real-time tool monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [MCP Implementation Spec](../../../src/gnn/mcp/mcp_implementation_spec.md)
- [MCP Integration Guide](../../../docs/mcp/)

### External Resources
- [Model Context Protocol Specification](https://modelcontextprotocol.io)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/mcp/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/mcp/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/mcp/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/mcp/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/mcp](../../../src/gnn/mcp)
