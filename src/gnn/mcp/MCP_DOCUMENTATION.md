# Model Context Protocol (MCP) Implementation Documentation

## Overview

The Model Context Protocol (MCP) implementation for the GeneralizedNotationNotation (GNN) project provides a server with features for tool discovery, registration, execution, and monitoring. This implementation extends the standard MCP specification with error handling, performance monitoring, caching, rate limiting, and thread safety.

## Key Features

### Core Features
- **JSON-RPC 2.0 Compliance**: Compliance with the MCP specification
- **Dynamic Module Discovery**: Automatic discovery and loading of MCP modules
- **Tool and Resource Registration**: Tool and resource management
- **Multiple Transport Layers**: Support for stdio and HTTP transport
- **CLI Interface**: Command-line access to MCP functionality

### Features
- **Advanced Error Handling**: Detailed error reporting with context information
- **Performance Monitoring**: Comprehensive metrics and health monitoring
- **Caching System**: Intelligent result caching with TTL support
- **Rate Limiting**: Configurable rate limiting per tool
- **Concurrent Execution Control**: Limits on concurrent tool executions
- **Thread Safety**: Full thread-safe operations with proper locking
- **Enhanced Validation**: Comprehensive parameter validation with detailed error messages
- **Health Monitoring**: Real-time health status and diagnostics
- **Memory Management**: Efficient memory usage and cleanup

## Architecture

### Core Components

```
src/gnn/mcp/
├── mcp.py                 # Enhanced core MCP server implementation
├── server_stdio.py        # stdio transport server
├── server_http.py         # HTTP transport server
├── cli.py                 # Command-line interface
├── meta_mcp.py           # Meta-tools for server introspection
├── sympy_mcp.py          # SymPy integration tools
├── sympy_mcp_client.py   # SymPy MCP client
├── npx_inspector.py      # NPX inspector utilities
├── README.md             # Basic documentation
└── *.md                  # Additional documentation files
```

### Enhanced Data Structures

#### MCPTool
Enhanced tool representation with additional metadata:

```python
@dataclass
class MCPTool:
    name: str  # Tool name
    func: Callable  # Tool function
    schema: Dict[str, Any]  # JSON schema for parameters
    description: str  # Tool description
    module: str = ""  # Source module
    category: str = ""  # Tool category
    version: str = "1.0.0"  # Tool version
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    examples: List[Dict[str, Any]] = field(default_factory=list)  # Usage examples
    experimental: bool = False  # Experimental flag
    timeout: Optional[float] = None  # Execution timeout
    max_concurrent: int = 1  # Max concurrent executions
    requires_auth: bool = False  # Authentication requirement
    rate_limit: Optional[float] = None  # Rate limit (requests/second)
    cache_ttl: Optional[float] = None  # Cache TTL
    input_validation: bool = True  # Enable input validation
    output_validation: bool = True  # Enable output validation
```

#### MCPResource
Enhanced resource representation:

```python
@dataclass
class MCPResource:
    uri_template: str  # URI template
    retriever: Callable  # Resource retriever function
    description: str  # Resource description
    module: str = ""  # Source module
    category: str = ""  # Resource category
    version: str = "1.0.0"  # Resource version
    mime_type: str = "application/json"  # MIME type
    cacheable: bool = True  # Cacheable flag
    tags: List[str] = field(default_factory=list)  # Tags
    timeout: Optional[float] = None  # Retrieval timeout
    requires_auth: bool = False  # Authentication requirement
    rate_limit: Optional[float] = None  # Rate limit
    cache_ttl: Optional[float] = None  # Cache TTL
    compression: bool = False  # Compression flag
    encryption: bool = False  # Encryption flag
```

### Error Classes

Errors carry both the wire-level JSON-RPC code and local context (tool name,
module name, optional payload) so server logs can be correlated with client
stack traces:

```python
class MCPError(Exception):
    """Base class for MCP related errors."""

    def __init__(
        self,
        message: str,
        code: int = -32000,
        data: Optional[Any] = None,
        tool_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.data = data or {}
        self.tool_name = tool_name
        self.module_name = module_name
        self.timestamp = time.time()


class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""


class MCPResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""


class MCPInvalidParamsError(MCPError):
    """Raised when tool parameters are invalid."""


class MCPToolExecutionError(MCPError):
    """Raised when tool execution fails."""


class MCPSDKNotFoundError(MCPError):
    """Raised when required SDK is not found."""


class MCPValidationError(MCPError):
    """Raised when validation fails."""


class MCPModuleLoadError(MCPError):
    """Raised when a module fails to load."""


class MCPPerformanceError(MCPError):
    """Raised when performance thresholds are exceeded."""
```

## Configuration

All runtime knobs live on the singleton :class:`mcp.MCP` instance and are
settable through ``initialize(...)`` or the pipeline step ``process_mcp(**kwargs)``.
The table below is the authoritative list — any additional configuration must
be added to :func:`mcp.mcp.initialize`.

| Knob | Type | Default | Effect |
|------|------|---------|--------|
| `performance_mode` | `"low"` \| `"medium"` \| `"high"` | `"low"` | Bulk toggle — sets the four booleans below to sensible preset values |
| `enable_caching` | `bool` | mode-derived | Toggle result-cache for tool invocations |
| `enable_rate_limiting` | `bool` | mode-derived | Enforce per-tool RPS limits |
| `strict_validation` | `bool` | mode-derived | Validate every call against its JSON schema |
| `cache_ttl` | `float` (s) | `300.0` | Default TTL for result-cache entries; a tool is cached only when it registered `cacheable=True`, and its own `cache_ttl` overrides this default |
| `tool_timeout` | `float` \| `None` | `None` | Default timeout (seconds) for tools that did not register their own timeout; `None` leaves those tools un-timed |
| `modules_allowlist` | `list[str]` \| `None` | `None` | Restrict discovery to the named modules under `src/` |
| `per_module_timeout` | `float` (s) | `30.0` | Max wall-clock per module during discovery |
| `overall_timeout` | `float` (s) | `120.0` | Max wall-clock for parallel discovery |
| `force_refresh` | `bool` | `False` | Re-run discovery even if modules are already loaded |

### Setting configuration at runtime

`21_mcp.py` exposes only the standard orchestrator flags (`--target-dir`,
`--output-dir`, `--verbose`); it does **not** define `--performance-mode`,
`--mcp-strict-validation`, `--mcp-cache-ttl`, `--mcp-modules-allowlist`,
`--mcp-per-module-timeout`, or `--mcp-overall-timeout`. Set those knobs by
passing them as `process_mcp(**kwargs)` (they are forwarded to `initialize()`)
or via the `GNN_MCP_*` environment variables:

```bash
python -c "from mcp import process_mcp; process_mcp('input/gnn_files', 'output', performance_mode='high', strict_validation=True, cache_ttl=120.0, modules_allowlist=['gnn','execute','render'])"
```

### Registered Tool Inventory

The MCP auto-discovers every module under `src/` that exposes `mcp.py`. A live
run should load all runtime MCP modules, expose canonical JSON schemas, and set
non-empty module/category metadata for every registered tool. Use the audit
tests for the current ground-truth inventory:

```bash
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py -q
PYTHONPATH=src uv run python src/gnn/mcp/validate_tools.py
```

The parent `gui` MCP module exposes the nested `oxdraw.*` tools; `doc` and
`lsp` intentionally register no callable tools while still participating in
module discovery.

## Installation and Setup

### Prerequisites
- Python 3.11 through 3.14, matching `pyproject.toml`
- Required dependencies resolved from `pyproject.toml` / `uv.lock` with `uv sync`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Generalized_Notation_Notation

# Install dependencies using UV (recommended)
uv sync

# Or install with optional MCP dependencies
uv sync --extra dev

# Verify installation
python -m gnn.mcp.cli --help
```

### Basic Usage

#### Starting the MCP Server

**stdio Transport (Recommended for local use):**
```bash
python -m gnn.mcp.server_stdio
```

**HTTP Transport (Authenticated local orchestration):**
```bash
GNN_MCP_TOKEN=local-dev-token python -m gnn.mcp.cli server --transport http --host 127.0.0.1 --port 8080
```

HTTP exposes only safe tools by default. Resource reads are denied unless an
exact URI is listed in `GNN_MCP_SAFE_RESOURCES`; keep stdio transport for broad
local resource access.
The HTTP capability response is filtered to this exposed safe surface.

#### Using the CLI

**List all available tools:**
```bash
python -m gnn.mcp.cli --format human list
```

**Execute a tool:**
```bash
python -m gnn.mcp.cli execute get_pipeline_status
```

**Get server status:**
```bash
python -m gnn.mcp.cli --format json status
```
#### JSON stdout contract (`--format json`)

Place `--format json` before the subcommand, for example
`python -m gnn.mcp.cli --format json status`. Finite commands use this
machine output format. A STDIO server streams protocol messages.
Routine log chatter is suppressed from stdout in this mode; `--verbose` is a
human-mode concern. On failure, the CLI prints a single error document to
stdout and exits with a nonzero code:

```json
{"error": {"operation": "getting tool info", "message": "Tool 'x' not found"}}
```

Note the sibling contract: the `gnn mcp` subcommand of the main `gnn` CLI
emits the standard CLI envelope (via its `--json` flag) instead of this raw
machine payload.

## API Reference

### Core MCP Class

#### Initialization
```python
from gnn.mcp import MCP

# Create MCP instance
mcp_instance = MCP()

# Initialize with module discovery
mcp_instance.discover_modules()
```

#### Tool Registration
```python
def my_tool(param1: str, param2: int) -> Dict[str, Any]:
    return {"result": f"{param1}_{param2}", "success": True}


schema = {
    "type": "object",
    "properties": {
        "param1": {"type": "string", "minLength": 1},
        "param2": {"type": "integer", "minimum": 0},
    },
    "required": ["param1", "param2"],
}

mcp_instance.register_tool(
    name="my_tool",
    func=my_tool,
    schema=schema,
    description="My custom tool",
    module="my_module",
    category="custom",
    version="1.0.0",
    tags=["custom", "example"],
    examples=[{"param1": "hello", "param2": 42}],
    timeout=30.0,
    max_concurrent=5,
    rate_limit=10.0,
    cache_ttl=300.0,
    input_validation=True,
    output_validation=True,
)
```

#### Resource Registration
```python
def my_resource_retriever(uri: str) -> Dict[str, Any]:
    return {"content": f"Resource content for {uri}", "uri": uri}


mcp_instance.register_resource(
    uri_template="my://{resource_id}",
    retriever=my_resource_retriever,
    description="My custom resource",
    module="my_module",
    category="custom",
    version="1.0.0",
    mime_type="application/json",
    cacheable=True,
    tags=["custom", "resource"],
    timeout=30.0,
    rate_limit=10.0,
    cache_ttl=300.0,
)
```

#### Tool Execution
```python
# Execute a tool
result = mcp_instance.execute_tool("my_tool", {"param1": "hello", "param2": 42})

# Get tool performance statistics
stats = mcp_instance.get_tool_performance_stats("my_tool")
```

#### Resource Access
```python
# Get a resource
result = mcp_instance.get_resource("my://my_resource")
```

#### Server Status and Monitoring
```python
# Get enhanced server status
status = mcp_instance.get_enhanced_server_status()

# Get basic server status
basic_status = mcp_instance.get_server_status()

# Get capabilities
capabilities = mcp_instance.get_capabilities()

# Clear caches
cache_stats = mcp_instance.clear_cache()
```

### Parameter Validation

Tool schemas support JSON-Schema-style type / range / format constraints and
object composition. Validation runs before the tool handler is invoked:

```python
# Complex schema with validation
schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 3,
            "maxLength": 50,
            "pattern": "^[a-zA-Z0-9_]+$",
        },
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {"type": "string"},
        },
        "settings": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "timeout": {"type": "number", "minimum": 0},
            },
            "required": ["enabled"],
        },
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}
```

### Caching System

Result caching is opt-in per tool and gated by the server-wide toggle:

- `enable_caching` (server) must be on (performance_mode "high", or an
  explicit `initialize(enable_caching=True)`).
- The tool must be registered with `cacheable=True` (default is `False`:
  mutating or side-effectful tools are never cached unless they opt in).
- The cache entry TTL is the tool's own `cache_ttl` when it registered one,
  otherwise the server-wide `cache_ttl` default (`300.0` s, settable via
  `initialize(cache_ttl=...)`).

```python
# Register a tool that opts into result caching
mcp_instance.register_tool(
    name="expensive_operation",
    func=expensive_function,
    schema=schema,
    description="An expensive operation with caching",
    cacheable=True,       # opt in to the result cache
    cache_ttl=3600.0,     # per-tool TTL overrides the server default
)

# Execute (executes the handler and stores the result)
result1 = mcp_instance.execute_tool("expensive_operation", {"param": "value"})

# Execute again (same params served from the result cache until the entry expires)
result2 = mcp_instance.execute_tool("expensive_operation", {"param": "value"})

# Clear cache
cache_stats = mcp_instance.clear_cache()
```

### Tool Timeouts

Each tool may register its own `timeout` (seconds). A server-wide default for
tools that did not register one is set via `initialize(tool_timeout=...)`;
the default `None` means un-registered tools run without a time bound. When
an effective timeout applies, the caller stops waiting at that bound and
receives `MCPToolTimeoutError` (wire code -32008).

```python
initialize(tool_timeout=30.0)  # un-registered tools stop waiting at 30 s
```

### Rate Limiting

Configure rate limiting per tool:

```python
# Register tool with rate limiting
mcp_instance.register_tool(
    name="rate_limited_tool",
    func=my_function,
    schema=schema,
    description="A rate-limited tool",
    rate_limit=5.0,  # 5 requests per second
)

# Execute (will be rate limited if exceeded)
result = mcp_instance.execute_tool("rate_limited_tool", {})
```

### Concurrent Execution Control

Limit concurrent executions:

```python
# Register tool with concurrent limits
mcp_instance.register_tool(
    name="concurrent_limited_tool",
    func=my_function,
    schema=schema,
    description="A tool with concurrent execution limits",
    max_concurrent=3,  # Only 3 concurrent executions allowed
)
```

## Transport Layers

### stdio Server

The stdio server provides local process communication:

```python
from gnn.mcp.server_stdio import StdioServer

# Create and start stdio server
server = StdioServer()
server.start()
```

**Usage:**
```bash
python -m gnn.mcp.server_stdio
```

### HTTP Server

The HTTP server provides authenticated JSON-RPC access. Keep it bound to
`127.0.0.1` unless a trusted reverse proxy provides TLS and access control.
Tool execution and resource reads are guarded by separate allowlists:
`GNN_MCP_SAFE_RESOURCES` must contain each URI that `mcp.resource.get` may
return over HTTP.
Rate limiting is checked before bearer authentication, so repeated missing or
invalid tokens are throttled by `GNN_MCP_RATE_LIMIT_PER_MINUTE`.

```python
from gnn.mcp.server_http import MCPHTTPHandler
from http.server import HTTPServer

# Create HTTP server
server = HTTPServer(("localhost", 8080), MCPHTTPHandler)
server.serve_forever()
```

**Usage:**
```bash
GNN_MCP_TOKEN=local-dev-token python -m gnn.mcp.cli server --transport http --host 127.0.0.1 --port 8080
```

## JSON-RPC API

### Standard MCP Methods

#### Get Capabilities
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "mcp.capabilities",
  "params": {}
}
```

#### Execute Tool
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "mcp.tool.execute",
  "params": {
    "name": "my_tool",
    "params": {
      "param1": "hello",
      "param2": 42
    }
  }
}
```

#### Get Resource
HTTP transport rejects resource reads unless the URI is explicitly allowlisted
through `GNN_MCP_SAFE_RESOURCES`; stdio/direct process use is unaffected.

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "mcp.resource.get",
  "params": {
    "uri": "my://my_resource"
  }
}
```

### Direct Tool Invocation

Tools can also be invoked directly:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "my_tool",
  "params": {
    "param1": "hello",
    "param2": 42
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Parameter 'param1' must be a string",
    "data": {
      "field": "param1",
      "value": 123,
      "tool_name": "my_tool",
      "module_name": "my_module"
    }
  }
}
```

### Error Codes

- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
- `-32000`: MCP-specific errors
- `-32001`: SDK not found
- `-32002`: Resource retrieval errors
- `-32003`: Module loading errors
- `-32004`: Performance errors

## Performance Monitoring

### Performance Metrics

Each tool dispatch records latency, cache hit/miss, error counts, and concurrency
snapshot. `get_enhanced_server_status()` returns the aggregated view:

```python
# Get performance metrics
status = mcp_instance.get_enhanced_server_status()
performance = status["performance"]

# Available metrics
print(f"Total requests: {performance['total_requests']}")
print(f"Success rate: {performance['success_rate']:.2%}")
print(f"Average execution time: {performance['average_execution_time']:.3f}s")
print(f"Cache hit ratio: {performance['cache_hit_ratio']:.2%}")
print(f"Concurrent requests: {performance['concurrent_requests']}")
```

### Health Monitoring

Real-time health status:

```python
# Get health status
status = mcp_instance.get_enhanced_server_status()
health = status["health"]

print(f"Status: {health['status']}")  # healthy, degraded, error
print(f"Error rate: {health['error_rate']:.2%}")
print(f"Cache efficiency: {health['cache_efficiency']:.2%}")
print(f"Concurrent load: {health['concurrent_load']:.2%}")
```

### Tool Performance Statistics

Get detailed statistics for specific tools:

```python
# Get tool performance stats
stats = mcp_instance.get_tool_performance_stats("my_tool")

if stats:
    print(f"Execution count: {stats['execution_count']}")
    print(f"Average execution time: {stats['average_execution_time']:.3f}s")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Error count: {stats['error_count']}")
```

## Module Development

### Creating MCP Modules

Create a new MCP module by adding an `mcp.py` file to your module directory:

```python
# src/my_module/mcp.py


def register_tools(mcp_instance):
    """Register tools for this module."""

    # Define tool function
    def my_tool(param: str) -> Dict[str, Any]:
        return {"result": f"Processed: {param}"}

    # Define schema
    schema = {
        "type": "object",
        "properties": {"param": {"type": "string", "minLength": 1}},
        "required": ["param"],
    }

    # Register tool
    mcp_instance.register_tool(
        name="my_tool",
        func=my_tool,
        schema=schema,
        description="My custom tool",
        module="my_module",
        category="custom",
        version="1.0.0",
        tags=["custom"],
        examples=[{"param": "example"}],
        timeout=30.0,
        rate_limit=10.0,
        cache_ttl=300.0,
    )

    # Register resource
    def my_resource_retriever(uri: str) -> Dict[str, Any]:
        return {"content": f"Resource: {uri}", "uri": uri}

    mcp_instance.register_resource(
        uri_template="my://{resource_id}",
        retriever=my_resource_retriever,
        description="My custom resource",
        module="my_module",
        category="custom",
        version="1.0.0",
        cacheable=True,
    )


# Module metadata
__version__ = "1.0.0"
__description__ = "My custom MCP module"
__dependencies__ = ["some_dependency"]
```

### Module Discovery

Modules are automatically discovered during initialization:

```python
# Discover modules
success = mcp_instance.discover_modules()

# Get module information
module_info = mcp_instance.get_module_info("my_module")
```

## Testing

### Running Tests

```bash
# All MCP tests
uv run --extra dev python -m pytest tests/mcp/test_mcp_overall.py tests/mcp/test_mcp_functional.py \
       tests/mcp/test_mcp_tools.py tests/mcp/test_mcp_performance.py \
       tests/mcp/test_mcp_audit.py -v

# Narrower slices
uv run --extra dev python -m pytest tests/mcp/test_mcp_tools.py -v          # registration + dispatch
uv run --extra dev python -m pytest tests/mcp/test_mcp_performance.py -v    # load / latency
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py -v          # spec / schema audit
```

### Test Categories

- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Performance and load testing
- **Error Handling Tests**: Error scenarios and edge cases

## Troubleshooting

### Common Issues

#### Module Loading Failures
```python
# Check module status
status = mcp_instance.get_enhanced_server_status()
modules = status["modules"]

for module_name, module_info in modules.items():
    if module_info["status"] != "loaded":
        print(f"Module {module_name}: {module_info['status']}")
        if module_info.get("error_message"):
            print(f"  Error: {module_info['error_message']}")
```

#### Performance Issues
```python
# Check performance metrics
status = mcp_instance.get_enhanced_server_status()
performance = status["performance"]

if performance["success_rate"] < 0.9:
    print("Low success rate detected")
    
if performance["average_execution_time"] > 5.0:
    print("High execution times detected")
    
if performance["cache_hit_ratio"] < 0.5:
    print("Low cache efficiency")
```

#### Memory Issues
```python
# Check memory usage
status = mcp_instance.get_enhanced_server_status()
resources = status["resources"]

if resources["cache_size"] > 1000:
    print("Large cache detected, consider clearing")
    mcp_instance.clear_cache()
```

### Debugging

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp")
```

### Performance Optimization

#### Caching Strategy
```python
# Use caching for expensive operations
mcp_instance.register_tool(
    name="expensive_operation",
    func=expensive_function,
    schema=schema,
    cache_ttl=3600.0,  # Cache for 1 hour
)
```

#### Rate Limiting
```python
# Prevent abuse with rate limiting
mcp_instance.register_tool(
    name="api_call",
    func=api_function,
    schema=schema,
    rate_limit=10.0,  # 10 requests per second
)
```

#### Concurrent Limits
```python
# Limit resource usage
mcp_instance.register_tool(
    name="resource_intensive",
    func=intensive_function,
    schema=schema,
    max_concurrent=2,  # Only 2 concurrent executions
)
```

## Security Considerations

### Authentication
```python
# Tools requiring authentication
mcp_instance.register_tool(
    name="sensitive_operation",
    func=sensitive_function,
    schema=schema,
    requires_auth=True,
)
```

### Input Validation
```python
# Always enable input validation
mcp_instance.register_tool(
    name="safe_tool",
    func=safe_function,
    schema=schema,
    input_validation=True,
    output_validation=True,
)
```

### Rate Limiting
```python
# Prevent abuse
mcp_instance.register_tool(
    name="public_api",
    func=public_function,
    schema=schema,
    rate_limit=5.0,  # 5 requests per second
)
```

## Best Practices

### Tool Design
1. **Clear Naming**: Use descriptive, consistent tool names
2. **Comprehensive Schemas**: Define detailed parameter schemas
3. **Error Handling**: Provide meaningful error messages
4. **Documentation**: Include clear descriptions and examples
5. **Validation**: Enable input and output validation

### Performance
1. **Caching**: Use caching for expensive operations
2. **Rate Limiting**: Implement appropriate rate limits
3. **Concurrent Limits**: Set reasonable concurrent execution limits
4. **Monitoring**: Monitor performance metrics regularly
5. **Optimization**: Profile and optimize slow operations

### Security
1. **Authentication**: Require authentication for sensitive operations
2. **Validation**: Always validate inputs and outputs
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **Logging**: Log security-relevant events
5. **Updates**: Keep dependencies updated

### Module Development
1. **Structure**: Follow the standard module structure
2. **Metadata**: Provide complete module metadata
3. **Error Handling**: Handle errors gracefully
4. **Testing**: Cover happy path, error path, and schema rejection
5. **Documentation**: Document all tools and resources

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints throughout
- Include docstrings with parameters, returns, and at least one example
- Write clear, readable code

### Testing
- Write unit tests for all new functionality
- Include integration tests for complex workflows
- Test error scenarios and edge cases
- Maintain good test coverage

## Summary

The MCP subsystem exposes every pipeline module as JSON-RPC tools discovered from
`src/*/mcp.py`. Wrappers around dispatch add per-tool caching, concurrency caps,
rate limiting, and metrics. `get_enhanced_server_status()` is the canonical
snapshot for debugging live servers.

For per-module tool inventories, see the `mcp.py` files in each module and the
main documentation in `docs/mcp/`.
