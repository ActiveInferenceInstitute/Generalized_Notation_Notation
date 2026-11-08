# Model Context Protocol (MCP) Integration

> **📋 Document Metadata**  
> **Type**: Integration Guide | **Audience**: AI Developers & Integrators | **Complexity**: Advanced  
> **Cross-References**: [API Documentation](../api/README.md) | [FastMCP Guide](fastmcp.md) | [docs/SPEC.md](../SPEC.md) (versioning policy)

## Overview
The GNN project implements Model Context Protocol (MCP) to provide structured APIs for AI assistants and LLM integrations. MCP enables external tools and AI systems to interact with GNN processing capabilities through standardized interfaces.

## Security

MCP servers expose tools over STDIO or HTTP: bind listeners to localhost in untrusted networks, authenticate HTTP deployments, and treat tool outputs like any sensitive pipeline data. See [security/README.md](../security/README.md).

## Architecture

### Core MCP System
- **Location**: `src/gnn/mcp/`
- **Registry**: `src/gnn/mcp/mcp.py` - Central MCP instance and tool registration
- **Server Components**: HTTP and STDIO server implementations
- **Tool Discovery**: Automatic registration from functional modules

### Distributed MCP Modules
Each functional module includes its own `mcp.py` file that registers domain-specific tools:

```
src/gnn/
├── mcp/               # Core MCP infrastructure (registry, transports)
├── export/mcp.py      # Export format tools
├── mcp/gnn_root.py    # Root GNN parsing, validation, and documentation tools
├── gui/mcp.py         # GUI tools (including oxdraw.* wrappers)
├── ontology/mcp.py    # Ontology processing tools
├── visualization/mcp.py # Visualization generation tools
└── llm/mcp.py         # LLM integration tools
```

## Available MCP Tools

### Core GNN Processing (`mcp/gnn_root.py`)
- `parse_gnn_content` - Parse GNN content into a structured model representation
- `validate_gnn_content` - Validate GNN content across supported validation levels
- `process_gnn_directory` - Process a directory of GNN files
- `get_gnn_documentation` - Retrieve maintained GNN format documentation

### Export Tools (`export/mcp.py`)
- `process_export` - Export GNN files from a directory
- `export_single_gnn_file` - Export one GNN file
- `list_export_formats` - Show available export formats
- `validate_export_format` - Check whether an export format is supported

### Visualization Tools (`visualization/mcp.py`)
- `process_visualization` - Run graph and matrix visualization
- `get_visualization_options` - List configurable visualization options
- `list_visualization_artifacts` - List generated visualization artifacts
- `get_visualization_module_info` - Return visualization module metadata

### Ontology Tools (`ontology/mcp.py`)
- `process_ontology` - Run ontology annotation processing
- `validate_ontology_terms` - Check Active Inference ontology compliance
- `extract_ontology_annotations` - Extract ontology annotations from GNN content
- `list_standard_ontology_terms` - List maintained ontology terms

### LLM Integration Tools (`llm/mcp.py`)
- `process_llm` - Run LLM analysis over GNN files
- `analyze_gnn_with_llm` - AI-powered GNN model analysis
- `generate_llm_documentation` - Natural-language documentation generation
- `get_llm_providers` - Report configured provider availability

### GUI and oxdraw Tools (`gui/mcp.py`)
- `process_gui` - Generate GUI artifacts
- `list_available_guis` - List GUI implementations
- `oxdraw.convert_to_mermaid` - Convert GNN to Mermaid for visual editing
- `oxdraw.convert_from_mermaid` - Convert Mermaid back to GNN
- `oxdraw.check_installation` - Check oxdraw CLI availability

## Using MCP Tools

### Direct Python Access
```python
from gnn.mcp import initialize, mcp_instance

initialize(halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True)
print(sorted(mcp_instance.tools))

result = mcp_instance.execute_tool(
    "parse_gnn_content",
    {
        "content": "## GNNSection\nActInfPOMDP\n",
        "format_hint": "markdown",
        "enhanced_validation": True,
    },
)
```

### HTTP Server
```bash
# Start MCP HTTP server
GNN_MCP_TOKEN=local-dev-token \
  uv run python -m gnn.mcp.cli server --transport http --host 127.0.0.1 --port 8080

# Execute an HTTP-safe tool through JSON-RPC
curl -X POST http://127.0.0.1:8080/ \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"status","method":"get_pipeline_status","params":{}}'
```

### STDIO Server (for AI assistants)
```bash
# Start STDIO server for AI assistant integration
uv run --frozen --no-sync python -m gnn.mcp.server_stdio
```

### Command Line Interface
```bash
# List capabilities and inspect a tool
uv run python -m gnn.mcp.cli list
uv run python -m gnn.mcp.cli info parse_gnn_content

# Execute a tool
uv run python -m gnn.mcp.cli execute parse_gnn_content \
  --params '{"content":"## GNNSection\nActInfPOMDP\n","format_hint":"markdown","enhanced_validation":true}'
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

## Tool Schema Examples

### GNN Content Parsing Tool
```json
{
  "name": "parse_gnn_content",
  "description": "Parse GNN content with enhanced multi-format support and return structured model representation.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "GNN file content to parse"
      },
      "format_hint": {
        "type": "string",
        "enum": ["markdown", "json", "xml", "yaml", "binary"],
        "default": "markdown"
      },
      "enhanced_validation": {
        "type": "boolean",
        "default": true
      }
    },
    "required": ["content"]
  }
}
```

### Render Tool
```json
{
  "name": "process_render",
  "description": "Render GNN models in a directory to all supported code frameworks.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "target_directory": {
        "type": "string",
        "description": "Directory containing GNN files"
      },
      "output_directory": {
        "type": "string",
        "description": "Directory to write rendered outputs"
      },
      "verbose": {
        "type": "boolean",
        "default": false
      }
    },
    "required": ["target_directory", "output_directory"]
  }
}
```

## Pipeline Integration

### Step 21: MCP Processing
The pipeline includes dedicated MCP analysis:

```bash
# Run MCP integration check
uv run python src/gnn/main.py --only-steps 21 --target-dir input/gnn_files --verbose

# Generate MCP integration report
uv run python src/gnn/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```

The MCP step generates comprehensive reports including:
- Available MCP tools across all modules
- Tool schemas and documentation
- Integration status and health checks
- API usage examples

## Development Guidelines

### Adding New MCP Tools

1. **Create tool function**:
```python
# In your module's mcp.py file
def my_new_tool(param1: str, param2: int = 10) -> dict:
    """
    Description of what this tool does.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
        
    Returns:
        Dictionary with results
    """
    # Implementation here
    return {"result": "success"}
```

2. **Register the tool**:
```python
def register_tools(mcp_instance):
    """Register all tools from this module."""
    mcp_instance.register_tool(
        "my_new_tool",
        my_new_tool,
        {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer", "default": 10},
            },
            "required": ["param1"],
        },
        "Run my new module action.",
        module=__package__,
        category="my_module",
    )
```

3. **Test the tool**:
```bash
uv run python -m gnn.mcp.cli execute my_new_tool --params '{"param1":"test"}'
```

### Tool Design Principles

- **Atomic Functions**: Each tool should do one thing well
- **Clear Schemas**: Provide complete input/output schemas
- **Error Handling**: Return structured error information
- **Documentation**: Include comprehensive docstrings
- **Type Hints**: Use proper type annotations for automatic schema generation

### Security Considerations

- **Input Validation**: Validate all inputs against schemas
- **File Access**: Restrict file operations to safe directories
- **API Keys**: Handle sensitive credentials securely
- **Resource Limits**: Implement timeouts and resource constraints

## Troubleshooting

### Common Issues

1. **Tool Not Found**
   - Check if module's `register_tools()` was called
   - Verify tool is in the module's MCP registration

2. **Schema Validation Errors**
   - Ensure input matches the tool's schema
   - Check required parameters are provided

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path includes src/ directory

### Debug Mode

```bash
# Run with verbose MCP logging
uv run python -m gnn.mcp.cli --verbose list

# Test tool with debugging
uv run python -m gnn.mcp.cli --verbose execute tool_name --params '{}'
```

## API Reference

### Core MCP Instance
```python
from gnn.mcp import mcp_instance

# Tool management
mcp_instance.register_tool(name, function)
mcp_instance.list_available_tools()
mcp_instance.execute_tool(name, parameters)

# Schema inspection
mcp_instance.get_tool_info(name)
```

### Server Components
- `server_http.py` - HTTP REST API server
- `server_stdio.py` - STDIO protocol server for AI assistants
- `cli.py` - Command-line interface

### Configuration
- Tool timeout settings
- Server port configuration
- API authentication (if enabled)
- Resource limits and constraints

## Integration Examples

### LLM Assistant Integration
```python
# Example of AI assistant using MCP tools
from pathlib import Path

from gnn.mcp import initialize, mcp_instance


def analyze_user_model(file_path: str) -> str:
    initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
    content = Path(file_path).read_text()
    parse_result = mcp_instance.execute_tool(
        "parse_gnn_content",
        {
            "content": content,
            "format_hint": "markdown",
            "enhanced_validation": True,
        },
    )
    return parse_result.get("summary", str(parse_result))
```

### External Tool Integration
```bash
curl -X POST http://127.0.0.1:8080/ \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"cap","method":"mcp.capabilities","params":{}}'
```
