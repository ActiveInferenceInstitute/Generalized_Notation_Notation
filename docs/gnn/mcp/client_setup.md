# GNN MCP Client Setup Guide

Connect a local MCP client to GNN's registered tools over STDIO.

## Prerequisites

From the repository root, provision the environment and inspect the finite tool list.

```bash
uv sync --frozen --python 3.14
uv run --frozen --no-sync python -m gnn.mcp.cli list
```

Replace `/path/to/Generalized_Notation_Notation` with the absolute path to your clone.
The examples use its installed environment. On Windows, use
`.venv/Scripts/python.exe` instead of `.venv/bin/python`.
The transport process is `gnn.mcp.server_stdio`. Pipeline Step 21 runs
discovery and writes reports, then exits.

## Claude Desktop

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "gnn": {
      "command": "/path/to/Generalized_Notation_Notation/.venv/bin/python",
      "args": ["-m", "gnn.mcp.server_stdio"]
    }
  }
}
```

Restart Claude Desktop and check its MCP server list.
See the [MCP guide to local servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

## VS Code

For a workspace using the VS Code Local harness, add this entry to `.vscode/mcp.json`.

```json
{
  "servers": {
    "gnn": {
      "type": "stdio",
      "command": "/path/to/Generalized_Notation_Notation/.venv/bin/python",
      "args": ["-m", "gnn.mcp.server_stdio"]
    }
  }
}
```

Follow the [VS Code MCP guide](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
to select the configuration destination and start the server.

## Cursor

Add the same `mcpServers` object from the Claude Desktop example to the
project's `.cursor/mcp.json` and start the server from Cursor's MCP settings.

## Using uv as the launcher

In a client that uses `mcpServers`, this configuration selects the checkout and
uses its provisioned environment. Set `command` to the absolute path to uv if
the desktop client's PATH omits it.

```json
{
  "mcpServers": {
    "gnn": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/Generalized_Notation_Notation",
        "--frozen", "--no-sync", "python", "-m", "gnn.mcp.server_stdio"
      ]
    }
  }
}
```

## Verifying the connection

Use the client to list tools or call `get_pipeline_status`. To try
`parse_gnn_content`, give it model text in `content`. For tools that take a
local path, use an absolute path so resolution does not depend on the client.

To verify with a Python script, run the following in a separate client environment
with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) installed.

```python
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command="/path/to/Generalized_Notation_Notation/.venv/bin/python",
        args=["-m", "gnn.mcp.server_stdio"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Connected: {len(tools.tools)} tools available")


asyncio.run(main())
```

## Available tools by use case

| Task | Tool or command |
|---|---|
| Parse GNN model text | `parse_gnn_content` |
| Validate a GNN file | `validate_gnn_file` |
| Run the full pipeline from the repository root | `gnn run` |
| Execute with PyMDP | `execute_pymdp_simulation` |
| Ask an LLM to analyze GNN | `analyze_gnn_with_llm` |
| Generate audio from GNN | `process_audio` |
| Export to multiple formats | `process_export` |
| Scan a GNN file for security findings | `scan_gnn_file` |

## Troubleshooting

If startup fails, confirm the configured interpreter exists and inspect the
client's stderr logs. If discovery fails, run the finite tool-list command above
from the checkout to inspect registration. Native execution and LLM tools may
need extra runtimes or credentials and can take minutes.

## See also

- [Tool reference](tool_reference.md)
- [Tool development guide](tool_development_guide.md)
- [Step 21 discovery and reporting](../modules/21_mcp.md)
- [FastMCP guide](../../mcp/fastmcp.md)
