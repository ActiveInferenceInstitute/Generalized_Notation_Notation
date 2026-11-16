# GNN Language Server (LSP)

## Overview

Minimal Language Server Protocol implementation for GNN files. Provides real-time diagnostics and hover information in editors that support LSP (VS Code, Neovim, etc.).

The installed `gnn lsp` command works with an ordinary GNN installation. It
uses the pygls server when that library is available and otherwise starts
the bundled JSON-RPC server on standard input and output.

## Features

| Feature | LSP Method | Description |
|---------|-----------|-------------|
| **Diagnostics on open** | `textDocument/didOpen` | Validates sections, state-space, and connections |
| **Diagnostics on edit** | `textDocument/didChange` | Re-publishes diagnostics as the document changes |
| **Diagnostics on save** | `textDocument/didSave` | Re-validates after changes |
| **Hover info** | `textDocument/hover` | Shows variable dimensions and type on hover |

## Usage

Use `gnn lsp` as the editor's language-server command. To use the pygls
implementation in a source checkout, install the frozen development environment
and launch the same command from it.

```bash
uv sync --frozen
uv run --frozen --no-sync gnn lsp
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "gnn-lsp.server.path": "gnn",
  "gnn-lsp.server.args": ["lsp"]
}
```

## Architecture

- **Server**: Built on `pygls` (Python Language Server framework)
- **Validation**: Delegates to `gnn.schema` (same validation as `gnn validate`)
- **Hover**: Parses state-space to show variable metadata at cursor position
- **Fallback**: Graceful degradation when `pygls` is not installed

The `gnn lsp` CLI subcommand launches the canonical pygls server from
`gnn.lsp` when pygls is importable; the pygls-free JSON-RPC server in
`src/gnn/cli/lsp.py` is the fallback when pygls is not installed.

## File Structure

```
src/gnn/lsp/
├── __init__.py    # Server implementation
├── mcp.py         # Model Context Protocol integration
├── AGENTS.md      # Agent documentation
├── README.md      # This file
├── SKILL.md       # Capability API
└── SPEC.md        # Module specification
```

## References

- [SPEC.md](SPEC.md) — Module specification
- [AGENTS.md](AGENTS.md) — Agent documentation
