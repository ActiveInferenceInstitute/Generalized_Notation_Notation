# Architecture Tests

## Overview

Dependency-direction and import-surface contracts for the `gnn` package.

- **Scope**: whole-tree static import edges (AST, including function-scope,
  relative, `TYPE_CHECKING` and literal `importlib.import_module` imports) and
  fresh-subprocess `sys.modules` checks.
- **Source under test**: `src/gnn/api`, `src/gnn/cli`, `src/gnn/operations`,
  `src/gnn/pipeline`, `src/gnn/report`, `src/gnn/template`.

## Rules

- Do not weaken a direction contract to admit a new edge; move the shared code
  below both packages (see `gnn.operations`) or keep a renderer with its
  producer and re-export it from the consumer.
- Import probes must run in a new `sys.executable` interpreter; the pytest
  process has already imported most of the tree.
- Keep the scanner self-checks: they prove the AST walk still finds relative,
  function-scope and dynamic imports.
