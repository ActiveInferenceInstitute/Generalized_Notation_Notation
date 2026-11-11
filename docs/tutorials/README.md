# GNN Tutorials

The maintained beginner tutorial is
[gnn/tutorials/quickstart_tutorial.md](../gnn/tutorials/quickstart_tutorial.md). It
creates a complete 2×2 navigation model, validates it, and demonstrates a focused
render/execute path.

## Recommended sequence

1. [GNN Overview](../gnn/gnn_overview.md)
2. [Quickstart Tutorial](../gnn/tutorials/quickstart_tutorial.md)
3. [Syntax Reference](../gnn/reference/gnn_syntax.md)
4. [GNN Examples](../gnn/tutorials/gnn_examples_doc.md)
5. [Templates](../templates/README.md)
6. [Framework Integration](../gnn/integration/framework_integration_guide.md)

## Runnable command reference

Run from the repository root:

```bash
# Install the development environment.
uv sync --extra dev

# Validate a single model file.
uv run gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# Parse and type-check a directory.
uv run python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5" \
  --verbose

# Render and execute selected frameworks.
uv run python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "11,12" \
  --frameworks "pymdp,jax" \
  --verbose

# Skip environment-dependent work.
uv run python src/gnn/main.py --skip-steps "2,13,15" --verbose
```

`--target-dir` receives a directory. GNN discovery uses Markdown files under that
directory; a single file path is not a directory substitute.

## Practice ideas

- Change the goal preference in the quickstart model and re-run Step 5.
- Add a second observation modality and compare the syntax against the examples.
- Render the same model to `pymdp` and `rxinfer` using the framework guide.
- Use `uv run gnn health` before selecting an optional backend.

For environment and configuration details, see [Setup](../SETUP.md) and
[Configuration](../configuration/README.md). For errors, see
[Troubleshooting](../troubleshooting/README.md).


## Documents in this folder

- [GNN Interactive Tutorial System (proposed)](interactive_tutorial_system.md)
