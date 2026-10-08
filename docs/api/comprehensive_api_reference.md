# GNN Comprehensive API Reference

> **📋 Document Metadata** \
> **Type**: API Reference | **Audience**: Developers & Integrators | **Complexity**: Intermediate-Advanced \
> **Cross-References**: [Pipeline Architecture](../gnn/operations/gnn_tools.md) | [Framework Integration](../gnn/integration/framework_integration_guide.md)

This comprehensive reference documents programmatic integration with the GeneralizedNotationNotation (GNN) system.

> **Verification status** \
> **Authoritative `gnn` exports:** [`src/gnn/__init__.py`](../../src/gnn/__init__.py). **Format counts / registries:** [`src/gnn/SPEC.md`](../../src/gnn/SPEC.md). \
> The September 2026 import probes are historical evidence. Current public exports, module specifications and exact-source consumer checks define accepted interfaces. Install each declared optional group before using its surface; importing a class does not establish native backend readiness. The [API guide](README.md) includes exercised discovery/parsing and CLI examples.

## API map

1. **Package `gnn` (Step 3)** — file discovery, parsing, validation, multi-format serialization under [`src/gnn/`](../../src/gnn/).
2. **Pipeline CLI** — `uv run python src/gnn/main.py`, numbered `src/gnn/N_*.py` scripts.
3. **Render / execute / viz / LLM / MCP** — under `src/gnn/render/`, `src/gnn/execute/`, `src/gnn/visualization/`, `src/gnn/llm/`, `src/gnn/mcp/`; each module's `AGENTS.md` describes its surface (see [`src/gnn/AGENTS.md`](../../src/gnn/AGENTS.md)).

### Quick start (current `gnn` exports)

```python
import logging
from pathlib import Path
from gnn import (
    discover_gnn_files,
    parse_gnn_file,
    process_gnn_directory,
    process_gnn_multi_format,
    validate_gnn_syntax,
    GNNParsingSystem,
    GNNFormat,
)

paths = discover_gnn_files(Path("input/gnn_files"))
info = parse_gnn_file(paths[0])

system = GNNParsingSystem()
result = system.parse_file(paths[0], format_hint=GNNFormat.MARKDOWN)

process_gnn_directory("input/gnn_files/basics", "/tmp/gnn-directory-example")

logger = logging.getLogger(__name__)
process_gnn_multi_format(Path("input/gnn_files/basics"), Path("/tmp/gnn-format-example"), logger)

ok, errors = validate_gnn_syntax(Path("input/gnn_files/discrete/two_state_bistable.md").read_text(encoding="utf-8"))
```

| Symbol | Role |
|--------|------|
| `GNNParsingSystem` | Registry-backed parse/serialize (`parsers/system.py`) |
| `GNNFormat` | Supported formats ([`SPEC.md`](../../src/gnn/SPEC.md)) |
| `discover_gnn_files`, `parse_gnn_file`, `process_gnn_directory` | Discovery and lightweight processing |
| `process_gnn_multi_format` | Step 3 multi-format output (**requires** `logging.Logger`) |
| `validate_gnn_syntax`, `validate_gnn_source` | Validation entry points |
| `schema_validator.GNNParser` | Section-level parser for strict validation (submodule import) |

## Parsing API (Step 3)

### **📄 GNNParsingSystem**

Registry-backed parse/serialize entry point ([`src/gnn/parsers/system.py`](../../src/gnn/parsers/system.py)).

```python
from gnn import GNNParsingSystem, GNNFormat

system = GNNParsingSystem()

# Supported input/serialization formats
print(system.get_supported_formats())

# Parse a file (format_hint optional — auto-detection otherwise)
result = system.parse_file("input/gnn_files/discrete/two_state_bistable.md", format_hint=GNNFormat.MARKDOWN)
if result.success:
    model = result.model
    print(result.source_file, result.parse_time)
```

Methods: `parse_file(file_path, format_hint=None)`, `parse_string`, `convert`, `convert_file`, `serialize`, `serialize_to_file`, `get_supported_formats`, `get_available_parsers`, `get_available_serializers`.

`parse_file` returns a `ParseResult` with fields `model`, `success`, `errors`, `warnings`, `parse_time`, `source_file`, `validation_result`, and `metadata`.

### **✅ Validation**

```python
from pathlib import Path
from gnn import validate_gnn_syntax, validate_gnn_source
from gnn.types import ValidationLevel

# Accepts a file path or raw content
ok, errors = validate_gnn_syntax(Path("input/gnn_files/discrete/two_state_bistable.md").read_text(encoding="utf-8"))
ok, errors = validate_gnn_syntax("input/gnn_files/discrete/two_state_bistable.md", validation_level=ValidationLevel.STANDARD)
```

- `validate_gnn_syntax(file_path_or_content, validation_level=ValidationLevel.STANDARD) -> (bool, List[str])`
  — delegates to the formal `schema_validator` pipeline
  (`GNNValidator.validate_file`), so returned messages are the formal
  validator's errors (e.g. `"Required section missing: ModelName"`).
  An existing file path and the same bytes passed as a content string
  produce identical verdicts (both are staged and validated through the
  same pipeline). Content is validated as GNN markdown text — for
  JSON/XML/YAML model files use `validate_gnn_file_comprehensive`,
  which honors the file extension. `validation_level` also accepts level
  strings (`"strict"`, `"STRICT"`) and `None` (the validator's default
  level, STANDARD); unknown level strings raise `ValueError`. Long
  inputs are never truncated and validate in linear time (the previous
  implementation crashed on long content strings with `ENAMETOOLONG`
  from its path probe and reported a bogus error tuple; such input now
  selects content mode). Validity is the formal validator's normative
  markdown gate — every required section with substantive body content
  — a deliberate tightening over the earlier regex heuristic; the formal
  level ladder applies at every level (BASIC included).
- `validate_gnn_source(source, *, is_content=False)`
- Section-level parser for strict validation: `from gnn.schema_validator import GNNParser` (`GNNParser(enhanced_validation=True)`).

## Pipeline API

### **⚙️ Configuration**

```python
from gnn.pipeline import PipelineConfig
from gnn.pipeline.config import (
    get_pipeline_config,
    get_pipeline_config_dict,
    get_output_dir_for_script,
)

config = PipelineConfig()  # defaults; accepts an optional config_path
```

- `PipelineConfig(config_path=None)` — pipeline-level configuration ([`src/gnn/pipeline/config.py`](../../src/gnn/pipeline/config.py)).
- `get_pipeline_config()`, `get_pipeline_config_dict()`, `get_output_dir_for_script(...)` — read/write the process-wide pipeline configuration and per-step output directories.

### **🚀 PipelineOrchestrator**

Programmatic execution of pipeline steps.

```python
from gnn.pipeline import PipelineOrchestrator, run_pipeline

orchestrator = PipelineOrchestrator(
    target_dir="input/gnn_files",
    output_dir="output",
    steps=[3, 5, 8, 11, 12],  # subset of steps 0-24
    verbose=True,
)
orchestrator.run()  # also: execute_pipeline(...), get_pipeline_steps()
```

Functional form:

```python
summary = run_pipeline(target_dir="input/gnn_files", output_dir="output", steps="all")
```

CLI equivalent (numbered scripts are thin orchestrators, e.g. `src/gnn/3_gnn.py`):

```bash
uv run python src/gnn/main.py --target-dir input/gnn_files --only-steps "3,5,8,11,12"
```

## Framework rendering API (Step 11)

### **🧩 Multi-framework renderers**

```python
from pathlib import Path
from gnn.render import (
    render_gnn_to_pymdp,
    render_gnn_to_rxinfer,
    render_gnn_to_discopy,
    render_gnn_spec,
)

ok, message, artifacts = render_gnn_to_pymdp(gnn_spec, Path("output/11_render_output/pymdp/model.py"))
ok, message, artifacts = render_gnn_to_rxinfer(gnn_spec, Path("output/11_render_output/rxinfer/model.jl"))
ok, message, artifacts = render_gnn_spec(gnn_spec, target="pymdp", output_directory="output/11_render_output")
```

- Every `render_gnn_to_*(gnn_spec, output_path, options=None)` returns `Tuple[bool, str, List[str]]` (success, message, written artifacts).
- Frameworks: `render_gnn_to_pymdp`, `render_gnn_to_rxinfer`, `render_gnn_to_discopy`, `render_gnn_to_numpyro`, `render_gnn_to_pytorch`, `render_gnn_to_activeinference_jl`, `render_stan`; code-generator helpers `generate_pymdp_code`, `generate_rxinfer_code`, `generate_discopy_code`, `generate_activeinference_jl_code`.
- `render_gnn_spec(gnn_spec, target, output_directory, options=None)` dispatches on `target` and accepts either a parsed model or a spec dict.
- Class-based renderers: `PyMDPRenderer(options=None)`, `JAXRenderer(options=None)`, and the `POMDPRenderProcessor` pipeline processor; `process_render` / `process_pomdp_for_frameworks` are the Step 11 entry points.
- Backend support matrix: [`src/gnn/render/framework_registry.py`](../../src/gnn/render/framework_registry.py) (per-framework support flags, including continuous-model support); `get_supported_frameworks()` and `get_available_renderers()` summarize it.

### **🐍 PyMDP**

```python
from pathlib import Path
from gnn.render import render_gnn_to_pymdp

ok, message, artifacts = render_gnn_to_pymdp(
    gnn_spec,
    Path("output/11_render_output/pymdp/agent.py"),
)
```

### **🔢 RxInfer.jl**

```python
from pathlib import Path
from gnn.render import render_gnn_to_rxinfer

ok, message, artifacts = render_gnn_to_rxinfer(
    gnn_spec,
    Path("output/11_render_output/rxinfer/agent.jl"),
)
```

The RxInfer strategies live under [`src/gnn/render/rxinfer/`](../../src/gnn/render/rxinfer/) (`_strategies*.py`, exposed as `gnn.render.rxinfer.model_strategies`).

### **🎨 DisCoPy**

```python
from pathlib import Path
from gnn.render import render_gnn_to_discopy

ok, message, artifacts = render_gnn_to_discopy(
    gnn_spec,
    Path("output/11_render_output/discopy/agent.py"),
)
```

## Execution API (Step 12)

### **🚀 Execute rendered simulations**

```python
from gnn.execute import GNNExecutor, execute_gnn_model

result = execute_gnn_model(
    "output/11_render_output/pymdp/agent.py",
    execution_type="pymdp",
)
```

- `execute_gnn_model(model_path, execution_type="pymdp", options=None) -> Dict[str, Any]`.
- `GNNExecutor(output_dir=None, cache=None)` — executor object with result caching; `clear_execution_cache()` resets it.
- Per-framework executors live under [`src/gnn/execute/`](../../src/gnn/execute/) (`pymdp`, `rxinfer`, `jax`, `pytorch`, `numpyro`, `stan`, `discopy`, `activeinference_jl`, `ngclearn`, `lean`, `bnlearn`); `execute_simulation_from_gnn` and `execute_pymdp_simulation` are the targeted entry points.
- Diagnostics: `check_dependencies()`, `collect_doctor_report()`.

## Visualization API (Step 8)

### **🎨 Visualizers**

```python
from pathlib import Path
from gnn.visualization import (
    GNNVisualizer,
    generate_matrix_visualizations,
    generate_network_visualizations,
)

viz = GNNVisualizer(output_dir="output/8_visualization_output")
diagram = viz.create_network_diagram(graph_data)   # graph structure -> figure data
html = viz.visualize_file("input/gnn_files/discrete/two_state_bistable.md")  # full per-model visualization set

artifacts = generate_matrix_visualizations(parsed_data, Path("output/8_visualization_output"), "model")
artifacts = generate_network_visualizations(parsed_data, Path("output/8_visualization_output"), "model")
```

- `GNNVisualizer(output_dir=None, project_root=None)`; methods include `create_network_diagram(graph_data=None)` and `visualize_file(file_path) -> str`.
- `MatrixVisualizer`, `OntologyVisualizer`, and `gnn.visualization.backends` for backend selection.
- Batch entry point: `generate_visualizations(logger, target_dir, output_dir, recursive=False, verbose=False) -> bool` (**requires** `logging.Logger`).

## LLM API (Step 13)

### **🧠 LLM analysis**

```python
from gnn.llm import (
    LLMAnalyzer,
    LLMConfig,
    analyze_gnn_model,
    DEFAULT_OLLAMA_MODEL,
    get_available_providers,
)

print(DEFAULT_OLLAMA_MODEL)  # "smollm2:135m-instruct-q4_K_S" (Ollama)

config = LLMConfig(model=DEFAULT_OLLAMA_MODEL)
analyzer = LLMAnalyzer()
insights = analyzer.analyze_content(content)
summary = analyze_gnn_model(content)
```

- `LLMConfig(model=None, max_tokens=None, temperature=None, ...)` — generation parameters.
- `LLMAnalyzer` exposes `analyze_content` and `extract_insights`; the module-level `analyze_gnn_model(model_content) -> Dict[str, Any]` and `analyze_gnn_file_with_llm` wrap model analysis.
- Processors: `LLMProcessor`, `UnifiedLLMProcessor`; `create_processor_from_env` / `initialize_global_processor` configure processors from the environment.
- `get_available_providers()` lists provider backends.

## MCP API (Step 21)

### **🔌 MCP tools**

```python
from gnn.mcp import MCPServer

server = MCPServer()
server.register_tool(
    name="parse_gnn_model",
    func=parse_gnn_model,  # your callable
    schema={
        "type": "object",
        "properties": {"filepath": {"type": "string"}},
        "required": ["filepath"],
    },
    description="Parse a GNN model file",
)
server.start()  # -> bool; server.stop() to shut down
```

- `MCPServer.register_tool(name, func, schema, description) -> bool`; `MCPServer(mcp_instance=None, capabilities_getter=None)`.
- Tool container: `MCPTool(name, func, schema, description, ...)`; registry: `MCPRegistry`.
- Convenience: `create_mcp_server() -> MCPServer`, `get_mcp_instance()`, `handle_mcp_request(request)`; tool discovery via `list_available_tools()`, `get_available_tools()`.
- HTTP and stdio server entry points live under [`src/gnn/mcp/`](../../src/gnn/mcp/AGENTS.md).

## Complete integration example

```python
"""Parse -> validate -> run steps -> render -> visualize -> LLM-summarize."""
from pathlib import Path

from gnn import GNNParsingSystem, discover_gnn_files, validate_gnn_syntax
from gnn.llm import analyze_gnn_model
from gnn.pipeline import PipelineOrchestrator
from gnn.render import render_gnn_spec
from gnn.visualization import GNNVisualizer

# 1. Discover and validate input models
paths = discover_gnn_files(Path("input/gnn_files"))
ok, errors = validate_gnn_syntax(paths[0])

# 2. Run pipeline steps programmatically
#    (CLI equivalent: uv run python src/gnn/main.py --target-dir input/gnn_files --only-steps "3,5,8,11,12")
orchestrator = PipelineOrchestrator(target_dir="input/gnn_files", output_dir="output", steps=[3, 5, 8, 11, 12])
orchestrator.run()

# 3. Render a parsed model to a framework target.
#    Render targets require the model's InitialParameterization section to define A/B/C/D.
model_path = Path("input/gnn_files/discrete/actinf_pomdp_agent.md")
spec = GNNParsingSystem().parse_file(model_path).model
ok, message, artifacts = render_gnn_spec(spec, target="pymdp", output_directory="output/11_render_output")

# 4. Visualize the model
viz = GNNVisualizer(output_dir="output/8_visualization_output")
viz.visualize_file(str(model_path))

# 5. Summarize with the default local LLM
summary = analyze_gnn_model(model_path.read_text(encoding="utf-8"))
print(summary)
```

---

**🔌 API Integration**: These surfaces mirror the pipeline's 25 steps — use the CLI for end-to-end runs and the APIs above to embed GNN processing in your own workflows.

---

**Status**: ✅ Production Ready \
**Compliance**: Professional documentation standards \
**Maintenance**: Regular updates with new API features and capabilities

- **Start Here**: [Overview](../../README.md)
- **Examples**: [Model Examples](../../docs/gnn/tutorials/gnn_examples_doc.md)
- **Development**: [Contribution Guide](../../CONTRIBUTING.md)
