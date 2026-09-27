# Pipeline Architecture

> **Environment**: Always use `uv` — `uv run python src/gnn/main.py`, `uv pip install -e .`

## 25-Step Pipeline (Steps 0–24)

| Step | Script | Module | Role |
|------|--------|--------|------|
| 0 | `0_template.py` | `template/` | Pipeline initialization |
| 1 | `1_setup.py` | `setup/` | Environment setup (**critical**) |
| 2 | `2_tests.py` | `tests/` | Full test suite |
| 3 | `3_gnn.py` | `gnn/` | GNN parsing, 21+ formats (**critical**) |
| 4 | `4_model_registry.py` | `model_registry/` | Model versioning |
| 5 | `5_type_checker.py` | `type_checker/` | Syntax/type validation |
| 6 | `6_validation.py` | `validation/` | Advanced validation |
| 7 | `7_export.py` | `export/` | Multi-format export |
| 8 | `8_visualization.py` | `visualization/` | Graph/matrix viz (**safe-to-fail**) |
| 9 | `9_advanced_viz.py` | `advanced_visualization/` | Interactive plots (**safe-to-fail**) |
| 10 | `10_ontology.py` | `ontology/` | Ontology processing |
| 11 | `11_render.py` | `render/` | Code generation (PyMDP/JAX/etc.) |
| 12 | `12_execute.py` | `execute/` | Simulation execution (**safe-to-fail**) |
| 13 | `13_llm.py` | `llm/` | LLM analysis |
| 14 | `14_ml_integration.py` | `ml_integration/` | ML integration |
| 15 | `15_audio.py` | `audio/` | Audio/sonification |
| 16 | `16_analysis.py` | `analysis/` | Statistical analysis |
| 17–19 | Integration, Security, Research | — | System coordination |
| 20 | `20_website.py` | `website/` | Static site generation |
| 21 | `21_mcp.py` | `mcp/` | MCP tool registration |
| 22 | `22_gui.py` | `gui/` | Interactive GUI |
| 23 | `23_report.py` | `report/` | Report generation |
| 24 | `24_intelligent_analysis.py` | `intelligent_analysis/` | AI pipeline analysis |

---

## Thin Orchestrator Pattern ⚠️ CRITICAL

Every numbered script (`N_module.py`) must be **≤150 lines** and:

```
Responsibility         | Script (N_module.py) | Module (src/module/)
-----------------------|----------------------|---------------------
Argument parsing       | ✅                   | ❌
Logging setup          | ✅                   | ❌
Output dir management  | ✅                   | ❌
Exit code handling     | ✅                   | ❌
Core processing logic  | ❌                   | ✅
Algorithm implementation| ❌                  | ✅
Helper functions       | ❌                   | ✅
```

### Standard Script Template

```python
#!/usr/bin/env python3
"""
Step N: Module Name (Thin Orchestrator)

Pipeline Flow: main.py → N_module.py (this) → module/ (implementation)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gnn.utils.pipeline_orchestration.pipeline_template import create_standardized_pipeline_script
from module_name import process_module

run_script = create_standardized_pipeline_script(
    "N_module.py",
    process_module,
    "Description of what this step does",
    additional_arguments={
        "extra_arg": {"type": str, "help": "Extra argument", "default": "default"}
    },
)


def main() -> int:
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
```

---

## Data Dependency Graph

Artifact-consuming edges (verified against each step's processor; consumer reads
the producer's output directory):

- Step 3 → 6, 7, 9 (`3_gnn_output` manifest + parsed JSON)
- Step 3 → 8 (`*_parsed.json` preferred, markdown re-parse fallback)
- Step 11 → 12 (rendered scripts + render manifest)
- Step 12 → 16 (execution results); 12 → 20 and 16 → 20 (website aggregation)
- `00_pipeline_summary/pipeline_execution_summary.json` → 23 and 24
- Optional enrichment reads (only when present): 10 → 13, 12 → 15, 11 → 17, 12 → 17

Steps 0, 4, 5, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22 re-parse the input GNN
files directly; steps 1, 2, 21 consume no pipeline data. Step 23 additionally
censuses every `N_*_output` directory.

**Automatic dependency resolution**: `--only-steps "11,12"` auto-includes step 3.

---

## Main Orchestrator (`src/gnn/main.py`)

- Executes steps 0–24 as **subprocesses** with proper working directory
- Tracks: timing, memory, exit codes, correlation IDs
- Generates `output/00_pipeline_summary/pipeline_execution_summary.json`
- Step timeouts: Tests=20min, LLM=10min, Execute=5min, others=5min
- Status codes: `SUCCESS`, `SUCCESS_WITH_WARNINGS`, `PARTIAL_SUCCESS`, `FAILED`, `TIMEOUT`

### Key Utilities (`src/gnn/utils/`, `src/gnn/pipeline/`)

| Utility | Purpose |
|---------|---------|
| `EnhancedArgumentParser` | Centralized arg parsing with explicit defaults |
| `setup_step_logging` | Standardized logging with correlation IDs |
| `get_output_dir_for_script` | Standardized output directory creation |
| `performance_tracker` | Resource usage monitoring |
| `create_standardized_pipeline_script` | Script factory (**preferred**) |

---

## Exit Code Conventions

| Code | Meaning | Pipeline Action |
|------|---------|----------------|
| `0` | Success | Continue |
| `1` | Critical Error | Stop (only steps 1, 3 use this) |
| `2` | Warnings | Continue |

**Steps 8, 9, 12**: ALWAYS return `0` — see [error_handling.md](error_handling.md).

---

**Last Updated**: 2026-09-26 | **Pipeline Version**: 3.6.0
