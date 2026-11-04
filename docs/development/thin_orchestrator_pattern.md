# Thin Orchestrator Pattern - GNN Pipeline Architecture

## Overview

The GNN Processing Pipeline follows a **thin orchestrator pattern** to ensure maintainability, modularity, and comprehensive testability. This pattern separates pipeline orchestration from core domain logic.

## 🏗️ Architectural Pattern

### Core Components

1. **Numbered Scripts** (e.g., `11_render.py`, `10_ontology.py`): Thin orchestrators that handle:
   - Pipeline orchestration
   - Argument parsing and validation
   - Logging setup and management
   - Output directory management
   - Result aggregation and reporting
   - Error handling and recovery

2. **Module `__init__.py`**: Imports and exposes functions from modular files within the module folder

3. **Modular Files** (e.g., `src/gnn/render/processor/rendering.py`, `src/gnn/ontology/processor.py`): Contain the actual implementation of core methods

4. **Tests**: All methods are tested in `tests/` with comprehensive test coverage

## 📁 File Organization

```
src/
├── 11_render.py                    # Thin orchestrator - imports from render/
├── render/
│   ├── __init__.py                 # Imports from renderer.py, pymdp/, etc.
│   ├── renderer.py                 # Core rendering functions
│   ├── pymdp/                      # PyMDP-specific rendering
│   │   ├── __init__.py
│   │   ├── pymdp_renderer.py
│   │   └── pymdp_templates.py
│   ├── rxinfer/                    # RxInfer.jl-specific rendering
│   │   ├── __init__.py
│   │   └── rxinfer_renderer.py
│   └── discopy/                    # DisCoPy-specific rendering
│       ├── __init__.py
│       └── discopy_renderer.py
├── 10_ontology.py                  # Thin orchestrator - imports from ontology/
├── ontology/
│   ├── __init__.py                 # Imports from processor.py
│   └── processor.py                # Core ontology processing functions
└── tests/
    ├── test_render_integration.py  # Tests for render module
    └── test_ontology_integration.py # Tests for ontology module
```

## ✅ Correct Pattern Examples

### Thin Orchestrator Script (`11_render.py`)

```python
#!/usr/bin/env python3
"""
Render GNN specifications to various target languages.
"""

from pathlib import Path
import logging

# Import core functionality from module
from gnn.render import (
    render_gnn_to_pymdp,
    render_gnn_to_rxinfer,
    render_gnn_to_activeinference_jl,
    render_gnn_to_discopy,
)


def main():
    """Main rendering orchestration."""
    # Setup logging, argument parsing, output directories
    # ...

    # Delegate core functionality to module
    ok, message, files = render_gnn_to_pymdp(model_data, output_dir / "model_pymdp.py")
    # ...
```

The snippets in this document illustrate the *split*, not literal signatures. Canonical
signatures live in [src/gnn/render/AGENTS.md](../../src/gnn/render/AGENTS.md).

### Module `__init__.py` (`src/gnn/render/__init__.py`)

```python
"""
Render module for GNN Processing Pipeline.
"""

# Import core functions from modular files
from .processor.rendering import render_gnn_spec
from .pymdp.pymdp_renderer import render_gnn_to_pymdp
from .rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
from .activeinference_jl.activeinference_renderer import render_gnn_to_activeinference_jl
from .discopy.discopy_renderer import render_gnn_to_discopy

__all__ = [
    "render_gnn_spec",
    "render_gnn_to_pymdp",
    "render_gnn_to_rxinfer",
    "render_gnn_to_activeinference_jl",
    "render_gnn_to_discopy",
]
```

### Modular File (`src/gnn/render/processor/rendering.py`)

This leaf owns single-spec dispatch. The per-framework generators live in their own
subpackages (`pymdp/pymdp_renderer.py`, `rxinfer/rxinfer_renderer.py`,
`activeinference_jl/activeinference_renderer.py`, `discopy/discopy_renderer.py`), and the
facade re-exports all of them.

```python
#!/usr/bin/env python3
"""
Renderer Module - Core rendering functionality.
"""


def render_gnn_spec(gnn_spec: Dict, target: str, output_directory: Path, options: Dict | None = None):
    """Route one parsed spec to the target framework's canonical renderer."""
    # Actual implementation here
    # ...
```

## ❌ Incorrect Pattern Examples

### Wrong: Defining Core Logic in Numbered Script

```python
# WRONG: 11_render.py should NOT contain this
def render_gnn_spec(gnn_spec: Dict, target: str, output_directory: Path):
    """Route one parsed spec to the target framework's canonical renderer."""
    # This should be in src/gnn/render/processor/rendering.py
    # ...
```

### Wrong: Long Domain Logic in Numbered Scripts

```python
# WRONG: domain behavior belongs in modular files
def process_complex_rendering_logic():
    """This is too long for a numbered script."""
    # 50+ lines of implementation
    # Should be moved to src/gnn/render/processor/rendering.py
```

## 🧪 Testing Pattern

### Test File (`tests/render/test_render_integration.py`)

```python
#!/usr/bin/env python3
"""
Test render integration functionality.
"""

import pytest
from gnn.render import render_gnn_to_pymdp, render_gnn_to_rxinfer


class TestRenderIntegration:
    """Test render integration functionality."""

    def test_render_gnn_to_pymdp(self, tmp_path):
        """Test PyMDP code generation."""
        gnn_spec = {"model_name": "test_model"}
        ok, message, files = render_gnn_to_pymdp(gnn_spec, tmp_path / "test_model_pymdp.py")
        assert ok, message
        assert files

    def test_render_gnn_to_rxinfer(self, tmp_path):
        """Test RxInfer.jl code generation."""
        gnn_spec = {"model_name": "test_model"}
        ok, message, files = render_gnn_to_rxinfer(gnn_spec, tmp_path / "test_model.jl")
        assert ok, message
        assert files
```

## 🎯 Benefits of This Pattern

1. **Maintainability**: Core logic is separated from orchestration
2. **Testability**: Each component can be tested independently
3. **Modularity**: Easy to add new functionality without modifying numbered scripts
4. **Reusability**: Core functions can be imported and used elsewhere
5. **Clarity**: Clear separation of concerns between pipeline flow and domain logic

## 🔧 Implementation Guidelines

### For New Pipeline Steps

1. **Create the numbered script** (e.g., `12_new_step.py`): Handle orchestration only
2. **Create the module directory** (e.g., `src/new_step/`): Organize core functionality
3. **Create modular files** (e.g., `src/new_step/processor.py`): Implement core logic
4. **Update module `__init__.py`**: Import and expose functions
5. **Create tests** (e.g., `tests/new_step/test_new_step_integration.py`): Comprehensive testing

### For Existing Steps

1. **Move core logic** from numbered scripts to modular files
2. **Update numbered scripts** to import from modules
3. **Update module `__init__.py`** to expose functions
4. **Create/update tests** for comprehensive coverage

## 📋 Checklist for Compliance

- [ ] Numbered scripts contain only orchestration logic
- [ ] Numbered scripts stay at or below 150 lines
- [ ] Core methods are defined in modular files within module folders
- [ ] Module `__init__.py` imports and exposes functions from modular files
- [ ] All methods are tested in `tests/`
- [ ] Helper methods in numbered scripts remain small and focused
- [ ] Clear separation between pipeline flow and domain logic
- [ ] Comprehensive error handling and logging
- [ ] Proper argument parsing and validation

## 🔍 Validation Commands

```bash
# Enforce the numbered-script line budget and basic script structure
uv run --extra dev python -m pytest tests/pipeline/test_pipeline_scripts.py::TestPipelineScriptDiscovery -q

# Inspect current numbered-script sizes
rg --files src -g '[0-9]*_*.py' | sort -V | xargs wc -l
```
