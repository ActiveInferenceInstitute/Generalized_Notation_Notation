# Step 14: Ml Integration

## Architectural Mapping

**Orchestrator**: `src/gnn/14_ml_integration.py` (52 lines)
**Implementation Layer**: `src/gnn/ml_integration/`

## Module Description

This module provides comprehensive machine learning integration capabilities for GNN models, including model training, evaluation, optimization, and integration with popular ML frameworks.


```
src/gnn/ml_integration/
├── __init__.py                    # Module initialization and exports
├── AGENTS.md                      # Agent scaffolding documentation
├── mcp.py                         # Model Context Protocol integration
├── processor.py                   # Scikit-learn training and feature extraction
├── README.md                      # This documentation
├── SKILL.md                       # Capability API
└── SPEC.md                        # Module specification



Main function for processing machine learning integration tasks.

**Features:**

## Agent Identity & Capabilities

# ML Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: Real Machine Learning integration using Scikit-Learn to train valid state-prediction models from GNN specifications.

**Pipeline Step**: Step 14: ML integration (14_ml_integration.py)

**Category**: Machine Learning / Model Training

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Integrate machine learning frameworks with GNN models
2. Provide model training and validation capabilities
3. Train classifiers on GNN structural features using scikit-learn (Decision Tree, Random Forest); PyTorch/TensorFlow/JAX are not currently implemented
4. Enable model optimization and hyperparameter tuning
5. Generate ML-ready datasets from GNN specifications

### Key Capabilities
- **Real Model Training**: Uses `scikit-learn` Decision Trees to learn state transition dynamics.
- **Feature Extraction**: Extracts actual dimensional and state properties from GNN markdown files.
- **Synthetic Data Generation**: Creates dynamically sized datasets matching GNN specifications for valid training simulation.
- **Model Serialization**: Saves actual `.pkl` model artifacts for deployment.

---

## API Reference

### Public Functions

#### `process_ml_integration(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main ML integration processing function called by orchestrator (14_ml_integration.py). Integrates machine learning frameworks with GNN models for training and evaluation.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for ML integration results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `**kwargs`: Accepted but not read by `process_ml_integration` — the function signature is `(target_dir, output_dir, recursive=False, verbose=False, **kwargs)` and no `model_type`, `training_mode`, `framework`, or `hyperparameter_optimization` key is ever pulled out of `kwargs`. Passing these values (e.g. via CLI flags, if added) currently has no effect; training always follows the scikit-learn path described below.

**Returns**: `bool` - True if ML integration processing succeeded, False otherwise

**Example**:
```python
from ml_integration import process_ml_integration
from pathlib import Path
import logging

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

---

## ML Framework Support

### Scikit-learn Integration
**Status**: ✅ Supported — this is the only ML framework actually implemented in `src/gnn/ml_integration/processor.py`
**Features**:
- Extracts structural features per GNN file (`num_states`, `num_observations`, `num_actions`, `num_variables`, `connectivity_ratio`, `max_dimension`, `total_parameters`, `has_precision`, `has_learning`, `has_ontology`, `has_parameterization`)
- Trains a `DecisionTreeClassifier` (`max_depth=4`) and a `RandomForestClassifier` (`n_estimators=10`, `max_depth=4`) side by side for comparison
- Classifies `model_family` when at least two distinct families are present across the input files; otherwise falls back to a `small`/`medium`/`large` complexity classification based on `total_parameters`
- Cross-validates (up to 5-fold, capped by sample count and the smallest class count) and reports mean/std accuracy plus per-model feature importance (top 5 features)
- Requires at least 2 GNN files with extractable features and a working `scikit-learn`/`numpy` install; otherwise falls back to writing a `structural_analysis` entry per file with no model trained
- Serializes each trained classifier to a `.pkl` artifact (`gnn_decision_tree.pkl`, `gnn_random_forest.pkl`) in the output directory

### PyTorch / TensorFlow-Keras / JAX-Flax
`src/gnn/ml_integration/processor.py` has no imports of or references to `torch`, `tensorflow`, or `jax` — this module trains only the scikit-learn classifiers described above. Neural-network training, custom loss functions, TensorBoard integration, and JIT-compiled models are outside this module's current scope.

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations for ML models
- `pandas` - Data manipulation for datasets
- `scikit-learn` - Traditional ML algorithms

### Optional Dependencies
- `torch` - PyTorch deep learning (recovery: simplified models)
- `tensorflow` - TensorFlow/Keras (recovery: scikit-learn)
- `jax` - JAX high-performance computing (recovery: numpy)
- `flax` - JAX neural networks (recovery: basic implementations)
- `optax` - JAX optimization (recovery: basic optimizers)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---


---

## Usage Examples

### Basic ML Integration
```python
from ml_integration.processor import process_ml_integration

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

---

## Output Specification

### Output Products

- `gnn_decision_tree.pkl` - Trained DecisionTreeClassifier artifact
- `gnn_random_forest.pkl` - Trained RandomForestClassifier artifact
- `ml_integration_results.json` - Processing summary with per-model metrics and per-file structural analysis

### Output Directory Structure
```
output/14_ml_integration_output/
├── gnn_decision_tree.pkl
├── gnn_random_forest.pkl
└── ml_integration_results.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~30-120 seconds (depending on model complexity)
- **Memory**: ~100-500MB for training
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: ~10-30s for simple models
- **Slow Path**: ~2-5min for complex neural networks
- **Memory**: ~50-200MB for typical models, ~500MB+ for large models

---

## Error Handling

### Graceful Degradation
- **No ML frameworks**: Recovery to statistical analysis only
- **Training failures**: Generate model evaluation report
- **Memory issues**: Reduce batch size and complexity

### Error Categories
1. **Framework Errors**: Missing or incompatible ML frameworks
2. **Training Errors**: Model training failures or convergence issues
3. **Data Errors**: Invalid or insufficient training data
4. **Resource Errors**: Memory or computational resource exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `14_ml_integration.py` (Step 14)
- **Function**: `process_ml_integration()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests/ml_integration/test_ml_integration_overall.py` - ML integration tests
- `main.py` - Pipeline orchestration

### Data Flow
```
GNN Models → ML Framework Selection → Dataset Preparation → Model Training → Evaluation → Deployment
```

---

## Testing

### Test Files
- `tests/ml_integration/test_ml_integration_overall.py` - Module-level tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/ml_integration/ --cov=gnn.ml_integration --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Framework detection and selection
2. Model generation and training across frameworks
3. Dataset preparation and validation
4. Performance evaluation and comparison
5. Error handling with missing frameworks

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/ml_integration/mcp.py`):

- `process_ml_integration` - Run Step 14 over a directory
- `check_ml_frameworks` - Report which ML dependencies are importable
- `list_ml_integration_targets` - List GNN files eligible for training
- `get_ml_module_info` - Module metadata

---

## Documentation
- **[README](../../../src/gnn/ml_integration/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/ml_integration/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/ml_integration/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/ml_integration/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/ml_integration](../../../src/gnn/ml_integration)
