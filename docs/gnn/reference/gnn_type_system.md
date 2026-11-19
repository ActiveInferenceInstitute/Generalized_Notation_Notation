# GNN Type System Guide

**Version**: v2.0.0 documentation bundle\
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: GNN type declarations and validation. See [framework implementations](../implementations/README.md) for current backend coverage.

This document provides comprehensive guidance on type checking and the type system for Generalized Notation Notation (GNN) models, including type inference, validation, and resource estimation.

## Overview

The GNN type system provides comprehensive type checking, validation, and resource estimation for GNN models. This ensures type safety, validates model structure, and estimates computational resources required for model execution.

The GNN type system provides:

- **Type Checking**: Comprehensive type checking and inference
- **Syntax Validation**: GNN syntax validation
- **Resource Estimation**: Computational resource estimation
- **Performance Prediction**: Performance prediction modeling
- **Structure Validation**: Model structure validation

## Type System Pipeline Integration

GNN type checking is integrated into the 25-step processing pipeline:

**Step 5: Type Checker** (`5_type_checker.py`)

- Type checking and validation
- Resource estimation
- See: [src/gnn/type_checker/AGENTS.md](../../../src/gnn/type_checker/AGENTS.md)

**Quick Start:**

```bash
# Run type checking
python src/gnn/main.py --only-steps "3,5" --target-dir input/gnn_files --verbose

# Strict type checking with resource estimation
python src/gnn/5_type_checker.py --target-dir input/gnn_files --output-dir output --strict --estimate-resources
```

## Type System Components

### Type Categories

The GNN type system categorizes types:

- **State Types**: Hidden state variable types
- **Observation Types**: Observation variable types
- **Action Types**: Action variable types
- **Matrix Types**: Matrix dimension and type information
- **Function Types**: Function signature and type information

### Type Inference

The type system automatically infers types from GNN specifications:

```mermaid
graph TD
    GNN[GNN Model] --> PARSE[Parse Model]
    PARSE --> EXTRACT[Extract Types]
    EXTRACT --> INFER[Type Inference]
    INFER --> VALIDATE[Type Validation]
    VALIDATE --> ESTIMATE[Resource Estimation]
    ESTIMATE --> REPORT[Type Report]
```

## Type Checking

### Type Validation

Type checking validates model structure and type consistency. The import
below is the stable public path; `resource_estimator.py` re-exports the class,
which is defined in `src/gnn/type_checker/estimation/estimator.py`:

```python
from type_checker.resource_estimator import GNNResourceEstimator

# Create type checker with optional pre-loaded type data
estimator = GNNResourceEstimator(
    type_check_data="output/5_type_checker_output/type_check_results.json"
)

# Estimate resources for a single GNN file
resources = estimator.estimate_from_file("input/gnn_files/model.md")
# Returns: {memory_kb, inference_units, storage_kb, complexity_metrics}

# Estimate resources for all GNN files in a directory
all_resources = estimator.estimate_from_directory("input/gnn_files", recursive=True)
```

### Validation Features

- **Syntax Validation**: GNN syntax validation
- **Type Consistency**: Type consistency checking
- **Structure Validation**: Model structure validation
- **Constraint Checking**: Constraint validation

## Resource Estimation

Resource estimation predicts computational resources required for model execution:

### Estimation Components

- **Memory Requirements**: Memory usage estimation
- **Computational Complexity**: Computational complexity analysis
- **Execution Time**: Execution time prediction
- **Storage Requirements**: Storage space estimation

### Resource Estimation Example

```python
from type_checker.resource_estimator import GNNResourceEstimator

estimator = GNNResourceEstimator()
resources = estimator.estimate_from_file("input/gnn_files/model.md")

# Resources include:
# - memory_kb: Memory estimate based on variable dimensions × bytes_per_type
# - inference_units: Relative cost from model type, variables, edges, equations
# - storage_kb: Disk space from file size + metadata overhead
# - complexity_metrics: {
#     state_space_complexity, graph_density, avg_in_degree,
#     avg_out_degree, cyclic_complexity, temporal_complexity,
#     equation_complexity, overall_complexity (0-10 weighted composite)
# }
```

## Performance Prediction

Performance prediction models estimate model performance:

### Prediction Features

- **Execution Time**: Execution time prediction
- **Memory Usage**: Memory usage prediction
- **Scalability**: Scalability analysis
- **Optimization Suggestions**: Performance optimization suggestions

## Type System Workflow

The type checking workflow:

```mermaid
graph TD
    INPUT[GNN File] --> PARSE[Parse]
    PARSE --> SYNTAX[Syntax Check]
    SYNTAX --> TYPES[Type Inference]
    TYPES --> VALID[Type Validation]
    VALID --> RESOURCE[Resource Estimation]
    RESOURCE --> PERF[Performance Prediction]
    PERF --> OUTPUT[Type Report]
```

## Type Safety

Type safety ensures models are correctly typed:

### Safety Features

- **Type Inference**: Automatic type inference
- **Type Checking**: Comprehensive type checking
- **Type Validation**: Type validation and verification
- **Error Reporting**: Detailed type error reporting

## Best Practices

### Type System Usage

- **Early Validation**: Validate types early in model development
- **Strict Mode**: Use strict type checking for production models
- **Resource Planning**: Use resource estimation for planning
- **Performance Optimization**: Apply performance optimization suggestions

### Model Development

- **Type Annotations**: Use explicit type annotations where helpful
- **Consistency**: Maintain type consistency across models
- **Validation**: Regular type validation during development
- **Documentation**: Document type requirements and constraints

## Integration with Pipeline

Type checking integrates throughout the pipeline:

1. **Core Processing** (Steps 0-9): Type checking during parsing and validation
2. **Simulation** (Steps 10-16): Type validation during execution
3. **Integration** (Steps 17-24): Type information in comprehensive outputs

## Related Documentation

- **[Type Checker Module](../../../src/gnn/type_checker/AGENTS.md)**: Implementation details
- **[GNN Syntax](gnn_syntax.md)**: Complete syntax reference
- **[GNN Tools](../operations/gnn_tools.md)**: Complete GNN tools reference
- **[GNN Standards](gnn_standards.md)**: GNN domain knowledge and standards

## See Also

- **[Pipeline Documentation](../operations/gnn_tools.md)**: Complete pipeline guide
- **[GNN Overview](../gnn_overview.md)**: Core GNN concepts
- **[GNN Implementation Guide](../integration/gnn_implementation.md)**: Implementation guidance

---

**Status**: Maintained
**Last Updated**: 2026-04-15  
**Version**: v2.0.0 documentation bundle
