# petri_nets

## Overview

This directory contains documentation and resources for the petri_nets subsystem.

**Status**: ✅ Documentation Module  
**Version**: 1.0  

---

## Purpose

Petri net modeling and analysis for workflow specification (documentation only)

> **Scope note**: Petri nets are not a render or execution framework. The only Petri-net support in the tree is the parse-only `PNMLParser` (`GNNFormat.PNML` in `src/gnn/parsers/xml_parser.py`) — no PNML serializer in `SERIALIZER_REGISTRY`, no PNML exporter in `src/gnn/export/`, no entry in `src/gnn/render/framework_registry.py`, and no Step 12 executor. The `.pnml`/`.xml` files here are sample models.

This subsystem is part of the broader GNN (Generalized Notation Notation) documentation ecosystem, integrated with the 25-step processing pipeline.

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [docs/README.md](../README.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
2. **Simulation** (Steps 10-16): Model execution and analysis  
3. **Integration** (Steps 17-24): System coordination and output

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

**Main GNN Documentation**: [gnn/](../gnn/)
- [GNN Overview](../gnn/gnn_overview.md)
- [GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)
- [GNN Examples](../gnn/tutorials/gnn_examples_doc.md)

**Pipeline Architecture**: [src/](../../src/)
- [Pipeline AGENTS](../../src/gnn/AGENTS.md)
- [Pipeline README](../../src/gnn/README.md)

---

**Status**: Documentation module — PNML samples and modeling notes (PNML is parse-only; no export, render, or execution support)  
**Maintenance**: Regular updates with new features
