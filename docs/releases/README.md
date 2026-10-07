# Release Management Documentation

> **📋 Document Metadata**  
> **Type**: Process Guide | **Audience**: Maintainers, Developers | **Complexity**: Intermediate  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Development Guide](../development/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation for release management, version control, and deployment processes for the GNN (Generalized Notation Notation) project. It covers semantic versioning, release cycles, quality assurance, and security release procedures.

**Status**: ✅ Production Ready  
**Version**: 1.0

## GNN 4.0.1 publication

[v4.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.1) was published on 2026-10-07 from
`17c72cf0f98d7d3bbf0159d1b1cce8c77c4e4daf`. Its
[publication receipt](../development/gnn_4_0_1_post_publication.json) records
normal companion integration, the annotated tag, exact hosted checks, zero open
issues and enabled Dependabot/CodeQL alerts, and ten directly verified assets.
The immutable release PDF retains its authored date; current main publication
metadata is renewed separately. No PyPI upload or new archival DOI is claimed.

## GNN 4.0.0 publication (historical)

[v4.0.0](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.0)
was published on 2026-10-07 from `1bc3a76eccccb2cc5ce8601770714075b3ec48db`.
The [publication receipt](../development/gnn_4_0_0_post_publication.json) records
normal companion merges, the annotated tag, exact checks and downloaded asset
hashes. The manuscript keeps its 2026-10-02 authored date; no PyPI upload or
version-specific archival DOI is claimed.

![GNN 4.0.0 overview of model notation, numerical execution and the 25-step workflow](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.0.0/GNN-4.0.0-release-overview.png)

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Development Guide](../development/README.md)**: Development workflows
- **[Deployment Guide](../deployment/README.md)**: Deployment strategies
- **[Security Guide](../security/README.md)**: Security considerations
- **[Testing Guide](../testing/README.md)**: Quality assurance

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

**Files**: 3 | **Subdirectories**: 0

### Core Files

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## Release Management

### Version Control Strategy

The GNN project follows semantic versioning:

- **Major Versions** (X.0.0): Breaking changes, major feature additions
- **Minor Versions** (0.X.0): New features and documented migrations
- **Patch Versions** (0.0.X): Bug fixes, minor improvements

### Release Cycles

- **Regular Releases**: Scheduled releases with feature additions and improvements
- **Security Releases**: Critical vulnerability response with immediate releases
- **Emergency Releases**: Urgent fixes for critical issues

### Quality Assurance

Release process includes:

- **Testing**: Comprehensive test suite execution
- **Validation**: Pipeline validation and verification
- **Documentation**: Documentation updates and review
- **Security Review**: Security scanning and validation

### Release Process

1. **Planning**: Feature planning and milestone definition
2. **Development**: Feature development and testing
3. **Quality Assurance**: Testing, validation, and review
4. **Release Preparation**: Documentation, changelog, version tagging
5. **Deployment**: Release deployment and announcement
6. **Post-Release**: Monitoring, support, and feedback collection

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - Release validation includes pipeline execution verification

2. **Simulation** (Steps 10-16): Model execution and analysis
   - Release testing includes framework execution validation

3. **Integration** (Steps 17-24): System coordination and output
   - Release artifacts include comprehensive outputs and documentation

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Development Guide](../development/README.md)**: Development workflows

### Process Documentation
- **[Deployment Guide](../deployment/README.md)**: Deployment strategies
- **[Security Guide](../security/README.md)**: Security considerations
- **[Testing Guide](../testing/README.md)**: Quality assurance procedures

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with process details
- **Functionality**: Describes actual release management capabilities
- **Completeness**: Comprehensive coverage of release processes
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[Release Management Cross-Reference](../CROSS_REFERENCE_INDEX.md#release-management)**: Cross-reference index entry
- **[Development Guide](../development/README.md)**: Development workflows
- **[Deployment Guide](../deployment/README.md)**: Deployment strategies
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new release processes and version management strategies
