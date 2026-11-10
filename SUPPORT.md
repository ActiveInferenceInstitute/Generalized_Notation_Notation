# Support for Generalized Notation Notation

> **📋 Document Metadata**  
> **Type**: Support Guide | **Audience**: All Users | **Complexity**: Beginner-Friendly  
> **Cross-References**: [README](README.md) | [Troubleshooting](docs/troubleshooting/README.md) | [Contributing](CONTRIBUTING.md)

Use this guide to troubleshoot Generalized Notation Notation (GNN) or contact the project.

## Documentation

Before seeking direct support, please check the extensive documentation provided with the project:

- **README.md**: The [main README file](./README.md) provides an overview of the project, setup instructions, and how to run the processing pipeline.
- **`docs/` Directory**: This directory contains detailed information on various aspects of GNN:
  - [`docs/gnn/about_gnn.md`](./docs/gnn/about_gnn.md): General information about GNN.
  - [`docs/gnn/gnn_overview.md`](./docs/gnn/gnn_overview.md): A high-level overview.
  - [`docs/gnn/reference/gnn_syntax.md`](./docs/gnn/reference/gnn_syntax.md): Detailed specification of GNN syntax.
  - [`docs/gnn/reference/gnn_file_structure_doc.md`](./docs/gnn/reference/gnn_file_structure_doc.md): Description of GNN file organization.
  - [`docs/gnn/tutorials/gnn_examples_doc.md`](./docs/gnn/tutorials/gnn_examples_doc.md): Examples and use cases.
  - And many other useful documents covering implementation, tools, and the GNN paper.
- **Example GNN Files**: Explore the exemplar corpus in `input/gnn_files/` (start from [`input/gnn_files/INDEX.md`](./input/gnn_files/INDEX.md), which maps task folders to model kinds); `src/gnn/gnn_examples/` holds the single example packaged with the `gnn` module.
- **Pipeline Documentation**: Check [`src/gnn/README.md`](./src/gnn/README.md) for detailed pipeline information.

## Quick Troubleshooting

### Common Issues and Solutions

**🐍 Python Version Issues**

```bash
uv run --frozen --no-sync python --version
```

The supported package range is CPython 3.11–3.14. Use Python 3.14 for the
[default frozen profile](README.md#runtime-defaults-and-required-profiles).

**📦 Dependency Issues**

```bash
uv sync --frozen --python 3.14
uv run --frozen --no-sync gnn doctor
```

**🔧 Pipeline Failures**

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md --strict
uv run --frozen --no-sync gnn run --target-dir input/gnn_files/basics \
  --only-steps 3 5 --skip-steps 0 1 2 --output-dir output/support-repro
```

**💾 Disk Space Issues**

```bash
df -h
du -sh output
```

Review existing run directories before moving them. Use a dedicated output root
for each reproduction so its results and evidence can be kept together.

### Getting Started Support

Follow the [tagged installation and first model](README.md#quick-start).
For native execution, choose a backend from the
[framework availability guide](docs/execution/FRAMEWORK_AVAILABILITY.md)
and follow its installation and admission requirements.

## GitHub Issues

For bug reports, feature requests, or specific questions that are not covered by the documentation, the primary place to seek support is through [GitHub Issues](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues).

- **Search Existing Issues**: Before creating a new issue, please search existing open and closed issues to see if your question has already been addressed.
- **Bug Reports**: If you are reporting a bug, please provide:
  - A clear and descriptive title.
  - Steps to reproduce the bug.
  - The version of GNN you are using (if applicable).
  - Any relevant error messages or logs.
  - Your operating system and Python version.
- **Feature Requests**: For feature requests, describe the feature, its potential benefits, and any proposed implementation ideas.

## Community Channels

Join the wider Active Inference community for discussions, questions, and collaboration related to GNN and Active Inference in general:

- **Active Inference Institute Community**: Visit the [Active Inference Institute website](https://activeinference.institute/) for links to community platforms such as Discord, forums, or mailing lists where GNN might be discussed.
- **GitHub Discussions**: Use [GitHub Discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions) for community discussions and questions.
- **Discord Community**: Join the [Active Inference Discord](https://discord.activeinference.institute/) for real-time discussions.

## Direct Contact (for specific inquiries)

For matters not suitable for public forums or GitHub issues (e.g., sensitive inquiries), you may try to reach out to the project maintainers. However, for general support and technical questions, public channels are preferred as they benefit the entire community.

- Refer to the project maintainers listed on the [GitHub repository page](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation).
- For security-related issues, see [SECURITY.md](./SECURITY.md) for proper reporting procedures.

## Commercial Support

The project is community-supported.

## Support Resources by Topic

### 🧠 Active Inference Concepts

- [About GNN](./docs/gnn/about_gnn.md) - Introduction to GNN and Active Inference
- [GNN Overview](./docs/gnn/gnn_overview.md) - High-level concepts
- [GNN Paper](./docs/gnn/gnn_paper.md) - Academic paper details

### 🛠️ Technical Implementation

- [GNN Syntax](./docs/gnn/reference/gnn_syntax.md) - Detailed syntax specification
- [File Structure](./docs/gnn/reference/gnn_file_structure_doc.md) - GNN file organization
- [Implementation Guide](./docs/gnn/integration/gnn_implementation.md) - Implementation details

### 🎯 Framework Integration

- [PyMDP Integration](./docs/pymdp/) - PyMDP framework integration
- [RxInfer.jl Integration](./docs/rxinfer/) - RxInfer.jl framework integration
- [ActiveInference.jl Integration](./docs/activeinference_jl/) - ActiveInference.jl framework integration
- [MCP Integration](./docs/mcp/) - Model Context Protocol

### 🔧 Pipeline and Tools

- [Pipeline Architecture](./ARCHITECTURE.md) - 25-step pipeline overview
- [API Reference](./docs/api/README.md) - Complete API documentation
- [Testing Guide](./docs/testing/README.md) - Testing strategies

### 🚨 Troubleshooting

- [Troubleshooting Guide](./docs/troubleshooting/) - Common issues and solutions
- [Security Guide](./docs/security/README.md) - Security considerations
- [Deployment Guide](./docs/deployment/README.md) - Production deployment

---

We strive to support our users and contributors. Using the channels above helps us manage requests efficiently and build a helpful knowledge base for everyone.
