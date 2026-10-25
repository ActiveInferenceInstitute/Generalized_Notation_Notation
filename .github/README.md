# Generalized Notation Notation (GNN)

**Describe an Active Inference generative model once. Validate its structure,
render framework-specific code, execute admitted models and inspect the evidence.**

[![Version: 5.0.1](https://img.shields.io/badge/version-5.0.1-00a6b8.svg)](../CHANGELOG.md#501---2026-10-24)
[![CI](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/workflows/ci.yml)
[![CI Python: 3.11–3.14](https://img.shields.io/badge/CI%20Python-3.11%E2%80%933.14-3776ab.svg)](workflows/ci.yml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](../LICENSE.md)

GNN is a human-readable, machine-parsable **Markdown notation** for
[Active Inference](https://activeinference.org/) generative models. Use it to
describe model assumptions in text, check their structure and run supported
models through framework-specific implementations.

[Quick start](#quick-start) · [Examples](#choose-a-model) ·
[Backends](#render-and-execute-backends) · [Documentation](#documentation-map) ·
[Release & paper](#release-artifacts-and-citation) · [Roadmap](#next-steps) ·
[Contribute](#contributing-and-support)

[![GNN: categorical and continuous model specifications move through Specify, Validate, Render, Execute and Inspect, with source identity, configuration and traceable artifacts connecting the workflow.](gnn-overview-2026-10-21.png)](#from-model-source-to-evidence)

*Five conceptual stages in the 25-step workflow. Backend admission depends on
the model contract; the illustration contains no measured results.*

This repository provides the notation, curated model sources and a **25-step
scientific workflow** spanning parsing, validation, visualization, simulation,
analysis and publication. Start with the CLI below, or explore the
[Python and service interfaces](#interfaces).

**Software:** 5.0.1. [Published release records](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases) own accepted tags and assets.

This major version adds replayable numerical witnesses, interactive evidence
inspection and configurable file workers. It targets coordinated FEP_Formal
2 qualification against the current scientific sources; see the
[paired release procedure](../docs/development/fep_lean_paired_revision.md).
Version metadata does not establish publication or completed pairing.
[Candidate release notes](../CHANGELOG.md#501---2026-10-24) describe the changes;
[package metadata](../pyproject.toml) owns version and dependency requirements.
**Page updated:** 2026-10-24.

## Start here

| Your goal | Best starting point |
| --- | --- |
| Run a first model | [Quick start below](#quick-start), then the [setup guide](../SETUP_GUIDE.md) |
| Write or understand GNN | [Language hub](../docs/gnn/language/README.md), [syntax reference](../docs/gnn/reference/gnn_syntax.md), [examples tutorial](../docs/gnn/tutorials/gnn_examples_doc.md) |
| Choose a scientific example | [Exemplar index](../input/gnn_files/INDEX.md) and [model-family manifest](../input/model_family_manifest.json) |
| Integrate a backend or interface | [Backend guide](../docs/gnn/integration/framework_integration_guide.md), [architecture](../ARCHITECTURE.md), [interface map](#interfaces) |
| Inspect the release or paper | [Release artifacts and citation](#release-artifacts-and-citation) |
| Contribute a focused improvement | [TO-DO.md](../TO-DO.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [AGENTS.md](../AGENTS.md) |

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.4** and use
Python **3.14** for the frozen default. Ordinary CI also tests Python 3.11–3.13.
See the [runtime exceptions and required profiles](../README.md#runtime-defaults-and-required-profiles)
for the Python 3.15 HDF5 constraint, Python-specific scientific stack and the
supported uv-only Gradio/Pillow override. Native GUI callbacks use Python 3.11–3.13.

### Install the released source

```bash
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation
git checkout v4.3.0
uv sync --frozen --python 3.14
```

These commands select the released source. Contributors can use `git switch main`
before installing the [development profile](../CONTRIBUTING.md); the release tag
continues to identify the published artifacts.

### Inspect one model

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md --strict
uv run --frozen --no-sync gnn extract input/gnn_files/discrete/two_state_bistable.md --compact
```

Validation reports **12 variables and 11 connections** for this committed
example. Extraction returns its two states, two observations, two actions and
source-declared parameters as structured JSON.

### Run a small local workflow

```bash
uv run --frozen --no-sync gnn run \
  --target-dir input/gnn_files/basics \
  --output-dir output/quickstart \
  --only-steps 3 5 6 7 8 \
  --skip-steps 0 1 2
```

This example parses, type-checks, validates, exports and visualizes the basic
models. `--target-dir` takes a **directory**; `validate` and `extract` take a
**file**. Pipeline prerequisites are resolved by the orchestrator. Each new run
gets its own identity; inspect its summary and artifacts under the selected
output directory.

**What to inspect:** validation diagnostics, exported model data and graph/matrix
views. This selected workflow does not run native inference. To simulate a model,
install its backend and follow that backend's admitted model examples below.

For simulation, use the [full quick-start guide](../docs/quickstart.md) and
[backend setup](../SETUP_GUIDE.md). The full workflow can include native
frameworks, Julia/Stan toolchains, GUI/audio and configured LLM providers;
provision those dependencies and resource budgets for the steps you select.
Configuration starts in [input/config.yaml](../input/config.yaml). LLM work
reports coverage and context refusals; completion is accepted only for the
source/prompt requests actually executed.

## What GNN 4 delivers

- **Model assumptions you can read.** Declare state spaces, connections,
  parameters, equations, time and ontology annotations for categorical POMDP
  and linear-Gaussian models.
- **Checks before simulation.** Validate dimensions, probabilities, covariance
  assumptions and scientific contracts, with explicit unsupported outcomes.
- **Native backend workflows.** Render code in Step 11 and supervise native
  scripts in Step 12 using the selected backend's packages and toolchain.
- **Evidence tied to its source.** Bind model identities, source hashes,
  configuration and artifact indexes to the current run. Required unfinished
  work prevents success. See the [v4 run-ownership guide](../docs/development/run_ownership_migration.md).
- **Inspectable outputs.** Produce graphs, matrix views, statistics, reports
  and static websites. Numerical comparisons require compatible model sources,
  inference semantics, inputs and precision.
- **Several ways to work.** Use CLI, Python, REST, MCP, editor/LSP or GUI;
  optional LLM, audio and ML integrations extend selected workflows.
  [Durable runs](../docs/development/durable-runs.md) support longer sessions.

The **4.3.0** numerical witness foundation independently reads a restricted
source subset and compares finite path enumeration and joint-Gaussian conditioning
with actual bnlearn/JAX filtering. It catches source, axis, action-time, horizon,
covariance and precision mismatches within that explicit scope. It does not
establish universal equivalence or a new Lean proof. See the
[minor changelog](../CHANGELOG.md#430---2026-10-21) and
[E4 contract](../docs/development/remaining_roadmap_contracts.md#e4-foundation-source-bound-numerical-witness-v1).

The **4.2.1** documentation update strengthens same-page/reference navigation
checks, replaces unsupported configuration examples and scopes the next minor
releases. See the [patch changelog](../CHANGELOG.md#421---2026-10-21).

The **4.2.0 release** updates RxInfer to 6.1.0 / ReactiveMP 7, refreshes
compatible dependencies and security fixes, and adds a fast local release preflight
and exact-source workflow collection. Independent Python tests and quality checks
run concurrently; native, manuscript and paired acceptance retain their full scope.
Read the [release notes](../CHANGELOG.md#420---2026-10-21),
[migration guide](../docs/development/run_ownership_migration.md) and
[workflow guide](workflows/README.md).

The **5.0.0 implementation** adds `gnn witness` for independent source-bound replay,
offline evidence filters and posterior/residual trace controls, and bounded
per-file process workers in Steps 3 and 5. The numerical and scalar Lean
evidence have separate assumptions; they do not establish universal
translation equivalence. See the [5.0.0 implementation](../CHANGELOG.md#500---2026-10-23)
and [CPU worker controls](../README.md#cpu-workers-for-multiple-files).

## From model source to evidence

A GNN model names its variables and dimensions, connects them and declares
parameters separately. This **excerpt** comes from the
[complete two-state model](../input/gnn_files/discrete/two_state_bistable.md):

```markdown
## StateSpaceBlock
A[2,2,type=float]       # Likelihood: observations × hidden states
s[2,1,type=float]       # State belief
o[2,1,type=int]         # Observation

## Connections
s-A
A-o

## InitialParameterization
A={
  (0.8, 0.2),
  (0.2, 0.8)
}
```

The overview image and the diagram below are conceptual workflows; the
[architecture guide](../ARCHITECTURE.md) and [module registry](../AGENTS.md)
document the actual artifact dependencies and step prerequisites.

```mermaid
flowchart LR
  S["GNN source + configuration"] --> I["Frozen selection + source identity"]
  I --> V["Parse, type-check, validate"]
  V --> R["11 · Render admitted code"]
  R --> E["12 · Execute native scripts"]
  E --> A["16 · Analyze compatible results"]
  A --> P["Reports, website, run receipts"]
  V --> D["Exports, graphs, ontology"]
```

Inputs live in [input/gnn_files/](../input/gnn_files/). Generated artifacts live
under the selected output directory; the repository's committed [output/](../output/)
contains publication evidence with its own recorded provenance. Consult the
current run summary before treating an artifact as evidence for a new run.

## Choose a model

| Model family or question | Example sources and guidance |
| --- | --- |
| Learn the notation | [Static perception](../input/gnn_files/basics/static_perception.md), [dynamic perception](../input/gnn_files/basics/dynamic_perception.md) |
| Minimal categorical agent | [Two-state bistable POMDP](../input/gnn_files/discrete/two_state_bistable.md), [simple MDP](../input/gnn_files/discrete/simple_mdp.md) |
| GridWorld and framework comparison | [GridWorld folder](../input/gnn_files/pomdp_gridworld/README.md) |
| Linear-Gaussian dynamics and control | [Continuous navigation](../input/gnn_files/continuous/continuous_navigation.md), [damped oscillator](../input/gnn_files/continuous/damped_oscillator_bias.md) |
| Explicit independent Gaussian agents | [Independent-agent exemplar](../input/gnn_files/continuous/independent_gaussian_agents.md); admitted native JAX/RxInfer contracts |
| Hierarchy and epistemic policies | [Block-reset hierarchy](../input/gnn_files/hierarchical/hierarchical_pomdp.md), [temporal controller](../input/gnn_files/hierarchical/temporal_hierarchy.md), [episodic T-maze](../input/gnn_files/discrete/tmaze_epistemic.md); explicit JAX contracts |
| Learning, precision and multi-agent models | [Full exemplar index](../input/gnn_files/INDEX.md), [cognitive phenomena](../docs/cognitive_phenomena/README.md) |
| Scaling studies | [PyMDP scaling examples](../input/gnn_files/pymdp_scaling_study/README.md); compare matched run receipts |

An exemplar's declared contract determines backend admission. Composed,
nonstationary and coupled models have specific supported/unsupported outcomes;
the [family manifest](../input/model_family_manifest.json) and
[scientific acceptance receipt](../docs/development/issue250_scientific_acceptance_2026_10_07.json)
record the selected cases and numerical evidence.

## Render and execute backends

The [renderer registry](../src/gnn/render/framework_registry.py) and
[executor specifications](../src/gnn/execute/executor_specs.py) describe the
maintained implementations. Generator availability, installed native
dependencies, model admission and numerical acceptance are separate checks.

| Backend | Scope and setup reference |
| --- | --- |
| [PyMDP](../docs/gnn/implementations/pymdp.md) | Categorical POMDP/MDP inference and action selection |
| [JAX](../docs/gnn/implementations/jax.md) | Admitted categorical and linear-Gaussian models, plus explicitly declared scientific/composed contracts |
| [RxInfer.jl](../docs/gnn/implementations/rxinfer.md) | Julia message-passing implementations for admitted categorical and Gaussian contracts |
| [ActiveInference.jl](../docs/gnn/implementations/activeinference_jl.md) | Julia categorical Active Inference |
| [PyTorch](../docs/gnn/implementations/pytorch.md), [NumPyro](../docs/gnn/implementations/numpyro.md), [Stan](../docs/gnn/implementations/stan.md) | Framework-specific categorical and linear-Gaussian render/execute paths; install the corresponding package/toolchain |
| [DisCoPy](../docs/gnn/implementations/discopy.md), [bnlearn](../docs/bnlearn/README.md) | Categorical diagrams and Bayesian-network workflows under their adapter contracts |
| [ngc-learn](../src/gnn/render/ngclearn/README.md) | Continuous linear-Gaussian/predictive-coding contract; see the adapter's model admission |
| [cpomdp](../docs/gnn/implementations/cpomdp.md) | Explicitly selected experimental released-wheel continuous backend, with bounded policy/resource options and numerical witnesses |
| [THRML](../docs/gnn/implementations/thrml.md) | Explicitly selected experimental finite categorical Gibbs smoothing under fixed actions, using released `thrml==0.1.4` |

THRML admits strictly positive models; structural zeros, continuous models,
cross-agent coupling and action optimization are outside its current contract.
Native sample witnesses support the reported empirical marginals; convergence,
exact inference and hardware execution require separate evidence.

See the [implementation index](../docs/gnn/implementations/README.md) for complete
adapter documentation and [setup guide](../SETUP_GUIDE.md) for optional extras
and Julia environments. [CatColab](../docs/gnn/implementations/catcolab.md) is
covered in the related modeling documentation.

## Pipeline: all 25 steps

Numbered scripts in [src/gnn/](../src/gnn/) delegate to their module owners.
Use selected steps for a focused task or the full pipeline for a provisioned
workflow. The [module documentation index](../docs/gnn/modules/README.md) and
[root AGENTS guide](../AGENTS.md) provide implementation details.

<details>
<summary>Expand the complete step map (0–24)</summary>

| Step | Module guide | Purpose |
| ---: | --- | --- |
| 0 | [Template](../docs/gnn/modules/00_template.md) | Initialize the pipeline |
| 1 | [Setup](../docs/gnn/modules/01_setup.md) | Inspect/setup dependencies |
| 2 | [Tests](../docs/gnn/modules/02_tests.md) | Execute test selections |
| 3 | [GNN](../docs/gnn/modules/03_gnn.md) | Discover and parse selected sources |
| 4 | [Model registry](../docs/gnn/modules/04_model_registry.md) | Model metadata and versioning |
| 5 | [Type checker](../docs/gnn/modules/05_type_checker.md) | Dimensions, types and resource estimates |
| 6 | [Validation](../docs/gnn/modules/06_validation.md) | Semantic and consistency checks |
| 7 | [Export](../docs/gnn/modules/07_export.md) | Generate exchange formats |
| 8 | [Visualization](../docs/gnn/modules/08_visualization.md) | Graph and matrix views |
| 9 | [Advanced visualization](../docs/gnn/modules/09_advanced_viz.md) | Additional plots and interactive artifacts |
| 10 | [Ontology](../docs/gnn/modules/10_ontology.md) | Map model terms to ontology concepts |
| 11 | [Render](../docs/gnn/modules/11_render.md) | Generate admitted backend code |
| 12 | [Execute](../docs/gnn/modules/12_execute.md) | Supervise native simulation scripts |
| 13 | [LLM](../docs/gnn/modules/13_llm.md) | Configured model interpretation and analysis |
| 14 | [ML integration](../docs/gnn/modules/14_ml_integration.md) | Machine-learning integrations |
| 15 | [Audio](../docs/gnn/modules/15_audio.md) | Sonification and audio artifacts |
| 16 | [Analysis](../docs/gnn/modules/16_analysis.md) | Statistics and source-compatible comparison |
| 17 | [Integration](../docs/gnn/modules/17_integration.md) | Coordinate cross-module outputs |
| 18 | [Security](../docs/gnn/modules/18_security.md) | Security validation and access controls |
| 19 | [Research](../docs/gnn/modules/19_research.md) | Research tooling and experimental features |
| 20 | [Website](../docs/gnn/modules/20_website.md) | Publish static run views |
| 21 | [MCP](../docs/gnn/modules/21_mcp.md) | Discover and expose model tools |
| 22 | [GUI](../docs/gnn/modules/22_gui.md) | Headless artifacts and interactive editors |
| 23 | [Report](../docs/gnn/modules/23_report.md) | Assemble current-run reports |
| 24 | [Intelligent analysis](../docs/gnn/modules/24_intelligent_analysis.md) | Summarize the current run |

</details>

## Interfaces

| Surface | Entry point and documentation |
| --- | --- |
| Python | Installed `gnn.*` package; [API reference](../docs/api/README.md) |
| CLI | `gnn`; [commands and exit codes](../src/gnn/cli/README.md) |
| REST | `gnn serve`; [service guide](../src/gnn/api/AGENTS.md) |
| MCP | `gnn mcp list` / `gnn mcp info`; [transport guide](../docs/gnn/mcp/README.md), [tool reference](../docs/gnn/mcp/tool_reference.md) |
| Editor/LSP | `gnn lsp`; [LSP guide](../src/gnn/lsp/README.md) |
| GUI | `gnn gui`; [GUI guide](../src/gnn/gui/README.md) |
| Templates | `gnn templates list` / `gnn pull`; [template documentation](../docs/templates/README.md) |

## Documentation map

| Read next | What it covers |
| --- | --- |
| [Guided start](../docs/START_HERE.md), [learning paths](../docs/learning_paths.md) | Onboarding for researchers and developers |
| [GNN documentation hub](../docs/gnn/README.md), [DOCS.md](../DOCS.md) | Language, pipeline and integration maps |
| [Syntax](../docs/gnn/reference/gnn_syntax.md), [schema](../docs/gnn/reference/gnn_schema.md), [type system](../docs/gnn/reference/gnn_type_system.md) | Model authoring and interpretation |
| [Architecture](../ARCHITECTURE.md), [SPEC.md](../SPEC.md) | Implementation boundaries and declared behavior |
| [Setup](../SETUP_GUIDE.md), [operations](../docs/gnn/operations/gnn_tools.md), [troubleshooting](../docs/gnn/operations/gnn_troubleshooting.md) | Installation, commands and diagnosis |
| [Active Inference](../docs/active_inference/README.md), [cognitive models](../docs/cognitive_phenomena/README.md) | Scientific background and example domains |
| [Testing guide](../docs/gnn/testing/README.md), [verification commands](../TO-DO.md#verification-and-execution-rules) | Acceptance and reproducibility |
| [Durable runs](../docs/development/durable-runs.md), [orchestration contracts](../docs/pipeline/v3_orchestration.md) | Manifests, resumption and container plans |
| [FEP/GEO paired revisions](../docs/development/fep_lean_paired_revision.md) | Cross-repository custody and coordinated owner changes |

## Release artifacts and citation

The published **4.2.0** record uses its [version release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.2.0)
for ordinary wheel/sdist, manuscript, source-bound verification records and
[checksums](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.2.0/SHA256SUMS).
Actual tag/source/check identities and directly downloaded asset hashes establish
acceptance.

Read the [4.2.0 manuscript PDF](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.2.0/GNN-4.2.0-manuscript.pdf)
and [release overview](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.2.0/GNN-4.2.0-release-overview.png).
Main may contain later documentation changes; release artifacts retain their
original source and tag identities. Historical releases and receipts remain
linked from [CHANGELOG.md](../CHANGELOG.md).

For academic use, follow [CITATION.cff](../CITATION.cff). The initial publication
is **Smékal, J., & Friedman, D. A. (2023), _Generalized Notation Notation for
Active Inference Models_, Active Inference Journal**. The
[project concept DOI](https://doi.org/10.5281/zenodo.7803313) identifies the project.
Exact-version source and distribution archival require separate acceptance;
see the [archive procedure](../docs/releases/exact_version_archival.md) and
[remaining archival work](../TO-DO.md#minor-work).

The repository is maintained by the
[Active Inference Institute](https://activeinference.org/) community and licensed
under [CC BY-NC-SA 4.0](../LICENSE.md).

## Next steps

The [4.3.0–4.9.0 delivery plan](../TO-DO.md#versioned-delivery-plan-430-through-490)
sets bounded scope, dependencies, review ownership and acceptance for each target.

The [forward-only roadmap](../TO-DO.md) owns scope, priority, acceptance evidence
and dependencies for remaining work. Browse [minor work](../TO-DO.md#minor-work)
and [major work](../TO-DO.md#major-work), or choose a scoped
[open issue](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues).
Effort size and release version are separate decisions.

## Contributing and support

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[agent/contributor guide](../AGENTS.md). Propose focused changes with a source
baseline and acceptance evidence; use current CI for test outcomes and timings.
The [handoff checklist](../docs/development/contributor_handoff.md) helps make
completed work reviewable.

- [Issues](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues): reproducible bugs and scoped work.
- [Discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions): modeling questions and ideas.
- [SUPPORT.md](../SUPPORT.md): help and community channels.
- [SECURITY.md](../SECURITY.md): vulnerability reporting.
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md): participation standards.
- [Contributors](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/graphs/contributors): contribution history.

## Repository and automation guide

| Path | Role |
| --- | --- |
| [src/gnn/](../src/gnn/) | Installed Python package, numbered orchestrators and module implementations |
| [input/](../input/) | Model sources, family manifest and configuration |
| [output/](../output/) | Committed publication artifacts and recorded provenance |
| [tests/](../tests/) | Behavior, contract and integration checks |
| [docs/](../docs/) | Language, framework, scientific and operator guides |
| [scripts/](../scripts/) | Audits, acceptance tools and publication tooling |
| [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock) | Package metadata and locked dependencies |
| [Root README](../README.md) | Extended project narrative and examples |

<details>
<summary>Maintainer reference: .github files, workflows and local checks</summary>

### Automation in this folder

[AGENTS.md](AGENTS.md) defines folder guardrails; [SPEC.md](SPEC.md) defines its
purpose. [workflows/README.md](workflows/README.md) documents triggers and exact
workflow commands, with [workflow AGENTS](workflows/AGENTS.md) and
[SPEC](workflows/SPEC.md) for maintainers. Dependabot configuration is in
[dependabot.yml](dependabot.yml); CodeQL configuration is in
[codeql/codeql-config.yml](codeql/codeql-config.yml).

### Landing-page artwork

The [overview PNG](gnn-overview-2026-10-21.png) was generated with the built-in
imagegen tool on 2026-10-21; its [exact prompt](gnn-overview-2026-10-21.prompt.txt)
is retained alongside it. The illustration explains the workflow and contains
no measured results. Its numbered cards are presentation stages, not pipeline
step numbers. The separately published [4.2.0 release overview](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.2.0/GNN-4.2.0-release-overview.png)
retains its original release identity.

### Workflow index

| Workflow or configuration | Purpose |
| --- | --- |
| [comprehensive-native-coverage.yml](workflows/comprehensive-native-coverage.yml) | Separate exact-source Python 3.11/3.12/3.13 native coverage reports with original scope and floor, process provenance and exact line unions; [procedure and limits](../docs/development/native_comprehensive_coverage.md) |
| [ci.yml](workflows/ci.yml) | Python 3.11/3.12/3.13/3.14 tests; 3.12 lint/types/docs/capability checks; pipeline contracts; optional-dependency and Bandit lanes |
| [installed-platforms.yml](workflows/installed-platforms.yml) | Ordinary installed-wheel native acceptance outside the checkout: Linux 3.11/3.12/3.14, macOS 3.14 and Windows 3.14; [acceptance scope and limits](../docs/development/installed_platform_acceptance.md) |
| [local-gates.yml](workflows/local-gates.yml) | Repository, manuscript-token and hydration gates |
| [docs-audit.yml](workflows/docs-audit.yml) | Focused strict documentation and terminology audits |
| [mcp-audit.yml](workflows/mcp-audit.yml) | MCP inventory regression gate |
| [codeql.yml](workflows/codeql.yml) | Python security analysis |
| [dependency-review.yml](workflows/dependency-review.yml) | PR dependency/license review; [fork limitations](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review#dependency-review-for-forked-repositories) |
| [supply-chain-audit.yml](workflows/supply-chain-audit.yml) | Scheduled vulnerability checks on locked exports |
| [full-extras.yml](workflows/full-extras.yml) | Scheduled/manual all-extras installation and acceptance |
| [rxinfer.yml](workflows/rxinfer.yml) | Native RxInfer 6.1 / ReactiveMP 7 tests on Julia 1.13.2, with environment-drift checks |
| [gridworld.yml](workflows/gridworld.yml) | GridWorld publication checks |
| [actionlint.yml](workflows/actionlint.yml) | Workflow YAML validation |
| [fep-lean-paired-revision.yml](workflows/fep-lean-paired-revision.yml), [fep-lean-pair.json](fep-lean-pair.json) | Pinned FEP bridge custody checks |
| [geo-infer-interchange.yml](workflows/geo-infer-interchange.yml), [gnn-pair.json](gnn-pair.json) | Pinned GEO interchange checks |
| [pair-pin-freshness.yml](workflows/pair-pin-freshness.yml) | Scheduled companion-pin freshness checks |
| [custody-re-render.yml](workflows/custody-re-render.yml) | Scheduled/manual fresh manuscript-render audit |

CI runs on documentation PRs too. Documentation, paired-custody, security and
repository gates provide complementary evidence. Exact selections, environment
requirements, schedules and permission scopes live in the workflow files.

### Local validation

From a development checkout on `main`, install the locked development tools,
then run the documentation checks appropriate to this page. `just release-preflight`
provides the shared fast preflight before hosted release qualification:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_repo_terminology.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_capability_contracts.py --strict
```

For source changes use the [current verification commands](../TO-DO.md#verification-and-execution-rules)
and the workflows' declared environments. Run `actionlint .github/workflows/*.yml`
when workflow YAML changes. Count-changing manuscript/source/test edits follow
the full rendering/custody procedure in [AGENTS.md](../AGENTS.md).

</details>
