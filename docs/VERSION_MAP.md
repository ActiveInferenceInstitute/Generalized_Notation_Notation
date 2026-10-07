# GNN Version Map

One-stop map of what changed at each release and where the authoritative record
lives. Current release: **4.0.0**, published 2026-10-07 from
`1bc3a76eccccb2cc5ce8601770714075b3ec48db`. The manuscript authored epoch
remains 2026-10-02. See [the publication receipt](development/gnn_4_0_0_post_publication.json),
[release notes](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.0),
[the active backlog](../TO-DO.md) and
[the historical verification ledger](development/verification_2026_10_01.json).

Package metadata now targets **4.0.1**, a patch candidate for GUI complexity,
dependency remediation, LLM diagnostics and complete-input delivery. Its
publication and exact final checks remain pending.

| Version | Date | Theme | Primary record |
| --- | --- | --- | --- |
| 4.0.1 | Unreleased patch candidate | GUI and dependency security fixes, truthful LLM summaries and complete-input subprocess delivery; scientific/dispatch issue closeout | [CHANGELOG §4.0.1](../CHANGELOG.md) |
| 4.0.0 | 2026-10-07 | Current-run model identity, strict scientific validation, shared deadlines and readiness, full-corpus LLM scheduling, manuscript custody, typed distributed execution, experimental cpomdp control and THRML categorical smoothing, explicit independent Gaussian agents | [CHANGELOG §4.0.0](../CHANGELOG.md) |
| 3.6.0 | 2026-09-26 | Composability & Offline Truth: step-20 website per-model detail pages with breadcrumbs and client-side search, pure-dict `generate_website(..., filesystem=False)` with zero disk collection, dependency-free step-catalogue leaf module, fully offline assets (system font stack, JSON-LD + meta, atomic manifest), complexity-estimator subpackage + benchmark CLI subcommands, dashboard fold-and-delete into MCP artifact tools, six website dead-seams wired-or-removed (`website_html_filename` gone end-to-end), render band splits (`pomdp_processor` + `processor` packages), execute/processor band split, GEO-INFER consumer conformance suite | [CHANGELOG §3.6.0](../CHANGELOG.md) |
| 3.5.0 | 2026-09-22 | Surface Truth & Integration: website step-20 statuses read from the recorded execution summary, standard MCP 2024-11-05 protocol in both transports, three new MCP tools (registry 142→146), `gnn gui` CLI subcommand, full API runs-delete contract with cancellation, one canonical framework tuple, step-24 LLM cache, didChange-aware LSP diagnostics, ngc-learn LGSSM exemplar + render/execute lane | [CHANGELOG §3.5.0](../CHANGELOG.md) |
| 3.4.0 | 2026-09-17 | Model-Kind Truth: docs+manuscript generalization to discrete/continuous/multi-agent kinds (~110 docs), snapshot-based token auto-injection (`GNN_VERSION`/`GNN_MODULE_COUNT`/`GNN_TOOL_COUNT`/`GNN_TEST_COUNT`), `gnn doctor` capability probe + MCP `get_doctor_report`, bnlearn Step 12 executor, Step-6 B-orientation diagnostics + `--transpose-b`, cover-page graphical abstract | [CHANGELOG §3.4.0](../CHANGELOG.md) |
| 3.3.0 | 2026-09-06 | One Corpus: input corpus closure (former top-level fixture dirs folded into `gnn_files/`), manuscript remediation reconciled onto the `src/gnn/` layout (Rule-8 path-claims gate, preamble PDF-metadata ownership, relocated manuscript gates), `gnn.*` as the single canonical import surface | [CHANGELOG §3.3.0](../CHANGELOG.md) |
| 3.2.0 | 2026-09-02 | Exemplar Gold Standard: pure continuous (linear-Gaussian) exemplars with native JAX/NumPyro/PyTorch/Stan/RxInfer.jl backends, `unsupported` render status for categorical backends, runnable Stan HMM/LGSSM programs + cmdstanpy executor, Step 12 per-folder summary merge, Julia pre-exec gate fix | [CHANGELOG §3.2.0](../CHANGELOG.md) |
| 3.1.0 | 2026-08-30 | Release hardening: `GNN_STEP_TIMEOUT_SCALE` for slow-storage checkouts, meta-analysis correctness fix, justfile repair | [CHANGELOG §3.1.0](../CHANGELOG.md) |
| 3.0.0 | 2026-06-20 | Long-Running Orchestration: durable streams, run sessions, resumable manifests, safe-by-design contracts in `src/gnn/pipeline/` | [src/gnn/pipeline/AGENTS.md](../src/gnn/pipeline/AGENTS.md) |
| 2.0.0 | 2026-06-12 | Major architecture revision | [CHANGELOG §2.0.0](../CHANGELOG.md) |
| 1.x (1.0.0–1.9.0) | 2025-12 → 2026-06 | Initial pipeline through iterative hardening | [CHANGELOG](../CHANGELOG.md) |

## Where version facts live

- **Package version**: `pyproject.toml` (`version` field) — single source of truth.
- **Release history**: [CHANGELOG.md](../CHANGELOG.md) (Keep-a-Changelog format).
- **Release process docs**: [releases/README.md](releases/README.md).
- **Roadmap / next target**: [TO-DO.md](../TO-DO.md) (post-release maintenance and qualified capability limits).

## Version-sensitive surfaces (check these after a bump)

- `pyproject.toml` `version`
- `CHANGELOG.md` release heading + compare links
- `CITATION.cff`
- `TO-DO.md` "Current Version" line
- Release test-evidence claims in `docs/releases/`
- `src/gnn/__init__.py` `__version__` (documented in [gnn/modules/init.md](gnn/modules/init.md); kept in lock-step with `pyproject.toml` at every release since 3.2.0)
- Package-version lines in [gnn/README.md](gnn/README.md), [gnn/AGENTS.md](gnn/AGENTS.md) and [gnn/reference/SPEC.md](gnn/reference/SPEC.md)
