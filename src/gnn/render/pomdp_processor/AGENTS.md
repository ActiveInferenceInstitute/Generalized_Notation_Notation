# POMDP Render Processor Package — Agent Scaffolding

## Module Overview

**Purpose**: Specialized processing that injects POMDP state spaces into framework-specific renderers (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, ngc-learn) with per-framework compatibility validation, canonical spec generation, and per-framework documentation.

**Pipeline Step**: Step 11: Code rendering (11_render.py), via the POMDP-aware path of [../README.md](../README.md)

**Category**: Code Generation / POMDP Renderer Orchestration

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-28

## File Structure

```
pomdp_processor/
├── __init__.py            # Package facade: module docstring, shared-helper re-exports, POMDPRenderProcessor assembly, public pomdp_to_gnn_spec / process_pomdp_for_frameworks
├── _routes.py             # RendererRoute dataclass and RENDERER_ROUTES dispatch table for nine framework renderers
├── _flow.py               # _ProcessFlowMixin: __init__ shared state and per-framework processing flow
├── _validation.py         # _CompatibilityValidationMixin: POMDP-renderer compatibility and state-space validation
├── _canonical.py          # _CanonicalSpecMixin: canonical POMDP spec composition helpers (strict A/B/C/D/E contract, B canonicalization)
├── _spec_generation.py    # _SpecGenerationMixin: GNN spec builders per model kind (continuous, nonstationary, structural, default)
├── _renderer_dispatch.py  # _RendererDispatchMixin: _invoke_renderer and _call_*_renderer methods
├── _support.py            # Cross-mixin typing contract (annotation-only declarations, no runtime members)
└── _documentation.py      # _DocumentationMixin: per-framework README generation
```

## Module Map

| Module | Owns |
| --- | --- |
| `__init__.py` | Facade: `POMDPRenderProcessor` assembly (`__init__.py:54`), `pomdp_to_gnn_spec` (`__init__.py:73`), `process_pomdp_for_frameworks` (`__init__.py:84`) |
| `_routes.py` | `RendererRoute` frozen dataclass (`_routes.py:27`) and `RENDERER_ROUTES` dispatch table for `pymdp`, `rxinfer`, `activeinference_jl`, `jax`, `discopy`, `pytorch`, `numpyro`, `stan`, `ngclearn` (`_routes.py:40-131`) |
| `_flow.py` | `_ProcessFlowMixin`: `__init__` shared state (`_flow.py:20`), `process_pomdp_for_all_frameworks` (`_flow.py:32`), `_process_single_framework` (`_flow.py:144`) |
| `_validation.py` | `_CompatibilityValidationMixin`: `_validate_pomdp_framework_compatibility` (`_validation.py:23`), spec/script state-space validation (`_validation.py:224`, `_validation.py:280`) |
| `_canonical.py` | `_CanonicalSpecMixin`: `_build_canonical_initialparameterization` (`_canonical.py:25`), factored composition (`_canonical.py:134`), B canonicalization (`_canonical.py:297`, `_canonical.py:317`) |
| `_spec_generation.py` | `_SpecGenerationMixin`: `_pomdp_to_gnn_spec` dispatcher (`_spec_generation.py:19`) plus continuous (`_spec_generation.py:142`), nonstationary (`_spec_generation.py:222`), and structural (`_spec_generation.py:322`) builders |
| `_renderer_dispatch.py` | `_RendererDispatchMixin`: route-driven `_call_framework_renderer` (`_renderer_dispatch.py:16`), `_invoke_renderer` skeleton (`_renderer_dispatch.py:42`), one `_call_*_renderer` per framework (`_renderer_dispatch.py:125-219`) |
| `_support.py` | `_POMDPProcessorSupportMixin`: annotation-only cross-mixin typing contract (`_support.py:24`) |
| `_documentation.py` | `_DocumentationMixin`: `_create_framework_documentation` writes a per-framework `README.md` (`_documentation.py:17`) |

## Mixin Architecture and Processing Flow

`POMDPRenderProcessor` is composed from six topic mixins in a fixed base order
(`__init__.py:54-61`): `_ProcessFlowMixin`, `_CompatibilityValidationMixin`,
`_CanonicalSpecMixin`, `_SpecGenerationMixin`, `_RendererDispatchMixin`,
`_DocumentationMixin`. Every mixin inherits
`_POMDPProcessorSupportMixin` (e.g. `_flow.py:19`), the annotation-only
contract in `_support.py` that keeps mypy green across composition
boundaries. Shared state (`base_output_dir`, `logger`, `framework_configs`) is
created only by `_ProcessFlowMixin.__init__` (`_flow.py:27-30`) from
[../framework_registry.py](../framework_registry.py) `get_pomdp_framework_configs`.

The per-model flow (`_flow.py:32` → `_flow.py:144`):

1. Resolve the framework list from `framework_configs` when unset (`_flow.py:51-52`).
2. Per framework: validate compatibility (`_flow.py:174`); unsupported model
   kinds are reported as `unsupported` and excluded from the success
   denominator (`_flow.py:113-122`).
3. Create the per-framework output directory (`_flow.py:193-194`).
4. Convert the POMDP space to a renderer GNN spec (`_flow.py:197`).
5. Invoke the renderer through the route table (`_flow.py:201`).
6. On success, write documentation (`_flow.py:207`) and collect code metrics
   via `count_code_metrics` (`_flow.py:216`).
7. Write `processing_summary.json` under the base output directory (`_flow.py:133-135`).

Dispatch is declarative: `RendererRoute` (`_routes.py:27`) binds each
framework to its renderer module, function, output suffix, and
validate/options/result/artifact modes; `_invoke_renderer`
(`_renderer_dispatch.py:42`) executes the shared invocation skeleton driven by
that route, so the `_call_*_renderer` wrappers (`_renderer_dispatch.py:125-219`)
are thin per-framework shims; a `_call_bnlearn_renderer` variant
(`_renderer_dispatch.py:169`) serves the non-routed bnlearn path.

## Canonical Spec Contract

The shared contract is `canonical_pomdp_v1`, defined in
[../pomdp_contract.py](../pomdp_contract.py) (`build_canonical_pomdp_spec` stamps
`spec["canonical_pomdp_schema"] = "canonical_pomdp_v1"` at
`pomdp_contract.py:682`); the mixin stamps the same schema marker when it
assembles a spec (`_spec_generation.py:116`). B is stored as
`(next_state, previous_state, action)`: `_canonicalise_time_indexed_B`
canonicalizes time-indexed B_t to that order (`_canonical.py:297-298`) and
`_canonicalise_factored_B` canonicalizes per-factor B
(`_canonical.py:317-321`). `_build_canonical_initialparameterization`
(`_canonical.py:25-29`) builds the strict canonical A/B/C/D/E contract used by
renderers, delegating to `build_canonical_pomdp_spec` when pre-canonical
matrices are absent (`_canonical.py:9`).

## Public Surface (facade contract)

Import rule: callers import only from the facade
(`from gnn.render.pomdp_processor import ...`). External call sites outside the
render package (e.g. analysis-side cross-framework execution) must use
`pomdp_to_gnn_spec` rather than reaching into
`POMDPRenderProcessor._pomdp_to_gnn_spec` (`__init__.py:76-81`). Leaf modules
(`_*.py`) and their `_*` members are private to the package.

Defined in this package:

| Name | Kind | Source |
| --- | --- | --- |
| `POMDPRenderProcessor` | class (six-mixin composition) | `__init__.py:54` |
| `pomdp_to_gnn_spec` | pure public conversion, no output writes | `__init__.py:73` |
| `process_pomdp_for_frameworks` | multi-framework convenience wrapper | `__init__.py:84` (documented at [../README.md:418](../README.md)) |

Re-exported from sibling modules of `gnn.render` (contract/typing surface,
`__init__.py:21-38`):

| Source module | Names | Cite |
| --- | --- | --- |
| `gnn.render.pomdp_contract` | `ModelKind`, `build_canonical_pomdp_spec`, `detect_pomdp_space_model_kind`, `detect_pomdp_space_model_kinds`, `unsupported_composition_reason`, `unsupported_nonstationary_reason` | `__init__.py:23-30` |
| `gnn.render.pomdp_math` | `_factor_action_counts`, `_is_kronecker_factorized_spec`, `_mixed_radix_digit`, `_normalise_columns`, `_normalise_prob_vector` | `__init__.py:31-37` |
| `gnn.render.framework_registry` | `get_pomdp_framework_configs` | `__init__.py:21` |
| `gnn.render.naming` | `safe_output_stem` | `__init__.py:22` |
| `gnn.utils.config_io.code_metrics` | `count_code_metrics` | `__init__.py:38` |

The `pomdp_math` private helpers are facade re-exports consumed by tests:
`_factor_action_counts` and `_is_kronecker_factorized_spec` are imported at
`tests/render/test_jax_factorized_pipeline.py:38-41`. Numeric/matrix helpers
live in [../pomdp_math.py](../pomdp_math.py); generic code-metrics counting
lives in `gnn.utils.config_io.code_metrics` (`__init__.py:8-9`).

## `_support.py` Cross-Mixin Typing Contract

`_support.py` exists because the composed class assembles six topic mixins,
and moved method bodies reference shared state and sibling-mixin methods that
mypy cannot infer from composition alone (`_support.py:5-9`). It declares the
shared cross-mixin surface of `POMDPRenderProcessor` under `TYPE_CHECKING`
(`_support.py:18`, `_support.py:27`):

- Shared state created by `_ProcessFlowMixin.__init__`: `logger`,
  `framework_configs` (`_support.py:29-30`).
- Sibling-mixin methods referenced across composition boundaries:
  validation (`_support.py:33-47`), canonical spec building and conversion
  (`_support.py:49-56`), documentation (`_support.py:58-64`), and renderer
  dispatch (`_support.py:66-72`).

Rule: this module stays annotation-only — it defines no runtime members
(`_support.py:11-12`); the real values and implementations are created by the
owning mixins (`_flow.py` state init; method bodies on their owning modules,
`_support.py:10-11`). Never add runtime code here; extending the composed
class surface means adding an annotation and the implementation on its owning
mixin.

## Provenance

This package was extracted in-place from the former monolithic
`render/processor.py`; the import sites now target `gnn.render.pomdp_processor`:

- `src/gnn/render/processor/parsing.py:108` imports `pomdp_to_gnn_spec`
- `src/gnn/render/processor/pipeline.py:72` imports `POMDPRenderProcessor`
- `src/gnn/render/rxinfer/rxinfer_renderer.py:64` imports `POMDPRenderProcessor`
- `src/gnn/analysis/rxinfer/cross_framework.py:884` imports `pomdp_to_gnn_spec`

## Testing & Verification

Test files that import this package (verified by grep at repo tip):

| Test | Import cite | Imports |
| --- | --- | --- |
| `tests/render/test_pomdp_renderer_regressions.py` | `:11` | `POMDPRenderProcessor` |
| `tests/render/test_composed_model_kinds.py` | `:27` | `POMDPRenderProcessor`, `pomdp_to_gnn_spec` |
| `tests/render/test_continuous_renderers.py` | `:29` | `pomdp_to_gnn_spec` |
| `tests/render/test_jax_factorized_pipeline.py` | `:38` | `_factor_action_counts`, `_is_kronecker_factorized_spec`, `pomdp_to_gnn_spec` |
| `tests/render/test_structural_model_kind.py` | `:37` | `POMDPRenderProcessor`, `pomdp_to_gnn_spec` |
| `tests/render/test_rxinfer_model_strategies.py` | `:28` | `pomdp_to_gnn_spec` |
| `tests/render/test_rxinfer_viz_log_contract.py` | `:24` | `POMDPRenderProcessor` |
| `tests/render/test_rxinfer_efe_correctness.py` | `:23` | `POMDPRenderProcessor` |
| `tests/render/test_rxinfer_multiagent_contract.py` | `:23` | `POMDPRenderProcessor` |
| `tests/render/test_stigmergic_multi_agent.py` | `:45` | `pomdp_to_gnn_spec` |
| `tests/render/test_ngclearn_renderer.py` | `:28` | `POMDPRenderProcessor`, `pomdp_to_gnn_spec` |
| `tests/render/test_render_contracts.py` | `:296` | `POMDPRenderProcessor` |
| `tests/analysis/test_rxinfer_cross_framework.py` | `:135` | `pomdp_to_gnn_spec` |
| `tests/gnn/test_pomdp_extractor_continuous.py` | `:10` | `POMDPRenderProcessor`, `pomdp_to_gnn_spec` |
| `tests/execute/test_discrete_models_pymdp.py` | `:59` | `POMDPRenderProcessor` |
| `tests/execute/test_pymdp_contracts.py` | `:39` | `POMDPRenderProcessor` |
| `tests/pipeline/test_pomdp_gridworld_cross_framework.py` | `:19` | `POMDPRenderProcessor` |

Run the regression and facade-contract coverage with:

```bash
uv run --extra dev python -m pytest \
    tests/render/test_pomdp_renderer_regressions.py \
    tests/render/test_composed_model_kinds.py \
    tests/render/test_continuous_renderers.py \
    tests/render/test_jax_factorized_pipeline.py \
    tests/render/test_structural_model_kind.py \
    tests/render/test_rxinfer_model_strategies.py \
    tests/render/test_rxinfer_viz_log_contract.py \
    tests/render/test_rxinfer_efe_correctness.py \
    tests/render/test_rxinfer_multiagent_contract.py \
    tests/render/test_stigmergic_multi_agent.py \
    tests/render/test_ngclearn_renderer.py \
    tests/render/test_render_contracts.py \
    tests/analysis/test_rxinfer_cross_framework.py \
    tests/gnn/test_pomdp_extractor_continuous.py \
    tests/execute/test_discrete_models_pymdp.py \
    tests/execute/test_pymdp_contracts.py \
    tests/pipeline/test_pomdp_gridworld_cross_framework.py \
    -q
```

---

**Last Updated**: 2026-09-28
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
