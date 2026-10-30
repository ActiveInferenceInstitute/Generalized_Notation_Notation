# Analysis/Complexity — Static Complexity Estimator

## Module Overview

**Purpose**: Static, deterministic per-backend complexity estimation for parsed GNN models. A parsed model (or spec-file path) becomes a pinned `gnn.complexity_estimate/v1` receipt: structure statistics, detected model kinds, and one ESTIMATE-labeled asymptotic bound row per backend.

**Pipeline Step**: none (pure library; consumed by the wave-8 CLI/benchmark lanes)

**Category**: Model Analysis / Static Complexity Estimation

**Status**: New (wave 8, lane W8-A)

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-25

---

## Core Functionality

### Primary Responsibilities
1. Compute deterministic structure statistics for a parsed model: variable/edge counts, per-factor arities, total and max state-space dims, declared discrete/continuous variable counts, and the time block (type, discretization, horizon, dynamic flag).
2. Classify the model via the render contract's `detect_model_kinds` (fed `model.to_dict()`; never re-implemented here).
3. Resolve symbolic dimensions through the type_checker content layer (`GNNTypeChecker.validate_content`), so variable-backed dims like `G[pi]` count their resolved size and the generic parser's symbolic→1 degradation never hides size.
4. Emit one bound row per backend from the static registry in `bounds.py`, in fixed backend order, with explicit numeric drivers.

### Non-Goals
- No empirical measurement (wall time / RSS belong to the wave-8 measurement lanes).
- No CLI, no pipeline step, no website surfacing (separate lanes).
- No edits to `type_checker/estimation/**` (the generic tier stays as is), `execute/**`, `cli/**`, `website/**`, or `render/pomdp_contract.py` (import only).

---

## API Reference

### `estimate_model_complexity(model_or_path) -> dict`
**Description**: Build the receipt for a `GNNInternalRepresentation` (from `gnn.parsers.parse_gnn_file_structured(file_path).model`) or a filesystem path.

**Receipt contract (PINNED cross-lane, emit exactly these keys)**:
```json
{"receipt_type": "gnn.complexity_estimate/v1",
 "model": {"name": str, "source_sha256": str, "path": str},
 "structure": {"variable_count": int, "edge_count": int, "factor_arities": [int],
   "total_state_space_dim": int, "max_variable_dim": int,
   "discrete_var_count": int, "continuous_var_count": int,
   "time": {"time_type": str, "discretization": str, "horizon": int|"Unbounded", "is_dynamic": bool}},
 "model_kinds": [str],
 "per_backend": [{"framework": str, "applicable": bool, "family": str, "asymptotic": str,
   "complexity_class": str, "drivers": {...}, "notes": str}],
 "estimator_version": "1"}
```

**Semantics**:
- `model.source_sha256`: sha256 of the raw file bytes (path form) or of the canonical sorted-key JSON of `model.to_dict()` (object form). `model.path` is `""` for the object form.
- `structure.total_state_space_dim` / `max_variable_dim`: sum/max of per-variable resolved dim products from the type_checker layer; fallback to parse-object dims when no raw content is available.
- `structure.time.horizon`: int when numeric; `"Unbounded"` when `None`/unbounded marker; a symbolic string (e.g. `"T"`) is kept verbatim and treated as non-numeric downstream.
- `model_kinds`: sorted `ModelKind.value` strings from `gnn.render.pomdp_contract.detect_model_kinds`.
- `per_backend`: one row per backend in `bounds.BACKEND_ORDER` (pymdp, rxinfer, discopy, activeinference_jl, jax, numpyro, pytorch, ngclearn, lean, stan, bnlearn). `jax` resolves its family by kinds (kronecker-factorized for discrete, dense-LGSSM for continuous). Inapplicable rows keep the primary family/asymptotic with `applicable: false` and empty `drivers`.

### `to_json_text(receipt) -> str`
**Description**: Stable JSON serialization (`sort_keys=True`, fixed indent) for the CLI lane.

---

## Bounds Registry (`bounds.py`)

| Framework | Family | Applicability | Asymptotic (ESTIMATE) |
|---|---|---|---|
| pymdp | exact-factorized | discrete kinds, not continuous/hybrid | O(T * prod_f s_f * prod_m o_m * a) |
| rxinfer | exact-dense-LGSSM | continuous kinds | O(T * d^3) time, O(d^2) memory |
| discopy | categorical-composition | discrete kinds, not continuous/hybrid | O(sum_f s_f + E) |
| activeinference_jl | exact-factorized | discrete kinds, not continuous/hybrid | O(T * prod_f s_f * prod_m o_m * a) |
| jax | exact-factorized / exact-dense-LGSSM | discrete / continuous kinds | O(T * sum_f s_f * o * a) / O(T * d^3) |
| numpyro | sampling | always | O(per_sample_cost) per sample |
| pytorch | sampling | always | O(per_sample_cost) per sample |
| ngclearn | sampling | always | O(per_sample_cost) per sample |
| lean | verification | always | class-only (no numeric bound) |
| stan | sampling | always | O(per_sample_cost) per sample |
| bnlearn | structure-learning | always (render-only, registry-gated) | O(2^N_v * score_cost) |

Non-numeric horizon (`"Unbounded"` or symbolic): exact-factorized and LGSSM bounds degrade to their per-step form and the asymptotic string says `horizon Unbounded: no numeric total bound`. Sampling bounds are horizon-agnostic (reported per sample). `bnlearn` is the render-only registry-gated backend (`render/framework_registry.py` bnlearn entry).

**Honesty rule**: every asymptotic is labeled `[ESTIMATE]`; drivers carry only computed numbers (horizon, state-space totals/max, joint continuous dim, max factor arity, declared agent count, declared regime tensor count, edge/variable counts). The agent count mirrors multi-agent kind detection: explicit `nr_agents` then `num_agents` (initialparameterization first, then model_parameters), else the highest per-agent matrix index (`A_agent2` -> 2); 1 when no agent evidence exists. Nothing is invented; the receipt is an argument, not a measurement.

---

## Conventions
- Pure stdlib (math, hashlib, json, re, pathlib, dataclasses) + existing GNN modules only; no jax/pymdp/executor imports.
- Deterministic: sorted kind keys, fixed backend order, stable JSON.
- Tests: `tests/analysis/complexity/test_complexity_estimator.py` (pure-unit, deterministic, zero-skip; committed exemplars are asserted to exist, so a missing exemplar FAILS — it never passes vacuously).

## Verification
Run `uv run --extra dev python -m pytest tests/analysis/complexity/test_complexity_estimator.py -q`.

## Custody note
New `src/gnn/**/*.py` files do not drift the fep_lean source-pin: the pinned
GNN roster is the declared notation contract, so implementation-tree additions
are not owners and need no re-seal. Emitted-document freshness is enforced at
run time by the bridge's `emit --check`. See the
[paired-revision guide](../../../../docs/development/fep_lean_paired_revision.md).
