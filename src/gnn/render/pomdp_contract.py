#!/usr/bin/env python3
"""Canonical POMDP render contract shared by framework renderers.

This module defines the canonical POMDP specification types and the
``build_canonical_pomdp_spec()`` function that normalises GNN specs into
``canonical_pomdp_v1`` form with explicit ``A/B/C/D`` (and optional ``E``)
matrices and B stored as ``(next_state, previous_state, action)``.

TypedDict definitions (``CanonicalPomdpSpec``, ``InitialParameterization``,
``RxInferSimulationV1``) and the ``ModelKind`` enum provide typed contracts
for renderers and consumers.
"""

from __future__ import annotations

import re
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple, TypedDict, cast

import numpy as np

CANONICAL_B_ORDER = "next_state_previous_state_action"
# Only source decimal rounding drift may be corrected; arbitrary weights
# require the explicit demo adapter and never enter strict canonical models.
DECLARED_PROBABILITY_MASS_ATOL = 1e-4


class ModelKind(Enum):
    """Detected model structure from the GNN spec.

    FLAT — single-factor POMDP (the common case).
    FACTORED — multiple independent hidden state factors.
    HIERARCHICAL — multi-level state hierarchy.
    MULTI_AGENT — multiple coordinated agents.
    HYBRID — discrete and continuous families declared together
    (explicit A/B/C/D[/E] keys AND an F/H/Q/R or Gaussian-prior block);
    no backend renders the mix whole.
    STRUCTURAL — structural wrapper (no discrete or continuous
    parameterization; render-only / informational).
    LEARNING — parameter learning (Dirichlet priors, etc.).
    CONTINUOUS — continuous state/observation spaces.
    NONSTATIONARY — time-indexed (B_t) or regime-switched (B_regime +
    schedule) transitions. Precedence note: a time-varying discrete model
    is NONSTATIONARY, not STRUCTURAL — it keeps its discrete A/B/C/D[/E]
    machinery and dispatches to the switching-capable route; STRUCTURAL
    remains the no-parameterization blanket case.
    """

    FLAT = "flat"
    FACTORED = "factored"
    HIERARCHICAL = "hierarchical"
    MULTI_AGENT = "multi_agent"
    HYBRID = "hybrid"
    STRUCTURAL = "structural"
    LEARNING = "learning"
    CONTINUOUS = "continuous"
    NONSTATIONARY = "nonstationary"


class InitialParameterization(TypedDict, total=False):
    """The A/B/C/D/E matrices of a canonical POMDP."""

    A: List[List[float]]  # likelihood [obs, state]
    B: List[List[List[float]]]  # transition [next, prev, action]
    C: List[float]  # log-preferences over observations
    D: List[float]  # prior over initial states
    E: List[float]  # habit / policy prior over actions


class CanonicalPomdpSpec(TypedDict, total=False):
    """Output of ``build_canonical_pomdp_spec()`` — a normalised GNN spec."""

    model_name: str
    model_parameters: Dict[str, Any]
    initialparameterization: InitialParameterization
    initial_parameterization: InitialParameterization
    matrix_provenance: Dict[str, Any]
    structured_pomdp: Dict[str, Any]
    canonical_pomdp_schema: str


class RxInferSimulationV1(TypedDict, total=False):
    """The ``rxinfer_simulation_v1`` output schema from a rendered RxInfer script."""

    schema_version: str
    success: bool
    framework: str
    model_name: str
    num_timesteps: int
    observations: List[int]
    true_states: List[int]
    actions: List[int]
    beliefs: List[List[float]]
    expected_free_energy: List[float]
    expected_free_energy_convention: str
    efe_per_action: List[List[float]]
    variational_free_energy: List[float]  # per-iteration VFE trace
    vfe_per_iteration: List[float]  # alias for variational_free_energy
    policy_posterior: List[List[float]]
    model_parameters: Dict[str, Any]
    matrix_provenance: Dict[str, Any]
    runtime_metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    validation: Dict[str, Any]


def nested_shape(value: Any) -> List[int]:
    """Return the shape of a regular nested list-like value."""
    shape: List[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def normalise_vector(
    value: Any, *, name: str, allow_weights: bool = False
) -> List[float]:
    """Return a finite probability vector."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all() or np.any(vector < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    if not allow_weights and not np.isclose(
        total, 1.0, rtol=0, atol=DECLARED_PROBABILITY_MASS_ATOL
    ):
        raise ValueError(f"{name} probability mass must be one, got {total}")
    return cast(List[float], (vector / total).astype(float).tolist())


def normalise_matrix_columns(
    value: Any, *, name: str, allow_weights: bool = False
) -> List[List[float]]:
    """Return a 2-D matrix with columns summing to one."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {matrix.shape}")
    if min(matrix.shape) <= 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    output = matrix.copy()
    for column in range(output.shape[1]):
        total = float(output[:, column].sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"{name} column {column} has invalid probability mass")
        if not allow_weights and not np.isclose(
            total, 1.0, rtol=0, atol=DECLARED_PROBABILITY_MASS_ATOL
        ):
            raise ValueError(
                f"{name} column {column} probability mass must be one, got {total}"
            )
        output[:, column] /= total
    return cast(List[List[float]], output.astype(float).tolist())


def _declared_b_order(model_parameters: Dict[str, Any]) -> str:
    """Handle declared b order for internal callers."""
    order = str(
        model_parameters.get("b_tensor_order")
        or model_parameters.get("B_tensor_order")
        or model_parameters.get("transition_tensor_order")
        or ""
    ).lower()
    return order.replace("-", "_").replace(" ", "_")


def canonicalise_b_matrix(
    value: Any,
    *,
    num_states: int,
    num_actions: int,
    model_parameters: Dict[str, Any],
) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
    """Return B as ``(next_state, previous_state, action)``."""
    raw = np.asarray(value, dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("B must contain finite nonnegative probabilities")
    declared_order = _declared_b_order(model_parameters)
    supported_orders = {
        CANONICAL_B_ORDER,
        "next_previous_action",
        "next_prev_action",
        "states_next_states_previous_actions",
        "action_next_state_previous_state",
        "action_next_previous",
        "actions_next_previous",
        "action_previous_state_next_state",
        "action_previous_next",
        "actions_previous_next",
    }
    if declared_order and declared_order not in supported_orders:
        raise ValueError(f"Unsupported declared B tensor order: {declared_order}")
    source_order = "inferred"

    if raw.ndim == 2:
        if raw.shape != (num_states, num_states):
            raise ValueError(
                f"B must be ({num_states}, {num_states}) for passive dynamics, got {raw.shape}"
            )
        tensor = raw[:, :, np.newaxis]
        source_order = "next_state_previous_state"
    elif raw.ndim == 3:
        if declared_order in {
            CANONICAL_B_ORDER,
            "next_previous_action",
            "next_prev_action",
            "states_next_states_previous_actions",
        }:
            tensor = raw
            source_order = CANONICAL_B_ORDER
        elif declared_order in {
            "action_next_state_previous_state",
            "action_next_previous",
            "actions_next_previous",
        }:
            tensor = raw.transpose(1, 2, 0)
            source_order = "action_next_state_previous_state"
        elif declared_order in {
            "action_previous_state_next_state",
            "action_previous_next",
            "actions_previous_next",
        }:
            tensor = raw.transpose(2, 1, 0)
            source_order = "action_previous_state_next_state"
        elif raw.shape == (num_actions, num_states, num_states):
            tensor = raw.transpose(2, 1, 0)
            source_order = "action_previous_state_next_state"
        elif raw.shape == (num_states, num_states, num_actions):
            tensor = raw
            source_order = CANONICAL_B_ORDER
        elif raw.shape == (1, num_states, num_states):
            tensor = raw.transpose(2, 1, 0)
            source_order = "single_action_previous_state_next_state"
        else:
            raise ValueError(
                "B must be 2-D passive, canonical "
                f"({num_states}, {num_states}, {num_actions}), or action-first "
                f"({num_actions}, {num_states}, {num_states}); got {raw.shape}"
            )
    else:
        raise ValueError(f"B must be 2-D or 3-D, got shape {raw.shape}")

    if tensor.shape[0] != num_states or tensor.shape[1] != num_states:
        raise ValueError(
            f"Canonical B shape {tensor.shape} does not match {num_states} states"
        )
    if tensor.shape[2] not in {1, num_actions}:
        raise ValueError(
            f"Canonical B action dimension {tensor.shape[2]} does not match {num_actions}"
        )

    output = tensor.astype(np.float64).copy()
    for action in range(output.shape[2]):
        for previous_state in range(output.shape[1]):
            total = float(output[:, previous_state, action].sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "B[:, previous_state, action] must have positive finite mass "
                    f"for previous_state={previous_state}, action={action}"
                )
            if not np.isclose(total, 1.0, rtol=0, atol=DECLARED_PROBABILITY_MASS_ATOL):
                raise ValueError(f"B probability mass must be one, got {total}")
            output[:, previous_state, action] /= total

    return (
        cast(List[List[List[float]]], output.astype(float).tolist()),
        {
            "source_order": source_order,
            "canonical_order": CANONICAL_B_ORDER,
            "shape": list(output.shape),
            "orientation_transformed": source_order != CANONICAL_B_ORDER,
            "normalized": bool(not np.array_equal(output, tensor)),
        },
    )


def _first_int(*values: Any, required: bool = True) -> int:
    """Handle first int for internal callers."""
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    if required:
        raise ValueError("required positive integer dimension is missing")
    return 0


def _infer_num_actions_from_b(value: Any, *, num_states: int) -> int:
    """Infer the control dimension from a passive or controlled B matrix."""
    shape = nested_shape(value)
    if len(shape) == 2:
        return 1 if shape == [num_states, num_states] else 0
    if len(shape) != 3:
        return 0
    if shape[0] == num_states and shape[1] == num_states:
        return shape[2]
    if shape[1] == num_states and shape[2] == num_states:
        return shape[0]
    return 0


def _require_keys(mapping: Dict[str, Any], keys: Iterable[str]) -> None:
    """Handle require keys for internal callers."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required initialparameterization keys: {missing}")


def _is_active_inference_matrix_key(key: str) -> bool:
    """Return whether an InitialParameterization key names a matrix/vector contract."""
    return key in {"A", "B", "C", "D", "E"} or any(
        key.startswith(f"{prefix}_") for prefix in ("A", "B", "C", "D", "E")
    )


_AGENT_MATRIX_KEY = re.compile(r"^[ABCDE]_agent\d+", re.IGNORECASE)
_AGENT_CONTINUOUS_KEY = re.compile(
    r"^(F|H|Q|R|prior_mean|prior_cov|goal_mean|control_gain)_agent[1-9]\d*$"
)
_LEVEL_MATRIX_KEY = re.compile(r"^[ABCDE]_level\d+", re.IGNORECASE)
_DIRICHLET_PRIOR_KEY = re.compile(r"^dirichlet_[ABCDE]$", re.IGNORECASE)
_CONTINUOUS_PARAM_KEYS = frozenset({"F", "H", "Q", "R"})
_FACTOR_CONTINUOUS_KEY = re.compile(
    r"^(F|H|Q|R|prior_mean|prior_cov)_f\d+$", re.IGNORECASE
)


_TIME_INDEXED_MATRIX_KEY = re.compile(r"^[ABCDE]_t\d*$", re.IGNORECASE)
_REGIME_SWITCHED_MATRIX_KEY = re.compile(r"^[ABCDE]_regime\d*$", re.IGNORECASE)
_NONSTATIONARY_SCHEDULE_KEYS = frozenset(
    {"b_regime_schedule", "regime_schedule", "b_schedule", "b_t_schedule"}
)


def _structured_matrix_keys(gnn_spec: Dict[str, Any]) -> List[str]:
    """Return raw (pre-composition) matrix key names from the spec.

    Per-level / per-agent matrix declarations survive canonicalization only
    inside ``structured_pomdp['matrices']`` — the top-level
    ``initialparameterization`` holds the composed joint A/B/C/D.
    """
    structured = gnn_spec.get("structured_pomdp")
    if not isinstance(structured, dict):
        return []
    matrices = structured.get("matrices")
    if not isinstance(matrices, dict):
        return []
    return [str(key) for key in matrices]


_KIND_PRECEDENCE: Tuple[ModelKind, ...] = (
    ModelKind.MULTI_AGENT,
    ModelKind.HYBRID,
    ModelKind.HIERARCHICAL,
    ModelKind.CONTINUOUS,
    ModelKind.NONSTATIONARY,
    ModelKind.LEARNING,
    ModelKind.FACTORED,
    ModelKind.STRUCTURAL,
    ModelKind.FLAT,
)


def detect_model_kinds(gnn_spec: Dict[str, Any]) -> frozenset[ModelKind]:
    """Detect the full composed model-kind set from a GNN spec — structurally.

    The composed counterpart of :func:`detect_model_kind`: every typed-field
    predicate is evaluated independently and ALL matching kinds are returned,
    so a spec that declares both a linear-Gaussian parameterization and
    multiple agents classifies as ``{CONTINUOUS, MULTI_AGENT}`` instead of
    silently collapsing to the first match of the precedence chain. Dispatch
    consumers render a single family at most; a non-singleton set therefore
    means the spec's families must be refused with an explicit
    unsupported-composition receipt (or rendered on every matching family)
    rather than silently rendered as the winner alone.
    ``HYBRID`` is the discrete+continuous family-mix case (explicit
    A/B/C/D[/E] contract keys declared alongside an F/H/Q/R or
    Gaussian-prior block); it is a composition like any other, so dispatch
    refuses it via the unsupported-composition receipt.

    Detection reads ONLY typed fields: the ``gnn_section`` value (the raw
    ``## GNNSection`` header, propagated by the extractor), declared matrix
    key patterns in ``structured_pomdp['matrices']``, explicit agent counts,
    and explicit ``model_parameters`` keys. Free-text scanning of the spec
    is deliberately gone: prose in a ModelName or annotation must never
    change how a model renders.

    Semantics shared with :func:`detect_model_kind`:

    - an explicit ``model_kind == "structural"`` stamp is the producer's
      authoritative classification and returns ``{STRUCTURAL}``;
    - when no family predicate matches, the set is ``{FLAT}`` for a spec
      with at least one discrete ``A/B/C/D[/E]`` contract key, else
      ``{STRUCTURAL}`` (the no-parameterization blanket case).
    """
    # An explicit structural stamp wins (mirrors is_continuous_spec's
    # ``model_kind == "continuous"`` check for the producer-classified case):
    # the producer already classified this spec — e.g. pomdp_processor's
    # structural spec view — so do not second-guess it with predicates.
    if gnn_spec.get("model_kind") == "structural":
        return frozenset({ModelKind.STRUCTURAL})
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization", {}
    )
    if not isinstance(initial, dict):
        raise ValueError(
            f"initialparameterization must be a mapping, got {type(initial).__name__}"
        )
    initial_keys = [str(key) for key in initial]
    matrix_keys = _structured_matrix_keys(gnn_spec)
    all_keys = initial_keys + matrix_keys
    section = str(gnn_spec.get("gnn_section") or "").lower()
    model_params = gnn_spec.get("model_parameters", {})
    if not isinstance(model_params, dict):
        model_params = {}

    kinds: set[ModelKind] = set()

    # Multi-agent: an explicit agent count > 1 or per-agent matrix keys.
    nr_agents = initial.get("nr_agents", model_params.get("nr_agents", 1))
    try:
        nr_agents = int(nr_agents)
    except (TypeError, ValueError):
        nr_agents = 1
    if nr_agents > 1 or any(
        _AGENT_MATRIX_KEY.match(key) or _AGENT_CONTINUOUS_KEY.fullmatch(key)
        for key in all_keys
    ):
        kinds.add(ModelKind.MULTI_AGENT)

    # Hybrid: discrete A/B/C/D[/E] contract keys declared alongside an
    # explicit linear-Gaussian parameterization (F/H/Q/R or a Gaussian
    # prior) — a family mix no backend renders whole, refused at dispatch
    # with an unsupported-composition receipt. The contract-key predicate
    # matches discrete A–E keys only, so a flat continuous spec
    # (F/H/Q/R with no A–E key) never classifies hybrid.
    if any(_is_active_inference_matrix_key(key) for key in all_keys) and (
        _CONTINUOUS_PARAM_KEYS.issubset(set(initial_keys))
        or {"prior_mean", "prior_cov"}.issubset(set(initial_keys))
    ):
        kinds.add(ModelKind.HYBRID)

    # Hierarchical: declared section or per-level matrix keys.
    if "hierarchical" in section or any(
        _LEVEL_MATRIX_KEY.match(key) for key in all_keys
    ):
        kinds.add(ModelKind.HIERARCHICAL)

    # Continuous: declared section or an explicit continuous
    # parameterization (F/H/Q/R system matrices or a Gaussian prior).
    if (
        "continuous" in section
        or _CONTINUOUS_PARAM_KEYS.issubset(set(initial_keys))
        or {"prior_mean", "prior_cov"}.issubset(set(initial_keys))
        # Per-factor continuous keys from a factored LGSSM (F_fN/H_fN/...)
        # satisfy the CONTINUOUS family the same way flat F/H/Q/R do —
        # without this, a per-factor spec classifies {FACTORED} only and
        # the generic discrete path demands static A/B/C/D keys.
        or any(_FACTOR_CONTINUOUS_KEY.fullmatch(str(key)) for key in initial_keys)
        or any(_AGENT_CONTINUOUS_KEY.fullmatch(str(key)) for key in initial_keys)
    ):
        kinds.add(ModelKind.CONTINUOUS)

    # Nonstationary: time-indexed (B_t) or regime-switched (B_regime)
    # transition keys, an explicit schedule parameter, or a declared
    # nonstationary GNN section. Precedence note: a time-varying discrete
    # model is NONSTATIONARY, not STRUCTURAL — it keeps its discrete
    # A/B/C/D[/E] machinery and dispatches to the switching-capable
    # route (pymdp), while STRUCTURAL stays the no-parameterization
    # blanket case.
    if (
        "nonstationary" in section
        or any(_TIME_INDEXED_MATRIX_KEY.match(key) for key in all_keys)
        or any(_REGIME_SWITCHED_MATRIX_KEY.match(key) for key in all_keys)
        or any(key in model_params for key in _NONSTATIONARY_SCHEDULE_KEYS)
    ):
        kinds.add(ModelKind.NONSTATIONARY)

    # Learning: declared section or explicit Dirichlet prior parameter keys
    # (a prose mention of "Dirichlet" in an annotation must not reroute).
    if "learning" in section or any(
        _DIRICHLET_PRIOR_KEY.match(key) for key in initial_keys
    ):
        kinds.add(ModelKind.LEARNING)

    # Factored: explicit num_factors declaration.
    num_factors = model_params.get("num_factors", 1)
    try:
        num_factors = int(num_factors)
    except (TypeError, ValueError):
        num_factors = 1
    if num_factors > 1:
        kinds.add(ModelKind.FACTORED)

    if not kinds:
        # No family predicate matched: STRUCTURAL is the blanket/wrapper case
        # (declares boundary structure only — no categorical A/B/C/D[/E]
        # contract key); a spec with at least one contract key falls through
        # to FLAT. Render-only / informational: structural wrappers are never
        # forced through the discrete render path.
        if any(
            _is_active_inference_matrix_key(key) or _DIRICHLET_PRIOR_KEY.match(key)
            for key in all_keys
        ):
            kinds.add(ModelKind.FLAT)
        else:
            kinds.add(ModelKind.STRUCTURAL)

    return frozenset(kinds)


def detect_model_kind(gnn_spec: Dict[str, Any]) -> ModelKind:
    """Single-winner kind: the max-precedence member of detect_model_kinds.

    Stable classification used by per-kind dispatch
    (precedence MULTI_AGENT > HYBRID > HIERARCHICAL > CONTINUOUS >
    NONSTATIONARY >
    LEARNING > FACTORED > STRUCTURAL > FLAT). A composed spec — e.g.
    continuous
    F/H/Q/R parameters plus ``nr_agents > 1`` — classifies to its most
    specific kind here and to the full set under :func:`detect_model_kinds`;
    dispatch consumers must consult the kind set so a composed spec is
    refused or rendered on every matching family, never silently rendered
    as the winner alone.
    """
    kinds = detect_model_kinds(gnn_spec)
    for kind in _KIND_PRECEDENCE:
        if kind in kinds:
            return kind
    raise AssertionError(  # pragma: no cover - kinds is never empty
        f"model-kind detection returned an empty kind set: {kinds}"
    )


def unsupported_composition_reason(kinds: frozenset[ModelKind]) -> str:
    """Receipt reason for a composed spec that no backend renders whole.

    The single stable ``unsupported-composition:`` prefix is the grep anchor
    for dispatch receipts: the spec declares the linear-Gaussian family
    alongside other families, so rendering the single winner would silently
    drop the rest.
    """
    others = [
        kind.value
        for kind in _KIND_PRECEDENCE
        if kind in kinds and kind is not ModelKind.CONTINUOUS
    ]
    return (
        "unsupported-composition: "
        + " × ".join(["continuous", *others])
        + " — the spec declares the linear-Gaussian (F/H/Q/R) family "
        "alongside additional model families; no framework renders the "
        "composition whole, so it is refused rather than silently rendered "
        "as one family with the other dropped"
    )


def unsupported_nonstationary_reason(kinds: frozenset[ModelKind]) -> str:
    """Receipt reason for a nonstationary spec a backend cannot express.

    The single stable ``unsupported-nonstationary:`` prefix is the grep
    anchor for dispatch receipts: the spec declares time-indexed or
    regime-switched transitions, and a backend that renders one static
    transition tensor would silently drop the time variation instead of
    refusing.
    """
    others = [
        kind.value
        for kind in _KIND_PRECEDENCE
        if kind in kinds and kind is not ModelKind.NONSTATIONARY
    ]
    return (
        "unsupported-nonstationary: "
        + " × ".join(["nonstationary", *others])
        + " — the spec declares time-indexed (B_t) or regime-switched "
        "(B_regime + b_regime_schedule) transitions; this backend renders "
        "a single static transition tensor and cannot express time "
        "variation, so it is refused rather than silently rendered with "
        "static dynamics"
    )


def detect_pomdp_space_model_kind(pomdp_space: Any) -> ModelKind:
    """Classify an extracted POMDPStateSpace through the shared taxonomy.

    Mirrors the extractor's own view (``initial_parameterization``,
    ``matrices``, ``gnn_section``, ``model_parameters``) into the spec shape
    ``detect_model_kind`` reads, so the render pipeline and direct spec
    callers agree on one classification — in particular, a structural
    blanket wrapper is STRUCTURAL, never a discrete POMDP.
    """
    return detect_model_kind(
        {
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "initialparameterization": getattr(
                pomdp_space, "initial_parameterization", None
            )
            or {},
            "model_parameters": getattr(pomdp_space, "model_parameters", None) or {},
            "structured_pomdp": {
                "matrices": getattr(pomdp_space, "matrices", None) or {},
            },
        }
    )


def detect_pomdp_space_model_kinds(pomdp_space: Any) -> frozenset[ModelKind]:
    """Composed kind set for an extracted POMDPStateSpace.

    Mirrors the extractor's own view into the spec shape
    :func:`detect_model_kinds` reads, so the render pipeline can refuse
    compositions (e.g. continuous × multi-agent) with an explicit receipt
    instead of silently rendering the single-winner family.
    """
    return detect_model_kinds(
        {
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "initialparameterization": getattr(
                pomdp_space, "initial_parameterization", None
            )
            or {},
            "model_parameters": getattr(pomdp_space, "model_parameters", None) or {},
            "structured_pomdp": {
                "matrices": getattr(pomdp_space, "matrices", None) or {},
            },
        }
    )


def build_canonical_pomdp_spec(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copied GNN spec with canonical POMDP matrices and provenance."""
    spec = deepcopy(gnn_spec)
    initial = spec.get("initialparameterization") or spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("GNN spec requires initialparameterization")
    _require_keys(initial, ("A", "B", "C", "D"))

    model_parameters = dict(spec.get("model_parameters") or {})
    num_states = _first_int(
        model_parameters.get("num_hidden_states"),
        model_parameters.get("num_states"),
        spec.get("num_states"),
        len(initial.get("D") or []),
    )
    num_observations = _first_int(
        model_parameters.get("num_obs"),
        model_parameters.get("num_observations"),
        spec.get("num_observations"),
        len(initial.get("A") or []),
    )
    num_actions = _first_int(
        model_parameters.get("num_actions"),
        model_parameters.get("num_controls"),
        spec.get("num_actions"),
        len(initial.get("E") or []),
        _infer_num_actions_from_b(initial["B"], num_states=num_states),
    )

    canonical_b, b_provenance = canonicalise_b_matrix(
        initial["B"],
        num_states=num_states,
        num_actions=num_actions,
        model_parameters=model_parameters,
    )
    action_count = len(canonical_b[0][0]) if canonical_b else num_actions

    canonical_initial: Dict[str, Any] = {
        "A": normalise_matrix_columns(initial["A"], name="A"),
        "B": canonical_b,
        "C": np.asarray(initial["C"], dtype=np.float64)
        .reshape(-1)
        .astype(float)
        .tolist(),
        "D": normalise_vector(initial["D"], name="D"),
    }
    if len(canonical_initial["A"]) != num_observations or any(
        len(row) != num_states for row in canonical_initial["A"]
    ):
        raise ValueError("A shape does not match declared observation/state dimensions")
    if (
        len(canonical_initial["D"]) != num_states
        or len(canonical_initial["C"]) != num_observations
    ):
        raise ValueError(
            "C/D lengths do not match declared observation/state dimensions"
        )
    if not np.isfinite(canonical_initial["C"]).all():
        raise ValueError("C must contain finite preferences")
    if "E" in initial:
        canonical_initial["E"] = normalise_vector(initial["E"], name="E")
        if len(canonical_initial["E"]) != action_count:
            raise ValueError("E length does not match declared action count")
    for key, value in initial.items():
        if key not in canonical_initial and not _is_active_inference_matrix_key(key):
            canonical_initial[key] = deepcopy(value)

    model_parameters.update(
        {
            "num_hidden_states": num_states,
            "num_obs": num_observations,
            "num_actions": action_count,
            "b_tensor_order": CANONICAL_B_ORDER,
        }
    )
    spec["model_parameters"] = model_parameters
    spec["initialparameterization"] = canonical_initial
    spec["initial_parameterization"] = canonical_initial

    provenance = dict(spec.get("matrix_provenance") or {})
    existing_b = dict(provenance.get("B") or {})
    provenance["B"] = {
        **existing_b,
        **b_provenance,
        "derived": existing_b.get("derived", False) or b_provenance["normalized"],
    }
    for key in ("A", "C", "D", "E"):
        if key in canonical_initial and key not in provenance:
            provenance[key] = {
                "source": "initialparameterization",
                "shape": nested_shape(canonical_initial[key]),
                "derived": False,
            }
        if key in canonical_initial and not np.array_equal(
            np.asarray(initial[key]),
            np.asarray(canonical_initial[key]).reshape(np.asarray(initial[key]).shape),
        ):
            provenance[key] = {
                **provenance[key],
                "derived": True,
                "normalization": "declared_precision_mass_correction",
                "source_values": deepcopy(initial[key]),
            }
    spec["matrix_provenance"] = provenance

    structured = dict(spec.get("structured_pomdp") or {})
    structured["matrix_provenance"] = provenance
    structured["canonical_b_order"] = CANONICAL_B_ORDER
    spec["structured_pomdp"] = structured
    spec["canonical_pomdp_schema"] = "canonical_pomdp_v1"
    return spec
