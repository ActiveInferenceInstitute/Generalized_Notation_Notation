#!/usr/bin/env python3
"""
POMDP state-space payload and error types for GNN extraction.

Mechanical extraction from ``gnn.extract.pomdp_extractor`` (M-01 band split):
this module holds the stdlib-only payload types — the canonical B tensor
semantic order constant, the error-collection mode contract, the structured
error type, the installed-version helper, :class:`POMDPStateSpace` (its
``to_dict`` mapping versioned via ``extraction_schema_version``), and
:func:`canonicalize_pomdp`. ``pomdp_extractor.py`` imports these names so
every consumer import path is unchanged.

This module imports only the standard library — nothing from the ``gnn``
package — so any sibling can import it first under any entry order.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

CANONICAL_B_ORDER = ["next_state", "previous_state", "action"]
"""Canonical B tensor semantic order: B[s', s, a] per pymdp 1.0.0
(control.py: B[f][s, v, u]; per-action slices column-stochastic over next_state)."""

ON_ERROR_MODES = ("lenient", "raise", "collect")
OnErrorMode = Literal["lenient", "raise", "collect"]


@dataclass
class GNNExtractionError(Exception):
    """Structured error emitted by the POMDP extractor.

    Local, hermetic type: structurally compatible with
    ``gnn.schema.GNNParseError`` (code/message/line + severity) so consumers can
    normalize across both surfaces, but deliberately not imported from there —
    probing ``from gnn.schema import GNNParseError`` shows the schema package core is
    import-light (ast/logging/re/dataclasses/typing only), yet routing through
    the ``gnn`` package ``__init__`` drags pipeline weight, and this module must
    stay importable headless. Line numbers are best-effort (relative to the
    enclosing GNN section, offset toward file-absolute when the section header
    is locatable).

    Codes: GNN-E002 (shape/orientation contradiction), GNN-E006 (parameter
    parse failure), GNN-E999 (unexpected extraction failure). Warning codes use
    the GNN-W* namespace.
    """

    code: str
    message: str
    line: Optional[int] = None
    section: Optional[str] = None

    @property
    def severity(self) -> str:
        """'warning' for GNN-W* codes, 'error' otherwise."""
        return "warning" if self.code.startswith("GNN-W") else "error"

    def __str__(self) -> str:
        location = f" (line {self.line})" if self.line is not None else ""
        section = f" [{self.section}]" if self.section else ""
        return f"{self.code}{section}{location}: {self.message}"


def _gnn_distribution_version() -> str:
    """Best-effort installed version of the generalized-notation-notation package."""
    try:
        from importlib.metadata import version

        return version("generalized-notation-notation")
    except Exception:  # noqa: BLE001 - any metadata failure degrades to "unknown"
        return "unknown"


@dataclass
class POMDPStateSpace:
    """Represents extracted POMDP state space information."""

    # Core dimensions
    num_states: int
    num_observations: int
    num_actions: int

    # Active Inference matrices and vectors
    A_matrix: Optional[List[List[float]]] = None  # Likelihood: P(o|s)
    B_matrix: Optional[List[List[List[float]]]] = None  # Transition: P(s'|s,a)
    C_vector: Optional[List[float]] = None  # Preferences over observations
    D_vector: Optional[List[float]] = None  # Prior beliefs over states
    E_vector: Optional[List[float]] = None  # Policy priors

    # State space variables
    state_variables: Optional[List[Dict[str, Any]]] = None
    observation_variables: Optional[List[Dict[str, Any]]] = None
    action_variables: Optional[List[Dict[str, Any]]] = None
    state_factors: Optional[List[Dict[str, Any]]] = None
    observation_modalities: Optional[List[Dict[str, Any]]] = None
    control_factors: Optional[List[Dict[str, Any]]] = None

    # Connections/relationships
    connections: Optional[List[Tuple[str, str, str]]] = (
        None  # (source, relation, target)
    )

    # Metadata
    model_name: Optional[str] = None
    model_annotation: Optional[str] = None
    gnn_section: Optional[str] = None  # raw ## GNNSection value (e.g. ActInfPOMDP)
    ontology_mapping: Optional[Dict[str, str]] = None
    num_timesteps: Optional[int] = None  # Simulation timesteps (from ModelParameters)
    model_parameters: Optional[Dict[str, Any]] = None
    matrices: Optional[Dict[str, Any]] = None
    matrix_provenance: Optional[Dict[str, Dict[str, Any]]] = None
    passive_model: bool = False
    adapter_notes: Optional[List[str]] = None
    initial_parameterization: Optional[Dict[str, Any]] = None
    # "discrete" (categorical POMDP/HMM) or "continuous" (linear-Gaussian
    # state-space model declared via F/H/Q/R + prior_mean/prior_cov).
    model_kind: str = "discrete"

    # Computed factor counts (bookkeeping excluded). A descriptor counts as a
    # factor unless its name matches *_prime (next-state/next-observation
    # aliases like s_prime/o_prime) or it is the policy symbol pi/π; the action
    # variable u DOES count as the control factor. The state_factors /
    # observation_modalities / control_factors lists keep ALL entries
    # (including bookkeeping, each tagged role='factor'|'bookkeeping'); only
    # these num_* counts exclude bookkeeping.
    num_state_factors: Optional[int] = None
    num_observation_modalities: Optional[int] = None
    num_control_factors: Optional[int] = None

    # Which _extract_dimensions priority level produced each core dimension:
    # {name: {"value": ..., "source": "ModelParameters|inferred_from_B_shape|
    # variable_dimensions|default"}}.
    dimension_provenance: Optional[Dict[str, Dict[str, Any]]] = None

    # Source sections retained verbatim for downstream semantic dispatch/audit.
    # Appended to preserve positional construction of existing payload fields.
    time: Optional[str] = None
    equations: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_states": self.num_states,
            "num_observations": self.num_observations,
            "num_actions": self.num_actions,
            "A_matrix": self.A_matrix,
            "B_matrix": self.B_matrix,
            "C_vector": self.C_vector,
            "D_vector": self.D_vector,
            "E_vector": self.E_vector,
            "state_variables": self.state_variables,
            "observation_variables": self.observation_variables,
            "action_variables": self.action_variables,
            "state_factors": self.state_factors,
            "observation_modalities": self.observation_modalities,
            "control_factors": self.control_factors,
            "connections": self.connections,
            "model_name": self.model_name,
            "model_annotation": self.model_annotation,
            "ontology_mapping": self.ontology_mapping,
            "num_timesteps": self.num_timesteps,
            "model_parameters": self.model_parameters,
            "matrices": self.matrices,
            "matrix_provenance": self.matrix_provenance,
            "passive_model": self.passive_model,
            "adapter_notes": self.adapter_notes,
            "initial_parameterization": self.initial_parameterization,
            "model_kind": self.model_kind,
            "gnn_version": _gnn_distribution_version(),
            "extraction_schema_version": "1.0.0",
            "num_state_factors": self.num_state_factors,
            "num_observation_modalities": self.num_observation_modalities,
            "num_control_factors": self.num_control_factors,
            "dimension_provenance": self.dimension_provenance,
            "time": self.time,
            "equations": self.equations,
        }


def canonicalize_pomdp(spec: POMDPStateSpace) -> POMDPStateSpace:
    """Return a NEW POMDPStateSpace with B in canonical (next, prev, action) order.

    Pure, stdlib-only copy: the input spec (and its B_matrix) is never
    mutated. Orientation is chosen from matrix_provenance['B']
    (detected_order/claimed_slice_convention) when decisive; a 3-D B stored
    as (action, previous_state, next_state) is transposed to
    (next_state, previous_state, action); canonical or ambiguous storage is
    copied unchanged. B shape/orientation provenance and matrices["B"] follow
    the transformed tensor, making repeated canonicalization idempotent.
    All unrelated fields are deep-copied as-is.
    """
    from copy import deepcopy

    canonical = deepcopy(spec)
    b_matrix = spec.B_matrix
    provenance = spec.matrix_provenance or {}
    b_meta = provenance.get("B") or {}
    stored_order = b_meta.get("detected_order") or (
        ["action", "previous_state", "next_state"]
        if b_meta.get("claimed_slice_convention") == "rows_previous_columns_next"
        else None
    )
    if (
        isinstance(b_matrix, (list, tuple))
        and len(self_shape := _shape_of(b_matrix)) == 3
        and stored_order
        and stored_order != list(CANONICAL_B_ORDER)
        and stored_order == ["action", "previous_state", "next_state"]
    ):
        # (action, prev, next) -> (next, prev, action):
        # canonical[n][p][a] = stored[a][p][n]
        canonical.B_matrix = [
            [
                [b_matrix[a][p][n] for a in range(self_shape[0])]
                for p in range(self_shape[1])
            ]
            for n in range(self_shape[2])
        ]
        canonical.matrix_provenance = deepcopy(provenance)
        canonical.matrix_provenance["B"] = {
            **b_meta,
            "original_detected_order": stored_order,
            "detected_order": list(CANONICAL_B_ORDER),
            "shape": _shape_of(canonical.B_matrix),
        }
        if canonical.matrices is not None and "B" in canonical.matrices:
            canonical.matrices["B"] = deepcopy(canonical.B_matrix)
    return canonical


def _shape_of(value: Any) -> List[int]:
    """Best-effort nested shape (module-level twin of _nested_shape)."""
    shape: List[int] = []
    current: Any = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape
