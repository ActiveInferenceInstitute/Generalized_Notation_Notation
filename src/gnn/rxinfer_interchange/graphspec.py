"""Validated daf-jev GraphSpec types, canonical ordering and JSON interchange."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping

logger = logging.getLogger(__name__)


GRAPH_SPEC_FORMAT: Final[str] = "dafjev.bayesnet/1"


_ALLOWED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"format", "variables", "edges", "cpts", "jev_factors"}
)


_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


_ROW_SUM_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class GraphVariable:
    """One discrete variable of a GraphSpec Bayes net."""

    key: str
    description: str
    states: tuple[str, ...]


@dataclass(frozen=True)
class GraphEdge:
    """One directed edge ``parent -> child``."""

    parent: str
    child: str


@dataclass(frozen=True)
class GraphCPT:
    """Conditional probability table for one child variable.

    ``rows`` holds ``(assignment, probabilities)`` pairs where
    ``assignment`` maps each parent key to a state label and
    ``probabilities`` is ordered by the child's state list. Rows are
    stored in the canonical parent-assignment odometer order.
    """

    child: str
    parents: tuple[str, ...]
    rows: tuple[tuple[Mapping[str, str], tuple[float, ...]], ...]


@dataclass(frozen=True)
class GraphSpec:
    """A validated discrete Bayes net in GraphSpec interchange shape.

    ``jev_factors`` is the RESERVED 'within' integration field (see the
    module docstring): Jev-derived factors preserved verbatim, not used
    by the emitter. Construct via :func:`load_graphspec`,
    :func:`parse_gnn_subset`, or directly (direct construction runs
    :meth:`validate`).
    """

    variables: tuple[GraphVariable, ...]
    edges: tuple[GraphEdge, ...]
    cpts: Mapping[str, GraphCPT]
    jev_factors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    # -- lookups ---------------------------------------------------------

    def variable(self, key: str) -> GraphVariable:
        for var in self.variables:
            if var.key == key:
                return var
        raise ValueError(
            f"GraphSpec variables: unknown variable key {key!r} "
            f"(known keys: {[v.key for v in self.variables]!r})"
        )

    def parents_of(self, key: str) -> tuple[str, ...]:
        return tuple(edge.parent for edge in self.edges if edge.child == key)

    def children_of(self, key: str) -> tuple[str, ...]:
        return tuple(edge.child for edge in self.edges if edge.parent == key)

    def _states_map(self) -> dict[str, tuple[str, ...]]:
        return {var.key: var.states for var in self.variables}

    def topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm with insertion-order tiebreak (deterministic)."""
        keys = [v.key for v in self.variables]
        parent_map = {
            key: tuple(e.parent for e in self.edges if e.child == key) for key in keys
        }
        return _topological_order(keys, parent_map)

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Re-check internal consistency; raise ``ValueError`` on violation."""
        keys = [var.key for var in self.variables]
        if not keys:
            raise ValueError("GraphSpec variables: at least one variable is required")
        seen: set[str] = set()
        for var in self.variables:
            if not _KEY_RE.match(var.key):
                raise ValueError(
                    f"GraphSpec variables: invalid key {var.key!r} "
                    "(must match [A-Za-z_][A-Za-z0-9_-]*)"
                )
            if var.key in seen:
                raise ValueError(f"GraphSpec variables: duplicate key {var.key!r}")
            seen.add(var.key)
            if len(var.states) < 2:
                raise ValueError(
                    f"GraphSpec variable {var.key}: needs at least 2 states, "
                    f"got {list(var.states)!r}"
                )
            if len(set(var.states)) != len(var.states):
                raise ValueError(
                    f"GraphSpec variable {var.key}: duplicate state labels "
                    f"{list(var.states)!r}"
                )
            for state in var.states:
                if not isinstance(state, str) or not state:
                    raise ValueError(
                        f"GraphSpec variable {var.key}: state labels must be "
                        f"non-empty strings, got {state!r}"
                    )
            if not isinstance(var.description, str):
                raise ValueError(
                    f"GraphSpec variable {var.key}: description must be a "
                    f"string, got {type(var.description).__name__}"
                )

        # Edge structure: unknown endpoints, self-loops, duplicates, cycles.
        _validate_edges_and_order(keys, self.edges)

        states_map = self._states_map()
        parent_map = {v.key: self.parents_of(v.key) for v in self.variables}
        if set(self.cpts) != seen:
            missing = sorted(seen - set(self.cpts))
            extra = sorted(set(self.cpts) - seen)
            raise ValueError(
                f"GraphSpec cpts: CPT keys must equal the variable keys "
                f"(missing: {missing!r}, unexpected: {extra!r})"
            )
        for child in self.variables:  # insertion order = variables order
            cpt = self.cpts[child.key]
            expected_parents = parent_map[child.key]
            if tuple(cpt.parents) != expected_parents:
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: parents must equal the "
                    f"graph parents in edge order {list(expected_parents)!r}, "
                    f"got {list(cpt.parents)!r}"
                )
            if cpt.child != child.key:
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: child field "
                    f"{cpt.child!r} does not match the mapping key"
                )
            canonical = _canonical_assignments(
                expected_parents,
                [states_map[p] for p in expected_parents],
            )
            if len(cpt.rows) != len(canonical):
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: expected {len(canonical)} "
                    f"rows (one per parent assignment), got {len(cpt.rows)}"
                )
            for i, ((assignment, probs), expected) in enumerate(
                zip(cpt.rows, canonical)
            ):
                if dict(assignment) != dict(expected):
                    raise ValueError(
                        f"GraphSpec cpts[{child.key}]: row {i} assignment "
                        f"{dict(assignment)!r} violates the canonical order "
                        f"(expected {dict(expected)!r}; rows must be in "
                        "parent-assignment odometer order, first parent "
                        "slowest)"
                    )
                for parent, label in assignment.items():
                    if label not in states_map[parent]:
                        raise ValueError(
                            f"GraphSpec cpts[{child.key}]: row {i} unknown "
                            f"state {label!r} for parent {parent!r} "
                            f"(states: {list(states_map[parent])!r})"
                        )
                if len(probs) != len(child.states):
                    raise ValueError(
                        f"GraphSpec cpts[{child.key}]: row {i} has "
                        f"{len(probs)} probabilities, expected "
                        f"{len(child.states)} (child states: "
                        f"{list(child.states)!r})"
                    )
                _check_probabilities(child.key, i, probs)

        for factor in self.jev_factors:
            if not isinstance(factor, Mapping) or not all(
                isinstance(k, str) for k in factor
            ):
                raise ValueError(
                    "GraphSpec jev_factors (RESERVED field): entries must be "
                    f"objects with string keys, got {factor!r}"
                )

    # -- interchange -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize to the pinned GraphSpec JSON schema (lossless)."""
        doc: dict[str, Any] = {
            "format": GRAPH_SPEC_FORMAT,
            "variables": [
                {
                    "key": var.key,
                    "description": var.description,
                    "states": list(var.states),
                }
                for var in self.variables
            ],
            "edges": [
                {"parent": edge.parent, "child": edge.child} for edge in self.edges
            ],
            "cpts": {
                child.key: {
                    "child": child.key,
                    "parents": list(cpt.parents),
                    "rows": [
                        {
                            "assignment": dict(assignment),
                            "probabilities": list(probs),
                        }
                        for assignment, probs in cpt.rows
                    ],
                }
                for child in self.variables
                for cpt in (self.cpts[child.key],)
            },
        }
        if self.jev_factors:
            doc["jev_factors"] = [dict(factor) for factor in self.jev_factors]
        return doc


def _check_probabilities(child: str, row: int, probs: tuple[float, ...]) -> None:
    """Fail-closed probability checks (finite, >= 0, sums to 1 within 1e-6)."""
    for j, p in enumerate(probs):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} must be "
                f"a number, got {p!r}"
            )
        if not math.isfinite(float(p)):
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} is not "
                f"finite: {p!r}"
            )
        if p < 0:
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} is negative: {p!r}"
            )
    total = float(sum(probs))
    if abs(total - 1.0) > _ROW_SUM_TOLERANCE:
        raise ValueError(
            f"GraphSpec cpts[{child}]: row {row} probabilities sum to "
            f"{total!r}, expected 1.0 within {_ROW_SUM_TOLERANCE}"
        )


def _canonical_assignments(
    parents: tuple[str, ...], states_lists: list[tuple[str, ...]]
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Parent-assignment odometer order; first parent slowest."""
    out: list[tuple[tuple[str, str], ...]] = []
    for combo in itertools.product(*states_lists):
        out.append(tuple(zip(parents, combo)))
    return tuple(out)


def _topological_order(
    keys: list[str], parent_map: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Kahn's algorithm with insertion-order tiebreak (deterministic)."""
    remaining = {key: len(parent_map[key]) for key in keys}
    children: dict[str, list[str]] = {key: [] for key in keys}
    for child, parents in parent_map.items():
        for parent in parents:
            children[parent].append(child)
    order: list[str] = []
    ready = [key for key in keys if remaining[key] == 0]
    while ready:
        key = ready.pop(0)
        order.append(key)
        for child in children[key]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(order) != len(keys):
        cycle = sorted(set(keys) - set(order))
        raise ValueError(f"GraphSpec edges: cycle detected involving {cycle!r}")
    return tuple(order)


def _validate_edges_and_order(
    keys: list[str], edges: tuple[GraphEdge, ...]
) -> dict[str, tuple[str, ...]]:
    """Validate edge structure (endpoints, self-loops, duplicates, acyclic)
    and return the child -> parent-tuple map in edge declaration order."""
    key_set = set(keys)
    edges_seen: set[tuple[str, str]] = set()
    parent_map: dict[str, tuple[str, ...]] = dict.fromkeys(keys, ())
    for edge in edges:
        for endpoint, role in ((edge.parent, "parent"), (edge.child, "child")):
            if endpoint not in key_set:
                raise ValueError(
                    f"GraphSpec edges: unknown {role} {endpoint!r} on edge "
                    f"{edge.parent!r} -> {edge.child!r}"
                )
        if edge.parent == edge.child:
            raise ValueError(f"GraphSpec edges: self-loop on {edge.parent!r}")
        pair = (edge.parent, edge.child)
        if pair in edges_seen:
            raise ValueError(
                f"GraphSpec edges: duplicate edge {edge.parent!r} -> {edge.child!r}"
            )
        edges_seen.add(pair)
        parent_map[edge.child] = parent_map[edge.child] + (edge.parent,)
    _topological_order(keys, parent_map)
    return parent_map


def load_graphspec(data: object) -> GraphSpec:
    """Load and fully validate a GraphSpec JSON document (duplicated rules)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"GraphSpec: expected a JSON object, got {type(data).__name__}"
        )
    fmt = data.get("format")
    if fmt != GRAPH_SPEC_FORMAT:
        raise ValueError(
            f"GraphSpec format: expected {GRAPH_SPEC_FORMAT!r}, got {fmt!r}"
        )
    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"GraphSpec: unknown top-level keys {unknown!r} "
            f"(allowed: {sorted(_ALLOWED_TOP_LEVEL_KEYS)!r})"
        )
    raw_vars = data.get("variables")
    if not isinstance(raw_vars, list) or not raw_vars:
        raise ValueError("GraphSpec variables: must be a non-empty list of objects")
    variables: list[GraphVariable] = []
    for i, raw in enumerate(raw_vars):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"GraphSpec variables[{i}]: expected an object, got {raw!r}"
            )
        key = raw.get("key")
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(
                f"GraphSpec variables[{i}]: invalid key {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if any(v.key == key for v in variables):
            raise ValueError(f"GraphSpec variables[{i}]: duplicate key {key!r}")
        description = raw.get("description")
        if not isinstance(description, str):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): description must be a "
                f"string, got {type(description).__name__}"
            )
        states = raw.get("states")
        if (
            not isinstance(states, list)
            or len(states) < 2
            or not all(isinstance(s, str) and s for s in states)
        ):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): states must be a list of "
                f"at least 2 non-empty strings, got {states!r}"
            )
        if len(set(states)) != len(states):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): duplicate state labels {states!r}"
            )
        variables.append(
            GraphVariable(key=key, description=description, states=tuple(states))
        )

    raw_edges = data.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("GraphSpec edges: must be a list of objects")
    edges: list[GraphEdge] = []
    for i, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            raise ValueError(f"GraphSpec edges[{i}]: expected an object, got {raw!r}")
        parent = raw.get("parent")
        child = raw.get("child")
        if not isinstance(parent, str) or not _KEY_RE.match(parent):
            raise ValueError(
                f"GraphSpec edges[{i}]: invalid parent {parent!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if not isinstance(child, str) or not _KEY_RE.match(child):
            raise ValueError(
                f"GraphSpec edges[{i}]: invalid child {child!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        edges.append(GraphEdge(parent=parent, child=child))

    keys = [v.key for v in variables]
    parent_map: dict[str, tuple[str, ...]] = _validate_edges_and_order(
        keys, tuple(edges)
    )

    raw_cpts = data.get("cpts")
    if not isinstance(raw_cpts, Mapping):
        raise ValueError("GraphSpec cpts: must be an object keyed by child")
    states_map = {v.key: v.states for v in variables}
    cpts: dict[str, GraphCPT] = {}
    for key in (v.key for v in variables):  # deterministic build order
        raw = raw_cpts.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"GraphSpec cpts[{key}]: missing or non-object CPT entry")
        child = raw.get("child")
        if child != key:
            raise ValueError(
                f"GraphSpec cpts[{key}]: child field {child!r} does not "
                "match the mapping key"
            )
        parents = raw.get("parents")
        if not isinstance(parents, list) or not all(
            isinstance(p, str) for p in parents
        ):
            raise ValueError(
                f"GraphSpec cpts[{key}]: parents must be a list of strings, "
                f"got {parents!r}"
            )
        if tuple(parents) != parent_map[key]:
            raise ValueError(
                f"GraphSpec cpts[{key}]: parents must equal the graph "
                f"parents in edge order {list(parent_map[key])!r}, "
                f"got {parents!r}"
            )
        raw_rows = raw.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"GraphSpec cpts[{key}]: rows must be a list of objects")
        canonical = _canonical_assignments(
            tuple(parents), [states_map[p] for p in parents]
        )
        if len(raw_rows) != len(canonical):
            raise ValueError(
                f"GraphSpec cpts[{key}]: expected {len(canonical)} rows (one "
                f"per parent assignment), got {len(raw_rows)}"
            )
        rows: list[tuple[dict[str, str], tuple[float, ...]]] = []
        for i, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} expected an object, "
                    f"got {raw_row!r}"
                )
            assignment = raw_row.get("assignment")
            probs_raw = raw_row.get("probabilities")
            if not isinstance(assignment, Mapping):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} assignment must be an "
                    f"object, got {assignment!r}"
                )
            if dict(assignment) != dict(canonical[i]):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} assignment "
                    f"{dict(assignment)!r} violates the canonical order "
                    f"(expected {dict(canonical[i])!r}; rows must be in "
                    "parent-assignment odometer order, first parent slowest)"
                )
            if not isinstance(probs_raw, list) or not probs_raw:
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} probabilities must be a "
                    f"non-empty list of numbers, got {probs_raw!r}"
                )
            if len(probs_raw) != len(states_map[key]):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} has {len(probs_raw)} "
                    f"probabilities, expected {len(states_map[key])} "
                    f"(child states: {list(states_map[key])!r})"
                )
            for j, p in enumerate(probs_raw):
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    raise ValueError(
                        f"GraphSpec cpts[{key}]: row {i} probability {j} is "
                        f"not a number, got {p!r}"
                    )
            probs = tuple(float(p) for p in probs_raw)
            _check_probabilities(key, i, probs)
            rows.append((dict(assignment), probs))
        cpts[key] = GraphCPT(child=key, parents=tuple(parents), rows=tuple(rows))

    jev_factors_raw = data.get("jev_factors", [])
    if not isinstance(jev_factors_raw, list):
        raise ValueError(
            "GraphSpec jev_factors (RESERVED field): must be a list of "
            f"objects, got {jev_factors_raw!r}"
        )
    for factor in jev_factors_raw:
        if not isinstance(factor, Mapping) or not all(
            isinstance(k, str) for k in factor
        ):
            raise ValueError(
                "GraphSpec jev_factors (RESERVED field): entries must be "
                f"objects with string keys, got {factor!r}"
            )
    jev_factors = tuple(dict(f) for f in jev_factors_raw)

    spec = GraphSpec(
        variables=tuple(variables),
        edges=tuple(edges),
        cpts=cpts,
        jev_factors=jev_factors,
    )
    logger.info(
        "loaded GraphSpec: variables=%d edges=%d jev_factors=%d",
        len(spec.variables),
        len(spec.edges),
        len(spec.jev_factors),
    )
    return spec


def load_graphspec_file(path: str | os.PathLike[str]) -> GraphSpec:
    """Load a GraphSpec JSON file from disk."""
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GraphSpec file {os.fspath(path)!r}: invalid JSON ({exc})"
            ) from exc
    return load_graphspec(data)
