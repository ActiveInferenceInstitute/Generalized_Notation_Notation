"""Strict Bayes-net GNN subset parsing and deterministic source rendering."""

from __future__ import annotations

import logging
import re
from typing import Any, Final

from .graphspec import (
    _KEY_RE,
    GraphCPT,
    GraphEdge,
    GraphSpec,
    GraphVariable,
    _canonical_assignments,
    _check_probabilities,
)

logger = logging.getLogger(__name__)


_GNN_SECTION_RE: Final[re.Pattern[str]] = re.compile(r"^##\s+(.+?)\s*$")


_GNN_SUBSECTION_RE: Final[re.Pattern[str]] = re.compile(r"^###\s+(.+?)\s*$")


_GNN_EDGE_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*?)\s*(?:->|>)\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$"
)


_GNN_CPT_OPEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*\{\s*$"
)


_GNN_ROW_RE: Final[re.Pattern[str]] = re.compile(
    r"^\((.*)\)\s*=\s*\(([^)]*)\)\s*,?\s*$"
)


_GNN_STATES_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\[Discrete\]\s*)?(.+)$")


def parse_gnn_subset(text: str) -> GraphSpec:
    """Parse the minimal Bayes-net ``.gnn`` markdown subset into a GraphSpec.

    Accepted subset (anything else is rejected with an actionable message):

    * ``# ...`` comment lines and blank lines anywhere.
    * ``## <key>`` variable blocks with ``### Description`` (free text) and
      ``### States`` (one ``[Discrete] label, label, ...`` line). A
      ``## Variables`` container with ``### <key>`` blocks is also accepted
      (inside a container, any ``### <key>`` that is not ``Description`` or
      ``States`` starts the next variable block).
    * ``## Connections`` with one ``parent>child`` (native GNN arrow) or
      ``parent->child`` line per edge, in declaration order.
    * ``## InitialParameterization`` with one CPT block per variable:
      ``name={`` ... rows ... ``}`` where each row is
      ``(p1=v1, p2=v2) = (0.1, 0.9)`` (assignment in graph parent order,
      probabilities in child state order) or ``() = (0.99, 0.01)`` for a
      prior. Rows may appear in any order; they are canonicalized into
      GraphSpec odometer order.

    Unknown headings, malformed rows, and undeclared endpoints fail
    closed with the line number and offending value.
    """
    variables: dict[str, dict[str, Any]] = {}
    var_order: list[str] = []
    edges: list[tuple[int, str, str]] = []
    cpts: dict[str, dict[str, Any]] = {}

    mode = "none"  # none | variables | varblock | connections | params
    current_key: str | None = None
    description_lines: list[str] = []
    states_line: str | None = None
    cpt_child: str | None = None
    cpt_rows: list[tuple[tuple[tuple[str, str], ...], tuple[float, ...]]] = []
    in_container = False

    def close_variable() -> None:
        nonlocal current_key, description_lines, states_line
        if current_key is None:
            return
        if states_line is None:
            raise ValueError(
                f".gnn variable block '## {current_key}': missing '### States'"
            )
        match = _GNN_STATES_RE.match(states_line)
        if not match:
            raise ValueError(
                f".gnn variable block '## {current_key}': unparsable states "
                f"line {states_line!r} (expected '[Discrete] s1, s2, ...')"
            )
        raw_states = match.group(1)
        states = tuple(s.strip() for s in raw_states.split(",") if s.strip()) or tuple(
            raw_states.split()
        )
        variables[current_key] = {
            "description": " ".join(s.strip() for s in description_lines).strip(),
            "states": states,
        }
        var_order.append(current_key)
        current_key = None
        description_lines = []
        states_line = None

    def close_cpt() -> None:
        nonlocal cpt_child, cpt_rows
        if cpt_child is None:
            return
        cpts[cpt_child] = {
            "assignments": [dict(a) for a, _ in cpt_rows],
            "probabilities": [p for _, p in cpt_rows],
        }
        cpt_child = None
        cpt_rows = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") and not line.startswith("##"):
            # Single-# lines are comments; ## / ### are structural headings.
            continue

        heading = _GNN_SECTION_RE.match(line)
        if heading:
            close_variable()
            close_cpt()
            name = heading.group(1)
            if name in ("Connections", "InitialParameterization"):
                mode = "connections" if name == "Connections" else "params"
                current_key = None
                in_container = False
            elif name == "Variables":
                mode = "variables"
                current_key = None
                in_container = True
            elif mode in ("connections", "params"):
                raise ValueError(
                    f".gnn line {lineno}: unexpected block '## {name}' inside "
                    f"the {mode} section"
                )
            else:
                if not _KEY_RE.match(name):
                    raise ValueError(
                        f".gnn line {lineno}: invalid variable key {name!r} "
                        "(must match [A-Za-z_][A-Za-z0-9_-]*)"
                    )
                if name in variables:
                    raise ValueError(
                        f".gnn line {lineno}: duplicate variable block '## {name}'"
                    )
                mode = "varblock"
                current_key = name
            continue

        subsection = _GNN_SUBSECTION_RE.match(line)
        if subsection:
            name = subsection.group(1)
            if mode == "variables":
                if name in ("Description", "States"):
                    raise ValueError(
                        f".gnn line {lineno}: '### {name}' appears before "
                        "any variable block in the Variables section"
                    )
                if not _KEY_RE.match(name):
                    raise ValueError(
                        f".gnn line {lineno}: invalid variable key {name!r}"
                    )
                if name in variables:
                    raise ValueError(
                        f".gnn line {lineno}: duplicate variable block '### {name}'"
                    )
                mode = "varblock"
                current_key = name
                continue
            if mode == "varblock":
                if name == "Description":
                    description_lines = []
                    continue
                if name == "States":
                    states_line = "__PENDING__"
                    continue
                if in_container:
                    # In a `## Variables` container, `### <key>` starts the
                    # next variable block.
                    close_variable()
                    if name in variables:
                        raise ValueError(
                            f".gnn line {lineno}: duplicate variable block '### {name}'"
                        )
                    current_key = name
                    continue
                raise ValueError(
                    f".gnn line {lineno}: unknown subsection '### {name}' "
                    "(accepted: Description, States)"
                )
            raise ValueError(
                f".gnn line {lineno}: unexpected subsection '### {name}' "
                f"in the {mode} section"
            )

        if mode == "varblock":
            if states_line == "__PENDING__":
                states_line = line
                continue
            description_lines.append(line)
            continue
        if mode == "connections":
            edge_match = _GNN_EDGE_RE.match(line)
            if not edge_match:
                raise ValueError(
                    f".gnn line {lineno}: unparsable edge {line!r} "
                    "(expected 'parent>child' or 'parent->child')"
                )
            edges.append((lineno, edge_match.group(1), edge_match.group(2)))
            continue
        if mode == "params":
            open_match = _GNN_CPT_OPEN_RE.match(line)
            if open_match:
                close_cpt()
                cpt_child = open_match.group(1)
                continue
            if cpt_child is not None:
                if line == "}":
                    close_cpt()
                    continue
                row_match = _GNN_ROW_RE.match(line)
                if not row_match:
                    raise ValueError(
                        f".gnn line {lineno}: unparsable CPT row {line!r} "
                        "(expected '(p1=v1, p2=v2) = (0.1, 0.9)' or "
                        "'() = (0.99, 0.01)')"
                    )
                assignment_part, probs_part = row_match.groups()
                assignment: list[tuple[str, str]] = []
                if assignment_part.strip():
                    for item in assignment_part.split(","):
                        if "=" not in item:
                            raise ValueError(
                                f".gnn line {lineno}: CPT assignment item "
                                f"{item!r} lacks '=' (expected 'parent=state')"
                            )
                        key, label = item.split("=", 1)
                        assignment.append((key.strip(), label.strip()))
                try:
                    probs = tuple(
                        float(p.strip()) for p in probs_part.split(",") if p.strip()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f".gnn line {lineno}: CPT probabilities "
                        f"{probs_part!r} are not all numbers ({exc})"
                    ) from exc
                cpt_rows.append((tuple(assignment), probs))
                continue
            raise ValueError(
                f".gnn line {lineno}: unparsable line {line!r} in the "
                "InitialParameterization section (expected 'name={' to "
                "open a CPT block)"
            )

    close_variable()
    close_cpt()

    if not variables:
        raise ValueError(
            ".gnn: no variable blocks found (expected at least one "
            "'## <key>' block with '### States')"
        )

    variables_out = tuple(
        GraphVariable(
            key=key,
            description=variables[key]["description"],
            states=variables[key]["states"],
        )
        for key in var_order
    )
    keys = set(var_order)
    edges_out = []
    for lineno, parent, child in edges:
        for role, value in (("parent", parent), ("child", child)):
            if value not in keys:
                raise ValueError(
                    f".gnn line {lineno}: edge references undeclared {role} {value!r}"
                )
        edges_out.append(GraphEdge(parent=parent, child=child))
    edges_tuple = tuple(edges_out)

    parent_map: dict[str, tuple[str, ...]] = {
        v.key: tuple(e.parent for e in edges_tuple if e.child == v.key)
        for v in variables_out
    }
    states_map = {v.key: v.states for v in variables_out}
    cpts_out: dict[str, GraphCPT] = {}
    for v in variables_out:
        raw = cpts.get(v.key)
        if raw is None:
            raise ValueError(
                f".gnn: missing CPT block '{v.key}={{...}}' in the "
                "InitialParameterization section"
            )
        parents = parent_map[v.key]
        canonical = _canonical_assignments(parents, [states_map[p] for p in parents])
        rows_raw: list[tuple[dict[str, str], tuple[float, ...]]] = [
            (dict(a), probs)
            for a, probs in zip(raw["assignments"], raw["probabilities"])
        ]
        if len(rows_raw) != len(canonical):
            raise ValueError(
                f".gnn CPT '{v.key}': expected {len(canonical)} rows (one "
                f"per parent assignment), got {len(rows_raw)}"
            )
        lookup: dict[tuple[tuple[str, str], ...], tuple[float, ...]] = {}
        for i, (assign_map, probs) in enumerate(rows_raw):
            if set(assign_map) != set(parents):
                raise ValueError(
                    f".gnn CPT '{v.key}': row {i} assignment "
                    f"{assign_map!r} does not match the declared parents "
                    f"{list(parents)!r}"
                )
            normalized = tuple((p, assign_map[p]) for p in parents)
            if normalized in lookup:
                raise ValueError(
                    f".gnn CPT '{v.key}': duplicate parent assignment {assign_map!r}"
                )
            for parent, label in normalized:
                if label not in states_map[parent]:
                    raise ValueError(
                        f".gnn CPT '{v.key}': row {i} unknown state "
                        f"{label!r} for parent {parent!r} (states: "
                        f"{list(states_map[parent])!r})"
                    )
            if len(probs) != len(v.states):
                raise ValueError(
                    f".gnn CPT '{v.key}': row {i} has {len(probs)} "
                    f"probabilities, expected {len(v.states)} (child "
                    f"states: {list(v.states)!r})"
                )
            _check_probabilities(v.key, i, probs)
            lookup[normalized] = probs
        ordered_rows = []
        for expected in canonical:
            matched_probs = lookup.get(expected)
            if matched_probs is None:
                raise ValueError(
                    f".gnn CPT '{v.key}': missing parent assignment {dict(expected)!r}"
                )
            ordered_rows.append((dict(expected), matched_probs))
        cpts_out[v.key] = GraphCPT(
            child=v.key, parents=parents, rows=tuple(ordered_rows)
        )

    spec = GraphSpec(
        variables=variables_out,
        edges=edges_tuple,
        cpts=cpts_out,
    )
    logger.info(
        "parsed .gnn subset: variables=%d edges=%d",
        len(spec.variables),
        len(spec.edges),
    )
    return spec


def render_gnn_subset(spec: GraphSpec) -> str:
    """Render a GraphSpec back to the minimal ``.gnn`` subset (deterministic).

    ``parse_gnn_subset(render_gnn_subset(spec)) == spec`` holds for every
    validated GraphSpec; rows are emitted in canonical odometer order and
    probabilities with ``repr(float)`` shortest form.
    """
    lines: list[str] = [
        "# Bayes-net .gnn subset for gnn.rxinfer_bridge",
        "# (GraphSpec interchange: dafjev.bayesnet/1; round-trips through",
        "# parse_gnn_subset/render_gnn_subset).",
        "",
    ]
    for var in spec.variables:
        lines.append(f"## {var.key}")
        lines.append("### Description")
        if var.description:
            lines.append(var.description)
        lines.append("### States")
        lines.append(f"[Discrete] {', '.join(var.states)}")
        lines.append("")
    lines.append("## Connections")
    lines.append("")
    for edge in spec.edges:
        lines.append(f"{edge.parent}>{edge.child}")
    lines.append("")
    lines.append("## InitialParameterization")
    lines.append("")
    for var in spec.variables:
        cpt = spec.cpts[var.key]
        lines.append(f"{var.key}={{")
        for assignment, probs in cpt.rows:
            if assignment:
                lhs = ", ".join(
                    f"{parent}={assignment[parent]}" for parent in cpt.parents
                )
            else:
                lhs = ""
            rhs = ", ".join(repr(float(p)) for p in probs)
            lines.append(f"  ({lhs}) = ({rhs})")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
