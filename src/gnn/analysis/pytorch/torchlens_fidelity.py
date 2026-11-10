"""Compare declared GNN ``## Connections`` with TorchLens-traced dataflow.

The discrete PyTorch script records a TorchLens summary in
``simulation_results.json`` when ``GNN_TORCHLENS_CAPTURE`` is set (see
``gnn.render.pytorch.discrete_script``). That summary carries the declared
connections, the module/buffer sites of each GNN variable, and the counted
dataflow edges between traced sites. This module turns it into a structural
fidelity report: for every declared connection, does the executed tensor
computation carry information along it?

The comparison reads JSON only; it never imports torch or torchlens. It
refuses to evaluate a capture that is not evidence of the run it describes
(trace not reproducing the simulation, or failed TorchLens validation).

Edge statuses, per expanded source/target variable pair:

``direct``
    A traced op at a target-owned site consumes a value produced at a
    source-owned site (one dataflow hop).
``indirect``
    Information reaches the target through other traced sites; the shortest
    site path is reported. The rollout is a closed perception-action loop, so
    a path may pass through later timesteps and most agent variables reach
    one another indirectly: ``direct`` is the discriminating status.
``reverse_only``
    A directed declaration whose only traced path runs target to
    source.
``not_observed``
    The required traced path was absent. Bidirectional declarations require
    a traced path in each direction.
``unmapped``
    An endpoint has no traced site, or the relation has no trace criterion.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "gnn_torchlens_fidelity_v1"
FIDELITY_JSON = "pytorch_torchlens_fidelity.json"
FIDELITY_MARKDOWN = "pytorch_torchlens_fidelity.md"
FIDELITY_FIGURE = "pytorch_torchlens_fidelity.png"

_STATUS_RANK = {"direct": 0, "indirect": 1, "reverse_only": 2, "not_observed": 3}
_STATUS_COLORS = {
    "direct": "#1b7837",
    "indirect": "#2166ac",
    "reverse_only": "#e08214",
    "not_observed": "#b2182b",
}


def _endpoint_members(name: str) -> List[str]:
    """Expand a grouped endpoint such as ``(s_f0, u)`` into its members."""
    text = str(name).strip()
    if text.startswith("(") and text.endswith(")"):
        return [part.strip() for part in text[1:-1].split(",") if part.strip()]
    return [text] if text else []


def site_owners(
    variable_sites: Mapping[str, Sequence[str]], sites: Iterable[str]
) -> Dict[str, Set[str]]:
    """Assign each traced site to the variables whose mapped site is its longest prefix.

    ``agent.efe.risk`` belongs to ``G`` when only ``G -> agent.efe`` is
    mapped, and to ``G_ins`` when that variable maps the risk site itself.
    """
    owners: Dict[str, Set[str]] = {}
    for site in sites:
        best_length = -1
        best: Set[str] = set()
        for variable, mapped in variable_sites.items():
            for prefix in mapped:
                if site == prefix or site.startswith(prefix + "."):
                    if len(prefix) > best_length:
                        best_length, best = len(prefix), {variable}
                    elif len(prefix) == best_length:
                        best.add(variable)
        if best:
            owners[site] = best
    return owners


def _adjacency(site_edges: Sequence[Sequence[Any]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for source, target, *_ in site_edges:
        graph.setdefault(str(source), set()).add(str(target))
        graph.setdefault(str(target), set())
    return graph


def _shortest_path(
    graph: Mapping[str, Set[str]], sources: Set[str], targets: Set[str]
) -> Optional[List[str]]:
    """Breadth-first shortest site path from any source site to any target site."""
    queue = deque((site, [site]) for site in sorted(sources) if site in graph)
    seen = {site for site, _ in queue}
    while queue:
        site, path = queue.popleft()
        for nxt in sorted(graph.get(site, ())):
            if nxt in targets:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def _owned_sites(variable: str, owners: Mapping[str, Set[str]]) -> Set[str]:
    return {site for site, names in owners.items() if variable in names}


def _declared_directions(
    source: str, target: str, relation: str
) -> Tuple[Tuple[str, str], ...]:
    if relation in ("<", "<-"):
        return ((target, source),)
    if relation in ("-", "<->"):
        return ((source, target), (target, source))
    if relation in (">", "->"):
        return ((source, target),)
    return ()


def _pair_status(
    source: str,
    target: str,
    relation: str,
    graph: Mapping[str, Set[str]],
    owners: Mapping[str, Set[str]],
    variable_sites: Mapping[str, Sequence[str]],
    unmapped: Mapping[str, str],
) -> Dict[str, Any]:
    pair: Dict[str, Any] = {"source": source, "target": target}
    directions = _declared_directions(source, target, relation)
    if not directions:
        pair["status"] = "unmapped"
        pair["reasons"] = {"relation": f"traced dataflow cannot establish {relation!r}"}
        return pair
    source, target = directions[0]
    missing = [name for name in (source, target) if name not in variable_sites]
    if missing:
        pair["status"] = "unmapped"
        pair["reasons"] = {
            name: unmapped.get(name, "no traced site recorded for this variable")
            for name in missing
        }
        return pair
    source_sites = _owned_sites(source, owners)
    target_sites = _owned_sites(target, owners)
    if not source_sites or not target_sites:
        pair["status"] = "not_observed"
        pair["reasons"] = {
            name: "mapped site never appeared in the trace"
            for name, sites in ((source, source_sites), (target, target_sites))
            if not sites
        }
        return pair
    forward = _shortest_path(graph, source_sites, target_sites)
    backward = _shortest_path(graph, target_sites, source_sites)
    if relation == "<->":
        if forward is not None:
            pair["path"] = forward
        if backward is not None and source != target:
            pair["reverse_path"] = backward
        if forward is None or backward is None:
            pair["status"] = "not_observed"
            pair["reasons"] = {"relation": "bidirectional relation requires both paths"}
        else:
            pair["status"] = (
                "direct" if max(len(forward), len(backward)) == 2 else "indirect"
            )
        return pair
    if relation != "-":
        if forward is None:
            pair["status"] = "reverse_only" if backward is not None else "not_observed"
            if backward is not None:
                pair["path"] = backward
            return pair
        chosen = forward
    else:
        # Undirected: either direction supports the declaration; the shorter
        # path wins and ties prefer the authored source-to-target direction.
        candidates = [
            (path, direction)
            for path, direction in ((forward, "forward"), (backward, "backward"))
            if path is not None
        ]
        if not candidates:
            pair["status"] = "not_observed"
            return pair
        chosen, pair["direction"] = min(candidates, key=lambda item: len(item[0]))
    pair["status"] = "direct" if len(chosen) == 2 else "indirect"
    pair["path"] = chosen
    return pair


def _overall_status(pairs: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(pair["status"]) for pair in pairs]
    if not statuses or "unmapped" in statuses:
        return "unmapped"
    return max(statuses, key=_STATUS_RANK.__getitem__)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _graph_problem(capture: Mapping[str, Any]) -> Optional[str]:
    variable_sites = capture.get("variable_sites")
    if not isinstance(variable_sites, Mapping) or not all(
        _text(name) and isinstance(sites, list) and sites and all(map(_text, sites))
        for name, sites in variable_sites.items()
    ):
        return "invalid or missing variable_sites"
    unmapped = capture.get("unmapped_variables", {})
    if not isinstance(unmapped, Mapping) or not all(
        _text(name) and _text(reason) for name, reason in unmapped.items()
    ):
        return "invalid unmapped_variables"
    site_edges = capture.get("site_edges")
    if not isinstance(site_edges, list) or not all(
        isinstance(edge, list)
        and len(edge) == 3
        and _text(edge[0])
        and _text(edge[1])
        and type(edge[2]) is int
        and edge[2] > 0
        for edge in site_edges
    ):
        return "invalid or missing site_edges"
    connections = capture.get("declared_connections")
    if not isinstance(connections, list) or not all(
        isinstance(row, Mapping)
        and _text(row.get("relation"))
        and all(
            _text(row.get(key)) and _endpoint_members(row[key])
            for key in ("source", "target")
        )
        for row in connections
    ):
        return "invalid or missing declared_connections"
    return None


def evaluate_capture(torchlens: Any) -> Dict[str, Any]:
    """Return the structural fidelity report for one TorchLens capture summary."""
    if not isinstance(torchlens, Mapping):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unverified",
            "reason": "TorchLens capture summary must be a JSON object",
        }
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "capture_status": torchlens.get("status"),
        "torchlens_version": torchlens.get("torchlens_version"),
        "num_ops": torchlens.get("num_ops"),
        "trace_reproduces_simulation": torchlens.get("trace_reproduces_simulation"),
        "validated": torchlens.get("validated"),
        "agent_step_validation": torchlens.get("agent_step_validation"),
    }
    if torchlens.get("status") != "captured":
        report["status"] = "unverified"
        report["reason"] = (
            f"TorchLens capture status is {torchlens.get('status')!r}: "
            f"{torchlens.get('reason', 'no reason recorded')}"
        )
        return report
    if torchlens.get("trace_reproduces_simulation") is not True:
        report["status"] = "unverified"
        report["reason"] = "the traced rollout did not reproduce the simulation outputs"
        return report
    if torchlens.get("validated") is not True:
        report["status"] = "unverified"
        report["reason"] = "TorchLens validation of the agent step did not pass"
        return report
    problem = _graph_problem(torchlens)
    if problem is not None:
        report["status"] = "unverified"
        report["reason"] = problem
        return report

    variable_sites = torchlens["variable_sites"]
    unmapped = torchlens.get("unmapped_variables", {})
    site_edges = torchlens["site_edges"]
    graph = _adjacency(site_edges)
    owners = site_owners(variable_sites, graph)

    edges: List[Dict[str, Any]] = []
    declared_pairs: Set[Tuple[str, str]] = set()
    for row in torchlens["declared_connections"]:
        relation = row["relation"]
        pairs = []
        for source in _endpoint_members(row["source"]):
            for target in _endpoint_members(row["target"]):
                pairs.append(
                    _pair_status(
                        source,
                        target,
                        relation,
                        graph,
                        owners,
                        variable_sites,
                        unmapped,
                    )
                )
                declared_pairs.update(_declared_directions(source, target, relation))
        edges.append(
            {
                "source": row.get("source"),
                "relation": relation,
                "target": row.get("target"),
                "status": _overall_status(pairs),
                "pairs": pairs,
            }
        )

    variable_edges: Dict[Tuple[str, str], int] = {}
    for source_site, target_site, count in site_edges:
        for source in owners.get(str(source_site), ()):
            for target in owners.get(str(target_site), ()):
                if source != target:
                    key = (source, target)
                    variable_edges[key] = variable_edges.get(key, 0) + int(count)
    undeclared = [
        {"source": s, "target": t, "op_edges": n}
        for (s, t), n in sorted(variable_edges.items())
        if (s, t) not in declared_pairs
    ]

    counts: Dict[str, int] = {}
    for edge in edges:
        counts[edge["status"]] = counts.get(edge["status"], 0) + 1
    mapped = [edge for edge in edges if edge["status"] != "unmapped"]
    supported = [edge for edge in mapped if edge["status"] in ("direct", "indirect")]
    report.update(
        {
            "status": "evaluated",
            "declared_edges": edges,
            "status_counts": dict(sorted(counts.items())),
            "mapped_declared_edges": len(mapped),
            "supported_declared_edges": len(supported),
            "structural_agreement": (len(supported) / len(mapped)) if mapped else None,
            "undeclared_dependencies": undeclared,
            "unmapped_variables": unmapped,
            "variable_sites": variable_sites,
            "intervention_witnesses": torchlens.get("intervention_witnesses"),
        }
    )
    return report


def _markdown(model_name: str, report: Mapping[str, Any]) -> str:
    lines = [
        f"# TorchLens structural fidelity: {model_name}",
        "",
        f"- Status: **{report.get('status')}**",
        f"- TorchLens {report.get('torchlens_version')}, {report.get('num_ops')} traced ops",
        f"- Trace reproduces simulation: {report.get('trace_reproduces_simulation')}",
        f"- Agent-step validation: {report.get('agent_step_validation')}",
    ]
    if report.get("status") != "evaluated":
        lines += ["", f"Not evaluated: {report.get('reason')}", ""]
        return "\n".join(lines)
    agreement = report.get("structural_agreement")
    lines += [
        f"- Declared connections supported by traced dataflow: "
        f"{report['supported_declared_edges']} of {report['mapped_declared_edges']} mapped"
        + (f" ({agreement:.0%})" if isinstance(agreement, float) else ""),
        "",
        "## Declared connections",
        "",
        "| Connection | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for edge in report["declared_edges"]:
        evidence = []
        for pair in edge["pairs"]:
            for key in ("path", "reverse_path"):
                if pair.get(key):
                    evidence.append(" → ".join(f"`{site}`" for site in pair[key]))
            if pair.get("reasons"):
                evidence.append(
                    ", ".join(f"{k}: {v}" for k, v in pair["reasons"].items())
                )
        label = f"`{edge['source']} {edge['relation']} {edge['target']}`"
        label = label.replace("|", r"\|")
        detail = "<br>".join(evidence).replace("|", r"\|") or "—"
        lines.append(f"| {label} | {edge['status']} | {detail} |")
    lines += ["", "## Undeclared traced dependencies", ""]
    if report["undeclared_dependencies"]:
        lines += ["| Source | Target | Op edges |", "| --- | --- | --- |"]
        lines += [
            f"| `{d['source']}` | `{d['target']}` | {d['op_edges']} |"
            for d in report["undeclared_dependencies"]
        ]
    else:
        lines.append("None.")
    if report["unmapped_variables"]:
        lines += ["", "## Unmapped variables", ""]
        lines += [
            f"- `{k}`: {v}" for k, v in sorted(report["unmapped_variables"].items())
        ]
    witnesses = report.get("intervention_witnesses")
    if isinstance(witnesses, Mapping) and witnesses:
        lines += [
            "",
            f"## Intervention witnesses (timestep {witnesses.get('timestep')})",
            "",
            "| Lesion | Site | KL(full ‖ lesioned) | argmax action |",
            "| --- | --- | --- | --- |",
            f"| none | — | 0 | {witnesses.get('argmax_action')} |",
        ]
        lesions = witnesses.get("lesions")
        if isinstance(lesions, Mapping):
            for name, lesion in sorted(lesions.items()):
                if isinstance(lesion, Mapping):
                    kl = lesion.get("kl_full_to_lesioned")
                    score = "—"
                    if type(kl) is int:
                        score = str(kl)
                    elif isinstance(kl, float) and math.isfinite(kl):
                        score = f"{kl:.6g}"
                    lines.append(
                        f"| {name} | `{lesion.get('site')}` | "
                        f"{score} | {lesion.get('argmax_action')} |"
                    )
    lines.append("")
    return "\n".join(lines)


def _plot(model_name: str, report: Mapping[str, Any], path: Path) -> bool:
    """Draw declared edges coloured by status over the variable graph."""
    from gnn.analysis.viz_base import MATPLOTLIB_AVAILABLE, plt

    if not MATPLOTLIB_AVAILABLE or plt is None or report.get("status") != "evaluated":
        return False
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(report["variable_sites"]))
    declared = []
    for edge in report["declared_edges"]:
        relation = edge["relation"]
        for pair in edge["pairs"]:
            if pair["status"] == "unmapped":
                continue
            directions = _declared_directions(pair["source"], pair["target"], relation)
            for source, target in dict.fromkeys(
                directions[:1] if relation == "-" else directions
            ):
                declared.append((source, target, pair["status"], relation != "-"))
                graph.add_edge(source, target)
    undeclared = [(d["source"], d["target"]) for d in report["undeclared_dependencies"]]
    graph.add_edges_from(undeclared)
    if graph.number_of_nodes() == 0:
        return False
    positions = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_size=1100,
            node_color="#f0f0f0",
            edgecolors="#404040",
        )
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=undeclared,
            ax=ax,
            node_size=1100,
            edge_color="#bdbdbd",
            style="dotted",
            arrows=True,
            connectionstyle="arc3,rad=0.12",
        )
        for status, color in _STATUS_COLORS.items():
            for directed in (True, False):
                subset = [
                    (s, t) for s, t, st, d in declared if st == status and d == directed
                ]
                if subset:
                    nx.draw_networkx_edges(
                        graph,
                        positions,
                        edgelist=subset,
                        ax=ax,
                        node_size=1100,
                        edge_color=color,
                        width=2.0,
                        style="solid" if status == "direct" else "dashed",
                        arrows=True,
                        arrowstyle="-|>" if directed else "-",
                        connectionstyle="arc3,rad=0.12",
                    )
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                ls="-" if status == "direct" else "--",
                label=f"declared: {status}",
            )
            for status, color in _STATUS_COLORS.items()
        ]
        handles.append(
            plt.Line2D(
                [0], [0], color="#bdbdbd", lw=1.5, ls=":", label="undeclared traced"
            )
        )
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
        ax.set_title(f"Declared connections vs TorchLens dataflow: {model_name}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return True


def write_fidelity_report(
    results: Mapping[str, Any], model_name: str, model_output_dir: Path
) -> List[str]:
    """Write the fidelity JSON/Markdown/PNG for one simulation result.

    Returns the written paths; an empty list when capture was not requested
    (``status == "disabled"``) or the result predates TorchLens capture.
    """
    torchlens = results.get("torchlens")
    if torchlens is None or (
        isinstance(torchlens, Mapping) and torchlens.get("status") == "disabled"
    ):
        return []
    report = evaluate_capture(torchlens)
    report["model_name"] = model_name
    model_output_dir = Path(model_output_dir)
    json_path = model_output_dir / FIDELITY_JSON
    md_path = model_output_dir / FIDELITY_MARKDOWN
    figure = model_output_dir / FIDELITY_FIGURE
    if any(path.is_symlink() for path in (json_path, md_path, figure)):
        logger.warning(
            "Skipping TorchLens fidelity output with a symbolic link: %s",
            model_output_dir,
        )
        return []
    model_output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    written.append(str(json_path))
    md_path.write_text(_markdown(model_name, report), encoding="utf-8")
    written.append(str(md_path))
    try:
        if _plot(model_name, report, figure):
            written.append(str(figure))
    except Exception as exc:  # the figure is a view of the JSON evidence
        logger.warning(f"TorchLens fidelity figure failed for {model_name}: {exc}")
    logger.info(f"✅ PyTorch TorchLens fidelity ({report['status']}): {model_name}")
    return written


__all__ = [
    "FIDELITY_FIGURE",
    "FIDELITY_JSON",
    "FIDELITY_MARKDOWN",
    "SCHEMA_VERSION",
    "evaluate_capture",
    "site_owners",
    "write_fidelity_report",
]
