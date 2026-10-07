"""Source-bound native-view dashboards and descriptive operational plots."""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from .comparison_contract import (
    compare_scientific_pair,
    prepare_scientific_record,
    revalidate_scientific_record,
)
from .result_adapter import model_family
from .viz_base import MATPLOTLIB_AVAILABLE, np, plt, safe_savefig
from .viz_schema import _normalize_framework_name

logger = logging.getLogger(__name__)


def _plot_groups(
    framework_data: Dict[str, Dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Admit individual native views and coherent cross-framework groups."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostics: dict[str, Any] = {
        "schema_version": "gnn.scientific_plot_admission/v1",
        "excluded": {},
        "admitted_models": [],
    }
    duplicates: set[str] = set()
    for label, data in framework_data.items():
        framework = _normalize_framework_name(str(data.get("framework", label)))
        if "scientific_records" in data:
            records = data["scientific_records"]
        elif data.get("simulation_data"):
            bindings = data.get("results") or [data]
            records = [
                prepare_scientific_record(data["simulation_data"], binding)
                for binding in bindings
            ]
        elif data.get("results"):
            records = [prepare_scientific_record(result) for result in data["results"]]
        else:
            records = [prepare_scientific_record(data)]
        for index, record in enumerate(records):
            record = revalidate_scientific_record(record)
            key = f"{label}[{index}]"
            if record["reasons"]:
                diagnostics["excluded"][key] = record["reasons"]
                continue
            model_id = record["metadata"]["model_id"]
            group = groups.setdefault(model_id, {})
            if framework in group:
                duplicates.add(model_id)
                diagnostics["excluded"][model_id] = [
                    "multiple execution results for one model/framework require explicit replicate selection"
                ]
            group[framework] = record
    for model_id, group in list(groups.items()):
        reasons = (
            list(diagnostics["excluded"].get(model_id, []))
            if model_id in duplicates
            else []
        )
        for (_, left), (_, right) in combinations(group.items(), 2):
            witness = compare_scientific_pair(left, right)
            if witness["status"] != "comparable":
                reasons.extend(witness["reasons"])
        if reasons:
            diagnostics["excluded"][model_id] = list(dict.fromkeys(reasons))
            del groups[model_id]
        else:
            diagnostics["admitted_models"].append(model_id)
    return groups, diagnostics


def _diagnostic(path: Path, diagnostics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


def _model_path(path: Path, model_id: str, group_count: int) -> Path:
    if group_count == 1:
        return path
    suffix = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _trace_present(value: Any) -> bool:
    return value is not None and hasattr(value, "__len__") and len(value) > 0


def _entropy(beliefs: Any) -> Any:
    terms = np.zeros_like(beliefs, dtype=float)
    positive = beliefs > 0
    terms[positive] = beliefs[positive] * np.log(beliefs[positive])
    return -terms.sum(axis=1)


def _save(path: Path, outputs: list[str]) -> None:
    plt.tight_layout()
    saved = safe_savefig(path, log=logger)
    if saved:
        outputs.append(saved)


def generate_unified_framework_dashboard(
    framework_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model_name: str = "Active Inference Model",
) -> List[str]:
    """Plot bound model/native views; Gaussian uncertainty comes from covariance.

    The caller's model_name is a display label. Every scientific title retains
    the bound model ID and view identity. Missing custody produces diagnostics,
    never an unbound cross-model or categorical-confidence plot.
    """
    groups, diagnostics = _plot_groups(framework_data)
    outputs: list[str] = []
    if not MATPLOTLIB_AVAILABLE:
        diagnostics["unavailable"] = "matplotlib unavailable"
    else:
        for model_id, group in groups.items():
            first = next(iter(group.values()))
            views = first["views"]
            fig, axes = plt.subplots(
                len(views), 1, figsize=(12, 4 * len(views)), squeeze=False
            )
            for row, name in enumerate(views):
                ax = axes[row, 0]
                gaussian = model_family(views[name]) == "continuous"
                for framework, record in group.items():
                    view = record["views"][name]
                    beliefs = np.asarray(view["beliefs"], dtype=float)
                    for state in range(min(8, beliefs.shape[1])):
                        line = ax.plot(
                            beliefs[:, state], label=f"{framework}: state {state}"
                        )[0]
                        if gaussian:
                            covariance = np.asarray(view["posterior_cov"], dtype=float)
                            std = np.sqrt(np.maximum(covariance[:, state, state], 0))
                            ax.fill_between(
                                np.arange(len(beliefs)),
                                beliefs[:, state] - 1.96 * std,
                                beliefs[:, state] + 1.96 * std,
                                color=line.get_color(),
                                alpha=0.12,
                            )
                ax.set_title(
                    f"{model_id}: {name}"
                    + (
                        " (first 8 states)"
                        if len(views[name]["beliefs"][0]) > 8
                        else ""
                    )
                )
                ax.set_xlabel("Trace position")
                ax.set_ylabel(
                    "Posterior mean (bands: ±1.96 posterior std)"
                    if gaussian
                    else "Categorical probability"
                )
                if not gaussian:
                    ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(alpha=0.3)
            _save(
                _model_path(
                    output_dir / "unified_belief_comparison.png", model_id, len(groups)
                ),
                outputs,
            )
            action_views = {
                name
                for name, view in views.items()
                if _trace_present(view.get("actions"))
                or _trace_present(view.get("controls"))
            }
            self_witnesses = {
                framework: compare_scientific_pair(record, record)
                for framework, record in group.items()
            }
            efe_available = all(
                "expected_free_energy" in witness for witness in self_witnesses.values()
            )
            if action_views or efe_available:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for framework, record in group.items():
                    for name in sorted(action_views):
                        view = record["views"][name]
                        values = (
                            view.get("controls")
                            if model_family(view) == "continuous"
                            else view.get("actions")
                        )
                        if _trace_present(values):
                            array = np.asarray(values, dtype=float)
                            if array.ndim == 1:
                                axes[0].step(
                                    np.arange(len(array)),
                                    array,
                                    label=f"{framework}: {name}",
                                    where="post",
                                )
                            else:
                                for component in range(min(8, array.shape[1])):
                                    axes[0].plot(
                                        array[:, component],
                                        label=f"{framework}: {name}/control {component}",
                                    )
                    if efe_available:
                        axes[1].plot(
                            record["payload"]["expected_free_energy"], label=framework
                        )
                axes[0].set_title(f"{model_id}: authored action/control trace")
                axes[0].set_ylabel(
                    "Control"
                    if all(
                        model_family(view) == "continuous" for view in views.values()
                    )
                    else "Action index"
                )
                axes[1].set_title(
                    f"{model_id}: scalar EFE"
                    if efe_available
                    else "Scalar EFE unavailable: missing aligned kind/convention"
                )
                axes[1].set_ylabel(
                    "Expected free energy" if efe_available else "Unavailable"
                )
                for ax in axes:
                    ax.set_xlabel("Trace position")
                    if ax.lines:
                        ax.legend()
                _save(
                    _model_path(
                        output_dir / "unified_action_efe_comparison.png",
                        model_id,
                        len(groups),
                    ),
                    outputs,
                )
            categorical = {
                name
                for name, view in views.items()
                if model_family(view) != "continuous"
            }
            if categorical:
                fig, ax = plt.subplots(figsize=(12, 5))
                for framework, record in group.items():
                    for name in sorted(categorical):
                        ax.plot(
                            _entropy(
                                np.asarray(
                                    record["views"][name]["beliefs"], dtype=float
                                )
                            ),
                            label=f"{framework}: {name}",
                        )
                ax.set_title(f"{model_id}: categorical posterior entropy")
                ax.set_xlabel("Trace position")
                ax.set_ylabel("Entropy (nats)")
                ax.legend()
                _save(
                    _model_path(
                        output_dir / "unified_entropy_comparison.png",
                        model_id,
                        len(groups),
                    ),
                    outputs,
                )
            else:
                diagnostics.setdefault("unavailable_metrics", {})[model_id] = {
                    "categorical_confidence": "Gaussian means are not categorical probabilities"
                }
    _diagnostic(output_dir / "unified_dashboard_admission.json", diagnostics)
    return outputs


def generate_cross_framework_comparison(
    framework_data: Dict[str, Dict[str, Any]], output_path: Path
) -> str:
    """Describe every attempt once, without asserting matched-workload performance."""
    aggregated: Dict[str, Dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "schema_version": "gnn.operational_plot_aggregation/v1",
        "frameworks": {},
    }

    def measured_time(value: Any) -> bool:
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(value))
            and value >= 0
        )

    def measured_steps(payload: Any) -> int | None:
        if not isinstance(payload, dict):
            return None
        explicit = payload.get("steps_completed")
        if (
            isinstance(explicit, (int, np.integer))
            and not isinstance(explicit, (bool, np.bool_))
            and explicit >= 0
        ):
            return int(explicit)
        lengths = [
            len(payload[key])
            for key in ("beliefs", "actions", "observations")
            if key in payload
            and isinstance(payload[key], (list, tuple, np.ndarray))
            and np.ndim(payload[key]) > 0
        ]
        for key in ("beliefs_by_agent", "beliefs_by_factor"):
            if isinstance(payload.get(key), dict):
                lengths.extend(
                    len(trace)
                    for trace in payload[key].values()
                    if isinstance(trace, (list, tuple, np.ndarray))
                    and np.ndim(trace) > 0
                )
        return max(lengths) if lengths else None

    for key, data in framework_data.items():
        framework = _normalize_framework_name(str(data.get("framework", key)))
        agg = aggregated.setdefault(
            framework,
            {
                "execution_times": [],
                "steps_completed": [],
                "success_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "total_count": 0,
                "invalid_timing_count": 0,
                "missing_timing_count": 0,
                "missing_steps_count": 0,
            },
        )
        if "results" in data:
            rows = data["results"]
            if not isinstance(rows, (list, tuple)):
                raise ValueError("Execution results must be an attempt sequence")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each execution attempt must be a mapping")
                agg["total_count"] += 1
                status = str(row.get("status", "")).lower()
                skipped = bool(row.get("skipped")) or status in {
                    "skipped",
                    "unsupported",
                }
                success = (
                    row.get("success") is True
                    or "success" not in row
                    and status in {"success", "completed"}
                ) and status in {"", "success", "completed"}
                outcome = (
                    "skipped_count"
                    if skipped
                    else ("success_count" if success else "failed_count")
                )
                agg[outcome] += 1
                if "execution_time" not in row:
                    agg["missing_timing_count"] += 1
                elif measured_time(row["execution_time"]):
                    agg["execution_times"].append(float(row["execution_time"]))
                else:
                    agg["invalid_timing_count"] += 1
                payload = row.get("simulation_data", row)
                if payload is row and len(rows) == 1:
                    payload = data.get("simulation_data", row)
                steps = measured_steps(payload)
                if steps is None:
                    agg["missing_steps_count"] += 1
                else:
                    agg["steps_completed"].append(steps)
        elif "total_count" in data:
            # The analysis API already aggregated attempts; retain its counts
            # instead of inventing one new attempt per framework container.
            for count in (
                "total_count",
                "success_count",
                "skipped_count",
                "failed_count",
            ):
                value = data.get(count, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid operational {count}")
                agg[count] += value
            for value in data.get("execution_times", []):
                if measured_time(value):
                    agg["execution_times"].append(float(value))
                else:
                    agg["invalid_timing_count"] += 1
            agg["invalid_timing_count"] += data.get("invalid_execution_time_count", 0)
            records = data.get("scientific_records", [])
            payloads = [record.get("payload", {}) for record in records]
            if not records:
                payloads = [data.get("simulation_data", data)]
            for payload in payloads:
                steps = measured_steps(payload)
                if steps is not None:
                    agg["steps_completed"].append(steps)
            agg["missing_timing_count"] += max(
                0, data["total_count"] - len(data.get("execution_times", []))
            )
            agg["missing_steps_count"] += max(
                0,
                data["total_count"]
                - sum(measured_steps(p) is not None for p in payloads),
            )
        else:
            agg["unavailable_attempts"] = (
                "No execution attempt rows or aggregate count supplied"
            )

    if not aggregated:
        raise ValueError("No framework data for comparison")

    frameworks = sorted(aggregated.keys())
    for fw in frameworks:
        agg = aggregated[fw]
        diagnostics["frameworks"][fw] = {
            **agg,
            "unavailable_metrics": {
                quantity: "No valid measured quantities supplied"
                for quantity in ("execution_times", "steps_completed")
                if not agg[quantity]
            },
        }
    _diagnostic(output_path.with_suffix(".operational.json"), diagnostics)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(frameworks)))
    for index, fw in enumerate(frameworks):
        agg = aggregated[fw]
        for ax, quantity in zip(axes[:2], ("execution_times", "steps_completed")):
            if agg[quantity]:
                ax.bar(index, float(np.mean(agg[quantity])), color=colors[index])
            else:
                ax.text(
                    index,
                    0.05,
                    "Unavailable",
                    ha="center",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                )
        if agg["total_count"]:
            rate = agg["success_count"] / agg["total_count"]
            axes[2].bar(index, rate, color=colors[index])
            axes[2].text(
                index,
                rate + 0.03,
                f"{agg['success_count']}/{agg['total_count']}\nskipped: {agg['skipped_count']}",
                ha="center",
            )
        else:
            axes[2].text(index, 0.05, "Unavailable", ha="center", rotation=90)
    for ax, ylabel, title in zip(
        axes,
        (
            "Mean measured runtime (s)",
            "Mean observed trace length",
            "Successful / all attempts",
        ),
        (
            "Observed Runtime Aggregate",
            "Observed Trace Length Aggregate",
            "Execution Outcomes",
        ),
    ):
        ax.set_xlabel("Framework")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(len(frameworks)))
        ax.set_xticklabels(frameworks, rotation=45, ha="right")
    axes[2].set_ylim(-0.1, 1.3)
    plt.suptitle(
        "Operational Backend Totals (workloads/environments may differ)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_confidence_comparison(
    framework_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> List[str]:
    """Compare validated categorical probabilities only within an admitted model."""
    groups, diagnostics = _plot_groups(framework_data)
    outputs: list[str] = []
    if MATPLOTLIB_AVAILABLE:
        for model_id, group in groups.items():
            if len(group) < 2:
                diagnostics["excluded"][model_id] = [
                    "confidence comparison requires two compatible frameworks"
                ]
                continue
            views = next(iter(group.values()))["views"]
            categorical = {
                name
                for name, view in views.items()
                if model_family(view) != "continuous"
            }
            if not categorical:
                diagnostics["excluded"][model_id] = [
                    "Gaussian means are not categorical probabilities; covariance plots provide uncertainty"
                ]
                continue
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for framework, record in group.items():
                for name in sorted(categorical):
                    beliefs = np.asarray(record["views"][name]["beliefs"], dtype=float)
                    axes[0].plot(beliefs.max(axis=1), label=f"{framework}: {name}")
                    axes[1].plot(_entropy(beliefs), label=f"{framework}: {name}")
            axes[0].set_ylabel("Maximum categorical probability")
            axes[0].set_ylim(0, 1)
            axes[1].set_ylabel("Categorical entropy (nats)")
            for ax in axes:
                ax.set_title(model_id)
                ax.set_xlabel("Trace position")
                ax.legend()
            _save(_model_path(output_path, model_id, len(groups)), outputs)
    else:
        diagnostics["unavailable"] = "matplotlib unavailable"
    _diagnostic(output_path.with_suffix(".admission.json"), diagnostics)
    return outputs


def generate_efe_convergence_comparison(
    framework_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> List[str]:
    """Plot aligned scalar EFE under the same declared numerical convention.

    Policy/action arrays are not silently averaged into scalar trajectories.
    A declining EFE series alone does not establish inference convergence.
    """
    groups, diagnostics = _plot_groups(framework_data)
    outputs: list[str] = []
    if MATPLOTLIB_AVAILABLE:
        for model_id, group in groups.items():
            if len(group) < 2:
                diagnostics["excluded"][model_id] = [
                    "EFE comparison requires two compatible frameworks"
                ]
                continue
            rows = list(group.values())
            witnesses = [
                compare_scientific_pair(left, right)
                for left, right in combinations(rows, 2)
            ]
            if any("expected_free_energy" not in witness for witness in witnesses):
                diagnostics["excluded"][model_id] = [
                    str(
                        witness.get("unavailable_metrics", {}).get(
                            "expected_free_energy", "EFE unavailable"
                        )
                    )
                    for witness in witnesses
                    if "expected_free_energy" not in witness
                ]
                continue
            fig, ax = plt.subplots(figsize=(12, 5))
            for framework, record in group.items():
                ax.plot(record["payload"]["expected_free_energy"], label=framework)
            ax.set_title(f"{model_id}: scalar expected free energy")
            ax.set_xlabel("Trace position")
            ax.set_ylabel(
                f"Expected free energy ({rows[0]['metadata']['expected_free_energy_convention']})"
            )
            ax.legend()
            _save(_model_path(output_path, model_id, len(groups)), outputs)
    else:
        diagnostics["unavailable"] = "matplotlib unavailable"
    _diagnostic(output_path.with_suffix(".admission.json"), diagnostics)
    return outputs


def generate_framework_radar(
    exec_summary_path: Path,
    framework_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> List[str]:
    """Retain the public entry point without asserting uncalibrated proxy scores."""
    _diagnostic(
        output_path.with_suffix(".admission.json"),
        {
            "schema_version": "gnn.scientific_plot_admission/v1",
            "status": "unavailable",
            "reason": "No calibrated belief-quality, data-richness, or matched-environment speed score is defined; descriptive operational aggregates remain available",
        },
    )
    return []
