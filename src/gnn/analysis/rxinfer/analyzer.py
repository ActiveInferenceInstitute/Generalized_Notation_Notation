"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.

This module is exercised for the pomdp_gridworld exemplar but is hardened to be
generic across ALL exemplars whose ``simulation_results.json`` conform to the
``rxinfer_simulation_v1`` schema. Different exemplars emit slightly different
mixtures of keys:

* ``beliefs`` / ``beliefs_by_factor`` (dict-shaped: ``{"joint_state": [...]}``)
* ``observations`` / ``observations_by_modality``
* ``true_states`` / ``hidden_states_by_factor``
* ``actions`` / ``actions_by_control_factor``
* ``expected_free_energy`` / ``variational_free_energy`` / ``efe_per_action``
* ``policy_posterior`` (optional)
* ``metrics``, ``validation`` (optional)

The analyzer normalises these into flat arrays and renders a consistent,
comprehensive visualization set for any result that carries the core belief /
observation arrays, skipping optional plots gracefully when their data is
absent rather than raising.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import shared visualization utilities (centralized matplotlib setup)
from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt


def _safe_close_figures(context: str) -> None:
    """Close open matplotlib figures, logging any cleanup failure (SC-17 tier 3).

    Replaces silent ``except: pass`` cleanup swallows so figure-close failures
    are observable instead of being discarded.
    """
    if plt is None:
        return
    try:
        plt.close()
    except (OSError, ValueError) as exc:
        logger.warning("Figure cleanup failed after %s: %s", context, exc)


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------
_CORE_PLOT_TYPES: List[str] = [
    "belief_evolution",
    "obs_vs_true",
    "belief_heatmap",
    "per_factor_beliefs",
    "belief_entropy",
    "accuracy",
    "action_frequencies",
    "belief_convergence",
    "belief_trace",
    "free_energy",
    "observations",
    "efe_per_action_heatmap",
    "convergence_diagnostics",
]


def _first_dict_value(value: Any) -> Any:
    """If value is a non-empty dict, return the first entry's value.

    RxInfer results often store arrays inside a ``{factor_name: [...]}`` dict
    (e.g. ``beliefs_by_factor`` -> ``{"joint_state": [...]}``). Helpers pull out
    the first such array so downstream code sees a plain list.
    """
    if isinstance(value, dict) and value:
        first_key = next(iter(value))
        return value[first_key]
    return value


def _as_flat_list(value: Any) -> List[Any]:
    """Best-effort conversion of a raw JSON value into a flat Python list.

    Handles lists, dicts whose single value is a list, and scalar values
    (wrapped in a single-element list).
    """
    if value is None:
        return []
    value = _first_dict_value(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    # numpy / scalar
    if np is not None and isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, (int, float)):
        return [value]
    return []


def _as_2d_list(value: Any) -> List[List[float]]:
    """Best-effort conversion into a list of sequence rows (2D-ish)."""
    value = _first_dict_value(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    rows: List[List[float]] = []
    for row in value:
        if isinstance(row, (list, tuple)):
            rows.append([float(x) for x in row])
        elif np is not None and isinstance(row, np.ndarray):
            rows.append([float(x) for x in row.tolist()])
        elif isinstance(row, (int, float)):
            rows.append([float(row)])
    return rows


def _normalise_beliefs(data: Dict[str, Any]) -> List[List[float]]:
    """Return beliefs as a list of rows, from either key form."""
    beliefs = data.get("beliefs")
    if beliefs is None:
        beliefs = data.get("beliefs_by_factor")
    rows = _as_2d_list(beliefs)
    if rows and all(np is not None and len(r) != len(rows[0]) for r in rows):
        # ragged rows cannot be plotted together; fall back to empty
        return []
    return rows


def _normalise_obs(data: Dict[str, Any]) -> List[float]:
    obs = data.get("observations")
    if obs is None:
        obs = data.get("observations_by_modality")
    return [float(x) for x in _as_flat_list(obs)]


def _normalise_true_states(data: Dict[str, Any]) -> List[float]:
    states = data.get("true_states")
    if states is None:
        states = data.get("hidden_states_by_factor")
    return [float(x) for x in _as_flat_list(states)]


def _normalise_actions(data: Dict[str, Any]) -> List[float]:
    actions = data.get("actions")
    if actions is None:
        actions = data.get("actions_by_control_factor")
    return _as_flat_list(actions)


def _normalise_free_energy(data: Dict[str, Any]) -> List[float]:
    """Best available expected / variational free energy trace.

    The ``variational_free_energy`` field in ``rxinfer_simulation_v1`` is now a
    per-iteration VFE trace (length = INFERENCE_ITERATIONS), not a per-step
    constant. This is the real convergence diagnostic from RxInfer's
    variational message passing. When ``vfe_per_iteration`` is present it is
    the authoritative source; otherwise we fall back to
    ``variational_free_energy`` or ``expected_free_energy``.
    """
    # Prefer vfe_per_iteration (the explicit per-iteration field)
    vfe_iter = data.get("vfe_per_iteration")
    if vfe_iter is not None:
        vfe_list = _as_flat_list(vfe_iter)
        if vfe_list:
            return [float(x) for x in vfe_list]
    # Fall back to variational_free_energy (also per-iteration now)
    efe = data.get("variational_free_energy")
    if efe is None:
        efe = data.get("expected_free_energy")
    efe_list = _as_flat_list(efe)
    if efe_list:
        return [float(x) for x in efe_list]
    # Fall back to per-action EFE mean across actions per step.
    efe_per_action = _as_2d_list(data.get("efe_per_action"))
    if efe_per_action and np is not None:
        arr = np.asarray(efe_per_action, dtype=float)
        if arr.ndim == 2:
            return [float(v) for v in np.mean(arr, axis=1)]
    return []


def _normalise_policy_posterior(data: Dict[str, Any]) -> List[List[float]]:
    pp = data.get("policy_posterior")
    if pp is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            pp = metrics.get("policy_posterior")
    return _as_2d_list(pp)


def _normalise_efe_per_action(data: Dict[str, Any]) -> List[List[float]]:
    eaa = data.get("efe_per_action")
    if eaa is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            eaa = metrics.get("efe_per_action")
    return _as_2d_list(eaa)


def _compute_convergence_diagnostics(free_energy: List[float]) -> Dict[str, Any]:
    """Compute convergence diagnostics from a per-iteration VFE trace.

    The ``variational_free_energy`` / ``vfe_per_iteration`` trace (length =
    INFERENCE_ITERATIONS) is the real convergence signal from RxInfer's
    variational message passing. This helper derives three diagnostics from it:

    * ``vfe_slope`` — slope of a linear regression over the *full* trace. A
      sustained negative slope indicates the variational bound is still
      improving; a slope near zero indicates the trace has flattened.
    * ``convergence_rate`` — slope of a linear regression over the *last 10*
      iterations. The tail slope estimates how fast VFE is still moving once
      the bulk of the descent is done.
    * ``iterations_to_convergence`` — first (1-indexed) iteration at which the
      absolute step-to-step VFE change drops below ``1e-4``, or ``None`` if the
      trace never settles. A lower value means the model posterior converged
      earlier in inference.

    Returns a ``convergence_diagnostics`` dict suitable for storing under a
    ``convergence_diagnostics`` key in the analysis results. Missing / empty
    traces yield a dict of ``None`` values (never raising).
    """
    diagnostics: Dict[str, Any] = {
        "vfe_slope": None,
        "convergence_rate": None,
        "iterations_to_convergence": None,
        "total_iterations": int(len(free_energy)),
    }
    if np is None or not free_energy:
        return diagnostics

    trace = np.asarray([float(x) for x in free_energy], dtype=float)
    n = trace.size
    if n == 0:
        return diagnostics

    iterations = np.arange(n, dtype=float)

    # Full-trace linear regression slope (vfe_slope)
    if n >= 2:
        slope, _intercept = np.polyfit(iterations, trace, 1)
        diagnostics["vfe_slope"] = float(slope)

    # Tail slope over the last 10 iterations (convergence_rate)
    tail = min(10, n)
    if tail >= 2:
        tail_iters = iterations[-tail:]
        tail_vfe = trace[-tail:]
        rate, _intercept = np.polyfit(tail_iters, tail_vfe, 1)
        diagnostics["convergence_rate"] = float(rate)

    # First iteration where the step-to-step change settles below threshold.
    # deltas[k] = |VFE[k+1] - VFE[k]|; a settle at deltas[k] corresponds to the
    # (k+2)-th 1-indexed iteration.
    if n >= 2:
        deltas = np.abs(np.diff(trace))
        settled = np.flatnonzero(deltas < 1e-4)
        if settled.size:
            diagnostics["iterations_to_convergence"] = int(settled[0] + 2)

    return diagnostics


def summarize_strategy_validation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize strategy-declared validation fields present in the results (FP-8).

    Reads ``runtime_metadata.model_kind`` (defaulting to ``"flat"`` for
    payloads written before the field existed), asks the registered
    render-side ``ModelStrategy`` which validation fields it contributes via
    ``get_validation_fields()``, and returns ``{field: value}`` for every
    declared field actually present in the results ``validation`` dict.

    Loud on an unknown kind (``ValueError``); tolerant (field simply absent
    from the summary) when a declared field is missing from the results.
    Every registered strategy declares its fields natively.
    """
    from gnn.render.pomdp_contract import ModelKind
    from gnn.render.rxinfer.model_strategies import get_model_strategy

    kind_value = str((data.get("runtime_metadata") or {}).get("model_kind", "flat"))
    try:
        kind = ModelKind(kind_value)
    except ValueError as exc:
        raise ValueError(
            f"unknown model_kind {kind_value!r} in runtime_metadata; "
            f"expected one of {[member.value for member in ModelKind]}"
        ) from exc

    fields = get_model_strategy(kind).get_validation_fields()

    validation = data.get("validation")
    if not isinstance(validation, dict):
        return {}
    return {field: validation[field] for field in fields if field in validation}


def compute_per_factor_beliefs(data: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    """Recover per-factor belief marginals from a flattened joint belief trace.

    Multi-agent / multi-factor models are rendered onto a single flat joint
    state space: the renderer enumerates ``itertools.product`` over
    ``state_factors`` in list order (C order, first factor slowest-varying) and
    builds A / B / D against that enumeration. A 256-state joint belief for
    ``(s_agent1=4, s_agent2=4, s_joint=16)`` is therefore a reshapeable
    ``4 x 4 x 16`` tensor, and each factor's marginal is the sum over the other
    axes.

    Args:
        data: An ``rxinfer_simulation_v1`` results dict. The factor structure is
            read from ``model_parameters.state_factors``, a list of
            ``{"name": str, "size": int}`` echoed from the GNN spec.

    Returns:
        A mapping of factor name to a per-timestep list of marginals, covering
        only factors with ``size > 1``. Size-1 factors participate in the
        reshape (they carry a real axis in the flattening) but are omitted from
        the output because a one-state distribution is always ``[1.0]``.

        An **empty dict** signals structural absence rather than failure, in
        three cases: ``state_factors`` is missing (flat models, or artifacts
        written before the key existed), there are no beliefs to decompose, or
        fewer than two factors have ``size > 1`` (the joint space *is* the
        single factor, so the marginal would just be the belief itself).

    Raises:
        ValueError: When ``state_factors`` is present but cannot describe the
            beliefs — a malformed descriptor, duplicate factor names, ragged
            belief rows, a size product that contradicts the joint width, or a
            timestep carrying no probability mass. These are contract
            violations between renderer and analyzer, never quietly absorbed.
    """
    model_parameters = data.get("model_parameters")
    if not isinstance(model_parameters, dict):
        return {}
    factors = model_parameters.get("state_factors")
    if not isinstance(factors, list) or not factors:
        return {}

    beliefs = _normalise_beliefs(data)
    if not beliefs:
        return {}

    names: List[str] = []
    sizes: List[int] = []
    for index, factor in enumerate(factors):
        if not isinstance(factor, dict) or factor.get("name") is None:
            raise ValueError(f"state_factors[{index}] is missing a 'name': {factor!r}")
        if factor.get("size") is None:
            raise ValueError(f"state_factors[{index}] is missing a 'size': {factor!r}")
        names.append(str(factor["name"]))
        sizes.append(int(factor["size"]))

    informative = [index for index, size in enumerate(sizes) if size > 1]
    if len(informative) < 2:
        return {}

    if len(set(names)) != len(names):
        raise ValueError(f"state_factors carry duplicate factor names: {names}")

    if np is None:
        raise RuntimeError("numpy is required to compute per-factor beliefs")

    joint_size = 1
    for size in sizes:
        joint_size *= size
    belief_width = len(beliefs[0])
    if joint_size != belief_width:
        # Whether a size mismatch is a contract violation depends on how the
        # payload was rendered. Joint-composed payloads (multi-agent) MUST
        # decompose — a mismatch there is renderer/analyzer breakage and
        # raises. For every other kind the state_factors list is descriptive
        # metadata that does not define the belief space: flat exemplars
        # legitimately declare next-state aliases (``s``/``s_prime``) as two
        # "factors" over a 3-state chain, and native factored/hierarchical
        # payloads carry per-factor marginals directly in beliefs_by_factor
        # instead of a flattened joint. Those are structurally not
        # decomposable joints, so the answer is {} — not an error.
        kind = str((data.get("runtime_metadata") or {}).get("model_kind", "flat"))
        if kind == "multi_agent":
            raise ValueError(
                f"state_factors {list(zip(names, sizes))} imply {joint_size} "
                f"joint states but beliefs carry {belief_width} per timestep "
                f"in a joint-composed multi_agent payload"
            )
        return {}

    marginals: Dict[str, List[List[float]]] = {names[i]: [] for i in informative}
    for step, row in enumerate(beliefs):
        if len(row) != belief_width:
            raise ValueError(
                f"belief row at timestep {step} has width {len(row)}, "
                f"expected {belief_width}"
            )
        q_nd = np.asarray(row, dtype=float).reshape(sizes)
        for i in informative:
            other_axes = tuple(j for j in range(len(sizes)) if j != i)
            marginal = q_nd.sum(axis=other_axes)
            mass = float(marginal.sum())
            if mass <= 0.0:
                raise ValueError(
                    f"belief at timestep {step} carries no probability mass "
                    f"for factor '{names[i]}'"
                )
            # Renormalise against accumulated float drift; the joint already
            # sums to ~1 so this is a correction, not a rescue.
            marginals[names[i]].append([float(v) for v in marginal / mass])

    return marginals


# ---------------------------------------------------------------------------
# Current-model helpers
# ---------------------------------------------------------------------------
def _current_rxinfer_models(execution_dir: Path) -> set[str] | None:
    """Return model names for RxInfer entries in the current Step 12 summary."""
    summary_file = execution_dir / "summaries" / "execution_summary.json"
    if not summary_file.exists():
        summary_file = execution_dir / "execution_summary.json"
    if not summary_file.exists():
        return None
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    details = data.get("execution_details")
    if not isinstance(details, list):
        details = data.get("execution_results", [])
    if not isinstance(details, list):
        return None
    models = {
        str(detail.get("model_name"))
        for detail in details
        if isinstance(detail, dict)
        and str(detail.get("framework", "")).lower() == "rxinfer"
        and detail.get("model_name")
    }
    return models or None


def _latest_current_results_file(sim_data_dir: Path) -> Path | None:
    """Select one current RxInfer result file from a simulation_data directory."""
    candidates = sorted(
        sim_data_dir.glob("*simulation_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("schema_version") == "rxinfer_simulation_v1":
            return candidate
    return candidates[0] if candidates else None


def generate_analysis_from_logs(
    execution_dir: Path, output_dir: Path, verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from RxInfer execution logs.

    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging

    Returns:
        List of generated visualization file paths
    """
    visualizations: list[Any] = []

    try:
        # Find RxInfer execution results
        current_models = _current_rxinfer_models(execution_dir)
        rxinfer_dirs = list(execution_dir.glob("*/rxinfer"))

        for rxinfer_dir in rxinfer_dirs:
            model_name = rxinfer_dir.parent.name
            if current_models and model_name not in current_models:
                continue
            sim_data_dir = rxinfer_dir / "simulation_data"
            if sim_data_dir.exists():
                # Load simulation results
                results_file = _latest_current_results_file(sim_data_dir)
                if results_file is not None:
                    try:
                        with open(results_file, "r") as f:
                            data = json.load(f)

                        viz_files = create_rxinfer_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)

                        # D7: also produce an animated GIF alongside the PNGs;
                        # multi-agent results get one GIF per agent.
                        try:
                            from .gif_animator import (
                                generate_gif_animation,
                                per_agent_views,
                            )

                            agent_views = per_agent_views(data)
                            gif_jobs = (
                                [
                                    (
                                        view,
                                        f"{model_name}_{agent}",
                                    )
                                    for agent, view in agent_views.items()
                                ]
                                if agent_views
                                else [(data, model_name)]
                            )
                            for gif_data, gif_label in gif_jobs:
                                gif_path: Path = output_dir / (
                                    f"{gif_label}_rxinfer_animation.gif"
                                )
                                gif_file = generate_gif_animation(
                                    gif_data, gif_path, model_name=gif_label
                                )
                                if gif_file:
                                    visualizations.append(gif_file)
                        except Exception as e:
                            logger.warning(
                                f"GIF generation failed for {model_name}: {e}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to process {results_file}: {e}")

    except Exception as e:
        logger.error(f"RxInfer analysis failed: {e}")

    return visualizations


def create_rxinfer_visualizations(
    data: Dict[str, Any], output_dir: Path, model_name: str, verbose: bool = False
) -> List[str]:
    """
    Create visualizations from RxInfer simulation data.

    Produces a comprehensive, consistent visualization set for any
    ``rxinfer_simulation_v1`` result. Core plots (belief evolution,
    observation-vs-true, belief heatmap, belief entropy) are emitted whenever
    the required arrays exist; optional plots (accuracy, action frequencies,
    belief convergence, belief trace, free energy, observations) are skipped
    gracefully when their source data is missing — never raising.

    Args:
        data: Simulation results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE or plt is None or np is None:
        logger.warning("Matplotlib unavailable, skipping RxInfer visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Normalise data into flat arrays (tolerates dict-shaped / missing keys)
    beliefs = _normalise_beliefs(data)
    observations = _normalise_obs(data)
    true_states = _normalise_true_states(data)
    actions = _normalise_actions(data)
    free_energy = _normalise_free_energy(data)
    efe_per_action = _normalise_efe_per_action(data)

    # --- Convergence diagnostics (D5): derived from the per-iteration VFE trace
    convergence_diagnostics = _compute_convergence_diagnostics(free_energy)
    # Store under a dedicated key so the diagnostics ride along in the results.
    data["convergence_diagnostics"] = convergence_diagnostics

    # --- Per-factor belief marginals (D4): empty for flat / single-factor models
    per_factor_beliefs = compute_per_factor_beliefs(data)
    data["per_factor_beliefs"] = per_factor_beliefs

    # --- Strategy-declared validation fields (FP-8): field -> value summary
    data["validation_summary"] = summarize_strategy_validation(data)

    beliefs_arr = np.asarray(beliefs, dtype=float) if beliefs else np.zeros((0, 0))
    have_beliefs = beliefs_arr.ndim >= 1
    have_2d_beliefs = beliefs_arr.ndim == 2 and beliefs_arr.shape[0] > 0

    def _record(plot_type: str) -> Optional[str]:
        viz_file = output_dir / f"{model_name}_rxinfer_{plot_type}.png"
        if viz_file.exists() and viz_file.stat().st_size > 0:
            visualizations.append(str(viz_file))
            logger.info(f"Generated {plot_type}: {viz_file.name}")
            return str(viz_file)
        return None

    # 1. Belief Evolution Plot
    if have_beliefs:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            if have_2d_beliefs:
                for i in range(beliefs_arr.shape[1]):
                    ax.plot(beliefs_arr[:, i], label=f"State {i + 1}", linewidth=2)
            else:
                ax.plot(beliefs_arr, label="Belief", linewidth=2)

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief Probability")
            ax.set_title(f"RxInfer Belief Evolution - {model_name}", fontweight="bold")
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_evolution.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief evolution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief plot: {e}")
            _safe_close_figures(f"{model_name} belief plot")

    # 2. Observation vs True State Plot
    if observations and true_states:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(len(observations))
            ax.scatter(x, observations, label="Observations", alpha=0.7, s=50)
            ax.scatter(x, true_states, label="True States", alpha=0.7, s=50, marker="x")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State/Observation")
            ax.set_title(
                f"RxInfer Observations vs True States - {model_name}", fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_obs_vs_true.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated obs vs true: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create obs plot: {e}")
            _safe_close_figures(f"{model_name} obs plot")

    # 3. Belief Heatmap (2D visualization of beliefs over time)
    if have_2d_beliefs and beliefs_arr.shape[0] > 1:
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(
                beliefs_arr.T,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State")
            ax.set_title(f"RxInfer Belief Heatmap - {model_name}", fontweight="bold")
            ax.set_yticks(range(beliefs_arr.shape[1]))
            ax.set_yticklabels([f"State {i + 1}" for i in range(beliefs_arr.shape[1])])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Belief Probability", fontweight="bold")

            viz_file = output_dir / f"{model_name}_rxinfer_belief_heatmap.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief heatmap: {e}")
            _safe_close_figures(f"{model_name} belief heatmap")

    # 3b. Per-Factor Belief Marginals (D4): one small-multiple panel per factor
    if per_factor_beliefs:
        factor_names = list(per_factor_beliefs)
        n_factors = len(factor_names)
        n_cols = min(3, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols
        fig, panels = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.0 * n_cols, 3.5 * n_rows),
            squeeze=False,
            layout="constrained",
        )
        for index, name in enumerate(factor_names):
            ax = panels[index // n_cols][index % n_cols]
            trajectory = np.asarray(per_factor_beliefs[name], dtype=float)
            factor_size = trajectory.shape[1]
            for state_index in range(factor_size):
                ax.plot(
                    trajectory[:, state_index],
                    linewidth=2,
                    label=f"State {state_index + 1}",
                )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Marginal Probability")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{name} ({factor_size} states)", fontweight="bold")
            ax.grid(True, alpha=0.3)
            # Timesteps are discrete — no fractional ticks.
            ax.locator_params(axis="x", integer=True)
            if factor_size <= 8:
                ax.legend(fontsize=8)
        for index in range(n_factors, n_rows * n_cols):
            panels[index // n_cols][index % n_cols].axis("off")

        fig.suptitle(
            f"RxInfer Per-Factor Belief Marginals - {model_name}", fontweight="bold"
        )
        viz_file = output_dir / f"{model_name}_rxinfer_per_factor_beliefs.png"
        plt.savefig(viz_file, dpi=300, bbox_inches="tight")
        plt.close()
        visualizations.append(str(viz_file))
        logger.info(f"Generated per-factor beliefs: {viz_file.name}")

    # 4. Belief Entropy (uncertainty tracking over time)
    if have_2d_beliefs:
        try:
            # Calculate entropy for each timestep: H = -sum(p * log(p))
            epsilon = 1e-10
            beliefs_clipped = np.clip(beliefs_arr, epsilon, 1.0)
            entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped), axis=1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(entropy, "purple", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(entropy)), entropy, alpha=0.3, color="purple")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief Entropy (bits)")
            ax.set_title(
                f"RxInfer Belief Uncertainty - {model_name}", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            max_entropy = np.log2(beliefs_arr.shape[1])
            ax.axhline(
                y=max_entropy,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Max Entropy ({max_entropy:.2f})",
            )
            ax.legend()

            viz_file = output_dir / f"{model_name}_rxinfer_belief_entropy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief entropy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create entropy plot: {e}")
            _safe_close_figures(f"{model_name} entropy plot")

    # 5. Inference Accuracy (if we have true states + 2D beliefs)
    if have_2d_beliefs and true_states:
        try:
            inferred_states = np.argmax(
                beliefs_arr, axis=1
            )  # 0-indexed (matching Julia output)
            true_arr = np.asarray(true_states[: len(inferred_states)], dtype=float)
            # true_states are 0-indexed from the generated script
            n = min(len(inferred_states), len(true_arr))

            matches = (inferred_states[:n] == true_arr[:n]).astype(int)
            cumulative_accuracy = np.cumsum(matches) / (np.arange(n) + 1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(cumulative_accuracy * 100, "green", linewidth=2)
            ax.fill_between(
                range(len(cumulative_accuracy)),
                cumulative_accuracy * 100,
                alpha=0.3,
                color="green",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Cumulative Accuracy (%)")
            ax.set_title(
                f"RxInfer Inference Accuracy - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

            if len(cumulative_accuracy) > 0:
                final_acc = cumulative_accuracy[-1] * 100
                ax.axhline(y=final_acc, color="navy", linestyle="--", alpha=0.5)
                ax.text(
                    len(cumulative_accuracy) - 1,
                    final_acc + 3,
                    f"Final: {final_acc:.1f}%",
                    ha="right",
                    fontweight="bold",
                )

            viz_file = output_dir / f"{model_name}_rxinfer_accuracy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated inference accuracy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create accuracy plot: {e}")
            _safe_close_figures(f"{model_name} accuracy plot")

    # 6. Action Frequencies (bar chart, if actions exist)
    if actions:
        try:
            actions_arr = np.asarray(actions, dtype=float)
            unique, counts = np.unique(actions_arr, return_counts=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(unique, counts, color="skyblue", edgecolor="navy")
            ax.set_xlabel("Action")
            ax.set_ylabel("Frequency")
            ax.set_title(
                f"RxInfer Action Frequencies - {model_name}", fontweight="bold"
            )
            ax.set_xticks(unique)
            ax.grid(True, alpha=0.3, axis="y")

            viz_file = output_dir / f"{model_name}_rxinfer_action_frequencies.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action frequencies: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action frequencies plot: {e}")
            _safe_close_figures(f"{model_name} action frequencies plot")

    # 6b. EFE Per Action Heatmap (structured EFE landscape over time)
    if efe_per_action:
        try:
            eaa_arr = np.asarray(efe_per_action, dtype=float)
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(
                eaa_arr.T,
                aspect="auto",
                cmap="RdBu_r",
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Action")
            ax.set_title(f"RxInfer EFE per Action - {model_name}")
            if eaa_arr.shape[1] > 0:
                ax.set_yticks(range(eaa_arr.shape[1]))
                ax.set_yticklabels([f"A{i + 1}" for i in range(eaa_arr.shape[1])])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Expected Free Energy")

            viz_file = output_dir / f"{model_name}_rxinfer_efe_per_action_heatmap.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated EFE per action heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create EFE per action heatmap: {e}")
            _safe_close_figures(f"{model_name} EFE per action heatmap")

    # 7. Belief Convergence (max belief probability over time)
    if have_2d_beliefs:
        try:
            max_probabilities = np.max(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(max_probabilities, color="brown", linewidth=2)
            ax.fill_between(
                range(len(max_probabilities)),
                max_probabilities,
                alpha=0.3,
                color="brown",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Max State Probability")
            ax.set_title(
                f"RxInfer Belief Convergence - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_convergence.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief convergence: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief convergence plot: {e}")
            _safe_close_figures(f"{model_name} belief convergence plot")

    # 8. Belief Trace (inferred most-likely state over time vs true state)
    if have_2d_beliefs:
        try:
            inferred_states = np.argmax(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(inferred_states, "o-", label="Inferred State", color="darkorange")
            if true_states:
                ax.plot(
                    np.asarray(true_states[: len(inferred_states)], dtype=float),
                    "s--",
                    label="True State",
                    color="steelblue",
                )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State Index")
            ax.set_title(f"RxInfer Belief Trace - {model_name}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_trace.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief trace: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief trace plot: {e}")
            _safe_close_figures(f"{model_name} belief trace plot")

    # 9. Free Energy (per-iteration VFE or expected free energy over time)
    if free_energy:
        try:
            fe_arr = np.asarray(free_energy, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(fe_arr, color="crimson", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(fe_arr)), fe_arr, alpha=0.3, color="crimson")
            # Label accurately: VFE is per-iteration when from variational_free_energy
            fe_key = (
                "vfe_per_iteration"
                if data.get("vfe_per_iteration") is not None
                else "variational_free_energy"
                if data.get("variational_free_energy") is not None
                else "expected_free_energy"
            )
            is_per_iteration = fe_key in (
                "vfe_per_iteration",
                "variational_free_energy",
            )
            xlabel = "Inference Iteration" if is_per_iteration else "Time Step"
            ylabel = "Variational Free Energy" if is_per_iteration else "Free Energy"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            title_prefix = "VFE (per-iteration)" if is_per_iteration else "Free Energy"
            ax.set_title(f"RxInfer {title_prefix} - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_free_energy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated free energy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create free energy plot: {e}")
            _safe_close_figures(f"{model_name} free energy plot")

    # 9b. Convergence Diagnostics (D5): VFE slope / tail rate / iterations to converge
    if free_energy:
        try:
            diag = convergence_diagnostics
            fe_arr = np.asarray(free_energy, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(fe_arr, color="crimson", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(fe_arr)), fe_arr, alpha=0.3, color="crimson")
            ax.set_xlabel("Inference Iteration")
            ax.set_ylabel("Variational Free Energy")
            ax.set_title(
                f"RxInfer Convergence Diagnostics - {model_name}", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            # Vertical marker at the iteration where VFE first settles
            itc = diag.get("iterations_to_convergence")
            if itc is not None and 1 <= itc <= len(fe_arr):
                ax.axvline(x=itc - 1, color="navy", linestyle="--", alpha=0.7)
                ax.text(
                    itc - 1,
                    fe_arr.max(),
                    f"Converged @ iter {itc}",
                    ha="right",
                    color="navy",
                    fontsize=9,
                    fontweight="bold",
                )

            # Annotate the derived diagnostics
            slope = diag.get("vfe_slope")
            rate = diag.get("convergence_rate")
            annotation_lines: List[str] = []
            if slope is not None:
                annotation_lines.append(f"VFE slope        : {slope:.4g}")
            if rate is not None:
                annotation_lines.append(f"Conv. rate (last10): {rate:.4g}")
            annotation_lines.append(
                f"Converged iter   : {itc if itc is not None else 'n/a'}"
            )
            ax.text(
                0.02,
                0.98,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round", facecolor="lightgoldenrodyellow", alpha=0.6
                ),
            )

            viz_file = output_dir / f"{model_name}_rxinfer_convergence_diagnostics.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated convergence diagnostics: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create convergence diagnostics plot: {e}")
            _safe_close_figures(f"{model_name} convergence diagnostics plot")

    # 10. Observations (raw observation trace over time)
    if observations:
        try:
            obs_arr = np.asarray(observations, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(obs_arr, "o-", color="teal", linewidth=2, markersize=4)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Observation")
            ax.set_title(f"RxInfer Observations - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_observations.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated observations: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create observations plot: {e}")
            _safe_close_figures(f"{model_name} observations plot")

    return visualizations


def extract_simulation_data(
    execution_dir: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract RxInfer simulation data from execution outputs.

    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data: dict[str, Any] = {
        "beliefs": [],
        "observations": [],
        "true_states": [],
        "time_steps": 0,
        "model_name": "",
        "framework": "rxinfer",
    }

    try:
        sim_data_dir = execution_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                with open(results_files[0], "r") as f:
                    results = json.load(f)
                data.update(results)

    except Exception as e:
        logger.warning(f"Failed to extract RxInfer data: {e}")

    return data


__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "compute_per_factor_beliefs",
    "summarize_strategy_validation",
    "extract_simulation_data",
]
