"""RxInfer result normalization and numerical diagnostics.

Consumes native categorical/factor result values without reading files or
rendering plots. Numerical availability uses the established shared owner.
"""

from typing import Any, Dict, List

from ..viz_base import np


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
    if rows and any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Belief traces must have regular rows")
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
