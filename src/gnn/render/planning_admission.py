"""Admission for source-authored horizons on the one-step planning adapters.

This checks declared intent before canonicalization can hide it. It neither
selects a pruned policy set nor certifies filtering/smoothing equivalence.
Versioned execution contracts and cpomdp own their separate planning semantics.
"""

from typing import Any

from gnn.render.framework_registry import get_planning_admission_frameworks

#: Frames whose declared planning semantics admit one-step declarations
#: (``render_capabilities["planning"] == ("one_step_only",)``).  Derived from
#: the canonical registry so the admission set and the declaration cannot
#: drift; a deeper authored horizon is refused here, not reduced to one step.
ONE_STEP_PLANNERS = frozenset(get_planning_admission_frameworks())
ALTERNATIVE_JAX_TARGETS = frozenset({"jax_pomdp", "jax_combined"})
HORIZON_KEYS = ("planning_horizon", "policy_len", "policy_horizon")
PLANNING_KEYS = (*HORIZON_KEYS, "num_policies", "policies", "policy_sequences")


def unsupported_planning(spec: dict[str, Any], framework: str) -> str | None:
    """Refuse declarations the selected one-step planner cannot preserve.

    Read all supported parameter containers independently: precedence must not
    erase a conflicting authored declaration. Integers are never coerced.
    Explicit policy lists are refused even when their length matches the action
    count, since these adapters do not consume such lists or their ordering.
    """
    if framework not in ONE_STEP_PLANNERS | ALTERNATIVE_JAX_TARGETS:
        return None
    containers = [("spec", spec)]
    for name in (
        "model_parameters",
        "initialparameterization",
        "initial_parameterization",
    ):
        value = spec.get(name)
        if isinstance(value, dict):
            containers.append((name, value))
    parameters = spec.get("parameters")
    if isinstance(parameters, list):
        for index, parameter in enumerate(parameters):
            if isinstance(parameter, dict) and parameter.get("name") in (
                *PLANNING_KEYS,
                "num_actions",
            ):
                containers.append(
                    (
                        f"parameters[{index}]",
                        {parameter["name"]: parameter.get("value")},
                    )
                )
    declared = [
        (f"{name}.{key}", key, values[key])
        for name, values in containers
        for key in PLANNING_KEYS
        if key in values
    ]
    if not declared:
        return None
    prefix = f"unsupported-planning: {framework} "
    if framework in ALTERNATIVE_JAX_TARGETS:
        return (
            prefix
            + "generator does not consume source-authored planning declarations; its solver/demo defaults cannot certify them"
        )
    for path, key, value in declared:
        if key in {"policies", "policy_sequences"}:
            return (
                prefix
                + f"does not consume authored {path}; policy identity/order cannot be discarded"
            )
        if type(value) is not int or value < 1:
            return prefix + f"requires {path} to be a positive integer, got {value!r}"
    horizons = [(path, value) for path, key, value in declared if key in HORIZON_KEYS]
    if len({value for _, value in horizons}) > 1:
        return prefix + f"cannot resolve conflicting horizon declarations: {horizons!r}"
    if horizons and horizons[0][1] != 1:
        return (
            prefix
            + f"implements one-step planning only; authored {horizons[0][0]}={horizons[0][1]} requires an explicit multi-step policy contract (a policy count does not define a pruned policy set)"
        )
    counts = [(path, value) for path, key, value in declared if key == "num_policies"]
    if counts:
        actions = [
            (f"{name}.num_actions", values["num_actions"])
            for name, values in containers
            if "num_actions" in values
        ]
        if (
            not actions
            or any(type(value) is not int or value < 1 for _, value in actions)
            or len({value for _, value in actions}) != 1
        ):
            return (
                prefix
                + "cannot validate num_policies without one unambiguous positive integer num_actions declaration"
            )
        if any(value != actions[0][1] for _, value in counts):
            return (
                prefix
                + f"requires one policy per action; authored policy counts {counts!r} differ from num_actions={actions[0][1]}"
            )
    return None
