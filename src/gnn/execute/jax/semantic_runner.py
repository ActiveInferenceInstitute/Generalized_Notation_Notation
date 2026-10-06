"""Native JAX inference for declared block and timed-controller semantics.

Only physical external states are sampled. Belief normalization is Bayesian
conditioning, never repair of authored probability tables. Traces distinguish
observations, controlled transitions, resets and approximate soft messages.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def posterior(weights: Any) -> Any:
    """Condition on positive evidence, refusing impossible observations."""
    total = jnp.sum(weights)
    if not np.isfinite(float(total)) or float(total) <= 0:
        raise ValueError("Impossible observation: zero or nonfinite evidence")
    return weights / total


def action_scores(a: Any, b: Any, c: Any, q: Any) -> Any:
    """One-step raw utility plus state/outcome mutual information (nats)."""
    predicted = jnp.einsum("ija,j->ia", b, q)
    outcomes = a @ predicted
    safe_a = jnp.where(a > 0, a, 1)
    safe_o = jnp.where(outcomes > 0, outcomes, 1)
    info = jnp.sum(
        predicted[None, :, :]
        * a[:, :, None]
        * (jnp.log(safe_a)[:, :, None] - jnp.log(safe_o)[:, None, :]),
        axis=(0, 1),
    )
    return c @ outcomes + info


def choose(a: Any, b: Any, c: Any, q: Any) -> int:
    """Select the lowest index attaining the largest counterfactual score."""
    scores = action_scores(a, b, c, q)
    # Algebraically tied scores can differ by a few arithmetic ULPs. This
    # affects selection only, never probabilities or posterior normalization.
    bound = 8 * jnp.finfo(scores.dtype).eps * jnp.maximum(1, jnp.max(jnp.abs(scores)))
    return int(jnp.argmax(scores >= jnp.max(scores) - bound))


def soft_update(a: Any, prior: Any, histogram: Any) -> Any:
    """Assimilate a unit-weight histogram once, not ten categorical samples."""
    message = histogram @ jnp.log(a)
    return posterior(prior * jnp.exp(message - jnp.max(message)))


def _sample(rng: Any, probabilities: Any) -> int:
    # Inverse CDF sampling avoids a second normalization by numpy.random.choice.
    p = np.asarray(probabilities)
    return min(
        int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), p.size - 1
    )


def block_filter(
    a: Any,
    b: Any,
    prior_map: Any,
    context_prior: Any,
    observations: list[int],
    actions: list[int],
) -> dict[str, Any]:
    """Exact block likelihood and joint posterior conditioned on executed actions."""
    if len(observations) != 5 or len(actions) != 4:
        raise ValueError("A block contains five observations and four actions")
    conditional = jnp.asarray(prior_map)
    for t, obs in enumerate(observations):
        if not 0 <= obs < a.shape[0]:
            raise ValueError("Observation out of range")
        if t:
            if not 0 <= actions[t - 1] < b.shape[2]:
                raise ValueError("Action out of range")
            conditional = b[:, :, actions[t - 1]] @ conditional
        conditional = a[obs, :, None] * conditional
    likelihood = jnp.sum(conditional, axis=0)
    joint = posterior(conditional * context_prior[None, :])
    return {
        "likelihood": np.asarray(likelihood).tolist(),
        "context_posterior": np.asarray(jnp.sum(joint, axis=0)).tolist(),
        "joint_posterior": np.asarray(joint).tolist(),
    }


def run_blocks(m: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """Simulate exact five-observation blocks with action-independent resets.

    Lower control uses the one-step utility+MI rule; higher C is diagnostic
    only and never participates in context inference or lower policy scores.
    """
    from gnn.render.execution_contracts import validate_run_parameters

    validate_run_parameters("block_reset_v1", p)
    a, b, c = (m[f"{k}_level1"] for k in "ABC")
    prior_map, context_b, context_c, context_prior = (m[f"{k}_level2"] for k in "ABCD")
    count = p.get("num_timesteps", 20)
    rng = np.random.default_rng(p.get("seed", 42))
    context = _sample(rng, context_prior)
    blocks = []
    context_posterior = context_prior
    for start in range(0, count, 5):
        if start:
            context = _sample(rng, context_b[:, context, 0])
            context_prior = context_b[:, :, 0] @ context_posterior
        lower = _sample(rng, prior_map[:, context])
        conditional = prior_map
        observations, actions, states, beliefs = [], [], [], []
        for t in range(5):
            obs = _sample(rng, a[:, lower])
            conditional = conditional * a[obs, :, None]
            joint = posterior(conditional * context_prior[None, :])
            lower_q = jnp.sum(joint, axis=1)
            observations.append(obs)
            states.append(lower)
            beliefs.append(np.asarray(lower_q).tolist())
            if t < 4:
                action = choose(a, b, c, lower_q)
                actions.append(action)
                conditional = b[:, :, action] @ conditional
                lower = _sample(rng, b[:, lower, action])
        context_posterior = jnp.sum(joint, axis=0)
        record = block_filter(a, b, prior_map, context_prior, observations, actions)
        record.update(
            {
                "start_observation": start,
                "context_state": context,
                "context_prior": np.asarray(context_prior).tolist(),
                "reset_lower_prior": np.asarray(prior_map @ context_prior).tolist(),
                "observations": observations,
                "actions": actions,
                "states": states,
                "lower_beliefs": beliefs,
                "higher_preference_diagnostic": float(
                    context_c @ (prior_map @ context_posterior)
                ),
            }
        )
        blocks.append(record)
    return {
        "inference": "exact_joint_filter_conditioned_on_actions",
        "lower_policy": "one_step_raw_utility_plus_mutual_information",
        "blocks": blocks,
        "num_observations": count,
        "num_controlled_transitions": 4 * len(blocks),
        "num_resets": len(blocks) - 1,
    }


def run_timed(m: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """Execute the approximate 10/100-clock controller in fast→medium→slow order."""
    from gnn.render.execution_contracts import validate_run_parameters

    validate_run_parameters("timed_soft_controller_v1", p)
    a = [m[f"A_level{k}"] for k in range(3)]
    b = [m[f"B_level{k}"] for k in range(3)]
    base = [m[f"C_level{k}"] for k in range(3)]
    q2 = m["D_level2"]
    q1 = m["N21"] @ q2
    q0 = m["D_level0"]
    rng = np.random.default_rng(p.get("seed", 42))
    state = _sample(rng, q0)
    obs = _sample(rng, a[0][:, state])
    q0 = posterior(q0 * a[0][obs])
    c0, c1 = base[0] + m["M10"] @ q1, base[1] + m["M21"] @ q2
    u0, u1, u2 = (
        choose(aa, bb, cc, qq)
        for aa, bb, cc, qq in zip(a, b, (c0, c1, base[2]), (q0, q1, q2))
    )
    trace = [
        {
            "transition": 0,
            "observation": obs,
            "state": state,
            "beliefs": [np.asarray(q).tolist() for q in (q0, q1, q2)],
            "next_actions": [u0, u1, u2],
            "preferences": [np.asarray(c).tolist() for c in (c0, c1, base[2])],
            "updates": ["initial_observation"],
        }
    ]
    fast_buffer, medium_buffer = [], []
    count = p.get("num_fast_transitions", 100)
    for t in range(1, count + 1):
        used = [u0, u1, u2]
        state = _sample(rng, b[0][:, state, u0])
        obs = _sample(rng, a[0][:, state])
        q0 = posterior((b[0][:, :, u0] @ q0) * a[0][obs])
        fast_buffer.append(q0)
        updates = ["fast_observation"]
        messages = {}
        if t % 10 == 0:
            histogram = jnp.mean(jnp.stack(fast_buffer), axis=0)
            q1 = soft_update(a[1], b[1][:, :, u1] @ q1, histogram)
            fast_buffer = []
            medium_buffer.append(q1)
            messages["medium"] = np.asarray(histogram).tolist()
            updates.append("medium_soft_message")
        if t % 100 == 0:
            histogram = jnp.mean(jnp.stack(medium_buffer), axis=0)
            q2 = soft_update(a[2], b[2][:, :, u2] @ q2, histogram)
            medium_buffer = []
            messages["slow"] = np.asarray(histogram).tolist()
            updates.append("slow_soft_message")
        # Recompute from bases, never accumulate preference increments.
        c0, c1 = base[0] + m["M10"] @ q1, base[1] + m["M21"] @ q2
        if t % 100 == 0:
            u2 = choose(a[2], b[2], base[2], q2)
        if t % 10 == 0:
            u1 = choose(a[1], b[1], c1, q1)
        u0 = choose(a[0], b[0], c0, q0)
        trace.append(
            {
                "transition": t,
                "observation": obs,
                "state": state,
                "beliefs": [np.asarray(q).tolist() for q in (q0, q1, q2)],
                "applied_actions": used,
                "next_actions": [u0, u1, u2],
                "preferences": [np.asarray(c).tolist() for c in (c0, c1, base[2])],
                "updates": updates,
                "messages": messages,
            }
        )
    return {
        "inference": "approximate_unit_weight_soft_message_controller",
        "num_fast_transitions": count,
        "medium_updates": count // 10,
        "slow_updates": count // 100,
        "trace": trace,
    }


def run_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a serialized contract and run its real JAX numerical path."""
    from gnn.render.execution_contracts import contract_payload

    # Validate again at execution: generated scripts are not trusted receipts.
    validated = contract_payload(
        {
            "model_parameters": payload["parameters"],
            "structured_pomdp": {"matrices": payload["matrices"]},
        }
    )
    contract = validated["execution_contract"]
    with (
        jax.enable_x64(True)
        if hasattr(jax, "enable_x64")
        else jax.experimental.enable_x64()
    ):
        m = {
            k: jnp.asarray(v, dtype=jnp.float64)
            for k, v in validated["matrices"].items()
        }
        if contract == "block_reset_v1":
            result = run_blocks(m, validated["parameters"])
        elif contract == "timed_soft_controller_v1":
            result = run_timed(m, validated["parameters"])
        else:
            from gnn.execute.jax.episodic_contingent import run_episodic

            result = run_episodic(m, validated["parameters"])
    return {
        "schema": "jax_execution_contract_v1",
        "framework": "jax",
        "execution_contract": contract,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "precision": "float64",
        "parameters": validated["parameters"],
        **result,
    }
