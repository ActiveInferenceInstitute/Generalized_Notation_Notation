"""Exact contingent two-transition episodic planning with raw reward utilities.

Enumerate complete observation trajectories conditional on fixed context. A
policy maps the observed (location, reward/cue) pair to its second action.
Terminal arrivals are scored once and have no continuation or terminal bonus.
Mutual information uses the complete trajectory once. No repeated marginal IG.
"""

from itertools import product
from typing import Any

import jax.numpy as jnp
import numpy as np


def evaluate_policies(
    m: dict[str, Any], p: dict[str, Any], location: int, context_prior: Any
) -> list[dict[str, Any]]:
    """Return every reachable contingent policy's exact utility and joint MI."""
    a, b, c = m["A_rew"], m["B_loc"], m["C_rew"]
    terminal = set(p["terminal_locations"])
    actions = b.shape[2]
    if location in terminal:
        raise ValueError(
            "Cannot plan after terminal arrival; reset the episode explicitly"
        )
    records = []
    for first_action in range(actions):
        first = [
            (loc, obs)
            for loc in range(a.shape[1])
            for obs in range(a.shape[0])
            if float(b[loc, location, first_action]) > 0
            and float(jnp.sum(a[obs, loc] * context_prior)) > 0
        ]
        branches = [(loc, obs) for loc, obs in first if loc not in terminal]
        if actions ** len(branches) > 100_000:
            raise ValueError(
                "Contingent policy enumeration exceeds the bounded contract budget"
            )
        for choices in product(range(actions), repeat=len(branches)):
            policy = dict(zip(branches, choices))
            trajectories, utilities = [], []
            for loc, obs in first:
                probability = b[loc, location, first_action] * a[obs, loc]
                if loc in terminal:
                    trajectories.append(probability)
                    utilities.append(c[obs])
                    continue
                second_action = policy[(loc, obs)]
                for loc2 in range(a.shape[1]):
                    for obs2 in range(a.shape[0]):
                        next_probability = b[loc2, loc, second_action] * a[obs2, loc2]
                        if float(jnp.sum(next_probability)) == 0:
                            continue
                        trajectories.append(probability * next_probability)
                        utilities.append(c[obs] + c[obs2])
            conditional = jnp.stack(trajectories)
            joint = conditional * context_prior[None, :]
            marginal = jnp.sum(joint, axis=1)
            utility = jnp.dot(marginal, jnp.asarray(utilities))
            log_ratio = (
                jnp.log(jnp.where(conditional > 0, conditional, 1))
                - jnp.log(jnp.where(marginal > 0, marginal, 1))[:, None]
            )
            information = jnp.sum(joint * log_ratio)
            records.append(
                {
                    "first_action": first_action,
                    "second_actions": [
                        {
                            "location": loc,
                            "observation": obs,
                            "action": policy[(loc, obs)],
                        }
                        for loc, obs in branches
                    ],
                    "expected_raw_utility": float(utility),
                    "trajectory_information": float(information),
                    "score": float(utility + information),
                }
            )
    return records


def run_episodic(m: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """Plan and sample one externally initialized, at-most-two-transition episode."""
    from gnn.execute.jax.semantic_runner import _sample, posterior
    from gnn.render.execution_contracts import validate_run_parameters

    validate_run_parameters("episodic_contingent_v1", p)
    rng = np.random.default_rng(p.get("seed", 42))
    loc = _sample(rng, m["D_loc"])
    context = _sample(rng, m["D_ctx"])
    # The common initial observation updates belief but earns no utility.
    initial_obs = _sample(rng, m["A_rew"][:, loc, context])
    q = posterior(m["D_ctx"] * m["A_rew"][initial_obs, loc])
    initial_loc = loc
    records = evaluate_policies(m, p, loc, q)
    scores = np.asarray([r["score"] for r in records])
    bound = 8 * np.finfo(scores.dtype).eps * max(1, np.max(abs(scores)))
    best = records[int(np.flatnonzero(scores >= scores.max() - bound)[0])]

    action = best["first_action"]
    trace = []
    for t in range(2):
        loc = _sample(rng, m["B_loc"][:, loc, action])
        obs = _sample(rng, m["A_rew"][:, loc, context])
        q = posterior(q * m["A_rew"][obs, loc])
        trace.append(
            {
                "transition": t + 1,
                "action": action,
                "location": loc,
                "observation": obs,
                "raw_utility": float(m["C_rew"][obs]),
                "context_posterior": np.asarray(q).tolist(),
            }
        )
        if loc in p["terminal_locations"]:
            break
        if t == 0:
            action = next(
                item["action"]
                for item in best["second_actions"]
                if item["location"] == loc and item["observation"] == obs
            )
    return {
        "inference": "exact_contingent_episodic_enumeration",
        "initial_location": initial_loc,
        "initial_observation": initial_obs,
        "context_state": context,
        "policies": records,
        "selected_policy": best,
        "trace": trace,
        "episode_raw_utility": sum(item["raw_utility"] for item in trace),
        "terminated": loc in p["terminal_locations"],
    }
