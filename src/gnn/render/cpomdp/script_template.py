"""Standalone cpomdp simulation script generator (continuous branch only).

``render.continuous_script`` hard-codes a Kalman filter plus GNN's
proportional controller. cpomdp brings its own filter (``KalmanBackend``) and
picks actions by scoring every policy over a declared finite action set, so it
gets its own template. The matrix literals come from
:func:`render.continuous_common.literal_block`, so every continuous backend
splices the same numbers.

The emitted script keeps the GNN continuous contract::

    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_{t-1} + u_{t-1} + N(0, Q)
    y_t = H x_t + N(0, R)

The control input adds straight to the state, so cpomdp's ``control_matrix``
is the identity. The prior is over ``x_1``, so the first reading is folded in
without a predict step. With a declared ``goal_mean``/``control_gain`` there
are two control modes. ``efe`` applies the first action of the policy with the
lowest expected free energy. ``parity`` applies GNN's
``u_t = control_gain * (goal_mean - mu_t)`` on cpomdp's filter, which
reproduces the ``jax`` backend on the same seed. Both score every policy each
step. Specs without a goal render as passive trackers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from gnn.render.continuous_common import ContinuousSpec, literal_block

OUTPUT_ENV = "CPOMDP_OUTPUT_DIR"
CONTROL_MODES = ("efe", "parity")
DEFAULT_ACTION_SCALE = 0.5
DEFAULT_HORIZON = 1
DEFAULT_GOAL_PRECISION = 1.0
ACTION_SET_VERSION = "gnn-compass-rose-v1"


def resolve_render_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply defaults to the cpomdp render options and validate them.

    Args:
        options: Caller options. Keys other than ``control_mode``,
            ``action_scale``, ``horizon`` and ``goal_precision`` are ignored.

    Returns:
        The four options, typed and defaulted.

    Raises:
        ValueError: If an option is outside its allowed range.
    """
    resolved = {k: v for k, v in dict(options or {}).items() if v is not None}
    control_mode = str(resolved.get("control_mode", "efe")).strip().lower()
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"cpomdp control_mode must be one of {CONTROL_MODES}, got {control_mode!r}"
        )
    action_scale = float(resolved.get("action_scale", DEFAULT_ACTION_SCALE))
    if not action_scale > 0.0:
        raise ValueError(f"cpomdp action_scale must be > 0, got {action_scale}")
    horizon = int(resolved.get("horizon", DEFAULT_HORIZON))
    if horizon < 1:
        raise ValueError(f"cpomdp horizon must be >= 1, got {horizon}")
    goal_precision = float(resolved.get("goal_precision", DEFAULT_GOAL_PRECISION))
    if not goal_precision > 0.0:
        raise ValueError(f"cpomdp goal_precision must be > 0, got {goal_precision}")
    return {
        "control_mode": control_mode,
        "action_scale": action_scale,
        "horizon": horizon,
        "goal_precision": goal_precision,
    }


_IMPORTS = '''
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("ERROR: JAX not installed. Install with: uv sync")
    sys.exit(1)

try:
    import cpomdp
    from cpomdp import Belief, KalmanBackend, LinearGaussianModel, Preference
    from cpomdp.efe import policy_efe
    from cpomdp.enumeration import EnumeratedEfeSearch, FiniteActionSet
except ImportError:
    print("ERROR: cpomdp not installed. Install with: uv sync --extra cpomdp")
    sys.exit(1)

FRAMEWORK = "cpomdp"
FRAMEWORK_VERSION = {
    "jax_version": jax.__version__,
    "cpomdp_version": cpomdp.__version__,
}
_KEY = [jax.random.PRNGKey(RANDOM_SEED)]


def arr(x):
    """Cast to a float64 JAX array."""
    return jnp.asarray(x, dtype=jnp.float64)


def mvn_sample(mean, cov):
    """Draw from N(mean, cov), in the same key order as the other backends."""
    _KEY[0], sub = jax.random.split(_KEY[0])
    return jax.random.multivariate_normal(sub, mean, cov)


def to_list(x):
    """Convert an array to nested Python floats for JSON."""
    return np.asarray(x, dtype=float).tolist()


def is_psd(cov):
    """Check that a covariance is positive semi-definite."""
    return bool(jnp.all(jnp.linalg.eigvalsh((cov + cov.T) / 2.0) > -1e-9))
'''

_MODEL = '''

def compass_rose(control_dim, scale):
    """Build the declared, versioned action set the policy search enumerates.

    A 2-D control gets the null action plus eight unit directions. Any other
    dimension gets the null action plus +/- scale along each axis.
    """
    if control_dim == 2:
        diag = scale / np.sqrt(2.0)
        actions = [
            [0.0, 0.0],
            [scale, 0.0],
            [diag, diag],
            [0.0, scale],
            [-diag, diag],
            [-scale, 0.0],
            [-diag, -diag],
            [0.0, -scale],
            [diag, -diag],
        ]
    else:
        actions = [[0.0] * control_dim]
        for axis in range(control_dim):
            for sign in (1.0, -1.0):
                action = [0.0] * control_dim
                action[axis] = sign * scale
                actions.append(action)
    return FiniteActionSet(actions, version=ACTION_SET_VERSION)


def build_model(controlled):
    """Build the cpomdp model from the GNN parameters."""
    dynamics_matrix = arr(F_RAW)  # F
    n_states = dynamics_matrix.shape[0]
    return LinearGaussianModel(
        dynamics_matrix,
        observation_matrix=arr(H_RAW),  # H
        dynamics_noise=arr(Q_RAW),  # Q
        observation_noise=arr(R_RAW),  # R
        prior=Belief(arr(PRIOR_MEAN_RAW), arr(PRIOR_COV_RAW)),
        control_matrix=jnp.eye(n_states) if controlled else None,
    )


def first_step_backend(model):
    """Filter for step 1, whose prior is already over x_1: an identity predict."""
    n_states = model.n_states
    return KalmanBackend(
        LinearGaussianModel(
            jnp.eye(n_states),
            observation_matrix=model.observation_matrix,
            dynamics_noise=jnp.zeros((n_states, n_states)),
            observation_noise=model.observation_noise,
            prior=model.prior,
        )
    )


def sample_observation(model, state):
    """Draw a reading of ``state`` from the fixed sensor."""
    noise = mvn_sample(jnp.zeros(model.n_observations), model.observation_noise)
    return model.observation_matrix @ state + noise


def build_planner(model, goal):
    """Return the policy search and a scorer covering every policy in one pass.

    The scorer returns each policy's G and its pragmatic and epistemic parts, so
    the chosen policy is never rolled out a second time.
    """
    action_set = compass_rose(model.n_controls, ACTION_SCALE)
    search = EnumeratedEfeSearch(model, action_set, horizon=HORIZON)
    preference = Preference(
        model.observation_matrix @ goal,
        GOAL_PRECISION * jnp.eye(model.n_observations),
    )
    policies = search.policies

    @jax.jit
    def score(belief):
        return jax.vmap(lambda pol: policy_efe(model, belief, pol, preference))(
            policies
        )

    return search, score
'''

_RUN = '''

def run_simulation():
    """Simulate the GNN process, filter it with cpomdp and write the results."""
    start = time.time()
    passive = GOAL_MEAN_RAW is None or CONTROL_GAIN is None
    model = build_model(controlled=not passive)
    backend = KalmanBackend(model)
    n_states, n_steps = model.n_states, NUM_TIMESTEPS
    no_action = jnp.zeros(n_states)

    true_states, observations, beliefs, covs, controls = [], [], [], [], []
    efe_history, epistemic, pragmatic, selected = [], [], [], []

    if GOAL_MEAN_RAW is None or CONTROL_GAIN is None:
        control_mode = "passive"
        search_stats = {"n_policies": 0, "cost_per_cycle": 0, "search_warrant": None}

        def choose(belief):
            return no_action

    else:
        control_mode = CONTROL_MODE
        goal = arr(GOAL_MEAN_RAW)
        gain = CONTROL_GAIN
        search, score = build_planner(model, goal)
        policies = search.policies
        search_stats = {
            "n_policies": search.n_policies,
            "cost_per_cycle": search.cost_per_cycle,
            "search_warrant": str(search.certificate),
        }

        def choose(belief):
            g, parts = score(belief)
            best = int(jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g)))
            efe_history.append(to_list(g))
            selected.append(best)
            epistemic.append(float(parts["epistemic"][best]))
            pragmatic.append(float(parts["pragmatic"][best]))
            if control_mode == "parity":
                return gain * (goal - belief.mean)
            return policies[best][0]

    def record(state, reading, belief, action):
        for buf, val in (
            (true_states, state),
            (observations, reading),
            (beliefs, belief.mean),
            (covs, belief.cov),
            (controls, action),
        ):
            buf.append(to_list(val))

    state = mvn_sample(model.prior.mean, model.prior.cov)
    reading = sample_observation(model, state)
    belief = first_step_backend(model).infer_states(reading, model.prior)
    action = choose(belief)
    record(state, reading, belief, action)

    for _step in range(1, n_steps):
        process_noise = mvn_sample(no_action, model.dynamics_noise)
        state = model.dynamics_matrix @ state + action + process_noise
        reading = sample_observation(model, state)
        belief = backend.infer_states(reading, belief, None if passive else action)
        action = choose(belief)
        record(state, reading, belief, action)

    beliefs_np = np.asarray(beliefs)
    rmse = float(np.sqrt(np.mean((beliefs_np - np.asarray(true_states)) ** 2)))
    validation = {
        "means_finite": bool(np.all(np.isfinite(beliefs_np))),
        "posterior_cov_psd": all(is_psd(arr(c)) for c in covs),
        "rmse_finite": bool(np.isfinite(rmse)),
        "controls_finite": bool(np.all(np.isfinite(np.asarray(controls)))),
        "efe_finite": bool(np.all(np.isfinite(np.asarray(efe_history, dtype=float)))),
    }
    validation["all_valid"] = all(validation.values())
    results = {
        "model_name": MODEL_NAME,
        "framework": FRAMEWORK,
        "model_kind": "continuous",
        "num_timesteps": n_steps,
        "num_states": n_states,
        "num_observations": model.n_observations,
        "beliefs": beliefs,
        "posterior_cov": covs,
        "true_states_continuous": true_states,
        "observations_continuous": observations,
        "controls": controls,
        "control_mode": control_mode,
        "rmse_vs_true": rmse,
        # Discrete-schema slots stay empty: nothing categorical is defined here.
        "observations": [],
        "actions": [],
        "validation": validation,
        "execution_time_seconds": round(time.time() - start, 4),
        "efe_history": efe_history,
        "epistemic_term": epistemic,
        "pragmatic_term": pragmatic,
        "selected_policy_index": selected,
        "horizon": HORIZON,
        "action_scale": ACTION_SCALE,
        **search_stats,
        **FRAMEWORK_VERSION,
    }

    output_dir = Path(os.environ.get(OUTPUT_ENV, "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"{FRAMEWORK} continuous active inference simulation complete: "
        f"{n_steps} steps, control_mode={control_mode}, rmse_vs_true={rmse:.4f}"
    )
    print(f"Results saved to: {out}")
    print(f"Validation: {validation}")
    return results


if __name__ == "__main__":
    res = run_simulation()
    sys.exit(0 if res["validation"]["all_valid"] else 1)
'''


def generate_cpomdp_script(spec: ContinuousSpec, opts: Mapping[str, Any]) -> str:
    """Return the full standalone cpomdp script for ``spec``.

    Args:
        spec: The parsed continuous model.
        opts: The output of :func:`resolve_render_options`.

    Returns:
        The script source, already in ruff's format.
    """
    lits = literal_block(spec)
    if not spec.has_control:
        control_desc = "passive tracker (no goal_mean/control_gain)"
    elif opts["control_mode"] == "parity":
        control_desc = "parity: GNN's proportional controller on cpomdp's filter"
    else:
        control_desc = f"efe: lowest-G policy, horizon {opts['horizon']}"
    header = f'''#!/usr/bin/env python3
"""cpomdp continuous active inference simulation of {spec.model_name}.

Generated by the GNN pipeline's cpomdp renderer. GNN continuous contract:
    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_(t-1) + u_(t-1) + N(0, Q)      ({spec.n}-dim latent state)
    y_t = H x_t + N(0, R)                    ({spec.m}-dim observation)
Control: {control_desc}.
Filter: cpomdp KalmanBackend. Policies: every sequence of a declared action set.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {json.dumps(spec.model_name)}
RANDOM_SEED = {spec.random_seed}
NUM_TIMESTEPS = {spec.num_timesteps}
OUTPUT_ENV = {json.dumps(OUTPUT_ENV)}
CONTROL_MODE = {json.dumps(opts["control_mode"])}
ACTION_SCALE = {opts["action_scale"]!r}
HORIZON = {opts["horizon"]}
GOAL_PRECISION = {opts["goal_precision"]!r}
ACTION_SET_VERSION = {json.dumps(ACTION_SET_VERSION)}

F_RAW = {lits["F"]}
H_RAW = {lits["H"]}
Q_RAW = {lits["Q"]}
R_RAW = {lits["R"]}
PRIOR_MEAN_RAW = {lits["prior_mean"]}
PRIOR_COV_RAW = {lits["prior_cov"]}
GOAL_MEAN_RAW = {lits["goal_mean"]}
CONTROL_GAIN = {lits["control_gain"]}
'''
    return "".join([header, _IMPORTS, _MODEL, _RUN])
