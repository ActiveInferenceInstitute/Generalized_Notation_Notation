#!/usr/bin/env python3
"""
NumPyro Renderer for GNN Specifications

Renders GNN POMDP models to standalone NumPyro simulation scripts.
Uses numpyro.distributions.Categorical for probabilistic sampling
within the standard Active Inference generative loop.

@Web: https://num.pyro.ai/
@Web: https://github.com/pyro-ppl/numpyro
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gnn.render.naming import atomic_write_text
from gnn.render.spec_matrices import extract_abcd_matrices, format_array_literal

logger = logging.getLogger(__name__)


def render_gnn_to_numpyro(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a NumPyro POMDP simulation script.

    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated NumPyro code.
        options: Optional rendering options.

    Returns:
        Tuple of (success: bool, message: str, artifact_paths: List[str])
    """
    try:
        model_name = gnn_spec.get("modelName", "numpyro_pomdp")
        logger.info(f"Rendering GNN spec to NumPyro: {model_name}")

        from gnn.render.continuous_common import (
            extract_continuous_spec,
            is_continuous_spec,
        )

        if is_continuous_spec(gnn_spec):
            # Continuous-state (linear-Gaussian) branch: Kalman filter + NUTS
            # on the same generative model. No A/B/C/D exist on this path.
            from gnn.render.continuous_script import generate_continuous_script
            from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

            if detect_model_kinds(gnn_spec) == frozenset(
                {ModelKind.FACTORED, ModelKind.CONTINUOUS}
            ):
                return (
                    False,
                    "unsupported-factored-continuous: numpyro renders the flat "
                    "linear-Gaussian family only; per-factor compositions are "
                    "refused rather than silently rendered flat",
                    [],
                )

            code = generate_continuous_script(
                extract_continuous_spec(gnn_spec), "numpyro"
            )
            output_path = Path(output_path)
            atomic_write_text(output_path, code)
            logger.info(f"✅ NumPyro continuous script written to: {output_path}")
            return (
                True,
                f"NumPyro continuous LGSSM script generated: {output_path}",
                [str(output_path)],
            )

        provenance = deepcopy(gnn_spec.get("matrix_provenance", {}))
        A, B, C, D = _extract_matrices(gnn_spec, provenance=provenance)

        # Validate shapes
        from gnn.render.matrix_utils import validate_abcd_shapes

        valid, msg = validate_abcd_shapes(A, B, C, D)
        if not valid:
            logger.warning(f"Shape validation warning: {msg}")

        # Ensure options has num_timesteps if available in gnn_spec
        options = options or {}
        model_params = gnn_spec.get("model_parameters", {})
        init_params = gnn_spec.get("initialparameterization", {})
        if "num_timesteps" not in options:
            extracted_t = model_params.get(
                "num_timesteps", init_params.get("num_timesteps")
            )
            if extracted_t is not None:
                options["num_timesteps"] = extracted_t

        code = _generate_numpyro_code(
            model_name,
            A,
            B,
            C,
            D,
            options,
            matrix_provenance=provenance,
        )

        output_path = Path(output_path)
        atomic_write_text(output_path, code)

        logger.info(f"✅ NumPyro script written to: {output_path}")
        return True, f"NumPyro script generated: {output_path}", [str(output_path)]

    except Exception as e:
        logger.error(f"❌ NumPyro rendering failed: {e}")
        return False, f"NumPyro rendering failed: {e}", []


def _extract_matrices(
    gnn_spec: Dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from GNN spec.

    Delegates to the shared :func:`render.spec_matrices.extract_abcd_matrices`.
    """
    return extract_abcd_matrices(gnn_spec, transformation_provenance=provenance)


def _format_jnp_array(arr: np.ndarray, indent: int = 4) -> str:
    """Format numpy array as jnp.array() literal."""
    return format_array_literal(arr, prefix="jnp.array", indent=indent)


def _generate_numpyro_code(
    model_name: str,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    options: Optional[Dict[str, Any]] = None,
    matrix_provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate standalone NumPyro POMDP simulation script."""
    provenance = matrix_provenance or {}
    num_timesteps = (options or {}).get("num_timesteps", 15)
    num_states = A.shape[1] if A.ndim == 2 else 2
    num_obs = A.shape[0] if A.ndim == 2 else num_states
    num_actions = B.shape[2] if B.ndim == 3 else 2

    A_str = _format_jnp_array(A)
    B_str = _format_jnp_array(B if B.ndim == 2 else B[:, :, 0])
    C_str = _format_jnp_array(C)
    D_str = _format_jnp_array(D)

    B_full_init = ""
    if B.ndim == 3:
        slices: list[Any] = []
        for a in range(B.shape[2]):
            slices.append(
                f"    B_slices.append({_format_jnp_array(B[:, :, a], indent=4)})"
            )
        B_full_init = (
            "\n    B_slices = []\n"
            + "\n".join(slices)
            + "\n    B = jnp.stack(B_slices, axis=2)"
        )
    else:
        B_full_init = f"\n    B = jnp.tile({B_str}[:, :, None], (1, 1, {num_actions}))"

    code = f'''\
#!/usr/bin/env python3
"""
NumPyro POMDP Simulation: {model_name}

Auto-generated by GNN Pipeline — NumPyro renderer.
Uses numpyro.distributions.Categorical for probabilistic sampling
within the standard Active Inference generative loop.
"""
import json
import os
import sys
import time
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
except ImportError:
    print("ERROR: JAX not installed. Install with: uv sync")
    sys.exit(1)

try:
    import numpyro
    import numpyro.distributions as dist
except ImportError:
    print("ERROR: NumPyro not installed. Install with: uv sync")
    sys.exit(1)

import numpy as np


def run_simulation(seed: int = 42):
    """Run POMDP simulation with NumPyro distributions."""
    start_time = time.time()
    key = jrandom.PRNGKey(seed)

    # --- Model Parameters ---
    num_states = {num_states}
    num_obs = {num_obs}
    num_actions = {num_actions}
    T = {num_timesteps}

    A = {A_str}
    C = {C_str}
    D = {D_str}
    {B_full_init}

    # --- Simulation State ---
    beliefs_history = []
    actions_history = []
    observations_history = []
    efe_history = []

    # Initialize true state from prior using NumPyro
    key, subkey = jrandom.split(key)
    true_state = int(dist.Categorical(probs=D).sample(subkey))
    beliefs = D.copy()

    for t in range(T):
        # 1. Generate observation
        key, subkey = jrandom.split(key)
        obs_probs = A[:, true_state]
        obs_probs = obs_probs / obs_probs.sum()
        observation = int(dist.Categorical(probs=obs_probs).sample(subkey))

        # 2. Bayesian belief update
        likelihood = A[observation, :]
        posterior = likelihood * beliefs
        posterior = posterior / (jnp.sum(posterior) + 1e-16)
        beliefs = posterior

        # 3. Expected Free Energy for each action
        efe = jnp.zeros(num_actions)
        for a_idx in range(num_actions):
            B_a = B[:, :, a_idx]
            predicted_state = B_a @ beliefs
            predicted_obs = A @ predicted_state

            # Ambiguity
            log_A = jnp.log(A + 1e-16)
            ambiguity = -jnp.sum(predicted_state * jnp.sum(A * log_A, axis=0))

            # Risk (KL from preferred)
            C_norm = jax.nn.softmax(C)
            risk = jnp.sum(predicted_obs * (jnp.log(predicted_obs + 1e-16) - jnp.log(C_norm + 1e-16)))

            efe = efe.at[a_idx].set(ambiguity + risk)

        # 4. Action selection via NumPyro Categorical
        key, subkey = jrandom.split(key)
        action_probs = jax.nn.softmax(-efe)
        action = int(dist.Categorical(probs=action_probs).sample(subkey))

        # 5. State transition
        key, subkey = jrandom.split(key)
        B_a = B[:, :, action]
        transition_probs = B_a[:, true_state]
        transition_probs = transition_probs / (jnp.sum(transition_probs) + 1e-16)
        true_state = int(dist.Categorical(probs=transition_probs).sample(subkey))

        beliefs_history.append(np.array(beliefs).tolist())
        actions_history.append(action)
        observations_history.append(observation)
        efe_history.append(np.array(efe).tolist())

    elapsed = time.time() - start_time

    # --- Validation ---
    beliefs_arr = np.array(beliefs_history)
    validation = {{
        "beliefs_in_range": bool(np.all((beliefs_arr >= 0) & (beliefs_arr <= 1))),
        "beliefs_sum_to_one": bool(np.allclose(beliefs_arr.sum(axis=1), 1.0, atol=1e-6)),
        "actions_in_range": all(0 <= a < num_actions for a in actions_history),
        "all_valid": True,
    }}
    validation["all_valid"] = all(validation.values())

    results = {{
        "schema_version": "numpyro_simulation_v1",
        "model_name": "{model_name}",
        "framework": "numpyro",
        "matrix_provenance": {provenance!r},
        "num_timesteps": T,
        "num_states": num_states,
        "num_observations": num_obs,
        "num_actions": num_actions,
        "beliefs": beliefs_history,
        "actions": actions_history,
        "observations": observations_history,
        "efe_history": efe_history,
        "validation": validation,
        "execution_time_seconds": round(elapsed, 4),
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
    }}

    output_dir = Path(os.environ.get("NUMPYRO_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"NumPyro POMDP simulation complete: {{T}} timesteps in {{elapsed:.3f}}s")
    print(f"Results saved to: {{output_file}}")
    print(f"Validation: {{validation}}")
    return results


if __name__ == "__main__":
    run_simulation()
'''
    return code
