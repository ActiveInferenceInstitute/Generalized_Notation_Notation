#!/usr/bin/env python3
"""
General JAX model script generation for GNN Step 11.

Extracted from ``render.jax.jax_renderer``.
"""

from typing import (
    Any,
    Dict,
    Optional,
)

from .jax_spec_extract import (
    _jax_model_name,
    _validated_jax_matrices,
)


def _json_dumps(value: str) -> str:
    """Return *value* as a JSON string literal (module-level import site)."""
    import json

    return json.dumps(value)


EFE_CONVENTION_JAX = (
    "jax renderer compute_expected_free_energy heuristic:"
    " EFE = obs_entropy - pragmatic_value + 0.1 * kl_divergence, where"
    " obs_entropy is the entropy of predicted observations,"
    " pragmatic_value = sum_o q(o) C[o] (a LINEAR payoff sum, not a KL"
    " against C), and kl_divergence = KL(next_belief || D) weighted by a"
    " fixed 0.1 coefficient. This is NOT the risk+ambiguity EFE"
    " decomposition (risk is the KL against C) and is not comparable to"
    " the pymdp neg_efe or the Lean"
    " expectedFreeEnergy_eq_risk_add_ambiguity without convention"
    " mapping (bridge finding O1)."
)
EFE_CONVENTION_JAX_JSON = EFE_CONVENTION_JAX.replace("\n", " ")
EFE_CONVENTION_JAX_JSON_LITERAL = _json_dumps(EFE_CONVENTION_JAX_JSON)


def _generate_jax_model_code(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]
) -> str:
    """Generate general JAX model code from GNN specification.

    This generates standalone JAX code that only requires jax and jax.numpy,
    without external dependencies like Flax or Optax.
    """

    try:
        model_name = _jax_model_name(gnn_spec, "GNNModel")
        matrices = _validated_jax_matrices(gnn_spec)

        # Pre-compute dimensions to avoid f-string issues
        A_matrix = matrices["A"]
        B_matrix = matrices["B"]
        C_vector = matrices["C"]
        D_vector = matrices["D"]

        num_states = A_matrix.shape[1]
        num_observations = A_matrix.shape[0]
        num_actions = B_matrix.shape[2]

        # Extract num_timesteps from model_parameters.
        model_params = gnn_spec.get("model_parameters", {})
        init_params = gnn_spec.get("initialparameterization", {})
        num_timesteps = model_params.get(
            "num_timesteps", init_params.get("num_timesteps", 20)
        )

        time_spec = gnn_spec.get("time_specification") or gnn_spec.get("time", "")
        time_type = (
            time_spec.get("time_type", "") if isinstance(time_spec, dict) else time_spec
        )
        # Match declarations, never a model name or incidental prose/comment.
        time_declarations = {
            line.split("#", 1)[0].strip().lower()
            for line in str(time_type).splitlines()
        }
        if {"static", "dynamic"} <= time_declarations:
            raise ValueError("Time cannot declare both Static and Dynamic")
        static_model = "static" in time_declarations
        passive_model = bool(model_params.get("passive_model", False))
        if passive_model and not static_model and num_actions != 1:
            raise ValueError(
                "Passive filtering requires one action-independent transition slice"
            )
        inference_only = static_model or passive_model
        estimand = "static_conditioning" if static_model else "filtering"
        if static_model:
            # The source's optional compatibility B does not define a time series.
            # Explicit multi-step calls fail below rather than silently rolling out.
            num_timesteps = 1

        # Convert matrices to lists for f-string insertion
        A_list = A_matrix.tolist()
        B_list = B_matrix.tolist()
        C_list = C_vector.tolist()
        D_list = D_vector.tolist()

        # nosec
        code = f'''"""
JAX Model Generated from GNN Specification: {model_name}

This model implements the GNN specification using pure JAX for high-performance computation.
Generated automatically by the GNN Processing Pipeline.

This is a standalone JAX implementation that only requires jax and numpy.
No external dependencies like Flax or Optax are required.

@Web: https://github.com/google/jax
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Model configuration
NUM_STATES = {num_states}
NUM_OBSERVATIONS = {num_observations}
NUM_ACTIONS = {num_actions}
STATIC_MODEL = {static_model}
PASSIVE_MODEL = {passive_model}
INFERENCE_ONLY = {inference_only}
INFERENCE_ESTIMAND = {estimand!r}


def _check_probability_mass(values, mass):
    """Host-side failure used by JAX callbacks, including under jit/vmap."""
    if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probability normalization requires finite nonnegative values and positive finite mass")


def normalize_probability(values):
    """Divide by actual positive mass; impossible evidence fails explicitly."""
    mass = jnp.sum(values)
    jax.debug.callback(_check_probability_mass, values, mass)
    return values / mass


def create_params() -> Dict[str, jnp.ndarray]:
    """Create model parameters from GNN specification."""
    return {{
        'A_matrix': jnp.array({A_list}),  # Observation model P(o|s)
        'B_matrix': jnp.array({B_list}),  # Transition model P(s'|s,a)
        'C_vector': jnp.array({C_list}),  # Preferences over observations
        'D_vector': jnp.array({D_list}),  # Prior over initial states
    }}


@jit
def belief_update(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> jnp.ndarray:
    """
    Bayesian belief update given an observation.
    
    P(s|o) ∝ P(o|s) * P(s) using the A matrix (likelihood)
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Updated belief state [num_states]
    """
    A_matrix = params['A_matrix']
    
    # Compute likelihood P(o|s) for each state
    # For each state s, compute sum over observations: A[o,s] * observation[o]
    likelihood = jnp.dot(observation, A_matrix)  # [num_states]
    
    # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
    updated_belief = belief * likelihood
    
    updated_belief = normalize_probability(updated_belief)
    
    return updated_belief


@jit
def compute_expected_free_energy(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                                  action: int) -> float:
    """
    Compute expected free energy (EFE) for a given action.
    
    EFE = -E[log P(o)] + KL[Q(s')||P(s')]
        = Entropy of predicted observations + KL divergence
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Expected free energy value
    """
    A_matrix = params['A_matrix']
    B_matrix = params['B_matrix']
    C_vector = params['C_vector']
    D_vector = params['D_vector']
    
    # Predict next state distribution using B matrix (transitions)
    # B_matrix[:, :, action] is [num_states, num_states]
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    next_belief = normalize_probability(next_belief)  # Normalize
    
    # Predict observation distribution using A matrix
    predicted_obs = jnp.dot(A_matrix, next_belief)
    predicted_obs = normalize_probability(predicted_obs)  # Normalize
    
    # Compute epistemic value (expected information gain)
    # This is the negative entropy of predicted observations
    obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
    
    # Compute pragmatic value (expected utility based on preferences)
    # Higher values in C_vector mean more preferred observations
    pragmatic_value = jnp.dot(predicted_obs, C_vector)
    
    # Compute KL divergence between predicted state and prior
    kl_divergence = jnp.sum(
        jnp.where(next_belief > 1e-8,
                  next_belief * jnp.log((next_belief + 1e-8) / (D_vector + 1e-8)),
                  0.0)
    )
    
    # Expected free energy = uncertainty - utility + complexity
    efe = obs_entropy - pragmatic_value + 0.1 * kl_divergence
    
    return efe


@jit
def choose_action(params: Dict[str, jnp.ndarray], belief: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """
    Choose action with minimum expected free energy.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        
    Returns:
        Tuple of (chosen_action_index, efe_values_for_all_actions)
    """
    # Compute EFE for all actions
    efe_values = jnp.array([
        compute_expected_free_energy(params, belief, a) 
        for a in range(NUM_ACTIONS)
    ])
    
    # Choose action with minimum EFE
    chosen_action = jnp.argmin(efe_values)
    
    return chosen_action, efe_values

# Batched operations for parallel execution across multiple agents or parallel rollouts
# JAX's vmap allows automatic vectorization of functions
batched_belief_update = jax.vmap(belief_update, in_axes=(None, 0, 0))
batched_compute_expected_free_energy = jax.vmap(compute_expected_free_energy, in_axes=(None, 0, None))

# Optional pmap for multiple devices if available
import os
try:
    if len(jax.devices()) > 1 and os.environ.get("JAX_ENABLE_PMAP", "1") == "1":
        # Multi-device data parallelism
        parallel_belief_update = jax.pmap(belief_update, in_axes=(None, 0, 0))
except (RuntimeError, ValueError):
    logger.debug("pmap unavailable (single-device setup or runtime error)")


@jit
def state_transition(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                     action: int) -> jnp.ndarray:
    """
    Predict next state distribution given current belief and action.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Predicted next state distribution [num_states]
    """
    B_matrix = params['B_matrix']
    
    # Apply transition model
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    
    # Normalize
    next_belief = normalize_probability(next_belief)
    
    return next_belief


def simulate_step(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> Dict[str, Any]:
    """
    Condition on one observation. Static models stop at that posterior;
    passive models predict with the sole transition slice, without planning.
    Active models additionally select an action using the labeled heuristic.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Dictionary with simulation results
    """
    # 1. Update belief based on observation
    updated_belief = belief_update(params, belief, observation)
    
    if INFERENCE_ONLY:
        return {{
            'belief': updated_belief,
            'action': None,
            'expected_free_energy': None,
            'all_efe_values': jnp.array([]),
            'predicted_next_state': (updated_belief if STATIC_MODEL else
                                     state_transition(params, updated_belief, 0)),
        }}

    # 2. Choose action
    chosen_action, efe_values = choose_action(params, updated_belief)
    
    # 3. Predict next state
    predicted_next_state = state_transition(params, updated_belief, chosen_action)
    
    return {{
        'belief': updated_belief,
        'action': chosen_action,
        'expected_free_energy': efe_values[chosen_action],
        'all_efe_values': efe_values,
        'predicted_next_state': predicted_next_state,
    }}


def run_simulation(params: Dict[str, jnp.ndarray], num_steps: int, 
                   initial_belief: jnp.ndarray = None, seed: int = 42) -> Dict[str, Any]:
    """
    Sample observations and perform filtering (or one static conditioning).
    Passive models have no selected actions or control objective.
    
    Args:
        params: Model parameters dictionary
        num_steps: Number of simulation timesteps
        initial_belief: Optional initial belief (default: source D)
        seed: Random seed for stochastic environment generation
        
    Returns:
        Dictionary with simulation trajectory
    """
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if STATIC_MODEL and num_steps != 1:
        raise ValueError("Time Static supports one observation; compatibility rollout is not static conditioning")
    key = jax.random.PRNGKey(seed)
    
    # Initialize belief
    if initial_belief is None:
        belief = params['D_vector'].copy()
    else:
        belief = initial_belief
        
    # Initialize true state from Prior
    key, subkey = jax.random.split(key)
    true_state_idx = jax.random.categorical(subkey, jnp.log(params['D_vector']))
    
    # Storage for trajectory
    beliefs = []
    actions = []
    efes = []
    observations_log = []
    
    # Run simulation
    for t in range(num_steps):
        # 1. Environment generates observation
        key, subkey = jax.random.split(key)
        obs_probs = params['A_matrix'][:, true_state_idx]
        obs_idx = jax.random.categorical(subkey, jnp.log(obs_probs))
        
        # Create one-hot observation for the agent
        obs_one_hot = jnp.zeros(params['A_matrix'].shape[0])
        obs_one_hot = obs_one_hot.at[obs_idx].set(1.0)
        observations_log.append(obs_idx)
        
        # 2. Agent perceives and acts
        result = simulate_step(params, belief, obs_one_hot)
        
        beliefs.append(result['belief'])
        if not INFERENCE_ONLY:
            actions.append(result['action'])
            efes.append(result['all_efe_values'])
        if STATIC_MODEL:
            belief = result['belief']
            continue
        action_idx = 0 if PASSIVE_MODEL else int(result['action'])

        # 3. Environment transitions true state
        key, subkey = jax.random.split(key)
        next_state_probs = params['B_matrix'][:, true_state_idx, action_idx]
        true_state_idx = jax.random.categorical(subkey, jnp.log(next_state_probs))
        
        # Update belief for next step (using predicted next state)
        belief = result['predicted_next_state']
    
    return {{
        'beliefs': jnp.stack(beliefs),
        'actions': jnp.array(actions),
        'expected_free_energies': jnp.array(efes),
        'observations': jnp.array(observations_log),
        'final_belief': beliefs[-1],
        'next_state_prediction': belief,
    }}


def get_model_summary() -> str:
    """Get a summary of the model architecture."""
    return f"""
{model_name} Model Summary (Pure JAX Implementation):
- Number of states: {{NUM_STATES}}
- Number of observations: {{NUM_OBSERVATIONS}}
- Declared transition slices: {{NUM_ACTIONS}}
- Inference estimand: {{INFERENCE_ESTIMAND}}
- Control mode: {{"none" if INFERENCE_ONLY else "active"}}
- Parameters: A matrix, B matrix, C vector, D vector

Key Functions:
- create_params(): Create model parameters
- belief_update(): Bayesian belief update
- compute_expected_free_energy(): Compute EFE for action
- choose_action(): Choose action with minimum EFE
- simulate_step(): One step of Active Inference
- run_simulation(): Full simulation over observations
"""


def save_simulation_results(trajectory: Dict[str, Any], params: Dict[str, jnp.ndarray],
                            model_name: str, output_dir: str = ".") -> str:
    """
    Save simulation results to structured JSON for downstream analysis.

    Args:
        trajectory: Simulation trajectory dictionary
        params: Model parameters
        model_name: Name of the model
        output_dir: Output directory for JSON file

    Returns:
        Path to the saved JSON file
    """
    import json
    import os
    from datetime import datetime

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Validate actual posterior arrays, including final_belief, before reporting.
    beliefs = np.asarray(trajectory['beliefs'])
    final_belief = np.asarray(trajectory['final_belief'])
    has_beliefs = beliefs.ndim == 2 and beliefs.shape[0] > 0 and beliefs.shape[1] == NUM_STATES
    shape_valid = has_beliefs and final_belief.shape == (NUM_STATES,)
    checked = np.concatenate([beliefs, final_belief[None, :]], axis=0) if shape_valid else np.array([])
    finite_nonnegative = bool(shape_valid and np.all(np.isfinite(checked)) and np.all(checked >= 0))
    # Retain the existing runtime mass tolerance; do not fabricate a success flag.
    mass_valid = bool(shape_valid and np.all(np.abs(checked.sum(axis=1) - 1.0) < 0.01))
    actions = np.asarray(trajectory['actions'])
    actions_valid = bool((actions.size == 0) if INFERENCE_ONLY else
                         (actions.shape == (len(beliefs),) and np.all(np.isfinite(actions)) and
                          np.all(actions == np.floor(actions)) and
                          np.all((actions >= 0) & (actions < NUM_ACTIONS))))
    all_valid = finite_nonnegative and mass_valid and actions_valid
    results = {{
        "success": all_valid,
        "inference_estimand": INFERENCE_ESTIMAND,
        "inference_description": ("P(s | o), one observation, no transition or planning" if STATIC_MODEL else
                                  "P(s_t | o_0:t, executed controls); no backward smoothing"),
        "control_mode": "none" if INFERENCE_ONLY else "active",
        "source_passive_model": PASSIVE_MODEL,
        "framework": "jax",
        "model_name": model_name,
        "num_timesteps": int(len(trajectory['beliefs'])),
        "timestamp": datetime.now().isoformat(),
        "simulation_trace": {{
            "observations": trajectory.get('observations', jnp.array([])).tolist(),
            "beliefs": [b.tolist() for b in trajectory.get('beliefs', [])],
            "actions": trajectory.get('actions', jnp.array([])).tolist(),
            "efe_history": trajectory.get('expected_free_energies', jnp.array([])).tolist(),
            "belief_confidence": [float(max(b)) if len(b) > 0 else 0.0 for b in trajectory.get('beliefs', [])],
        }},
        "observations": trajectory.get('observations', jnp.array([])).tolist(),
        "beliefs": [b.tolist() for b in trajectory['beliefs']],
        "actions": trajectory['actions'].tolist(),
        "final_belief": trajectory['final_belief'].tolist(),
        "model_parameters": {{
            "A_shape": list(params['A_matrix'].shape),
            "B_shape": list(params['B_matrix'].shape),
            "C_shape": list(params['C_vector'].shape),
            "D_shape": list(params['D_vector'].shape),
            "num_states": NUM_STATES,
            "num_observations": NUM_OBSERVATIONS,
            "num_actions": 0 if INFERENCE_ONLY else NUM_ACTIONS,
            "transition_slices": NUM_ACTIONS
        }},
        "metrics": {{
            "expected_free_energy": trajectory['expected_free_energies'].tolist(),
            "expected_free_energy_convention": None if INFERENCE_ONLY else {EFE_CONVENTION_JAX_JSON_LITERAL},
            "average_efe": None if INFERENCE_ONLY else float(jnp.mean(trajectory['expected_free_energies'])),
            "belief_confidence": [float(max(b)) for b in trajectory['beliefs']],
        }},
        "validation": {{
            "all_beliefs_valid": finite_nonnegative and mass_valid,
            "beliefs_sum_to_one": mass_valid,
            "belief_mass_tolerance": 0.01,
            "max_belief_mass_error": float(np.max(np.abs(checked.sum(axis=1) - 1.0))) if finite_nonnegative else None,
            "actions_in_range": actions_valid
        }}
    }}

    # Save to JSON
    output_file = os.path.join(output_dir, "simulation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return output_file


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("JAX Model: {model_name} ({estimand})")
    print("=" * 60)

    # Create parameters
    params = create_params()
    print("\\n✅ Model parameters created")
    print(f"   A matrix shape: {{params['A_matrix'].shape}}")
    print(f"   B matrix shape: {{params['B_matrix'].shape}}")
    print(f"   C vector shape: {{params['C_vector'].shape}}")
    print(f"   D vector shape: {{params['D_vector'].shape}}")

    # Print model summary
    print(get_model_summary())

    print("\\nRunning {num_timesteps} observation(s): {estimand}")
    trajectory = run_simulation(params, num_steps={num_timesteps})
    print(f"   Final posterior: {{trajectory['final_belief']}}")
    if not INFERENCE_ONLY:
        print(f"   Actions taken: {{trajectory['actions']}}")

    # Save structured results for analysis step
    output_dir = os.environ.get('GNN_OUTPUT_DIR', 'jax_outputs')
    results_file = save_simulation_results(trajectory, params, "{model_name}", output_dir)
    print(f"\\n💾 Saved simulation results to: {{results_file}}")

    import json
    with open(results_file) as receipt_file:
        if not json.load(receipt_file)["success"]:
            raise ValueError("Generated simulation failed posterior/action validation")
    print("\\n✅ JAX model test successful!")
'''

        return code

    except Exception as e:
        raise ValueError(f"JAX model generation failed: {e}") from e
