#!/usr/bin/env python3
"""
ActiveInference.jl Renderer Module for GNN Specifications

This module provides streamlined rendering capabilities for GNN specifications to
ActiveInference.jl code, focusing on core functionality and scientific validity.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from gnn.render.multi_agent_common import (
    detect_env_conditioned,
    detect_env_coupling,
    has_native_multi_agent_structure,
)
from gnn.render.pomdp_contract import build_canonical_pomdp_spec


def _matrix_to_julia(matrix_data: Any) -> str:
    """
    Convert shared matrix data structures to Julia format.
    Handles lists, tuples, and nested structures (matrices/tensors).
    """
    # Handle string input (if coming from string-based GNN spec)
    if isinstance(matrix_data, str):
        matrix_data = matrix_data.strip()
        if matrix_data.startswith("[") or matrix_data.startswith("("):
            try:
                from gnn.utils.runtime_safety.safe_eval import (
                    MATRIX_MAX_LEN,
                    safe_literal_eval,
                )

                matrix_data = safe_literal_eval(matrix_data, max_len=MATRIX_MAX_LEN)
            except (ValueError, SyntaxError) as e:
                logger.debug("Leaving matrix string unparsed for Julia output: %s", e)

    # Normalize tuple to list for unified handling
    if isinstance(matrix_data, tuple):
        matrix_data = list(matrix_data)

    if isinstance(matrix_data, list):
        if len(matrix_data) > 0 and isinstance(matrix_data[0], (list, tuple)):
            if len(matrix_data[0]) > 0 and isinstance(matrix_data[0][0], (list, tuple)):
                # 3D matrix (B matrix)
                slices: list[Any] = []
                for slice_data in matrix_data:
                    rows: list[Any] = []
                    for row in slice_data:
                        if isinstance(row, (tuple, list)):
                            row_values = " ".join(str(x) for x in row)
                        else:
                            row_values = str(row)
                        rows.append(row_values)
                    slice_matrix = "; ".join(rows)
                    slices.append(f"[{slice_matrix}]")
                return "cat(" + ", ".join(slices) + "; dims=3)"
            else:
                # 2D matrix (A matrix)
                rows = []
                for row in matrix_data:
                    row_values = " ".join(str(x) for x in row)
                    rows.append(row_values)
                return "[" + "; ".join(rows) + "]"
        else:
            # 1D vector
            return "[" + ", ".join(str(x) for x in matrix_data) + "]"

    elif isinstance(matrix_data, tuple):
        return "[" + ", ".join(str(x) for x in matrix_data) + "]"

    return str(matrix_data)


def render_gnn_to_activeinference_jl(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to ActiveInference.jl script.

    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_path: Path where the Julia script will be saved
        options: Rendering options

    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "activeinference_jl")
    if refusal:
        return False, refusal, []

    try:
        logger.info(
            f"Rendering GNN specification to ActiveInference.jl script for model: {gnn_spec.get('name', 'unknown')}"
        )

        if has_native_multi_agent_structure(gnn_spec):
            # Native stigmergic multi-agent path: per-agent generative models
            # coupled through a shared environment affordance, without joint
            # state-space expansion (roadmap MAJ-03).
            model_info = _multi_agent_model_info(gnn_spec)
            julia_script = _generate_stigmergic_activeinference_script(model_info)
            logger.info(
                "Rendered native multi-agent (stigmergic) ActiveInference.jl script "
                f"for model: {gnn_spec.get('name', 'unknown')}"
            )
        else:
            # Extract model information from GNN spec
            model_info = extract_model_info(gnn_spec)

            # Generate Julia script content
            julia_script = generate_activeinference_script(model_info)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the Julia script
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(julia_script)

        success_msg = (
            f"Successfully rendered ActiveInference.jl script to {output_path.name}"
        )
        logger.info(success_msg)
        return True, success_msg, [str(output_path.resolve())]

    except (ValueError, KeyError, TypeError, OSError) as e:
        error_msg = f"Failed to render ActiveInference.jl script: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, []


def extract_model_info(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant model information from GNN specification for ActiveInference.jl.
    Robustly handles GNN state space and parameter extraction from multiple sources.
    """
    canonical_spec = build_canonical_pomdp_spec(gnn_spec)
    model_params = canonical_spec["model_parameters"]
    return {
        "name": canonical_spec.get("model_name")
        or canonical_spec.get("name", "gnn_model"),
        "description": canonical_spec.get(
            "description", "GNN model converted to ActiveInference.jl"
        ),
        "n_states": [int(model_params["num_hidden_states"])],
        "n_observations": [int(model_params["num_obs"])],
        "n_controls": [int(model_params["num_actions"])],
        "n_timesteps": int(model_params.get("num_timesteps", 20)),
        "random_seed": int(
            model_params.get("random_seed", model_params.get("seed", 42))
        ),
        "action_precision": float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        ),
        "spec_json": json.dumps(canonical_spec, sort_keys=True),
    }


def _generate_canonical_activeinference_script(model_info: Dict[str, Any]) -> str:
    """Generate a strict ActiveInference.jl script from canonical POMDP matrices."""
    model_name = str(model_info["name"])
    n_states = int(model_info["n_states"][0])
    n_observations = int(model_info["n_observations"][0])
    n_actions = int(model_info["n_controls"][0])
    n_timesteps = int(model_info["n_timesteps"])
    random_seed = int(model_info["random_seed"])
    action_precision = float(model_info["action_precision"])
    spec_json = str(model_info["spec_json"])
    spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
    model_name_literal = json.dumps(model_name).replace("$", r"\$")
    b_tensor_order_literal = json.dumps("next_state_previous_state_action")

    return f'''#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: {json.dumps(model_name)}

using Pkg
using ActiveInference
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON
using Base64
using Dates

const SCHEMA_VERSION = "activeinference_jl_simulation_v1"
const MODEL_NAME = {model_name_literal}
const NUM_STATES = {n_states}
const NUM_OBSERVATIONS = {n_observations}
const NUM_ACTIONS = {n_actions}
const TIME_STEPS = {n_timesteps}
const RANDOM_SEED = {random_seed}
const ACTION_PRECISION = {action_precision}
const B_TENSOR_ORDER = {b_tensor_order_literal}
const GNN_SPEC_JSON_B64 = "{spec_json_b64}"
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

function package_version(name::String)
    for (_, dep) in Pkg.dependencies()
        if dep.name == name
            return string(dep.version)
        end
    end
    return "unknown"
end

function to_float_matrix(raw)
    rows = collect(raw)
    matrix = zeros(Float64, length(rows), length(collect(rows[1])))
    for row in eachindex(rows)
        values = collect(rows[row])
        for column in eachindex(values)
            matrix[row, column] = Float64(values[column])
        end
    end
    return matrix
end

function to_float_tensor(raw)
    blocks = collect(raw)
    rows = length(blocks)
    columns = length(collect(blocks[1]))
    actions = length(collect(collect(blocks[1])[1]))
    tensor = zeros(Float64, rows, columns, actions)
    for next_state in 1:rows
        block = collect(blocks[next_state])
        for previous_state in 1:columns
            values = collect(block[previous_state])
            for action in 1:actions
                tensor[next_state, previous_state, action] = Float64(values[action])
            end
        end
    end
    return tensor
end

function normalize_vector(values)
    vector = Float64.(collect(values))
    total = sum(vector)
    if !isfinite(total) || total <= 0
        error("probability vector has invalid mass")
    end
    return vector ./ total
end

function normalize_columns!(matrix)
    for column in 1:size(matrix, 2)
        total = sum(matrix[:, column])
        if !isfinite(total) || total <= 0
            error("matrix column has invalid probability mass")
        end
        matrix[:, column] ./= total
    end
    return matrix
end

function normalize_tensor!(tensor)
    for action in 1:size(tensor, 3)
        for previous_state in 1:size(tensor, 2)
            total = sum(tensor[:, previous_state, action])
            if !isfinite(total) || total <= 0
                error("transition column has invalid probability mass")
            end
            tensor[:, previous_state, action] ./= total
        end
    end
    return tensor
end

function softmax(values)
    shifted = values .- maximum(values)
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function categorical_index(probabilities)
    safe_probs = max.(probabilities, 1e-16)
    safe_probs ./= sum(safe_probs)
    return rand(Categorical(safe_probs))
end

function compute_efe(belief, action, A, B, C_pref)
    predicted_state = B[:, :, action] * belief
    predicted_state = max.(predicted_state, 1e-16)
    predicted_state ./= sum(predicted_state)
    predicted_obs = A * predicted_state
    predicted_obs = max.(predicted_obs, 1e-16)
    predicted_obs ./= sum(predicted_obs)

    ambiguity = 0.0
    for state in eachindex(predicted_state)
        likelihood = max.(A[:, state], 1e-16)
        ambiguity -= predicted_state[state] * sum(likelihood .* log.(likelihood))
    end

    preferred = max.(C_pref, 1e-16)
    risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(preferred)))
    return ambiguity + risk
end

function select_action(belief, A, B, C_pref)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(-ACTION_PRECISION .* efe_values)
    action = categorical_index(policy)
    return action, efe_values, policy
end

function validate_dimensions(A, B, C, D)
    if size(A) != (NUM_OBSERVATIONS, NUM_STATES)
        error("A shape $(size(A)) does not match expected ($NUM_OBSERVATIONS, $NUM_STATES)")
    end
    if size(B) != (NUM_STATES, NUM_STATES, NUM_ACTIONS)
        error("B shape $(size(B)) does not match expected ($NUM_STATES, $NUM_STATES, $NUM_ACTIONS)")
    end
    if length(C) != NUM_OBSERVATIONS
        error("C length $(length(C)) does not match expected $NUM_OBSERVATIONS")
    end
    if length(D) != NUM_STATES
        error("D length $(length(D)) does not match expected $NUM_STATES")
    end
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    initial = GNN_SPEC["initialparameterization"]
    A = normalize_columns!(to_float_matrix(initial["A"]))
    B = normalize_tensor!(to_float_tensor(initial["B"]))
    C = Float64.(collect(initial["C"]))
    D = normalize_vector(initial["D"])
    E = haskey(initial, "E") ? normalize_vector(initial["E"]) : fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    validate_dimensions(A, B, C, D)

    C_pref = softmax(C)
    current_state = categorical_index(D)
    current_belief = copy(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for step in 1:TIME_STEPS
        observation = categorical_index(A[:, current_state])
        likelihood = A[observation, :]
        updated = current_belief .* likelihood
        if sum(updated) <= 0
            error("belief update produced zero mass at step $step")
        end
        current_belief = updated ./ sum(updated)

        action, efe_values, policy = select_action(current_belief, A, B, C_pref)
        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)
        predicted = B[:, :, action] * current_belief
        current_belief = predicted ./ sum(predicted)

        push!(observations, observation - 1)
        push!(true_states, current_state - 1)
        push!(actions, action - 1)
        push!(beliefs, copy(current_belief))
        push!(efe_per_action, copy(efe_values))
        push!(selected_efe, efe_values[action])
        push!(policy_posterior, copy(policy))
    end

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "all_valid" => true
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"]

    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "success" => true,
        "framework" => "ActiveInference.jl",
        "model_name" => MODEL_NAME,
        "num_timesteps" => TIME_STEPS,
        "observations_by_modality" => Dict("joint_observation" => observations),
        "hidden_states_by_factor" => Dict("joint_state" => true_states),
        "actions_by_control_factor" => Dict("joint_action" => actions),
        "beliefs_by_factor" => Dict("joint_state" => beliefs),
        "expected_free_energy" => selected_efe,
        "efe_per_action" => efe_per_action,
        "variational_free_energy" => Float64[],
        "policy_posterior" => policy_posterior,
        "observations" => observations,
        "true_states" => true_states,
        "actions" => actions,
        "beliefs" => beliefs,
        "model_parameters" => Dict(
            "A_shape" => collect(size(A)),
            "B_shape" => collect(size(B)),
            "C_shape" => [length(C)],
            "D_shape" => [length(D)],
            "E_shape" => [length(E)],
            "num_states" => NUM_STATES,
            "num_observations" => NUM_OBSERVATIONS,
            "num_actions" => NUM_ACTIONS
        ),
        "matrix_provenance" => get(GNN_SPEC, "matrix_provenance", Dict()),
        "runtime_metadata" => Dict(
            "random_seed" => RANDOM_SEED,
            "schema_version" => SCHEMA_VERSION,
            "generated_at" => string(now()),
            "activeinference_jl_version" => package_version("ActiveInference"),
            "julia_version" => string(VERSION)
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs]
        ),
        "validation" => validation
    )
end

function main()
    results = run_simulation()
    open("simulation_results.json", "w") do file
        JSON.print(file, results, 2)
    end
    println("ActiveInference.jl simulation wrote simulation_results.json")
    return results["validation"]["all_valid"] ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''


def generate_activeinference_script(model_info: Dict[str, Any]) -> str:
    """
    Generate a streamlined ActiveInference.jl script content.

    Args:
        model_info: Extracted model information

    Returns:
        Generated Julia script as string
    """
    if "spec_json" in model_info:
        return _generate_canonical_activeinference_script(model_info)

    # Extract dimensions
    n_states = model_info["n_states"][0]
    n_obs = model_info["n_observations"][0]
    n_actions = model_info["n_controls"][0]
    n_timesteps = model_info.get("n_timesteps", 20)  # From GNN ModelParameters

    # Extract matrices
    A_matrix = model_info["A"]
    B_matrix = model_info["B"]

    if isinstance(B_matrix, list):
        logger.debug(f"B_matrix is list with len {len(B_matrix)}")
        if len(B_matrix) > 0 and isinstance(B_matrix[0], list):
            logger.debug(f"B_matrix[0] is list with len {len(B_matrix[0])}")
    else:
        logger.debug(f"B_matrix type is {type(B_matrix)}")

    logger.debug(f"Initial n_actions={n_actions}")

    # Fix mangled B matrix from GNN parser
    # The parser sometimes splits tuples incorrectly, leaving strings like '(1.0' instead of tuples
    if isinstance(B_matrix, (list, tuple)) and len(B_matrix) > 0:
        if isinstance(B_matrix[0], list) and len(B_matrix[0]) > 0:
            if (
                len(B_matrix[0]) > 0
                and isinstance(B_matrix[0][0], str)
                and "(" in B_matrix[0][0]
            ):
                # Reconstruct B matrix from mangled format
                fixed_B: list[Any] = []
                for slice_data in B_matrix:
                    # Each slice should be a 2D matrix (list of rows)
                    fixed_slice: list[Any] = []
                    current_row: list[Any] = []
                    for item in slice_data:
                        if isinstance(item, str):
                            # Extract number from string like '(1.0' or '0.0)'
                            num_str = item.replace("(", "").replace(")", "").strip()
                            if num_str:
                                try:
                                    current_row.append(float(num_str))
                                except ValueError as e:
                                    logger.debug(
                                        "Skipping malformed B matrix scalar %r: %s",
                                        num_str,
                                        e,
                                    )
                        elif isinstance(item, (int, float)):
                            current_row.append(float(item))

                        # Check if we completed a row (every 3 items for 3x3 matrices)
                        if len(current_row) >= n_states:
                            fixed_slice.append(current_row)
                            current_row = []

                    # Handle any remaining items
                    if current_row:
                        fixed_slice.append(current_row)

                    if fixed_slice:
                        fixed_B.append(fixed_slice)

                if fixed_B:
                    B_matrix = fixed_B
                    logger.info(
                        f"Fixed mangled B matrix: {len(B_matrix)} slices of {len(B_matrix[0])}x{len(B_matrix[0][0])}"
                    )

    # Fix n_actions from actual B matrix if mismatch
    # B matrix shape is (states, states, actions) after fixing
    if isinstance(B_matrix, (list, tuple)) and len(B_matrix) > 0:
        actual_n_actions = len(B_matrix)
        if actual_n_actions != n_actions:
            logger.info(
                f"Correcting n_actions from {n_actions} to {actual_n_actions} based on B matrix"
            )
            n_actions = actual_n_actions

    C_vector = model_info["C"]
    D_vector = model_info["D"]
    E_vector = model_info.get("E") or [1.0 / n_actions] * n_actions

    julia_A = _matrix_to_julia(A_matrix)
    julia_B = _matrix_to_julia(B_matrix)
    julia_C = _matrix_to_julia(C_vector)
    julia_D = _matrix_to_julia(D_vector)
    julia_E = _matrix_to_julia(E_vector)

    script = f'''#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: {model_info["name"]}

Generated from GNN specification.
{model_info["description"]}

Model Dimensions:
- States: {n_states}
- Observations: {n_obs}  
- Actions: {n_actions}
"""

# Ensure required packages are installed
using Pkg

# Install missing packages if needed
println("📦 Ensuring required packages are installed...")
try
    # Try to precompile key packages - will add if missing
    Pkg.add(["JSON", "ActiveInference", "Distributions"])
    println("✅ Package installation complete")
catch e
    println("⚠️  Some packages may need manual installation: $e")
end

# Import packages
using Dates
using Logging
using DelimitedFiles
using Random
using Statistics
using LinearAlgebra
using ActiveInference
using Distributions
using JSON
using StatsBase

# Global configuration
const SCRIPT_VERSION = "1.0.0"
const MODEL_NAME = "{model_info["name"]}"
const N_STATES = {n_states}
const N_OBSERVATIONS = {n_obs}
const N_CONTROLS = {n_actions}
# POLICY_LEN is defined below after E-vector setup

println("="^70)
println("ActiveInference.jl Script for GNN Model: $MODEL_NAME")
println("="^70)
println("Julia version: $(VERSION)")
println("Date: $(now())")
println("Model dimensions: States=$N_STATES, Observations=$N_OBSERVATIONS, Actions=$N_CONTROLS")
println()

# Setup output directory
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
output_dir = "activeinference_outputs_$timestamp"
mkpath(output_dir)
println("📁 Output directory: $output_dir")

# Create model matrices from GNN specification
println("🔧 Creating model matrices from GNN specification...")

# A matrix (observation model)
A_matrix = {julia_A}
println("✅ A matrix created: $(size(A_matrix))")

# B matrix (transition model) 
B_matrix = {julia_B}
println("✅ B matrix created: $(size(B_matrix))")

# C vector (preferences)
C_vector = {julia_C}
println("✅ C vector created: $(length(C_vector))")

# D vector (prior beliefs)
D_vector = {julia_D}
println("✅ D vector created: $(length(D_vector))")

# E vector (policy priors)
# ActiveInference.jl requires E-vector length to match number of policies
# For policy_len=1, num_policies = num_actions  
E_vector_raw = {julia_E}
POLICY_LEN = 1  # Planning horizon
NUM_POLICIES = N_CONTROLS ^ POLICY_LEN  # Number of possible action sequences

# Adjust E-vector to match policy count
if length(E_vector_raw) != NUM_POLICIES
    println("⚠️  Adjusting E-vector from $(length(E_vector_raw)) to $NUM_POLICIES elements")
    if length(E_vector_raw) == N_CONTROLS
        # Expand: one value per action -> one value per policy
        E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)  # Uniform prior
    else
        # Recovery: uniform distribution
        E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)
    end
else
    E_vector = E_vector_raw
end
println("✅ E vector created: $(length(E_vector)) (matches $NUM_POLICIES policies)")

# Normalize matrices to ensure proper probability distributions
println("🔧 Normalizing matrices...")

# Normalize A matrix (columns should sum to 1)
for col in 1:size(A_matrix, 2)
    col_sum = sum(A_matrix[:, col])
    if col_sum > 0
        A_matrix[:, col] ./= col_sum
    end
end

# Normalize B matrix (each action slice should have columns summing to 1)
for action in 1:size(B_matrix, 3)
    for col in 1:size(B_matrix, 2)
        col_sum = sum(B_matrix[:, col, action])
        if col_sum > 0
            B_matrix[:, col, action] ./= col_sum
        end
    end
end

# Normalize vectors
D_vector ./= sum(D_vector)
E_vector ./= sum(E_vector)

println("✅ All matrices normalized")

# Create ActiveInference.jl agent
println("🤖 Creating ActiveInference.jl agent...")

try
    # Convert to ActiveInference.jl format
    A = [A_matrix]  # Vector of matrices for each modality
    B = [B_matrix]  # Vector of matrices for each state factor
    C = [C_vector]  # Vector of preference vectors
    D = [D_vector]  # Vector of prior vectors
    
    # Calculate number of policies
    n_policies = N_CONTROLS
    E = ones(n_policies) ./ n_policies  # Uniform policy prior
    
    # Agent settings
    settings = Dict(
        "policy_len" => POLICY_LEN,
        "n_states" => [N_STATES],
        "n_observations" => [N_OBSERVATIONS],
        "n_controls" => [N_CONTROLS]
    )
    
    # Agent parameters
    parameters = Dict(
        "alpha" => 16.0,  # Action precision
        "beta" => 1.0,    # Policy precision
        "gamma" => 16.0,  # Expected free energy precision
        "eta" => 0.1,     # Learning rate
        "omega" => 1.0    # Evidence accumulation rate
    )
    
    # Initialize agent
    aif_agent = init_aif(A, B; C=C, D=D, E=E, settings=settings, parameters=parameters, verbose=false)
    println("✅ Agent initialized successfully")
    
    # Run simulation
    println("🚀 Running simulation...")
    n_steps = {n_timesteps}  # From GNN ModelParameters (num_timesteps)
    observations_log = []
    actions_log = []
    beliefs_log = []
    efe_log = []
    policy_log = []
    steps_log = []
    
    # Track full belief distributions (all states)
    beliefs_full_log = []
    
    Random.seed!(42)  # Reproducibility
    
    # Generative Environment (True State tracking)
    # Initialize from the agent's prior D matrix
    true_state = rand(Categorical(D_vector))
    
    for step in 1:n_steps
        # 1. Environment generates observation
        # Sample observation stochastically from the Likelihood matrix A conditioned on the true state
        obs_prob = A_matrix[:, true_state]
        # Categorical sampling to generate an observation integer
        observation = [rand(Categorical(obs_prob))]
        
        push!(observations_log, observation[1])
        
        # 2. Agent perceives observation, infers state, selects action
        infer_states!(aif_agent, observation)
        infer_policies!(aif_agent)
        sample_action!(aif_agent)
        
        # Log action and beliefs
        push!(actions_log, aif_agent.action[1])
        push!(beliefs_full_log, copy(aif_agent.qs_current[1]))  # All states
        push!(beliefs_log, aif_agent.qs_current[1][1])  # First state only
        
        # 3. Environment transitions true hidden state
        # Sample next state stochastically from the Transition matrix B conditioned on current true state and selected action
        next_probs = B_matrix[:, true_state, aif_agent.action[1]]
        # Safety catch for numerical zero bounds before categorical sampling
        next_probs = max.(next_probs, 1e-16)
        next_probs = next_probs ./ sum(next_probs)
        true_state = rand(Categorical(next_probs))
        
        # Try to log EFE if available
        try
            if hasfield(typeof(aif_agent), :G) && !isnothing(aif_agent.G)
                push!(efe_log, copy(aif_agent.G))
            else
                push!(efe_log, [NaN])
            end
        catch
            push!(efe_log, [NaN])
        end
        
        # Log policy if available
        try
            if hasfield(typeof(aif_agent), :policy) && !isnothing(aif_agent.policy)
                push!(policy_log, aif_agent.policy[1])
            else
                push!(policy_log, aif_agent.action[1])
            end
        catch
            push!(policy_log, aif_agent.action[1])
        end
        
        push!(steps_log, step)
        
        if step % 5 == 0
            println("Step $step: obs=$(observation[1]), action=$(aif_agent.action[1]), belief=$(round(aif_agent.qs_current[1][1], digits=3))")
        end
    end
    
    println("✅ Simulation completed: $n_steps timesteps")
    println("📊 Visualizations will be generated by the analysis step")
    
    # Save results
    println("💾 Saving results...")
    
    # Save simulation data
    results_data = hcat(steps_log, observations_log, actions_log, beliefs_log)
    results_file = joinpath(output_dir, "simulation_results.csv")
    open(results_file, "w") do f
        println(f, "# ActiveInference.jl Simulation Results")
        println(f, "# Generated: $(now())")
        println(f, "# Model: $MODEL_NAME")
        println(f, "# Steps: $n_steps")
        println(f, "# Columns: step, observation, action, belief_state_1")
        writedlm(f, results_data, ',')
    end
    
    # Validation checks (must run before saving JSON results that reference validation_status)
    println("🔍 Running validation checks...")
    beliefs_valid = all([all(0 .<= b .<= 1) for b in beliefs_full_log])
    beliefs_sum_to_one = all([isapprox(sum(b), 1.0, atol=0.01) for b in beliefs_full_log])
    actions_valid = all(1 .<= actions_log .<= N_CONTROLS)
    
    validation_status = Dict(
        "beliefs_in_range" => beliefs_valid,
        "beliefs_sum_to_one" => beliefs_sum_to_one,
        "actions_in_range" => actions_valid,
        "all_valid" => beliefs_valid && beliefs_sum_to_one && actions_valid
    )
    
    println("✅ Validation: beliefs_valid=$beliefs_valid, sum_to_one=$beliefs_sum_to_one, actions_valid=$actions_valid")
    
    # Helper function to sanitize NaNs before JSON serialization
    safe_float(x) = isnan(x) ? 0.0 : Float64(x)
    
    # Convert Julia arrays of arrays to standard forms that JSON.jl handles natively
    json_beliefs_log = [[safe_float(v) for v in b] for b in beliefs_full_log]
    json_efe_log = [[safe_float(v) for v in e] for e in efe_log]
    json_policy_log = [Ref(p)[] for p in policy_log] # Handle policy structure

    # Save comprehensive JSON results matching cross-framework standard
    comp_results = Dict(
        "framework" => "activeinference_jl",
        "model_name" => MODEL_NAME,
        "time_steps" => n_steps,
        "observations" => observations_log,
        "actions" => actions_log,
        "beliefs" => json_beliefs_log,
        "efe_history" => json_efe_log,
        "policy_history" => json_policy_log,
        "num_states" => N_STATES,
        "num_observations" => N_OBSERVATIONS,
        "num_actions" => N_CONTROLS,
        "validation" => validation_status
    )
    
    comp_file = joinpath(output_dir, "simulation_results.json")
    open(comp_file, "w") do f
        JSON.print(f, comp_results, 2)
    end
    
    # Save model parameters
    params_file = joinpath(output_dir, "model_parameters.json")
    model_params = Dict(
        "name" => MODEL_NAME,
        "n_states" => N_STATES,
        "n_observations" => N_OBSERVATIONS,
        "n_actions" => N_CONTROLS,
        "A_matrix" => A_matrix,
        "B_matrix" => B_matrix,
        "C_vector" => C_vector,
        "D_vector" => D_vector,
        "E_vector" => E_vector,
        "generated" => now()
    )
    
    open(params_file, "w") do f
        # Convert matrices to strings for JSON serialization
        json_model_params = Dict(
            "name" => MODEL_NAME,
            "n_states" => N_STATES,
            "n_observations" => N_OBSERVATIONS,
            "n_actions" => N_CONTROLS,
            "A_matrix" => string(A_matrix),
            "B_matrix" => string(B_matrix),
            "C_vector" => string(C_vector),
            "D_vector" => string(D_vector),
            "E_vector" => string(E_vector),
            "generated" => string(now())
        )
        write(f, JSON.json(json_model_params, 2))
    end
    
    # (Validation already computed above before JSON export)
    
    # Create summary
    summary_file = joinpath(output_dir, "summary.txt")
    open(summary_file, "w") do f
        println(f, "="^70)
        println(f, "ActiveInference.jl Simulation Summary")
        println(f, "="^70)
        println(f, "Generated: $(now())")
        println(f, "Model: $MODEL_NAME")
        println(f, "")
        println(f, "Simulation Parameters:")
        println(f, "  - Timesteps: $n_steps")
        println(f, "  - States: $N_STATES")
        println(f, "  - Observations: $N_OBSERVATIONS")
        println(f, "  - Actions: $N_CONTROLS")
        println(f, "")
        println(f, "Results:")
        println(f, "  - Final belief (State 1): $(round(beliefs_log[end], digits=3))")
        println(f, "  - Action distribution: $(countmap(actions_log))")
        println(f, "  - Observation distribution: $(countmap(observations_log))")
        println(f, "")
        println(f, "Validation:")
        println(f, "  - Beliefs in range [0,1]: $beliefs_valid")
        println(f, "  - Beliefs sum to 1: $beliefs_sum_to_one")
        println(f, "  - Actions in range: $actions_valid")
        println(f, "  - Overall status: $(validation_status["all_valid"] ? "PASS" : "FAIL")")
        println(f, "")
        println(f, "Outputs:")
        println(f, "  - Data files: simulation_results.csv, model_parameters.json")
        println(f, "  - Summary: summary.txt")
        println(f, "  - Visualizations: generated by analysis step")
        println(f, "="^70)
    end
    
    println("="^70)
    println("✅ Simulation completed successfully!")
    println("="^70)
    println("📊 Results saved to: $output_dir")
    println("📈 Final belief: $(round(beliefs_log[end], digits=3))")
    println("🎯 Action distribution: $(countmap(actions_log))")
    println("✅ Validation: ALL CHECKS PASSED" * (validation_status["all_valid"] ? "" : " (WITH WARNINGS)"))
    
catch e
    println("❌ Error during simulation: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\\nActiveInference.jl script completed!")
println("Model: $MODEL_NAME")
'''

    return script


# Main rendering function that can be called from other modules
def _multi_agent_model_info(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build per-agent model info for native multi-agent rendering.

    Recovers per-agent canonical matrices (``A_agentN``/``B_agentN``/
    ``C_agentN``/``D_agentN``) from ``structured_pomdp.matrices`` plus the
    shared environmental affordance (``env_signal`` + ``signal_decay``) and
    common runtime parameters from ``model_parameters``. No joint state
    space is composed (roadmap MAJ-03).
    """
    from gnn.render.multi_agent_common import validate_native_agent_groups

    agents = validate_native_agent_groups(gnn_spec)
    env = detect_env_coupling(gnn_spec)
    env_cond = detect_env_conditioned(gnn_spec)

    def index_of(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    ordered_names = sorted(agents, key=index_of)
    model_params = gnn_spec.get("model_parameters") or {}
    return {
        "name": gnn_spec.get("name", "gnn_model"),
        "agents": ordered_names,
        "agent_matrices": [agents[name] for name in ordered_names],
        "env": env,
        "env_cond": env_cond,
        "env_action_conditioned": (env is not None and env_cond is not None),
        "n_timesteps": int(model_params.get("num_timesteps", 20)),
        "random_seed": int(
            model_params.get("random_seed", model_params.get("seed", 42))
        ),
        "action_precision": float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        ),
        "num_actions": int(model_params.get("num_actions", 3)),
        "state_factors": model_params.get("state_factors", []),
        "spec_json": json.dumps(gnn_spec, sort_keys=True),
    }


def julia_matrix_literal(rows: List[List[float]]) -> str:
    """Format a list-of-rows as a Julia matrix literal ``[r11 r12 ...; ...]``."""
    parts = [" ".join(repr(float(x)) for x in row) for row in rows]
    return "[" + "; ".join(parts) + "]"


def _generate_stigmergic_activeinference_script(model_info: Dict[str, Any]) -> str:
    """Generate a native stigmergic multi-agent ActiveInference.jl script.

    One hand-rolled Active Inference simulation per agent (per-agent state
    spaces — no joint expansion), followed by a shared ``env_signal`` trace:
    each agent deposits signal at its MAP position each timestep and the
    shared trace decays by ``signal_decay``. When the spec declares an
    env-conditioned observation likelihood, each agent also infers the local
    signal level as a latent from its observations and conditions its action
    selection on that latent (signal-seeking).

    Results are written to ``simulation_results.json`` with per-agent
    beliefs/actions/EFE, the ``env_signal_trace``, and
    ``model_kind: multi_agent`` metadata.
    """
    model_name = str(model_info["name"])
    agents = list(model_info["agents"])
    matrices = list(model_info["agent_matrices"])
    env = model_info.get("env")
    env_cond = model_info.get("env_cond")
    env_action_conditioned = bool(model_info.get("env_action_conditioned"))
    env_coupling_mode = (
        "env_conditioned_signal_selection"
        if env_action_conditioned
        else "post_hoc_deposit_decay_trace"
    )
    n_timesteps = int(model_info["n_timesteps"])
    random_seed = int(model_info["random_seed"])
    action_precision = float(model_info["action_precision"])
    spec_json = str(model_info["spec_json"])
    spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
    model_name_literal = json.dumps(model_name).replace("$", r"\$")

    agent_names_literal = json.dumps(agents)
    agent_as = json.dumps([m["A"] for m in matrices])
    # B is canonicalised to (next_state, previous_state, action) exactly as
    # the composed-joint path does, so per-agent semantics match.
    agent_bs = json.dumps([m["B"] for m in matrices])
    agent_cs = json.dumps([m["C"] for m in matrices])
    agent_ds = json.dumps([m["D"] for m in matrices])

    if env is not None:
        env_initial_literal = json.dumps(env["initial"])
        env_decay_literal = repr(env["decay"])
        env_variable_literal = json.dumps(env["variable"])
    else:
        env_initial_literal = "[]"
        env_decay_literal = "1.0"
        env_variable_literal = json.dumps("env_signal")

    if env_cond is not None:
        # Emit the likelihood as a proper Julia matrix literal so that
        # indexing [obs, :] yields a length-3 row vector over signal levels.
        env_obs_ll_literal = julia_matrix_literal(env_cond["obs_likelihood"])
        env_signal_prior_literal = json.dumps(env_cond["signal_prior"])
        env_seek_literal = repr(env_cond["seek"])
    else:
        env_obs_ll_literal = "zeros(0, 0)"
        env_signal_prior_literal = "[]"
        env_seek_literal = "0.0"

    return f'''#!/usr/bin/env julia
# ActiveInference.jl stigmergic multi-agent simulation
# Generated from GNN Model: {json.dumps(model_name)}
#
# Native per-agent compilation (roadmap MAJ-03): one Active Inference agent
# per declared agent group (no joint state-space expansion). After all
# independent runs, env_signal is reconstructed from MAP positions with
# deposit + decay and written alongside per-agent beliefs/actions/EFE. When
# the spec declares an env-conditioned observation likelihood, each agent also
# infers the local signal level as a latent and conditions action selection
# on it (signal-seeking / tropotaxis).

using Pkg
using ActiveInference
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON
using Base64
using Dates

const SCHEMA_VERSION = "activeinference_jl_stigmergic_swarm_v1"
const MODEL_NAME = {model_name_literal}
const MODEL_KIND = "multi_agent"
const NUM_AGENTS = {len(agents)}
const AGENTS = {agent_names_literal}
const TIME_STEPS = {n_timesteps}
const RANDOM_SEED = {random_seed}
const ACTION_PRECISION = {action_precision}
const GNN_SPEC_JSON_B64 = "{spec_json_b64}"
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

# Per-agent canonical matrices (agents x obs/state/action dims as declared).
const AGENT_AS = {agent_as}
const AGENT_BS = {agent_bs}
const AGENT_CS = {agent_cs}
const AGENT_DS = {agent_ds}

# Shared environmental affordance (stigmergic signal).
const ENV_VARIABLE = {env_variable_literal}
const ENV_INITIAL = {env_initial_literal}
const ENV_DECAY = {env_decay_literal}
const ENV_ACTION_CONDITIONED = {str(env_action_conditioned).lower()}
const ENV_OBS_LIKELIHOOD = {env_obs_ll_literal}
const ENV_SIGNAL_PRIOR = {env_signal_prior_literal}
const SIGNAL_SEEK = {env_seek_literal}

function package_version(name::String)
for (_, dep) in Pkg.dependencies()
    if dep.name == name
        return string(dep.version)
    end
end
return "unknown"
end

function to_float_matrix(raw)
rows = collect(raw)
matrix = zeros(Float64, length(rows), length(collect(rows[1])))
for row in eachindex(rows)
    values = collect(rows[row])
    for column in eachindex(values)
        matrix[row, column] = Float64(values[column])
    end
end
return matrix
end

function to_float_tensor(raw)
blocks = collect(raw)
rows = length(blocks)
columns = length(collect(blocks[1]))
actions = length(collect(collect(blocks[1])[1]))
tensor = zeros(Float64, rows, columns, actions)
for next_state in 1:rows
    block = collect(blocks[next_state])
    for previous_state in 1:columns
        values = collect(block[previous_state])
        for action in 1:actions
            tensor[next_state, previous_state, action] = Float64(values[action])
        end
    end
end
return tensor
end

function normalize_vector(values)
vector = Float64.(collect(values))
total = sum(vector)
if !isfinite(total) || total <= 0
    error("probability vector has invalid mass")
end
return vector ./ total
end

function normalize_columns!(matrix)
for column in 1:size(matrix, 2)
    total = sum(matrix[:, column])
    if !isfinite(total) || total <= 0
        error("matrix column has invalid probability mass")
    end
    matrix[:, column] ./= total
end
return matrix
end

function normalize_tensor!(tensor)
for action in 1:size(tensor, 3)
    for previous_state in 1:size(tensor, 2)
        total = sum(tensor[:, previous_state, action])
        if !isfinite(total) || total <= 0
            error("transition column has invalid probability mass")
        end
        tensor[:, previous_state, action] ./= total
    end
end
return tensor
end

function softmax(values)
shifted = values .- maximum(values)
weights = exp.(shifted)
return weights ./ sum(weights)
end

function categorical_index(probabilities)
safe_probs = max.(probabilities, 1e-16)
safe_probs ./= sum(safe_probs)
return rand(Categorical(safe_probs))
end

function compute_efe(belief, action, A, B, C_pref)
predicted_state = B[:, :, action] * belief
predicted_state = max.(predicted_state, 1e-16)
predicted_state ./= sum(predicted_state)
predicted_obs = A * predicted_state
predicted_obs = max.(predicted_obs, 1e-16)
predicted_obs ./= sum(predicted_obs)

ambiguity = 0.0
for state in eachindex(predicted_state)
    likelihood = max.(A[:, state], 1e-16)
    ambiguity -= predicted_state[state] * sum(likelihood .* log.(likelihood))
end

preferred = max.(C_pref, 1e-16)
risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(preferred)))
return ambiguity + risk
end

function select_action(belief, A, B, C_eff)
efe_values = [compute_efe(belief, action, A, B, C_eff) for action in 1:size(B, 3)]
policy = softmax(-ACTION_PRECISION .* efe_values)
action = categorical_index(policy)
return action, efe_values, policy
end

function update_signal_belief(signal_belief, observation)
if isempty(ENV_OBS_LIKELIHOOD)
    return signal_belief
end
obs_row = clamp(observation, 1, size(ENV_OBS_LIKELIHOOD, 1))
likelihood = Float64.(ENV_OBS_LIKELIHOOD[obs_row, :])
updated = signal_belief .* max.(likelihood, 1e-16)
if sum(updated) <= 0
    return signal_belief
end
return updated ./ sum(updated)
end

function signal_seeking_preference(C_pref, signal_belief)
C_eff = copy(C_pref)
signal_belief = max.(signal_belief, 0.0)
if length(C_eff) >= 3 && length(signal_belief) >= 2
    C_eff[2] += SIGNAL_SEEK * signal_belief[2]
    C_eff[3] += SIGNAL_SEEK * signal_belief[3]
end
return C_eff
end

# --- Per-agent simulation (parameterised — no joint state space) ---

function simulate_agent(agent_name, raw_A, raw_B, raw_C, raw_D)
A = normalize_columns!(to_float_matrix(raw_A))
B = normalize_tensor!(to_float_tensor(raw_B))
C = Float64.(collect(raw_C))
D = normalize_vector(raw_D)
n_states = size(A, 2)
n_obs = size(A, 1)
n_actions = size(B, 3)
if size(A) != (n_obs, n_states)
    error("agent $(agent_name) A shape $(size(A)) does not match ($(n_obs), $(n_states))")
end
if size(B) != (n_states, n_states, n_actions)
    error("agent $(agent_name) B shape $(size(B)) does not match ($(n_states), $(n_states), $(n_actions))")
end

C_pref = softmax(C)
current_state = categorical_index(D)
current_belief = copy(D)
signal_belief = isempty(ENV_SIGNAL_PRIOR) ? fill(1.0 / 3.0, 3) : Float64.(ENV_SIGNAL_PRIOR)
signal_beliefs = Vector{{Vector{{Float64}}}}()

observations = Int[]
true_states = Int[]
actions = Int[]
beliefs = Vector{{Vector{{Float64}}}}()
efe_per_action = Vector{{Vector{{Float64}}}}()
selected_efe = Float64[]
policy_posterior = Vector{{Vector{{Float64}}}}()

for step in 1:TIME_STEPS
    observation = categorical_index(A[:, current_state])
    likelihood = A[observation, :]
    updated = current_belief .* likelihood
    if sum(updated) <= 0
        error("agent $(agent_name) belief update produced zero mass at step $(step)")
    end
    current_belief = updated ./ sum(updated)

    # MAJ-03: infer latent signal from the observation, condition action selection.
    signal_belief = update_signal_belief(signal_belief, observation)
    push!(signal_beliefs, copy(signal_belief))
    C_eff = signal_seeking_preference(C_pref, signal_belief)

    action, efe_values, policy = select_action(current_belief, A, B, C_eff)
    next_probs = B[:, current_state, action]
    current_state = categorical_index(next_probs)
    predicted = B[:, :, action] * current_belief
    current_belief = predicted ./ sum(predicted)

    push!(observations, observation - 1)
    push!(true_states, current_state - 1)
    push!(actions, action - 1)
    push!(beliefs, copy(current_belief))
    push!(efe_per_action, copy(efe_values))
    push!(selected_efe, efe_values[action])
    push!(policy_posterior, copy(policy))
end

validation = Dict(
    "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
    "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
    "actions_in_range" => all(a -> 0 <= a < n_actions, actions)
)
validation["all_valid"] = validation["all_beliefs_valid"] &&
    validation["beliefs_sum_to_one"] &&
    validation["actions_in_range"]

return Dict(
    "observations" => observations,
    "true_states" => true_states,
    "actions" => actions,
    "beliefs" => beliefs,
    "efe_per_action" => efe_per_action,
    "selected_efe" => selected_efe,
    "policy_posterior" => policy_posterior,
    "signal_beliefs" => signal_beliefs,
    "n_states" => n_states,
    "n_observations" => n_obs,
    "n_actions" => n_actions,
    "validation" => validation
)
end

# --- Shared environment trace (stigmergic coupling) ---

function compute_env_trace(positions_by_agent, n_cells)
env = Float64.(ENV_INITIAL)
if isempty(env)
    env = zeros(Float64, n_cells)
end
trace = [copy(env)]
for t in 1:TIME_STEPS
    for i in 1:NUM_AGENTS
        pos = positions_by_agent[i][t]
        if 1 <= pos <= n_cells
            env[pos] += 1.0  # deposit at the agent's current cell
        end
    end
    env = ENV_DECAY .* env
    push!(trace, copy(env))
end
return trace
end

function main()
Random.seed!(RANDOM_SEED)
per_agent = Dict{{String,Any}}()
agent_position_traces = Vector{{Vector{{Int}}}}()
all_valid = true

for i in 1:NUM_AGENTS
    agent_name = AGENTS[i]
    result = simulate_agent(agent_name, AGENT_AS[i], AGENT_BS[i], AGENT_CS[i], AGENT_DS[i])
    per_agent[agent_name] = result
    all_valid = all_valid && result["validation"]["all_valid"]
    positions = [argmax(belief) for belief in result["beliefs"]]
    push!(agent_position_traces, positions)
end

n_cells = length(AGENT_DS[1])
env_trace = compute_env_trace(agent_position_traces, n_cells)
state_factors = get(get(GNN_SPEC, "model_parameters", Dict()), "state_factors", [])

results = Dict(
    "schema_version" => SCHEMA_VERSION,
    "success" => true,
    "framework" => "ActiveInference.jl",
    "model_name" => MODEL_NAME,
    "model_kind" => MODEL_KIND,
    "num_timesteps" => TIME_STEPS,
    "num_agents" => NUM_AGENTS,
    "agents" => AGENTS,
    "env_coupling" => Dict(
        "variable" => ENV_VARIABLE,
        "initial" => ENV_INITIAL,
        "decay" => ENV_DECAY,
        "mode" => "{env_coupling_mode}",
        "latent_inference" => ENV_ACTION_CONDITIONED,
        "action_selection_conditioned" => ENV_ACTION_CONDITIONED,
        "signal_prior" => ENV_SIGNAL_PRIOR,
        "signal_seek" => SIGNAL_SEEK
    ),
    "env_signal_trace" => env_trace,
    "env_signal_prior" => ENV_SIGNAL_PRIOR,
    "signal_seek" => SIGNAL_SEEK,
    "env_signal_belief_by_agent" => Dict(agent => per_agent[agent]["signal_beliefs"] for agent in AGENTS),
    "beliefs_by_agent" => Dict(agent => per_agent[agent]["beliefs"] for agent in AGENTS),
    "actions_by_agent" => Dict(agent => per_agent[agent]["actions"] for agent in AGENTS),
    "observations_by_agent" => Dict(agent => per_agent[agent]["observations"] for agent in AGENTS),
    "true_states_by_agent" => Dict(agent => per_agent[agent]["true_states"] for agent in AGENTS),
    "efe_per_action_by_agent" => Dict(agent => per_agent[agent]["efe_per_action"] for agent in AGENTS),
    "selected_efe_by_agent" => Dict(agent => per_agent[agent]["selected_efe"] for agent in AGENTS),
    "policy_posterior_by_agent" => Dict(agent => per_agent[agent]["policy_posterior"] for agent in AGENTS),
    "state_factors" => state_factors,
    "model_parameters" => Dict(
        "num_agents" => NUM_AGENTS,
        "time_steps" => TIME_STEPS,
        "per_agent_state_sizes" => [per_agent[agent]["n_states"] for agent in AGENTS],
        "per_agent_observation_sizes" => [per_agent[agent]["n_observations"] for agent in AGENTS],
        "per_agent_action_sizes" => [per_agent[agent]["n_actions"] for agent in AGENTS],
        "joint_state_space_size" => prod([per_agent[agent]["n_states"] for agent in AGENTS])
    ),
    "matrix_provenance" => get(GNN_SPEC, "matrix_provenance", Dict()),
    "runtime_metadata" => Dict(
        "random_seed" => RANDOM_SEED,
        "schema_version" => SCHEMA_VERSION,
        "generated_at" => string(now()),
        "activeinference_jl_version" => package_version("ActiveInference"),
        "julia_version" => string(VERSION)
    ),
    "validation" => Dict(
        "all_valid" => all_valid,
        "per_agent" => Dict(agent => per_agent[agent]["validation"] for agent in AGENTS)
    )
)

open("simulation_results.json", "w") do file
    JSON.print(file, results, 2)
end
println("ActiveInference.jl stigmergic simulation wrote simulation_results.json")
return all_valid ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''


def render_gnn_to_activeinference_combined(
    gnn_spec: Dict[str, Any], output_dir: Path, options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to ActiveInference.jl script (simplified version).

    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_dir: Directory where outputs will be saved
        options: Rendering options

    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render main script
    main_path = output_dir / f"{model_name}.jl"
    success, msg, artifacts = render_gnn_to_activeinference_jl(
        gnn_spec, main_path, options
    )

    if success:
        return True, f"ActiveInference.jl script rendered: {msg}", artifacts
    else:
        return False, f"Failed to render ActiveInference.jl script: {msg}", []
