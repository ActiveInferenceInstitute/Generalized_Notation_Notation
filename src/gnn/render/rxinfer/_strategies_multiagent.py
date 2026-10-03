#!/usr/bin/env python3
"""Native stigmergic multi-agent RxInfer.jl code generation (roadmap MAJ-03).

``MultiAgentStrategy`` previously rendered multi-agent specs as the POMDP
extractor's *composed joint* model: one joint state space of size
``prod(factor_sizes)`` (e.g. 9^3 = 729 states for the three-agent swarm),
with per-agent beliefs recovered only downstream by marginalising the joint
posterior.

This module implements the native alternative: when a spec declares two or
more complete agent groups (``A_agentN`` / ``B_agentN`` / ``C_agentN`` /
``D_agentN``) plus a shared environmental affordance (``env_signal`` +
``signal_decay``), the generated Julia script runs one genuine RxInfer
``pomdp_model`` inference per agent (native, per-agent state spaces — no
joint expansion) and reconstructs the shared environment trace:

- each agent deposits signal at its MAP position each timestep
  (stigmergy: indirect coordination through the environment);
- the shared trace decays by ``signal_decay`` per timestep;
- the resulting ``env_signal_trace`` is written to the results JSON
  alongside per-agent beliefs/actions/EFE and ``model_kind: multi_agent``.

Residual (documented in the module SPEC/roadmap): conditioning *action
selection* on the environment (inferring ``env_signal`` as a latent from
observations) requires env-conditioned likelihoods the swarm exemplar does
not declare; the current trace is post hoc and does not affect inference or
action selection.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, List

from gnn.render.multi_agent_common import (
    detect_env_conditioned,
    detect_env_coupling,
    has_env_conditioned_action_selection,
    validate_native_agent_groups,
)

__all__ = ["_generate_stigmergic_code"]


def _ordered_agent_groups(
    agents: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return agent groups ordered by numeric agent index."""

    def index_of(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    return [agents[name] for name in sorted(agents, key=index_of)]


def julia_matrix_literal(rows: List[List[float]]) -> str:
    """Format a list-of-rows as a Julia matrix literal ``[r11 r12 ...; ...]``."""
    parts = [" ".join(repr(float(x)) for x in row) for row in rows]
    return "[" + "; ".join(parts) + "]"


def _generate_stigmergic_code(
    gnn_spec: Dict[str, Any],
    model_display_name: str,
    kind_value: str,
) -> str:
    """Generate a native stigmergic multi-agent RxInfer.jl script.

    Args:
        gnn_spec: The parsed GNN specification dict.
        model_display_name: Model name for the script header / results.
        kind_value: The stamped ``model_kind`` (``"multi_agent"``).

    Returns:
        The Julia script text. Per-agent ``pomdp_model`` inference plus a
        post-hoc shared ``env_signal`` trace; no joint state-space expansion.
    """
    agents = validate_native_agent_groups(gnn_spec)
    env = detect_env_coupling(gnn_spec)
    env_cond = detect_env_conditioned(gnn_spec)
    env_action_conditioned = has_env_conditioned_action_selection(gnn_spec)
    env_coupling_mode = (
        "env_conditioned_signal_selection"
        if env_action_conditioned
        else "post_hoc_deposit_decay_trace"
    )
    ordered = _ordered_agent_groups(agents)
    agent_names = list(agents.keys())

    model_params = gnn_spec.get("model_parameters") or {}
    num_timesteps = int(model_params.get("num_timesteps", 20))
    seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
    action_precision = float(
        model_params.get("action_precision", model_params.get("gamma", 4.0))
    )
    inference_iterations = int(model_params.get("inference_iterations", 20))

    # Runtime only consumes descriptors/provenance from this record. Per-agent
    # numeric matrices are emitted below; the composed joint A/B/C/D tensors
    # are unused and can otherwise expand one script by tens of megabytes.
    embedded_spec = {
        key: value
        for key, value in gnn_spec.items()
        if key
        not in {
            "initialparameterization",
            "initial_parameterization",
            "structured_pomdp",
            "matrix_provenance",
        }
    }
    embedded_spec["matrix_provenance"] = {
        key: value
        for key, value in (gnn_spec.get("matrix_provenance") or {}).items()
        if key not in {"A", "B", "C", "D", "E"}
    }
    embedded_spec["matrix_provenance"].update(
        {
            f"{key}_{name}": metadata
            for name, group in agents.items()
            for key, metadata in group["matrix_provenance"].items()
        }
    )
    embedded_spec["native_agent_matrix_provenance"] = {
        name: group["matrix_provenance"] for name, group in agents.items()
    }
    embedded_spec["embedding_contract"] = {
        "schema": "native_agent_metadata_v1",
        "omitted": "unused composed joint tensors; native agent matrices emitted explicitly",
        "source_contract_sha256": hashlib.sha256(
            json.dumps(gnn_spec, sort_keys=True).encode()
        ).hexdigest(),
    }
    spec_json = json.dumps(embedded_spec, sort_keys=True)
    spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
    model_name_literal = json.dumps(str(model_display_name)).replace("$", r"\$")
    agent_names_literal = json.dumps(agent_names)

    # Per-agent matrix literals (JSON object syntax is valid Julia for
    # nested numeric arrays; tuples from the extractor serialise as arrays).
    # B is canonicalised to (next_state, previous_state, action) exactly as
    # the composed-joint path does, so per-agent semantics match.
    agent_as = json.dumps([m["A"] for m in ordered])
    agent_bs = json.dumps([m["B"] for m in ordered])
    agent_cs = json.dumps([m["C"] for m in ordered])
    agent_ds = json.dumps([m["D"] for m in ordered])

    if env is not None:
        env_initial_literal = json.dumps(env["initial"])
        env_decay_literal = repr(env["decay"])
        env_variable_literal = json.dumps(env["variable"])
    else:
        # No explicit env coupling: emit an empty initial trace so consumers
        # can rely on the key being present.
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
        # No env-conditioned likelihood: emit neutral defaults so the script
        # is well-formed but does not condition actions on a latent signal.
        env_obs_ll_literal = "zeros(0, 0)"
        env_signal_prior_literal = "[]"
        env_seek_literal = "0.0"

    return f'''#!/usr/bin/env julia
# RxInfer.jl stigmergic multi-agent simulation — native per-agent compilation
# Generated from GNN Model: {json.dumps(str(model_display_name))}
#
# This script runs one genuine RxInfer.jl pomdp_model inference per agent
# (native per-agent state spaces; NO joint state-space expansion). After all
# independent agent runs, it reconstructs env_signal from MAP positions using
# deposit + decay and writes that trace alongside per-agent beliefs/actions/EFE.
# When the spec declares an env-conditioned observation likelihood, each agent
# also infers the local signal level as a latent from its observations and
# conditions its action selection on that latent (signal-seeking / tropotaxis).

using Pkg
using RxInfer
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON
using Base64
using Dates

using GnnRxInferModels: pomdp_model

const SCHEMA_VERSION = "rxinfer_stigmergic_swarm_v1"
const MODEL_NAME = {model_name_literal}
const MODEL_KIND = "{kind_value}"
const NUM_AGENTS = {len(agent_names)}
const AGENTS = {agent_names_literal}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const ACTION_PRECISION = {action_precision}
const INFERENCE_ITERATIONS = {inference_iterations}
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

# Env-conditioned observation likelihood + latent signal prior (MAJ-03).
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

# --- Custom EFE computation (Active Inference domain, not RxInfer's) ---

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

function select_action(belief, A, B, C_eff, E_prior)
efe_values = [compute_efe(belief, action, A, B, C_eff) for action in 1:size(B, 3)]
policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
action = categorical_index(policy)
return action, efe_values, policy
end

function compute_efe_and_policy(belief, A, B, C_eff, E_prior)
efe_values = [compute_efe(belief, action, A, B, C_eff) for action in 1:size(B, 3)]
policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
return efe_values, policy
end

function signal_seeking_preference(C_pref, signal_belief)
# Env-conditioned action selection: boost preference for signal-rich
# observations (signal_low=idx2, signal_high=idx3) proportionally to the
# inferred latent signal level. C is over observations.
C_eff = copy(C_pref)
signal_belief = max.(signal_belief, 0.0)
if length(C_eff) >= 3 && length(signal_belief) >= 2
    C_eff[2] += SIGNAL_SEEK * signal_belief[2]  # signal_low
    C_eff[3] += SIGNAL_SEEK * signal_belief[3]  # signal_high
end
return C_eff
end

function update_signal_belief(signal_belief, observation)
# Bayes update of the latent signal-level belief from the observed category
# (obs indexing 1-based into ENV_OBS_LIKELIHOOD rows). If no likelihood was
# declared, keep the belief unchanged.
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

# --- Per-agent simulation: forward data collection + genuine RxInfer batch
# inference (no fallback — real RxInfer inference or nothing). Parameterised
# by the agent's own matrices so no joint state space is ever built.

function to_matrix(raw)
rows = length(raw)
cols = length(raw[1])
matrix = zeros(Float64, rows, cols)
for r in 1:rows
    for c in 1:cols
        matrix[r, c] = Float64(raw[r][c])
    end
end
return matrix
end

function to_tensor(raw)
ns = length(raw)
ps = length(raw[1])
ac = length(raw[1][1])
tensor = zeros(Float64, ns, ps, ac)
for i in 1:ns
    for j in 1:ps
        for k in 1:ac
            tensor[i, j, k] = Float64(raw[i][j][k])
        end
    end
end
return tensor
end

function simulate_agent(agent_name, raw_A, raw_B, raw_C, raw_D)
    A = to_matrix(raw_A)
    B = to_tensor(raw_B)
    C = Float64.(collect(raw_C))
    D = Float64.(collect(raw_D))
    n_states = size(A, 2)
    n_obs = size(A, 1)
    n_actions = size(B, 3)
    E = fill(1.0 / n_actions, n_actions)

    # Validate per-agent dimensions.
    if size(B) != (n_states, n_states, n_actions)
        error("agent $(agent_name) B shape $(size(B)) is not (n_states, n_states, n_actions)")
    end
    if length(C) != n_obs
        error("agent $(agent_name) C length $(length(C)) does not match n_obs $(n_obs)")
    end
    if length(D) != n_states
        error("agent $(agent_name) D length $(length(D)) does not match n_states $(n_states)")
    end

    C_pref = softmax(C)

    # MAJ-03: latent env-signal belief. When the spec declares an env-conditioned
    # observation likelihood, each agent maintains a belief over the local signal
    # level (none/low/high) updated from its observations via Bayes, and uses it
    # to modulate its action selection (signal-seeking). Otherwise a uniform
    # belief is kept and no conditioning is applied.
    signal_belief = isempty(ENV_SIGNAL_PRIOR) ? fill(1.0 / 3.0, 3) : Float64.(ENV_SIGNAL_PRIOR)
    signal_beliefs = Vector{{Vector{{Float64}}}}()

    # Phase 1 — forward simulation for data collection (hand-rolled EFE).
    current_state = categorical_index(D)
    current_belief = copy(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    action_seq_full = Int[]

    for step in 1:TIME_STEPS
        observation = categorical_index(A[:, current_state])
        emitting_state = current_state

        obs_onehot = [i == observation ? 1.0 : 0.0 for i in 1:n_obs]
        likelihood = A[observation, :]
        updated = current_belief .* likelihood
        if sum(updated) <= 0
            error("agent $(agent_name) belief update produced zero mass at step $(step)")
        end
        current_belief = updated ./ sum(updated)

        # MAJ-03: update the latent signal belief from this observation, then
        # condition action selection on the inferred signal (C_eff).
        signal_belief = update_signal_belief(signal_belief, observation)
        push!(signal_beliefs, copy(signal_belief))
        C_eff = signal_seeking_preference(C_pref, signal_belief)

        action, _efe_values, _policy = select_action(current_belief, A, B, C_eff, E)
        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)
        predicted = B[:, :, action] * current_belief
        current_belief = predicted ./ sum(predicted)

        push!(observations, observation - 1)
        push!(true_states, emitting_state - 1)
        push!(actions, action - 1)
        push!(action_seq_full, action)
    end

    # Phase 2 — genuine RxInfer batch inference (no fallback).
    obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:n_obs] for obs in observations]
    model_actions = copy(action_seq_full)
    while length(model_actions) < TIME_STEPS
        push!(model_actions, 1)
    end

    result = infer(
        model = pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS),
        data = (y = obs_seq,),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )

    # Phase 3 — posterior extraction (smoothed posteriors).
    posteriors_s = result.posteriors[:s]
    final_iter = posteriors_s[end]
    posterior_per_step = isa(final_iter, Vector) ? final_iter : [final_iter]

    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for t in 1:TIME_STEPS
        cat_dist = posterior_per_step[t]
        belief = copy(cat_dist.p)
        belief = max.(belief, 1e-16)
        belief ./= sum(belief)
        push!(beliefs, belief)

        # MAJ-03: condition the policy on the inferred latent signal level.
        sig_bel = isempty(signal_beliefs) ? signal_belief : signal_beliefs[t]
        C_eff = signal_seeking_preference(C_pref, sig_bel)
        efe_vals, pol = compute_efe_and_policy(belief, A, B, C_eff, E)
        push!(efe_per_action, efe_vals)
        push!(selected_efe, efe_vals[action_seq_full[t]])
        push!(policy_posterior, pol)
    end

    vfe_per_iteration = Float64.(result.free_energy)

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < n_actions, actions),
        "vfe_present" => !isempty(vfe_per_iteration) && all(v -> v > 0, vfe_per_iteration)
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
        "vfe_per_iteration" => vfe_per_iteration,
        "n_states" => n_states,
        "n_observations" => n_obs,
        "n_actions" => n_actions,
        "validation" => validation
    )
end

# --- Shared environment trace (stigmergic coupling) ---
# env[t+1] = ENV_DECAY * env[t] + sum of one-agent-per-timestep deposits at
# each agent's MAP position. The trace starts with the declared initial
# environment and has length TIME_STEPS + 1.

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
        A = AGENT_AS[i]
        B = AGENT_BS[i]
        C = AGENT_CS[i]
        D = AGENT_DS[i]
        result = simulate_agent(agent_name, A, B, C, D)
        per_agent[agent_name] = result
        all_valid = all_valid && result["validation"]["all_valid"]
        # MAP position per timestep (1-indexed) for the env trace.
        positions = [argmax(belief) for belief in result["beliefs"]]
        push!(agent_position_traces, positions)
    end

    n_cells = length(AGENT_DS[1])
    env_trace = compute_env_trace(agent_position_traces, n_cells)

    state_factors = get(get(GNN_SPEC, "model_parameters", Dict()), "state_factors", [])

    results = Dict(
        "schema_version" => SCHEMA_VERSION,
        "success" => true,
        "framework" => "RxInfer.jl",
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
        "vfe_per_iteration_by_agent" => Dict(agent => per_agent[agent]["vfe_per_iteration"] for agent in AGENTS),
        "state_factors" => state_factors,
        "model_parameters" => Dict(
            "num_agents" => NUM_AGENTS,
            "time_steps" => TIME_STEPS,
            "inference_iterations" => INFERENCE_ITERATIONS,
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
            "rxinfer_version" => package_version("RxInfer"),
            "julia_version" => string(VERSION),
            "inference_converged" => all(per_agent[agent]["validation"]["vfe_present"] for agent in AGENTS)
        ),
        "validation" => Dict(
            "all_valid" => all_valid,
            "per_agent" => Dict(agent => per_agent[agent]["validation"] for agent in AGENTS)
        )
    )

    open("simulation_results.json", "w") do file
        JSON.print(file, results, 2)
    end
    println("RxInfer.jl stigmergic simulation wrote simulation_results.json")
    return all_valid ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''
