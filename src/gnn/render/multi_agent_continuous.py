"""Native independent Gaussian agent composition with explicit source blocks.

No agent is cloned from a shared flat system. Every declared agent supplies its
own complete LGSSM, and independent coupling must be declared by the author.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import numpy as np

from gnn.render.continuous_common import (
    REQUIRED_KEYS,
    ContinuousSpec,
    _scalar,
    extract_continuous_spec,
)

AGENT_CONTINUOUS_KEY = re.compile(
    r"^(F|H|Q|R|prior_mean|prior_cov|goal_mean|control_gain)_agent([1-9]\d*)$"
)
REQUIRED_DECLARATIONS = "declare agent_coupling: independent and complete F_agentN/H_agentN/Q_agentN/R_agentN/prior_mean_agentN/prior_cov_agentN blocks for every agent"


def extract_multi_agent_continuous_specs(
    spec: dict[str, Any],
) -> dict[str, ContinuousSpec]:
    """Validate all source agent identities, Gaussian blocks and control pairs."""
    initial = spec.get("initialparameterization") or spec.get(
        "initial_parameterization"
    )
    params = spec.get("model_parameters") or {}
    if not isinstance(initial, dict) or params.get("agent_coupling") != "independent":
        raise ValueError(f"unsupported-composition: {REQUIRED_DECLARATIONS}")
    if any(key in initial for key in REQUIRED_KEYS):
        raise ValueError(
            "unsupported-composition: a shared flat Gaussian block cannot be assigned independent agent semantics"
        )
    count = params.get("nr_agents", params.get("num_agents"))
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Independent continuous agents require integer nr_agents >= 2")
    groups: dict[int, dict[str, Any]] = {}
    for key, value in initial.items():
        match = AGENT_CONTINUOUS_KEY.fullmatch(str(key))
        if match:
            groups.setdefault(int(match[2]), {})[match[1]] = value
    if set(groups) != set(range(1, count + 1)):
        raise ValueError(
            f"Agent blocks must match contiguous declared identities agent1..agent{count}"
        )
    base_seed = _scalar(params.get("random_seed", params.get("seed", 42)))
    if not np.isfinite(base_seed) or base_seed < 0 or base_seed != int(base_seed):
        raise ValueError("random_seed must be a nonnegative integer")
    result = {}
    for index in range(1, count + 1):
        agent_params = {
            **params,
            "random_seed": params.get(
                f"random_seed_agent{index}", int(base_seed) + index - 1
            ),
        }
        try:
            result[f"agent{index}"] = extract_continuous_spec(
                {
                    "model_name": f"{spec.get('model_name', 'Continuous agents')}: agent{index}",
                    "initialparameterization": groups[index],
                    "model_parameters": agent_params,
                }
            )
        except ValueError as exc:
            raise ValueError(f"agent{index}: {exc}") from exc
    return result


def agent_result_record(
    spec: dict[str, Any], backend: str, names: list[str]
) -> dict[str, Any]:
    """Immutable source and schema record shared by both native generators."""
    import hashlib

    return {
        "schema_version": "gnn_multi_agent_continuous_v1",
        "model_kind": "multi_agent_continuous",
        "model_name": spec.get("model_name", spec.get("name", "Continuous agents")),
        "framework": backend,
        "agents": names,
        "agent_coupling": "independent",
        "num_timesteps": (spec.get("model_parameters") or {}).get("num_timesteps", 20),
        "runtime_metadata": {
            "model_kind": "multi_agent_continuous",
            "inference_semantics": "independent agent Gaussian posteriors",
            "seed_derivation": "random_seed_agentN if declared, otherwise random_seed + N - 1",
            "source_spec_sha256": hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest(),
        },
    }


def generate_multi_agent_continuous_script(spec: dict[str, Any], backend: str) -> str:
    """Compose genuine native continuous programs in isolated module namespaces."""
    agents = extract_multi_agent_continuous_specs(spec)
    names = list(agents)
    record = agent_result_record(spec, backend, names)
    keys = (
        "beliefs",
        "posterior_cov",
        "true_states_continuous",
        "observations_continuous",
        "controls",
        "vfe_per_iteration",
        "variational_free_energy",
    )
    if backend == "jax":
        import textwrap

        from gnn.render.continuous_script import generate_continuous_script

        functions = []
        for name, agent in agents.items():
            program = generate_continuous_script(agent, "jax").rsplit(
                '\nif __name__ == "__main__":', 1
            )[0]
            functions.append(
                f"def run_{name}():\n"
                + textwrap.indent(program, "    ")
                + "\n    return run_simulation()\n"
            )
        native_functions = "\n".join(functions)
        function_map = "{" + ", ".join(f"{name!r}: run_{name}" for name in names) + "}"
        return f'''#!/usr/bin/env python3
"""Native JAX independent continuous agents; source-declared Gaussian blocks."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
RECORD = {record!r}
KEYS = {keys!r}
AGENT_SEEDS = { {name: agent.random_seed for name, agent in agents.items()}!r}
{native_functions}
FUNCTIONS = {function_map}

def run_simulation():
    output = Path(os.environ.get("GNN_OUTPUT_DIR", ".")).resolve()
    previous = os.environ.get("GNN_OUTPUT_DIR")
    results = {{}}
    try:
        with TemporaryDirectory(prefix="gnn-agent-lgssm-") as directory:
            for name, native in FUNCTIONS.items():
                os.environ["GNN_OUTPUT_DIR"] = str(Path(directory) / name)
                results[name] = native()
    finally:
        if previous is None:
            os.environ.pop("GNN_OUTPUT_DIR", None)
        else:
            os.environ["GNN_OUTPUT_DIR"] = previous
    record = {{**RECORD, "runtime_metadata": {{**RECORD["runtime_metadata"]}}}}
    for key in KEYS:
        record[key + "_by_agent"] = {{name: result.get(key, []) for name, result in results.items()}}
    record["validation"] = {{"all_valid": all(r["validation"]["all_valid"] for r in results.values()), "per_agent": {{name: r["validation"] for name, r in results.items()}}}}
    record["runtime_metadata"]["agent_runtime"] = {{name: {{"jax_version": r.get("jax_version"), "seed": AGENT_SEEDS[name]}} for name, r in results.items()}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulation_results.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    return record

if __name__ == "__main__":
    raise SystemExit(0 if run_simulation()["validation"]["all_valid"] else 1)
'''
    if backend != "rxinfer":
        raise ValueError(
            f"unsupported-composition: independent continuous agents are supported by JAX and RxInfer; got {backend}"
        )
    from gnn.render.rxinfer._strategies_continuous import _generate_continuous_code

    initial = (
        spec.get("initialparameterization")
        or spec.get("initial_parameterization")
        or {}
    )
    programs = []
    for name, agent in agents.items():
        block = {
            key: value for key, value in initial.items() if key.endswith("_" + name)
        }
        block = {key[: -(len(name) + 1)]: value for key, value in block.items()}
        child = {
            "model_name": agent.model_name,
            "model_kind": "continuous",
            "initialparameterization": block,
            "model_parameters": {
                **spec.get("model_parameters", {}),
                "nr_agents": 1,
                "random_seed": agent.random_seed,
            },
        }
        programs.append(
            base64.b64encode(
                _generate_continuous_code(
                    child, agent.model_name, "continuous"
                ).encode()
            ).decode()
        )
    encoded = base64.b64encode(json.dumps(record).encode()).decode()
    return f'''#!/usr/bin/env julia
# Native RxInfer independent continuous agents: genuine @model + infer() per agent.
using JSON, Base64
const AGENTS = {json.dumps(names)}
const PROGRAMS = {json.dumps(programs)}
const RECORD = JSON.parse(String(base64decode("{encoded}")))
const RESULT_KEYS = {json.dumps(keys)}
function main()
    agent_results = Dict{{String,Any}}()
    for (index, name) in enumerate(AGENTS)
        native = Module(Symbol("GnnContinuousAgent", index))
        Base.include_string(native, String(base64decode(PROGRAMS[index])), name * ".jl")
        agent_results[name] = Base.invokelatest(getproperty(native, :run_simulation))
    end
    results = deepcopy(RECORD)
    for key in RESULT_KEYS
        results[key * "_by_agent"] = Dict(name => get(agent_results[name], key, []) for name in AGENTS)
    end
    results["validation"] = Dict("all_valid" => all(agent_results[name]["validation"]["all_valid"] for name in AGENTS), "per_agent" => Dict(name => agent_results[name]["validation"] for name in AGENTS))
    results["runtime_metadata"]["agent_runtime"] = Dict(name => agent_results[name]["runtime_metadata"] for name in AGENTS)
    open("simulation_results.json", "w") do io
        JSON.print(io, results, 2)
    end
    return results["validation"]["all_valid"] ? 0 : 1
end
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''
