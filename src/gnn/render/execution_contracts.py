"""Versioned, structure-preserving scientific execution contracts.

These contracts are opt-in semantic declarations, independent of model names.
A backend must implement the whole contract; ordinary flat POMDP adapters may
not silently consume one. Values are validated, never repaired or normalized.
"""

from typing import Any

import numpy as np

CONTRACTS = ("block_reset_v1", "timed_soft_controller_v1", "episodic_contingent_v1")


def execution_contract(spec: object) -> str | None:
    """Return an explicitly declared execution contract, including unknown versions."""
    if not isinstance(spec, dict):
        return None
    parameters = spec.get("model_parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("model_parameters must be a mapping")
    value = parameters.get("execution_contract")
    return str(value) if value is not None else None


def unsupported_contract(spec: dict[str, Any], framework: str) -> str | None:
    """Fail closed for unknown versions and backends lacking full semantics."""
    from gnn.render.framework_registry import FRAMEWORK_REGISTRY

    contract = execution_contract(spec)
    supported = FRAMEWORK_REGISTRY.get(framework, {}).get("execution_contracts", ())
    if contract is not None and (
        contract not in CONTRACTS or contract not in supported
    ):
        return f"unsupported-execution-contract: {framework} cannot execute {contract}"
    return None


def validate_run_parameters(contract: str, params: dict[str, Any]) -> None:
    """Reject lossy run counts/seeds and incomplete blocks without coercion."""
    for key in ("seed", "num_timesteps", "num_fast_transitions"):
        if key in params:
            value = params[key]
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be an integer in its valid range")
    if contract == "block_reset_v1" and params.get("num_timesteps", 20) % 5:
        raise ValueError("block_reset_v1 requires complete five-observation blocks")


def contract_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize component matrices without forming a flat world."""
    from gnn.render.pomdp_contract import DECLARED_PROBABILITY_MASS_ATOL

    contract = execution_contract(spec)
    if contract not in CONTRACTS:
        raise ValueError(f"unsupported-execution-contract: {contract}")
    params = dict(spec.get("model_parameters") or {})
    validate_run_parameters(contract, params)
    if params.get("b_tensor_order") != "next_state_previous_state_action":
        raise ValueError(
            "Execution contracts require explicit canonical b_tensor_order"
        )
    raw = {
        **(spec.get("initialparameterization") or {}),
        **((spec.get("structured_pomdp") or {}).get("matrices") or {}),
    }
    matrices: dict[str, Any] = {}

    def array(name: str, probability: bool = False, vector: bool = False) -> np.ndarray:
        if name not in raw:
            raise ValueError(f"Missing execution contract parameter {name}")
        value = np.asarray(raw[name], dtype=float)
        if vector:
            value = value.reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        if probability and (
            np.any(value < 0)
            or not np.allclose(
                value.sum(axis=0), 1, rtol=0, atol=DECLARED_PROBABILITY_MASS_ATOL
            )
        ):
            raise ValueError(f"{name} probability mass must be one and nonnegative")
        matrices[name] = value.tolist()
        return value

    def level(suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = array(f"A_{suffix}", True)
        b = array(f"B_{suffix}", True)
        c = array(f"C_{suffix}", vector=True)
        d = array(f"D_{suffix}", True, True)
        if b.ndim == 2:
            b = b[:, :, None]
            matrices[f"B_{suffix}"] = b.tolist()
        if (
            a.ndim != 2
            or b.ndim != 3
            or b.shape[:2] != (d.size, d.size)
            or a.shape != (c.size, d.size)
            or b.shape[2] < 1
        ):
            raise ValueError(f"Inconsistent dimensions for {suffix}")
        return a, b, c, d

    if contract == "block_reset_v1":
        a, b, c, d = level("level1")
        a2, b2, c2, d2 = level("level2")
        if a2.shape[0] != d.size or b2.shape[2] != 1:
            raise ValueError(
                "Block reset requires a lower-state prior map and passive context"
            )
        if not np.allclose(d, a2 @ d2, atol=DECLARED_PROBABILITY_MASS_ATOL, rtol=0):
            raise ValueError("D_level1 must be the derived context mixture")
        if params.get("timescale_ratio") != 5:
            raise ValueError("block_reset_v1 requires five observations per block")
    elif contract == "timed_soft_controller_v1":
        levels = [level(f"level{k}") for k in range(3)]
        a0, _, _, d0 = levels[0]
        a1, _, _, d1 = levels[1]
        a2, _, _, d2 = levels[2]
        m10, m21, n21 = array("M10"), array("M21"), array("N21", True)
        if (
            a1.shape[0] != d0.size
            or a2.shape[0] != d1.size
            or m10.shape != (a0.shape[0], d1.size)
            or m21.shape != (a1.shape[0], d2.size)
            or n21.shape != (d1.size, d2.size)
        ):
            raise ValueError(
                "Inconsistent cross-level message/preference map dimensions"
            )
        if not np.allclose(d1, n21 @ d2, atol=DECLARED_PROBABILITY_MASS_ATOL, rtol=0):
            raise ValueError("D_level1 must equal N21 D_level2")
        if any(
            params.get(k) != 10 for k in ("timescale_ratio_1_0", "timescale_ratio_2_1")
        ):
            raise ValueError(
                "timed_soft_controller_v1 requires 10/100 transition clocks"
            )
        if np.any(a1 <= 0) or np.any(a2 <= 0):
            raise ValueError("Soft-message likelihoods must be strictly positive")
    else:
        al = array("A_loc", True)
        ar = array("A_rew", True)
        bl = array("B_loc", True)
        bc = array("B_ctx", True)
        dl = array("D_loc", True, True)
        dc = array("D_ctx", True, True)
        cl = array("C_loc", vector=True)
        cr = array("C_rew", vector=True)
        terminal = params.get("terminal_locations")
        if isinstance(terminal, str):
            import json

            terminal = json.loads(terminal)
        params["terminal_locations"] = terminal
        if (
            not isinstance(terminal, (list, tuple))
            or not terminal
            or any(type(x) is not int or x < 0 or x >= dl.size for x in terminal)
        ):
            raise ValueError(
                "terminal_locations must declare valid absorbing location indices"
            )
        if (
            al.shape != (dl.size, dl.size)
            or not np.array_equal(al, np.eye(dl.size))
            or ar.shape != (cr.size, dl.size, dc.size)
            or bl.ndim != 3
            or bl.shape[:2] != (dl.size, dl.size)
            or cl.shape != (dl.size,)
        ):
            raise ValueError(
                "Episodic contract requires observed locations and factored reward likelihood"
            )
        if bc.ndim == 3 and bc.shape[2] == 1:
            bc = bc[:, :, 0]
        if not np.array_equal(bc, np.eye(dc.size)):
            raise ValueError("Episodic context must remain fixed")
        if np.any(cl != 0):
            raise ValueError(
                "episodic_contingent_v1 scores C_rew only; C_loc must be zero"
            )
        for loc in terminal:
            if not np.array_equal(
                bl[:, loc, :],
                np.repeat(np.eye(dl.size)[:, loc, None], bl.shape[2], axis=1),
            ):
                raise ValueError(
                    "Terminal locations must be absorbing for every action"
                )
        if params.get("policy_horizon") != 2:
            raise ValueError("episodic_contingent_v1 requires two action transitions")
    return {"execution_contract": contract, "matrices": matrices, "parameters": params}


def generate_contract_script(
    spec: dict[str, Any], options: dict[str, Any] | None
) -> str:
    """Emit a thin runnable JAX driver with source-bound validated parameters."""
    import json

    payload = contract_payload(spec)
    opts = options or {}
    for key in ("seed", "num_fast_transitions", "num_timesteps"):
        if key in opts:
            payload["parameters"][key] = opts[key]
    validate_run_parameters(payload["execution_contract"], payload["parameters"])
    return f"""# Generated structure-preserving JAX execution contract.
import json
import os
import sys
from pathlib import Path
if os.environ.get("GNN_PROJECT_ROOT"):
    sys.path.insert(0, str(Path(os.environ["GNN_PROJECT_ROOT"]) / "src"))
from gnn.execute.jax.semantic_runner import run_contract
PAYLOAD = json.loads({json.dumps(payload)!r})
if __name__ == "__main__":
    result = run_contract(PAYLOAD)
    out = Path(os.environ.get("GNN_OUTPUT_DIR", os.environ.get("JAX_OUTPUT_DIR", ".")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "simulation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
"""
