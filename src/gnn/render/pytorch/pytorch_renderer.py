#!/usr/bin/env python3
"""
PyTorch Renderer for GNN Specifications

Renders GNN POMDP models to standalone PyTorch simulation scripts.
Generates the standard generative loop (same environment dynamics as PyMDP/JAX)
using torch.tensor operations.

@Web: https://pytorch.org/docs/stable/
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gnn.render.naming import atomic_write_text
from gnn.render.spec_matrices import extract_abcd_matrices, format_array_literal

logger = logging.getLogger(__name__)


def render_gnn_to_pytorch(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a PyTorch POMDP simulation script.

    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated PyTorch code.
        options: Optional rendering options.

    Returns:
        Tuple of (success: bool, message: str, artifact_paths: List[str])
    """
    try:
        model_name = gnn_spec.get("modelName", "pytorch_pomdp")
        logger.info(f"Rendering GNN spec to PyTorch: {model_name}")

        # Continuous-state (LGSSM) branch: no A/B/C/D exist on this path, so
        # it must never reach the discrete extractors below.
        if _is_continuous_spec(gnn_spec):
            from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

            if detect_model_kinds(gnn_spec) == frozenset(
                {ModelKind.FACTORED, ModelKind.CONTINUOUS}
            ):
                return (
                    False,
                    "unsupported-factored-continuous: pytorch renders the flat "
                    "linear-Gaussian family only; per-factor compositions are "
                    "refused rather than silently rendered flat",
                    [],
                )
            return _render_continuous(gnn_spec, Path(output_path), options, model_name)

        # Extract matrices
        provenance = deepcopy(gnn_spec.get("matrix_provenance", {}))
        A, B, C, D = _extract_matrices(gnn_spec, provenance=provenance)

        # Validate shapes
        from gnn.render.matrix_utils import validate_abcd_shapes

        valid, msg = validate_abcd_shapes(A, B, C, D)
        if not valid:
            logger.warning(f"Shape validation warning: {msg}")

        # Generate code
        code = _generate_pytorch_code(
            model_name,
            A,
            B,
            C,
            D,
            _extract_num_timesteps(gnn_spec, options),
            matrix_provenance=provenance,
        )

        # Write output
        output_path = Path(output_path)
        atomic_write_text(output_path, code)

        logger.info(f"✅ PyTorch script written to: {output_path}")
        return True, f"PyTorch script generated: {output_path}", [str(output_path)]

    except Exception as e:
        logger.error(f"❌ PyTorch rendering failed: {e}")
        return False, f"PyTorch rendering failed: {e}", []


def _is_continuous_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True for continuous-state (linear-Gaussian) specs."""
    from gnn.render.continuous_common import is_continuous_spec

    return is_continuous_spec(gnn_spec)


def _render_continuous(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]],
    model_name: str,
) -> Tuple[bool, str, List[str]]:
    """Emit the standalone PyTorch Kalman-filter LGSSM script."""
    from gnn.render.continuous_common import extract_continuous_spec
    from gnn.render.continuous_script import generate_continuous_script

    code = generate_continuous_script(extract_continuous_spec(gnn_spec), "pytorch")
    atomic_write_text(output_path, code)
    logger.info(f"✅ PyTorch continuous script written to: {output_path}")
    return (
        True,
        f"PyTorch continuous LGSSM script generated: {output_path}",
        [str(output_path)],
    )


def _extract_matrices(
    gnn_spec: Dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from GNN spec.

    Delegates to the shared :func:`render.spec_matrices.extract_abcd_matrices`.
    """
    return extract_abcd_matrices(gnn_spec, transformation_provenance=provenance)


def _extract_num_timesteps(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None
) -> int:
    """Extract simulation horizon from render options or parsed GNN metadata."""
    options = options or {}
    model_params = gnn_spec.get("model_parameters", {})
    init_params = gnn_spec.get("initialparameterization", {})
    return int(
        options.get(
            "num_timesteps",
            model_params.get("num_timesteps", init_params.get("num_timesteps", 10)),
        )
    )


def _format_tensor(arr: np.ndarray, indent: int = 4) -> str:
    """Format a numpy array as a torch.tensor() literal."""
    return format_array_literal(
        arr, prefix="torch.tensor", suffix=", dtype=torch.float64", indent=indent
    )


def _generate_pytorch_code(
    model_name: str,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    num_timesteps: int = 10,
    matrix_provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate standalone PyTorch POMDP simulation script."""
    provenance = matrix_provenance or {}
    num_states = A.shape[1] if A.ndim == 2 else 2
    num_obs = A.shape[0] if A.ndim == 2 else num_states
    num_actions = B.shape[2] if B.ndim == 3 else 2

    A_str = _format_tensor(A)
    B_str = _format_tensor(B if B.ndim == 2 else B[:, :, 0])
    C_str = _format_tensor(C)
    D_str = _format_tensor(D)

    B_full_init = ""
    if B.ndim == 3:
        slices: list[Any] = []
        for a in range(B.shape[2]):
            slices.append(
                f"    B_slices.append({_format_tensor(B[:, :, a], indent=4)})"
            )
        B_full_init = (
            "\n    B_slices = []\n"
            + "\n".join(slices)
            + "\n    B = torch.stack(B_slices, dim=2)"
        )
    else:
        B_full_init = (
            f"\n    B = {B_str}.unsqueeze(2).expand(-1, -1, {num_actions}).clone()"
        )

    code = f'''\
#!/usr/bin/env python3
"""
PyTorch POMDP Simulation: {model_name}

Auto-generated by GNN Pipeline — PyTorch renderer.
Implements the standard Active Inference generative loop using PyTorch tensors.
"""
import json
import os
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Install with: uv sync --extra ml-ai")
    sys.exit(1)

import numpy as np


def run_simulation():
    """Run POMDP simulation with PyTorch tensors."""
    start_time = time.time()

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

    # Initialize true state from prior
    true_state = torch.multinomial(D, 1).item()
    beliefs = D.clone()

    for t in range(T):
        # 1. Generate observation from true state
        obs_probs = A[:, true_state]
        obs_probs = obs_probs / obs_probs.sum()
        observation = torch.multinomial(obs_probs, 1).item()

        # 2. Belief update (Bayesian filtering)
        likelihood = A[observation, :]
        posterior = likelihood * beliefs
        posterior = posterior / (posterior.sum() + 1e-16)
        beliefs = posterior

        # 3. Compute Expected Free Energy (EFE) for each action
        efe = torch.zeros(num_actions, dtype=torch.float64)
        for a_idx in range(num_actions):
            B_a = B[:, :, a_idx]
            predicted_state = B_a @ beliefs
            predicted_obs = A @ predicted_state

            # Ambiguity (expected entropy of observations)
            log_A = torch.log(A + 1e-16)
            ambiguity = -(predicted_state * (A * log_A).sum(dim=0)).sum()

            # Risk (KL from preferred observations)
            log_pred = torch.log(predicted_obs + 1e-16)
            C_norm = torch.softmax(C, dim=0)
            log_pref = torch.log(C_norm + 1e-16)
            risk = (predicted_obs * (log_pred - log_pref)).sum()

            efe[a_idx] = ambiguity + risk

        # 4. Action selection (softmax over negative EFE)
        action_probs = torch.softmax(-efe, dim=0)
        action = torch.multinomial(action_probs, 1).item()

        # 5. State transition
        B_a = B[:, :, action]
        transition_probs = B_a[:, true_state]
        transition_probs = transition_probs / (transition_probs.sum() + 1e-16)
        true_state = torch.multinomial(transition_probs, 1).item()

        # Record history
        beliefs_history.append(beliefs.cpu().numpy().tolist())
        actions_history.append(action)
        observations_history.append(observation)
        efe_history.append(efe.cpu().numpy().tolist())

    elapsed = time.time() - start_time

    # --- Validation ---
    beliefs_arr = np.array(beliefs_history)
    validation = {{
        "beliefs_in_range": bool(np.all((beliefs_arr >= 0) & (beliefs_arr <= 1))),
        "beliefs_sum_to_one": bool(np.allclose(beliefs_arr.sum(axis=1), 1.0, atol=1e-6)),
        "actions_in_range": all(0 <= a < num_actions for a in actions_history),
        "all_valid": True
    }}
    validation["all_valid"] = all(validation.values())

    results = {{
        "schema_version": "pytorch_simulation_v1",
        "model_name": "{model_name}",
        "framework": "pytorch",
        "matrix_provenance": {provenance!r},
        "num_timesteps": T,
        "num_states": num_states,
        "num_observations": num_obs,
        "num_actions": num_actions,
        "beliefs": beliefs_history,
        "actions": actions_history,
        "observations": observations_history,
        "efe_history": efe_history,
        "expected_free_energy_convention": "pytorch renderer efe_history:"
            " EFE = ambiguity + risk where ambiguity = -(sum_s' q(s') *"
            " sum_o A[o,s'] log A[o,s']) (expected entropy of"
            " observations) and risk = sum_o q(o) (log q(o) - log"
            " softmax(C)[o]) (KL of predicted observations against"
            " softmax-normalised preferences). Closest to the Lean"
            " expectedFreeEnergy_eq_risk_add_ambiguity shape, but the"
            " preference reference is softmax(C), not the raw C vector,"
            " so values are not directly comparable without convention"
            " mapping (bridge finding O1).",
        "validation": validation,
        "execution_time_seconds": round(elapsed, 4),
        "torch_version": torch.__version__
    }}

    # Save results
    output_dir = Path(os.environ.get("PYTORCH_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"PyTorch POMDP simulation complete: {{T}} timesteps in {{elapsed:.3f}}s")
    print(f"Results saved to: {{output_file}}")
    print(f"Validation: {{validation}}")
    return results


if __name__ == "__main__":
    run_simulation()
'''
    return code
