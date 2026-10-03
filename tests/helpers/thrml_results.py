"""Discrete result witnesses for contract tests, separate from native acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def categorical_result() -> dict[str, Any]:
    """An exact hand-counted sample witness and its declared replicate prediction."""
    return {
        "schema_version": 1,
        "framework": "thrml",
        "success": True,
        "model_kind": "discrete",
        "model_name": "two_state",
        "inference_mode": "gibbs_monte_carlo",
        "posterior_semantics": "fixed_actions_full_sequence_smoothing",
        "predictive_semantics": "replicate_observation_given_smoothed_state",
        "action_semantics": "transition_into_next_state",
        "sampling_dtype": "float64",
        "num_timesteps": 2,
        "parameters": {
            "A": [[0.7, 0.3], [0.3, 0.7]],
            "B": [[[0.8, 0.6], [0.3, 0.4]], [[0.2, 0.4], [0.7, 0.6]]],
            "C": [0.0, 0.0],
            "D": [0.6, 0.4],
            "E": [0.5, 0.5],
        },
        "sample_states": [[0, 0], [0, 1], [1, 1], [1, 1]],
        "beliefs": [[0.5, 0.5], [0.25, 0.75]],
        "predictive_observations": [[0.5, 0.5], [0.4, 0.6]],
        "observations": [0, 1],
        "transition_actions": [0],
        "sampling": {
            "num_timesteps": 2,
            "num_samples": 4,
            "burn_in": 5,
            "thin": 1,
            "seed": 0,
        },
        "validation": {"mapping_supported": True, "all_conditionals_valid": True},
        "dependencies": {"thrml": "0.1.4", "jax": "receipt-only-fixture"},
        "source_identity": {
            "model_id": "stable-test",
            "source_sha256": "a" * 64,
            "artifact_stem": "stable-test",
            "semantic_sha256": "c" * 64,
            "claim": "parameter/mapping identity; extraction and Monte Carlo accuracy require separate witnesses",
        },
        "runtime_metadata": {
            "execution_id": "test-receipt",
            "script_sha256": "b" * 64,
            "jax_enable_x64": True,
            "execution_plane": "protocol_fixture_only",
        },
    }


def write_protocol_script(path: Path, *, once: bool = False, mutate: str = "") -> Path:
    """Write a real child artifact protocol fixture; this is not THRML inference."""
    path.write_text(
        "import hashlib, json, os\nfrom pathlib import Path\n"
        + f"payload = json.loads({json.dumps(categorical_result())!r})\n"
        + "payload['runtime_metadata'].update(execution_id=os.environ['GNN_THRML_EXECUTION_ID'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())\n"
        + mutate
        + "\n"
        + "target = Path(os.environ['THRML_OUTPUT_DIR']) / 'simulation_data' / 'simulation_results.json'\n"
        + "target.parent.mkdir(parents=True, exist_ok=True)\n"
        + ("if not target.exists():\n    " if once else "")
        + "target.write_text(json.dumps(payload))\n"
        + "print('protocol-child-completed')\n"
    )
    return path
