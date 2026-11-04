"""Capability-contract regressions for the Python-render backends.

The audit that motivates these tests measured the full 38-source corpus against
the five Python-render frames (190 cells: 119 rendered, 71 refused, 0 failed)
and replayed every executed artifact's beliefs against the canonical matrices
(deviation <= 2.1e-07 for pymdp and numpyro over 45 discrete models).

These tests pin the two properties that make those numbers meaningful:

* a frame either renders a class or refuses it with a typed, actionable reason
  and no output files — it never fails silently and never emits a partial run;
* the two canonical matrices containers the renderers consume agree on the
  transition tensor: ``initialparameterization["B"]`` is
  ``(next_state, previous_state, action)`` per the spec's declared
  ``b_tensor_order`` and ``structured_pomdp.matrices["B"]`` is action-major.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_processor import POMDPRenderProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "input" / "gnn_files"

# (source, framework, reason-fragment) for refusals the audit verified as
# deliberate. Each fragment is the stable class marker the refusal carries.
REFUSALS = [
    # Capability: pymdp declares supports_continuous=false.
    ("continuous/stochastic_dynamics.md", "pymdp", "supports discrete POMDPs only"),
    # Capability: ngclearn declares continuous_only=true.
    ("discrete/actinf_pomdp_agent.md", "ngclearn", "supports continuous linear-Gaussian models only"),
    # Composition: independent continuous agents need the declared composition.
    ("continuous/multi_agent_lgssm.md", "pymdp", "unsupported-composition"),
    # Factored continuous LGSSM is rendered by the per-factor JAX path only.
    ("continuous/factored_continuous_lgssm.md", "numpyro", "unsupported-factored-continuous"),
    # Nonstationary dynamics need per-step transition switching (pymdp only).
    ("discrete/time_varying_dynamics.md", "jax", "unsupported-nonstationary"),
    ("discrete/regime_switched_dynamics.md", "numpyro", "unsupported-nonstationary"),
    # Execution contracts are declared by the JAX frame only.
    ("hierarchical/hierarchical_pomdp.md", "pytorch", "unsupported-execution-contract"),
    ("hierarchical/temporal_hierarchy.md", "pymdp", "unsupported-execution-contract"),
    ("discrete/tmaze_epistemic.md", "pytorch", "unsupported-execution-contract"),
    # One-step planning only: a multi-step authored horizon is refused.
    ("discrete/deep_planning_horizon.md", "pytorch", "unsupported-planning"),
]


@pytest.mark.integration
@pytest.mark.parametrize(("source", "framework", "fragment"), REFUSALS)
def test_declared_unsupported_cells_refuse_with_reason(
    source: str, framework: str, fragment: str, tmp_path: Path
) -> None:
    """Every declared-unsupported cell refuses, names its class and emits nothing."""
    pomdp = extract_pomdp_from_file(MODELS / source, strict_validation=True)
    assert pomdp is not None
    processor = POMDPRenderProcessor(tmp_path)
    result = processor._process_single_framework(pomdp, framework)
    assert result["unsupported"] is True, result
    assert result["status"] == "unsupported", result
    assert result["success"] is False, result
    assert fragment in result["message"], result
    assert result["output_files"] == [], result


@pytest.mark.integration
@pytest.mark.parametrize(
    ("source", "framework"),
    [
        ("basics/static_perception.md", "pymdp"),
        ("discrete/actinf_pomdp_agent.md", "jax"),
        ("discrete/actinf_pomdp_agent.md", "numpyro"),
        ("continuous/stochastic_dynamics.md", "jax"),
        ("continuous/stochastic_dynamics.md", "ngclearn"),
    ],
)
def test_declared_supported_cells_render_with_matrices(
    source: str, framework: str, tmp_path: Path
) -> None:
    """Every declared-supported class renders a non-empty artifact that carries its matrices."""
    pomdp = extract_pomdp_from_file(MODELS / source, strict_validation=True)
    assert pomdp is not None
    processor = POMDPRenderProcessor(tmp_path)
    result = processor._process_single_framework(pomdp, framework)
    assert result["success"] is True, result
    assert result.get("unsupported") is None, result
    output_files = result.get("output_files") or []
    assert output_files, result
    emitted = Path(result["output_files"][0]).read_text()
    assert len(emitted) > 500, result
    compile(emitted, str(result["output_files"][0]), "exec")
    # No placeholder values may reach an emitted artifact.
    for token in ("TODO", "FIXME", "PLACEHOLDER", "NotImplementedError("):
        assert token not in emitted, f"{token} in {framework} artifact for {source}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "source",
    [
        "discrete/two_state_bistable.md",
        "discrete/actinf_pomdp_agent.md",
        "discrete/multi_armed_bandit.md",
    ],
)
def test_canonical_b_containers_agree_on_the_transition_tensor(source: str, tmp_path: Path) -> None:
    """The declared-order container and the action-major container hold the same slices."""
    pomdp = extract_pomdp_from_file(MODELS / source, strict_validation=True)
    assert pomdp is not None
    spec = POMDPRenderProcessor(tmp_path)._pomdp_to_gnn_spec(pomdp)
    declared_order = spec["model_parameters"]["b_tensor_order"]
    assert declared_order == "next_state_previous_state_action"

    declared = np.asarray(spec["initialparameterization"]["B"], dtype=float)
    structured = np.asarray(spec["structured_pomdp"]["matrices"]["B"], dtype=float)
    assert declared.ndim == 3 and structured.ndim == 3
    assert declared.shape == structured.shape
    for action in range(declared.shape[2]):
        assert np.allclose(declared[:, :, action], structured[action]), action
