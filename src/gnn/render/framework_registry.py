"""Canonical render framework inventory."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from gnn.frameworks import LITE_FRAMEWORKS

FRAMEWORK_REGISTRY: Mapping[str, Dict[str, Any]] = MappingProxyType(
    {
        "pymdp": {
            "name": "PyMDP",
            "description": "Python Markov Decision Process library for Active Inference",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "POMDP",
                "MDP",
                "Belief State Updates",
                "Active Inference",
            ],
            "function": "render_gnn_to_pymdp",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": False,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "rxinfer": {
            "name": "RxInfer.jl",
            "description": "Julia reactive message passing inference engine",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": [
                "Message Passing",
                "Probabilistic Programming",
                "Bayesian Inference",
            ],
            "function": "render_gnn_to_rxinfer",
            "output_format": "julia",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": False,
            "supports_multi_factor": False,
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "activeinference_jl": {
            "name": "ActiveInference.jl",
            "description": "Julia Active Inference library",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": [
                "Free Energy Minimization",
                "Active Inference",
                "POMDP",
            ],
            "function": "render_gnn_to_activeinference_jl",
            "output_format": "julia",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": False,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "jax": {
            "execution_contracts": [
                "block_reset_v1",
                "timed_soft_controller_v1",
                "episodic_contingent_v1",
            ],
            "name": "JAX",
            "description": "High-performance numerical computing with automatic differentiation",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "GPU Acceleration",
                "Automatic Differentiation",
                "JIT Compilation",
            ],
            "function": "render_gnn_to_jax",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "discopy": {
            "name": "DisCoPy",
            "description": "Python library for computing with string diagrams",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Categorical Diagrams",
                "String Diagrams",
                "Compositional Models",
            ],
            "function": "render_gnn_to_discopy",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            # The DisCoPy translator draws categorical POMDP string diagrams
            # (A/B/C/D/E boxes over discrete state counts); it has no
            # linear-Gaussian diagram semantics, so continuous models are
            # reported unsupported rather than drawn as a discrete stand-in.
            "supports_execution": True,
            "supports_continuous": False,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "pytorch": {
            "name": "PyTorch",
            "description": "PyTorch tensor backend for Active Inference-style simulation",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Tensor Simulation", "POMDP", "Neural Integration"],
            "function": "render_gnn_to_pytorch",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            # Rendering is codegen-only (torch is imported by the emitted
            # script, not by the renderer). torch>=2.13.0 — the ``torch``
            # extra — resolves GHSA-rrmf-rvhw-rf47, lifting the previous
            # exclusion; Step 12 keeps its own dynamic import gate.
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "numpyro": {
            "name": "NumPyro",
            "description": "NumPyro probabilistic programming backend",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Probabilistic Programming", "POMDP", "JAX Backend"],
            "function": "render_gnn_to_numpyro",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "stan": {
            "name": "Stan",
            "description": "Stan probabilistic programming model generation",
            "language": "Stan",
            "file_extension": ".stan",
            "supported_features": ["Probabilistic Programming", "Variable Graphs"],
            "function": "render_stan",
            "output_format": "stan",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "bnlearn": {
            "name": "bnlearn",
            "description": "Python package for learning Bayesian network structure",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Structure Learning",
                "Parameter Learning",
                "Exact Inference",
                "Causal Discovery",
            ],
            "function": "generate_bnlearn_code",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            # Executed by `src/gnn/execute/bnlearn/` (Step 12 script path,
            # `BNLEARN_OUTPUT_DIR`); scripts skip when the `bnlearn` extra is
            # absent (`gnn.utils.runtime_safety.framework_availability`).
            "supports_execution": True,
            "supports_continuous": False,
            "continuous_only": False,
            "unavailable_reason": None,
        },
        "cpomdp": {
            "name": "cpomdp",
            "description": "Experimental released continuous active-inference backend",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Kalman Filtering", "Finite Policy EFE Search"],
            "function": "render_gnn_to_cpomdp",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["F", "H", "Q", "R", "prior_mean", "prior_cov"],
            "supports_multi_modality": False,
            "supports_multi_factor": False,
            "available": True,
            "supports_execution": True,
            "supports_continuous": True,
            "continuous_only": True,
            "experimental": True,
            "unavailable_reason": None,
        },
        "thrml": {
            "name": "THRML",
            "description": "Experimental released categorical Gibbs sampling backend",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Conditional Gibbs Sampling",
                "Fixed-action Posterior Smoothing",
                "Independent Component Composition",
                "Bounded Factored and Observation Composition",
            ],
            "function": "render_gnn_to_thrml",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_execution": True,
            "supports_continuous": False,
            "continuous_only": False,
            "experimental": True,
            "unavailable_reason": None,
        },
        "ngclearn": {
            "name": "ngc-learn",
            "description": "ngc-learn predictive-processing neural simulation backend",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Predictive Processing",
                "Neuromorphic Simulation",
                "JAX Backend",
            ],
            "function": "render_gnn_to_ngclearn",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["F", "H", "Q", "R", "prior_mean", "prior_cov"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            # Codegen-only renderer (pytorch precedent): the generated script
            # imports ngclearn, never the renderer. Execution skips without
            # the `ngclearn` extra (`uv sync --extra ngclearn`, py3.12 marker;
            # `gnn.utils.runtime_safety.framework_availability`).
            "supports_execution": True,
            "supports_continuous": True,
            # Continuous-only backend: no discrete A/B/C/D[/E] machinery
            # exists in ngc-learn — discrete POMDPs are first-class
            # unsupported (see pomdp_processor compatibility validation).
            "continuous_only": True,
            "unavailable_reason": None,
        },
    }
)


def get_supported_frameworks() -> list[str]:
    """Return canonical supported framework names in render order."""
    return list(FRAMEWORK_REGISTRY.keys())


def get_default_frameworks() -> list[str]:
    """Return maintained targets; experimental backends require selection."""
    return [
        name
        for name, spec in FRAMEWORK_REGISTRY.items()
        if not spec.get("experimental", False)
    ]


def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Return renderer metadata without POMDP validation-only fields."""
    renderer_fields = {
        "name",
        "description",
        "language",
        "file_extension",
        "supported_features",
        "function",
        "output_format",
        "pomdp_compatible",
        "supports_execution",
        "experimental",
    }
    return {
        name: {
            key: deepcopy(value)
            for key, value in spec.items()
            if key in renderer_fields
        }
        for name, spec in FRAMEWORK_REGISTRY.items()
    }


def get_pomdp_framework_configs() -> Dict[str, Dict[str, Any]]:
    """Return POMDP processor configs derived from the canonical registry."""
    return {
        name: {
            "output_subdir": name,
            "file_extension": spec["file_extension"],
            "requires_matrices": deepcopy(spec["requires_matrices"]),
            "optional_matrices": deepcopy(spec["optional_matrices"]),
            "supports_multi_modality": bool(spec["supports_multi_modality"]),
            "supports_multi_factor": bool(spec["supports_multi_factor"]),
            "supports_execution": bool(spec.get("supports_execution", True)),
            "supports_continuous": bool(spec.get("supports_continuous", False)),
            "continuous_only": bool(spec.get("continuous_only", False)),
            "experimental": bool(spec.get("experimental", False)),
            "name": spec["name"],
        }
        for name, spec in FRAMEWORK_REGISTRY.items()
        if spec.get("pomdp_compatible", False)
    }


def get_framework_availability(framework: str) -> Tuple[bool, Optional[str]]:
    """Return (available, reason) for a framework from the canonical registry.

    Args:
        framework: Name of the framework (e.g. ``"bnlearn"``).

    Returns:
        Tuple of ``(available, reason)`` where *reason* is ``None`` when the
        framework is available, or a human-readable string explaining why it
        is intentionally unavailable.
    """
    spec = FRAMEWORK_REGISTRY.get(framework)
    if spec is None:
        return True, None  # unknown frameworks are assumed available
    available = spec.get("available", True)
    reason = spec.get("unavailable_reason")
    return bool(available), reason


def validate_framework_requested(framework: str) -> None:
    """Validate that a requested framework is available for rendering.

    Raises ``ValueError`` with a clear, actionable message when the framework
    is intentionally unavailable.  Does nothing when the framework is available
    or unknown (unknown frameworks are treated as available to avoid
    over-blocking external tools).

    Args:
        framework: Name of the framework to validate.

    Raises:
        ValueError: If the framework is intentionally marked unavailable in
            the canonical registry.
    """
    available, reason = get_framework_availability(framework)
    if not available and reason is not None:
        raise ValueError(f"Framework '{framework}' is not available: {reason}")


#: Frameworks served by the lightweight ``"lite"`` preset in
#: ``processor.process_render`` (no Julia toolchain, no GPU stack). Re-exported
#: from the canonical name enumeration so the two cannot drift.


def get_lite_frameworks() -> list[str]:
    """Return the canonical ``"lite"`` preset framework list."""
    return list(LITE_FRAMEWORKS)
