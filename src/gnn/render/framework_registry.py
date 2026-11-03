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
            # The DisCoPy renderer draws the specification's declared
            # categorical connection graph as one box per Connections entry.
            # It refuses a spec without explicit wiring rather than
            # synthesizing a discrete circuit, and it has no linear-Gaussian
            # diagram semantics, so continuous models are reported unsupported.
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
            "description": "Native finite-model conditional Bayesian filtering",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Source CPT Filtering",
                "Exact Inference",
                "Synthetic Parameter Learning Demo",
            ],
            "function": "generate_bnlearn_code",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": False,
            "supports_multi_factor": False,
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
            "supported_features": [
                "Kalman Filtering",
                "Finite Policy EFE Search",
                "Source-Declared Finite Controls",
            ],
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


#: Closed vocabulary for the render dimensions every frame declares in
#: :data:`RENDER_CAPABILITY_DECLARATIONS`.  A dimension's tuple lists every
#: value a frame may declare; a frame declares the subset it supports and the
#: renderer refuses a model that requires an undeclared value (absence is
#: refusal — fail closed: an undeclared value is refused, never rendered).
#:
#: Dimension meanings:
#:   ``continuous_composition`` -- which continuous families the frame renders
#:     whole: ``flat`` (one linear-Gaussian F/H/Q/R block), ``factored``
#:     (per-factor ``*_fN`` blocks), ``independent_agents`` (per-agent
#:     ``*_agentN`` blocks with declared independent coupling).
#:   ``dynamics`` -- ``stationary``/``nonstationary`` (time-indexed ``B_t`` or
#:     regime-switched ``B_regime`` transitions with a step schedule).
#:   ``planning`` -- ``none`` (no planning semantics; authored horizons are not
#:     gated) or ``one_step_only`` (admits one-step declarations and refuses
#:     deeper or conflicting authored horizons rather than reducing them).
#:   ``mixed_family_composition`` -- ``whole`` when the frame renders a spec
#:     that declares more than one render family (e.g. continuous x hybrid)
#:     as one model; empty means the spec is refused instead of silently
#:     rendering one family with the others dropped.
#:   ``connection_graph`` -- ``explicit`` when the frame builds its artifact
#:     from the specification's declared ``Connections`` and refuses a spec
#:     with no wiring; empty when the frame does not consume the connection
#:     graph. DisCoPy is the explicit-wiring frame: its boxes are the declared
#:     connections, not a fixed circuit.
#:
#: ``execution_contracts`` remains a separate top-level registry key consumed
#: by :func:`gnn.render.execution_contracts.unsupported_contract`; the
#: discrete/continuous split is declared by ``supports_continuous`` and
#: ``continuous_only``.  Together these are the machine-checked render
#: capabilities; prose ``supported_features`` is documentation, not a gate.
RENDER_DIMENSION_VALUES: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "continuous_composition": ("flat", "factored", "independent_agents"),
        "dynamics": ("stationary", "nonstationary"),
        "planning": ("none", "one_step_only"),
        "mixed_family_composition": ("whole",),
        "connection_graph": ("explicit",),
    }
)

#: Per-frame render-capability declarations.  Values reproduce the enforced
#: render behaviour exactly; ``tests/render/`` conformance checks fail when a
#: declaration and the renderer's decisions drift.  A frame absent from this
#: table declares nothing and therefore supports nothing here (fail closed).
RENDER_CAPABILITY_DECLARATIONS: Mapping[str, Mapping[str, Tuple[str, ...]]] = (
    MappingProxyType(
        {
            "pymdp": {
                "continuous_composition": (),
                "dynamics": ("stationary", "nonstationary"),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "rxinfer": {
                "continuous_composition": ("flat", "independent_agents"),
                "dynamics": ("stationary",),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "activeinference_jl": {
                "continuous_composition": (),
                "dynamics": ("stationary",),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "jax": {
                "continuous_composition": (
                    "flat",
                    "factored",
                    "independent_agents",
                ),
                "dynamics": ("stationary",),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "discopy": {
                "continuous_composition": (),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": ("explicit",),
            },
            "pytorch": {
                "continuous_composition": ("flat",),
                "dynamics": ("stationary",),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "numpyro": {
                "continuous_composition": ("flat",),
                "dynamics": ("stationary",),
                "planning": ("one_step_only",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "stan": {
                "continuous_composition": ("flat",),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "bnlearn": {
                "continuous_composition": (),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "cpomdp": {
                "continuous_composition": ("flat",),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "thrml": {
                "continuous_composition": (),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
            "ngclearn": {
                "continuous_composition": ("flat",),
                "dynamics": ("stationary",),
                "planning": ("none",),
                "mixed_family_composition": (),
                "connection_graph": (),
            },
        }
    )
)


def get_render_capabilities(framework: str) -> Mapping[str, Tuple[str, ...]]:
    """Return the frame's declared render capabilities.

    An undeclared frame returns an empty mapping: unknown frames support
    nothing by declaration (fail closed).
    """
    return RENDER_CAPABILITY_DECLARATIONS.get(framework, {})


def supports_render_dimension(framework: str, dimension: str, value: str) -> bool:
    """Return whether *framework* declares support for (*dimension*, *value*).

    Fail closed: an undeclared frame, dimension or value is not supported.
    """
    return value in get_render_capabilities(framework).get(dimension, ())


def get_planning_admission_frameworks() -> list[str]:
    """Return frames whose declared planning semantics gate authored horizons."""
    return [
        name
        for name in FRAMEWORK_REGISTRY
        if supports_render_dimension(name, "planning", "one_step_only")
    ]


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
            **{
                key: deepcopy(value)
                for key, value in spec.items()
                if key in renderer_fields
            },
            "render_capabilities": deepcopy(
                RENDER_CAPABILITY_DECLARATIONS.get(name, {})
            ),
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
            "render_capabilities": deepcopy(
                RENDER_CAPABILITY_DECLARATIONS.get(name, {})
            ),
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
