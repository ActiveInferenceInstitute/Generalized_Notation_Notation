#!/usr/bin/env python3
"""Compatibility-validation mixin for the POMDP render processor."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from gnn.render.pomdp_contract import (
    ModelKind,
    detect_pomdp_space_model_kind,
    detect_pomdp_space_model_kinds,
    unsupported_composition_reason,
    unsupported_nonstationary_reason,
)

from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _CompatibilityValidationMixin(_POMDPProcessorSupportMixin):
    def _validate_pomdp_framework_compatibility(
        self, pomdp_space: "POMDPStateSpace", framework: str
    ) -> Dict[str, Any]:
        """
        Validate that POMDP is compatible with target framework.

        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name

        Returns:
            Validation result dictionary
        """
        config = self.framework_configs[framework]
        warnings: list[Any] = []

        # A composed spec declares more than one render family (e.g. a
        # linear-Gaussian F/H/Q/R block alongside nr_agents > 1). Rendering
        # the single-winner kind would silently drop the other family, so
        # every framework refuses a composed continuous spec with an explicit
        # unsupported-composition receipt — the same unsupported accounting
        # structural wrappers get, never a silent wrong-family render.
        kinds = detect_pomdp_space_model_kinds(pomdp_space)
        factored_continuous = kinds == frozenset(
            {ModelKind.FACTORED, ModelKind.CONTINUOUS}
        )
        if kinds == frozenset({ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}):
            from gnn.render.multi_agent_continuous import (
                extract_multi_agent_continuous_specs,
            )

            try:
                extract_multi_agent_continuous_specs(
                    {
                        "initialparameterization": pomdp_space.initial_parameterization,
                        "model_parameters": pomdp_space.model_parameters,
                    }
                )
            except ValueError as exc:
                if not str(exc).startswith("unsupported-composition:"):
                    raise
                return {
                    "compatible": False,
                    "unsupported": True,
                    "reason": str(exc),
                    "warnings": warnings,
                }
            if framework in {"jax", "rxinfer"}:
                return {"compatible": True, "reason": None, "warnings": warnings}
            return {
                "compatible": False,
                "unsupported": True,
                "reason": f"unsupported-composition: {framework} does not render independent continuous agents",
                "warnings": warnings,
            }
        if ModelKind.CONTINUOUS in kinds and len(kinds) > 1 and not factored_continuous:
            return {
                "compatible": False,
                "unsupported": True,
                "reason": unsupported_composition_reason(kinds),
                "warnings": warnings,
            }
        if factored_continuous:
            # Per-factor LGSSM composition: JAX renders it through the
            # factored per-factor path; every other framework is refused
            # rather than silently rendered as one flat LGSSM.
            if framework != "jax":
                return {
                    "compatible": False,
                    "unsupported": True,
                    "reason": (
                        "unsupported-factored-continuous: "
                        f"{config.get('name', framework)} renders the flat "
                        "linear-Gaussian family only; per-factor compositions "
                        "are refused rather than silently rendered flat"
                    ),
                    "warnings": warnings,
                }
            return {"compatible": True, "reason": None, "warnings": warnings}

        # A nonstationary spec declares time-indexed (B_t) or regime-switched
        # (B_regime + b_regime_schedule) transitions. Only pymdp executes
        # the switching semantics (per-step Agent rebuild); every other
        # framework renders one static transition tensor and is refused
        # with an explicit receipt instead of silently rendering static
        # dynamics.
        if ModelKind.NONSTATIONARY in kinds and framework != "pymdp":
            return {
                "compatible": False,
                "unsupported": True,
                "reason": unsupported_nonstationary_reason(kinds),
                "warnings": warnings,
            }

        if getattr(pomdp_space, "model_kind", "discrete") == "continuous":
            if not config.get("supports_continuous", False):
                return {
                    "compatible": False,
                    "unsupported": True,
                    "reason": (
                        f"continuous-state model: {config.get('name', framework)} "
                        "supports discrete POMDPs only"
                    ),
                    "warnings": warnings,
                }
            matrices = getattr(pomdp_space, "matrices", None) or {}
            missing = [
                key
                for key in ("F", "H", "Q", "R", "prior_mean", "prior_cov")
                if key not in matrices
            ]
            if missing:
                return {
                    "compatible": False,
                    "reason": f"Missing continuous parameters: {missing}",
                    "warnings": warnings,
                }
            return {"compatible": True, "reason": None, "warnings": warnings}

        # Structural wrapper specs (e.g. Markov-blanket patterns) declare
        # boundary structure only — no discrete A/B/C/D[/E] and no continuous
        # F/H/Q/R parameterization. They are render-only / informational:
        # reported unsupported like a categorical backend facing a continuous
        # model, never forced through a discrete render that would fail on
        # missing matrices.
        if detect_pomdp_space_model_kind(pomdp_space) is ModelKind.STRUCTURAL:
            return {
                "compatible": False,
                "unsupported": True,
                "reason": (
                    "structural-spec: no renderable form — declares boundary "
                    "structure only (no discrete A/B/C/D[/E] and no "
                    "continuous F/H/Q/R parameterization)"
                ),
                "warnings": warnings,
            }

        # Continuous-only backends (registry ``continuous_only``) declare no
        # discrete A/B/C/D[/E] machinery: a discrete POMDP is first-class
        # unsupported for them — never recorded as a failed render, and never
        # forced through a renderer that would refuse it anyway.
        if config.get("continuous_only", False):
            return {
                "compatible": False,
                "unsupported": True,
                "reason": (
                    f"discrete POMDP: {config.get('name', framework)} "
                    "supports continuous linear-Gaussian models only"
                ),
                "warnings": warnings,
            }

        # Check required matrices are present, allowing factored matrices when
        # they can be composed into a canonical execution contract.
        missing_matrices: list[Any] = []
        required_matrices = cast(list[str], config["requires_matrices"])
        for required_matrix in required_matrices:
            if not self._has_matrix_or_factored_matrix(pomdp_space, required_matrix):
                missing_matrices.append(required_matrix)

        if missing_matrices:
            return {
                "compatible": False,
                "reason": f"Missing required matrices: {missing_matrices}",
                "warnings": warnings,
            }

        if framework == "pymdp" and ModelKind.NONSTATIONARY in kinds:
            if not self._nonstationary_b_declared(pomdp_space):
                # The time variation sits in A/C/D/E parameterization, not in
                # a B_t/B_regime transition: pymdp has no executor route for
                # it and would roll the model out as a static B.
                return {
                    "compatible": False,
                    "unsupported": True,
                    "reason": unsupported_nonstationary_reason(kinds),
                    "warnings": warnings,
                }
        elif framework == "pymdp":
            try:
                self._build_canonical_initialparameterization(pomdp_space)
            except ValueError as exc:
                return {"compatible": False, "reason": str(exc), "warnings": warnings}

        # Framework-specific checks
        if framework == "rxinfer" and not config["supports_multi_modality"]:
            if pomdp_space.num_observations > 1:  # This is a simplistic check
                warnings.append(f"{framework} has limited multi-modality support")

        # Check dimension limits
        max_reasonable_dim = 100  # Reasonable limit for most frameworks
        if (
            pomdp_space.num_states > max_reasonable_dim
            or pomdp_space.num_observations > max_reasonable_dim
            or pomdp_space.num_actions > max_reasonable_dim
        ):
            warnings.append("Large state spaces may cause performance issues")

        return {"compatible": True, "reason": None, "warnings": warnings}

    def _has_matrix_or_factored_matrix(
        self, pomdp_space: "POMDPStateSpace", matrix_name: str
    ) -> bool:
        """Return whether matrix or factored matrix."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        if matrix_name in matrices:
            return True
        if matrix_name == "B" and self._time_indexed_transition_key(matrices):
            return True
        return any(key.startswith(f"{matrix_name}_") for key in matrices)

    def _time_indexed_transition_key(self, matrices: Dict[str, Any]) -> Optional[str]:
        """Return the single supported time-indexed transition tensor key."""
        if "B_t" in matrices:
            return "B_t"
        time_keys = sorted(
            key for key in matrices if key.startswith("B_t") and key[3:].isdigit()
        )
        if len(time_keys) == 1:
            return time_keys[0]
        return None

    def _nonstationary_b_declared(self, pomdp_space: "POMDPStateSpace") -> bool:
        """Return whether the space declares a pymdp-executable nonstationary
        transition: a ``B_t`` / ``B_regime`` key (optional numeric suffix)."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        return any(
            re.match(r"^B_(t|regime)\d*$", str(key), re.IGNORECASE) for key in matrices
        )

    def _validate_state_spaces_in_spec(
        self, gnn_spec: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """
        Validate that state spaces are present in GNN spec.

        Args:
            gnn_spec: GNN specification dictionary
            framework: Target framework name

        Returns:
            Validation result dictionary
        """
        warnings: list[Any] = []
        initial_params = gnn_spec.get("initialparameterization", {})
        config = self.framework_configs[framework]

        if gnn_spec.get("model_kind") == "continuous":
            # Linear-Gaussian contract: compatibility (incl. the F/H/Q/R
            # presence check) was already decided per framework upstream.
            return {"valid": True, "critical": False, "warnings": warnings}

        if framework in {"rxinfer", "activeinference_jl"}:
            from gnn.render.multi_agent_common import (
                has_native_multi_agent_structure,
                validate_native_agent_groups,
            )

            if has_native_multi_agent_structure(gnn_spec):
                try:
                    validate_native_agent_groups(gnn_spec)
                except (ValueError, TypeError) as exc:
                    return {
                        "valid": False,
                        "critical": True,
                        "reason": str(exc),
                        "warnings": warnings,
                    }
                return {"valid": True, "critical": False, "warnings": warnings}

        # Check required matrices
        missing_required: list[Any] = []
        required_matrices = cast(list[str], config["requires_matrices"])
        for required_matrix in required_matrices:
            if required_matrix in initial_params:
                continue
            if required_matrix == "B" and any(
                re.fullmatch(r"B_(t|regime)\d*", key, re.IGNORECASE)
                for key in initial_params
            ):
                # A NONSTATIONARY spec's time-varying (B_t) or regime-switched
                # (B_regime) transition tensor satisfies the required B: pymdp
                # is the only framework that reaches this validator for such
                # specs (others are refused upstream), and its raw-passthrough
                # contract consumes B_t/B_regime directly.
                continue
            missing_required.append(required_matrix)

        if missing_required:
            return {
                "valid": False,
                "critical": True,
                "reason": f"Missing required matrices: {missing_required}",
                "warnings": warnings,
            }

        # Check optional matrices
        optional_matrices = cast(list[str], config.get("optional_matrices", []))
        for optional_matrix in optional_matrices:
            if optional_matrix not in initial_params:
                warnings.append(f"Optional matrix {optional_matrix} not found")

        return {"valid": True, "critical": False, "warnings": warnings}

    def _validate_state_spaces_in_script(
        self, script_path: Path, gnn_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that state spaces are present in generated script.

        Args:
            script_path: Path to generated script
            gnn_spec: Original GNN specification

        Returns:
            Validation result dictionary
        """
        warnings: list[Any] = []

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()

            initial_params = gnn_spec.get("initialparameterization", {})

            # Check if matrices are referenced in script
            for matrix_name in ["A", "B", "C", "D", "E"]:
                if matrix_name in initial_params:
                    # Check if matrix is present in script (as variable or in data structure)
                    if (
                        matrix_name not in script_content
                        and f'"{matrix_name}"' not in script_content
                    ):
                        warnings.append(
                            f"Matrix {matrix_name} may not be properly injected into script"
                        )

            return {"valid": len(warnings) == 0, "warnings": warnings}
        except Exception as e:
            return {"valid": False, "warnings": [f"Failed to validate script: {e}"]}
