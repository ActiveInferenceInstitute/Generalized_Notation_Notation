#!/usr/bin/env python3
"""Spec-generation mixin for the POMDP render processor."""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from gnn.render.pomdp_contract import ModelKind, detect_pomdp_space_model_kind

from ._routes import _continuous_shape
from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _SpecGenerationMixin(_POMDPProcessorSupportMixin):
    def _pomdp_to_gnn_spec(
        self, pomdp_space: "POMDPStateSpace", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert POMDP state space to GNN spec format expected by renderers.

        Args:
            pomdp_space: POMDP state space data
            **kwargs: Additional options like timesteps

        Returns:
            GNN specification dictionary
        """
        # Extract optional config params
        timesteps = kwargs.get("timesteps")
        if timesteps is None and hasattr(pomdp_space, "num_timesteps"):
            timesteps = pomdp_space.num_timesteps

        sim_params = kwargs.get("simulation_params", "{}")
        try:
            parsed_sim_params = (
                json.loads(sim_params) if isinstance(sim_params, str) else sim_params
            )
        except json.JSONDecodeError:
            self.logger.warning(
                f"Invalid simulation_params string: {sim_params}. Using empty dict."
            )
            parsed_sim_params = {}

        if getattr(pomdp_space, "model_kind", "discrete") == "continuous":
            return self._continuous_pomdp_to_gnn_spec(
                pomdp_space, timesteps=timesteps, simulation_params=parsed_sim_params
            )

        if detect_pomdp_space_model_kind(pomdp_space) is ModelKind.STRUCTURAL:
            return self._structural_pomdp_to_gnn_spec(
                pomdp_space, timesteps=timesteps, simulation_params=parsed_sim_params
            )
        if detect_pomdp_space_model_kind(pomdp_space) is ModelKind.NONSTATIONARY:
            return self._nonstationary_pomdp_to_gnn_spec(
                pomdp_space, timesteps=timesteps, simulation_params=parsed_sim_params
            )

        if kwargs.get("preserve_discrete_structure"):
            # A component-aware backend owns its bounded composition. Never
            # allocate a dense joint tensor before that backend admits it.
            parameters = {
                **(pomdp_space.model_parameters or {}),
                "num_actions": pomdp_space.num_actions,
                "simulation_params": parsed_sim_params,
                **({"num_timesteps": timesteps} if timesteps is not None else {}),
            }
            return {
                "name": pomdp_space.model_name,
                "model_name": pomdp_space.model_name,
                "gnn_section": pomdp_space.gnn_section,
                "model_kind": getattr(pomdp_space, "model_kind", "discrete"),
                "model_parameters": parameters,
                "initialparameterization": pomdp_space.initial_parameterization or {},
                "structured_pomdp": {
                    "matrices": pomdp_space.matrices or {},
                    "matrix_provenance": pomdp_space.matrix_provenance or {},
                    "state_factors": pomdp_space.state_factors or [],
                    "observation_modalities": pomdp_space.observation_modalities or [],
                    "control_factors": pomdp_space.control_factors or [],
                    "adapter_notes": pomdp_space.adapter_notes or [],
                },
                "matrix_provenance": pomdp_space.matrix_provenance or {},
                "connections": [
                    {"source": edge[0], "relation": edge[1], "target": edge[2]}
                    for edge in (pomdp_space.connections or [])
                ],
                "variables": (pomdp_space.state_variables or [])
                + (pomdp_space.observation_variables or [])
                + (pomdp_space.action_variables or []),
                "canonical_pomdp_schema": "raw_discrete_components_v1",
            }

        from gnn.render.multi_agent_common import has_native_multi_agent_structure

        if kwargs.get("native_agents") and has_native_multi_agent_structure(
            {"structured_pomdp": {"matrices": pomdp_space.matrices}}
        ):
            return self._native_agent_pomdp_to_gnn_spec(
                pomdp_space, timesteps=timesteps, simulation_params=parsed_sim_params
            )

        initial_parameterization, matrix_provenance, canonical_model_parameters = (
            self._build_canonical_initialparameterization(pomdp_space)
        )
        raw_initial_parameterization = (
            getattr(pomdp_space, "initial_parameterization", None) or {}
        )
        matrix_keys = set(getattr(pomdp_space, "matrices", None) or {})
        preserved_initial_metadata = {
            key: value
            for key, value in raw_initial_parameterization.items()
            if key not in matrix_keys
        }
        initial_parameterization = {
            **initial_parameterization,
            **preserved_initial_metadata,
        }
        raw_model_parameters = getattr(pomdp_space, "model_parameters", None) or {}
        state_factors = getattr(pomdp_space, "state_factors", None) or []
        observation_modalities = (
            getattr(pomdp_space, "observation_modalities", None) or []
        )
        control_factors = getattr(pomdp_space, "control_factors", None) or []

        gnn_spec: dict[str, Any] = {
            "name": pomdp_space.model_name or "POMDP_Model",
            "model_name": pomdp_space.model_name or "POMDP_Model",
            "description": pomdp_space.model_annotation or "Extracted POMDP model",
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "model_parameters": {
                **raw_model_parameters,
                **canonical_model_parameters,
                "num_state_factors": len(state_factors)
                or canonical_model_parameters.get("num_state_factors")
                or raw_model_parameters.get("num_state_factors"),
                "num_modalities": len(observation_modalities)
                or canonical_model_parameters.get("num_modalities")
                or raw_model_parameters.get("num_modalities"),
                "state_factors": state_factors,
                "observation_modalities": observation_modalities,
                "control_factors": control_factors,
                "passive_model": getattr(pomdp_space, "passive_model", False),
                "simulation_params": parsed_sim_params,
                **({"num_timesteps": timesteps} if timesteps else {}),
            },
            "initialparameterization": initial_parameterization,
            "structured_pomdp": {
                "matrices": getattr(pomdp_space, "matrices", None) or {},
                "matrix_provenance": matrix_provenance,
                "state_factors": state_factors,
                "observation_modalities": observation_modalities,
                "control_factors": control_factors,
                "adapter_notes": getattr(pomdp_space, "adapter_notes", None) or [],
            },
            "matrix_provenance": matrix_provenance,
            "canonical_pomdp_schema": "canonical_pomdp_v1",
            "variables": [],
            "connections": [],
        }

        # Add variable definitions
        if pomdp_space.state_variables:
            gnn_spec["variables"].extend(pomdp_space.state_variables)
        if pomdp_space.observation_variables:
            gnn_spec["variables"].extend(pomdp_space.observation_variables)
        if pomdp_space.action_variables:
            gnn_spec["variables"].extend(pomdp_space.action_variables)

        # Add connections
        if pomdp_space.connections:
            gnn_spec["connections"] = [
                {"source": conn[0], "relation": conn[1], "target": conn[2]}
                for conn in pomdp_space.connections
            ]

        # Add ontology mapping if available
        if pomdp_space.ontology_mapping:
            gnn_spec["ontology_mapping"] = pomdp_space.ontology_mapping

        return gnn_spec

    def _native_agent_pomdp_to_gnn_spec(
        self,
        pomdp_space: "POMDPStateSpace",
        *,
        timesteps: Optional[int],
        simulation_params: Any,
    ) -> Dict[str, Any]:
        """Preserve native agent declarations without allocating a joint tensor."""
        from gnn.render.multi_agent_common import validate_native_agent_groups

        matrices = pomdp_space.matrices or {}
        provenance = pomdp_space.matrix_provenance or {}
        params = {
            **(pomdp_space.model_parameters or {}),
            "simulation_params": simulation_params,
            "state_factors": pomdp_space.state_factors or [],
            **({"num_timesteps": timesteps} if timesteps is not None else {}),
        }
        spec = {
            "name": pomdp_space.model_name,
            "model_name": pomdp_space.model_name,
            "gnn_section": pomdp_space.gnn_section,
            "model_kind": "multi_agent",
            "model_parameters": params,
            "initialparameterization": pomdp_space.initial_parameterization or {},
            "structured_pomdp": {
                "matrices": matrices,
                "matrix_provenance": provenance,
                "state_factors": pomdp_space.state_factors or [],
                "observation_modalities": pomdp_space.observation_modalities or [],
                "control_factors": pomdp_space.control_factors or [],
            },
            "matrix_provenance": provenance,
            "canonical_pomdp_schema": "native_agent_pomdp_v1",
            "variables": (pomdp_space.state_variables or [])
            + (pomdp_space.observation_variables or [])
            + (pomdp_space.action_variables or []),
            "connections": [
                {"source": c[0], "relation": c[1], "target": c[2]}
                for c in (pomdp_space.connections or [])
            ],
            "ontology_mapping": pomdp_space.ontology_mapping or {},
        }
        validate_native_agent_groups(spec)
        return spec

    def _continuous_pomdp_to_gnn_spec(
        self,
        pomdp_space: "POMDPStateSpace",
        *,
        timesteps: Optional[int],
        simulation_params: Any,
    ) -> Dict[str, Any]:
        """Spec for a linear-Gaussian model: F/H/Q/R + prior passed verbatim.

        No canonical A/B/C/D is built — deriving categorical matrices from a
        continuous system would fabricate a stand-in.
        """
        matrices = dict(getattr(pomdp_space, "matrices", None) or {})
        raw_initial = getattr(pomdp_space, "initial_parameterization", None) or {}
        raw_model_parameters = dict(
            getattr(pomdp_space, "model_parameters", None) or {}
        )
        model_parameters: dict[str, Any] = {
            **raw_model_parameters,
            "num_states": pomdp_space.num_states,
            "num_observations": pomdp_space.num_observations,
            "num_hidden_states": pomdp_space.num_states,
            "num_obs": pomdp_space.num_observations,
            "num_actions": pomdp_space.num_actions,
            "passive_model": getattr(pomdp_space, "passive_model", True),
            "simulation_params": simulation_params,
            "dt": raw_model_parameters.get("dt", 1.0),
            "random_seed": raw_model_parameters.get(
                "random_seed", raw_model_parameters.get("seed", 42)
            ),
        }
        if timesteps is not None:
            model_parameters["num_timesteps"] = timesteps
        provenance = {
            key: {
                "source": "InitialParameterization",
                "shape": _continuous_shape(value),
                "derived": False,
            }
            for key, value in matrices.items()
        }
        preserved = {k: v for k, v in raw_initial.items() if k not in matrices}
        gnn_spec: dict[str, Any] = {
            "name": pomdp_space.model_name or "Continuous_Model",
            "model_name": pomdp_space.model_name or "Continuous_Model",
            "description": pomdp_space.model_annotation or "Continuous LGSSM model",
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "model_kind": "continuous",
            "model_parameters": model_parameters,
            "initialparameterization": {**matrices, **preserved},
            "initial_parameterization": {**matrices, **preserved},
            "structured_pomdp": {
                "matrices": matrices,
                "matrix_provenance": provenance,
                "state_factors": [],
                "observation_modalities": [],
                "control_factors": getattr(pomdp_space, "control_factors", None) or [],
                "adapter_notes": getattr(pomdp_space, "adapter_notes", None) or [],
            },
            "matrix_provenance": provenance,
            "canonical_pomdp_schema": "continuous_lgssm_v1",
            "variables": [],
            "connections": [],
        }
        for attr in ("state_variables", "observation_variables", "action_variables"):
            values = getattr(pomdp_space, attr, None)
            if values:
                gnn_spec["variables"].extend(values)
        if pomdp_space.connections:
            gnn_spec["connections"] = [
                {"source": c[0], "relation": c[1], "target": c[2]}
                for c in pomdp_space.connections
            ]
        if pomdp_space.ontology_mapping:
            gnn_spec["ontology_mapping"] = pomdp_space.ontology_mapping
        return gnn_spec

    def _nonstationary_pomdp_to_gnn_spec(
        self,
        pomdp_space: "POMDPStateSpace",
        *,
        timesteps: Optional[int],
        simulation_params: Any,
    ) -> Dict[str, Any]:
        """Spec for a nonstationary model: A/C/D[/E] plus raw B_t/B_regime.

        No static canonical B is built — projecting a time-indexed or
        regime-switched tensor to one static B would silently drop the
        switching semantics. The declared time tensor (and the schedule in
        model_parameters) passes through verbatim;
        ``run_pymdp_simulation`` resolves the per-step transition from it
        and refuses a ``B_regime`` declared without ``b_regime_schedule``.
        """
        matrices = dict(getattr(pomdp_space, "matrices", None) or {})
        raw_initial = getattr(pomdp_space, "initial_parameterization", None) or {}
        raw_model_parameters = dict(
            getattr(pomdp_space, "model_parameters", None) or {}
        )
        initial: Dict[str, Any] = {}
        provenance: Dict[str, Dict[str, Any]] = {}
        for key, value in sorted(matrices.items()):
            if key in {"A", "B", "C", "D", "E"} or key.startswith(("B_t", "B_regime")):
                initial[key] = value
                shape = list(np.asarray(value).shape)
                if key.startswith(("B_t", "B_regime")):
                    provenance[key] = {
                        "source": "nonstationary_raw_passthrough",
                        "source_key": key,
                        "shape": shape,
                        "derived": False,
                        "reason": (
                            "Nonstationary transition tensor passes through "
                            "verbatim; run_pymdp_simulation resolves the "
                            "per-step transition"
                        ),
                    }
                else:
                    provenance[key] = {
                        "source": "InitialParameterization",
                        "shape": shape,
                        "derived": False,
                    }
        preserved = {
            key: value
            for key, value in raw_initial.items()
            if key not in matrices and key not in initial
        }
        initial = {**initial, **preserved}
        model_parameters: dict[str, Any] = {
            **raw_model_parameters,
            "num_hidden_states": pomdp_space.num_states,
            "num_states": pomdp_space.num_states,
            "num_obs": pomdp_space.num_observations,
            "num_observations": pomdp_space.num_observations,
            "num_actions": pomdp_space.num_actions,
            "passive_model": getattr(pomdp_space, "passive_model", False),
            "simulation_params": simulation_params,
        }
        if timesteps is not None:
            model_parameters["num_timesteps"] = timesteps
        gnn_spec: dict[str, Any] = {
            "name": pomdp_space.model_name or "Nonstationary_Model",
            "model_name": pomdp_space.model_name or "Nonstationary_Model",
            "description": pomdp_space.model_annotation or "Nonstationary POMDP model",
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "model_parameters": model_parameters,
            "initialparameterization": initial,
            "initial_parameterization": initial,
            "structured_pomdp": {
                "matrices": matrices,
                "matrix_provenance": provenance,
                "state_factors": getattr(pomdp_space, "state_factors", None) or [],
                "observation_modalities": getattr(
                    pomdp_space, "observation_modalities", None
                )
                or [],
                "control_factors": getattr(pomdp_space, "control_factors", None) or [],
                "adapter_notes": getattr(pomdp_space, "adapter_notes", None) or [],
            },
            "matrix_provenance": provenance,
            "canonical_pomdp_schema": "nonstationary_raw_v1",
            "variables": [],
            "connections": [],
        }
        for attr in ("state_variables", "observation_variables", "action_variables"):
            values = getattr(pomdp_space, attr, None)
            if values:
                gnn_spec["variables"].extend(values)
        if pomdp_space.connections:
            gnn_spec["connections"] = [
                {"source": c[0], "relation": c[1], "target": c[2]}
                for c in pomdp_space.connections
            ]
        if pomdp_space.ontology_mapping:
            gnn_spec["ontology_mapping"] = pomdp_space.ontology_mapping
        return gnn_spec

    def _structural_pomdp_to_gnn_spec(
        self,
        pomdp_space: "POMDPStateSpace",
        *,
        timesteps: Optional[int],
        simulation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Spec for a structural wrapper: declared keys passed through verbatim.

        A blanket pattern declares structure, not values — fabricating a
        categorical A/B/C/D[/E] parameterization (or a continuous one) would
        be a stand-in, so the raw InitialParameterization keys survive
        unchanged and the spec is stamped ``model_kind: "structural"`` for
        the render contract to report as render-only / informational.
        """
        raw_initial = dict(getattr(pomdp_space, "initial_parameterization", None) or {})
        raw_model_parameters = dict(
            getattr(pomdp_space, "model_parameters", None) or {}
        )
        model_parameters: dict[str, Any] = {
            **raw_model_parameters,
            "passive_model": getattr(pomdp_space, "passive_model", True),
            "simulation_params": simulation_params,
        }
        if timesteps is not None:
            model_parameters["num_timesteps"] = timesteps
        gnn_spec: dict[str, Any] = {
            "name": pomdp_space.model_name or "Structural_Model",
            "model_name": pomdp_space.model_name or "Structural_Model",
            "description": pomdp_space.model_annotation
            or "Structural wrapper specification",
            "gnn_section": getattr(pomdp_space, "gnn_section", None),
            "model_kind": "structural",
            "model_parameters": model_parameters,
            "initialparameterization": raw_initial,
            "structured_pomdp": {
                "matrices": dict(getattr(pomdp_space, "matrices", None) or {}),
                "matrix_provenance": dict(
                    getattr(pomdp_space, "matrix_provenance", None) or {}
                ),
                "state_factors": getattr(pomdp_space, "state_factors", None) or [],
                "observation_modalities": getattr(
                    pomdp_space, "observation_modalities", None
                )
                or [],
                "control_factors": getattr(pomdp_space, "control_factors", None) or [],
                "adapter_notes": getattr(pomdp_space, "adapter_notes", None) or [],
            },
            "canonical_pomdp_schema": "structural_spec_v1",
            "variables": [],
            "connections": [],
        }
        for attr in ("state_variables", "observation_variables", "action_variables"):
            values = getattr(pomdp_space, attr, None)
            if values:
                gnn_spec["variables"].extend(values)
        if pomdp_space.connections:
            gnn_spec["connections"] = [
                {"source": c[0], "relation": c[1], "target": c[2]}
                for c in pomdp_space.connections
            ]
        if pomdp_space.ontology_mapping:
            gnn_spec["ontology_mapping"] = pomdp_space.ontology_mapping
        return gnn_spec
