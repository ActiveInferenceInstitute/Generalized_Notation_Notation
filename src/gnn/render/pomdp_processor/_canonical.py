#!/usr/bin/env python3
"""Canonical-spec mixins for the POMDP render processor."""

import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from gnn.render.pomdp_contract import build_canonical_pomdp_spec, canonicalise_b_matrix
from gnn.render.pomdp_math import (
    _factor_action_counts,
    _is_kronecker_factorized_spec,
    _mixed_radix_digit,
    _normalise_columns,
    _normalise_prob_vector,
)

from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _CanonicalSpecMixin(_POMDPProcessorSupportMixin):
    def _build_canonical_initialparameterization(
        self,
        pomdp_space: "POMDPStateSpace",
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Build the strict canonical A/B/C/D/E contract used by renderers."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        provenance = dict(getattr(pomdp_space, "matrix_provenance", None) or {})
        has_canonical = all(key in matrices for key in ("A", "B", "C", "D"))

        if has_canonical:
            initial = {key: matrices[key] for key in ("A", "B", "C", "D")}
            if "E" in matrices:
                initial["E"] = matrices["E"]
            gnn_spec = {
                "initialparameterization": initial,
                "model_parameters": getattr(pomdp_space, "model_parameters", None)
                or {},
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
                "matrix_provenance": provenance,
            }
            canonical = build_canonical_pomdp_spec(gnn_spec)
            return (
                canonical["initialparameterization"],
                canonical["matrix_provenance"],
                canonical["model_parameters"],
            )

        time_indexed_b = self._time_indexed_transition_key(matrices)
        if time_indexed_b and all(key in matrices for key in ("A", "C", "D")):
            b_tensor = self._canonicalise_time_indexed_B(
                matrices[time_indexed_b],
                max(1, int(getattr(pomdp_space, "num_actions", 1))),
            )
            initial = {
                "A": matrices["A"],
                "B": b_tensor.tolist(),
                "C": matrices["C"],
                "D": matrices["D"],
            }
            if "E" in matrices:
                initial["E"] = matrices["E"]
            provenance["B"] = {
                "source": "time_indexed_transition_projection",
                "source_key": time_indexed_b,
                "shape": list(b_tensor.shape),
                "derived": True,
                "reason": "PyMDP static transition contract uses the declared B_t tensor for execution",
            }
            gnn_spec = {
                "initialparameterization": initial,
                "model_parameters": getattr(pomdp_space, "model_parameters", None)
                or {},
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
                "matrix_provenance": provenance,
            }
            canonical = build_canonical_pomdp_spec(gnn_spec)
            return (
                canonical["initialparameterization"],
                canonical["matrix_provenance"],
                canonical["model_parameters"],
            )

        if any(str(key).startswith("E_agent") for key in matrices):
            raise ValueError(
                "Agent-specific habit priors E_agentN require a supported composition; declared values cannot be omitted"
            )
        joint, joint_provenance = self._compose_factored_pomdp(pomdp_space)
        initial = {
            "A": joint["A"],
            "B": joint["B"],
            "C": joint["C"],
            "D": joint["D"],
        }
        if "E" in matrices:
            initial["E"] = matrices["E"]
        provenance.update(joint_provenance)
        joint_num_actions = int(
            joint.get("num_actions") or pomdp_space.num_actions or 1
        )
        canonical_model_parameters = {
            **(getattr(pomdp_space, "model_parameters", None) or {}),
            "num_hidden_states": len(initial["D"]),
            "num_obs": len(initial["A"]),
            "num_actions": joint_num_actions,
        }
        # Composition has already produced canonical axes. The original
        # per-factor/source declaration must never re-transpose this derived
        # joint tensor, including equal-sized state/action dimensions.
        canonical_model_parameters["b_tensor_order"] = (
            "next_state_previous_state_action"
        )
        gnn_spec = {
            "initialparameterization": initial,
            "model_parameters": canonical_model_parameters,
            "num_states": len(initial["D"]),
            "num_observations": len(initial["A"]),
            "num_actions": joint_num_actions,
            "matrix_provenance": provenance,
        }
        canonical = build_canonical_pomdp_spec(gnn_spec)
        return (
            canonical["initialparameterization"],
            canonical["matrix_provenance"],
            canonical["model_parameters"],
        )

    def _compose_factored_pomdp(
        self, pomdp_space: "POMDPStateSpace"
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Compose factored POMDP matrices into a joint PyMDP model without dropping factors."""
        matrices = getattr(pomdp_space, "matrices", None) or {}
        state_factors = [
            factor
            for factor in (getattr(pomdp_space, "state_factors", None) or [])
            if factor.get("size")
        ]
        obs_modalities = [
            modality
            for modality in (getattr(pomdp_space, "observation_modalities", None) or [])
            if modality.get("size")
        ]

        if not state_factors:
            raise ValueError("Factored POMDP is missing state factor metadata")
        if not obs_modalities:
            raise ValueError("Factored POMDP is missing observation modality metadata")

        state_sizes = [int(factor["size"]) for factor in state_factors]
        obs_sizes = [int(modality["size"]) for modality in obs_modalities]
        state_tuples = list(itertools.product(*[range(size) for size in state_sizes]))
        obs_tuples = list(itertools.product(*[range(size) for size in obs_sizes]))
        num_states = len(state_tuples)
        num_obs = len(obs_tuples)

        a_keys = sorted(key for key in matrices if key.startswith("A_"))
        b_keys = sorted(key for key in matrices if key.startswith("B_"))
        c_keys = sorted(key for key in matrices if key.startswith("C_"))
        d_keys = sorted(key for key in matrices if key.startswith("D_"))

        # MAJ-02: Kronecker-factorized specs carry independent per-factor
        # action spaces; the joint action space is their product. Shared-
        # control factored specs (gridworld) and multi-agent specs keep the
        # extractor's flat ``num_actions`` (one joint action index applied to
        # every factor).
        kronecker_factorized = _is_kronecker_factorized_spec(pomdp_space)
        factor_action_counts = _factor_action_counts(matrices, b_keys)
        if kronecker_factorized and factor_action_counts:
            num_actions = max(1, int(np.prod(factor_action_counts)))
        else:
            num_actions = max(1, int(getattr(pomdp_space, "num_actions", 1)))
            if any(key.startswith("B_agent") for key in b_keys):
                active_action_counts = {
                    count for count in factor_action_counts if count > 1
                }
                if len(active_action_counts) > 1 or (
                    active_action_counts and num_actions not in active_action_counts
                ):
                    raise ValueError(
                        "unsupported-agent-action-composition: per-agent action dimensions "
                        f"{factor_action_counts} cannot use the declared shared action count "
                        f"{num_actions}"
                    )

        if not a_keys:
            raise ValueError("Factored POMDP is missing A_* likelihood matrices")
        if not b_keys:
            raise ValueError("Factored POMDP is missing B_* transition matrices")
        if not d_keys:
            raise ValueError("Factored POMDP is missing D_* prior vectors")

        A_joint = np.ones((num_obs, num_states), dtype=np.float64)
        for key in a_keys:
            matrix = np.asarray(matrices[key], dtype=np.float64)
            if matrix.ndim not in (2, 3):
                raise ValueError(
                    f"{key} must be 2D or 3D for PyMDP composition, got shape {matrix.shape}"
                )
            source_shape: tuple[int, ...] = matrix.shape
            matrix = _normalise_columns(matrix.reshape(source_shape[0], -1)).reshape(
                source_shape
            )
            obs_index = self._match_descriptor_index(
                key, obs_modalities, source_shape[0]
            )
            state_indices = self._match_state_indices_for_matrix(
                key, state_factors, source_shape[1:]
            )
            for obs_flat, obs_tuple in enumerate(obs_tuples):
                for state_flat, state_tuple in enumerate(state_tuples):
                    matrix_index: list[Any] = [obs_tuple[obs_index]]
                    matrix_index.extend(state_tuple[index] for index in state_indices)
                    A_joint[obs_flat, state_flat] *= float(matrix[tuple(matrix_index)])
        A_joint = _normalise_columns(A_joint, allow_weights=True)

        B_joint = np.ones((num_states, num_states, num_actions), dtype=np.float64)
        component_b_provenance: dict[str, Any] = {}
        for key, factor_actions in zip(b_keys, factor_action_counts):
            factor_index = self._match_descriptor_index(key, state_factors)
            factor_size = state_sizes[factor_index]
            tensor, transition_provenance = (
                self._canonicalise_factored_B_with_provenance(
                    matrices[key],
                    factor_size,
                    factor_actions,
                    agent_matrix=key.startswith("B_agent"),
                    model_parameters={
                        **(getattr(pomdp_space, "model_parameters", None) or {}),
                        **(
                            {
                                "b_tensor_order": (
                                    getattr(pomdp_space, "model_parameters", None) or {}
                                )[f"b_tensor_order_{key[2:]}"]
                            }
                            if f"b_tensor_order_{key[2:]}"
                            in (getattr(pomdp_space, "model_parameters", None) or {})
                            else {}
                        ),
                    },
                )
            )
            # The structured matrices remain original source values. The
            # derived joint B has its own canonical order and must never
            # overwrite the retained component tensor's source orientation.
            component_b_provenance[key] = {
                **(
                    (getattr(pomdp_space, "matrix_provenance", None) or {}).get(key)
                    or {}
                ),
                "source_order": transition_provenance["source_order"],
                "source_shape": list(np.asarray(matrices[key]).shape),
                "declared_order_explicit": transition_provenance[
                    "declared_order_explicit"
                ],
                "canonicalization": transition_provenance,
            }
            for action in range(num_actions):
                if kronecker_factorized:
                    source_action = _mixed_radix_digit(
                        action, factor_action_counts, factor_index
                    )
                else:
                    source_action = action if tensor.shape[2] > 1 else 0
                for prev_flat, prev_tuple in enumerate(state_tuples):
                    for next_flat, next_tuple in enumerate(state_tuples):
                        B_joint[next_flat, prev_flat, action] *= float(
                            tensor[
                                next_tuple[factor_index],
                                prev_tuple[factor_index],
                                source_action,
                            ]
                        )
        for action in range(num_actions):
            B_joint[:, :, action] = _normalise_columns(
                B_joint[:, :, action], allow_weights=True
            )

        if c_keys:
            C_joint = np.zeros(num_obs, dtype=np.float64)
            for key in c_keys:
                vector = np.asarray(matrices[key], dtype=np.float64).flatten()
                obs_index = self._match_descriptor_index(
                    key, obs_modalities, vector.shape[0]
                )
                for obs_flat, obs_tuple in enumerate(obs_tuples):
                    C_joint[obs_flat] += float(vector[obs_tuple[obs_index]])
        elif getattr(pomdp_space, "passive_model", False):
            C_joint = np.zeros(num_obs, dtype=np.float64)
        else:
            raise ValueError("Factored POMDP is missing C_* preference vectors")

        D_joint = np.ones(num_states, dtype=np.float64)
        for key in d_keys:
            vector = _normalise_prob_vector(np.asarray(matrices[key], dtype=np.float64))
            factor_index = self._match_descriptor_index(
                key, state_factors, vector.shape[0]
            )
            for state_flat, state_tuple in enumerate(state_tuples):
                D_joint[state_flat] *= float(vector[state_tuple[factor_index]])
        D_joint = _normalise_prob_vector(D_joint, allow_weights=True)

        provenance: dict[str, Any] = {
            **component_b_provenance,
            "A": {
                "source": "factored_joint_composition",
                "normalization_scope": "derived_product_of_validated_source_conditionals",
                "source_keys": a_keys,
                "shape": list(A_joint.shape),
                "derived": True,
            },
            "B": {
                "source": "factored_joint_composition",
                "normalization_scope": "derived_product_of_validated_source_conditionals",
                "source_keys": b_keys,
                "shape": list(B_joint.shape),
                "derived": True,
                "factor_action_counts": factor_action_counts,
                "kronecker_factorized": kronecker_factorized,
            },
            "C": {
                "source": "factored_joint_composition",
                "normalization_scope": "derived_product_of_validated_source_conditionals",
                "source_keys": c_keys,
                "shape": list(C_joint.shape),
                "derived": True,
            },
            "D": {
                "source": "factored_joint_composition",
                "normalization_scope": "derived_product_of_validated_source_conditionals",
                "source_keys": d_keys,
                "shape": list(D_joint.shape),
                "derived": True,
            },
        }

        return (
            {
                "A": A_joint.tolist(),
                "B": B_joint.tolist(),
                "C": C_joint.tolist(),
                "D": D_joint.tolist(),
                "num_actions": num_actions,
                "kronecker_factorized": kronecker_factorized,
            },
            provenance,
        )

    def _canonicalise_time_indexed_B(self, value: Any, num_actions: int) -> np.ndarray:
        """Canonicalize B_t to (next_state, previous_state, action)."""
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim == 2:
            tensor = raw[:, :, np.newaxis]
        elif raw.ndim == 3:
            if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
                tensor = raw.transpose(1, 2, 0)
            elif raw.shape[0] == raw.shape[1] and raw.shape[2] in {1, num_actions}:
                tensor = raw
            else:
                raise ValueError(
                    f"B_t must be action-first or canonical 3D tensor, got shape {raw.shape}"
                )
        else:
            raise ValueError(f"B_t must be 2D or 3D, got shape {raw.shape}")
        for action in range(tensor.shape[2]):
            tensor[:, :, action] = _normalise_columns(tensor[:, :, action])
        return tensor

    def _canonicalise_factored_B(
        self,
        value: Any,
        factor_size: int,
        num_actions: int,
        *,
        agent_matrix: bool = False,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Apply the shared declared-axis and source-probability boundary."""
        return self._canonicalise_factored_B_with_provenance(
            value,
            factor_size,
            num_actions,
            agent_matrix=agent_matrix,
            model_parameters=model_parameters,
        )[0]

    def _canonicalise_factored_B_with_provenance(
        self,
        value: Any,
        factor_size: int,
        num_actions: int,
        *,
        agent_matrix: bool = False,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Bind retained source axes to the same strict joint canonicalization."""
        parameters = dict(model_parameters or {})
        raw = np.asarray(value)
        declared_order = any(
            parameters.get(k)
            for k in ("b_tensor_order", "B_tensor_order", "transition_tensor_order")
        )
        orientation_resolution = (
            "declared_model_parameter" if declared_order else "shape_inferred"
        )
        if (
            agent_matrix
            and not declared_order
            and raw.ndim == 3
            and raw.shape[0] in {1, num_actions}
            and raw.shape[1:] == (factor_size, factor_size)
        ):
            parameters["b_tensor_order"] = "action_next_state_previous_state"
            orientation_resolution = "native_agent_action_first_default"
        canonical, provenance = canonicalise_b_matrix(
            value,
            num_states=factor_size,
            num_actions=num_actions,
            model_parameters=parameters,
        )
        return np.asarray(canonical), {
            **provenance,
            "orientation_resolution": orientation_resolution,
            "declared_order_explicit": bool(declared_order),
        }

    def _match_descriptor_index(
        self,
        matrix_key: str,
        descriptors: List[Dict[str, Any]],
        required_size: Optional[int] = None,
    ) -> int:
        """Handle match descriptor index for internal callers."""
        suffix = (
            matrix_key.split("_", 1)[1].lower()
            if "_" in matrix_key
            else matrix_key.lower()
        )
        matches = [
            index
            for index, descriptor in enumerate(descriptors)
            if suffix in str(descriptor.get("name", "")).lower()
        ]
        if required_size is not None:
            matches = [
                index
                for index in matches
                if int(descriptors[index].get("size") or -1) == int(required_size)
            ] or [
                index
                for index, descriptor in enumerate(descriptors)
                if int(descriptor.get("size") or -1) == int(required_size)
            ]
        if len(matches) == 1:
            return matches[0]
        if not matches and len(descriptors) == 1:
            return 0
        raise ValueError(
            f"Could not map {matrix_key} to descriptors {[d.get('name') for d in descriptors]}"
        )

    def _match_state_indices_for_matrix(
        self,
        matrix_key: str,
        state_factors: List[Dict[str, Any]],
        matrix_state_shape: tuple[int, ...],
    ) -> List[int]:
        """Handle match state indices for matrix for internal callers."""
        if len(matrix_state_shape) == 1:
            return [
                self._match_descriptor_index(
                    matrix_key, state_factors, matrix_state_shape[0]
                )
            ]
        indices: List[int] = []
        used: set[int] = set()
        for size in matrix_state_shape:
            matches = [
                index
                for index, factor in enumerate(state_factors)
                if index not in used and int(factor.get("size") or -1) == int(size)
            ]
            if not matches:
                raise ValueError(
                    f"Could not map {matrix_key} state shape {matrix_state_shape} to state factors"
                )
            index = matches[0]
            used.add(index)
            indices.append(index)
        return indices
