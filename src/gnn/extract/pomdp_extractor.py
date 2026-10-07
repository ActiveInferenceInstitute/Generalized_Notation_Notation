#!/usr/bin/env python3
"""
POMDP State Space Extractor for GNN Active Inference Models

This module is the public entry point for POMDP extraction from GNN
specifications, with focus on Active Inference model structures.

Mechanical sibling split (M-01 band split): the extraction surface now lives
in ``pomdp_state.py`` (payload types + canonicalization), ``pomdp_support.py``
(shared shape and error-recording helpers), ``pomdp_sections.py``,
``pomdp_parameters.py``, and ``pomdp_orientation.py``. This module re-exports
every public name and composes :class:`POMDPExtractor` from the sibling
mixins, so consumer import paths and signatures are unchanged.

Dependency contract
-------------------
This module is stdlib-only at import time. The only non-stdlib-adjacent import
(``gnn.utils.runtime_safety.safe_eval`` for literal matrix evaluation) is performed lazily inside
_parse_parameter_value; the heavy pipeline (numpy, jax, pymdp, renderers, ...)
is NOT required. Headless consumers can use ``gnn.extract`` (CLI) or import
``gnn.extract.pomdp_extractor`` directly under a blocked-import environment.

Stability promise: the mapping produced by :meth:`POMDPStateSpace.to_dict` is
versioned via its ``extraction_schema_version`` key (currently "1.0.0"). The 26
pre-existing keys keep identical semantics within schema version 1.x; new keys
may be appended in minor versions.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload

from .pomdp_orientation import POMDPOrientationMixin
from .pomdp_parameters import POMDPParametersMixin
from .pomdp_sections import POMDPSectionsMixin
from .pomdp_state import (
    CANONICAL_B_ORDER,
    ON_ERROR_MODES,
    GNNExtractionError,
    OnErrorMode,
    POMDPStateSpace,
    canonicalize_pomdp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CANONICAL_B_ORDER",
    "GNNExtractionError",
    "ON_ERROR_MODES",
    "OnErrorMode",
    "POMDPExtractor",
    "POMDPStateSpace",
    "canonicalize_pomdp",
    "extract_pomdp_from_content",
    "extract_pomdp_from_file",
]


class POMDPExtractor(POMDPSectionsMixin, POMDPParametersMixin, POMDPOrientationMixin):
    """
    Specialized extractor for POMDP state spaces from GNN specifications.

    Features:
    - Parses Active Inference matrix structures (A, B, C, D, E)
    - Extracts state space dimensions and variable definitions
    - Handles initial parameterization with matrix values
    - Maps ontology annotations to Active Inference concepts
    - Validates POMDP structural consistency
    """

    def __init__(self, strict_validation: bool = True) -> None:
        """
        Initialize POMDP extractor.

        Args:
            strict_validation: Enable strict validation of POMDP structure
        """
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(__name__)

        # Error-collection state (reset at the start of each extraction call,
        # but initialized here so private helpers are safe in isolation).
        self._on_error: str = "lenient"
        self._errors: List[GNNExtractionError] = []
        self._parse_failures: Dict[str, Dict[str, Any]] = {}
        self._section_line_offset: int = 0
        self._dimension_sources: Dict[str, str] = {}

        # Patterns for parsing GNN content
        self.SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        self.VARIABLE_PATTERN = re.compile(
            r"^([A-Za-z_π][A-Za-z0-9_π]*)\[([^\]]+)\](?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$"
        )
        self.CONNECTION_PATTERN = re.compile(
            r"^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$"
        )
        self.PARAMETER_PATTERN = re.compile(
            r"^([A-Za-z_π][A-Za-z0-9_π]*)\s*=\s*\{(.+)\}", re.MULTILINE | re.DOTALL
        )

    @overload
    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: Literal["lenient", "raise"] = ...,
        insert_default_c: bool = ...,
    ) -> Optional[POMDPStateSpace]: ...

    @overload
    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: Literal["collect"],
        insert_default_c: bool = ...,
    ) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...

    def extract_from_gnn_content(
        self,
        content: str,
        *,
        on_error: str = "lenient",
        insert_default_c: bool = True,
    ) -> Union[
        Optional[POMDPStateSpace],
        Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
    ]:
        """
        Extract POMDP state space from GNN content.

        Args:
            content: Raw GNN file content
            on_error: Failure policy.
                - 'lenient' (default): log faults and keep going; parse-error
                  provenance/adapter_notes records are still written.
                - 'raise': raise GNNExtractionError at the first fault.
                - 'collect': return (spec_or_None, list[GNNExtractionError]).
                Invalid values raise ValueError.
            insert_default_c: When True (default) a passive model without a C
                vector keeps the zero-preference adapter behavior. When False
                the faithful read is preserved: C_vector stays None and no
                passive_model_adapter provenance/adapter_notes entry is made.

        Returns:
            POMDPStateSpace (lenient/raise modes) or a
            (POMDPStateSpace | None, list[GNNExtractionError]) tuple
            (collect mode).
        """
        if on_error not in ON_ERROR_MODES:
            raise ValueError(
                f"on_error must be one of {ON_ERROR_MODES}, got {on_error!r}"
            )
        self._on_error = on_error
        self._errors = []
        self._parse_failures = {}
        self._section_line_offset = self._section_line_offset_for(
            content, "InitialParameterization"
        )
        try:
            sections = self._parse_sections(content)

            # Extract basic information
            model_name = self._extract_model_name(sections)
            model_annotation = self._extract_model_annotation(sections)
            gnn_section = sections.get("GNNSection", "").strip() or None

            # Parse state space block
            state_space_info = self._parse_state_space_block(
                sections.get("StateSpaceBlock", "")
            )

            # Parse initial parameterization FIRST (needed for dimension inference)
            initial_params = self._parse_initial_parameterization(
                sections.get("InitialParameterization", "")
            )
            model_parameters = self._parse_model_parameters(
                sections.get("ModelParameters", "")
            )

            # Continuous-state (linear-Gaussian) models declare F/H/Q/R and a
            # Gaussian prior instead of categorical A/B/C/D. Their dimensions
            # come from the system matrices, never from a discrete stand-in.
            is_continuous = self._is_continuous_model(gnn_section, initial_params)

            if is_continuous:
                num_states, num_observations, num_actions, num_timesteps = (
                    self._extract_continuous_dimensions(
                        state_space_info, initial_params, model_parameters
                    )
                )
            else:
                # Extract dimensions (now with access to sections and initial_params for better inference)
                num_states, num_observations, num_actions, num_timesteps = (
                    self._extract_dimensions(
                        state_space_info,
                        sections=sections,
                        initial_params=initial_params,
                    )
                )

            # Parse connections
            connections = self._parse_connections(sections.get("Connections", ""))

            # Parse ontology mapping
            ontology_mapping = self._parse_ontology_annotations(
                sections.get("ActInfOntologyAnnotation", "")
            )

            if is_continuous:
                matrices = self._collect_continuous_parameters(initial_params)
            else:
                matrices = self._collect_matrix_parameters(initial_params)
            matrix_provenance = self._build_matrix_provenance(matrices)
            state_factors = self._describe_variables(
                state_space_info.get("state_variables"), "state_factor"
            )
            observation_modalities = self._describe_variables(
                state_space_info.get("observation_variables"), "observation_modality"
            )
            control_factors = self._describe_variables(
                state_space_info.get("action_variables"), "control_factor"
            )
            if is_continuous:
                passive_model = not (
                    "goal_mean" in initial_params and "control_gain" in initial_params
                )
            else:
                passive_model = self._is_passive_model(
                    model_parameters=model_parameters,
                    initial_params=initial_params,
                    num_actions=num_actions,
                    connections=connections,
                )

            A_matrix = None if is_continuous else initial_params.get("A")
            B_matrix = None if is_continuous else initial_params.get("B")
            C_vector = None if is_continuous else initial_params.get("C")
            D_vector = None if is_continuous else initial_params.get("D")
            E_vector = None if is_continuous else initial_params.get("E")
            adapter_notes: list[Any] = []

            # Failed parameter blocks are never silently dropped: record
            # parse_error provenance + adapter_notes entries in every mode.
            for failed_name, failure in self._parse_failures.items():
                if failed_name in ("A", "B", "C", "D", "E") or failed_name.startswith(
                    ("A_", "B_", "C_", "D_", "E_")
                ):
                    matrix_provenance[failed_name] = {
                        "source": "parse_error",
                        "code": failure["code"],
                        "message": failure["message"],
                    }
                adapter_notes.append(
                    f"parse_error:{failed_name} [{failure['code']}]: {failure['message']}"
                )

            if (
                insert_default_c
                and not is_continuous
                and C_vector is None
                and passive_model
                and num_observations > 0
            ):
                C_vector = [0.0] * num_observations
                matrices["C"] = C_vector
                matrix_provenance["C"] = {
                    "source": "passive_model_adapter",
                    "shape": [num_observations],
                    "derived": True,
                    "reason": "zero preferences for passive HMM/Markov model",
                }
                adapter_notes.append("passive_model_zero_preferences")

            # B-orientation metadata (detection only — the stored B_matrix is
            # never transposed here; render/execute depend on as-written nesting).
            if not is_continuous and isinstance(B_matrix, (list, tuple)):
                b_provenance = matrix_provenance.get("B")
                if (
                    b_provenance is not None
                    and b_provenance.get("source") == "InitialParameterization"
                ):
                    orientation = self._analyze_b_orientation(
                        sections.get("StateSpaceBlock", ""),
                        sections.get("InitialParameterization", ""),
                        B_matrix,
                    )
                    b_provenance.update(orientation)
                    if orientation.get("contradiction"):
                        message = (
                            f"B orientation contradiction: {orientation.get('reason')}"
                        )
                        if self.strict_validation and on_error in ("raise", "collect"):
                            self._record_error(
                                "GNN-E002", message, section="StateSpaceBlock"
                            )
                        else:
                            self.logger.warning("B orientation: %s", message)

            # Computed factor counts (bookkeeping excluded; see dataclass).
            num_state_factors = sum(
                1 for descriptor in state_factors if descriptor.get("role") == "factor"
            )
            num_observation_modalities = sum(
                1
                for descriptor in observation_modalities
                if descriptor.get("role") == "factor"
            )
            num_control_factors = sum(
                1
                for descriptor in control_factors
                if descriptor.get("role") == "factor"
            )

            dimension_provenance = self._build_dimension_provenance(
                num_states, num_observations, num_actions, num_timesteps
            )

            # Preserve section bodies separately from the whitespace-normalized
            # matrix parser. Equations are evidence, not executable expressions.
            source_sections = {}
            headers = list(self.SECTION_PATTERN.finditer(content))
            for index, header in enumerate(headers):
                end = (
                    headers[index + 1].start()
                    if index + 1 < len(headers)
                    else len(content)
                )
                source_sections[header.group(1).strip()] = content[
                    header.end() : end
                ].strip()

            # Create POMDP state space
            pomdp_space = POMDPStateSpace(
                num_states=num_states,
                num_observations=num_observations,
                num_actions=num_actions,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                C_vector=C_vector,
                D_vector=D_vector,
                E_vector=E_vector,
                state_variables=state_space_info.get("state_variables"),
                observation_variables=state_space_info.get("observation_variables"),
                action_variables=state_space_info.get("action_variables"),
                state_factors=state_factors,
                observation_modalities=observation_modalities,
                control_factors=control_factors,
                connections=connections,
                model_name=model_name,
                model_annotation=model_annotation,
                gnn_section=gnn_section,
                ontology_mapping=ontology_mapping,
                num_timesteps=num_timesteps,
                model_parameters=model_parameters,
                matrices=matrices,
                matrix_provenance=matrix_provenance,
                passive_model=passive_model,
                adapter_notes=adapter_notes,
                initial_parameterization=initial_params,
                model_kind="continuous" if is_continuous else "discrete",
                num_state_factors=num_state_factors,
                num_observation_modalities=num_observation_modalities,
                num_control_factors=num_control_factors,
                dimension_provenance=dimension_provenance,
                time=source_sections.get("Time"),
                equations=source_sections.get("Equations"),
            )

            # Validate if strict validation enabled (discrete contract only —
            # continuous models are shape-checked in _extract_continuous_dimensions)
            if self.strict_validation and not is_continuous:
                validation_result = self._validate_pomdp_structure(pomdp_space)
                if not validation_result["valid"]:
                    self.logger.warning(
                        f"POMDP validation warnings: {validation_result['warnings']}"
                    )

            if on_error == "collect":
                return (pomdp_space, self._errors)
            return pomdp_space

        except GNNExtractionError:
            raise
        except Exception as e:
            error = GNNExtractionError(
                code="GNN-E999",
                message=f"unexpected extraction failure: {e}",
            )
            if on_error == "raise":
                raise error from e
            self.logger.error(f"Failed to extract POMDP state space: {e}")
            self._errors.append(error)
            if on_error == "collect":
                return (None, self._errors)
            return None

    @overload
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: Literal["lenient", "raise"] = ...,
        insert_default_c: bool = ...,
    ) -> Optional[POMDPStateSpace]: ...

    @overload
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: Literal["collect"],
        insert_default_c: bool = ...,
    ) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        *,
        on_error: str = "lenient",
        insert_default_c: bool = True,
    ) -> Union[
        Optional[POMDPStateSpace],
        Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
    ]:
        """
        Extract POMDP state space from GNN file.

        Args:
            file_path: Path to GNN file
            on_error: 'lenient' (default) | 'raise' | 'collect' — see
                extract_from_gnn_content. Invalid values raise ValueError.
            insert_default_c: Preserve (True, default) or suppress (False) the
                passive-model zero-C adapter.

        Returns:
            POMDPStateSpace (lenient/raise modes) or a
            (POMDPStateSpace | None, list[GNNExtractionError]) tuple
            (collect mode).
        """
        if on_error not in ON_ERROR_MODES:
            raise ValueError(
                f"on_error must be one of {ON_ERROR_MODES}, got {on_error!r}"
            )
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                error = GNNExtractionError(
                    code="GNN-E999",
                    message=f"file not found: {file_path}",
                )
                if on_error == "raise":
                    raise error
                self.logger.error("File not found: %s", file_path)
                self._errors.append(error)
                if on_error == "collect":
                    return (None, self._errors)
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return self.extract_from_gnn_content(
                content,
                on_error=cast(OnErrorMode, on_error),
                insert_default_c=insert_default_c,
            )

        except GNNExtractionError:
            raise
        except Exception as e:
            error = GNNExtractionError(
                code="GNN-E999",
                message=f"failed to read file {file_path}: {e}",
            )
            if on_error == "raise":
                raise error from e
            self.logger.error("Failed to read file %s: %s", file_path, e)
            self._errors.append(error)
            if on_error == "collect":
                return (None, self._errors)
            return None

    def _validate_pomdp_structure(self, pomdp_space: POMDPStateSpace) -> Dict[str, Any]:
        """Validate POMDP structure for consistency."""
        warnings: list[Any] = []

        # Check dimension consistency
        try:
            if pomdp_space.A_matrix and isinstance(pomdp_space.A_matrix, list):
                if len(pomdp_space.A_matrix) > 0 and isinstance(
                    pomdp_space.A_matrix[0], list
                ):
                    expected_a_dims = (
                        pomdp_space.num_observations,
                        pomdp_space.num_states,
                    )
                    actual_a_dims = (
                        len(pomdp_space.A_matrix),
                        len(pomdp_space.A_matrix[0]),
                    )
                    if expected_a_dims != actual_a_dims:
                        warnings.append(
                            f"A matrix dimensions {actual_a_dims} don't match expected {expected_a_dims}"
                        )
        except (TypeError, IndexError) as e:
            warnings.append(f"A matrix has invalid structure: {e}")

        try:
            if pomdp_space.B_matrix and isinstance(pomdp_space.B_matrix, list):
                if (
                    len(pomdp_space.B_matrix) > 0
                    and isinstance(pomdp_space.B_matrix[0], list)
                    and len(pomdp_space.B_matrix[0]) > 0
                    and isinstance(pomdp_space.B_matrix[0][0], list)
                ):
                    expected_b_dims = (
                        pomdp_space.num_states,
                        pomdp_space.num_states,
                        pomdp_space.num_actions,
                    )
                    stored_order = (
                        (pomdp_space.matrix_provenance or {}).get("B") or {}
                    ).get("detected_order")
                    if stored_order == ["action", "previous_state", "next_state"]:
                        expected_b_dims = (
                            pomdp_space.num_actions,
                            pomdp_space.num_states,
                            pomdp_space.num_states,
                        )
                    actual_b_dims = (
                        len(pomdp_space.B_matrix),
                        len(pomdp_space.B_matrix[0]),
                        len(pomdp_space.B_matrix[0][0]),
                    )
                    if expected_b_dims != actual_b_dims:
                        warnings.append(
                            f"B matrix dimensions {actual_b_dims} don't match expected {expected_b_dims}"
                        )
        except (TypeError, IndexError) as e:
            warnings.append(f"B matrix has invalid structure: {e}")

        try:
            if pomdp_space.C_vector and isinstance(pomdp_space.C_vector, list):
                if len(pomdp_space.C_vector) != pomdp_space.num_observations:
                    warnings.append(
                        f"C vector length {len(pomdp_space.C_vector)} doesn't match num_observations {pomdp_space.num_observations}"
                    )
        except TypeError as e:
            warnings.append(f"C vector has invalid structure: {e}")

        try:
            if pomdp_space.D_vector and isinstance(pomdp_space.D_vector, list):
                if len(pomdp_space.D_vector) != pomdp_space.num_states:
                    warnings.append(
                        f"D vector length {len(pomdp_space.D_vector)} doesn't match num_states {pomdp_space.num_states}"
                    )
        except TypeError as e:
            warnings.append(f"D vector has invalid structure: {e}")

        return {"valid": len(warnings) == 0, "warnings": warnings}


@overload
def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = ...,
    *,
    on_error: Literal["lenient", "raise"] = ...,
    insert_default_c: bool = ...,
) -> Optional[POMDPStateSpace]: ...


@overload
def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = ...,
    *,
    on_error: Literal["collect"],
    insert_default_c: bool = ...,
) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...


def extract_pomdp_from_file(
    file_path: Union[str, Path],
    strict_validation: bool = True,
    *,
    on_error: str = "lenient",
    insert_default_c: bool = True,
) -> Union[
    Optional[POMDPStateSpace],
    Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
]:
    """
    Convenience function to extract POMDP state space from a GNN file.

    Args:
        file_path: Path to GNN file
        strict_validation: Enable strict validation
        on_error: 'lenient' (default) | 'raise' | 'collect'. 'raise' raises
            GNNExtractionError at the first fault; 'collect' returns
            (spec_or_None, list[GNNExtractionError]); invalid -> ValueError.
        insert_default_c: True (default) preserves the passive-model zero-C
            adapter; False keeps C None with no adapter provenance.

    Returns:
        POMDPStateSpace (lenient/raise) or a (spec | None, errors) tuple
        (collect mode).
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_file(
        file_path,
        on_error=cast(OnErrorMode, on_error),
        insert_default_c=insert_default_c,
    )


@overload
def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = ...,
    *,
    on_error: Literal["lenient", "raise"] = ...,
    insert_default_c: bool = ...,
) -> Optional[POMDPStateSpace]: ...


@overload
def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = ...,
    *,
    on_error: Literal["collect"],
    insert_default_c: bool = ...,
) -> Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]]: ...


def extract_pomdp_from_content(
    content: str,
    strict_validation: bool = True,
    *,
    on_error: str = "lenient",
    insert_default_c: bool = True,
) -> Union[
    Optional[POMDPStateSpace],
    Tuple[Optional[POMDPStateSpace], List[GNNExtractionError]],
]:
    """
    Convenience function to extract POMDP state space from GNN content.

    Args:
        content: GNN file content
        strict_validation: Enable strict validation
        on_error: 'lenient' (default) | 'raise' | 'collect'. 'raise' raises
            GNNExtractionError at the first fault; 'collect' returns
            (spec_or_None, list[GNNExtractionError]); invalid -> ValueError.
        insert_default_c: True (default) preserves the passive-model zero-C
            adapter; False keeps C None with no adapter provenance.

    Returns:
        POMDPStateSpace (lenient/raise) or a (spec | None, errors) tuple
        (collect mode).
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_gnn_content(
        content,
        on_error=cast(OnErrorMode, on_error),
        insert_default_c=insert_default_c,
    )
