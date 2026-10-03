#!/usr/bin/env python3
"""
Parameter and continuous-collection parsing for the POMDP extractor.

Mechanical extraction from ``gnn.extract.pomdp_extractor`` (M-01 band split):
``POMDPParametersMixin`` holds the verbatim parameter-parsing machinery —
model-kind detection, continuous dimension extraction, per-factor continuous
parameter collection, matrix provenance, variable descriptions, initial
parameterization, and the literal value parsers (the lazy
``gnn.utils.runtime_safety.safe_eval`` import stays inside
``_parse_parameter_value``).

This module is stdlib-only at import time.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from .pomdp_support import POMDPExtractorSupportMixin

if TYPE_CHECKING:
    import logging


class POMDPParametersMixin(POMDPExtractorSupportMixin):
    """Verbatim parameter-parsing methods moved from ``POMDPExtractor``."""

    if TYPE_CHECKING:
        _dimension_sources: Dict[str, str]
        logger: logging.Logger
    CONTINUOUS_REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")
    CONTINUOUS_OPTIONAL_KEYS = ("goal_mean", "control_gain")
    # Per-factor LGSSM keys (Kronecker ^([ABCD])_f(\d+)$ analogue), consumed
    # by render.continuous_common.extract_factored_continuous_spec.
    _FACTOR_CONTINUOUS_KEY = re.compile(
        r"^(F|H|Q|R|prior_mean|prior_cov|goal_mean|control_gain)_f(\d+)$"
    )
    _AGENT_CONTINUOUS_KEY = re.compile(
        r"^(F|H|Q|R|prior_mean|prior_cov|goal_mean|control_gain)_agent([1-9]\d*)$"
    )

    def _is_continuous_model(
        self, gnn_section: Optional[str], initial_params: Dict[str, Any]
    ) -> bool:
        """Continuous when the section says so or a linear-Gaussian block is declared.

        A per-factor linear-Gaussian block classifies continuous with or
        without the section, matching the render contract's factored path.
        """
        if gnn_section and "continuous" in gnn_section.lower():
            return True
        if any(self._FACTOR_CONTINUOUS_KEY.match(key) for key in initial_params):
            return True
        if any(self._AGENT_CONTINUOUS_KEY.match(key) for key in initial_params):
            return True
        return all(key in initial_params for key in ("F", "H", "Q", "R"))

    def _extract_continuous_dimensions(
        self,
        state_space_info: Dict[str, Any],
        initial_params: Dict[str, Any],
        model_parameters: Dict[str, Any],
    ) -> Tuple[int, int, int, Optional[int]]:
        """Dimensions of a linear-Gaussian model: n from F, m from H."""
        if any(self._AGENT_CONTINUOUS_KEY.fullmatch(key) for key in initial_params):
            from gnn.render.multi_agent_continuous import (
                extract_multi_agent_continuous_specs,
            )

            agents = extract_multi_agent_continuous_specs(
                {
                    "initialparameterization": initial_params,
                    "model_parameters": model_parameters,
                }
            )
            first = next(iter(agents.values()))
            self._dimension_sources = {
                "num_states": "per_agent_block",
                "num_observations": "per_agent_block",
                "num_actions": "declared_continuous_controls",
                "num_timesteps": "ModelParameters",
            }
            return (
                first.n,
                first.m,
                int(any(agent.has_control for agent in agents.values())),
                first.num_timesteps,
            )
        from gnn.render.continuous_common import (
            extract_continuous_spec,
            extract_factored_continuous_spec,
        )

        spec = {
            "initialparameterization": initial_params,
            "model_parameters": model_parameters,
        }
        factor_keys = [
            key for key in initial_params if self._FACTOR_CONTINUOUS_KEY.fullmatch(key)
        ]
        if factor_keys and not all(
            key in initial_params for key in self.CONTINUOUS_REQUIRED_KEYS
        ):
            validated_factors = extract_factored_continuous_spec(spec)
            first_factor = validated_factors.factors[0]
            states, observations, timesteps = (
                first_factor.n,
                first_factor.m,
                validated_factors.num_timesteps,
            )
            dimension_source = "per_factor_block"
        else:
            missing = [
                key
                for key in self.CONTINUOUS_REQUIRED_KEYS
                if key not in initial_params
            ]
            if missing:
                raise ValueError(
                    f"continuous model is missing linear-Gaussian parameters {missing}"
                )
            validated = extract_continuous_spec(spec)
            states, observations, timesteps = (
                validated.n,
                validated.m,
                validated.num_timesteps,
            )
            dimension_source = "variable_dimensions"
        self._dimension_sources = {
            "num_states": dimension_source,
            "num_observations": dimension_source,
            "num_actions": "declared_continuous_controls"
            if state_space_info.get("action_variables")
            else "default",
            "num_timesteps": "ModelParameters"
            if "num_timesteps" in model_parameters
            else "default",
        }
        return (
            states,
            observations,
            int(bool(state_space_info.get("action_variables"))),
            timesteps if "num_timesteps" in model_parameters else None,
        )

    def _collect_continuous_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the linear-Gaussian parameter block (no discrete stand-in)."""
        keys = self.CONTINUOUS_REQUIRED_KEYS + self.CONTINUOUS_OPTIONAL_KEYS
        out: Dict[str, Any] = {}
        for key in keys:
            if key in params:
                value = params[key]
                if key == "control_gain":
                    # scalar declared as {(v)} parses to [v]
                    if isinstance(value, (list, tuple)):
                        while isinstance(value, (list, tuple)) and len(value) == 1:
                            value = value[0]
                    value = float(value)
                out[key] = value
        for key in params:
            if self._FACTOR_CONTINUOUS_KEY.match(
                key
            ) or self._AGENT_CONTINUOUS_KEY.fullmatch(key):
                value = params[key]
                if key.startswith("control_gain_f"):
                    # scalar declared as {(v)} parses to [v] (per-factor twin)
                    if isinstance(value, (list, tuple)):
                        while isinstance(value, (list, tuple)) and len(value) == 1:
                            value = value[0]
                    value = float(value)
                out[key] = value
        return out

    def _collect_matrix_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return all Active Inference matrices/vectors without collapsing factors."""
        matrices: Dict[str, Any] = {}
        for key, value in params.items():
            if key in {"A", "B", "C", "D", "E"} or any(
                key.startswith(f"{prefix}_") for prefix in ("A", "B", "C", "D", "E")
            ):
                matrices[key] = value
        return matrices

    def _build_matrix_provenance(
        self, matrices: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Describe where each extracted matrix came from and what shape it has."""
        return {
            key: {
                "source": "InitialParameterization",
                "shape": self._nested_shape(value),
                "derived": False,
            }
            for key, value in matrices.items()
        }

    def _describe_variables(
        self,
        variables: Optional[List[Dict[str, Any]]],
        fallback_prefix: str,
    ) -> List[Dict[str, Any]]:
        """Create factor/modality/control descriptors from parsed variables.

        Every descriptor carries 'role': 'factor' or 'bookkeeping'. Bookkeeping
        entries are next-state/next-observation aliases matching *_prime and
        the policy symbol π/pi; the action variable u is a real factor. Lists
        keep ALL entries; the num_* counts on POMDPStateSpace exclude
        bookkeeping.
        """
        descriptors: list[Any] = []
        for index, variable in enumerate(variables or []):
            name = variable.get("name") or f"{fallback_prefix}_{index}"
            name_lower = str(name).lower()
            if fallback_prefix == "state_factor" and not name_lower.startswith("s"):
                continue
            if fallback_prefix == "observation_modality" and not name_lower.startswith(
                "o"
            ):
                continue
            if (
                fallback_prefix == "control_factor"
                and name_lower not in {"u", "π", "pi"}
                and not name_lower.startswith("pi")
            ):
                continue
            dimensions = variable.get("dimensions") or []
            size = next((dim for dim in dimensions if isinstance(dim, int)), None)
            is_bookkeeping = name_lower.endswith("_prime") or (
                fallback_prefix == "control_factor" and name_lower in {"π", "pi"}
            )
            descriptors.append(
                {
                    "name": name,
                    "size": size,
                    "dimensions": dimensions,
                    "type": variable.get("type"),
                    "comment": variable.get("comment"),
                    "index": index,
                    "role": "bookkeeping" if is_bookkeeping else "factor",
                }
            )
        return descriptors

    def _is_passive_model(
        self,
        *,
        model_parameters: Dict[str, Any],
        initial_params: Dict[str, Any],
        num_actions: int,
        connections: List[Tuple[str, str, str]],
    ) -> bool:
        """Detect passive HMM/Markov models that have no control-dependent choices."""
        model_type = str(model_parameters.get("model_type", "")).lower()
        if any(term in model_type for term in ("hmm", "markov", "passive")):
            return True
        if int(num_actions) == 1:
            return True
        b_matrix = initial_params.get("B")
        if b_matrix is not None and len(self._nested_shape(b_matrix)) == 2:
            return True
        return False

    def _parse_initial_parameterization(self, content: str) -> Dict[str, Any]:
        """Parse InitialParameterization section."""
        params: dict[Any, Any] = {}

        # Split content into lines and process each parameter block
        lines = content.split("\n")
        current_param = None
        current_value = ""
        in_param_block = False

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            # Skip comments
            if line.startswith("#") or not line:
                continue

            # Check if this line starts a matrix/vector block parameter definition.
            if "={" in line and not in_param_block:
                # Start of parameter block
                param_name = line.split("={")[0].strip()
                raw_value = line.split("=", 1)[1].strip()
                raw_inner = (
                    raw_value[1:-1].strip()
                    if raw_value.startswith("{") and raw_value.endswith("}")
                    else ""
                )
                if ":" in raw_inner:
                    try:
                        params[param_name] = self._parse_assignment_value(raw_value)
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(param_name, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(param_name, e, line_no)
                    current_param = None
                    current_value = ""
                    continue

                current_param = param_name
                current_value = (
                    raw_value[1:] if raw_value.startswith("{") else raw_value
                )

                # Check if parameter ends on the same line
                if "}" in current_value:
                    # Single-line parameter
                    current_value = current_value.split("}")[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(current_param, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(current_param, e, line_no)
                    current_param = None
                    current_value = ""
                else:
                    # Multi-line parameter
                    in_param_block = True

            elif in_param_block and current_param:
                # Continue collecting parameter value
                if "}" in line:
                    # End of parameter block
                    current_value += " " + line.split("}")[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except (ValueError, SyntaxError) as e:
                        self._record_parameter_failure(current_param, e, line_no)
                    except Exception as e:  # unexpected — still never dropped
                        self._record_parameter_failure(current_param, e, line_no)
                    in_param_block = False
                    current_param = None
                    current_value = ""
                else:
                    # Add line to current value
                    current_value += " " + line

            elif "=" in line and not in_param_block:
                param_name, raw_value = line.split("=", 1)
                param_name = param_name.strip()
                raw_value = raw_value.strip()
                if not param_name:
                    continue
                try:
                    params[param_name] = self._parse_assignment_value(raw_value)
                except (ValueError, SyntaxError) as e:
                    self._record_parameter_failure(param_name, e, line_no)
                except Exception as e:  # unexpected — still never dropped
                    self._record_parameter_failure(param_name, e, line_no)

        return params

    def _parse_assignment_value(self, value_str: str) -> Any:
        """Parse a complete InitialParameterization assignment value."""
        value_str = value_str.strip()
        if value_str.startswith("{") and value_str.endswith("}"):
            inner = value_str[1:-1].strip()
            if ":" not in inner:
                value_str = inner
        return self._parse_parameter_value(value_str)

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse parameter value string into appropriate data structure."""

        value_str = value_str.strip()

        # Handle simple numeric values
        try:
            if re.match(r"^[-+]?\d*\.\d+$", value_str):
                return float(value_str)
            if re.match(r"^[-+]?\d+$", value_str):
                return int(value_str)
        except ValueError:
            self.logger.debug(
                "Value '%s' is not a simple numeric, trying structured formats",
                value_str[:40],
            )

        # Handle structured data (tuples/nested lists)
        if "(" in value_str or "[" in value_str:
            try:
                try:
                    from gnn.utils.runtime_safety.safe_eval import (
                        MATRIX_MAX_LEN,
                        safe_literal_eval,
                    )
                except ImportError as e:
                    # Heavy pipeline not importable: eval-free path keeps
                    # working; a broken/suspicious safe_eval is a structured
                    # GNN-E006 fault, never a silent drop.
                    raise ImportError(
                        f"gnn.utils.runtime_safety.safe_eval unavailable ({e}); cannot safely "
                        "evaluate structured parameter literal"
                    ) from e

                # Convert ( ) to [ ] for literal_eval if needed, or just let it handle tuples
                # Better to convert to a standard format
                clean_str = value_str.replace("(", "[").replace(")", "]")
                # Handle cases like ( (1,2), (3,4) ) -> [ [1,2], [3,4] ]
                # Remove extra commas if any (e.g., from trailing commas in GNN)
                clean_str = re.sub(r",\s*\]", "]", clean_str)
                # Matrix/tensor literals are shallow but legitimately large
                # (scaling-study B tensors reach ~2.6M chars), so use the
                # matrix length bound rather than the scalar default.
                return cast(
                    "list[Any] | float | int",
                    safe_literal_eval(clean_str, max_len=MATRIX_MAX_LEN),
                )
            except (ValueError, SyntaxError) as e:
                self.logger.warning(
                    f"ast.literal_eval failed for {value_str}: {e}. Falling back to manual parsing."
                )
                fallback = self._parse_nested_structure_safe(value_str)
                junk = self._find_string_tokens(fallback)
                if junk:
                    # Non-numeric tokens inside a bracketed literal are a
                    # genuine parse failure (e.g. '(0.05, 0.9, oops)'), not a
                    # tolerated string value: raise so the parameter is
                    # recorded as a structured parse_error, never silently
                    # degraded.
                    raise ValueError(
                        f"non-numeric token(s) {junk} in matrix literal"
                    ) from e
                return fallback

    @staticmethod
    def _find_string_tokens(value: Any) -> List[str]:
        """Collect string tokens that leaked into a parsed numeric structure."""
        found: List[str] = []
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(POMDPParametersMixin._find_string_tokens(item))
        return found

    def _parse_nested_structure_safe(self, value_str: str) -> List:
        """
        Robust manual parser for nested structures as a last resort.
        Handles nested parentheses/brackets by tracking depth.
        """
        value_str = value_str.strip()
        if not value_str:
            return []

        result: list[Any] = []
        current = ""
        depth = 0

        # Normalize delimiters
        value_str = value_str.replace("(", "[").replace(")", "]")

        if value_str.startswith("[") and value_str.endswith("]"):
            content = value_str[1:-1].strip()
        else:
            content = value_str

        i = 0
        while i < len(content):
            char = content[i]
            if char == "[":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    # Found a complete nested group
                    group = content[start_idx : i + 1]
                    result.append(self._parse_nested_structure_safe(group))
            elif char == "," and depth == 0:
                if current.strip():
                    try:
                        val = current.strip()
                        if "." in val:
                            result.append(float(val))
                        else:
                            result.append(int(val))
                    except ValueError:
                        result.append(current.strip())
                    current = ""
            elif depth == 0:
                current += char
            i += 1

        if current.strip():
            try:
                val = current.strip()
                if "." in val:
                    result.append(float(val))
                else:
                    result.append(int(val))
            except ValueError:
                result.append(current.strip())

        return result
