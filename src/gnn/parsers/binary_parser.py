"""
Binary Parser for GNN - Pickle

This module provides parsing capabilities for Python pickle serialization
format specifications of GNN models.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import base64
import io
import logging
import pickle  # nosec B403

# Opcode inspection does not reconstruct or execute objects.
import pickletools  # nosec B403
from typing import Any, Dict, List, cast

logger = logging.getLogger(__name__)

from .common import (
    BaseGNNParser,
    DataType,
    GNNInternalRepresentation,
    Parameter,
    ParseResult,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Restricted unpickling for untrusted .pickle input (CVE-2023-23933 class).
# Only plain data containers and the repo's own GNN model classes may be
# reconstructed; any other global (which could execute code via __reduce__)
# raises UnpicklingError and routes to the textual fallback parse.
# ---------------------------------------------------------------------------
_RESTRICTED_UNPICKLE_ALLOWLIST: Dict[str, set[str]] = {
    "builtins": {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "complex",
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "NoneType",
    },
    "datetime": {
        "datetime",
        "date",
        "time",
        "timedelta",
        "timezone",
    },
    # The parser can be imported as a package (gnn.parsers.common) or, when src/
    # is on sys.path, as a top-level module (parsers.common); both are allowed.
    "gnn.parsers.common": {
        "ASTNode",
        "Connection",
        "ConnectionType",
        "DataType",
        "Equation",
        "GNNFormat",
        "GNNInternalRepresentation",
        "OntologyMapping",
        "Parameter",
        "ParseResult",
        "TimeSpecification",
        "Variable",
        "VariableType",
    },
    "parsers.common": {
        "ASTNode",
        "Connection",
        "ConnectionType",
        "DataType",
        "Equation",
        "GNNFormat",
        "GNNInternalRepresentation",
        "OntologyMapping",
        "Parameter",
        "ParseResult",
        "TimeSpecification",
        "Variable",
        "VariableType",
    },
}


class _RestrictedUnpickler(pickle.Unpickler):  # nosec B301 - allowlist-gated
    """Unpickler that refuses any global outside the GNN safe allowlist."""

    def find_class(self, module: str, name: str) -> Any:
        """Block arbitrary class reconstruction; only allowlist is permitted."""
        if name not in _RESTRICTED_UNPICKLE_ALLOWLIST.get(module, set()):
            raise pickle.UnpicklingError(
                "GNN restricted unpickler refused to reconstruct "
                f"{module}.{name} from untrusted input"
            )
        return super().find_class(module, name)


def safe_pickle_load(file_obj: Any) -> Any:
    """Read one byte snapshot and reconstruct one restricted GNN record."""
    return safe_pickle_loads(file_obj.read())


def safe_pickle_loads(data: bytes) -> Any:
    """Reconstruct one record; reject cached extension globals and trailing data."""
    stop_position: int | None = None
    for opcode, _, position in pickletools.genops(data):
        # A cached EXT global bypasses Unpickler.find_class entirely.
        if opcode.name in {"EXT1", "EXT2", "EXT4"}:
            raise pickle.UnpicklingError("Pickle extension globals are not permitted")
        stop_position = position
    if stop_position is None or stop_position + 1 != len(data):
        raise pickle.UnpicklingError("Trailing data after GNN record pickle")
    return _RestrictedUnpickler(io.BytesIO(data)).load()


class PickleGNNParser(BaseGNNParser):
    """Parser for Python pickle binary format with enhanced round-trip support."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".pkl", ".pickle"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            # Try binary read first
            with open(file_path, "rb") as f:
                data = safe_pickle_load(f)
            return self._parse_pickle_data(data)
        except Exception as binary_error:
            # If binary fails, try reading as base64-encoded text
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                return self.parse_string(content)
            except Exception as text_error:
                result = ParseResult(model=self.create_empty_model())
                result.add_error(
                    f"Failed to read pickle file as binary: {binary_error}"
                )
                result.add_error(f"Failed to read pickle file as text: {text_error}")
                return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse base64-encoded pickle content from string."""
        try:
            # Try to decode base64 content
            binary_data = base64.b64decode(content)
            data = safe_pickle_loads(binary_data)
            return self._parse_pickle_data(data)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to decode base64 pickle data: {e}")
            return result

    def _parse_pickle_data(self, data: Any) -> ParseResult:
        """Parse pickled data into GNN representation with enhanced support."""
        result = ParseResult(model=self.create_empty_model())

        try:
            if isinstance(data, dict):
                # Check if this is our enhanced format with complete model data
                if self._is_enhanced_gnn_data(data):
                    result.model = self._reconstruct_model_from_enhanced_data(data)
                else:
                    # Previous format
                    result.model.model_name = "PickleModel"
                    self._parse_dict_data(data, result)
            elif isinstance(data, GNNInternalRepresentation):
                # Already a GNN model
                result.model = data
            elif hasattr(data, "__dict__"):
                # Object with attributes
                result.model.model_name = "PickleModel"
                self._parse_object_data(data, result)
            else:
                # Simple value
                result.model.model_name = "PickleModel"
                parameter = Parameter(
                    name="pickle_value",
                    value=data,
                    description=f"Pickled {type(data).__name__}",
                )
                result.model.parameters.append(parameter)

            if not result.model.annotation:
                result.model.annotation = "Parsed from Python pickle file"

        except Exception as e:
            result.add_error(f"Error parsing pickle data: {e}")

        return result

    def _is_enhanced_gnn_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is from our enhanced GNN serialization format."""
        required_fields: list[Any] = [
            "model_name",
            "variables",
            "connections",
            "parameters",
        ]
        return all(field in data for field in required_fields)

    def _reconstruct_model_from_enhanced_data(
        self, data: Dict[str, Any]
    ) -> GNNInternalRepresentation:
        """Reconstruct GNN model from enhanced pickle data for perfect round-trip."""
        from datetime import datetime

        from .common import (
            Connection,
            ConnectionType,
            DataType,
            Parameter,
            Variable,
            VariableType,
        )

        model = GNNInternalRepresentation(
            model_name=data.get("model_name", "PickleModel")
        )

        # Basic metadata
        model.version = data.get("version", "1.0")
        model.annotation = data.get("annotation", "")
        model.checksum = data.get("checksum", "")

        # Parse timestamps
        try:
            if "created_at" in data:
                model.created_at = datetime.fromisoformat(data["created_at"])
            if "modified_at" in data:
                model.modified_at = datetime.fromisoformat(data["modified_at"])
        except (ValueError, TypeError) as e:
            logger.debug("Timestamp parsing failed, skipping optional metadata: %s", e)

        # Reconstruct variables
        for var_data in data.get("variables", []):
            var = Variable(
                name=var_data["name"],
                var_type=VariableType(var_data.get("var_type", "hidden_state")),
                data_type=DataType(var_data.get("data_type", "categorical")),
                dimensions=var_data.get("dimensions", []),
            )
            model.variables.append(var)

        # Reconstruct connections
        for conn_data in data.get("connections", []):
            conn = Connection(
                annotation=conn_data.get("annotation"),
                source_variables=conn_data.get("source_variables", []),
                target_variables=conn_data.get("target_variables", []),
                connection_type=ConnectionType(
                    conn_data.get("connection_type", "directed")
                ),
            )
            model.connections.append(conn)

        # Reconstruct parameters
        for param_data in data.get("parameters", []):
            param = Parameter(name=param_data["name"], value=param_data["value"])
            model.parameters.append(param)

        # Reconstruct time specification
        if data.get("time_specification"):
            from .common import TimeSpecification

            time_data = data["time_specification"]
            model.time_specification = TimeSpecification(
                time_type=time_data.get("time_type", "dynamic"),
                discretization=time_data.get("discretization"),
                horizon=time_data.get("horizon"),
                step_size=time_data.get("step_size"),
            )

        # Reconstruct ontology mappings
        for mapping_data in data.get("ontology_mappings", []):
            from .common import OntologyMapping

            mapping = OntologyMapping(
                variable_name=mapping_data.get("variable_name", ""),
                ontology_term=mapping_data.get("ontology_term", ""),
                description=mapping_data.get("description"),
            )
            model.ontology_mappings.append(mapping)

        return model

    def _parse_dict_data(self, data: Dict[str, Any], result: ParseResult) -> Any:
        """Parse dictionary data."""
        # Look for common GNN structure patterns
        if "model_name" in data:
            result.model.model_name = str(data["model_name"])

        if "variables" in data:
            self._parse_variables_list(data["variables"], result)

        if "parameters" in data:
            self._parse_parameters_dict(data["parameters"], result)

        # Parse other dictionary items as parameters
        for key, value in data.items():
            if key not in ["model_name", "variables", "parameters"]:
                parameter = Parameter(
                    name=key,
                    value=value,
                    description=f"Pickled parameter: {type(value).__name__}",
                )
                result.model.parameters.append(parameter)

    def _parse_object_data(self, obj: Any, result: ParseResult) -> Any:
        """Parse object data."""
        result.model.model_name = f"{type(obj).__name__}Model"

        for attr_name in dir(obj):
            if not attr_name.startswith("_"):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        parameter = Parameter(
                            name=attr_name,
                            value=attr_value,
                            description=f"Object attribute: {type(attr_value).__name__}",
                        )
                        result.model.parameters.append(parameter)
                except (AttributeError, TypeError) as e:
                    logger.debug("Skipping non-readable attribute: %s", e)

    def _parse_variables_list(self, variables: List[Any], result: ParseResult) -> Any:
        """Parse variables from list."""
        for var_data in variables:
            if isinstance(var_data, dict):
                variable = Variable(
                    name=var_data.get("name", f"var_{len(result.model.variables)}"),
                    var_type=self._parse_variable_type(
                        var_data.get("type", "hidden_state")
                    ),
                    dimensions=var_data.get("dimensions", [1]),
                    data_type=self._parse_data_type(
                        var_data.get("data_type", "categorical")
                    ),
                    description=var_data.get("description", "Pickled variable"),
                )
                result.model.variables.append(variable)

    def _parse_parameters_dict(self, parameters: Any, result: ParseResult) -> Any:
        """Parse parameters from dictionary or list."""
        if isinstance(parameters, dict):
            for param_name, param_value in parameters.items():
                parameter = Parameter(
                    name=param_name,
                    value=param_value,
                    description=f"Pickled parameter: {type(param_value).__name__}",
                )
                result.model.parameters.append(parameter)
        elif isinstance(parameters, list):
            for param_data in parameters:
                if isinstance(param_data, dict):
                    parameter = Parameter(
                        name=param_data.get(
                            "name", f"param_{len(result.model.parameters)}"
                        ),
                        value=param_data.get("value", ""),
                        description=param_data.get("description", "Pickled parameter"),
                    )
                    result.model.parameters.append(parameter)
                else:
                    parameter = Parameter(
                        name=f"param_{len(result.model.parameters)}",
                        value=param_data,
                        description=f"Pickled parameter: {type(param_data).__name__}",
                    )
                    result.model.parameters.append(parameter)

    def _parse_variable_type(self, type_str: str) -> VariableType:
        """Parse variable type from string."""
        type_mapping: dict[str, Any] = {
            "hidden_state": VariableType.HIDDEN_STATE,
            "observation": VariableType.OBSERVATION,
            "action": VariableType.ACTION,
            "policy": VariableType.POLICY,
            "likelihood_matrix": VariableType.LIKELIHOOD_MATRIX,
            "transition_matrix": VariableType.TRANSITION_MATRIX,
            "preference_vector": VariableType.PREFERENCE_VECTOR,
            "prior_vector": VariableType.PRIOR_VECTOR,
        }
        return cast(
            "VariableType",
            type_mapping.get(type_str.lower(), VariableType.HIDDEN_STATE),
        )

    def _parse_data_type(self, type_str: str) -> DataType:
        """Parse data type from string."""
        type_mapping: dict[str, Any] = {
            "categorical": DataType.CATEGORICAL,
            "continuous": DataType.CONTINUOUS,
            "binary": DataType.BINARY,
            "integer": DataType.INTEGER,
            "float": DataType.FLOAT,
            "complex": DataType.COMPLEX,
        }
        return cast(
            "DataType", type_mapping.get(type_str.lower(), DataType.CATEGORICAL)
        )
