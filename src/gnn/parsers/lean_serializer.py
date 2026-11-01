"""GNN Lean serializer: canonical ``FEP.GnnDocument`` typed-AST emission.

Serializes GNN internal representations to the frozen 13-section typed
surface defined by ``fep_formal/src/fep_formal/formal/gnn_document.lean``
(``FEP.GnnDocument``, FEP_Formal bridge contract v0.7). Every section kind is
emitted exactly once, in canonical rank order, as a typed Lean value; the
aggregate ``document`` value assembles them into a ``GnnDocument``.

A round-trip ``-- MODEL_DATA:`` JSON payload is appended for parser round-trip
fidelity (``gnn.parsers.lean_parser`` consumes it via ``BaseGNNParser``). The
payload carries the canonical typed-payload keys (``schema_version``,
``model_family``, ``state_spaces``, ``parameterizations``,
``ontology_bindings``) plus the parser-facing keys the shared strict reconstruction
path reads (``model_name``, ``variables``, ``connections``, ``parameters``,
``equations``, ``time_specification``, ``ontology_mappings``).
"""

import json
from typing import Any, cast

from .base_serializer import BaseGNNSerializer
from .common import GNNInternalRepresentation

#: Frozen section inventory in canonical rank order (``GnnSectionKind``).
_SECTION_KINDS: tuple[str, ...] = (
    "gnnSection",
    "gnnVersionAndFlags",
    "modelName",
    "modelAnnotation",
    "stateSpaceBlock",
    "connections",
    "initialParameterization",
    "equations",
    "time",
    "actInfOntologyAnnotation",
    "modelParameters",
    "footer",
    "signature",
)

#: Variable/parameter names that mark a model as the continuous family.
#: ``F`` is deliberately absent: discrete models carry free-energy readouts.
_CONTINUOUS_MARKERS: frozenset[str] = frozenset(
    {"x", "H", "Q", "R", "priorMean", "priorCov"}
)

#: ``DataType.value`` → ``GnnValueType`` constructor (frozen three-value enum;
#: the categorical/continuous/complex GNN types are folded onto the closest
#: frozen member, documented here rather than invented in Lean).
_VALUE_TYPES: dict[str, str] = {
    "float": "floatT",
    "continuous": "floatT",
    "complex": "floatT",
    "integer": "intT",
    "categorical": "intT",
    "binary": "boolT",
}


def lean_escape(text: str) -> str:
    """Escape ``text`` for inclusion in a Lean string literal."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


class LeanSerializer(BaseGNNSerializer):
    """Serializer for the canonical ``FEP.GnnDocument`` Lean typed surface."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Emit ``model`` as a canonical ``FEP.GnnDocument`` typed document."""
        identifier = self._section_identifier(model)
        scalar_params, brace_params = self._split_parameters(model)

        sections: dict[str, str] = {
            "gnnSection": f'.gnnSection "{lean_escape(identifier)}"',
            "gnnVersionAndFlags": f".gnnVersionAndFlags {self._lean_version(model.version)} []",
            "modelName": f'.modelName "{lean_escape(model.model_name)}"',
            "modelAnnotation": f'.modelAnnotation "{lean_escape(model.annotation)}"',
            "stateSpaceBlock": f".stateSpaceBlock {self._decls(model)}",
            "connections": f".connections {self._edges(model)}",
            "initialParameterization": (
                f".initialParameterization {self._param_entries(brace_params)}"
            ),
            "equations": f'.equations "{lean_escape(self._equations_text(model))}"',
            "time": f".time {self._time_entries(model)}",
            "actInfOntologyAnnotation": (
                f".actInfOntologyAnnotation {self._bindings(model)}"
            ),
            "modelParameters": f".modelParameters {self._model_params(scalar_params)}",
            "footer": (
                '.footer "'
                + lean_escape(
                    f"GNN model {model.model_name} emitted as the canonical "
                    "FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."
                )
                + '"'
            ),
            "signature": (
                '.signature "'
                + lean_escape(
                    "serializer=gnn.parsers.lean_serializer; "
                    "contract=FEP_Formal bridge v0.7; "
                    f"model_family={self._model_family(model)}"
                )
                + '"'
            ),
        }

        lines: list[str] = []
        lines.append("-- Canonical: FEP.GnnDocument (FEP_Formal bridge v0.7)")
        lines.append(f"-- Model: {model.model_name}")
        if model.annotation:
            lines.append(f"-- {model.annotation}")
        lines.append("import FepSketches.gnn_document")
        lines.append("")
        lines.append("open FEP.GnnDocument")
        lines.append("")
        for kind in _SECTION_KINDS:
            lines.append(f"-- GnnSectionKind.{kind}")
            lines.append(f"def _{kind} : GnnSection := {sections[kind]}")
            lines.append("")
        lines.append("def document : GnnDocument :=")
        lines.append(
            "  { sections := [" + ", ".join(f"_{k}" for k in _SECTION_KINDS) + "] }"
        )
        lines.append("")

        model_data = self._model_data(model, brace_params)
        lines.append(
            "-- MODEL_DATA: "
            + json.dumps(model_data, separators=(",", ":"), ensure_ascii=False)
        )
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Typed-surface emitters
    # ------------------------------------------------------------------

    def _section_identifier(self, model: GNNInternalRepresentation) -> str:
        """Derive the ``GNNSection`` identifier from the model name."""
        allowed = "".join(c for c in model.model_name if c.isalnum() or c in "_π'")
        return allowed or "GNNModel"

    def _lean_version(self, version: str) -> str:
        """Map a GNN version string onto the frozen ``GnnVersion`` enum."""
        if version.startswith("1.1"):
            return "GnnVersion.v1_1"
        if version.startswith("1.0"):
            return "GnnVersion.v1_0"
        return "GnnVersion.v1"

    def _decls(self, model: GNNInternalRepresentation) -> str:
        """Emit ``GnnDecl`` values for the state-space block."""
        decls: list[str] = []
        for var in sorted(model.variables, key=lambda v: v.name):
            dims = ", ".join(self._dim(d) for d in var.dimensions)
            value_type = _VALUE_TYPES.get(var.data_type.value, "floatT")
            decls.append(
                "{ name := "
                + f'"{lean_escape(var.name)}"'
                + ", dims := ["
                + dims
                + "], valueType := GnnValueType."
                + value_type
                + ", defaultValue := none }"
            )
        return "[" + ", ".join(decls) + "]"

    def _dim(self, dim: Any) -> str:
        """Encode one dimension as ``GnnDim.lit`` or ``GnnDim.ref``."""
        if isinstance(dim, bool):  # bool is an int subclass; reject first
            return f"GnnDim.lit {int(dim)}"
        if isinstance(dim, int):
            return f"GnnDim.lit {dim}"
        text = str(dim)
        if text.isdigit():
            return f"GnnDim.lit {text}"
        return f'GnnDim.ref "{lean_escape(text)}"'

    def _edges(self, model: GNNInternalRepresentation) -> str:
        """Emit ``GnnConnection`` values (cross product of sources × targets)."""
        edges: list[str] = []
        for conn in model.connections:
            kind = (
                "ConnKind.undirected"
                if conn.connection_type.value == "undirected"
                else "ConnKind.directed"
            )
            label = "none"
            if conn.annotation and _valid_name(conn.annotation):
                label = f'some "{lean_escape(conn.annotation)}"'
            for src in conn.source_variables or []:
                for dst in conn.target_variables or []:
                    edges.append(
                        "{ src := "
                        + f'"{lean_escape(src)}"'
                        + ", kind := "
                        + kind
                        + ", dst := "
                        + f'"{lean_escape(dst)}"'
                        + ", label := "
                        + label
                        + " }"
                    )
        return "[" + ", ".join(edges) + "]"

    def _param_entries(self, brace_params: list[tuple[str, str]]) -> str:
        """Emit ``GnnParamEntry`` values with verbatim brace payloads."""
        entries = [
            "{ varName := "
            + f'"{lean_escape(name)}"'
            + ", payload := "
            + f'"{lean_escape(payload)}"'
            + " }"
            for name, payload in brace_params
        ]
        return "[" + ", ".join(entries) + "]"

    def _time_entries(self, model: GNNInternalRepresentation) -> str:
        """Emit ``GnnTimeEntry`` values from the time specification."""
        entries: list[str] = []
        time_spec = model.time_specification
        if time_spec is not None:
            for key, value in (
                ("time_type", getattr(time_spec, "time_type", None)),
                ("discretization", getattr(time_spec, "discretization", None)),
                ("horizon", getattr(time_spec, "horizon", None)),
                ("step_size", getattr(time_spec, "step_size", None)),
            ):
                if value is None:
                    continue
                entries.append(
                    "{ key := "
                    + f'"{key}"'
                    + ", value := some "
                    + f'"{lean_escape(str(value))}"'
                    + " }"
                )
        return "[" + ", ".join(entries) + "]"

    def _bindings(self, model: GNNInternalRepresentation) -> str:
        """Emit ``GnnBinding`` values from the ontology mappings."""
        bindings = [
            "{ varName := "
            + f'"{lean_escape(mapping.variable_name)}"'
            + ", term := "
            + f'"{lean_escape(mapping.ontology_term)}"'
            + " }"
            for mapping in model.ontology_mappings
        ]
        return "[" + ", ".join(bindings) + "]"

    def _model_params(self, scalar_params: list[tuple[str, str]]) -> str:
        """Emit ``GnnParameter`` values for scalar model parameters."""
        params = [
            "{ key := "
            + f'"{lean_escape(name)}"'
            + ", value := "
            + f'"{lean_escape(value)}"'
            + " }"
            for name, value in scalar_params
        ]
        return "[" + ", ".join(params) + "]"

    # ------------------------------------------------------------------
    # Model-family and payload helpers
    # ------------------------------------------------------------------

    def _split_parameters(
        self, model: GNNInternalRepresentation
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Split parameters into scalar (ModelParameters) and brace-shaped
        (InitialParameterization) entries."""
        scalars: list[tuple[str, str]] = []
        braces: list[tuple[str, str]] = []
        for param in model.parameters:
            text = str(param.value) if param.value is not None else ""
            if text.startswith("{") and text.endswith("}"):
                braces.append((param.name, text))
            else:
                scalars.append((param.name, text))
        return scalars, braces

    def _equations_text(self, model: GNNInternalRepresentation) -> str:
        """Join equation contents into the free-text equations payload."""
        return "\n".join(
            eq.content for eq in model.equations if getattr(eq, "content", "")
        )

    def _model_family(self, model: GNNInternalRepresentation) -> str:
        """Classify the model as the finite or continuous family.

        Deterministic marker heuristic: any variable or parameter named in
        ``_CONTINUOUS_MARKERS`` marks the continuous family; ``F`` is excluded
        because discrete models carry free-energy readouts named ``F``.
        """
        names = {var.name for var in model.variables} | {
            param.name for param in model.parameters
        }
        return "continuous" if names & _CONTINUOUS_MARKERS else "finite"

    def _model_data(
        self,
        model: GNNInternalRepresentation,
        brace_params: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Build the round-trip payload: canonical keys, then parser-facing keys."""
        data: dict[str, Any] = {
            "schema_version": 1,
            "model_family": self._model_family(model),
            "state_spaces": [
                {
                    "decl": var.name,
                    "dims": list(var.dimensions),
                    "value_type": var.data_type.value,
                }
                for var in sorted(model.variables, key=lambda v: v.name)
            ],
            "parameterizations": [
                {"var_name": name, "payload": payload} for name, payload in brace_params
            ],
            "ontology_bindings": [
                {
                    "var_name": mapping.variable_name,
                    "term": mapping.ontology_term,
                }
                for mapping in model.ontology_mappings
            ],
            # Parser-facing keys: read by the shared strict reconstruction path in
            # gnn.parsers.common.BaseGNNParser._parse_from_embedded_data.
            "model_name": model.model_name,
            "annotation": model.annotation,
            "variables": [
                {
                    "name": var.name,
                    "var_type": var.var_type.value,
                    "data_type": var.data_type.value,
                    "dimensions": list(var.dimensions),
                }
                for var in sorted(model.variables, key=lambda v: v.name)
            ],
            "connections": [
                {
                    "annotation": conn.annotation,
                    "source_variables": list(conn.source_variables),
                    "target_variables": list(conn.target_variables),
                    "connection_type": conn.connection_type.value,
                }
                for conn in model.connections
            ],
            "parameters": [
                {
                    "name": param.name,
                    "value": param.value,
                    "param_type": getattr(param, "type_hint", None) or "constant",
                }
                for param in model.parameters
            ],
            "equations": [eq.content for eq in model.equations],
            "time_specification": self._serialize_time_spec(model.time_specification)
            if model.time_specification
            else None,
            "ontology_mappings": self._serialize_ontology_mappings(
                model.ontology_mappings
            ),
        }
        return data

    # ------------------------------------------------------------------
    # Round-trip payload helpers (build the parser-facing keys)
    # ------------------------------------------------------------------

    def _serialize_time_spec(self, time_spec: Any) -> Any:
        """Serialize time specification object."""
        return {
            "time_type": getattr(time_spec, "time_type", None),
            "discretization": getattr(time_spec, "discretization", None),
            "horizon": getattr(time_spec, "horizon", None),
            "step_size": getattr(time_spec, "step_size", None),
        }

    def _serialize_ontology_mappings(self, mappings: Any) -> Any:
        """Serialize ontology mappings."""
        return [
            {
                "variable_name": mapping.variable_name,
                "ontology_term": mapping.ontology_term,
                "description": getattr(mapping, "description", None),
            }
            for mapping in (mappings or [])
        ]

    def _map_to_lean_type(self, data_type: str) -> str:
        """Map GNN data types to Lean types."""
        mapping = {
            "float": "Float",
            "continuous": "Float",
            "int": "Int",
            "integer": "Int",
            "binary": "Bool",
            "bool": "Bool",
        }
        return mapping.get(data_type, "String")


def _valid_name(text: str) -> bool:
    """Return True when ``text`` is a valid GNN name (syntax doc §2)."""
    return bool(text) and all(c.isalnum() or c in "_π'" for c in text)
