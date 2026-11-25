"""Single canonical GNN section contract (W2-01).

One home for the GNN section universe and the required/optional
classification. Consumers:

- ``schemas/json.json`` / ``schemas/yaml.yaml`` formal artifacts — pinned to
  this table by ``tests/gnn/test_section_contract.py``; edit the table, not
  the schema files.
- ``schema_validator.GNNValidator`` markdown-structure gate (compliance).
- ``validation.mcp.check_schema_compliance_mcp`` (compliance).
- ``type_checker`` section-presence maps and report renderers (display).

Two gate levels exist by design:

- *Compliance level* — ``REQUIRED_SECTIONS`` below: the normative GNN v1
  contract, i.e. the nine sections the formal schema artifacts require.
- *Parse level* — ``gnn.schema.parser`` keeps a deliberately lighter
  five-section gate (``GNN-E001``) so structurally incomplete documents can
  still be parsed and reported on rather than rejected outright.
"""

# Canonical GNN section headers, in declared order. Used to build
# section-presence maps and to drive section-aware parsing.
CANONICAL_GNN_SECTIONS: tuple[str, ...] = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "BoundaryAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)

# Normative required sections (compliance level), in declared order —
# mirrors the ``required`` list in schemas/json.json and
# ``required_sections`` in schemas/yaml.yaml.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Time",
    "Footer",
)

# Recognised-but-optional sections (compliance level) — mirrors
# ``optional_sections`` in schemas/yaml.yaml.
OPTIONAL_SECTIONS: tuple[str, ...] = (
    "ImageFromPaper",
    "BoundaryAnnotation",
    "Equations",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Signature",
)

# Section header name -> snake_case property key used by schemas/json.json.
JSON_SECTION_KEYS: dict[str, str] = {
    "GNNSection": "gnn_section",
    "GNNVersionAndFlags": "gnn_version_and_flags",
    "ModelName": "model_name",
    "ModelAnnotation": "model_annotation",
    "BoundaryAnnotation": "boundary_annotation",
    "StateSpaceBlock": "state_space_block",
    "Connections": "connections",
    "InitialParameterization": "initial_parameterization",
    "Time": "time",
    "Footer": "footer",
    "Equations": "equations",
    "ActInfOntologyAnnotation": "act_inf_ontology_annotation",
    "ModelParameters": "model_parameters",
    "Signature": "signature",
    "ImageFromPaper": "image_from_paper",
}
