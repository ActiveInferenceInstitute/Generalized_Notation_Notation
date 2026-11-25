# ADR 0050 — Optional BoundaryAnnotation

**Status** Proposed  
**Date** 2026-11-15

NEST's MESSAGEix and STURM wrappers are the first intended consumers. Their
GNN files can declare the meaning and shape of their own inputs and outputs.
A separate versioned seam contract can refer to these declarations or to
another model interface. The seam carries connections and translation rules,
ownership, solve order, convergence, and the uncertainty of exchanged values.
The records for individual runs carry evidence and provenance.

## Decision

Add an optional Markdown section and a corresponding structured representation
keyed by variable name. Each variable already exists in the same model's
`StateSpaceBlock`. A declaration uses one line per variable. Fields are
separated by semicolons and support spaces inside values. For example, a
synthetic temperature model can declare:

```markdown
## BoundaryAnnotation
temperature: unit=K; convention=thermodynamic temperature; support=cell x step; form=point
flux: unit=?; convention.reference=as received; support=face x step; form=interval
```

`unit`, `convention`, `support`, and `form` describe declared model
semantics. `?` marks an undeclared field. A field may also be omitted.
Conventions can use dotted keys for named aspects. `form` is one of
`point`, `range`, `interval`, `distribution`, `samples`, or `?`. Preserve
and report unknown fields. Reject duplicate variables or fields, unsupported
forms, and references to variables absent from the model. Unit text is
preserved verbatim in substance. A unit grammar and conversion rules require
a separate decision, including currency year and calendar conventions.

## Verification

The implementation should preserve existing models and demonstrate
round trips in Markdown, JSON, and YAML. The XML, Protobuf, ASN.1, and Pkl
schema files describe the optional declarations. The registered exporters
guard against losing them in formats without round-trip support. The
consumer handoff requires validation of both wrappers at a documented GNN
commit. Use synthetic tests in GNN. Existing NEST sidecars remain
authoritative through a reviewed migration.

[ADR 048](048-composition-seam.md) provides a precedent for naming a
consumer and landing implementation, tests, and documentation together.
