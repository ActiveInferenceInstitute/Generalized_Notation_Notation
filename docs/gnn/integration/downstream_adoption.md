# Downstream adoption: using GNN in a sibling repository

**Status**: Maintained guide · **Last reviewed**: 2026-11-01
**Scope**: how a sibling repository makes GNN the specification of its Active
Inference generative models, and what each adoption pattern proves.

This guide describes the patterns sibling repositories actually use. The
authority for *which* GNN symbols a sibling may import is the
[composability standard](../../standards/composability.md#consumer-seams-declared-cross-repo-surfaces);
this guide adds the adoption mechanics that the standard deliberately leaves to
each consumer.

---

## 1  What a sibling repository may consume

Everything in `src/gnn/` outside the five declared surfaces is private to this
repository. The surfaces a sibling uses in practice are:

| Need | Surface | Form |
| --- | --- | --- |
| Parse a model into matrices | Runtime-bridge surface | `gnn.extract.pomdp_extractor.extract_pomdp_from_file`, `gnn.schema.parse_state_space` |
| Parse a model into sections/variables | Parser entry point | `gnn.parsers.markdown_parser.MarkdownGNNParser` |
| Exchange a model as data | Export surface | `gnn-geo-infer/1`, `gnn-geo-infer/2`, `gnn-geo-infer/factored/1`, Gaussian variants |
| Author against the language | Syntax surface | [`docs/gnn/gnn_syntax.md`](../gnn_syntax.md) |
| Verify custody | Pin files | the consumer's own pin file, see §3 |

Do not reach into `gnn.render`, `gnn.pipeline` or `gnn.main` internals from a
sibling. Rendering is a pipeline step with its own output contract; a sibling
that needs rendered code should consume the artifact, not the renderer.

## 2  Four adoption patterns

Each pattern below is in use in the sibling repositories of this ecosystem. They
differ in what they prove, not in quality.

### A  Build-time compile

The repository authors the model once as a GNN file, then a build step parses it
with the upstream parser and emits a language-native artifact (JSON or source).
The emitted artifact records the specification digest and the upstream revision,
so a stale artifact is detectable.

*Proves*: the shipped runtime model descends from the committed GNN source.
*Cost*: a build step and a checked-in artifact.
*Fits*: repositories whose runtime is not Python, or that must ship without a
Python environment.

### B  Runtime import

The repository injects the pinned GNN checkout onto `sys.path` and imports the
runtime-bridge surface, constructing its runtime model from the parsed state
space.

*Proves*: the running model is read from the GNN file at run time.
*Cost*: a provisioned GNN checkout and a resolved interpreter.
*Fits*: Python repositories that can own a pinned dependency and an admission
step.

### C  Artifact interchange

GNN exports the model as JSON under a versioned contract; the consumer validates
that contract and runs the model. Neither environment imports the other's source.

*Proves*: deterministic replay of a versioned artifact between two installed
repositories.
*Cost*: two pinned revisions, one per side, plus a paired check.
*Fits*: repositories with independent toolchains, or a language boundary.

### D  Structure-only declaration

When the tensors are estimated from data rather than authored, the GNN file owns
the *structural* contract — factor names and cardinalities, observation
modalities and their dependencies, control sizes, the conditional-independence
flag, and the estimation/tuning contracts — while the code owns the estimates.
The GNN file carries no invented matrices.

*Proves*: the model's declared structure is reviewable in one place and cannot
drift from the code.
*Cost*: one conformance test (see §4).
*Fits*: learning-based and data-fitted models, where an authored matrix would be
a fabricated source of truth.

Patterns compose: a repository may compile structure from a GNN file (D + A) and
still export artifacts to a downstream consumer (C).

## 3  Pin discipline

A sibling that consumes GNN records exactly one pin, in a committed file, with:

- the upstream repository URL,
- the exact 40-hex commit revision that was reviewed,
- the upstream version string from that revision's `pyproject.toml`,
- the date the pin was recorded.

One concrete shape for that file:

```json
{
  "repository": "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation",
  "commit": "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1",
  "version": "5.0.3",
  "date_recorded": "2026-11-01",
  "tag": "v5.0.3",
  "parser_module": "gnn.parsers.system",
  "parser_class": "GNNParsingSystem"
}
```

The four core keys are what a consumer's check reads; extra keys below them
(`tag`, `parser_module` and `parser_class` here are optional convenience keys)
are allowed, and divergent-but-visible pins between siblings are expected.

Rules that follow from the custody contract:

- A pin names a revision that **resolves in the upstream repository**. A pin
  that exists only on an unreviewed local branch is not a pin.
- Re-pinning is a paired change: the consumer bumps its pin in the same change
  set that reviews the upstream difference, never by relaxing the check.
- A red paired or freshness check is the drift signal. Repair it by re-pinning,
  never by suppressing the check.
- Different consumers pin independently. Divergent pins between siblings are
  expected; a *silent* divergence is not.

## 4  The drift guard

Two checks carry the weight of an adoption:

1. **Conformance (required).** A test that asserts the GNN-derived declaration
   equals the in-code declaration field by field — factor names, sizes,
   dependency tuples, control sizes. This is the load-bearing test: it fails the
   moment either side is edited alone.
2. **Upstream validation (required where a checkout is available).** A test that
   runs the real GNN validator against the committed model files. It may skip
   only with an explicit, printed reason when the pinned checkout is absent; it
   must never fail the consumer's suite merely because a sibling repository is
   not provisioned.

A conformance test that only compares a file against itself proves nothing. The
comparison must cross the boundary between the GNN declaration and the code that
implements it.

## 5  What adoption does not prove

- A green conformance test proves the declaration and the implementation agree.
  It does not prove the model is scientifically adequate, that its tensors are
  correctly estimated, or that any inference result is correct.
- A green upstream-validator run proves the file is well-formed GNN. It does not
  prove the model's semantics.
- A pin proves the reviewed revision is the one recorded. It does not prove the
  consumer's runtime used it.

Keep these planes separate in reports, exactly as the
[evidence-plane discipline](../../standards/composability.md#evidence-plane-discipline)
requires inside this repository.

## Related

- [Composability standard: declared cross-repo consumer surfaces](../../standards/composability.md#consumer-seams-declared-cross-repo-surfaces)
- [Paired-revision procedure](../../development/fep_lean_paired_revision.md)
- [GEO_INFER interchange contract](../../../src/gnn/export/geo_infer_contract.md)
- [GNN syntax specification](../gnn_syntax.md)
