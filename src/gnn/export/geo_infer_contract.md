# GNN / GEO-INFER artifact contract version 1

GNN owns this format's producer and notation semantics. GEO-INFER owns the
consumer validator and spatial/temporal inference runtime. Exchange JSON data
between independently installed repositories; neither environment imports the
other repository's source. `geo_infer` is an opt-in export registry format,
currently supported by `export_model` and the standalone CLI. The five default
Step 7 exports retain their existing behavior.

## Required structure

Every object below has exactly the listed keys. Unknown keys or versions fail
on the consumer. Top-level `schema_version` is `gnn-geo-infer/1`, `model_type`
is `categorical`, and `model_name` is a nonempty string. The current paired consumer accepts the explicit-metadata schemas
`gnn-geo-infer/1`, `gnn-geo-infer/2` and `gnn-geo-infer/factored/1`.
Producer-side notation derivation can attach `provenance.metadata_derivation`
after base validation, but the current consumer rejects that added key. See
*Notation-derived metadata* for this separate opt-in boundary. Unknown keys
are never silently stripped to make an artifact pass.

| Object | Required keys and interpretation |
| --- | --- |
| dimensions | `states`, `observations`, `actions`: positive integers |
| matrices | `A`, `B`, `C`, `D`, `E`: finite numeric arrays |
| space | `kind`: `categorical` or `h3`; `state_ids`: unique strings in matrix state order |
| time | `step_seconds`: positive seconds per B transition, explicitly supplied by the caller or read from an explicit notation declaration with recorded provenance |
| provenance | `producer`: nonempty label; `source_sha256`: lowercase SHA-256 of original UTF-8 source; producer-only optional `metadata_derivation`: rejected by the current paired consumer |

A axes are `[observation,state]`; B axes are `[next_state,current_state,action]`.
C contains log preferences, D is the initial state prior, and E is the prior over
one-step policies in action order. A/B/D/E sum to one along axis zero (absolute
tolerance 1e-8), with nonnegative entries. C has one value per observation; D one
per state; E one per action. Matrices are not normalized, resized or substituted
on export. All five must be explicit in `InitialParameterization`.

Only one state factor, observation modality and control factor are accepted.
Noncanonical, contradictory or numerically unverifiable B axis provenance
fails; doubly-stochastic (or otherwise ambiguous) data is never reordered on
the strength of prose declarations alone, and users must explicitly resolve
the source model. The general `canonicalize_pomdp` helper separately
supports action-first B conversion and maintains its resulting orientation
metadata, including repeated-call idempotence and unequal action/state counts.

H3 IDs must be canonical cells at one resolution, and must be supplied in the
existing matrix order. Labels do not change matrix axes. H3 validation requires
the optional `geo-infer` dependency group. Categorical IDs default to numeric
strings; they make no geographic claim.

## Producer usage

```bash
uv sync --extra dev --extra geo-infer
PYTHONPATH=src uv run --no-sync python -m gnn.export.geo_infer \
  input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md \
  /tmp/gridworld.geo-infer.json --step-seconds 60
```

For spatial states, add `--space-kind h3 --state-ids /path/to/ordered_cells.json`.
The IDs file must contain a JSON list. The exporter never guesses seconds or
CRS: seconds come from the caller, or — when the caller passes
`--geo-derive-metadata` — from explicit notation declarations recorded in
provenance (see *Notation-derived metadata*). Input source and IDs files are
bounded to four MiB each; the dense matrix entry budget is 1,000,000.

```python
from pathlib import Path
from gnn.export.processor import export_model

result = export_model({
    'raw_content': Path('model.md').read_text(),
    'geo_infer': {'step_seconds': 60},
}, Path('/tmp/exports'), formats=['geo_infer'])
assert result['success'], result
```

## Consumer timing and verification

GEO-INFER-ACT's `GNNArtifact` loads bounded JSON (32 MiB) and rejects duplicate
keys. `run_gnn_inference` accepts records containing `timestamp` and an integer
`observation` index. D is the prior at the first timestamp. Each timestamp
conditions once; the selected action produces the next prior through B. TIME
requires UTC-convertible aware timestamps and fixed intervals, rejecting gaps,
reversals and duplicates. Default execution budget: 10,000 steps. Zero-likelihood
observations fail. E reaches real pymdp 1.0.3 policy inference.

Artifact digests use sorted compact Python JSON, UTF-8, without a trailing
newline; source digests identify bytes and do not authenticate the producer.
This format does not claim RFC 8785 canonicalization.

Run `tests/export/` and `tests/gnn/test_pomdp_extractor*` locally. GEO's
`GEO-INFER-TEST/validate_gnn_interchange.py` takes explicit GNN checkout and Python
paths, checks the real gridworld and a SPACE-generated seven-cell H3 model, and
verifies exact matrix/order preservation and deterministic real inference.

Continuous GNN, factorized models, irregular time and longer policies are
covered by the separate contracts below ("Linear Gaussian v2" and the factored
contract); v1 itself does not accept them. Historical GNN-02/GNN-03/GNN-05
identifiers tracked these contracts and Step 7 wiring. Current remaining scope
is in [TO-DO](../../../TO-DO.md); exact pairing follows the
[paired-revision procedure](../../../docs/development/fep_lean_paired_revision.md).

## Historical companion implementation checkpoint

The paired GEO implementation is
[`e028aa90`](https://github.com/ActiveInferenceInstitute/GEO_INFER/commit/e028aa9060e05f765762224499f5e2c714cf25a3).
Its receipt records identical categorical/H3 artifacts and complete traces across
GEO Python 3.11 and 3.12, with GNN running separately on Python 3.11. This identifies the tested topic epoch, not a current pin or 4.1.0 acceptance.
Historical published pairing is recorded by the
[4.0.1 receipt](../../../docs/development/gnn_4_0_1_post_publication.json);
current pairing belongs to the version's exact source/check/release records.

## Linear Gaussian v2

`geo_infer_gaussian.build_geo_infer_gaussian_artifact` exports the explicit
`gnn-geo-infer/2`, `linear_gaussian` model type. The categorical v1 contract above
remains unchanged. The GNN source must explicitly declare `Discrete` in `Time`;
continuous state is supported, while continuous-time generators require a separate
explicit discretization before export. No Euler approximation is performed here.

| Field | Required value or shape |
| --- | --- |
| `dimensions` | Positive integer `states=n`, `observations=m`, `controls=k` |
| `matrices.F` | State transition, `[next_state, previous_state]`, shape `(n,n)` |
| `matrices.G` | Control map, `[next_state, control]`, shape `(n,k)` |
| `matrices.H` | Observation map, `[observation, state]`, shape `(m,n)` |
| `matrices.Q` | Per-interval process covariance `(n,n)`, symmetric positive semidefinite |
| `matrices.R` | Observation covariance `(m,m)`, symmetric positive definite |
| `initial_belief` | `mean` `(n,)` and positive definite `covariance` `(n,n)` |
| `units` | `states`, `observations`, `controls`: one nonempty unit string per coordinate |
| `time` | Exactly `domain="discrete"`, positive finite `step_seconds` |
| `provenance` | Producer string and SHA-256 of the exact UTF-8 GNN source |

Every matrix and initial belief must occur explicitly in GNN
`InitialParameterization` as `F`, `G`, `H`, `Q`, `R`, `prior_mean`, `prior_cov`.
Units describe the coordinates; matrix units follow their row/column ratios and
covariance units follow coordinate products. No unit conversion is implicit.
The artifact accepts only the listed top-level fields plus `schema_version`,
`model_type`, and nonempty `model_name`. JSON numbers must be finite and cannot
be booleans. Dense storage is bounded to one million entries. GEO additionally
bounds JSON input to 32 MiB and rejects duplicate keys.

```bash
PYTHONPATH=src uv run python -m gnn.export.geo_infer \
  tests/export/gaussian_rectangular.md gaussian.geo-infer.json \
  --model-type linear_gaussian --step-seconds 2 --units units.json
```

For this fixture, `units.json` contains
`{"states":["m","m/s","K"],"observations":["m","m/s"],"controls":["N"]}`.
The three state coordinates, two observations and one control provide a
non-square axis regression fixture.

With `--geo-derive-metadata` / `geo_derive_metadata=True`, `step_seconds` may
come from an explicit notation declaration (`TimeStep`, `StepSeconds` or `dt`)
instead of the command line; explicit values win and the derivation is
recorded in `provenance.metadata_derivation`.

GEO's `GaussianGNNArtifact` loads the artifact. Its
`run_gaussian_gnn_inference` accepts records with exactly `timestamp`,
`observation` and `control`. Timestamps are aware and regularly spaced by
`step_seconds`; observations and controls are explicit numeric vectors.
The initial belief applies at the first measurement, which is conditioned once.
The record's control then predicts the next prior using `F @ mean + G @ control`
and `F @ covariance @ F.T + Q`. The trace includes prior, posterior, next prior,
and negative log evidence. The runner performs filtering with supplied controls;
it does not synthesize an active-inference policy or label supplied controls as
optimal actions.

## Explicit Step 7 metadata

`process_export(..., formats=["geo_infer"], geo_infer_options=...)` supports
per-source options keyed by exact `file_name` in the Step 3 manifest:

```python
process_export(
    target_dir, output_dir,
    formats=["json", "geo_infer"],
    geo_infer_options={
        "navigation.md": {
            "model_type": "linear_gaussian",
            "step_seconds": 2,
            "units": {
                "states": ["m", "m/s", "K"],
                "observations": ["m", "m/s"], "controls": ["N"],
            },
        },
        "gridworld.md": {"step_seconds": 60},
    },
)
```

The interchange writer reads bounded original source bytes under `target_dir`,
checks resolved containment (including symlinks), and computes source provenance
from those bytes. It does not reconstruct source from the Step 3 summary. Missing
per-model options produce a failed format entry in `export_results.json`. A run
with any model export failure returns `False`, even when another model succeeds.
The default format set remains JSON, XML, GraphML, GEXF and pickle. No physical
step, Gaussian units, or H3 identities are inferred from filenames or examples.
A `metadata_derivation` key inside an explicit options mapping is stripped
before export: only the notation-derivation layer may record derivation
provenance, so explicit JSON cannot claim it.

Gaussian Markdown uses one explicit `StateSpaceBlock` with continuous `x`, `y`
and `u` coordinate declarations matching state, observation and control sizes.
Present matrix and prior declarations must match their literal values. Duplicate
semantic sections, additional unsupported coordinates and contradictory shapes
are rejected. The numeric covariance check rejects nonfinite eigenspectra and
uses overflow-safe symmetrization before checking definiteness.

For recursive Step 3 input, metadata keys may be contained relative paths such as
`first/navigation.md`. Basename lookup is accepted only when that basename is
unique in the manifest. Source identity comes from the contained Step 3
`file_path`, whose basename must match `file_name`; nested output directories
preserve distinct models with equal basenames. Existing output-directory,
artifact-file and manifest-file symlinks cannot redirect writes outside the
requested output directory.

## Notation-derived metadata (opt-in)

The producer can read metadata from explicit declarations inside the GNN
source itself, but only when the
caller opts in, only for fields the explicit options leave unset, and always
with the derivation recorded in the artifact. The concern behind the original
rule was silent guessing; the revision keeps every value either
caller-explicit or notation-explicit with provenance.

Step 7 accepts `--geo-derive-metadata`; `process_export` accepts
`geo_derive_metadata=True`. The resolution order per model is:

1. per-model explicit options (`geo_infer_options`, keyed by the Step 3
   manifest entry);
2. the global CLI `geo_infer` mapping (`--geo-step-seconds` et al.);
3. notation derivation (`gnn.export.notation_metadata.derive_geo_metadata`)
   as the last fallback before the "explicit geo_infer_options required"
   failure.

Derivation is per field: explicit options always win. When explicit options
carry `step_seconds`, nothing is derived and the artifact provenance stays
clean. When explicit options lack `step_seconds` (for example a
`linear_gaussian` entry that declares only `model_type` and `units`),
derivation fills the missing field and `provenance.metadata_derivation`
records both the derived value and the `explicit_overrides` it respected.

Accepted declarations, all read from the bounded original source:

| Field | Notation surface | Notes |
| --- | --- | --- |
| `step_seconds` | `## Time` entries `TimeStep=60s`, `StepSeconds=60`, `dt=0.5`; or `## ModelParameters` entry `dt: 0.1` | Bare numbers are read as seconds and the interpretation is recorded. Unit suffixes `s`/`ms`/`min`/`h`/`days` and a `TimeUnits=<unit>` companion key are accepted. |
| `space_kind` | `## Time` entry `SpaceKind=categorical\|h3` | Absent declarations resolve to the categorical writer default, recorded as non-derived. `h3` still requires explicit `state_ids`; derivation never fabricates H3 cells. |
| `units` | none | Never derived: the notation declares no per-coordinate unit surface. Supply units explicitly for `linear_gaussian` exports. |
| time index | `Time=<var>` / `DiscreteTime=<var>`, the `StateSpaceBlock` declaration of `<var>`, and its `ActInfOntologyAnnotation` term | Corroboration only; recorded in provenance. |

Derivation fails visibly instead of guessing: a `NotationMetadataError` (a
`ValueError`) names every section and key searched when no time step is
declared, rejects conflicting or ambiguous declarations (including a
`## Time`/`## ModelParameters` disagreement), rejects a `Continuous` time
declaration, and rejects nonpositive, unparseable or unsupported-unit values.
No default `step_seconds` is ever substituted. Gaussian exports additionally
keep their own `Discrete`-without-`Continuous` requirement.

`provenance.metadata_derivation` is a producer-side extension attached after
base artifact validation. It does not change the current consumer's exact-key
contract or establish paired-consumer acceptance:

```json
"provenance": {
  "producer": "GNN strict POMDP extractor",
  "source_sha256": "…",
  "metadata_derivation": {
    "step_seconds": {
      "value": 60.0,
      "derived": true,
      "sources": [
        {"section": "Time", "key": "timestep", "raw_value": "60s",
         "interpretation": "60 s = 60 seconds"}
      ]
    },
    "space_kind": {"value": "categorical", "derived": false,
                   "source": "writer default"},
    "units": {"derived": false, "reason": "…"},
    "time_index": {"variable": "t",
                   "state_space_declaration": "t[1,type=int]",
                   "ontology_term": "Time"},
    "discrete_time_declared": true
  }
}
```

The current paired GEO consumer rejects unknown provenance sub-keys, including
this record. Use explicit metadata with the three current schemas. Exchanging
derived provenance requires a separately reviewed consumer contract and a
paired revision that actually accepts it; the producer flag alone does not
establish readiness. Never drop provenance silently to bypass refusal.

## Numbered Step 7 command

After Step 3, select the interchange format and explicit metadata file:

```bash
PYTHONPATH=src uv run --extra dev --extra geo-infer python src/gnn/7_export.py \
  --target-dir input/gnn_files --output-dir output \
  --formats geo_infer --geo-infer-options-file geo-options.json
```

With `--geo-derive-metadata`, a partial `geo-options.json` (for example
`{"continuous_navigation.md": {"model_type": "linear_gaussian", "units":
{"states": ["m", "m"], "observations": ["m", "m"], "controls": ["N", "N"]}}}`)
lets the exporter fill `step_seconds` from the notation (`dt: 0.1`) and record
the derivation in the artifact. Without the flag the behavior is unchanged.
Passing the flag without any options file exports every model whose notation
declares a step; models without declarations fail visibly with the searched
keys listed.

`geo-options.json` maps relative source paths to the same `geo_infer` options
shown above. Duplicate JSON keys are rejected. Nested source directories retain
their relative identity; basename metadata is accepted only when unambiguous.
The default invocation still emits the original five formats. These flags also
pass through the main pipeline argument registry. A requested export with absent
metadata returns failure, while retaining per-model diagnostics.
