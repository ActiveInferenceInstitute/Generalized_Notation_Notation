# GNN–FEP workflow group: one pin reader for every consumer

This contract binds the twenty-two repositories that apply Generalized Notation
Notation (GNN) to one implementation of three concerns: reading an upstream pin,
validating it (together with the `.github/GNN.md` pointer), and validating the
`*.gnn.md` specifications under it. The reader is `src/gnn/importers.py` in this
repository. A member calls it; a member does not carry a second copy.

The machine-readable form of this page is the `workflow_groups."gnn-fep"` block
of [importers.json](../downstream/importers.json). `scripts/check_importer_registry.py`
keeps the two forms one artifact in the same way it already keeps the downstream
tables one artifact, and `./run.sh gates` runs that check in its `docs` arm.

## What this contract decides

1. **Pin parsing has one home.** `gnn.importers.read_pin` normalizes every pin
   shape the estate uses: `commit` or `revision`, `tag` or `release`,
   `repository` as `org/name` or URL, optional `spec_files`, seven candidate
   locations, and a party-qualified sub-record for a two-party custody pin.
2. **Pin and pointer validation have one home.** `check_importer` reports
   findings with stable codes; `check_pointer` additionally requires
   `.github/GNN.md` to name the pin file, the pinned revision, the conformance
   workflow and the job key.
3. **Specification validation has one home.** `check_importer` parses each
   declared or discovered `*.gnn.md` with the parser installed in the pinned
   revision — the same parser the pin names.

## Members

Every member runs the same command. The `gate` column is the member's own front
door, which the arm is added to; it is never replaced by this contract.

<!-- gnn-fep-members:start -->
| Repository | Role | Gate | Docs index | Manuscript producer | GNN pin | Reader arm |
| --- | --- | --- | --- | --- | --- | --- |
| [Generalized_Notation_Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation) | upstream | `./run.sh gates` | `docs/README.md` and `docs/INDEX.md` | `scripts/z_generate_manuscript_variables.py` | none: this repository is the upstream | `python -m gnn.importers check --repo .` |
| [fep_formal](https://github.com/ActiveInferenceInstitute/fep_formal) | importer | `bash scripts/gate.sh gnn` | `docs/README.md` | none | `specs/gnn-bridge-w2-source-custody/source-pin.json`, party `gnn` | `python -m gnn.importers check --repo .` |
| [pymdp](https://github.com/ActiveInferenceInstitute/pymdp) | importer | none: no gate front door in the checkout | none | none | none in a file: `pymdp/gnn.py` constants | `python -m gnn.importers check --repo .` |
| [pymcp](https://github.com/ActiveInferenceInstitute/pymcp) | importer | `./run.sh all` | `docs/README.md` and `docs/documentation-index.md` | none | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [Active_InferAnts](https://github.com/ActiveInferenceInstitute/Active_InferAnts) | importer | `./run.sh gate` | `docs/README.md` and `docs/documentation-index.md` | none | `gnn/upstream.json` | `python -m gnn.importers check --repo .` |
| [ants](https://github.com/ActiveInferenceInstitute/ants) | importer | `python scripts/gate.py all` | `docs/README.md` and `docs/documentation-index.md` | none | `gnn/upstream.json` | `python -m gnn.importers check --repo .` |
| [ActInf_RxInfer](https://github.com/ActiveInferenceInstitute/ActInf_RxInfer) | importer | `./run.sh all` | `docs/README.md` and `docs/documentation-index.md` | none | `specs/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [ActiveBlockference](https://github.com/ActiveInferenceInstitute/ActiveBlockference) | importer | `uv run python scripts/release_check.py --arm all` | `docs/README.md` and `docs/documentation-index.md` | `scripts/manuscript_variables.py` | `specs/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [Active_Fedference](https://github.com/ActiveInferenceInstitute/Active_Fedference) | importer | `python scripts/gate.py` | `docs/README.md` | `scripts/z_generate_manuscript_variables.py` | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [Active_Cardference](https://github.com/ActiveInferenceInstitute/Active_Cardference) | importer | `./run.sh` | `docs/README.md` and `docs/documentation-index.md` | `scripts/z_generate_manuscript_variables.py` | `specs/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [active_fractal_rabbit](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit) | importer | `./run.sh all` | `docs/README.md` and `docs/documentation-index.md` | `scripts/z_generate_manuscript_variables.py` | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [policy_entanglement](https://github.com/ActiveInferenceInstitute/policy_entanglement) | importer | `make ci` | `docs/README.md` | `scripts/manuscript_variables.py` | `gnn/upstream.json` | `python -m gnn.importers check --repo .` |
| [on_policy_distillation](https://github.com/ActiveInferenceInstitute/on_policy_distillation) | importer | `python scripts/gate.py` | `docs/README.md` | `scripts/z_generate_manuscript_variables.py` | `gnn/upstream.json` | `python -m gnn.importers check --repo .` |
| [active_inference_battleship](https://github.com/ActiveInferenceInstitute/active_inference_battleship) | importer | `bun run check` | `docs/README.md` and `docs/documentation-index.md` | `research/variables.py` | `models/gnn.pin.json` | `python -m gnn.importers check --repo .` from `scripts/parse_gnn.py` |
| [active_inference_power](https://github.com/ActiveInferenceInstitute/active_inference_power) | importer | `./run.sh` | `docs/README.md` and `docs/documentation-index.md` | `scripts/z_generate_manuscript_variables.py` | `models/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [active_inference_language_model](https://github.com/ActiveInferenceInstitute/active_inference_language_model) | importer | `scripts/verify_release.py full` | `docs/README.md` and `docs/documentation-index.md` | `scripts/z_generate_manuscript_variables.py` | `gnn/upstream.json` | `python -m gnn.importers check --repo .` |
| [COGANT](https://github.com/ActiveInferenceInstitute/COGANT) | importer | `python run_gate.py` | `docs/README.md` | `scripts/z_generate_manuscript_variables.py` | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [CEREBRUM](https://github.com/ActiveInferenceInstitute/CEREBRUM) | importer | `python scripts/gate.py all` | `docs/README.md` and `docs/documentation-index.md` | none | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [Biofirm](https://github.com/ActiveInferenceInstitute/Biofirm) | importer | `./run.sh gate` | `docs/README.md` | none | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [GEO_INFER](https://github.com/ActiveInferenceInstitute/GEO_INFER) | importer | `python scripts/gate.py gates` | `docs/README.md` and `docs/INDEX.md` | `scripts/z_generate_manuscript_variables.py` | `.github/gnn-pair.json` | `python -m gnn.importers check --repo .` |
| [AEOS](https://github.com/ActiveInferenceInstitute/AEOS) | importer | `./run.sh gate` | `docs/README.md` and `docs/documentation-index.md` | none | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
| [Active_Inference_Ontology](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology) | importer | `./run.sh all` | `docs/README.md` and `docs/documentation-index.md` | none | `.github/gnn-pin.json` | `python -m gnn.importers check --repo .` |
<!-- gnn-fep-members:end -->

Two members have no gate front door to add an arm to (`pymdp`), or a gate in a
language that cannot import the reader (`active_inference_battleship`). Both are
recorded in the JSON with `install` and, for the second, a `wrapper` file and a
`reason`; every other member installs nothing and calls the reader directly.

## Why the canonical form is a direct call, not a vendored copy

A member's `gnn` arm already materializes the pinned revision. The evidence is in
the checkouts, not in a claim: `pymcp/.gnn-venv/`, `AEOS/.venv-gnn/`,
`COGANT/cogant/.venv-gnn/` and `ActInf_RxInfer/.gate/gnn-runtime/` each hold an
installed `gnn` package at the pinned commit. So `python -m gnn.importers` runs
inside the environment the arm has already built, against the parser the pin
names.

A vendored copy would therefore add, not remove, a surface: a second parser, a
second candidate list, and a digest to keep in step with this file. The rule is
one reader, called from the environment that already exists.

The exception is a gate that cannot import Python. `active_inference_battleship`
runs under Bun; its recorded wrapper (`scripts/parse_gnn.py`) shells out to the
same command instead of re-implementing the pin shape in TypeScript.

## The reader is not for this repository's own tree

`find_pin` treats `.github/gnn-pair.json` as a GNN pin, because for `GEO_INFER`
it is one. This repository's `.github/gnn-pair.json` points the other way: it
pins `GEO_INFER` at `21b65a26`, and `.github/fep-lean-pair.json` pins
`fep_formal` at `48206d6d`. Discovery would read either as a GNN pin and report a
fork. That is why `check_importer` accepts `pin_path`, and why the upstream
member records no pin: the arm is for importers, and the upstream's own pins are
*dependencies it consumes*, not a claim about which GNN revision it is.

## Shared concerns and their canonical homes

| Concern | Canonical home | Copies measured in the group |
| --- | --- | --- |
| Pin discovery and shape normalization | `src/gnn/importers.py` (`PIN_CANDIDATES`, `read_pin`) | 21 members; see the table below |
| Pin and pointer validation | `src/gnn/importers.py` (`check_importer`, `check_pointer`) | `AEOS/scripts/check_gnn.py:29` declares its own `POINTER_PATH`; `GEO_INFER/GEO-INFER-TEST/tests/unit/test_gate_contracts.py:25` repeats the pointer path and asserts its content at `:192` |
| Specification validation | `src/gnn/importers.py` (`_validate_spec` → `gnn.parsers.basic.validate_gnn_syntax`) | `pymcp/scripts/gnn_conformance.py:232` and `AEOS/scripts/check_gnn.py:159` import the validator and repeat the call-and-report shape |
| Estate registry | `docs/downstream/importers.json` + `scripts/check_importer_registry.py` | one copy, in this repository; no member mirrors it |
| Docs link and index coverage | this repository's `docs/development/docs_audit.py` | `pymcp/scripts/check_docs.py:27`, `ActInf_RxInfer/scripts/check_links.py:37`, `AEOS/scripts/check_docs.py:19`, `Active_Inference_Ontology/scripts/check_docs.py:29`, `active_inference_battleship/scripts/check-docs.ts:66`, and `CEREBRUM/scripts/gate.py:103` with `scripts/docgen/doc_index.py` |
| Manuscript variable production | `scripts/z_generate_manuscript_variables.py` in each repository, GNN-derived keys from `gnn.importers.importer_variables` | 8 members already use the template's filename; `ActiveBlockference/scripts/manuscript_variables.py:57` and `policy_entanglement/scripts/manuscript_variables.py` use different names; `active_inference_battleship/research/variables.py` is a third |

### Measured copies of pin reading

Each of these reads a pin file (or a pin-shaped constant) and decides for itself
which key holds the revision. Six concerns copied across the estate is a
contract; twenty-one is the migration.

| Member | Copy |
| --- | --- |
| `fep_formal` | `src/fep_formal/bridge/operations.py:140` (`pin_sources`), over the nested `source-pin.json` |
| `pymdp` | `pymdp/gnn.py:97` and `:100` (release and revision as module constants) |
| `pymcp` | `scripts/gnn_conformance.py:41`, `:71`, `:80`–`:93` |
| `Active_InferAnts` | `gnn_conformance.py:33`, `:63`, `:125` |
| `ants` | `gnn_conformance.py:33`, `:60`, `:120`; `scripts/gate.py:43`, `:144` |
| `ActInf_RxInfer` | `scripts/validate_gnn_spec.py:59`, `:121`; `scripts/gnn_parser_probe.py:6` |
| `ActiveBlockference` | `blockference/gnn.py:52`; `scripts/validate_gnn_spec.py:105`; `scripts/manuscript_variables.py:21`, `:57`–`:61` |
| `Active_Fedference` | `.github/workflows/ci.yml:126`; `scripts/gate.py:52`–`:54`; `src/fedference/gnn/loader.py:38` |
| `Active_Cardference` | `scripts/validate_gnn_spec.py:57`; `scripts/gate.py:47`; `src/gnn_cards/spec.py:310`; `scripts/manuscript/build_figures.py:510`–`:517` |
| `active_fractal_rabbit` | `scripts/check_gnn_conformance.py:38`, `:44`, `:69`–`:72` |
| `policy_entanglement` | `src/gnn/upstream.py:42`; `scripts/gate.py:68` |
| `on_policy_distillation` | `src/gnn/upstream.py:41`; `scripts/gate.py:46` |
| `active_inference_battleship` | `.github/workflows/ci.yml:61`–`:63` (three `bun -e` reads); `scripts/import-model.ts:20` |
| `active_inference_power` | `src/active_inference_power/gnn_specs.py:39`; `scripts/validate_gnn_specs.py:4` |
| `active_inference_language_model` | `scripts/fetch_gnn_upstream.py:40`; `scripts/verify_release.py:825`–`:828` |
| `COGANT` | `tools/gnn_upstream_audit_surface.py:6`–`:9` takes the expected commit from the `upstream` extra in `cogant/pyproject.toml` and `cogant/uv.lock`, a second source beside its `.github/gnn-pin.json` |
| `CEREBRUM` | `scripts/gnn_checkout.py:29`; `tests/core/test_gnn_spec.py:48` |
| `Biofirm` | `tests/test_gnn.py:40`, `:62`–`:63`, with field-by-field assertions at `:137`–`:144` |
| `GEO_INFER` | `GEO-INFER-TEST/src/geo_infer_test/gnn_pin_freshness.py:19`; `GEO-INFER-TEST/src/geo_infer_test/companion_releases.py:294` |
| `AEOS` | `scripts/check_gnn.py:28`, `:142` |
| `Active_Inference_Ontology` | `scripts/validate_gnn_conformance.py:25`, `:91`–`:96` |

## The pointer contract

Every importer publishes `.github/GNN.md`, and it is what the estate's online
re-read looks for (`scripts/check_importer_registry.py:376`). The reader's
`check_pointer` makes the pointer checkable rather than decorative: it requires
the pointer text to name

- the pin file, as the path the registry records;
- the pinned revision, in full or as its first twelve characters;
- the conformance workflow, by filename, when `--workflow` is given;
- the job key, when `--job` is given.

Each miss is its own finding: `pointer-pin-unnamed`, `pointer-revision-unnamed`,
`pointer-workflow-unnamed`, `pointer-job-unnamed`, `pointer-missing`.
`--require-pointer` is opt-in: a repository turns it on in the change that makes
its pointer complete, so this contract does not retroactively fail twenty-one
gates.

## Manuscript variables for GNN-derived models

The render pipeline invokes `scripts/z_generate_manuscript_variables.py` before a
PDF render, which is why that filename, not a name of the author's choosing, is
the producer contract. Of the twenty-two members, eight already use it and three
use a different name.

A manuscript that states which GNN release it consumes must not re-read the pin
JSON. It calls `gnn.importers.importer_variables`, which returns the
`gnn-importer-variables/v1` map:

| Key | Value |
| --- | --- |
| `schema` | `gnn-importer-variables/v1` |
| `gnn_pin_path` | the pin, relative to the repository root |
| `gnn_pin_schema` | the pin's declared schema, or empty |
| `gnn_repository` | the normalized `org/name`, or empty |
| `gnn_tag` | the release tag, or empty |
| `gnn_commit` | the 40-character commit |
| `gnn_job_key` | the conformance job key the registry records |
| `gnn_job_name` | that job's display name |

The reader does not parse workflow YAML, so the two job keys are passed in from
the registry row. `ActiveBlockference/scripts/manuscript_variables.py:57`–`:61`
is the copy this replaces; its `gnn_*` keys keep their names, so prose does not
change.

## The template render lock, the project link, and a mismatch

A member that renders a manuscript consumes `docxology/template`. Two things
about that consumption are declarable, and the group's rule is that both are
declared rather than assumed.

**The renderer revision** is declared in `manuscript/config.yaml` under
`template_renderer` with schema `template-renderer-lock-v1`:

```yaml
template_renderer:
  schema_version: "template-renderer-lock-v1"
  repository: "docxology/template"
  git_commit: "7558ee58c624f7554d45b85dce1c9cc42f754610"
```

That block is `Active_Fedference/manuscript/config.yaml:151`–`:154` and, at the
same commit, `Active_Cardference/manuscript/config.yaml`. In this group those are
the only two declarations of the lock; the other manuscript-bearing members'
`config.yaml` record no `git_commit`, so their renderer revision is whatever
checkout the operator happens to have.

**The project link** is a path inside the template checkout:
`projects/working/<repository-name>`, with `projects/active/` holding
`_test_project`. On this workstation `projects/working/` holds
`active_fractal_rabbit` and `active_inference_language_model` among others, and
`projects/active_fedference` exists nowhere under `projects/`.

**When the local reference is at a different commit than the member locks, the
group does not render.** The reference checkout on this workstation is at
`f1fc24de3a2989531e954bf5098360dbfb47ec3d`; `Active_Fedference` locks
`7558ee58c624f7554d45b85dce1c9cc42f754610` and has no project link, so its
render is blocked by the mismatch and the missing link, not by a missing tool
(`pandoc`, `xelatex`, `qpdf` and `pdfinfo` are present). The rule:

1. Never edit the reference checkout to match a member. It is read-only, shared
   by every project.
2. Check the locked revision out into its own location, or link the member's
   project into the checkout that *is* at the locked revision, and render there.
3. A member that wants the newer reference re-locks its own `config.yaml`
   through a reviewed change to that member, never by editing the template.
4. A render performed against a different revision than the one the member
   locks is not evidence for that member, whatever the artifacts look like.

The `fast`-arm result recorded below makes the same point about gates: a member
whose render inputs are inconsistent is a finding to report, not a green.

## Measured member arm evidence

No member gate is claimed green by this contract. What has actually been run is
recorded here, with the observation's source, so a later claim can be checked
against a name rather than a memory:

| Member | Arms run | Result | Source |
| --- | --- | --- | --- |
| `Active_Fedference` | `fast` | timed out at 600 s with two failures in `tests/analysis/test_stage_process.py` | the member's own worker, 2026-11-12 |
| `Active_Fedference` | `full`, `shard`, `verify`, `gnn` | not run | same |
| every other member | none | not run by this contract | this page |

`Active_Fedference/scripts/gate.py:38`–`:44` defines the `fast` arm as the whole
`tests/` tree under `-m "not slow and not publication"`, so
`tests/analysis/test_stage_process.py` is inside that selection; the timeout is a
property of the arm, not of an unrelated test.

For the lead repository, see the measured arms below.

## `fep_formal` and the Lean boundary

`fep_formal` stays a row of its own, and not as a GNN `spec` row. Three facts
decide it:

1. Its custody record is not a release pin. `source-pin.json` has
   `schema_version: 1` and two sub-objects, `fep_lean` and `gnn`; the `gnn`
   sub-object names descriptive commit `2778e65c` with six SHA-256 owner
   digests and no tag. It is a content-addressed custody artifact, so the reader
   takes `--pin-party gnn` and reports the missing tag as `pin-schema-unversioned`
   rather than inventing one.
2. Its evidence is a Lean proof surface, not a parser run. The conformance job
   runs `bash scripts/gate.sh gnn`, which drives `generate_probe.py --check`,
   `check_geo_notation_bridge.py --check`, `validate_retained.py` and
   `test_composition.py`; the only `*.gnn.md` it reads is the Cascadia fixture.
3. The bridge is mirrored in both directions. This repository pins `fep_formal`
   at `48206d6d` in `.github/fep-lean-pair.json` and names the bridge contract
   among the files a paired revision re-drifts.

So: one row, `pin.kind` `content-hash`, party `gnn`, and no `spec` kind. The
reader validates the pin's shape and the fixture's syntax; the Lean boundary,
its owner digests and its paired-revision procedure stay `fep_formal`'s own
gates. Treating it as a GNN specification row would claim that a Lean theorem
catalogue is validated by a Markdown parser, which it is not.

## Migration: the step every member takes

All twenty-two members take the same step. What differs is the copy to delete,
which the table above locates.

1. In the members' `gnn` arm, after the pinned revision is materialized, run
   the reader from that environment:

   ```bash
   python -m gnn.importers check --repo . --require-pointer \
       --workflow .github/workflows/ci.yml --job gnn-conformance
   ```

   `--require-pointer` and the two pointer flags are added in the same change
   that makes the pointer complete; a member whose pointer is already complete
   adds them at once. Members whose conformance job has a different key or
   workflow pass their own values, as the registry row records them.
2. Complete `.github/GNN.md` so it names the pin path, the revision, the
   workflow and the job key. `check_pointer` reports the missing fact.
3. Delete the member's own pin reader and its tests for pin shape. The reader
   already reports `pin-missing`, `pin-unreadable`, `pin-commit-shape`,
   `pin-commit-mismatch`, `pin-tag-mismatch`, `pin-repository-mismatch`,
   `pin-schema-unversioned`, `spec-missing`, `spec-invalid` and the pointer
   codes, so a member's duplicate of that logic is not carrying unique coverage.
4. If the member's manuscript states the pin, replace the inline JSON read with
   `gnn.importers.importer_variables` and keep the `gnn_*` key names.
5. Leave the member's other arms alone. This contract adds one arm; it does not
   replace a member's own gate, and the member's `gate` command above stays its
   front door.

Member-specific notes:

- `pymdp` has no pin file, no pointer and no gate front door. It adopts
  `.github/gnn-pin.json`, adds `.github/GNN.md`, and keeps `pymdp/gnn.py`'s
  constants only as the code-facing bridge surface, generated from the pin.
- `fep_formal` passes `--pin specs/gnn-bridge-w2-source-custody/source-pin.json
  --pin-party gnn` and keeps its owner-digest and paired-revision gates.
- `GEO_INFER` keeps `gnn-pair.json` as its pin path; the reader already accepts
  `revision`.
- `active_inference_battleship` calls the reader from `scripts/parse_gnn.py`,
  and its recorded `reason` says why the gate cannot import it.
- `ActiveBlockference` and `policy_entanglement` rename their
  `scripts/manuscript_variables.py` to the template's filename when they next
  render a manuscript, because the pipeline invokes that name.
- `COGANT` folds its `pyproject.toml`/`uv.lock` derivation into the pin read,
  so one file states the revision.
- The upstream member does not run the arm on itself; see above.

## Verification

The lead repository's arms are the only measured evidence on this page. Both
columns were measured in one worktree with the same frozen environment, switching
between `origin/main` and this branch, so the comparison is the change and not a
cold install:

| Arm | `origin/main` | this branch |
| --- | --- | --- |
| `./run.sh docs` | 13.97 s | 16.54 s |
| `python scripts/check_importer_registry.py` alone | 0.05–0.08 s (three runs) | 0.05–0.07 s (three runs) |

The registry check itself did not move; the arm grew by the three documentation
files this change adds to the tree the `docs` arm walks. `./run.sh lint` and
`./run.sh type` (781 source files) are green on the branch. The two new test
files contribute twenty-five tests. The registry reports 42 repositories and
`group gnn-fep: 22 members, 20 call the reader directly`; the two that do not are
the upstream member and the wrapper.

Member arms were not run except where the table above records one. Each member's
own gate remains its evidence, and this page makes no claim about a member's
current green state.

### One custody cost this change carries

This page adds three files under `docs/`, so `GNN_DOC_FILE_COUNT` moves from 675
to 678. The token map, the repo-metrics figure and the hydrated manuscript are
not regenerated here: `scripts/check_manuscript_tokens.py --strict` already fails
on `origin/main` before this change (`committed token_checksum 22f31c37db3f !=
producer-at-HEAD 8f9a7688c733`), and a partial ritual — new variables without a
fresh render — would leave the figures and the PDF disagreeing. The whole SC-22
ritual, including the template render, stays owed against `HEAD`; the PR-time
hydration gate is unaffected and green, because it compares the committed
artifacts against the committed map.

## Non-goals

- This contract does not pin members to one GNN release. Each member's pin file
  owns its revision; this page owns how that file is read.
- It does not add a member-side workflow, job or required check. The arm runs
  inside a gate the member already has.
- It does not delete a member's tests, except a test whose only subject is the
  pin shape the reader now owns.
- It does not touch another workflow group's repositories.

## Related

- [Downstream registry](../downstream/README.md) and its machine-readable form
  [importers.json](../downstream/importers.json)
- [GNN documentation hub](../README.md)
- [Downstream agent guidance](../downstream/AGENTS.md)
