# Gate, CI and manuscript run guide

Run commands from the repository root. [run.sh](../run.sh) owns the ordinary
commands; CI calls its arms rather than maintaining another copy. Every arm
uses `uv.lock` through `uv sync --frozen --extra dev`, followed by
`uv run --frozen --no-sync`. The aggregate synchronizes once per invocation.
Hosted Actions is billing-locked for this estate; the evidence below is local.

## Eight separately runnable arms

| Command | Scope | CI owner |
| --- | --- | --- |
| `./run.sh sync` | Frozen development environment | Ordinary test and docs setup |
| `./run.sh lint` | Ruff format check and lint | `ci.yml`: `quality` / `quality (3.12)` |
| `./run.sh type` | Mypy on `src/gnn` and the release gate | `ci.yml`: `quality` |
| `./run.sh test` | Default parallel pytest selection, without coverage instrumentation | Developer arm; same selection as CI coverage |
| `./run.sh coverage` | Same selection, with the existing coverage floor | `ci.yml`: `test`, `pytest312` / `pytest (3.12)` |
| `./run.sh docs` | Relative links/anchors, terminology, doc patterns, documentation and capability contracts | `docs-audit.yml`: `markdown-audit` |
| `./run.sh contracts` | MCP functional/auth/transport, framework availability and capability tests; whole-tree collection | `ci.yml`: `quality` |
| `./run.sh acceptance` | Strict v3 orchestration acceptance | `ci.yml`: `quality` |

`./run.sh gates` (also the no-argument default and `just gates`) combines lint,
type, coverage, docs, contracts and acceptance. It does not run `test` again:
coverage already executes that selection. The existing default is preserved.

`test`, `coverage` and `contracts` forward trailing pytest arguments, so CI can
request JUnit/coverage files without copying selections or pytest commands:

```bash
./run.sh coverage --cov-report=json:/tmp/gnn-coverage.json --junitxml=/tmp/gnn-tests.xml
./run.sh contracts --junitxml=/tmp/gnn-contracts.xml
```

The default marker selection excludes `pipeline`, `mcp`, `ollama`, `env_heavy`
and `toolchain`. The MCP subset runs in `contracts`; pipeline contracts have
their own `ci.yml` job, not the MCP arm. Native compilation, all-extras/full
matrices, live providers, paired companions and release lanes are not included
in these arm measurements. This bounded selection is not the complete native
suite; [testing profiles](testing/README.md) retain their separate scopes.

Documentation audits have one CI owner. Their duplicate invocations were
removed from `quality` and `local-gates`; status/job names, ordinary test
matrices, coverage denominator and reporting artifacts remain unchanged.
Separate runners still need independent environments; no cross-runner `.venv`
is shared. No composite setup action was introduced.

## Measured wall time

Baseline: `70757fc8e7c0c8591e267342b12f11f1a7f491b6`, macOS arm64,
Python 3.14.10, uv 0.13.7 (allowed by `uv.toml`'s `>=0.13.6,<0.14`). The
frozen environment was provisioned before timing; caches were not cleared.
These are single observations, not statistically controlled performance data.
The first mypy run and subsequent cached runs are not equivalent conditions.

| Arm | Baseline seconds | Final seconds |
| --- | ---: | ---: |
| sync | 1.7 | 4.2 |
| lint | 2.3 | 4.8 |
| type | 28.5 | 8.6 |
| test | 129.8 | 528.5 |
| coverage | 152.1 | 492.2 |
| docs | 33.1 | 40.4 |
| contracts | unavailable as a CLI arm | 47.0 |
| acceptance | 2.4 | 5.2 |

Final implementation source: `fab18063ecf30bb11f4324df7cb6d27d3932355c`.
Each arm completed under GitVault's qualified offline test runner with a
source-bound intent and `duration.upper_seconds=600`; times include its launch
overhead. Rendering overlapped the sweep, and host load/cache state were not
controlled. These observations establish completion, not a speedup.

`test` and `coverage` each passed 9,013 tests and 10 subtests; coverage was
78.1285% against the existing 60% floor. `contracts` passed 97 tests and
completed whole-tree collection. The first final docs run failed on two new
Markdown anchors in 20.2 seconds; the table records the successful corrected
rerun. The baseline CLI rejected `contracts` with exit 2; its existing function
completed in 30.0 seconds after exposure.

A 600-second timeout would be a failed/incomplete observation, not a passing
shortened suite. No arm reached that limit. Native compilation, complete
native/all-extras matrices, live providers, paired companions and release
qualification were deliberately not run. Exact commands are retained in the
PR body; `coverage` and `contracts` exercised the forwarded report arguments.

## Upstream pins and consumer pointers

This repository is the GNN **producer**, not a self-pinned importer. It has no
`.github/gnn-pin.json`, `.github/GNN.md` or `gnn-conformance` job. Do not invent
one or confuse its companion pins with importer schema `gnn-importer-pin/v1`.
The existing [downstream registry](downstream/README.md) records consumers'
actual pin paths, `.github/GNN.md` pointers, job keys and validation scopes.
Its recorded release pin is `v5.0.3` at
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`; the pointer explains what a consumer
parses, not universal coverage of every sketch. The commonly published display
name is `gnn conformance (pinned)`; some job keys differ, as the registry states.
The registry is unchanged by this work.

GNN's own companion pins are [`.github/gnn-pair.json`](../.github/gnn-pair.json)
(GEO_INFER `21b65a26dd15288d6e578baef86a8bcfd6a7b5db`) and
[`.github/fep-lean-pair.json`](../.github/fep-lean-pair.json)
(fep_formal `48206d6de664966d4a721080aee76a6c3bdf3e0a`). Their workflows are
[Paired GEO_INFER interchange](../.github/workflows/geo-infer-interchange.yml)
and [Paired fep_formal revision](../.github/workflows/fep-lean-paired-revision.yml).
They do not use the importer conformance job or importer pin schema.

The GNN workflows use pinned `actions/setup-python` and `astral-sh/setup-uv`,
not `ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`.
That composite action is not a claim about this repository's setup.

## Manuscript rendering and custody

The authoritative renderer is the template's `scripts/pipeline/stage_03_render.py`.
The [manuscript reproduction section](../manuscript/05_reproducibility.md#manuscript-reproducibility)
and [custody workflow](../.github/workflows/custody-re-render.yml) own the ritual:
regenerate variables, rebuild figures, render, verify the old chain, then record
the renewed manifest after a successful render. Figures are declared in
`output/figures/figure_registry.json`; variables are in
`output/data/manuscript_variables.json`.

Keep the reference template read-only. A disposable copy can hold this project
as a real directory at `projects/active/GeneralizedNotationNotation`; project
symlinks are refused. An exported template copy without `.git` needs an explicit
`SOURCE_DATE_EPOCH` from the reference commit. Use a prepared template
interpreter without synchronizing or changing the reference environment:

```bash
# From the disposable template root; set these paths to prepared checkouts.
export SOURCE_DATE_EPOCH="$(git -C "$REFERENCE_TEMPLATE" show -s --format=%ct HEAD)"
export GNN_TEMPLATE_REVISION="$(git -C "$REFERENCE_TEMPLATE" rev-parse HEAD)"
"$REFERENCE_TEMPLATE/.venv/bin/python" scripts/pipeline/stage_03_render.py \
  --project GeneralizedNotationNotation
```

Observed on 2026-11-12 at template
`f1fc24de3a2989531e954bf5098360dbfb47ec3d`, pandoc 3.13,
pandoc-crossref 0.3.26 and TeX Live 2026: rendering completed in 74 seconds,
producing a 32-page, 1,450,176-byte PDF plus 10 HTML sections and 10 slide decks.
The project's environment must be prepared for hydration. An ignored `out/`
tree containing symlinked pytest artifacts blocked admission in the first
copy; excluding it allowed the real render, without changing either reference
checkout or source manuscript.

The old manifest named template `0d4eeb2f`; verification correctly reported
`[STALE-CHAIN]` against `f1fc24de`. This change also rendered clean source
`fab18063e` in 405.8 seconds (32 pages, 1,449,735 bytes), rebuilt figures and
recorded a renewed manifest bound to that actual render. Token and hydrated
prose audits and all 35 focused custody/figure checks passed. The generated
publication artifacts are retained together; no hashes were hand-edited.
