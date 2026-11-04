# Manuscript Agent Notes - GeneralizedNotationNotation

This directory follows the docxology/template manuscript contract:

- `00_` through `09_` files are main sections.
- `S01_` files are supplemental material.
- `98_` and `99_` files are back matter.
- Every manuscript section file starts with one H1 and a stable `{#sec:...}` label.
- Citations must use Pandoc syntax and resolve in `references.bib`.
- Generated numbers belong behind `{{TOKEN}}` variables, not hard-coded prose.

## Editing Rules

- Treat this scaffold as an outline until project-specific evidence is bound.
- Do not fabricate results, benchmark numbers, citations, DOIs, or release claims.
- Keep project-specific computation in source modules and scripts; keep manuscript files as prose and evidence maps.
- Prefer explicit paths to source surfaces when describing evidence.
- If adding figures, write them under `../output/figures/` and reference them with Pandoc-crossref labels.

## Current Scope

A standardized text-based language and processing pipeline for Active Inference generative models, transforming specifications into validation, visualization, simulation, and analysis outputs.

Evidence boundary: Do not treat root output churn as manuscript evidence until the specific run target and generated artifacts are recorded.

## Figures

- `output/figures/figure_registry.json` is **generated** by
  `python -m scripts.manuscript_build_figures`, never hand-edited. Its label,
  filename and generator columns come from the same table the build loop runs, so
  it cannot drift from the figures that were actually produced. Add a figure by
  adding a row to `_FIGURES` in that script; the build fails if the manuscript
  declares a `{#fig:...}` label the table does not carry, or vice versa.
- `alt_text` is the one authored field. It must say what the figure *shows* for a
  reader who cannot see it — different words from the caption, which is a title.
- `scripts/check_manuscript_tokens.py` re-checks registry coverage independently of
  the template. The template's own `validate_figure_registry` runs at
  `stage_04_validate`, not `stage_03_render`, so relying on it alone let a 29-page
  PDF ship with no registry at all.

### A figure is a second renderer of the token map, and it does not re-run

Prose is re-hydrated from `output/data/manuscript_variables.json` on every render.
A figure is a committed PNG: move a count in the token map and the prose follows,
while the figure keeps printing the old value. `fig:repo_metrics` shipped
"Test files: 365" on the same PDF page as prose reading 367, under a caption
claiming it was measured at the stamped commit — the count had moved 365 → 366 →
367 across three commits and no one re-ran the build.

So: **any commit that moves a count must re-run
`python -m scripts.manuscript_build_figures` and commit the PNGs.** This is now
mechanical rather than remembered. Generators read the token map through
`scripts/lib/manuscript_figure_tokens.load_tokens()`, which records the
`{key: value}` pairs each generator actually reads; the build writes them, plus
the PNG's SHA-256, into `figure_registry.json`; and
`src/tests/test_manuscript_figure_freshness.py` fails when a recorded value
disagrees with the live token map, or when a committed PNG is not the one the
registry records. A new generator that opens the token map directly fails that
suite too, because bypassing the loader would leave its numbers unchecked.

## Paths are claims, and are checked like counts

Do not type a model family's target directory into prose. The manifest owns it,
and the producer emits the whole sentence:
`{{GNN_UNSCANNED_CORPUS_NOTE}}` for the coverage paragraph and
`{{GNN_TARGET_DIR_COVERAGE_NOTE}}` for the `--target-dir` note. Both flip with
the manifest, including between the "all inside `input/gnn_files`" and "some
outside" wordings.

This exists because a typed path is the one claim every other check was blind
to. A commit repointed the `multiagent` family's `target_dir` from
`input/multi_agent_models` into `input/gnn_files/multiagent`; three prose sites
went on describing the old layout, every count beside them stayed correct, and
`check_manuscript_tokens.py --strict` reported "clean" over the contradiction —
so the false text shipped in the PDF.

Rule 8 of that gate now cross-checks the manifest: an `input/...` literal must
exist on disk, and a literal typed next to a family name must be that family's
declared `target_dir` (or an ancestor of it, so `input/gnn_files` still reads as
covering every family). `src/tests/test_manuscript_path_claims.py` pins both the
generated sentences and the gate rule, and its two live-repository tests fail on
the pre-fix prose.

## Commands are claims too, and "it exists" is not the claim

§6 promises it "lists only commands that exist in the repository, so that a
reader with a clean checkout can reproduce the pipeline". It shipped
`scripts/run_model_family_acceptance.py --manifest ... --strict`, which exits 2:
`--output-dir` is `required=True`. The file existed, so nothing complained —
"exists" had quietly come to mean "the path is there" rather than "the
invocation runs".

`src/tests/test_manuscript_published_commands.py` reads every command out of the
manuscript's `bash` blocks and checks it against the script's own `argparse`
declarations: every flag passed must be declared, and every `required=True`
option must be passed. It is static — it proves the repository would accept the
arguments, not that the command succeeds — so a published command still has to
be run before it is called verified. One command is exempt by name, the
template's `stage_03_render.py`, because §6.3 says in prose that it runs from a
separate checkout; a test pins that sentence so the exemption cannot outlive it.


## Known benign LaTeX diagnostics

`output/pdf/_combined_manuscript.log` carries
`Infinite glue shrinkage found in box being split` warnings. **This is settled,
not open.** Do not re-open it; the experiments below are the reason.

### Count them like this, not with a bare grep

```bash
tr -d '\n' < output/pdf/_combined_manuscript.log \
  | grep -o 'Infinite glue shrinkage found in box being split' | wc -l
```

TeX breaks its own log lines, and it will split this message mid-word. The log
this note was written against contains, verbatim (lines 1284-1286):

```
ignored: Infinite glue shrinkage found in box being split [24]
ignored: In
finite glue shrinkage found in box being split [25]
```

so `grep -c 'Infinite glue'` reported **3** against a true count of **4**, and
the earlier version of this note prescribed exactly that under-counting probe.
Treat every log probe this way: join the lines before matching.
`src/tests/test_manuscript_latex_log.py` pins the split line verbatim and the
counting that survives it.

### One per wide table — not one per page break

The old note (and the audit that raised this) said "one at every longtable page
break". That is false; the folios coincided. Dropping each `longtable` from
`_combined_manuscript.tex` in turn and re-rendering gives:

| dropped table | column spec | occurrences |
|---|---|---|
| — (baseline) | — | 4 |
| `tbl:pipeline_steps` | `p{...}` ×3 | 3 |
| `tbl:backend_registry` | `lll` | **4** |
| `tbl:model_families` | `p{...}` ×3 | 3 |
| `tbl:gnn_constructs` | `p{...}` ×2 | 3 |
| `tbl:actinf_symbols` | `p{...}` ×2 | 3 |

In this document each of the four paragraph-column tables owns exactly one
message, and the one `lll` table owns none. The *bound* is structural rather
than a tally: `\LT@start` (`longtable.sty:196`) does `\let\LT@start\endgraf`
on its first call, so its `\vsplit` runs once per `longtable` and one table can
contribute at most one message. `src/tests/test_manuscript_latex_log.py`
asserts that bound against the shipped log, so a fifth message would fail the
suite instead of being re-discovered by the next audit.

### The emitter is proven, not inferred

`\endlongtable` was wrapped in `\message` markers and the document re-rendered:

```latex
\makeatletter
\let\GNNorigendlongtable\endlongtable
\def\endlongtable{\message{^^JPROBE-BEGIN^^J}\GNNorigendlongtable\message{^^JPROBE-END^^J}}
\makeatother
```

Every occurrence lands strictly between a `PROBE-BEGIN` and its `PROBE-END`.
`\vsplit` appears exactly once in `longtable.sty` (line 212), inside
`\LT@start`, which `\endlongtable` calls — so that is the emitter.
`latex.ltx`'s `\@doclearpage` (line 20594) — the kernel's other `\vsplit`
user — was hooked in the same run as a control: it fires once, at
`\end{document}`, and emits nothing between its markers.

### The cause is the `p{...}` column, and nothing above it

Reduced to one table in a document that reproduces exactly one occurrence, then
changed one thing at a time:

| variant | occurrences |
|---|---|
| as Pandoc emits it | 1 |
| `>{\raggedright\arraybackslash}p{...}` → `lll` | **0** |
| `minipage[b]{\linewidth}\raggedright` headers removed | 1 |
| `\raggedright\arraybackslash` dropped, plain `p{...}` kept | 1 |
| `\caption` removed | 1 |

Only the column type matters. The `minipage` headers — the cause the first
version of this note asserted — are irrelevant, and so are `\raggedright` and
the caption. It is also position-sensitive: the same table in a short document
emits 0, 1 or 0 as filler pushes it down the page, so this is not a property of
the table alone and cannot be fixed by editing the table.

Pandoc chose `p{...}` for the four tables whose cells wrap and `lll` for the one
whose cells do not, so the only manuscript-level change that removes the message
is narrowing `tbl:gnn_constructs`, `tbl:actinf_symbols`, `tbl:model_families`
and `tbl:pipeline_steps` until their cells fit on one line — which would destroy
their content. **There is no fix, and there is nothing to fix.**

### Why the message cannot affect the shipped page

The `\vsplit` at `longtable.sty:212` splits a *copy*:

```latex
\setbox\tw@\copy\z@                          % 211: a COPY of the chunk box
\setbox\tw@\vsplit\tw@ to \ht\@arstrutbox    % 212: the split TeX warns about
\setbox\tw@\vbox{\unvbox\tw@}%               % 213
...\ht\tw@...  ...\dp\tw@...                % 216, 218: its ONLY consumers
```

`grep -n 'tw@' $(kpsewhich longtable.sty)` confirms box `\tw@` is never `\box`ed
or `\unvbox`ed onto the page in that macro: it is measured and dropped. The
measurement feeds one decision — whether the table's first row fits in the space
left on the current page, or whether to `\vfil\break` first. The typeset output
is not the box that was split. Corroborating: `Overfull` count is 0, and every
data row of `tbl:gnn_constructs` and `tbl:actinf_symbols` is present in the
rendered PDF.

### Two remedies that were tested and do not work

Kept on record so they are not retried. Measured again in the one-table rig
above, whose baseline is 1 occurrence, 0 underfull, 0 overfull:

| applied | occurrences | underfull |
|---|---|---|
| baseline | 1 | 0 |
| `\setlength{\LTpre}{0pt}\setlength{\LTpost}{0pt}` | 1 | 0 |
| `\setlength{\@flushglue}{0pt plus 2em}` | 1 | 15 |

1. `\LTpre`/`\LTpost` is the audit's own prescription and it changes nothing.
   `\LTpre` is applied at `longtable.sty:198` (`\vskip\LTpre`), *before and
   outside* the box whose copy line 212 splits, so its value is irrelevant.
2. `\@flushglue` is strictly worse — the warning survives and loose lines
   appear (15 here; a full render on 2026-09-05 produced 144). `\@flushglue`
   is horizontal (`\rightskip`); the reported shrinkage is vertical.

### What is still not known, and why it does not matter

The specific glue node has not been named: dumping the chunk box with
`\showbox\z@` immediately before `\LT@start` shows only row `\hbox`es and
`\glue(\lineskip) 0.0` at top level, with no infinite shrink. That is a question
about TeX's `vpack` internals, not about this manuscript — the bound, the
emitter, the cause and the harmlessness are all established without it. Do not
spend another pass on it, and do not suppress the message by dropping a table.

### Underfull hboxes are the price of readable identifiers

Count them the same way as everything else in this log — join the lines first:

```bash
tr -d '\n' < output/pdf/_combined_manuscript.log | grep -o 'Underfull' | wc -l
```

Which of them are live moves with pagination (one render had three, the next
two), so this is a catalogue of *causes*, not a census. Do **not** buy any of
them back by unbounding `\breaktt` — that would reintroduce a correctness
defect. Each was traced to its paragraph in `_combined_manuscript.tex` via the
line range the log prints:

| badness | cause |
|---|---|
| 1019 | `\breaktt{output/data/manuscript\_variables.json}` |
| 3657 | `\texttt{output/}` / `\texttt{manuscript/}` in a narrow `\item` measure |
| 1852 | four consecutive `\breaktt{A=LikelihoodMatrix}`-style spans |

Two of the three causes are `\breaktt`, which deliberately forbids line breaks inside an
identifier's first 2 and last 5 characters. That bound exists because unbounded
`\seqsplit` rendered `model_family_manifest.json` as `model_family_manifest.js` +
`on` — and the first fragment is itself a valid filename, so a reader could not
tell a line break from a name. An unbreakable token in a narrow measure makes the
line stretch; that is the cost, and it is the right one to pay.

An underfull hbox is a *loose* line, not lost content: every badness here is far
under TeX's 10000 ceiling and the `Overfull` count is 0 (asserted by
`src/tests/test_manuscript_latex_log.py`), so nothing is clipped or run into
the margin. The 3657 case is a plain `\texttt` list item and has nothing to do
with `\breaktt`.
