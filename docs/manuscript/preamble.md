# LaTeX Preamble

This file contains project-local LaTeX additions consumed by the template renderer.
Keep it minimal until the manuscript needs additional math, table, or figure support.

Every package below ships with a base TeX Live installation. Anything outside that
set must be loaded through the `\IfFileExists` guard the template itself uses for
`seqsplit` (see `output/pdf/_combined_manuscript.tex`), so a missing package
degrades instead of failing the build:

```
\IfFileExists{pkg.sty}{\usepackage{pkg}}{<fallback definition>}
```

`cleveref` was loaded here unguarded until 2026-09-05. It resolved only from a
hand-installed personal `texmf` tree (`kpsewhich cleveref.sty` →
`~/Library/texmf/...`, an eight-year-old copy that triggered LaTeX's "First Aid"
patch), it is absent from the `texlive-basic` scheme, and the document contains
zero `\cref`/`\Cref` calls — every cross-reference goes through pandoc-crossref,
which emits plain `\ref`. Reinstating it means guarding it AND converting the
`\ref` calls.

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{natbib}

% Contents listing: article.cls prefixes every \l@section entry with
% \addvspace{1.0em}, which across ten sections costs about ten lines and pushed
% the ~40-entry listing one line past page 2 — leaving a third sheet carrying a
% single "10 References" line. Tighten that inter-section gap for the duration
% of \tableofcontents only, inside a group, so body spacing is untouched and no
% entry is dropped (reducing \tocdepth would have hidden 29 real subsections).
\makeatletter
\let\GNNoriginaltableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \begingroup
  \setlength{\parskip}{0pt}%
  \renewcommand{\addvspace}[1]{\vskip 2\p@}%
  \GNNoriginaltableofcontents
  \endgroup}
\makeatother
% Document metadata. config.yaml declares a subtitle and a keyword list, and
% the producer emits both as tokens, but the template's \hypersetup writes only
% pdftitle/pdfauthor/pdflang -- so `pdfinfo` reported no Subject and no
% Keywords. This block runs after the template's, so it wins.
%
% The two values below are WRITTEN by scripts/z_generate_manuscript_variables.py
% (sync_preamble_metadata), never typed. A double-brace token cannot be used here:
% preamble.md is substituted into output/manuscript/ like any other section, but
% the template's _manuscript_source.py then copies the raw file back over that
% copy, so an unresolved token would reach hyperref and land in the PDF's
% metadata verbatim.
\hypersetup{
  pdfsubject={A text language and modular processing pipeline for Active Inference generative models},
  pdfkeywords={GNN, active inference, generative models, model notation, pipeline automation}}

% Bounded identifier splitting. The template defines
%   \protected\def\breaktt#1{\begingroup\ttfamily\seqsplit{#1}\endgroup}
% and \seqsplit permits a break after EVERY character with no continuation cue.
% In the 2026-09-05 PDF that rendered `model_family_manifest.json` as
% `model_family_manifest.js` + `on` and `src/manuscript_variables.py` as
% `(src/manusc` + `ript_variables.py` — the first is itself a valid filename, so
% a reader cannot tell a line break from a name. Keep seqsplit's guarantee that
% no code token overflows the measure, but forbid breaks inside the last
% \GNNttTail characters (an extension is never orphaned) and inside the first
% \GNNttHead (a one- or two-letter stub is never left behind).
\makeatletter
\newcount\GNN@ttlen
\newcount\GNN@ttpos
\newcount\GNN@ttleft
\def\GNNttHead{2}
\def\GNNttTail{5}
\def\GNN@ttstop{}
\long\def\GNN@ttcount#1{%
  \ifx#1\GNN@ttstop \let\GNN@ttnext\relax
  \else \advance\GNN@ttlen\@ne \let\GNN@ttnext\GNN@ttcount \fi
  \GNN@ttnext}
\long\def\GNN@ttemit#1{%
  \ifx#1\GNN@ttstop \let\GNN@ttnext\relax
  \else
    #1%
    \advance\GNN@ttpos\@ne
    \GNN@ttleft\GNN@ttlen \advance\GNN@ttleft-\GNN@ttpos
    \ifnum\GNN@ttpos>\GNNttHead
      \ifnum\GNN@ttleft>\GNNttTail \discretionary{}{}{}\fi
    \fi
    \let\GNN@ttnext\GNN@ttemit
  \fi
  \GNN@ttnext}
\protected\def\breaktt#1{%
  \begingroup
  \ttfamily
  \GNN@ttlen\z@ \GNN@ttcount#1\GNN@ttstop
  \GNN@ttpos\z@ \GNN@ttemit#1\GNN@ttstop
  \endgroup}
\makeatother

% Declarations the template's combined-PDF renderer requires
% (infrastructure/rendering/_pdf_combined_preamble.py). Absent here, it warns
% "Preamble is missing declarations required by manuscript content" and
% appends these exact lines itself; declaring them explicitly silences the
% warning without changing the typeset document.
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
```
