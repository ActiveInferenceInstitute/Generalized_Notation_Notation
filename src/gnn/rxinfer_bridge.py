"""Bayes-net <-> RxInfer.jl bridge over the daf-jev GraphSpec interchange.

This module is the GNN-side half of the pinned cross-repo contract with
daf-jev (``dafjev.bayesnet/1`` GraphSpec JSON — see the daf-jev contract
section 3). It consumes and produces GraphSpec JSON, parses a minimal
Bayes-net-oriented ``.gnn`` markdown subset into a GraphSpec, and emits a
standalone deterministic RxInfer.jl script (``@model`` with one
``DiscreteTransition`` factor per CPT) that loads the JSON, conditions on
evidence, and prints marginals.

Jev integration points (daf-jev is the Jev client; GNN is the engine):

* **upstream** — CPTs and network structure are produced by daf-jev
  elicitation (``propose_structure`` + ``elicit_cpts``) and exchanged as
  GraphSpec JSON (format ``dafjev.bayesnet/1``). ``load_graphspec`` /
  ``load_graphspec_file`` consume that interchange; ``parse_gnn_subset``
  and ``render_gnn_subset`` are the GNN-flavored authoring surface.
* **within** — RESERVED. The optional GraphSpec top-level field
  ``jev_factors`` is reserved for Jev-derived factors (zero-shot
  probabilistic factors injected as extra factors at inference time).
  Its semantics are intentionally unpinned: the loader validates only
  the outer shape (a list of string-keyed objects) and preserves the
  entries verbatim through ``to_json``; ``emit_rxinfer_jl`` emits a
  comment noting the reserved field and otherwise ignores it. Any use
  of ``jev_factors`` must be coordinated across both repos in one wave.
* **downstream** — the emitted script prints posterior marginals for the
  non-evidence variables and can write a posteriors sidecar JSON
  (``--out FILE``, format ``dafjev.bayesnet-posteriors/1`` — a plain
  sidecar, deliberately NOT the pinned GraphSpec schema). Re-asking
  daf-jev for follow-up evidence queries then goes through the daf-jev
  CLI (see ``examples/rxinfer/README.md``).
  ``parse_marginals`` parses that printed block back into
  ``{key: {state: probability}}`` and ``write_marginals`` serializes it as
  a ``gnn.marginals/1`` JSON sidecar for daf-jev calibration/re-ask.

Validation here is a deliberate duplicate of the daf-jev ``BayesNet``
rules (this repo must not import ``daf_jev``): unknown edge endpoints,
duplicate edges, self-loops, cycles, missing/duplicate CPTs, CPT parent
set AND order mismatch, non-canonical row order, unknown assignment
labels, wrong row count, non-finite/negative probabilities, and row sums
outside 1e-6 of 1.0 are all rejected with ``ValueError`` messages naming
the context and the offending value (fail-closed).

Canonical CPT row order is the parent-assignment odometer over each
parent's state list, first parent slowest (``itertools.product`` order)
— identical to the daf-jev pinned ordering.

The ``.gnn`` subset accepted by ``parse_gnn_subset`` is documented on
that function; full GNN pipeline files (POMDP tensor blocks and the 25
steps) are out of scope — unknown constructs are rejected with
actionable messages rather than partially parsed.

RxInfer note (validated empirically against RxInfer 5.5.0 and 5.5.2):
GraphPPL requires each ``@model`` argument to be supplied exactly once.
The emitted models keep every latent free and add one observation
interface per variable (``e_<key> ~ DiscreteTransition(<key>,
EYE_<key>)``); evidence is a one-hot through ``data``, unobserved
interfaces go through ``predictvars``/``missing``. Single-parent nets run
end-to-end with correct posteriors; nets containing a multi-parent
``DiscreteTransition`` hit a ReactiveMP structured-rule limitation (see
``examples/rxinfer/README.md`` for the verified gap matrix).
The learning variant replaces fixed CPT arguments with latent
``Dirichlet``/``DirichletCollection`` priors plus the mean-field cut and
marginal initialization RxInfer requires for learning transition tensors.

Implementation owners live under ``gnn.rxinfer_interchange``: validated data,
source authoring, Julia emission and posterior sidecars. This module preserves
the released public import surface.
"""

from .rxinfer_interchange.graphspec import (
    GRAPH_SPEC_FORMAT as GRAPH_SPEC_FORMAT,
)
from .rxinfer_interchange.graphspec import (
    GraphCPT as GraphCPT,
)
from .rxinfer_interchange.graphspec import (
    GraphEdge as GraphEdge,
)
from .rxinfer_interchange.graphspec import (
    GraphSpec as GraphSpec,
)
from .rxinfer_interchange.graphspec import (
    GraphVariable as GraphVariable,
)
from .rxinfer_interchange.graphspec import (
    load_graphspec as load_graphspec,
)
from .rxinfer_interchange.graphspec import (
    load_graphspec_file as load_graphspec_file,
)
from .rxinfer_interchange.julia_emitter import (
    emit_rxinfer_jl as emit_rxinfer_jl,
)
from .rxinfer_interchange.marginals import (
    MARGINALS_FORMAT as MARGINALS_FORMAT,
)
from .rxinfer_interchange.marginals import (
    parse_marginals as parse_marginals,
)
from .rxinfer_interchange.marginals import (
    write_marginals as write_marginals,
)
from .rxinfer_interchange.subset import (
    parse_gnn_subset as parse_gnn_subset,
)
from .rxinfer_interchange.subset import (
    render_gnn_subset as render_gnn_subset,
)

__all__ = [
    "GRAPH_SPEC_FORMAT",
    "MARGINALS_FORMAT",
    "GraphVariable",
    "GraphEdge",
    "GraphCPT",
    "GraphSpec",
    "load_graphspec",
    "load_graphspec_file",
    "parse_gnn_subset",
    "render_gnn_subset",
    "emit_rxinfer_jl",
    "parse_marginals",
    "write_marginals",
]
