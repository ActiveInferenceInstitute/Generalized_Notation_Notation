# `gnn/render/pomdp_processor/` — README

Specialized POMDP state-space injection into framework-specific renderers
(PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro,
Stan, ngc-learn): `POMDPRenderProcessor` composes six topic mixins over the
`RENDERER_ROUTES` dispatch table, validates per-framework compatibility,
builds `canonical_pomdp_v1` specs, and writes per-framework documentation.

Import only from the package facade
(`from gnn.render.pomdp_processor import ...`); for pure conversion without
output writes use `pomdp_to_gnn_spec`. The `_*.py` leaf modules are private
to the package; `_support.py` is an annotation-only mypy contract and must
stay runtime-free.

Mixin architecture, canonical-spec contract, facade surface, and the
gating tests are in [`AGENTS.md`](AGENTS.md); the parent module's
[`AGENTS.md`](../AGENTS.md) covers the Step 11 surface.
