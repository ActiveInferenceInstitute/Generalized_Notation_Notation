"""Canonical framework enumeration for the GNN pipeline.

One tuple, one source: every framework list that enumerates the pipeline's
backends derives from ``ALL_FRAMEWORKS`` here, so a backend added or removed
in one place is added or removed everywhere. The rich per-framework render
configuration stays in ``gnn.render.framework_registry.FRAMEWORK_REGISTRY``
(the canonical render inventory); this module owns only the name
enumeration. Self-consistency between the two is enforced by
``tests/render/test_framework_availability.py``.

Capability-scoped subsets (executor registry-runners, planning language
classification, analyzer-backed frameworks, the profiled
``MAINTAINED_FRAMEWORKS`` trio in ``pipeline.cross_framework_reliability``)
are intentional subsets, not alternate enumerations, and keep their own
definitions with a comment naming the capability they encode.
"""

from __future__ import annotations

from typing import Final

#: Every backend the pipeline can render or execute, in render order
#: (matching ``FRAMEWORK_REGISTRY`` keys) with the execution-only ``lean``
#: backend appended. ``lean`` runs through the executor registry-runner path
#: (``ExecutorFrameworkSpec``) and has no render entry.
ALL_FRAMEWORKS: Final[tuple[str, ...]] = (
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "stan",
    "bnlearn",
    "cpomdp",
    "thrml",
    "ngclearn",
    "lean",
)

#: Frameworks served by the lightweight ``"lite"`` preset (no Julia
#: toolchain, no GPU stack). ``gnn.render.framework_registry.LITE_FRAMEWORKS``
#: re-exports this tuple so the two cannot drift.
LITE_FRAMEWORKS: Final[tuple[str, ...]] = (
    "pymdp",
    "jax",
    "discopy",
    "bnlearn",
)

# The execution-only Lean bridge has no renderer. This leaf subset lets the
# CLI build without importing optional scientific dependencies.
RENDER_FRAMEWORKS: Final[tuple[str, ...]] = tuple(
    name for name in ALL_FRAMEWORKS if name != "lean"
)

__all__ = ["ALL_FRAMEWORKS", "LITE_FRAMEWORKS", "RENDER_FRAMEWORKS"]
