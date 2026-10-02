"""Multi-line token renderers for the manuscript token map.

The step, family, backend, framework-capability and model-kind tables are
generated from parsed source surfaces (never typed by hand), and the
cross-framework family selection is the one function every surface
highlighting that set must share. ``_caption`` emits the pandoc-crossref
caption line each table requires.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def _humanize_step(script_name: str) -> str:
    stem = script_name.removesuffix(".py")
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").title()


def _caption(text: str, label: str) -> str:
    """Return a pandoc-crossref table caption line.

    The caption must be the line immediately after the table with no blank line
    between them; ``{#tbl:label}`` is what makes the table numbered and
    referenceable as ``[@tbl:label]``.
    """
    return f": {text} {{#tbl:{label}}}"


def _render_step_table(steps: list[tuple[int, str]], purposes: dict[int, str]) -> str:
    """Render the per-step markdown table consumed by the manuscript."""
    rows = ["| Step | Module | Purpose |", "|---:|---|---|"]
    for number, script in steps:
        purpose = purposes.get(number) or _humanize_step(script)
        rows.append(f"| {number} | `{script}` | {purpose} |")
    rows.append(
        _caption(
            "The pipeline steps with the thin orchestrator module that owns "
            "each and the one-line purpose parsed from the master table in "
            "`src/gnn/STEP_INDEX.md`. Read the Step column against the arrows "
            "of [@fig:pipeline]: each row's purpose is the transformation that "
            "module owns, and the table is regenerated from the index at every "
            "build, so it cannot drift from the code it describes.",
            "pipeline_steps",
        )
    )
    return "\n".join(rows)


def _capability_clause(axis: str, specs: dict[str, dict]) -> str:
    """Render a family's native/unsupported split from the registry flags.

    The manifest declares *which* capability axis a family exercises (e.g.
    ``"capability_axis": "supports_continuous"``); the split itself is read from
    ``src/gnn/render/framework_registry.py``. The manifest must never restate the
    split in prose — that is how ``discopy`` came to be listed as natively
    supporting the continuous family while the registry, the acceptance ledger
    and CLAUDE.md all reported it ``unsupported``.
    """
    if not axis or not specs:
        return ""
    native = [
        str(spec.get("name", key))
        for key, spec in specs.items()
        if bool(spec.get(axis, False))
    ]
    unsupported = [
        str(spec.get("name", key))
        for key, spec in specs.items()
        if not bool(spec.get(axis, False))
    ]
    if not native and not unsupported:
        return ""
    parts = []
    if native:
        parts.append(f"Native on {', '.join(native)}")
    if unsupported:
        parts.append(f"reported unsupported (not failed) on {', '.join(unsupported)}")
    return "; ".join(parts) + "."


def _render_family_table(families: list[dict], specs: dict[str, dict]) -> str:
    """Render the model-family markdown table.

    The Description cell is the manifest's authored purpose sentence plus, for
    families that declare a ``capability_axis``, a backend split generated from
    the framework registry.
    """
    rows = ["| Family | Frameworks | Description |", "|---|---|---|"]
    for fam in families:
        name = fam.get("name", "?")
        frameworks = str(fam.get("frameworks", "")).replace(",", ", ")
        desc = str(fam.get("description", "")).strip()
        clause = _capability_clause(str(fam.get("capability_axis", "")), specs)
        if clause:
            desc = f"{desc} {clause}".strip()
        rows.append(f"| `{name}` | {frameworks} | {desc} |")
    rows.append(
        _caption(
            "Model families declared in `input/model_family_manifest.json` "
            "with the simulation frameworks each family targets and a "
            "generated description. The *Frameworks* column is the family's "
            "declared targets, verbatim from the manifest; the capability "
            "splits appended to some descriptions are generated from "
            "`src/gnn/render/framework_registry.py` flags, not authored in the "
            "manifest, so they cannot disagree with the registry. Read with "
            "[@tbl:backend_registry] (what each backend is) and "
            "[@fig:family_matrix] (the same coverage as a matrix).",
            "model_families",
        )
    )
    return "\n".join(rows)


def _render_backend_table(backends: list[tuple[str, str, bool]]) -> str:
    """Render the backend registry markdown table."""
    rows = ["| Registry key | Backend | Executes |", "|---|---|---|"]
    n_executes = sum(1 for _, _, executes in backends if executes)
    for key, name, executes in backends:
        rows.append(f"| `{key}` | {name} | {'yes' if executes else 'render-only'} |")
    rows.append(
        _caption(
            "Render targets declared in `src/gnn/render/framework_registry.py`, "
            "one row per registry key in registry order. The *Executes* column "
            "is the registry's own `supports_execution` flag: "
            f"{n_executes} of {len(backends)} backends have a Step-12 executor, "
            "and a `render-only` entry would mean generated code with no "
            "Step-12 executor. Read this table against [@tbl:model_families], "
            "which shows which families actually target each backend.",
            "backend_registry",
        )
    )
    return "\n".join(rows)


def _render_framework_capability_table(specs: dict[str, dict]) -> str:
    """Render the per-framework model-kind capability table.

    *Discrete render* requires ``pomdp_compatible`` and excludes
    ``continuous_only`` targets. *Continuous render* is ``supports_continuous``,
    and *Executor status* is ``supports_execution`` (whether a Step-12
    executor exists). An ``unsupported`` continuous cell is the renderer's
    own status for continuous-state models — reported as unsupported, not
    failed.
    """
    rows = [
        "| Framework | Discrete render | Continuous render | Executor status |",
        "|---|---|---|---|",
    ]
    for key, spec in specs.items():
        name = str(spec.get("name", key))
        discrete = (
            "yes"
            if spec.get("pomdp_compatible") and not spec.get("continuous_only")
            else "no"
        )
        continuous = "yes" if spec.get("supports_continuous") else "unsupported"
        executor = "executor" if spec.get("supports_execution") else "render-only"
        rows.append(f"| {name} | {discrete} | {continuous} | {executor} |")
    rows.append(
        _caption(
            "Model kinds each render framework supports, read entirely from "
            "the flags in `src/gnn/render/framework_registry.py` — the same "
            "source as [@tbl:backend_registry]. A continuous cell of "
            "`unsupported` is the renderer's own status for continuous-state "
            "models; continuous-only targets do not render categorical "
            "models. An executor entry declares dispatch support, not "
            "acceptance of every model or inference mode; a render-only "
            "entry has no Step-12 executor. Read with "
            "[@tbl:model_kinds], which groups the same flags by model kind.",
            "framework_capability",
        )
    )
    return "\n".join(rows)


def _render_model_kind_table(specs: dict[str, dict]) -> str:
    """Render the model-kind table the generalized manuscript is built around.

    One row per model kind the pipeline represents and executes. The
    Renderer(s)/Executor(s) cells are generated from the registry's
    ``pomdp_compatible``/``supports_continuous``/``supports_execution``
    flags, excluding continuous-only targets from categorical coverage.
    The registry has no universal multi-agent capability flag. Multi-agent
    composition is model- and backend-dependent and cannot inherit a
    categorical renderer's coverage. The recursive row is an execution mode
    with no notation or render target of its own.
    """
    rows = [
        "| Model kind | Notation block | Exemplar folder | Renderer(s) | Executor(s) |",
        "|---|---|---|---|---|",
    ]
    discrete_renderers: list[str] = []
    discrete_executors: list[str] = []
    continuous_renderers: list[str] = []
    continuous_executors: list[str] = []
    for key, spec in specs.items():
        name = str(spec.get("name", key))
        if spec.get("pomdp_compatible") and not spec.get("continuous_only"):
            discrete_renderers.append(name)
            if spec.get("supports_execution"):
                discrete_executors.append(name)
        if spec.get("supports_continuous"):
            continuous_renderers.append(name)
            if spec.get("supports_execution"):
                continuous_executors.append(name)

    def _coverage_cell(names: list[str], universe: int) -> str:
        if not names:
            return "—"
        if len(names) == universe:
            return f"all {universe} registry frameworks"
        return ", ".join(names)

    n_frameworks = len(specs)
    rows.extend(
        [
            (
                f"| Discrete categorical | `A`/`B`/`C`/`D`[/`E`] "
                f"(column-stochastic `B` slices) | "
                f"`input/gnn_files/discrete/` "
                f"| {_coverage_cell(discrete_renderers, n_frameworks)} "
                f"| {_coverage_cell(discrete_executors, n_frameworks)} |"
            ),
            (
                f"| Continuous linear-Gaussian | `F`/`H`/`Q`/`R` + "
                f"priors (optional closed-loop pair) "
                f"| `input/gnn_files/continuous/` "
                f"| {_coverage_cell(continuous_renderers, n_frameworks)} "
                f"| {_coverage_cell(continuous_executors, n_frameworks)} |"
            ),
            (
                "| Multi-agent | `nr_agents` + per-agent categorical or "
                "linear-Gaussian parameters | `input/gnn_files/multiagent/` "
                "| Model/backend dependent composition "
                "| Model/backend dependent; explicit acceptance required |"
            ),
            ("| Recursive | — | `input/gnn_files/recursive/` | — | — |"),
        ]
    )
    rows.append(
        _caption(
            "The model kinds the pipeline represents and executes, one row "
            "per kind. Notation blocks are the parameterization each kind "
            "declares; exemplar folders are the committed corpus under "
            "`input/gnn_files/`. Renderer(s)/Executor(s) cells are generated "
            "from the registry flags (see [@tbl:framework_capability]): the "
            "two parameterization rows carry their own flag sets, excluding "
            "continuous-only targets from categorical coverage. These flags "
            "do not establish universal multi-agent support: THRML accepts "
            "independent categorical components, and coupled models require "
            "a backend-specific representation and acceptance evidence. "
            "Native independent linear-Gaussian agent exemplars are verified "
            "on JAX and RxInfer in the v4 acceptance ledger; `recursive/` is "
            "reserved for bounded `--autonomous` proposal-loop runs and "
            "holds no committed models.",
            "model_kinds",
        )
    )
    return "\n".join(rows)


def select_cross_framework_family(families: list[dict]) -> dict | None:
    """Return the manifest family used as the cross-framework reference.

    The cross-framework family is the one the manifest marks
    ``"cross_framework": true``; when no family declares the flag the widest
    family (most frameworks) is used, so the selection cannot silently move to a
    different family just because the manifest's ordering changed. ``None`` when
    no family lists more than one framework.

    Public because every surface that highlights the cross-framework set —
    the token map here and ``scripts/manuscript_fig_backend_matrix.py`` — must
    select the *same* family. Two independent implementations is how the figure
    came to highlight a union of every multi-framework family while the prose
    described one.
    """
    multi = [f for f in families if "," in str(f.get("frameworks", ""))]
    if not multi:
        return None
    flagged = [f for f in multi if f.get("cross_framework") is True]
    if flagged:
        return flagged[0]
    return max(multi, key=lambda f: len(str(f.get("frameworks", "")).split(",")))


def _framework_keys(family: Mapping[str, object] | None) -> list[str]:
    """Split a family's comma-separated ``frameworks`` field into keys."""
    if not family:
        return []
    return [
        k.strip() for k in str(family.get("frameworks", "")).split(",") if k.strip()
    ]


def _cross_framework_selection(
    families: list[dict],
    backends: list[tuple[str, str, bool]],
    maintained: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    """Resolve the cross-framework family and its declared/profiled backends.

    Returns ``(family_name, declared_keys, profiled_keys)``. ``profiled_keys``
    is ``declared_keys`` intersected with the reliability gate's
    ``MAINTAINED_FRAMEWORKS``: a framework the manifest declares but the gate
    refuses to profile (``stan`` today) is *not* one of the engines the
    reference comparison runs on, and the manuscript must never count it as
    one.

    Raises:
        ValueError: If the selected family declares frameworks but none of them
            are profiled by the gate — a manifest/gate contradiction that would
            otherwise ship as an empty backend list in the manuscript.
    """
    family = select_cross_framework_family(families)
    if family is None:
        return "", [], []
    declared = _framework_keys(family)
    if not maintained:  # registry unreadable (tarball); do not silently drop
        return str(family.get("name", "")), declared, declared
    profiled = [key for key in declared if key in set(maintained)]
    if declared and not profiled:
        raise ValueError(
            f"cross-framework family {family.get('name', '?')!r} declares "
            f"{declared} but the reliability gate profiles none of them "
            f"(MAINTAINED_FRAMEWORKS={list(maintained)}); fix "
            "input/model_family_manifest.json or "
            "src/gnn/pipeline/cross_framework_reliability.py before rendering"
        )
    return str(family.get("name", "")), declared, profiled
