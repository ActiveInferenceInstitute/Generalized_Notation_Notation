#!/usr/bin/env python3
"""Declaration-reading audit for executed belief histories.

Judges an executed artifact's recorded belief history against the reading the
repository declares, and raises a layout finding when only an alternative
reading explains it. The declaration is read from the repository, never fitted
to the artifact:

- ``A`` is ``[observation, state]``. ``gnn.render.jax.jax_spec_extract`` refuses
  any other shape ("A must be 2-D [observation, state]"), every emitted filter
  reads ``A[observation, :]``, and the framework READMEs state the same axes.
- ``B`` is ``(next_state, previous_state, action)``:
  ``gnn.render.pomdp_contract.CANONICAL_B_ORDER``, stamped into
  ``model_parameters["b_tensor_order"]``; ``gnn.render.execution_contracts``
  refuses an artifact that lacks the explicit canonical value.

Per cell the tool reports numbers: the row-sum error, the first-posterior error
under both ``A`` readings, and the recursion error for the four layout readings
(``A`` row/column x ``B`` as-is/transposed) under both temporal conventions
(prior-first "offset", observe-first "classic"). The verdict is decided by the
declared reading alone (``--atol``, default 1e-6):

- ``declared-reading-reproduced``: the declared layout reproduces under at
  least one temporal convention. Alternative readings are still reported per
  cell; a symmetric likelihood makes several readings reproduce, and that is a
  report, not a finding.
- ``layout-finding``: the declared layout does not reproduce while an
  alternative layout does. The artifact is only explicable under a transposed
  layout - a renderer layout defect, named with its evidence.
- ``not-reproduced``: no layout reproduces. Reported, never a silent pass;
  frames that run learning or Monte-Carlo inference are expected here.

Exit status: 1 when any layout finding is raised, 0 when every judged cell
reproduced, 2 otherwise (nothing judged, or judged cells did not reproduce
without a layout finding). ``--soft`` prints the report and always exits 0.
Only frames named by ``--exact-frames`` (default ``jax,pytorch``; measured
2026-11-04: 23/23 discrete PyTorch cells and the JAX frame reproduce an exact
filter) are judged for reproduction, because other frames' executors run
learning or Monte-Carlo inference. Layout findings are raised for every frame:
an artifact that fits a transposed layout and not the declared one is a layout
signal whatever the inference scheme.

Superseded instruments (local audit tools, 2026-11-03/04). Their false
negatives are why this tool judges instead of fits:

- ``audit_replay.py`` fixed one ``A`` reading (``A[:, o]``) across its four
  variants, so cells whose artifacts follow the declared row reading failed:
  measured on the audit run root it reported 18 of the 45 comparable
  pymdp+numpyro cells and 9 of the 23 discrete PyTorch cells ``not-reproduced``
  although all reproduce to <= 2.22e-16 under the declared reading.
- ``audit_consistency.py`` tried both ``A`` readings and took the best as the
  pass path (``best_recursion <= 1e-6``). Fitting cannot separate a renderer
  that follows the declaration from one that transposes it: a genuine
  transposed-``A`` renderer bug passes as ``exact-filter-reproduced``. This
  tool's verdict for that artifact is ``layout-finding``.

Results discovery: every ``*simulation_results.json`` under the execute
directory, read for the recorded ``beliefs``/``actions``/``observations`` (or a
nested ``simulation_trace``). Frame and model come from the path, for the
pipeline layout ``<model>/<frame>/simulation_data/simulation_results.json`` and
for the audit harness layout ``<frame>/<model>/simulation_results.json``.
Declared matrices come from the canonical spec the renderer was handed
(``gnn.extract`` + ``gnn.render.pomdp_processor``), matched to the recorded
belief width; an ambiguous or missing model source is reported, never guessed.

Usage:

    uv run --no-sync python scripts/audit_declared_reading.py \
        --execute-dir output/12_execute_output \
        --models-dir input/gnn_files
    uv run --no-sync python scripts/audit_declared_reading.py \
        --execute-dir output/12_execute_output --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "input" / "gnn_files"
DEFAULT_ATOL = 1e-6
DEFAULT_EXACT_FRAMES = ("jax", "pytorch")

# Declared readings; the module docstring names their repository sites.
DECLARED_A_READING = "row"  # A[observation, state]
DECLARED_B_ORDER = "next_state_previous_state_action"
DECLARED_READING = "A[observation, state] B[next_state, previous_state, action]"
LAYOUT_READINGS: tuple[tuple[str, str, bool], ...] = (
    ("declared", DECLARED_A_READING, False),
    ("b-transposed", DECLARED_A_READING, True),
    ("a-column", "col", False),
    ("a-column+b-transposed", "col", True),
)
TEMPORAL_CONVENTIONS = ("offset", "classic")


def normalize(vector: np.ndarray) -> np.ndarray:
    total = vector.sum()
    return vector / total if total else vector


def transition_slice(
    b_matrix: np.ndarray, action: int, *, transposed: bool
) -> np.ndarray:
    """``B[next_state, previous_state]`` for one action, declared or flipped."""
    if b_matrix.ndim == 2:
        slice_2d = b_matrix
    else:
        slice_2d = b_matrix[:, :, action]
    return slice_2d.T if transposed else slice_2d


def likelihood(a_matrix: np.ndarray, observation: int, *, reading: str) -> np.ndarray:
    """Observation likelihood over states under one ``A`` reading."""
    if reading == "row":
        return a_matrix[observation]  # declared A[observation, state]
    return a_matrix[:, observation]  # alternative A[state, observation]


def filter_recursion(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    d_vector: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    *,
    variant: str,
    reading: str = DECLARED_A_READING,
    transposed: bool = False,
) -> np.ndarray:
    """Belief history under one layout reading and temporal convention."""

    def observe(predicted: np.ndarray, observation: int) -> np.ndarray:
        return normalize(
            predicted * likelihood(a_matrix, int(observation), reading=reading)
        )

    if variant == "offset":
        first = likelihood(a_matrix, int(observations[0]), reading=reading)
        posterior = normalize(first * normalize(d_vector))
        history = [posterior]
        for index in range(len(actions) - 1):
            predicted = normalize(
                transition_slice(b_matrix, int(actions[index]), transposed=transposed)
                @ posterior
            )
            posterior = observe(predicted, observations[index + 1])
            history.append(posterior)
        return np.asarray(history)
    posterior = normalize(d_vector)
    history = [posterior]
    for index in range(1, len(observations)):
        predicted = normalize(
            transition_slice(b_matrix, int(actions[index - 1]), transposed=transposed)
            @ posterior
        )
        posterior = observe(predicted, observations[index])
        history.append(posterior)
    return np.asarray(history)


def _max_deviation(history: np.ndarray, beliefs: np.ndarray) -> float | None:
    steps = min(len(history), len(beliefs))
    if steps == 0 or history[0].shape != beliefs[0].shape:
        return None
    deviations = [
        float(np.max(np.abs(normalize(history[index]) - normalize(beliefs[index]))))
        for index in range(steps)
    ]
    return max(deviations)


def layout_errors(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    d_vector: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    beliefs: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Recursion error per layout reading, per temporal convention."""
    errors: dict[str, dict[str, float | None]] = {}
    for label, reading, transposed in LAYOUT_READINGS:
        per_convention: dict[str, float | None] = {}
        for variant in TEMPORAL_CONVENTIONS:
            try:
                history = filter_recursion(
                    a_matrix,
                    b_matrix,
                    d_vector,
                    actions,
                    observations,
                    variant=variant,
                    reading=reading,
                    transposed=transposed,
                )
            except (IndexError, ValueError):
                per_convention[variant] = None
                continue
            per_convention[variant] = _max_deviation(history, beliefs)
        errors[label] = per_convention
    return errors


def _best(per_convention: Mapping[str, float | None]) -> float | None:
    values = [value for value in per_convention.values() if value is not None]
    return min(values) if values else None


def judge_cell(
    name: str,
    *,
    a_matrix: Any,
    b_matrix: Any,
    d_vector: Any,
    actions: Any,
    observations: Any,
    beliefs: Any,
    frame: str = "",
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    """Judge one executed belief history against the declared reading."""
    record: dict[str, Any] = {"cell": name, "frame": frame}
    a = np.asarray(a_matrix, dtype=float)
    b = np.asarray(b_matrix, dtype=float)
    d = np.asarray(d_vector, dtype=float).reshape(-1)
    history = np.asarray(beliefs, dtype=float)
    action_list = np.asarray(actions, dtype=int).reshape(-1)
    observation_list = np.asarray(observations, dtype=int).reshape(-1)
    record["shapes"] = {
        "A": list(a.shape),
        "B": list(b.shape),
        "beliefs": list(history.shape),
    }
    if (
        a.ndim != 2
        or history.ndim != 2
        or history.size == 0
        or a.shape[1] != len(d)
        or history.shape[1] != len(d)
    ):
        record["outcome"] = "not-comparable"
        record["reason"] = (
            "declared A/D width does not match the recorded belief width "
            f"(A {list(a.shape)}, D {len(d)}, beliefs {list(history.shape)})"
        )
        return record
    if len(action_list) == 0 or len(observation_list) == 0:
        record["outcome"] = "not-comparable"
        record["reason"] = "no recorded actions or observations"
        return record
    record["steps"] = int(len(history))
    record["rows_sum_to_one"] = float(np.abs(history.sum(axis=1) - 1.0).max())
    first_options: dict[str, float] = {}
    for label, reading in (("A[o,:]", "row"), ("A[:,o]", "col")):
        try:
            expected = normalize(
                likelihood(a, int(observation_list[0]), reading=reading) * normalize(d)
            )
        except IndexError:
            continue
        if expected.shape == history[0].shape:
            first_options[label] = float(
                np.max(np.abs(normalize(expected) - normalize(history[0])))
            )
    record["first_belief"] = first_options
    errors = layout_errors(a, b, d, action_list, observation_list, history)
    record["layout_errors"] = errors
    record["declared_error"] = _best(errors["declared"])
    alternatives = {
        label: _best(per_convention)
        for label, per_convention in errors.items()
        if label != "declared"
    }
    record["alternative_errors"] = alternatives
    reproduced = sorted(
        label
        for label, value in alternatives.items()
        if value is not None and value <= atol
    )
    record["alternative_readings_reproduced"] = reproduced
    record["beliefs_vary"] = int(len({tuple(np.round(row, 9)) for row in history}))
    declared_error = record["declared_error"]
    if declared_error is not None and declared_error <= atol:
        record["outcome"] = "declared-reading-reproduced"
        record["alternative_also_reproduces"] = bool(reproduced)
        return record
    if reproduced:
        record["outcome"] = "layout-finding"
        record["finding"] = {
            "declared_reading": DECLARED_READING,
            "declared_error": declared_error,
            "alternative_readings": reproduced,
            "alternative_errors": {label: alternatives[label] for label in reproduced},
        }
        return record
    record["outcome"] = "not-reproduced"
    return record


def _derive_cell_names(results_path: Path) -> tuple[str, str]:
    """``(frame, model)`` for the pipeline and audit-harness result layouts."""
    parent = results_path.parent
    if parent.name == "simulation_data":
        return parent.parent.name, parent.parent.parent.name
    return parent.parent.name, parent.name


def _belief_payload(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorded beliefs/actions/observations from a results payload."""
    trace = payload.get("simulation_trace") or {}
    beliefs = payload.get("beliefs")
    actions = payload.get("actions")
    observations = payload.get("observations")
    beliefs = beliefs if beliefs is not None else trace.get("beliefs")
    actions = actions if actions is not None else trace.get("actions")
    observations = (
        observations if observations is not None else trace.get("observations")
    )
    return (
        np.asarray(beliefs, dtype=float),
        np.asarray(actions, dtype=int).reshape(-1),
        np.asarray(observations, dtype=int).reshape(-1),
    )


def _model_source_index(models_dir: Path) -> dict[str, list[Path]]:
    """Map model stems to their source files under the corpus."""
    index: dict[str, list[Path]] = {}
    for path in sorted(models_dir.rglob("*.md")):
        if path.name in {"README.md", "INDEX.md", "CHANGELOG.md", "LICENSE.md"}:
            continue
        index.setdefault(path.stem, []).append(path)
    return index


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _declared_matrices(
    source: Path, *, beliefs_width: int
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """``(container, A, B, D)`` from the canonical spec the renderer was handed."""
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    space = extract_pomdp_from_file(source, strict_validation=False)
    spec = pomdp_to_gnn_spec(
        space, native_agents=False, preserve_discrete_structure=False
    )
    containers = {
        "initialparameterization": spec.get("initialparameterization") or {},
        "structured_pomdp.matrices": (spec.get("structured_pomdp") or {}).get(
            "matrices"
        )
        or {},
    }
    for name, container in containers.items():
        if not {"A", "B", "D"} <= set(container):
            continue
        a_matrix = np.asarray(container["A"], dtype=float)
        d_vector = np.asarray(container["D"], dtype=float).reshape(-1)
        if a_matrix.ndim == 2 and a_matrix.shape[1] == beliefs_width == len(d_vector):
            b_matrix = np.asarray(container["B"], dtype=float)
            if name == "structured_pomdp.matrices" and b_matrix.ndim == 3:
                # The structured container is action-major; the filter reads
                # (next state, previous state, action).
                b_matrix = np.moveaxis(b_matrix, 0, 2)
            return name, a_matrix, b_matrix, d_vector
    raise ValueError("no declared container matches the recorded belief width")


def audit(
    execute_dir: Path,
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    frames: Sequence[str] | None = None,
    atol: float = DEFAULT_ATOL,
    exact_frames: Sequence[str] = DEFAULT_EXACT_FRAMES,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Judge every executed result under ``execute_dir``."""
    records: list[dict[str, Any]] = []
    sources = _model_source_index(models_dir)
    exact = set(exact_frames)
    for results_path in sorted(execute_dir.rglob("*simulation_results.json")):
        frame, model = _derive_cell_names(results_path)
        if frames and frame not in frames:
            continue
        cell = f"{frame}/{model}"
        record: dict[str, Any] = {
            "cell": cell,
            "frame": frame,
            "model": model,
            "results": str(results_path),
        }
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            beliefs, actions, observations = _belief_payload(payload)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record["outcome"] = "not-comparable"
            record["reason"] = f"unreadable results ({type(exc).__name__}: {exc})"
            records.append(record)
            continue
        candidates = sources.get(model) or []
        if len(candidates) != 1:
            record["outcome"] = "not-comparable"
            record["reason"] = (
                f"{len(candidates)} model sources named {model!r} under {models_dir}"
            )
            records.append(record)
            continue
        source = candidates[0]
        record["source"] = _display_path(source, repo_root)
        try:
            container, a_matrix, b_matrix, d_vector = _declared_matrices(
                source,
                beliefs_width=int(beliefs.shape[1]) if beliefs.ndim == 2 else 0,
            )
        except Exception as exc:  # noqa: BLE001 - availability is a reported state
            record["outcome"] = "not-comparable"
            record["reason"] = (
                f"canonical spec unavailable ({type(exc).__name__}: {exc})"
            )
            records.append(record)
            continue
        record["container"] = container
        judged = judge_cell(
            cell,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            d_vector=d_vector,
            actions=actions,
            observations=observations,
            beliefs=beliefs,
            frame=frame,
            atol=atol,
        )
        record.update(judged)
        if record.get("outcome") != "not-comparable":
            record["judged"] = frame in exact
        records.append(record)
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    findings = [
        record for record in records if record.get("outcome") == "layout-finding"
    ]
    judged = [
        record
        for record in records
        if record.get("judged") and record.get("outcome") != "not-comparable"
    ]
    reproduced = [
        record
        for record in judged
        if record.get("outcome") == "declared-reading-reproduced"
    ]
    return {
        "cells": len(records),
        "not_comparable": sum(
            1 for record in records if record.get("outcome") == "not-comparable"
        ),
        "judged": len(judged),
        "reproduced": len(reproduced),
        "not_reproduced": sum(
            1 for record in judged if record.get("outcome") == "not-reproduced"
        ),
        "layout_findings": len(findings),
        "finding_cells": [record["cell"] for record in findings],
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1e}"


def render_text(records: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        cell = str(record.get("cell", "?"))
        if record.get("outcome") == "not-comparable":
            lines.append(f"{cell:<44s} not-comparable  {record.get('reason', '')}")
            continue
        first = record.get("first_belief") or {}
        reproduced = record.get("alternative_readings_reproduced") or []
        lines.append(
            f"{cell:<44s} {str(record.get('outcome')):<28s} "
            f"rows_sum={_fmt(record.get('rows_sum_to_one'))} "
            f"first_belief={_fmt(min(first.values()) if first else None)} "
            f"declared={_fmt(record.get('declared_error'))} "
            f"alternatives_reproduced={','.join(reproduced) or '-'}"
        )
    summary = summarize(records)
    lines.append("")
    lines.append(
        f"cells={summary['cells']} judged={summary['judged']} "
        f"reproduced={summary['reproduced']} "
        f"not_reproduced={summary['not_reproduced']} "
        f"not_comparable={summary['not_comparable']} "
        f"layout_findings={summary['layout_findings']}"
    )
    for record in records:
        finding = record.get("finding")
        if not finding:
            continue
        evidence = ", ".join(
            f"{label}={_fmt(value)}"
            for label, value in finding["alternative_errors"].items()
        )
        lines.append(
            f"layout finding {record.get('cell')}: declared reading "
            f"{finding['declared_reading']} error "
            f"{_fmt(finding['declared_error'])}; reproduced under "
            f"{', '.join(finding['alternative_readings'])} ({evidence})"
        )
    return "\n".join(lines) + "\n"


def render_json(records: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "declared_reading": DECLARED_READING,
        "summary": summarize(records),
        "cells": list(records),
    }
    return json.dumps(payload, indent=2) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    cells: Sequence[dict[str, Any]] | None = None,
    execute_dir: Path | None = None,
    models_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Run the audit; ``cells`` injects judged records (test seam).

    Mirrors ``scripts/audit_line_bands.py``'s injection seam: tests pass
    hand-built cells instead of a run directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Judge executed belief histories against the declared filter reading."
        )
    )
    parser.add_argument(
        "--execute-dir",
        type=Path,
        default=None,
        help="directory holding executed *simulation_results.json files",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"corpus searched for model sources (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--frames", default="", help="comma-separated frame filter")
    parser.add_argument(
        "--exact-frames",
        default=",".join(DEFAULT_EXACT_FRAMES),
        help="frames judged for exact-filter reproduction",
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument("--soft", action="store_true", help="always exit 0")
    args = parser.parse_args(argv)

    exact_frames = [name for name in args.exact_frames.split(",") if name]
    records: list[dict[str, Any]]
    if cells is not None:
        records = [dict(record) for record in cells]
    else:
        target = execute_dir if execute_dir is not None else args.execute_dir
        if target is None:
            parser.error("--execute-dir is required")
        frames = [name for name in args.frames.split(",") if name]
        records = audit(
            target,
            models_dir=models_dir or args.models_dir or DEFAULT_MODELS_DIR,
            frames=frames or None,
            atol=args.atol,
            exact_frames=exact_frames,
            repo_root=repo_root,
        )
    for record in records:
        record.setdefault("judged", record.get("frame") in exact_frames)
    print(render_json(records) if args.json else render_text(records), end="")
    if args.soft:
        return 0
    summary = summarize(records)
    if summary["layout_findings"]:
        return 1
    if summary["judged"] and summary["reproduced"] == summary["judged"]:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
