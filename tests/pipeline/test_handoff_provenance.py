"""Versioned Step 3 -> 16 model/result handoffs and COGANT summary readers.

Covers the ``gnn_parsed_model_v1`` admission at Steps 7, 8, 9 and 11, the
``gnn_execution_handoff_v1`` identity Step 12 stamps onto every execution
detail, and its admission by the Step 15 sweep collector and Step 16 analysis:
current handoffs are admitted with their provenance, stale and mixed-run
handoffs fail closed, and historical unversioned fixtures keep their historical
readers when no run context governs them. The COGANT contract test feeds the
emitted summaries to COGANT's mtime and invocation/run-ID summary readers.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from gnn.pipeline.artifact_ownership import (
    EXECUTION_HANDOFF_SCHEMA,
    admit_execution_summary,
    changed_artifacts,
)
from gnn.pipeline.parse_receipt import PARSED_MODEL_SCHEMA
from gnn.pipeline.run_context import build_run_context

pytestmark = pytest.mark.pipeline

EXEMPLAR = Path(__file__).parents[2] / "input/gnn_files/discrete/simple_mdp.md"
LOGGER = logging.getLogger(__name__)


def _activate(context: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path(context.output_root) / "context.json"
    context.write(path)
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(path))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)


def _inventory(output: Path, run_id: str, *directories: Path) -> None:
    """Publish the current-run artifact inventory the way ``gnn.main`` does."""
    path = output / "00_pipeline_summary" / "current_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "steps": [
                    {"artifacts": changed_artifacts(directory, output, {})}
                    for directory in directories
                ],
            }
        ),
        encoding="utf-8",
    )


def _rehash_record(manifest_path: Path, artifact: Path) -> None:
    manifest = json.loads(manifest_path.read_bytes())
    (record,) = manifest["processed_files"]
    record["parsed_model_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 3 -> Steps 7, 8, 9 and 11
# ---------------------------------------------------------------------------


def _run_export(view: Path, output: Path) -> tuple[Any, list[Any]]:
    from gnn.export.processor import process_export

    step_dir = output / "7_export_output"
    outcome = process_export(view, step_dir, formats=["json"])
    receipt = json.loads((step_dir / "export_results.json").read_bytes())
    return outcome, [
        (entry.get("parsed_input"), entry.get("error"))
        for entry in receipt["files_exported"]
    ]


def _run_visualization(view: Path, output: Path) -> tuple[Any, list[Any]]:
    from gnn.visualization import process_visualization

    step_dir = output / "8_visualization_output"
    outcome = process_visualization(view, step_dir)
    manifests = sorted(step_dir.rglob("*_viz_manifest.json"))
    return outcome, [
        (json.loads(path.read_bytes())["viz_meta"].get("parsed_input"), None)
        for path in manifests
    ]


def _run_advanced_visualization(view: Path, output: Path) -> tuple[Any, list[Any]]:
    from gnn.advanced_visualization.processor import process_advanced_viz

    step_dir = output / "9_advanced_viz_output"
    outcome = process_advanced_viz(view, step_dir, LOGGER, viz_type="statistical")
    summary_path = step_dir / "advanced_viz_summary.json"
    summary = json.loads(summary_path.read_bytes()) if summary_path.is_file() else {}
    return outcome, [
        (identity, None) for identity in (summary.get("parsed_inputs") or {}).values()
    ]


def _run_render(view: Path, output: Path) -> tuple[Any, list[Any]]:
    from gnn.render.processor import process_render

    step_dir = output / "11_render_output"
    outcome = process_render(view, step_dir, frameworks=["pymdp"])
    summary = json.loads((step_dir / "render_processing_summary.json").read_bytes())
    return outcome, [
        (record.get("parsed_input"), record.get("error"))
        for record in summary["file_results"].values()
    ]


CONSUMERS: dict[int, Callable[[Path, Path], tuple[Any, list[Any]]]] = {
    7: _run_export,
    8: _run_visualization,
    9: _run_advanced_visualization,
    11: _run_render,
}


@pytest.mark.parametrize("step", sorted(CONSUMERS))
@pytest.mark.parametrize("mode", ["current", "tampered", "mixed-run"])
def test_step3_handoff_admission_at_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: int, mode: str
) -> None:
    from gnn.processing.multi_format_processor import process_gnn_multi_format

    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_bytes(EXEMPLAR.read_bytes())
    output = tmp_path / "output"
    context = build_run_context(source, output, "handoff-run", [3, step], {})
    _activate(context, monkeypatch)

    step3_dir = output / "3_gnn_output"
    assert process_gnn_multi_format(
        context.input_view(3), output, LOGGER, serialize_preset="minimal"
    )
    manifest_path = step3_dir / "gnn_processing_results.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["parsed_model_schema"] == PARSED_MODEL_SCHEMA
    (record,) = manifest["processed_files"]
    artifact = Path(record["parsed_model_file"])
    payload = json.loads(artifact.read_bytes())
    if mode == "mixed-run":
        payload["_gnn_provenance"]["run_id"] = "another-run"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        _rehash_record(manifest_path, artifact)
    _inventory(output, context.run_id, step3_dir)
    if mode == "tampered":
        payload["model_name"] = "Different model"
        artifact.write_text(json.dumps(payload), encoding="utf-8")

    outcome, admitted = CONSUMERS[step](context.input_view(step), output)
    record = json.loads(manifest_path.read_bytes())["processed_files"][0]
    if mode == "current":
        assert outcome, admitted
        assert admitted, "consumer emitted no provenance"
        for identity, _ in admitted:
            assert identity == {
                **payload["_gnn_provenance"],
                "parsed_model_sha256": record["parsed_model_sha256"],
            }
            assert identity["run_id"] == context.run_id
            assert identity["source_sha256"] == context.models[0].sha256
    else:
        assert outcome is False
        assert all(identity is None for identity, _ in admitted)


def test_standalone_unversioned_parse_receipt_keeps_flat_readers(
    tmp_path: Path,
) -> None:
    """Historical unversioned Step 3 output is read through the migration path."""
    from gnn.processing.multi_format_processor import process_gnn_multi_format

    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_bytes(EXEMPLAR.read_bytes())
    output = tmp_path / "output"
    assert process_gnn_multi_format(source, output, LOGGER, serialize_preset="minimal")
    manifest_path = output / "3_gnn_output" / "gnn_processing_results.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.pop("parsed_model_schema")
    for record in manifest["processed_files"]:
        record.pop("parsed_model_sha256", None)
        artifact = Path(record["parsed_model_file"])
        payload = json.loads(artifact.read_bytes())
        payload.pop("_gnn_provenance", None)
        artifact.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for step in (7, 8, 11):
        outcome, admitted = CONSUMERS[step](source, output)
        assert outcome, (step, admitted)
        assert all(identity is None for identity, _ in admitted)


def test_render_refuses_source_edited_after_parse(tmp_path: Path) -> None:
    from gnn.processing.multi_format_processor import process_gnn_multi_format

    source = tmp_path / "input"
    source.mkdir()
    model = source / "model.md"
    model.write_bytes(EXEMPLAR.read_bytes())
    output = tmp_path / "output"
    assert process_gnn_multi_format(source, output, LOGGER, serialize_preset="minimal")
    model.write_bytes(model.read_bytes() + b"\nEdited after parsing.\n")
    outcome, admitted = _run_render(source, output)
    assert outcome is False
    ((identity, error),) = admitted
    assert identity is None
    assert "Step 3 handoff refused" in error


# ---------------------------------------------------------------------------
# Step 12 -> Steps 15 and 16
# ---------------------------------------------------------------------------


def _execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None,
    *,
    run_id: str = "execution-run",
) -> tuple[Any, Path, dict[str, Any]]:
    """Write a real Step 12 summary for one rendered PyMDP script."""
    from gnn.execute.processor.summary import (
        _init_execution_summary,
        _write_execution_summaries,
    )

    source = tmp_path / "input"
    source.mkdir(exist_ok=True)
    (source / "model.md").write_bytes(EXEMPLAR.read_bytes())
    output = tmp_path / "output"
    context = build_run_context(source, output, run_id, [11, 12, 15, 16], {})
    if monkeypatch is not None:
        _activate(context, monkeypatch)
    (model,) = context.models
    render_dir = output / "11_render_output"
    script = render_dir / model.artifact_stem / "pymdp" / f"{model.artifact_stem}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('simulate')\n", encoding="utf-8")
    render_summary = render_dir / "render_processing_summary.json"
    render_summary.write_text(
        json.dumps(
            {
                "file_results": {
                    str((source / "model.md").resolve()): {
                        "source_identity": {
                            "path": str((source / "model.md").resolve()),
                            "sha256": model.sha256,
                        },
                        "model_id": model.model_id,
                        "source_relative_path": model.relative_path,
                        "framework_results": {
                            "pymdp": {
                                "artifact_identities": [
                                    {
                                        "path": str(script.resolve()),
                                        "sha256": hashlib.sha256(
                                            script.read_bytes()
                                        ).hexdigest(),
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    execute_dir = output / "12_execute_output"
    sim = execute_dir / model.artifact_stem / "pymdp" / "simulation_results.json"
    sim.parent.mkdir(parents=True, exist_ok=True)
    sim.write_text(
        json.dumps({"success": True, "num_timesteps": 3, "metrics": {}}),
        encoding="utf-8",
    )
    results = _init_execution_summary(render_dir, execute_dir, 1, False)
    results["run_id"] = run_id
    results["execution_details"] = [
        {
            "script_path": str(script),
            "script_name": script.name,
            "framework": "pymdp",
            "executor": "python",
            "model_name": model.artifact_stem,
            "success": True,
            "execution_time": 0.5,
            "timestamp": "2026-11-15T00:00:00",
        }
    ]
    _write_execution_summaries(
        execute_dir, results, False, LOGGER, render_summary=render_summary
    )
    summary = json.loads(
        (execute_dir / "summaries" / "execution_summary.json").read_bytes()
    )
    return context, execute_dir, summary


@pytest.mark.parametrize("governed", [True, False])
def test_step12_summary_stamps_deterministic_execution_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, governed: bool
) -> None:
    context, execute_dir, summary = _execution(
        tmp_path, monkeypatch if governed else None
    )
    (model,) = context.models
    assert summary["execution_handoff_schema"] == EXECUTION_HANDOFF_SCHEMA
    (detail,) = summary["execution_details"]
    script = Path(detail["script_path"])
    assert detail["_gnn_provenance"] == {
        "schema": EXECUTION_HANDOFF_SCHEMA,
        "run_id": context.run_id,
        "model_id": model.model_id,
        "artifact_stem": model.artifact_stem,
        "source_relative_path": model.relative_path,
        "source_sha256": model.sha256,
        "rendered_artifact_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    receipt = admit_execution_summary(summary) if governed else None
    if governed:
        assert receipt == {
            "schema": EXECUTION_HANDOFF_SCHEMA,
            "run_id": context.run_id,
            "admitted_details": 1,
            "models": [
                {
                    "model_id": model.model_id,
                    "artifact_stem": model.artifact_stem,
                    "source_relative_path": model.relative_path,
                    "source_sha256": model.sha256,
                }
            ],
            "unverified_rendered_artifacts": [],
        }


def _mutate(summary: dict[str, Any], mode: str, script: Path) -> None:
    (detail,) = summary["execution_details"]
    if mode == "mixed-run":
        detail["_gnn_provenance"]["run_id"] = "another-run"
    elif mode == "stale-run":
        summary["run_id"] = "previous-run"
        detail["_gnn_provenance"]["run_id"] = "previous-run"
    elif mode == "stale-source":
        detail["_gnn_provenance"]["source_sha256"] = "0" * 64
        detail.pop("source_sha256", None)
    elif mode == "stale-script":
        script.write_text("print('edited after execution')\n", encoding="utf-8")
    elif mode == "unversioned":
        summary.pop("execution_handoff_schema")
        detail.pop("_gnn_provenance")
    elif mode == "orphan-handoff":
        summary.pop("execution_handoff_schema")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("mixed-run", "mixes runs"),
        ("stale-run", "different run"),
        ("stale-source", "source is stale"),
        ("stale-script", "changed after execution"),
        ("unversioned", "not versioned"),
        ("orphan-handoff", "requires a versioned summary"),
    ],
)
def test_execution_handoff_refuses_stale_and_mixed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    _, _, summary = _execution(tmp_path, monkeypatch)
    _mutate(summary, mode, Path(summary["execution_details"][0]["script_path"]))
    with pytest.raises(ValueError, match=message):
        admit_execution_summary(summary)


def test_unversioned_execution_summary_is_admitted_only_without_a_run(
    tmp_path: Path,
) -> None:
    summary = {
        "execution_details": [
            {"model_name": "m", "framework": "pymdp", "success": True}
        ]
    }
    assert admit_execution_summary(summary) is None


def _collect(execute_dir: Path) -> tuple[Any, list[tuple[Any, ...]]]:
    from gnn.integration.meta_analysis.collector import SweepDataCollector

    collector = SweepDataCollector(execute_dir, logger=LOGGER)
    records = collector.collect()
    return collector, sorted(
        (
            r.model_name,
            r.framework,
            r.execution_time,
            r.success,
            r.num_timesteps,
            json.dumps(r.provenance, sort_keys=True),
        )
        for r in records
    )


def _plant_stale_outputs(execute_dir: Path) -> None:
    """Outputs a previous run left behind in the same output root."""
    stale = execute_dir / "old_model" / "pymdp" / "simulation_results.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"success": True}), encoding="utf-8")
    log = execute_dir / "old_model" / "execution_logs" / "old_results.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "model_name": "old_model",
                "framework": "jax",
                "execution_time": 9.0,
                "success": True,
            }
        ),
        encoding="utf-8",
    )


def test_collector_admits_versioned_handoff_and_ignores_stale_outputs(
    tmp_path: Path,
) -> None:
    context, execute_dir, _ = _execution(tmp_path, None)
    _plant_stale_outputs(execute_dir)
    collector, records = _collect(execute_dir)
    (model,) = context.models
    assert collector.handoff_refusal is None
    assert collector.handoff["run_id"] == context.run_id
    assert [record[:2] for record in records] == [(model.artifact_stem, "pymdp")]
    provenance = json.loads(records[0][-1])
    assert provenance["run_id"] == context.run_id
    assert provenance["source_sha256"] == model.sha256


@pytest.mark.parametrize("mode", ["mixed-run", "orphan-handoff", "stale-script"])
def test_collector_and_meta_analysis_fail_closed(tmp_path: Path, mode: str) -> None:
    from gnn.integration.meta_analysis import run_meta_analysis

    _, execute_dir, summary = _execution(tmp_path, None)
    summary_path = execute_dir / "summaries" / "execution_summary.json"
    _mutate(summary, mode, Path(summary["execution_details"][0]["script_path"]))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    collector, records = _collect(execute_dir)
    assert records == []
    assert collector.handoff_refusal
    meta_dir = tmp_path / "meta"
    assert run_meta_analysis(execute_dir, meta_dir, logger=LOGGER) is None
    refusal = json.loads((meta_dir / "execution_handoff.json").read_bytes())
    assert refusal["status"] == "refused"
    assert not (meta_dir / "meta_statistics.json").exists()


def test_collector_keeps_unversioned_fixture_behaviour(tmp_path: Path) -> None:
    _, execute_dir, summary = _execution(tmp_path, None)
    _mutate(summary, "unversioned", Path("unused"))
    summary_path = execute_dir / "summaries" / "execution_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _plant_stale_outputs(execute_dir)
    collector, records = _collect(execute_dir)
    assert collector.handoff is None and collector.handoff_refusal is None
    # Unversioned collection still mines execution logs and every simulation result.
    assert {record[:2] for record in records} >= {
        ("old_model", "jax"),
        ("old_model", "pymdp"),
    }


def test_collector_refuses_prior_run_summary_inside_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, execute_dir, _ = _execution(tmp_path, monkeypatch, run_id="previous-run")
    current = build_run_context(
        tmp_path / "input", tmp_path / "output", "current-run", [15, 16], {}
    )
    _activate(current, monkeypatch)
    collector, records = _collect(execute_dir)
    assert records == []
    assert "different run" in (collector.handoff_refusal or "")


def test_replay_and_resume_reproduce_identical_handoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rewriting the same run's summary (resume/replay) is byte-stable."""
    from gnn.execute.processor.summary import _write_execution_summaries

    context, execute_dir, first = _execution(tmp_path, monkeypatch)
    _, first_records = _collect(execute_dir)
    replay = dict(first)
    replay["execution_details"] = [
        {
            key: value
            for key, value in detail.items()
            if key not in {"_gnn_provenance", "script_identity", "source_scope"}
        }
        for detail in first["execution_details"]
    ]
    _write_execution_summaries(
        execute_dir,
        replay,
        False,
        LOGGER,
        render_summary=tmp_path
        / "output"
        / "11_render_output"
        / "render_processing_summary.json",
    )
    second = json.loads(
        (execute_dir / "summaries" / "execution_summary.json").read_bytes()
    )
    assert [d["_gnn_provenance"] for d in second["execution_details"]] == [
        d["_gnn_provenance"] for d in first["execution_details"]
    ]
    assert admit_execution_summary(second) == admit_execution_summary(first)
    _, second_records = _collect(execute_dir)
    assert second_records == first_records
    assert second["run_id"] == context.run_id


def test_analysis_fails_closed_on_mixed_run_execution_summary(
    tmp_path: Path,
) -> None:
    from gnn.analysis.processor import process_analysis

    _, execute_dir, summary = _execution(tmp_path, None)
    _mutate(summary, "mixed-run", Path("unused"))
    (execute_dir / "summaries" / "execution_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    results_dir = tmp_path / "output" / "16_analysis_output"
    assert process_analysis(tmp_path / "input", results_dir) is False
    results = json.loads((results_dir / "analysis_results.json").read_bytes())
    assert results["execution_handoff"]["status"] == "refused"
    assert "mixes runs" in results["execution_handoff"]["reason"]
    assert not results.get("post_simulation_analysis")


def test_analysis_records_admitted_execution_handoff(tmp_path: Path) -> None:
    from gnn.analysis.processor import process_analysis

    context, _, _ = _execution(tmp_path, None)
    results_dir = tmp_path / "output" / "16_analysis_output"
    process_analysis(tmp_path / "input", results_dir)
    results = json.loads((results_dir / "analysis_results.json").read_bytes())
    assert results["execution_handoff"]["schema"] == EXECUTION_HANDOFF_SCHEMA
    assert results["execution_handoff"]["run_id"] == context.run_id


# ---------------------------------------------------------------------------
# COGANT summary-reader compatibility
# ---------------------------------------------------------------------------

_COGANT_BRIDGE = Path("cogant/py/cogant/gnn/upstream_bridge/pipeline.py")
# Live COGANT checkouts: a sibling ``COGANT`` clone next to this repository,
# plus any listed in ``GNN_COGANT_CHECKOUTS`` (``os.pathsep``-separated). The
# vendored readers below always run, so the contract holds without COGANT.
_COGANT_CHECKOUTS = tuple(
    dict.fromkeys(
        [Path(__file__).resolve().parents[3] / "COGANT"]
        + [
            Path(entry)
            for entry in os.environ.get("GNN_COGANT_CHECKOUTS", "").split(os.pathsep)
            if entry
        ]
    )
)


def _vendored_mtime_reader(path: Path, started_at: float) -> dict[str, Any] | None:
    """COGANT's mtime summary reader (``_read_summary(path, started_at)``)."""
    try:
        if path.stat().st_mtime < started_at:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _vendored_invocation_reader(
    output_root: Path, invocation_id: str
) -> dict[str, Any] | None:
    """COGANT's invocation reader (``_read_summary(output_root, invocation_id)``)."""
    output_root = output_root.resolve()
    path = output_root / "00_pipeline_summary" / "pipeline_execution_summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(data, dict) or data.get("run_id") != invocation_id:
        raise ValueError("upstream summary run ID differs from this invocation")
    steps = data.get("steps", [])
    if any("run_id" in step and step["run_id"] != invocation_id for step in steps):
        raise ValueError("upstream step run ID differs from this invocation")
    if data.get("artifact_inventory_contract") is not None:
        if data["artifact_inventory_contract"] != "current-run-v1":
            raise ValueError("unrecognized upstream artifact inventory contract")
        for step in steps:
            for record in step.get("artifacts", []):
                relative = Path(record["path"])
                artifact = (output_root / relative).resolve()
                if relative.is_absolute() or not artifact.is_relative_to(output_root):
                    raise ValueError("upstream evidence path escapes")
                with artifact.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != record.get("sha256"):
                    raise ValueError(f"upstream artifact checksum differs: {relative}")
    return data


def _live_cogant_readers() -> list[tuple[str, Callable[..., Any]]]:
    """Extract COGANT's stdlib-only summary readers from local checkouts."""
    readers = []
    for checkout in _COGANT_CHECKOUTS:
        bridge = checkout / _COGANT_BRIDGE
        if not bridge.is_file():
            continue
        tree = ast.parse(bridge.read_text(encoding="utf-8"))
        wanted = {"_evidence_path", "_verify_inventory", "_read_summary"}
        body: list[ast.stmt] = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in wanted)
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "_SUMMARY_RELPATH"
                    for t in node.targets
                )
            )
        ]
        module = ast.Module(
            body=[ast.parse("from __future__ import annotations").body[0], *body],
            type_ignores=[],
        )
        namespace: dict[str, Any] = {
            "hashlib": hashlib,
            "json": json,
            "Path": Path,
            "Any": Any,
        }
        exec(compile(module, str(bridge), "exec"), namespace)  # noqa: S102
        readers.append((str(checkout), namespace["_read_summary"]))
    return readers


def _emit_pipeline_summary(output: Path, run_id: str, step_dirs: list[Path]) -> Path:
    summary_path = output / "00_pipeline_summary" / "pipeline_execution_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_inventory_contract": "current-run-v1",
                "steps": [
                    {
                        "script_name": directory.name,
                        "run_id": run_id,
                        "status": "SUCCESS",
                        "artifacts": changed_artifacts(directory, output, {}),
                    }
                    for directory in step_dirs
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_cogant_summary_readers_accept_emitted_handoff_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_at = time.time() - 1
    context, execute_dir, summary = _execution(tmp_path, monkeypatch)
    output = Path(context.output_root)
    assert summary["execution_handoff_schema"] == EXECUTION_HANDOFF_SCHEMA
    summary_path = _emit_pipeline_summary(
        output, context.run_id, [output / "11_render_output", execute_dir]
    )

    readers: list[tuple[str, Callable[..., Any], str]] = [
        ("vendored", _vendored_invocation_reader, "invocation"),
    ] + [(name, reader, "live") for name, reader in _live_cogant_readers()]
    receipt = {}
    for name, reader, _ in readers:
        try:
            data = reader(output, context.run_id)
        except TypeError:
            # Older COGANT checkouts use the mtime reader signature.
            data = reader(summary_path, started_at)
        assert data is not None, name
        assert data["run_id"] == context.run_id
        receipt[name] = "accepted"
    assert _vendored_mtime_reader(summary_path, started_at)["run_id"] == (
        context.run_id
    )

    # A summary from another invocation, or a handoff whose bytes changed
    # after the inventory, is refused by the invocation reader.
    with pytest.raises(ValueError, match="run ID differs"):
        _vendored_invocation_reader(output, "another-run")
    (execute_dir / "summaries" / "execution_summary.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum differs"):
        _vendored_invocation_reader(output, context.run_id)
    present = {
        str(checkout)
        for checkout in _COGANT_CHECKOUTS
        if (checkout / _COGANT_BRIDGE).is_file()
    }
    assert set(receipt) == {"vendored"} | present
