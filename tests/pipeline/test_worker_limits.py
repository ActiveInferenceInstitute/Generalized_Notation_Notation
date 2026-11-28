"""Worker admission and propagation preserve explicit caller limits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gnn.pipeline.admission import validate_worker_count
from gnn.utils.arguments.arg_parsing import ArgumentParser, build_step_command_args
from gnn.utils.arguments.pipeline_arguments import PipelineArguments
from gnn.utils.arguments.pipeline_config_merge import apply_input_config_defaults


@pytest.mark.parametrize("value", [False, True, 0, -1, 257, 2.5, "4", [], {}])
@pytest.mark.parametrize("name", ["file_workers", "step_workers"])
def test_invalid_workers_refuse_before_output_lease(tmp_path, value, name):
    from gnn.main import main

    output = tmp_path / "output"
    args = PipelineArguments(target_dir=tmp_path, output_dir=output, **{name: value})
    assert main(override_args=args, override_config={}) == 1
    assert not output.exists()


@pytest.mark.parametrize("value", [1, 4, 256])
def test_worker_admission_preserves_requested_integer(value):
    assert validate_worker_count(value, field_name="file_workers") == value
    assert (
        validate_worker_count(None, field_name="step_workers", allow_none=True) is None
    )
    with pytest.raises(ValueError):
        validate_worker_count(None, field_name="file_workers")


def test_yaml_limits_apply_only_when_cli_omits_them():
    parser = ArgumentParser.create_main_parser()
    omitted = parser.parse_args([])
    args = PipelineArguments()
    full = {"pipeline": {"file_workers": 4, "step_workers": 2}}
    apply_input_config_defaults(args, full, omitted)
    assert (args.file_workers, args.step_workers) == (4, 2)
    explicit = parser.parse_args(["--file-workers", "1", "--step-workers", "1"])
    args = PipelineArguments(file_workers=1, step_workers=1)
    apply_input_config_defaults(args, full, explicit)
    assert (args.file_workers, args.step_workers) == (1, 1)


@pytest.mark.parametrize("value", [True, "4", 0, 257])
def test_invalid_yaml_worker_limits_refuse(value):
    args = PipelineArguments()
    with pytest.raises(ValueError):
        apply_input_config_defaults(
            args, {"pipeline": {"file_workers": value}}, argparse.Namespace()
        )


@pytest.mark.parametrize(
    "name,value", [("file_workers", True), ("step_workers", "2"), ("file_workers", 0)]
)
def test_cli_yaml_worker_admission_precedes_output_creation(
    tmp_path, monkeypatch, name, value
):
    import sys

    import gnn.main as orchestrator

    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gnn",
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--only-steps",
            "3",
        ],
    )
    monkeypatch.setattr(
        orchestrator, "_read_input_config", lambda _path: {"pipeline": {name: value}}
    )
    assert orchestrator.main() == 1
    assert not output.exists()


@pytest.mark.parametrize("step", ["3_gnn.py", "5_type_checker.py"])
def test_file_limit_reaches_step_parser_and_command(step):
    args = PipelineArguments(file_workers=4, step_workers=2)
    command = build_step_command_args(step, args, "python", Path("src/gnn") / step)
    assert command[command.index("--file-workers") + 1] == "4"
    assert "--step-workers" not in command
    parsed = ArgumentParser.create_step_parser(step).parse_args(["--file-workers", "4"])
    assert parsed.file_workers == 4


def test_run_cli_forwards_resource_limits(monkeypatch):
    import gnn.main
    from gnn.cli.handlers_pipeline import _cmd_run
    from gnn.cli.parser import build_parser

    observed = []
    monkeypatch.setattr(gnn.main, "main", lambda *, argv: observed.extend(argv) or 0)
    args = build_parser().parse_args(
        ["run", "--file-workers", "4", "--parallel", "--step-workers", "2"]
    )
    assert _cmd_run(args) == 0
    assert observed[observed.index("--file-workers") + 1] == "4"
    assert observed[observed.index("--step-workers") + 1] == "2"


def test_programmatic_pipeline_preserves_resource_precedence(tmp_path, monkeypatch):
    import gnn.main
    from gnn.pipeline.execution import run_pipeline

    observed = []
    monkeypatch.setattr(
        gnn.main, "main", lambda **kw: observed.append(kw["override_args"]) or 1
    )
    run_pipeline(
        {
            "target_dir": tmp_path,
            "output_dir": tmp_path / "output",
            "file_workers": 8,
            "step_workers": 4,
        },
        steps=[0],
        file_workers=2,
        step_workers=1,
    )
    assert len(observed) == 1
    assert (observed[0].file_workers, observed[0].step_workers) == (2, 1)


def test_file_process_pool_keeps_supervised_step_boundary(monkeypatch):
    from gnn.pipeline.summary_wiring import _consolidated_step_selected

    monkeypatch.delenv("GNN_RUN_CONTEXT_FILE", raising=False)
    args = PipelineArguments(consolidated_steps=True, file_workers=2)
    assert not _consolidated_step_selected("3_gnn.py", args, {})
    assert not _consolidated_step_selected("5_type_checker.py", args, {})


def test_typed_configuration_retains_worker_limits(tmp_path):
    from gnn.utils.config_io.config_loader import GNNPipelineConfig, PipelineConfig

    source = tmp_path / "config.yaml"
    source.write_text("pipeline:\n  file_workers: 4\n  step_workers: 2\n")
    config = GNNPipelineConfig.load_from_file(source)
    assert (config.pipeline.file_workers, config.pipeline.step_workers) == (4, 2)
    with pytest.raises(ValueError):
        PipelineConfig(file_workers=True)
    source.write_text("pipeline:\n  file_workers: true\n")
    with pytest.raises(ValueError):
        GNNPipelineConfig.load_from_file(source)
