"""Tests for the GNN CLI module."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest

# Import the CLI main function
try:
    import gnn.cli as cli
    from gnn.cli import main
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import gnn.cli as cli
    from gnn.cli import main


class _CallTracker:
    def __init__(self) -> None:
        self.called = False
        self.call_args: tuple[Any, ...] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.called = True
        self.call_args = args


def test_cli_help() -> Any:
    """Test that 'gnn --help' works and returns success."""
    orig_stdout = sys.stdout
    captured_out = StringIO()
    sys.stdout = captured_out
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])

        assert exit_info.value.code == 0
        output = captured_out.getvalue()
        assert "GNN Processing Pipeline" in output
        assert "Available commands" in output
        assert "run" in output
        assert "validate" in output
    finally:
        sys.stdout = orig_stdout


def test_cli_invalid_command() -> Any:
    """Test that an invalid command returns an error or help."""
    orig_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with pytest.raises(SystemExit):
            main(["nonexistent-command"])
    finally:
        sys.stderr = orig_stderr


def test_cli_validate_parser(tmp_path: Any) -> Any:
    """Test the 'validate' subcommand parser."""
    test_file = tmp_path / "test.md"
    test_file.touch()

    orig_validate = getattr(cli, "_cmd_validate", None)
    tracker = _CallTracker()
    cli._cmd_validate = tracker

    try:
        main(["validate", str(test_file)])
        assert tracker.called is True
    finally:
        if orig_validate:
            cli._cmd_validate = orig_validate


def test_cli_verbose_flag(tmp_path: Any) -> Any:
    """Test that the --verbose flag is correctly handled."""
    orig_health = getattr(cli, "_cmd_health", None)
    tracker = _CallTracker()
    cli._cmd_health = tracker

    try:
        main(["--verbose", "health"])
        assert tracker.call_args is not None
        args = tracker.call_args[0]
        assert getattr(args, "verbose", False) is True
    finally:
        if orig_health:
            cli._cmd_health = orig_health


def test_cli_health_strict_flag_routes_to_handler() -> Any:
    """Test that health --strict reaches the handler."""
    orig_health = getattr(cli, "_cmd_health", None)
    tracker = _CallTracker()
    cli._cmd_health = tracker

    try:
        main(["health", "--strict"])
        assert tracker.call_args is not None
        args = tracker.call_args[0]
        assert getattr(args, "strict", False) is True
    finally:
        if orig_health:
            cli._cmd_health = orig_health


def test_health_default_informational_when_environment_has_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gnn.pipeline as pipeline
    import gnn.pipeline.preflight
    import gnn.render as render
    import gnn.render.health  # noqa: F401

    monkeypatch.setattr(
        render.health,
        "check_renderers",
        lambda: {"pymdp": SimpleNamespace(available=True)},
    )
    monkeypatch.setattr(
        pipeline.preflight,
        "check_environment",
        lambda: SimpleNamespace(
            checks_passed=1,
            checks_failed=1,
            is_ok=False,
            issues=[SimpleNamespace(severity="error", message="missing backend")],
        ),
    )

    assert cli._cmd_health(argparse.Namespace(strict=False)) == 2
    captured = capsys.readouterr()
    assert "generator modules importable" in captured.out
    assert "pass --strict to fail on errors" in captured.out


def test_health_strict_fails_when_environment_has_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.pipeline as pipeline
    import gnn.pipeline.preflight
    import gnn.render as render
    import gnn.render.health  # noqa: F401

    monkeypatch.setattr(
        render.health,
        "check_renderers",
        lambda: {"pymdp": SimpleNamespace(available=True)},
    )
    monkeypatch.setattr(
        pipeline.preflight,
        "check_environment",
        lambda: SimpleNamespace(
            checks_passed=1,
            checks_failed=1,
            is_ok=False,
            issues=[SimpleNamespace(severity="error", message="missing backend")],
        ),
    )

    assert cli._cmd_health(argparse.Namespace(strict=True)) == 1


def test_run_combines_and_serializes_skip_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI forwarding uses the pipeline's single comma-separated step value."""
    import gnn.main as pipeline_module

    captured_argv: list[str] = []
    original_argv = list(sys.argv)

    def fake_pipeline_main(*, argv: list[str]) -> int:
        captured_argv.extend(argv)
        return 0

    monkeypatch.setattr(pipeline_module, "main", fake_pipeline_main)
    result = cli._cmd_run(
        argparse.Namespace(
            target_dir="input/gnn_files",
            output_dir="output",
            verbose=False,
            log_format="human",
            skip_llm=True,
            skip_steps=[2, 1],
        )
    )

    assert result == 0
    assert captured_argv[captured_argv.index("--skip-steps") + 1] == "1,2,13"
    assert sys.argv == original_argv


def test_run_serializes_only_steps_for_pipeline_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.main as pipeline_module

    captured_argv: list[str] = []

    def fake_pipeline_main(*, argv: list[str]) -> int:
        captured_argv.extend(argv)
        return 0

    monkeypatch.setattr(pipeline_module, "main", fake_pipeline_main)
    assert main(["run", "--only-steps", "3", "11"]) == 0
    assert captured_argv[captured_argv.index("--only-steps") + 1] == "3,11"


def test_run_rejects_overlapping_only_and_skip_steps() -> None:
    assert main(["run", "--only-steps", "3", "--skip-steps", "3"]) == 1


def test_concurrent_runs_use_their_own_output_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gnn.main as pipeline_module
    from gnn.pipeline import output_lease
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments

    original_argv = sys.argv
    original_values = list(original_argv)
    pipeline_main = pipeline_module.main
    admitted = Barrier(2)
    leased: list[Path] = []
    observed: list[tuple[Path, Path]] = []

    def synchronize(*args: Any, **kwargs: Any) -> int:
        admitted.wait(timeout=5)
        return pipeline_main(*args, **kwargs)

    @contextmanager
    def lease(root: Path, _run_id: str):
        leased.append(root)
        try:
            yield
        finally:
            leased.pop()

    def record_run(args: PipelineArguments, *_: Any) -> int:
        observed.append((leased[0], args.output_dir))
        return 0

    monkeypatch.setattr(pipeline_module, "main", synchronize)
    monkeypatch.setattr(pipeline_module, "_run_pipeline", record_run)
    monkeypatch.setattr(output_lease, "OutputLease", lease)

    destinations = [tmp_path / "first", tmp_path / "second"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(main, ["run", "--only-steps", "3", "--output-dir", str(path)])
            for path in destinations
        ]
        assert [future.result(timeout=10) for future in futures] == [0, 0]
    assert sorted(observed) == [(path, path) for path in destinations]
    assert sys.argv is original_argv and sys.argv == original_values


def test_verbose_is_accepted_after_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    def fake_health(args: Any) -> int:
        seen.append(args.verbose)
        return 0

    monkeypatch.setattr(cli, "_cmd_health", fake_health)
    assert main(["health", "--verbose"]) == 0
    assert seen == [True]


def test_unexpected_handler_error_returns_error_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_health(args: Any) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_cmd_health", broken_health)
    assert main(["health"]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--skip-steps", "25"],
        ["run", "--only-steps", "25"],
        ["serve", "--port", "0"],
        ["serve", "--port", "65536"],
    ],
)
def test_cli_rejects_out_of_range_numeric_arguments(argv: list[str]) -> None:
    """Invalid steps and ports are parsing errors with argparse exit code 2."""
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


def test_cli_subcommand_routing(tmp_path: Any) -> Any:
    """Test that subcommands are routed to the correct handlers."""
    commands: list[Any] = [
        ("run", "_cmd_run"),
        ("validate", "_cmd_validate"),
        ("parse", "_cmd_parse"),
        ("render", "_cmd_render"),
        ("report", "_cmd_report"),
        ("health", "_cmd_health"),
    ]

    test_file = tmp_path / "test.md"
    test_file.touch()
    str_test_file = str(test_file)

    for cmd_name, handler_name in commands:
        orig_handler = getattr(cli, handler_name, None)
        tracker = _CallTracker()
        setattr(cli, handler_name, tracker)

        try:
            if cmd_name in ["validate", "parse", "render", "graph"]:
                main([cmd_name, str_test_file])
            elif cmd_name == "reproduce":
                main([cmd_name, "abc123def456"])
            elif cmd_name == "watch":
                main([cmd_name, "."])
            else:
                main([cmd_name])

            assert tracker.called is True
        finally:
            if orig_handler:
                setattr(cli, handler_name, orig_handler)
