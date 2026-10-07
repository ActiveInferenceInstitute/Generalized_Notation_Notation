"""Interface guardrails for step arguments and dependency closure."""

import logging
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from gnn.utils.arguments.arg_parsing import (
    ArgumentParser,
    audit_step_contracts,
    build_step_command_args,
)
from gnn.utils.arguments.pipeline_arguments import (
    DEFAULT_ONTOLOGY_TERMS_FILE,
    PipelineArguments,
)
from gnn.utils.arguments.step_config import StepConfiguration
from gnn.utils.errors.error_handling import CRITICAL_STEP_NUMBERS
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    resolve_step_dependencies,
)
from gnn.utils.pipeline_orchestration.pipeline_template import (
    _create_fallback_parser,
    _parse_step_args,
)


def test_step_specific_arguments_are_exposed() -> None:
    step0 = ArgumentParser.parse_step_arguments("0_template.py", ["--simulate-error"])
    assert step0.simulate_error is True

    step2 = ArgumentParser.parse_step_arguments("2_tests.py", ["--fast-only"])
    assert step2.fast_only is True

    step4 = ArgumentParser.parse_step_arguments(
        "4_model_registry.py", ["--registry-path", "custom_registry.json"]
    )
    assert step4.registry_path == Path("custom_registry.json")

    step6 = ArgumentParser.parse_step_arguments(
        "6_validation.py", ["--profile", "--strict"]
    )
    assert step6.profile is True
    assert step6.strict is True

    step9 = ArgumentParser.parse_step_arguments(
        "9_advanced_viz.py",
        [
            "--viz-type",
            "network",
            "--no-interactive",
            "--export-formats",
            "html",
            "json",
        ],
    )
    assert step9.viz_type == "network"
    assert step9.interactive is False
    assert step9.export_formats == ["html", "json"]

    step22 = ArgumentParser.parse_step_arguments(
        "22_gui.py",
        [
            "--interactive",
            "--gui-types",
            "gui_1,oxdraw",
            "--open-browser",
            "--launch-editor",
        ],
    )
    assert step22.interactive is True
    assert step22.gui_types == "gui_1,oxdraw"
    assert step22.open_browser is True
    assert step22.launch_editor is True

    step22_default = ArgumentParser.parse_step_arguments("22_gui.py", [])
    assert step22_default.launch_editor is False

    step24 = ArgumentParser.parse_step_arguments(
        "24_intelligent_analysis.py",
        [
            "--analysis-model",
            "local-test",
            "--bottleneck-threshold",
            "12.5",
            "--skip-llm",
        ],
    )
    assert step24.analysis_model == "local-test"
    assert step24.bottleneck_threshold == 12.5
    assert step24.skip_llm is True


def test_step16_animation_argument_contract() -> None:
    step16_default = ArgumentParser.parse_step_arguments("16_analysis.py", [])
    assert step16_default.generate_animations is True

    step16_disabled = ArgumentParser.parse_step_arguments(
        "16_analysis.py", ["--no-animations"]
    )
    assert step16_disabled.generate_animations is False


def test_step16_no_animations_propagates_from_main_command() -> None:
    script_path = Path("src/gnn/16_analysis.py")
    enabled_args = PipelineArguments(generate_animations=True)
    disabled_args = PipelineArguments(generate_animations=False)

    enabled_cmd = build_step_command_args(
        "16_analysis", enabled_args, sys.executable, script_path
    )
    disabled_cmd = build_step_command_args(
        "16_analysis", disabled_args, sys.executable, script_path
    )

    assert "--no-animations" not in enabled_cmd
    assert "--no-animations" in disabled_cmd
    assert "False" not in disabled_cmd


def test_recursive_false_propagates_to_child_steps() -> None:
    script_path = Path("src/gnn/8_visualization.py")
    enabled_cmd = build_step_command_args(
        "8_visualization",
        PipelineArguments(recursive=True),
        sys.executable,
        script_path,
    )
    disabled_cmd = build_step_command_args(
        "8_visualization",
        PipelineArguments(recursive=False),
        sys.executable,
        script_path,
    )

    assert "--recursive" in enabled_cmd
    assert "--no-recursive" in disabled_cmd


def test_store_true_flags_only_propagate_when_enabled() -> None:
    script_path = Path("src/gnn/2_tests.py")
    disabled_cmd = build_step_command_args(
        "2_tests",
        PipelineArguments(comprehensive=False),
        sys.executable,
        script_path,
    )
    enabled_cmd = build_step_command_args(
        "2_tests",
        PipelineArguments(comprehensive=True),
        sys.executable,
        script_path,
    )

    assert "--comprehensive" not in disabled_cmd
    assert "--comprehensive" in enabled_cmd


def test_setup_optional_group_arguments_match_docs() -> None:
    args = ArgumentParser.parse_step_arguments(
        "1_setup.py",
        ["--install-optional", "--optional-groups", "gui,audio"],
    )
    assert args.install_optional is True
    assert args.optional_groups == "gui,audio"
    assert args.target_dir == Path("input/gnn_files")


def test_step_specific_defaults_are_preserved() -> None:
    step9 = ArgumentParser.parse_step_arguments("9_advanced_viz.py", [])
    step22 = ArgumentParser.parse_step_arguments("22_gui.py", [])
    assert step9.interactive is True
    assert step22.interactive is False


def test_registered_step_contract_audit_is_clean() -> None:
    assert audit_step_contracts(sys.executable, Path("src/gnn")) == []


def test_exit_code_contract_docs_use_canonical_wording() -> None:
    canonical = "0=success, 1=error, 2=success with warnings/skipped"
    checked_paths = [
        Path("ARCHITECTURE.md"),
        Path("docs/PIPELINE_SCRIPTS.md"),
        Path("src/gnn/STEP_INDEX.md"),
        Path("src/gnn/template/README.md"),
        Path("tests/README.md"),
    ]
    stale_phrases = [
        "1 = warning/non-critical failure",
        "1=warning",
        "2=warnings)",
        "2 warnings)",
        "2=skipped/warnings",
    ]

    for path in checked_paths:
        text = path.read_text()
        assert canonical in text, f"{path} missing canonical exit-code contract"
        for phrase in stale_phrases:
            assert phrase not in text, f"{path} contains stale phrase: {phrase}"


def test_critical_metadata_matches_canonical_set() -> None:
    actual = {
        int(step_name.split("_", 1)[0])
        for step_name, config in StepConfiguration.STEP_CONFIGS.items()
        if config.get("critical")
    }
    assert actual == set(CRITICAL_STEP_NUMBERS)


def test_registered_step_parsers_support_recursive_round_trip() -> None:
    for step_name, config in StepConfiguration.STEP_CONFIGS.items():
        all_args = cast("list[Any]", config.get("required_args", [])) + cast(
            "list[Any]", config.get("optional_args", [])
        )
        if "recursive" not in all_args:
            continue

        script_name = f"{step_name}.py"
        default_args = ArgumentParser.create_step_parser(script_name).parse_args([])
        enabled_args = ArgumentParser.create_step_parser(script_name).parse_args(
            ["--recursive"]
        )
        disabled_args = ArgumentParser.create_step_parser(script_name).parse_args(
            ["--no-recursive"]
        )

        assert default_args.recursive == cast(
            "dict[str, Any]", config.get("defaults", {})
        ).get("recursive", True)
        assert enabled_args.recursive is True
        assert disabled_args.recursive is False


def test_fallback_parser_matches_step_recursive_defaults() -> None:
    for step_name, config in StepConfiguration.STEP_CONFIGS.items():
        all_args = cast("list[Any]", config.get("required_args", [])) + cast(
            "list[Any]", config.get("optional_args", [])
        )
        if "recursive" not in all_args:
            continue

        script_name = f"{step_name}.py"
        fallback = _create_fallback_parser("fallback", {}, script_name)
        fallback_default = fallback.parse_args([])
        fallback_disabled = fallback.parse_args(["--no-recursive"])

        assert fallback_default.recursive == cast(
            "dict[str, Any]", config.get("defaults", {})
        ).get("recursive", True)
        assert fallback_disabled.recursive is False


def test_script_additional_args_do_not_trigger_recovery_parser(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cases = [
        (
            "0_template.py",
            ["--simulate-error"],
            {"simulate_error": {"type": bool, "help": "Simulate an error"}},
            "simulate_error",
            True,
        ),
        (
            "2_tests.py",
            ["--fast-only"],
            {
                "fast_only": {
                    "action": "store_true",
                    "default": True,
                    "flag": "--fast-only",
                }
            },
            "fast_only",
            True,
        ),
        (
            "4_model_registry.py",
            ["--registry-path", "registry.json"],
            {"registry_path": {"type": str, "help": "Registry path"}},
            "registry_path",
            Path("registry.json"),
        ),
        (
            "6_validation.py",
            ["--profile", "--strict"],
            {
                "profile": {"type": bool, "help": "Profile"},
                "strict": {"type": bool, "help": "Strict"},
            },
            "profile",
            True,
        ),
    ]

    caplog.set_level(logging.WARNING)
    for step_name, argv, additional, attr_name, expected in cases:
        caplog.clear()
        monkeypatch.setattr(sys, "argv", [step_name, *argv])
        parsed = _parse_step_args(step_name, "test parser", additional)
        assert getattr(parsed, attr_name) == expected
        assert not any("using recovery parser" in r.message for r in caplog.records)


def test_invalid_step_argument_fails_instead_of_silent_defaults() -> None:
    with pytest.raises(SystemExit):
        ArgumentParser.parse_step_arguments("22_gui.py", ["--not-a-real-flag"])


def test_only_steps_dependency_closure_is_recursive() -> None:
    assert resolve_step_dependencies([23]) == [3, 23]
    assert resolve_step_dependencies([17]) == [3, 17]
    assert resolve_step_dependencies([24]) == [24]


def test_default_ontology_terms_file_resolves_to_bundled_vocabulary() -> None:
    """The packaged default must hit src/gnn/ontology/, not the stale
    arguments-subpackage-relative src/gnn/utils/ontology/ path."""
    assert DEFAULT_ONTOLOGY_TERMS_FILE.parent.name == "ontology"
    assert DEFAULT_ONTOLOGY_TERMS_FILE.name == "act_inf_ontology_terms.json"
    assert DEFAULT_ONTOLOGY_TERMS_FILE.exists()
    assert PipelineArguments().ontology_terms_file == DEFAULT_ONTOLOGY_TERMS_FILE


def test_removed_website_html_filename_flag_is_rejected() -> None:
    """The dead ``--website-html-filename`` knob (accepted-and-ignored
    through four duplicated defaults) was removed end-to-end in v3.5.0:
    the step parser must reject it and the defaults namespace must not
    carry the field."""
    with pytest.raises(SystemExit):
        ArgumentParser.parse_step_arguments(
            "20_website.py", ["--website-html-filename", "out.html"]
        )
    defaults = ArgumentParser.parse_step_arguments("20_website.py", [])
    assert not hasattr(defaults, "website_html_filename")
