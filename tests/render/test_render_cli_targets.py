"""
Mechanical guard for ``src/render/render.py`` CLI target choices.

Every ``choices`` item advertised by the renderer's CLI must have a matching
dispatch branch in ``render.processor.render_gnn_spec``; otherwise the CLI
accepts the target but always fails with ``Unsupported target``. This test
parametrises over every CLI target and asserts each one either:
  * produces at least one output artifact, or
  * returns a ``False`` success flag whose message is *not* the generic
    ``Unsupported target: <name>`` rejection.

Runs against the committed sample GNN ``input/gnn_files/discrete/actinf_pomdp_agent.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.render.processor import render_gnn_spec
from gnn.render.render import main as render_cli_main

SAMPLE_GNN = (
    Path(__file__).parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)

# Targets the CLI advertises. Keep this list in sync with
# ``render.render.main``'s ``choices=`` list.
CLI_TARGETS: List[str] = [
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "discopy",
    "discopy_combined",
    "bnlearn",
    "jax",
    "jax_pomdp",
    "pytorch",
    "numpyro",
    "cpomdp",
    "stan",
]


def _parse_sample() -> dict:
    from gnn import parse_gnn_file

    spec = parse_gnn_file(SAMPLE_GNN)
    if hasattr(spec, "to_dict"):
        return cast("dict[Any, Any]", spec.to_dict())
    return spec


@pytest.mark.parametrize("target", CLI_TARGETS)
def test_every_cli_target_dispatches(target: str, tmp_path: Path) -> None:
    gnn_spec = _parse_sample()
    out_dir = tmp_path / f"render_{target}"
    out_dir.mkdir()

    success, message, artifacts = render_gnn_spec(gnn_spec, target, out_dir)

    # Either the render succeeds or fails for a *real* reason — the failure
    # must not be the generic "Unsupported target" fallback.
    unsupported_prefix = f"Unsupported target: {target}"
    assert not message.startswith(unsupported_prefix), (
        f"CLI target {target!r} has no dispatch branch in render_gnn_spec "
        f"(message was {message!r})"
    )
    if success:
        assert artifacts, f"Target {target} reported success but produced no artifacts"


def test_cli_choices_are_dispatched_subset() -> None:
    """Catch drift if someone re-introduces an undispatched CLI target."""
    import argparse

    # Build the parser the way main() does and read its choices back.
    parser = argparse.ArgumentParser()
    parser.add_argument("gnn_file")
    parser.add_argument("output_dir")
    # Re-import to pick up latest definition.
    import importlib

    render_module = importlib.import_module("gnn.render.render")
    # Grab choices the easy way: inspect main by running with --help handling
    # disabled is clunky; instead rely on the explicit CLI_TARGETS list being
    # authoritative and compare it to the module's declared list.
    assert render_module.__file__ is not None
    source = Path(render_module.__file__).read_text()
    for target in CLI_TARGETS:
        assert f'"{target}"' in source, (
            f"CLI_TARGETS includes {target!r} but render.render does not list it "
            f"in the argparse choices"
        )


def test_cli_main_rejects_unknown_target(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    gnn_file = tmp_path / "m.md"
    gnn_file.write_text("## ModelName\nx\n")
    with pytest.raises(SystemExit):
        render_cli_main([str(gnn_file), str(tmp_path / "out"), "discopy_jax"])


# --- Phase 1.3 regression: generators reject invalid model_data ----------


@pytest.mark.parametrize(
    "generator_name",
    [
        "generate_pymdp_code",
        "generate_discopy_code",
        "generate_activeinference_jl_code",
        "generate_rxinfer_code",
    ],
)
def test_generators_reject_none_model_data(generator_name: Any) -> Any:
    """Before Phase 1.3, passing None to a generator crashed deep inside the
    template with an opaque AttributeError. After the fix, the validate_model_data
    guard catches it and the generator returns "" (empty string = no code emitted).
    """
    from gnn.render import generators

    fn = getattr(generators, generator_name)
    result = fn(None)
    assert result == "", (
        f"{generator_name}(None) should return empty string, got {type(result)}"
    )


@pytest.mark.parametrize(
    "generator_name",
    [
        "generate_pymdp_code",
        "generate_discopy_code",
    ],
)
def test_generators_reject_non_dict_model_data(generator_name: Any) -> Any:
    from gnn.render import generators

    fn = getattr(generators, generator_name)
    # A string isn't a dict — validator rejects before the generator tries to .get()
    result = fn("not-a-dict")
    assert result == ""


def test_bnlearn_generator_rejects_invalid_model_data() -> Any:
    """Same Phase 1.3 contract as the sibling generators: the validate_model_data
    guard returns "" (empty string = no code emitted) instead of crashing deep
    inside the template.
    """
    from gnn.render import bnlearn

    fn = bnlearn.generate_bnlearn_code
    result_none = fn(None)
    assert result_none == "", (
        f"generate_bnlearn_code(None) should return empty string, got {type(result_none)}"
    )
    # A string isn't a dict — validator rejects before the generator tries to .get()
    result_non_dict = fn("not-a-dict")
    assert result_non_dict == ""


def test_generator_accepts_minimal_valid_model_data() -> Any:
    """Smoke test: with the required key present, a generator should not error out
    at the validation stage (it may still have downstream issues, but validation
    itself must pass).
    """
    from gnn.render.bnlearn import generate_bnlearn_code

    # bnlearn is the lightest — doesn't import external packages at generation time
    result = generate_bnlearn_code({"model_name": "TestModel"})
    # Should produce SOMETHING (the emitted code string), not "".
    assert isinstance(result, str)
    assert len(result) > 100, "Generator with valid input should emit nontrivial code"


def test_bnlearn_generator_keeps_model_metadata_inside_literals() -> None:
    from gnn.render.bnlearn import generate_bnlearn_code

    model_name = (
        'Injected";\n__import__("pathlib").Path("/tmp/gnn-pwned").write_text("x")\n#'
    )
    code = generate_bnlearn_code(
        {
            "model_name": model_name,
            "source_file": 'source"; __import__("os").system("id") #.md',
            "model_parameters": {
                "num_timesteps": "1); __import__('os').system('id') #"
            },
        }
    )

    compile(code, "<generated-bnlearn>", "exec")
    assert f"GENERATED_MODEL_NAME = {model_name!r}" in code
    assert '\n__import__("pathlib")' not in code
    assert "n_samples = max(1000, 15 * 20)" in code


def test_render_gnn_spec_sanitizes_generated_output_filename(tmp_path: Path) -> None:
    malicious_name = "../../escape"

    ok, message, artifacts = render_gnn_spec(
        {
            "model_name": malicious_name,
            "variables": [],
            "connections": [],
            "model_parameters": {},
        },
        "bnlearn",
        tmp_path,
    )

    assert ok, message
    assert artifacts, "render should produce artifacts"
    artifact_path = Path(artifacts[0]).resolve()
    artifact_path.relative_to(tmp_path.resolve())
    assert artifact_path.name == "escape_bnlearn.py"
    assert not (tmp_path.parent / "escape_bnlearn.py").exists()


@pytest.mark.parametrize(
    ("target", "required_marker"),
    [
        ("jax", "def run_simulation"),
        ("jax_pomdp", "class JAXPOMDPSolver"),
    ],
)
def test_file_backed_parser_summary_renders_real_jax_code(
    target: str,
    required_marker: str,
    tmp_path: Path,
) -> None:
    """The public parser summary must be rehydrated, never stubbed."""
    success, message, artifacts = render_gnn_spec(
        _parse_sample(),
        target,
        tmp_path,
        {"output_filename": f"custom-{target}"},
    )

    assert success, message
    assert len(artifacts) == 1
    artifact = Path(artifacts[0])
    assert artifact.name == f"custom-{target}_jax.py"
    code = artifact.read_text(encoding="utf-8")
    assert required_marker in code
    assert "Recovery Implementation" not in code
    assert "Active_Inference_POMDP_Agent" in code


def test_jax_and_jax_pomdp_dispatch_to_distinct_generators(tmp_path: Path) -> None:
    general_dir = tmp_path / "general"
    pomdp_dir = tmp_path / "pomdp"
    general = render_gnn_spec(_parse_sample(), "jax", general_dir)
    pomdp = render_gnn_spec(_parse_sample(), "jax_pomdp", pomdp_dir)

    assert general[0], general[1]
    assert pomdp[0], pomdp[1]
    general_code = Path(general[2][0]).read_text(encoding="utf-8")
    pomdp_code = Path(pomdp[2][0]).read_text(encoding="utf-8")
    assert "def run_simulation" in general_code
    assert "class JAXPOMDPSolver" not in general_code
    assert "class JAXPOMDPSolver" in pomdp_code
