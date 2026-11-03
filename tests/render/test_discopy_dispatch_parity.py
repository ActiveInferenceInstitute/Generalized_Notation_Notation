"""One renderer and one artifact for every discopy entry point.

Before this contract, Step 11 called ``render_gnn_to_discopy`` while the
CLI/MCP single-spec door called ``generators.generate_discopy_code``. Both
reported success and wrote the same filename, but the artifacts differed in
category (monoidal versus rigid) and structure (declared connections versus a
fixed eight-box circuit). These tests pin the shared renderer, the literal
source provenance in the header, and the typed refusal when wiring is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SAMPLE_GNN = (
    Path(__file__).parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


def _load_generated(path: Path | str) -> ModuleType:
    path = Path(path)
    module_name = f"discopy_parity_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", ["discopy", "discopy_combined"])
def test_single_spec_targets_emit_connection_driven_artifact(
    target: str, tmp_path: Path
) -> None:
    from gnn import parse_gnn_file
    from gnn.render.processor import render_gnn_spec

    ok, message, artifacts = render_gnn_spec(
        parse_gnn_file(SAMPLE_GNN), target, tmp_path
    )

    assert ok, message
    assert len(artifacts) == 1
    code = Path(artifacts[0]).read_text(encoding="utf-8")
    assert "Generated from GNN specification: actinf_pomdp_agent.md" in code
    assert "unknown.md" not in code
    assert "from discopy.monoidal import" in code
    assert "discopy.rigid" not in code
    assert "def build_diagram" in code
    assert "EnhancedGnnModelCategoricalAnalyzer" not in code

    module = _load_generated(artifacts[0])
    _diagram, boxes, signatures = module.build_diagram()
    assert len(boxes) == 11
    assert len(signatures) == 11
    assert signatures[0]["source"] == ["D"]
    assert signatures[0]["target"] == ["s"]


def test_step11_and_cli_doors_emit_identical_bytes(tmp_path: Path) -> None:
    from gnn import parse_gnn_file
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import process_pomdp_for_frameworks
    from gnn.render.processor import render_gnn_spec

    pomdp_space = extract_pomdp_from_file(SAMPLE_GNN, strict_validation=True)
    assert pomdp_space is not None
    step11 = process_pomdp_for_frameworks(
        pomdp_space,
        tmp_path / "step11",
        ["discopy"],
        gnn_file_path=SAMPLE_GNN,
    )
    step11_result = step11["framework_results"]["discopy"]
    assert step11_result["success"], step11_result
    step11_file = Path(step11_result["output_files"][0])

    ok, message, cli_files = render_gnn_spec(
        parse_gnn_file(SAMPLE_GNN),
        "discopy",
        tmp_path / "cli",
        {"output_filename": step11_file.stem.removesuffix("_discopy")},
    )
    assert ok, message
    assert len(cli_files) == 1
    assert step11_file.read_bytes() == Path(cli_files[0]).read_bytes()


def test_matrix_permutation_options_reach_shared_renderer(tmp_path: Path) -> None:
    from gnn.render.processor import render_gnn_spec

    permutation = {"A": [1, 0]}
    spec: dict[str, Any] = {
        "model_name": "permutation_probe",
        "source_file": "permutation_probe.md",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "matrix_permutations": permutation,
        },
        "connections": [{"source": "D", "relation": ">", "target": "s"}],
    }

    ok, message, artifacts = render_gnn_spec(
        spec,
        "discopy",
        tmp_path,
        {"matrix_permutations": permutation},
    )

    assert ok, message
    module = _load_generated(artifacts[0])
    assert module.MATRIX_PERMUTATION_METADATA["A"]["permutation"] == [1, 0]
    assert module.MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM is False


def test_missing_connections_refuses_through_single_spec_door(
    tmp_path: Path,
) -> None:
    from gnn.render.processor import render_gnn_spec

    spec = {
        "model_name": "no_wiring",
        "initialparameterization": {"A": [[1.0]]},
    }

    ok, message, artifacts = render_gnn_spec(spec, "discopy", tmp_path)

    assert not ok
    assert message.startswith("unsupported-discopy-connections:")
    assert artifacts == []
    assert not list(tmp_path.glob("*_discopy.py"))


def test_legacy_generator_is_not_in_target_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn import parse_gnn_file
    from gnn.render import generators
    from gnn.render.processor import render_gnn_spec

    calls: list[tuple[Any, ...]] = []

    def _legacy(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(generators, "generate_discopy_code", _legacy)
    ok, message, artifacts = render_gnn_spec(
        parse_gnn_file(SAMPLE_GNN), "discopy", tmp_path
    )

    assert ok, message
    assert artifacts
    assert calls == []
