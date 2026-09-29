"""Continuous-state exemplars: extraction, model kind, and framework support matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import process_render

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
FILES = sorted(CONTINUOUS_DIR.glob("*.md"))

UNSUPPORTED = {"pymdp", "activeinference_jl", "bnlearn", "discopy"}
SUPPORTED = {"jax", "numpyro", "pytorch", "stan", "rxinfer", "cpomdp"}


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_continuous_exemplar_extracts_lgssm(path: Path) -> None:
    pomdp = extract_pomdp_from_file(path, strict_validation=True)
    assert pomdp is not None
    assert pomdp.model_kind == "continuous"
    assert pomdp.A_matrix is None and pomdp.B_matrix is None
    assert pomdp.matrices is not None
    if path.stem == "factored_continuous_lgssm":
        # Per-factor LGSSM block: every per-factor key is collected, the
        # optional goal/control pair on factor 1 included; joint dimensions
        # come from factor 1.
        for suffix in ("1", "2"):
            for prefix in ("F", "H", "Q", "R", "prior_mean", "prior_cov"):
                assert f"{prefix}_f{suffix}" in pomdp.matrices
        assert "goal_mean_f1" in pomdp.matrices
        assert "control_gain_f1" in pomdp.matrices
        assert pomdp.num_states == len(pomdp.matrices["F_f1"])
        assert pomdp.num_observations == len(pomdp.matrices["H_f1"])
    else:
        for key in ("F", "H", "Q", "R", "prior_mean", "prior_cov"):
            assert key in pomdp.matrices, key
        assert pomdp.num_states == len(pomdp.matrices["F"])
        assert pomdp.num_observations == len(pomdp.matrices["H"])
    spec = pomdp_to_gnn_spec(pomdp)
    assert spec["model_kind"] == "continuous"
    if path.stem == "hybrid_discrete_continuous":
        # The discrete family survives extraction verbatim; the refusal
        # happens at dispatch, never in extraction.
        assert "A" in spec["initialparameterization"]
    else:
        assert "A" not in spec["initialparameterization"]


def test_navigation_is_closed_loop_others_passive() -> None:
    kinds: dict[str, bool] = {}
    for p in FILES:
        pomdp = extract_pomdp_from_file(p, strict_validation=True)
        assert pomdp is not None
        kinds[p.stem] = pomdp.passive_model
    assert kinds["continuous_navigation"] is False
    assert kinds["predictive_coding_agent"] is True
    assert kinds["stochastic_dynamics"] is True


def test_unsupported_frameworks_are_flagged_not_failed(tmp_path: Path) -> None:
    # Pin the pure passive exemplar by name: alphabetical ordering must not
    # decide which spec the discrete-only receipt is asserted against.
    pomdp = extract_pomdp_from_file(
        CONTINUOUS_DIR / "stochastic_dynamics.md", strict_validation=True
    )
    assert pomdp is not None
    proc = POMDPRenderProcessor(tmp_path)
    for fw in sorted(UNSUPPORTED):
        result = proc._process_single_framework(pomdp, fw)
        assert result["unsupported"] is True and result["status"] == "unsupported"
        assert "supports discrete POMDPs only" in result["message"]
        assert result["output_files"] == []
    factored = extract_pomdp_from_file(
        CONTINUOUS_DIR / "factored_continuous_lgssm.md", strict_validation=True
    )
    assert factored is not None
    for fw in sorted(UNSUPPORTED):
        result = proc._process_single_framework(factored, fw)
        assert result["unsupported"] is True and result["status"] == "unsupported"
        assert "unsupported-factored-continuous" in result["message"]
        assert result["output_files"] == []


def test_process_render_counts_unsupported_separately(tmp_path: Path) -> None:
    import json

    outcome = process_render(
        target_dir=CONTINUOUS_DIR,
        output_dir=tmp_path,
        frameworks=[
            "jax",
            "numpyro",
            "pytorch",
            "stan",
            "rxinfer",
            "cpomdp",
            "discopy",
            "pymdp",
            "activeinference_jl",
        ],
        verbose=False,
    )
    assert outcome is True
    summary = json.loads((tmp_path / "render_processing_summary.json").read_text())
    assert summary["total_files"] == len(FILES)
    assert summary["successful_files"] == len(FILES)
    assert summary["failed_framework_renderings"] == []
    # Composed exemplars (e.g. multi_agent_lgssm: {continuous, multi_agent})
    # intentionally receive an unsupported-composition receipt from EVERY
    # framework — no renderer handles the whole composition yet. The census
    # below counts only PURE continuous files (singleton kind); composed
    # files must still be explicit receipts, never failed renders.
    from gnn.render.pomdp_contract import detect_pomdp_space_model_kinds

    composed_stems = {
        p.stem
        for p in FILES
        if len(
            detect_pomdp_space_model_kinds(
                extract_pomdp_from_file(p, strict_validation=False)
            )
        )
        > 1
    }
    unsupported = {
        u["framework"]
        for u in summary["unsupported_framework_renderings"]
        if Path(u["file"]).stem not in composed_stems
    }
    assert unsupported == {"pymdp", "activeinference_jl", "discopy"}
    for stem, res in summary["file_results"].items():
        statuses = {
            fw: r.get("status", "ok") for fw, r in res["framework_results"].items()
        }
        if Path(stem).stem in composed_stems:
            assert all(
                r.get("unsupported") or r.get("success")
                for r in res["framework_results"].values()
            )
            continue
        assert all(
            r["success"]
            for fw, r in res["framework_results"].items()
            if fw in SUPPORTED
        )
        assert statuses["pymdp"] == "unsupported"
