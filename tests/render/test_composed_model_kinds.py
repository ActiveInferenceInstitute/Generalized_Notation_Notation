"""Composed model-kind detection and dispatch honesty.

``detect_model_kinds`` returns the full composed kind set while
``detect_model_kind`` stays the max-precedence single winner (plain specs
classify identically to the former single-winner chain). A spec that declares
the linear-Gaussian (F/H/Q/R) family alongside multi-agent structure must
classify as ``{CONTINUOUS, MULTI_AGENT}`` and be refused at every dispatch
site with the explicit ``unsupported-composition`` receipt — never silently
rendered as one family with the other dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_contract import (
    ModelKind,
    detect_model_kind,
    detect_model_kinds,
    detect_pomdp_space_model_kind,
    detect_pomdp_space_model_kinds,
)
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import process_render, render_gnn_spec
from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"
COMPOSED_REL = "continuous/multi_agent_lgssm.md"

#: The full linear-Gaussian contract every continuous exemplar declares.
_LGSSM_BLOCK = {
    "F": [[1.0, 0.0], [0.0, 1.0]],
    "H": [[1.0, 0.0], [0.0, 1.0]],
    "Q": [[0.05, 0.0], [0.0, 0.05]],
    "R": [[0.1, 0.0], [0.0, 0.1]],
    "prior_mean": [0.0, 0.0],
    "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
}

#: One representative spec per plain (non-composed) kind; the kind-set API
#: must return a singleton set identical to the single-winner classification.
_PLAIN_KIND_SPECS = {
    "flat": {
        "initialparameterization": {
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "B": [[1.0]],
            "C": [1.0, 1.0],
            "D": [1.0, 1.0],
        }
    },
    "continuous": {"initialparameterization": dict(_LGSSM_BLOCK)},
    "multi_agent": {"initialparameterization": {"nr_agents": 2, "agent1_id": "a1"}},
    "hierarchical": {
        "gnn_section": "hierarchical",
        "initialparameterization": {"A_level1": [[1.0, 0.0], [0.0, 1.0]]},
    },
    "learning": {
        "initialparameterization": {
            "dirichlet_A": [1.0, 1.0],
            "A": [[1.0]],
            "B": [[1.0]],
            "C": [1.0],
            "D": [1.0],
        }
    },
    "factored": {
        "model_parameters": {"num_factors": 2},
        "initialparameterization": {"A": [[1.0]]},
    },
    "structural": {"model_kind": "structural"},
}


def _composed_spec() -> dict:
    """A continuous LGSSM block declared with a two-agent composition."""
    return {
        "initialparameterization": dict(_LGSSM_BLOCK),
        "model_parameters": {"nr_agents": 2},
    }


def _hybrid_spec(prior_only: bool = False) -> dict:
    """Discrete A-E contract keys alongside the linear-Gaussian family."""
    initial: dict = {
        "A": [[0.8, 0.2], [0.1, 0.9]],
        "B": [[[1.0, 0.0], [0.0, 1.0]]],
        "C": [1.0, 0.0],
        "D": [0.5, 0.5],
    }
    if prior_only:
        initial["prior_mean"] = [0.0, 0.0]
        initial["prior_cov"] = [[0.5, 0.0], [0.0, 0.5]]
    else:
        initial.update(_LGSSM_BLOCK)
    return {"initialparameterization": initial}


def _hybrid_multi_agent_spec() -> dict:
    """A linear-Gaussian block declared under per-agent discrete keys."""
    initial = dict(_LGSSM_BLOCK)
    initial["A_agent1"] = [[0.8], [0.2]]
    initial["A_agent2"] = [[0.7], [0.3]]
    return {"initialparameterization": initial}


def _factored_continuous_spec() -> dict:
    """``num_factors > 1`` declared alongside the flat Gaussian family."""
    return {
        "initialparameterization": dict(_LGSSM_BLOCK),
        "model_parameters": {"num_factors": 2},
    }


class TestKindSets:
    """detect_model_kinds returns the full set; plain specs stay singleton."""

    @pytest.mark.parametrize(
        "kind_name", sorted(_PLAIN_KIND_SPECS), ids=sorted(_PLAIN_KIND_SPECS)
    )
    def test_plain_specs_have_singleton_kind_sets(self, kind_name: str) -> None:
        spec = _PLAIN_KIND_SPECS[kind_name]
        kinds = detect_model_kinds(spec)
        winner = detect_model_kind(spec)
        assert kinds == frozenset({winner})
        assert winner.value == kind_name

    def test_plain_specs_never_classify_hybrid(self) -> None:
        """HYBRID needs explicit discrete+continuous keys; no plain spec mixes."""
        for kind_name, spec in _PLAIN_KIND_SPECS.items():
            assert ModelKind.HYBRID not in detect_model_kinds(spec), kind_name

    def test_composed_spec_detects_both_kinds(self) -> None:
        """F/H/Q/R + nr_agents in model_parameters classifies as both."""
        spec = _composed_spec()
        assert detect_model_kinds(spec) == {
            ModelKind.CONTINUOUS,
            ModelKind.MULTI_AGENT,
        }
        # Single-winner contract: the max-precedence winner is unchanged.
        assert detect_model_kind(spec) is ModelKind.MULTI_AGENT

    def test_composed_nr_agents_in_initialparameterization(self) -> None:
        """The agent count may live in either declared section."""
        initial = dict(_LGSSM_BLOCK)
        initial["nr_agents"] = 2
        spec = {"initialparameterization": initial}
        assert detect_model_kinds(spec) == {
            ModelKind.CONTINUOUS,
            ModelKind.MULTI_AGENT,
        }

    def test_composed_prior_only_pair(self) -> None:
        """The Gaussian-prior pair alone counts as the continuous family."""
        initial = {"prior_mean": [0.0, 0.0], "prior_cov": [[1.0, 0.0], [0.0, 1.0]]}
        spec = {"initialparameterization": {**initial, "nr_agents": 2}}
        assert detect_model_kinds(spec) == {
            ModelKind.CONTINUOUS,
            ModelKind.MULTI_AGENT,
        }

    def test_structural_stamp_short_circuits_composition(self) -> None:
        """The producer's structural stamp is authoritative, never re-guessed."""
        spec = {
            "model_kind": "structural",
            "initialparameterization": {"nr_agents": 2, **_LGSSM_BLOCK},
        }
        assert detect_model_kinds(spec) == frozenset({ModelKind.STRUCTURAL})

    def test_malformed_initialparameterization_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            detect_model_kinds({"initialparameterization": ["A", "B"]})


class TestHybridDetection:
    """HYBRID is the discrete+continuous family mix, keyed explicitly."""

    def test_hybrid_kinds(self) -> None:
        """Discrete A-E keys alongside F/H/Q/R classify as the family mix."""
        spec = _hybrid_spec()
        assert detect_model_kinds(spec) == {
            ModelKind.HYBRID,
            ModelKind.CONTINUOUS,
        }
        assert detect_model_kind(spec) is ModelKind.HYBRID

    def test_hybrid_prior_pair_branch(self) -> None:
        """The Gaussian-prior pair satisfies the predicate without F/H/Q/R."""
        spec = _hybrid_spec(prior_only=True)
        assert detect_model_kinds(spec) == {
            ModelKind.HYBRID,
            ModelKind.CONTINUOUS,
        }
        assert detect_model_kind(spec) is ModelKind.HYBRID

    def test_hybrid_with_multi_agent_keeps_multi_agent_winner(self) -> None:
        """The max-precedence winner is MULTI_AGENT, not HYBRID."""
        spec = _hybrid_multi_agent_spec()
        assert detect_model_kinds(spec) == {
            ModelKind.MULTI_AGENT,
            ModelKind.HYBRID,
            ModelKind.CONTINUOUS,
        }
        assert detect_model_kind(spec) is ModelKind.MULTI_AGENT

    def test_flat_continuous_never_classifies_hybrid(self) -> None:
        """F/H/Q/R alone (no A-E contract key) stays the plain continuous kind."""
        spec = {"initialparameterization": dict(_LGSSM_BLOCK)}
        assert detect_model_kinds(spec) == frozenset({ModelKind.CONTINUOUS})
        assert detect_model_kind(spec) is ModelKind.CONTINUOUS

    def test_factored_continuous_kinds(self) -> None:
        """num_factors > 1 alongside the Gaussian family is {FACTORED, CONTINUOUS}."""
        spec = _factored_continuous_spec()
        assert detect_model_kinds(spec) == {
            ModelKind.FACTORED,
            ModelKind.CONTINUOUS,
        }
        assert detect_model_kind(spec) is ModelKind.CONTINUOUS


class TestComposedExemplar:
    """The composed exemplar classifies as {CONTINUOUS, MULTI_AGENT}."""

    def test_extracted_pomdp_space_kind_set(self) -> None:
        gnn_file = GNN_FILES / COMPOSED_REL
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None, f"extraction failed for {gnn_file}"
        kinds = detect_pomdp_space_model_kinds(pomdp)
        assert kinds == {ModelKind.CONTINUOUS, ModelKind.MULTI_AGENT}
        assert detect_pomdp_space_model_kind(pomdp) is ModelKind.MULTI_AGENT

    def test_render_spec_dict_kind_set(self) -> None:
        gnn_file = GNN_FILES / COMPOSED_REL
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None
        spec = pomdp_to_gnn_spec(pomdp)
        assert detect_model_kinds(spec) == {
            ModelKind.CONTINUOUS,
            ModelKind.MULTI_AGENT,
        }
        assert detect_model_kind(spec) is ModelKind.MULTI_AGENT

    def test_corpus_has_no_other_composed_spec(self) -> None:
        """The four maintained composed exemplars have explicit kind sets.

        The composed receipts change behavior only for the sanctioned composed
        exemplars — continuous × multi-agent, the factored-continuous LGSSM
        ({FACTORED, CONTINUOUS}) and the hybrid ({HYBRID, CONTINUOUS}) — every
        plain exemplar must keep a singleton kind set.
        """
        from gnn.processing.discovery import is_model_source_path

        expected_composed = {
            COMPOSED_REL: frozenset({ModelKind.CONTINUOUS, ModelKind.MULTI_AGENT}),
            "continuous/independent_gaussian_agents.md": frozenset(
                {ModelKind.CONTINUOUS, ModelKind.MULTI_AGENT}
            ),
            "continuous/factored_continuous_lgssm.md": frozenset(
                {ModelKind.FACTORED, ModelKind.CONTINUOUS}
            ),
            "continuous/hybrid_discrete_continuous.md": frozenset(
                {ModelKind.HYBRID, ModelKind.CONTINUOUS}
            ),
        }
        mismatches: list[str] = []
        for gnn_file in sorted(GNN_FILES.rglob("*.md")):
            if not is_model_source_path(gnn_file):
                continue
            pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
            assert pomdp is not None, f"extraction failed for {gnn_file}"
            kinds = detect_pomdp_space_model_kinds(pomdp)
            rel = str(gnn_file.relative_to(GNN_FILES))
            if rel in expected_composed:
                if kinds != expected_composed[rel]:
                    mismatches.append(
                        f"{rel}: expected "
                        f"{sorted(k.value for k in expected_composed[rel])}, "
                        f"got {sorted(k.value for k in kinds)}"
                    )
            elif len(kinds) != 1:
                mismatches.append(f"{rel}: kind set {sorted(k.value for k in kinds)}")
        assert not mismatches, "unexpected composed kinds:\n" + "\n".join(mismatches)


class TestDispatchHonesty:
    """A composed spec is receipted, never rendered as one family."""

    def test_pipeline_validation_refuses_composition_for_every_framework(
        self, tmp_path: Path
    ) -> None:
        gnn_file = GNN_FILES / COMPOSED_REL
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None
        processor = POMDPRenderProcessor(tmp_path)
        result = processor.process_pomdp_for_all_frameworks(pomdp, gnn_file)
        assert result["overall_success"] is True
        summary = json.loads((tmp_path / "processing_summary.json").read_text())
        assert summary["frameworks_unsupported"] == summary["frameworks_requested"]
        assert summary["frameworks_processed"] == []
        assert summary["frameworks_failed"] == []
        for framework, framework_result in result["framework_results"].items():
            assert framework_result["unsupported"] is True, framework
            assert framework_result["status"] == "unsupported"
            assert "unsupported-composition" in framework_result["message"], framework

    @pytest.mark.parametrize("target", ["pymdp", "rxinfer", "jax", "stan", "discopy"])
    def test_render_gnn_spec_refuses_composed_spec(
        self, target: str, tmp_path: Path
    ) -> None:
        success, message, files = render_gnn_spec(_composed_spec(), target, tmp_path)
        assert success is False
        assert files == []
        assert "unsupported-composition" in message

    def test_render_gnn_to_rxinfer_refuses_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "composed_rxinfer.jl"
        success, message, _warnings = render_gnn_to_rxinfer(
            pomdp_to_gnn_spec(extract_pomdp_from_file(GNN_FILES / COMPOSED_REL)),  # type: ignore[arg-type]
            script,
        )
        assert success is False
        assert "unsupported-composition" in message
        assert not script.exists()

    def test_plain_continuous_spec_is_not_refused(self, tmp_path: Path) -> None:
        """The composition gate must not fire for a plain continuous spec."""
        spec = {
            "model_name": "Plain LGSSM",
            "gnn_section": "continuous",
            "initialparameterization": dict(_LGSSM_BLOCK),
            "model_parameters": {},
        }
        success, message, files = render_gnn_spec(spec, "jax", tmp_path)
        assert success is True, message
        assert files and "unsupported-composition" not in message

    def test_process_render_receipts_the_composition(self, tmp_path: Path) -> None:
        """Step 11: six native programs and three unsupported compositions."""
        result = process_render(
            target_dir=GNN_FILES / "continuous",
            output_dir=tmp_path / "11_render_output",
            frameworks=["rxinfer"],
            verbose=False,
        )
        assert result is True
        summary = json.loads(
            (
                tmp_path / "11_render_output" / "render_processing_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert summary["total_files"] == 9
        assert summary["successful_files"] == 9
        assert summary["successful_framework_renderings"] == 6
        expected_receipts = {
            "multi_agent_lgssm": "unsupported-composition",
            # The factored exemplar is receipted by the pomdp_processor
            # validation gate before rxinfer's own composed refusal.
            "factored_continuous_lgssm": "unsupported-factored-continuous",
            "hybrid_discrete_continuous": "unsupported-composition",
        }
        receipts = summary["unsupported_framework_renderings"]
        for stem, prefix in expected_receipts.items():
            entries = [entry for entry in receipts if stem in entry["file"]]
            assert len(entries) == 1, stem
            assert entries[0]["framework"] == "rxinfer"
            assert prefix in entries[0]["message"], stem
        rendered_jl = list((tmp_path / "11_render_output").rglob("*.jl"))
        assert len(rendered_jl) == 6
        assert not any(
            stem in path.name for stem in expected_receipts for path in rendered_jl
        )
