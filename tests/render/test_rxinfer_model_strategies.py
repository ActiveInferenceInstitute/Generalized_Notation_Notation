"""Regression tests for RxInfer ModelKind detection + strategy dispatch.

Pins two invariants the 2026-08-05 red-team review found broken:

1. ``detect_model_kind`` is STRUCTURAL — prose in a ModelName or annotation
   must never change how a model renders (the old ``str(gnn_spec)``
   substring scan misrouted ``temporal_hierarchy.md`` on the word
   "Hierarchy" in its name and made every exemplar one doc-comment away
   from a render failure).
2. Every model receives its intended structural taxonomy and an explicit
   rendered, scientifically invalid, or unsupported outcome. Strict rendering
   preserves authored values and never silently normalizes malformed examples.

Pure Python — no Julia required, zero skips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_contract import (
    ModelKind,
    build_canonical_pomdp_spec,
    detect_model_kind,
)
from gnn.render.pomdp_processor import pomdp_to_gnn_spec
from gnn.render.rxinfer.model_strategies import (
    ContinuousStrategy,
    FactoredStrategy,
    FlatStrategy,
    HierarchicalStrategy,
    LearningStrategy,
    MultiAgentStrategy,
    get_model_strategy,
)
from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"

EXEMPLAR_COUNT = 38
INVALID_SOURCE_CONDITIONALS = {
    "basics/dynamic_perception.md": "probability mass must be one",
    "basics/static_perception.md": "probability mass must be one",
    "discrete/hmm_baseline.md": "probability mass must be one",
    "discrete/tmaze_epistemic.md": "positive finite mass",
    "hierarchical/hierarchical_pomdp.md": "probability mass must be one",
    "hierarchical/temporal_hierarchy.md": "probability mass must be one",
}

# The intended kind for every non-flat exemplar; everything else is FLAT.
EXPECTED_NON_FLAT = {
    "continuous/continuous_navigation.md": ModelKind.CONTINUOUS,
    "continuous/damped_oscillator_bias.md": ModelKind.CONTINUOUS,
    "continuous/ngclearn_lgssm.md": ModelKind.CONTINUOUS,
    "continuous/predictive_coding_agent.md": ModelKind.CONTINUOUS,
    "continuous/multi_agent_lgssm.md": ModelKind.MULTI_AGENT,
    "continuous/independent_gaussian_agents.md": ModelKind.MULTI_AGENT,
    "continuous/stochastic_dynamics.md": ModelKind.CONTINUOUS,
    "continuous/factored_continuous_lgssm.md": ModelKind.CONTINUOUS,
    "continuous/hybrid_discrete_continuous.md": ModelKind.HYBRID,
    "discrete/time_varying_dynamics.md": ModelKind.NONSTATIONARY,
    "discrete/regime_switched_dynamics.md": ModelKind.NONSTATIONARY,
    "hierarchical/hierarchical_pomdp.md": ModelKind.HIERARCHICAL,
    "hierarchical/temporal_hierarchy.md": ModelKind.HIERARCHICAL,
    "learning/dirichlet_likelihood_learning.md": ModelKind.LEARNING,
    "multiagent/multi_agent_coordination.md": ModelKind.MULTI_AGENT,
    "multiagent/multi_agent_coordination_acceptance.md": ModelKind.MULTI_AGENT,
    "multiagent/stigmergic_swarm.md": ModelKind.MULTI_AGENT,
    "structured/factorized_posterior.md": ModelKind.FACTORED,
}


def _exemplar_files() -> list:
    from gnn.processing.discovery import is_model_source_path

    files = [f for f in sorted(GNN_FILES.rglob("*.md")) if is_model_source_path(f)]
    assert len(files) == EXEMPLAR_COUNT, (
        f"expected {EXEMPLAR_COUNT} exemplars, found {len(files)}"
    )
    return files


def _canonical_spec(gnn_file: Path) -> dict:
    pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
    assert pomdp is not None, f"extraction failed for {gnn_file}"
    spec = pomdp_to_gnn_spec(pomdp)
    initial = spec.get("initialparameterization") or {}
    has_static_contract = all(key in initial for key in ("A", "B", "C", "D"))
    if spec.get("model_kind") == "continuous" or not has_static_contract:
        # Continuous specs carry F/H/Q/R and never A/B/C/D; factored
        # per-factor and non-stationary B_t/B_regime specs likewise lack a
        # static B — canonicalisation would demand one and drop the family
        # keys, so kind detection runs on the raw spec instead.
        return spec
    return build_canonical_pomdp_spec(spec)


class TestExemplarKindTaxonomy:
    """Every exemplar detects its intended kind through the real path."""

    def test_all_exemplars_detect_expected_kind(self) -> None:
        mismatches = []
        for gnn_file in _exemplar_files():
            rel = str(gnn_file.relative_to(GNN_FILES))
            expected = EXPECTED_NON_FLAT.get(rel, ModelKind.FLAT)
            # Structural classification is independent of numeric validity;
            # strict construction/refusal is checked separately below.
            pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
            assert pomdp is not None
            raw_spec = pomdp_to_gnn_spec(pomdp, preserve_discrete_structure=True)
            actual = detect_model_kind(raw_spec)
            if actual != expected:
                mismatches.append(
                    f"{rel}: expected {expected.value}, got {actual.value}"
                )
        assert not mismatches, "kind misdetections:\n" + "\n".join(mismatches)

    def test_all_exemplars_render(self, tmp_path: Path) -> None:
        """Every selected source gets a strict, explicit scientific outcome."""
        failures = []
        outcomes = {"rendered": set(), "invalid": set(), "unsupported": set()}
        for gnn_file in _exemplar_files():
            pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
            assert pomdp is not None, f"extraction failed for {gnn_file}"
            script = tmp_path / f"{gnn_file.stem}_rxinfer.jl"
            rel = str(gnn_file.relative_to(GNN_FILES))
            if rel in INVALID_SOURCE_CONDITIONALS:
                with pytest.raises(ValueError, match=INVALID_SOURCE_CONDITIONALS[rel]):
                    pomdp_to_gnn_spec(pomdp, native_agents=True)
                assert not script.exists()
                outcomes["invalid"].add(rel)
                continue
            spec = pomdp_to_gnn_spec(pomdp, native_agents=True)
            receipted = {
                "continuous/multi_agent_lgssm.md": "unsupported-composition",
                "continuous/hybrid_discrete_continuous.md": ("unsupported-composition"),
                "continuous/factored_continuous_lgssm.md": ("unsupported-composition"),
                "discrete/time_varying_dynamics.md": ("unsupported-nonstationary"),
                "discrete/regime_switched_dynamics.md": ("unsupported-nonstationary"),
            }
            if rel in receipted:
                success, message, _warnings = render_gnn_to_rxinfer(spec, script)
                assert not success
                assert receipted[rel] in message
                assert not script.exists()
                outcomes["unsupported"].add(rel)
                continue
            success, message, _warnings = render_gnn_to_rxinfer(spec, script)
            if not success:
                failures.append(f"{gnn_file.name}: {message}")
            else:
                assert script.exists() and script.stat().st_size > 0
                outcomes["rendered"].add(rel)
        assert not failures, "render failures:\n" + "\n".join(failures)
        assert outcomes["invalid"] == set(INVALID_SOURCE_CONDITIONALS)
        assert len(outcomes["rendered"]) == 27
        assert len(outcomes["unsupported"]) == 5
        assert set.union(*outcomes.values()) == {
            str(path.relative_to(GNN_FILES)) for path in _exemplar_files()
        }


class TestStructuralDetection:
    """detect_model_kind reads typed fields only — never prose."""

    _BASE_INITIAL = {
        "A": [[1.0, 0.0], [0.0, 1.0]],
        "B": [[[1.0], [0.0]], [[0.0], [1.0]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
    }

    def test_prose_mentions_do_not_reroute(self) -> None:
        """The exact words that used to misroute renders are now inert."""
        spec = {
            "model_name": "A Hierarchy of Multi_Agent Dirichlet Learning Models",
            "description": (
                "hierarchy multi_agent dirichlet stochastic_dynamics learning"
            ),
            "model_parameters": {},
            "initialparameterization": dict(self._BASE_INITIAL),
        }
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_agent_metadata_key_does_not_misroute(self) -> None:
        """A scalar 'agent_*' metadata key is not an agent declaration."""
        initial = dict(self._BASE_INITIAL)
        initial["agent_note"] = "single agent model"
        spec = {"initialparameterization": initial, "model_parameters": {}}
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_non_mapping_initialparameterization_raises(self) -> None:
        """No silent iteration over lists/strings — fail loud."""
        for bad in (["agent_config", "B"], "agent", [("A", 1)], 42):
            with pytest.raises(ValueError, match="must be a mapping"):
                detect_model_kind({"initialparameterization": bad})

    def test_nr_agents_one_stays_flat(self) -> None:
        initial = dict(self._BASE_INITIAL)
        initial["nr_agents"] = 1
        spec = {"initialparameterization": initial, "model_parameters": {}}
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_agent_matrix_keys_in_structured_pomdp_detect_multi_agent(self) -> None:
        spec = {
            "initialparameterization": dict(self._BASE_INITIAL),
            "model_parameters": {},
            "structured_pomdp": {
                "matrices": {"A_agent1": [[1.0]], "A_agent2": [[1.0]]}
            },
        }
        assert detect_model_kind(spec) == ModelKind.MULTI_AGENT

    def test_level_matrix_keys_detect_hierarchical(self) -> None:
        spec = {
            "initialparameterization": dict(self._BASE_INITIAL),
            "model_parameters": {},
            "structured_pomdp": {
                "matrices": {"A_level1": [[1.0]], "A_level2": [[1.0]]}
            },
        }
        assert detect_model_kind(spec) == ModelKind.HIERARCHICAL

    def test_continuous_parameterization_keys_detect_continuous(self) -> None:
        initial = dict(self._BASE_INITIAL)
        initial.update({"F": [[1.0]], "H": [[1.0]], "Q": [[0.1]], "R": [[0.1]]})
        spec = {"initialparameterization": initial, "model_parameters": {}}
        # Discrete A-E contract keys + a flat Gaussian parameterization is
        # the HYBRID family mix: refused at dispatch with an
        # unsupported-composition receipt, never silently rendered as one
        # family with the other dropped.
        assert detect_model_kind(spec) == ModelKind.HYBRID

    def test_pure_continuous_parameterization_detects_continuous(self) -> None:
        spec = {
            "initialparameterization": {
                "F": [[1.0]],
                "H": [[1.0]],
                "Q": [[0.1]],
                "R": [[0.1]],
            },
            "model_parameters": {},
        }
        assert detect_model_kind(spec) == ModelKind.CONTINUOUS


class TestStrategyDispatchAndCodegen:
    """Strategies stamp their own kind and generate runnable-shaped code."""

    def test_joint_composition_strategies_inherit_flat_codegen(self) -> None:
        """Multi-agent keeps the joint composition as its fallback path.

        MultiAgentStrategy renders natively (per-agent + shared env trace)
        when the spec declares >= 2 complete agent groups, and falls back to
        the documented joint composition otherwise — both through the flat
        codegen lineage. FactoredStrategy went native (D3): it must NOT reuse
        the flat codegen.
        """
        assert isinstance(get_model_strategy(ModelKind.FACTORED), FactoredStrategy)
        assert isinstance(get_model_strategy(ModelKind.MULTI_AGENT), MultiAgentStrategy)
        assert isinstance(MultiAgentStrategy(), FlatStrategy)
        # FactoredStrategy went native (D3): it must NOT reuse the flat codegen.
        assert not isinstance(FactoredStrategy(), FlatStrategy)

    def test_every_kind_has_a_registered_strategy(self) -> None:
        """Every renderable kind has a strategy. STRUCTURAL, HYBRID and
        NONSTATIONARY are deliberately strategy-less: a wrapper spec is
        render-only / informational, and hybrid / non-stationary family
        mixes are refused with receipts upstream
        (``unsupported-composition`` / ``unsupported-nonstationary``) —
        never silently rendered as one family with the other dropped."""
        strategy_less = {
            ModelKind.STRUCTURAL,
            ModelKind.HYBRID,
            ModelKind.NONSTATIONARY,
        }
        for kind in ModelKind:
            if kind in strategy_less:
                with pytest.raises((KeyError, ValueError)):
                    get_model_strategy(kind)
            else:
                assert get_model_strategy(kind).kind is kind

    def test_multi_agent_script_stamps_true_kind_and_echoes_factors(self) -> None:
        gnn_file = GNN_FILES / "multiagent" / "multi_agent_coordination.md"
        code = MultiAgentStrategy().generate_model_code(
            _canonical_spec(gnn_file), "multi_agent_coordination"
        )
        assert 'const MODEL_KIND = "multi_agent"' in code
        assert '"state_factors"' in code

    def test_flat_script_contains_habit_prior_policy(self) -> None:
        """D2: E enters action selection via log-add."""
        gnn_file = GNN_FILES / "discrete" / "actinf_pomdp_agent.md"
        code = FlatStrategy().generate_model_code(
            _canonical_spec(gnn_file), "actinf_pomdp_agent"
        )
        assert "log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values" in code
        assert '"E" => E' in code

    def test_hierarchical_two_level_generates_native_script(self) -> None:
        gnn_file = GNN_FILES / "hierarchical" / "hierarchical_pomdp.md"
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None
        # An explicit test conditional, not a repair of the authored example
        # whose A_level2 columns have mass two. Distinct contexts stay distinct.
        likelihood = [[0.6, 0.1], [0.1, 0.6], [0.2, 0.1], [0.1, 0.2]]
        pomdp.matrices["A_level2"] = likelihood
        spec = build_canonical_pomdp_spec(pomdp_to_gnn_spec(pomdp))
        expected = np.kron(np.asarray(pomdp.matrices["A_level1"]), likelihood)
        np.testing.assert_allclose(
            spec["initialparameterization"]["A"], expected, rtol=0, atol=1e-15
        )
        assert spec["structured_pomdp"]["matrices"]["A_level2"] == likelihood
        code = HierarchicalStrategy().generate_model_code(
            spec, "hierarchical_valid_test_fixture"
        )
        assert "hierarchical_pomdp_model" in code
        assert "hierarchical_constraints()" in code
        assert "hierarchical_initialization(NUM_FAST, NUM_SLOW)" in code
        assert 'const MODEL_KIND = "hierarchical"' in code
        assert '"slow_context"' in code

    def test_hierarchical_three_level_renders_joint_composition(self) -> None:
        """3+ declared levels use the documented joint-composition path."""
        gnn_file = GNN_FILES / "hierarchical" / "temporal_hierarchy.md"
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None
        # Each fixture column is an explicitly authored conditional. They are
        # asymmetric and distinguish all levels without normalizing the source.
        likelihoods = {
            "A_level0": [
                [0.8, 0.1, 0.1, 0.2],
                [0.1, 0.8, 0.1, 0.3],
                [0.1, 0.1, 0.8, 0.5],
            ],
            "A_level1": [
                [0.7, 0.1, 0.1],
                [0.1, 0.7, 0.1],
                [0.1, 0.1, 0.7],
                [0.1, 0.1, 0.1],
            ],
            "A_level2": [[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]],
        }
        pomdp.matrices.update(likelihoods)
        spec = build_canonical_pomdp_spec(pomdp_to_gnn_spec(pomdp))
        expected = np.kron(
            likelihoods["A_level0"],
            np.kron(likelihoods["A_level1"], likelihoods["A_level2"]),
        )
        np.testing.assert_allclose(
            spec["initialparameterization"]["A"], expected, rtol=0, atol=1e-15
        )
        for key, value in likelihoods.items():
            assert spec["structured_pomdp"]["matrices"][key] == value
        code = HierarchicalStrategy().generate_model_code(
            spec, "temporal_hierarchy_valid_test_fixture"
        )
        assert 'const MODEL_KIND = "hierarchical"' in code
        # Joint composition means the flat pomdp_model, not the native chain.
        assert "using GnnRxInferModels: pomdp_model" in code

    def test_hierarchical_missing_level_matrices_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
            "structured_pomdp": {
                "matrices": {"A_level1": [[1.0]], "A_level2": [[1.0]]}
            },
        }
        with pytest.raises(ValueError, match="missing per-level"):
            HierarchicalStrategy().generate_model_code(spec, "broken_hierarchical")

    def test_hierarchical_validation_fields_extend_flat(self) -> None:
        fields = HierarchicalStrategy().get_validation_fields()
        assert "context_beliefs_valid" in fields
        assert "context_beliefs_sum_to_one" in fields
        assert "belief_accuracy" in fields


class TestFactoredNativeCodegen:
    """D3: two-factor exemplars render the native mean-field chain."""

    def _code(self) -> str:
        gnn_file = GNN_FILES / "structured" / "factorized_posterior.md"
        return FactoredStrategy().generate_model_code(
            _canonical_spec(gnn_file), "factorized_posterior"
        )

    def test_uses_the_native_factored_model(self) -> None:
        code = self._code()
        assert "factored_pomdp_model" in code
        assert "factored_constraints()" in code
        assert "factored_initialization(N_F0, N_F1)" in code
        # The joint composition's flat model must be gone.
        assert "using GnnRxInferModels: pomdp_model" not in code

    def test_stamps_kind_and_real_factor_names(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "factored"' in code
        assert 'const FACTOR0_NAME = "s_f0"' in code
        assert 'const FACTOR1_NAME = "s_f1"' in code
        assert '"beliefs_by_factor" => Dict(' in code
        assert '"posterior_family" => "mean_field_factorized"' in code

    def test_loads_per_factor_matrices_not_the_joint(self) -> None:
        code = self._code()
        for key in ("A_m0", "A_m1", "B_f0", "B_f1", "D_f0", "D_f1"):
            assert f'matrices["{key}"]' in code
        assert 'const B_TENSOR_ORDER = "next_state_previous_state_action"' in code

    def test_validation_fields_cover_both_factors(self) -> None:
        fields = FactoredStrategy().get_validation_fields()
        assert "beliefs_sum_to_one" in fields
        assert "factor1_beliefs_valid" in fields
        assert "factor1_beliefs_sum_to_one" in fields

    def test_missing_per_factor_matrices_raises(self) -> None:
        spec = {
            "model_parameters": {"num_factors": 2, "num_modalities": 2},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
            "structured_pomdp": {"matrices": {"A_m0": [[[1.0]]]}},
        }
        with pytest.raises(ValueError, match="cannot render natively"):
            FactoredStrategy().generate_model_code(spec, "broken_factored")

    def test_wrong_factor_count_raises(self) -> None:
        """Three factors is not the two-factor native path — fail loud."""
        gnn_file = GNN_FILES / "structured" / "factorized_posterior.md"
        spec = _canonical_spec(gnn_file)
        spec["model_parameters"]["num_factors"] = 3
        with pytest.raises(ValueError, match="num_factors is 3"):
            FactoredStrategy().generate_model_code(spec, "factorized_posterior")


class TestContinuousNativeCodegen:
    """A2: continuous exemplars render the linear-Gaussian state-space model."""

    def _code(self, stem: str = "continuous_navigation") -> str:
        gnn_file = GNN_FILES / "continuous" / f"{stem}.md"
        return ContinuousStrategy().generate_model_code(_canonical_spec(gnn_file), stem)

    def test_uses_the_native_continuous_model(self) -> None:
        code = self._code()
        assert "using GnnRxInferModels: continuous_pomdp_model" in code
        assert "continuous_pomdp_model(F = F, H = H, Q = Q, R = R," in code
        # Fully conjugate: no constraints/initialization are needed or passed.
        assert "constraints =" not in code
        assert "initialization =" not in code

    def test_stamps_kind_and_parameterization(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "continuous"' in code
        assert '"parameterization" => "linear_gaussian_state_space"' in code
        assert "posterior_cov" in code
        assert '"true_states_continuous" => true_states_continuous' in code

    def test_validation_uses_finiteness_not_positive_free_energy(self) -> None:
        """Continuous Bethe FE is routinely negative — vfe > 0 would be wrong."""
        code = self._code()
        assert "all(isfinite, vfe_per_iteration)" in code
        assert "all(v -> v > 0, vfe_per_iteration)" not in code
        assert "posterior_cov_psd" in code
        assert "rmse_vs_true" in code

    def test_emits_no_fabricated_policy_data(self) -> None:
        code = self._code()
        assert '"efe_per_action" => Vector{Vector{Float64}}(),' in code
        assert '"policy_posterior" => Vector{Vector{Float64}}(),' in code

    def test_continuous_exemplars_render(self) -> None:
        for stem in (
            "continuous_navigation",
            "predictive_coding_agent",
            "stochastic_dynamics",
            "damped_oscillator_bias",
        ):
            assert 'const MODEL_KIND = "continuous"' in self._code(stem)

    def test_validation_fields(self) -> None:
        fields = ContinuousStrategy().get_validation_fields()
        assert fields == [
            "vfe_finite",
            "means_finite",
            "posterior_cov_psd",
            "inference_converged",
            "rmse_vs_true",
            "rmse_finite",
        ]

    def test_missing_continuous_parameterization_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
        }
        with pytest.raises(
            ValueError, match=r"continuous spec is missing.*'F'.*'prior_cov'"
        ):
            ContinuousStrategy().generate_model_code(spec, "no_lgssm")

    def test_partial_continuous_parameterization_names_the_gap(self) -> None:
        initial = dict(TestStructuralDetection._BASE_INITIAL)
        initial.update({"F": [[1.0]], "H": [[1.0]]})
        spec = {"model_parameters": {}, "initialparameterization": initial}
        with pytest.raises(ValueError) as excinfo:
            ContinuousStrategy().generate_model_code(spec, "partial_lgssm")
        message = str(excinfo.value)
        assert "'Q'" in message and "'R'" in message
        assert "'F'" not in message


class TestLearningNativeCodegen:
    """D1: dirichlet_A exemplars render the latent-likelihood model."""

    def _code(self) -> str:
        gnn_file = GNN_FILES / "learning" / "dirichlet_likelihood_learning.md"
        return LearningStrategy().generate_model_code(
            _canonical_spec(gnn_file), "dirichlet_likelihood_learning"
        )

    def test_uses_the_native_learning_model(self) -> None:
        code = self._code()
        assert "learning_pomdp_model" in code
        assert "learning_constraints()" in code
        assert "learning_initialization(prior_counts, NUM_STATES)" in code
        assert "limit_stack_depth = 500" in code

    def test_stamps_kind_and_learned_parameters(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "learning"' in code
        assert '"learned_parameters" => ["A"]' in code
        assert 'initial["dirichlet_A"]' in code
        assert '"learned_A_mean" => matrix_rows(A_learned_mean)' in code

    def test_agent_acts_on_the_prior_mean_not_the_true_likelihood(self) -> None:
        """The environment uses true A; the agent uses its Dirichlet belief."""
        code = self._code()
        assert "A_prior_mean = prior_counts ./ sum(prior_counts, dims = 1)" in code
        assert "observation = categorical_index(A_true[:, current_state])" in code
        assert "select_action(current_belief, A_prior_mean, B, C_pref, E)" in code

    def test_learning_is_a_hard_gate(self) -> None:
        code = self._code()
        assert "a_distance_prior" in code
        assert "a_distance_posterior" in code
        assert 'validation["a_learning_improved"]' in code

    def test_vfe_present_means_finite_for_dirichlet_models(self) -> None:
        code = self._code()
        assert "vfe_present = !isempty(vfe_per_iteration) && all(isfinite," in code
        assert "all(v -> v > 0, vfe_per_iteration)" not in code

    def test_validation_fields(self) -> None:
        fields = LearningStrategy().get_validation_fields()
        assert "a_learning_improved" in fields
        assert "a_posterior_columns_normalized" in fields
        assert "belief_accuracy" in fields

    def test_missing_dirichlet_counts_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
        }
        with pytest.raises(ValueError, match="dirichlet_A"):
            LearningStrategy().generate_model_code(spec, "no_dirichlet")


class TestOnlineInferenceMode:
    """A1: inference_mode='online' generates the per-timestep filtering loop."""

    def _spec(self) -> dict:
        gnn_file = GNN_FILES / "discrete" / "simple_mdp.md"
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None, f"extraction failed for {gnn_file}"
        return pomdp_to_gnn_spec(pomdp)

    def test_online_option_generates_filtering_script(self, tmp_path: Path) -> None:
        script = tmp_path / "online.jl"
        success, message, _ = render_gnn_to_rxinfer(
            self._spec(), script, options={"inference_mode": "online"}
        )
        assert success, message
        text = script.read_text(encoding="utf-8")
        assert 'const INFERENCE_MODE = "online"' in text
        assert "filtered_posterior" in text
        assert '"inference_mode" => INFERENCE_MODE' in text
        # Action selection still uses the habit prior + EFE
        assert "log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION" in text

    def test_default_stays_batch(self, tmp_path: Path) -> None:
        script = tmp_path / "batch.jl"
        success, message, _ = render_gnn_to_rxinfer(self._spec(), script)
        assert success, message
        text = script.read_text(encoding="utf-8")
        assert "INFERENCE_MODE" not in text
        assert "filtered_posterior" not in text

    def test_spec_declaration_wins_over_option(self, tmp_path: Path) -> None:
        spec = self._spec()
        spec["model_parameters"]["inference_mode"] = "batch"
        script = tmp_path / "declared.jl"
        success, message, _ = render_gnn_to_rxinfer(
            spec, script, options={"inference_mode": "online"}
        )
        assert success, message
        assert "filtered_posterior" not in script.read_text(encoding="utf-8")

    def test_invalid_mode_raises(self) -> None:
        spec = self._spec()
        spec["model_parameters"]["inference_mode"] = "streaming"
        with pytest.raises(ValueError, match="inference_mode"):
            FlatStrategy().generate_model_code(
                build_canonical_pomdp_spec(spec), "simple_mdp"
            )
