"""Contract tests for render module shared helpers and processor policy.

Pins real behavior of the pieces extracted/refactored in the 2026-09-04
render composability pass:
  - ``render.naming`` (shared output-stem sanitization + atomic writes)
  - ``render.spec_matrices`` (shared discrete A/B/C/D extraction + literals)
  - ``render.framework_registry`` lite preset
  - ``render.processor.parse_frameworks_selection`` (CLI --frameworks policy)
  - ``render.processor._render_succeeded`` (success-policy contract)
  - ``render.validate_render`` facade contract
  - ``render.mcp.render_spec_to_format_mcp`` (single-framework MCP tool)
  - POMDPRenderProcessor unknown-framework dispatch contract

All tests are deterministic, isolated (tmp_path), and network-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gnn.render import validate_render
from gnn.render.framework_registry import LITE_FRAMEWORKS, get_lite_frameworks
from gnn.render.naming import MAX_STEM_LENGTH, atomic_write_text, safe_output_stem
from gnn.render.processor import (
    _render_succeeded,
    parse_frameworks_selection,
)
from gnn.render.spec_matrices import (
    extract_abcd_matrices,
    format_array_literal,
    parse_gnn_matrix_value,
)

SAMPLE_GNN = (
    Path(__file__).parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


class TestSafeOutputStem:
    """Output-stem sanitization contract (shared by both processors)."""

    def test_replaces_unsafe_characters(self) -> None:
        assert safe_output_stem("a b/c") == "a_b_c"

    def test_strips_leading_and_trailing_separators(self) -> None:
        assert safe_output_stem("..model__") == "model"

    def test_empty_falls_back(self) -> None:
        assert safe_output_stem("") == "model"
        assert safe_output_stem("///") == "model"
        assert safe_output_stem("", fallback="pomdp_model") == "pomdp_model"

    def test_truncates_to_120_chars(self) -> None:
        assert len(safe_output_stem("x" * 500)) == MAX_STEM_LENGTH

    def test_keeps_safe_characters(self) -> None:
        assert safe_output_stem("Model-1.2_3") == "Model-1.2_3"


class TestAtomicWriteText:
    def test_writes_content_and_creates_nested_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "out.py"
        result = atomic_write_text(target, "print('hi')\n")
        assert result == target
        assert target.read_text() == "print('hi')\n"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")
        assert target.read_text() == "second"

    def test_leaves_no_temp_files_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "x")
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []


class TestExtractAbcdMatrices:
    def test_empty_spec_gets_neutral_defaults(self) -> None:
        a, b, c, d = extract_abcd_matrices({})
        assert a.shape == (2, 2)
        assert np.allclose(np.diag(a), 1.0)
        assert b.shape == (2, 2)
        assert c.tolist() == [1.0, 0.0]
        assert np.allclose(d, [0.5, 0.5])

    def test_state_space_parameters_take_precedence(self) -> None:
        spec = {
            "stateSpace": {"size": 3, "parameters": {"A": [[1.0, 2.0, 3.0]] * 3}},
            "initialparameterization": {"A": [[9.0]]},
        }
        a, _, _, _ = extract_abcd_matrices(spec)
        assert a.shape == (3, 3)
        assert np.allclose(a, 1.0 / 3.0)  # columns normalized from [3, 6, 9] sums

    def test_initialparameterization_used_when_state_space_empty(self) -> None:
        spec = {"initialparameterization": {"A": [[1.0, 1.0], [1.0, 1.0]]}}
        a, _, _, _ = extract_abcd_matrices(spec)
        assert np.allclose(a, 0.5)

    def test_d_is_normalized_to_probability_vector(self) -> None:
        _, _, _, d = extract_abcd_matrices({"parameters": {"D": [2.0, 2.0, 4.0]}})
        assert np.allclose(d, [0.25, 0.25, 0.5])

    def test_string_matrix_parsed_via_safe_literal(self) -> None:
        spec = {"parameters": {"A": "[[0.6, 0.25], [0.4, 0.75]]"}}
        a, _, _, _ = extract_abcd_matrices(spec)
        assert np.allclose(a, [[0.6, 0.25], [0.4, 0.75]])

    def test_parse_gnn_matrix_value_invalid_string_returns_default(self) -> None:
        sentinel = np.eye(2)
        assert parse_gnn_matrix_value("not a literal", sentinel) is sentinel
        assert parse_gnn_matrix_value(None, sentinel) is sentinel


class TestFormatArrayLiteral:
    def test_one_dimensional(self) -> None:
        out = format_array_literal(np.array([0.5, 0.25]), prefix="jnp.array")
        assert out == "jnp.array([0.5, 0.25])"

    def test_two_dimensional_with_suffix_and_indent(self) -> None:
        out = format_array_literal(
            np.array([[1.0, 2.0]]),
            prefix="torch.tensor",
            suffix=", dtype=torch.float64",
            indent=2,
        )
        assert out == ("torch.tensor([\n      [1.0, 2.0]\n  ], dtype=torch.float64)")
        assert out.startswith("torch.tensor([")
        assert out.endswith("dtype=torch.float64)")

    def test_normalized_distribution_round_trips_to_a_simplex(self) -> None:
        # Uniform priors over 3 and 256 states must still sum to 1 once the
        # literal is evaluated: 6-decimal rounding produced 0.999999 / 0.999936,
        # which NumPyro's Categorical rejected at run time.
        for n in (3, 256):
            arr = np.full(n, 1.0 / n)
            out = format_array_literal(arr, prefix="list")
            values = eval(out)  # nosec B307 - literal produced by the formatter
            assert abs(sum(values) - 1.0) < 1e-12
            assert values == arr.tolist()

    def test_three_dimensional_falls_back_to_repr(self) -> None:
        arr = np.ones((1, 1, 1))
        out = format_array_literal(arr, prefix="jnp.array")
        assert out == f"jnp.array({arr.tolist()})"


class TestLiteFrameworkPreset:
    def test_preset_matches_registry_constant(self) -> None:
        assert get_lite_frameworks() == list(LITE_FRAMEWORKS)

    def test_preset_contents_are_registered_frameworks(self) -> None:
        from gnn.render.framework_registry import get_supported_frameworks

        supported = get_supported_frameworks()
        for name in LITE_FRAMEWORKS:
            assert name in supported

    def test_preset_excludes_julia_backends(self) -> None:
        assert "rxinfer" not in LITE_FRAMEWORKS
        assert "activeinference_jl" not in LITE_FRAMEWORKS
        assert "stan" not in LITE_FRAMEWORKS


class TestParseFrameworksSelection:
    def test_none_means_all_frameworks(self) -> None:
        frameworks, explicit = parse_frameworks_selection(None)
        assert frameworks is None
        assert explicit is False

    def test_all_keyword_normalizes_to_none(self) -> None:
        assert parse_frameworks_selection("all") == (None, False)
        assert parse_frameworks_selection("  ALL ") == (None, False)

    def test_lite_resolves_to_registry_preset(self) -> None:
        frameworks, explicit = parse_frameworks_selection("lite")
        assert frameworks == get_lite_frameworks()
        assert explicit is False

    def test_comma_separated_string_is_explicit(self) -> None:
        frameworks, explicit = parse_frameworks_selection("pymdp, jax")
        assert frameworks == ["pymdp", "jax"]
        assert explicit is True

    def test_list_selection_is_explicit(self) -> None:
        frameworks, explicit = parse_frameworks_selection(["rxinfer"])
        assert frameworks == ["rxinfer"]
        assert explicit is True


class TestRenderSucceededPolicy:
    def test_no_files_returns_exit_code_two(self) -> None:
        assert (
            _render_succeeded(
                success_count=0,
                total_files=0,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            == 2
        )

    def test_strict_mode_requires_every_framework(self) -> None:
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=1,
                total_framework_attempts=2,
                strict_framework_success=True,
            )
            is False
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=1,
                total_framework_attempts=2,
                strict_framework_success=False,
            )
            is True
        )

    def test_aggregate_policy_requires_eighty_percent_or_any_file_success(
        self,
    ) -> None:
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=79,
                total_framework_attempts=100,
            )
            is True
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=1,
                total_framework_successes=0,
                total_framework_attempts=100,
            )
            is True
        )  # success_count > 0 keeps partial-success semantics
        assert (
            _render_succeeded(
                success_count=0,
                total_files=1,
                total_framework_successes=79,
                total_framework_attempts=100,
            )
            is False
        )

    def test_no_framework_attempts_falls_back_to_file_counting(self) -> None:
        assert (
            _render_succeeded(
                success_count=2,
                total_files=2,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            is True
        )
        assert (
            _render_succeeded(
                success_count=1,
                total_files=2,
                total_framework_successes=0,
                total_framework_attempts=0,
            )
            is False
        )


class TestValidateRenderContract:
    def test_none_result_raises(self) -> None:
        with pytest.raises(ValueError, match="None"):
            validate_render(None)

    def test_empty_string_result_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_render("")

    def test_any_other_result_passes(self) -> None:
        assert validate_render("code") is True
        assert validate_render(["a.py"]) is True


class TestUnknownFrameworkDispatch:
    def test_unknown_framework_message_contract(self, tmp_path: Path) -> None:
        from gnn.render.pomdp_processor import POMDPRenderProcessor

        result = POMDPRenderProcessor(tmp_path)._call_framework_renderer(
            "no_such_backend", {}, tmp_path
        )
        assert result["success"] is False
        assert result["message"] == "No renderer implemented for no_such_backend"
        assert result["artifacts"] == []


class TestRenderSpecToFormatMcp:
    def test_renders_single_framework_end_to_end(self, tmp_path: Path) -> None:
        from gnn.render.mcp import render_spec_to_format_mcp

        out_dir = tmp_path / "out"
        result = render_spec_to_format_mcp(
            str(SAMPLE_GNN), str(out_dir), framework="bnlearn"
        )
        assert result["success"] is True, result.get("error") or result.get("message")
        assert result["framework"] == "bnlearn"
        assert len(result["output_files"]) == 1
        artifact = Path(result["output_files"][0])
        assert artifact.name.endswith("_bnlearn.py")
        assert artifact.exists()

    def test_missing_file_reports_error(self, tmp_path: Path) -> None:
        from gnn.render.mcp import render_spec_to_format_mcp

        result = render_spec_to_format_mcp(
            str(tmp_path / "missing.md"), str(tmp_path / "out")
        )
        assert result["success"] is False
        assert "missing.md" in result["error"]

    def test_unsupported_target_reports_failure_not_error(self, tmp_path: Path) -> None:
        from gnn.render.mcp import render_spec_to_format_mcp

        result = render_spec_to_format_mcp(
            str(SAMPLE_GNN), str(tmp_path / "out"), framework="definitely_not_real"
        )
        assert result["success"] is False
        assert "Unsupported target" in result["message"]


CORPUS_DISCRETE = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/discrete/actinf_pomdp_agent.md"
)
CORPUS_MULTIAGENT = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/multiagent/multi_agent_coordination.md"
)
CORPUS_CONTINUOUS = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/continuous/continuous_navigation.md"
)
CORPUS_BASICS = (
    Path(__file__).resolve().parents[2] / "input/gnn_files/basics/static_perception.md"
)


def _render_corpus_file(
    source: Path, framework: str, tmp_path: Path
) -> tuple[str, str]:
    """Render one corpus model through the canonical render_gnn_spec path."""
    from gnn import parse_gnn_file
    from gnn.render.framework_registry import FRAMEWORK_REGISTRY
    from gnn.render.processor import render_gnn_spec

    gnn_spec = parse_gnn_file(source)
    out_dir = tmp_path / framework
    out_dir.mkdir(parents=True, exist_ok=True)
    success, message, artifacts = render_gnn_spec(gnn_spec, framework, out_dir)
    assert success, message
    extension = str(FRAMEWORK_REGISTRY[framework]["file_extension"])
    canonical = [a for a in artifacts if Path(str(a)).suffix == extension]
    assert canonical, f"no {extension} artifact among {artifacts}"
    artifact = Path(canonical[0])
    return artifact.name, artifact.read_text()


class TestMaintainedOutputContracts:
    """``CONTRACTS`` stays pinned to real ``render_gnn_spec`` output shapes.

    Each case renders a corpus model through the canonical dispatch and
    asserts the contract passes on the emitted canonical artifact, so the
    contract table cannot silently rot away from the maintained
    delegated-executor output shapes again.
    """

    @pytest.mark.parametrize(
        ("framework", "source"),
        [
            ("pymdp", CORPUS_DISCRETE),
            ("rxinfer", CORPUS_DISCRETE),
            ("rxinfer", CORPUS_MULTIAGENT),
            ("activeinference_jl", CORPUS_DISCRETE),
            ("jax", CORPUS_DISCRETE),
            ("jax", CORPUS_CONTINUOUS),
            ("pytorch", CORPUS_DISCRETE),
            ("pytorch", CORPUS_CONTINUOUS),
            ("numpyro", CORPUS_DISCRETE),
            ("numpyro", CORPUS_CONTINUOUS),
            ("stan", CORPUS_DISCRETE),
            ("discopy", CORPUS_DISCRETE),
            ("bnlearn", CORPUS_BASICS),
        ],
    )
    def test_corpus_render_satisfies_contract(
        self, framework: str, source: Path, tmp_path: Path
    ) -> None:
        from gnn.render.contracts import validate_rendered_output

        artifact_name, code = _render_corpus_file(source, framework, tmp_path)
        violations = validate_rendered_output(code, framework, file_path=artifact_name)
        assert violations == []

    def test_contract_requires_maintained_delegated_pymdp_shape(self) -> None:
        """A pre-delegation inlined-matrix pymdp script fails the contract."""
        from gnn.render.contracts import validate_rendered_output

        legacy = (
            "import numpy\n"
            "A = [[0.9, 0.1], [0.1, 0.9]]\n"
            "B = [[[1.0, 0.0], [0.0, 1.0]]]\n"
        )
        violations = validate_rendered_output(
            legacy, "pymdp", file_path="legacy_pymdp.py"
        )
        assert any(violation.field == "import" for violation in violations)

    def test_rxinfer_contract_requires_model_or_shared_module(self) -> None:
        from gnn.render.contracts import validate_rendered_output

        code = "using RxInfer\nresult = infer(model = foo())\n"
        violations = validate_rendered_output(code, "rxinfer", file_path="x.jl")
        assert any(violation.field == "pattern" for violation in violations)

    def test_bnlearn_contract_closes_ninth_framework_gap(self) -> None:
        from gnn.render.contracts import CONTRACTS, validate_rendered_output

        assert "bnlearn" in CONTRACTS
        with pytest.raises(ValueError):
            validate_rendered_output("", "unknown_framework")


class TestFailureMessageActionability:
    """Render failure messages carry remediation hints and root causes."""

    def test_get_remediation_covers_dependency_backends(self) -> None:
        from gnn.render.health import get_remediation

        for framework in ("jax", "discopy", "pytorch", "numpyro", "stan", "bnlearn"):
            hint = get_remediation(framework)
            assert hint is not None, framework
            assert "uv add" in hint or "julia" in hint.lower()

    def test_unsupported_target_message_lists_known_targets(
        self, tmp_path: Path
    ) -> None:
        from gnn import parse_gnn_file
        from gnn.render.processor import render_gnn_spec

        success, message, _artifacts = render_gnn_spec(
            parse_gnn_file(SAMPLE_GNN), "definitely_not_real", tmp_path
        )
        assert success is False
        assert message.startswith("Unsupported target: definitely_not_real")
        assert "pymdp" in message and "jax_pomdp" in message

    def test_generator_write_failure_propagates_cause(self, tmp_path: Path) -> None:
        """Generator exceptions must reach callers (receipt messages), not be
        swallowed behind a print-and-empty-string sentinel."""
        from gnn.render.bnlearn import generate_bnlearn_code

        target = tmp_path / "out"
        target.mkdir()
        model_data = {"model_name": "m", "variables": [], "connections": []}
        with pytest.raises(OSError):
            generate_bnlearn_code(model_data, target)


_CONTINUOUS_PROBE_SPEC = {
    "model_name": "Probe",
    "initialparameterization": {
        "F": [[1.0, 0.1], [0.0, 1.0]],
        "H": [[1.0, 0.0], [0.0, 1.0]],
        "Q": [[0.01, 0.0], [0.0, 0.01]],
        "R": [[0.05, 0.0], [0.0, 0.05]],
        "prior_mean": [0.0, 0.0],
        "prior_cov": [[1.0, 0.0], [0.0, 1.0]],
        "goal_mean": [1.0, 1.0],
        "control_gain": [0.5],
    },
    "num_timesteps": 5,
    "random_seed": 42,
    "dt": 1.0,
}


class TestEmittedArtifactHygiene:
    """Emitted scripts must not reference names they never bind.

    Regression pin for the continuous-script split: the shared body used to
    emit the numpyro-only ``run_mcmc`` branch for jax/pytorch too, where
    ``run_mcmc`` was never defined (static NameError risk). The scan is the
    same conservative implementation the benchmark consumes.
    """

    def test_undefined_names_flags_never_bound_names(self) -> None:
        from gnn.render.emitted_artifact_checks import undefined_names

        findings, star = undefined_names('if FRAMEWORK == "numpyro":\n    run_mcmc()\n')
        assert star is False
        assert ("run_mcmc", 2) in findings

    def test_undefined_names_skips_star_imports(self) -> None:
        from gnn.render.emitted_artifact_checks import undefined_names

        findings, star = undefined_names("from discopy import *\nTy('x')\n")
        assert star is True
        assert findings == []

    @pytest.mark.parametrize("backend", ["jax", "pytorch", "numpyro"])
    def test_continuous_scripts_have_no_undefined_names(self, backend: str) -> None:
        from gnn.render.continuous_common import extract_continuous_spec
        from gnn.render.continuous_script import generate_continuous_script
        from gnn.render.emitted_artifact_checks import undefined_names

        spec = extract_continuous_spec(_CONTINUOUS_PROBE_SPEC)
        code = generate_continuous_script(spec, backend)
        compile(code, backend, "exec")
        findings, star = undefined_names(code)
        assert findings == []
        assert ("run_mcmc" in code) == (backend == "numpyro")


class TestRenderDeterminism:
    """Same input + same renderer must produce byte-identical artifacts.

    Regression pin for the generation-time wall-clock removal: pymdp and
    discopy headers embedded ``datetime.now()`` values, and rxinfer strategy
    headers carried ``now()`` timestamps, so re-rendering the same model
    produced different bytes. The benchmark's determinism gate covers the
    full corpus per run; these pins cover the previously clocked backends
    in isolation.
    """

    @pytest.mark.parametrize("framework", ["pymdp", "rxinfer", "discopy"])
    def test_repeated_renders_are_byte_identical(
        self, framework: str, tmp_path: Path
    ) -> None:
        from gnn import parse_gnn_file
        from gnn.render.processor import render_gnn_spec

        extension = {"pymdp": ".py", "rxinfer": ".jl", "discopy": ".py"}[framework]
        artifacts: list[bytes] = []
        for index in range(2):
            out_dir = tmp_path / f"pass_{index}"
            out_dir.mkdir()
            success, message, paths = render_gnn_spec(
                parse_gnn_file(SAMPLE_GNN), framework, out_dir
            )
            assert success, message
            primary = next(
                Path(path) for path in paths if Path(path).suffix == extension
            )
            artifacts.append(primary.read_bytes())
        assert artifacts[0] == artifacts[1]


class TestFirstPartyImportResolution:
    """Emitted scripts' ``gnn.*`` imports must resolve from the repository.

    Regression pin for the stale-template drift class: a template importing
    a renamed first-party module compiles clean and passes every name-based
    check, yet ImportErrors the moment the emitted script runs.
    """

    def test_checker_flags_unresolvable_first_party_import(self) -> None:
        from gnn.render.emitted_artifact_checks import (
            first_party_unresolvable_imports,
        )

        code = "from gnn.execute.does_not_exist import thing\n"
        findings = first_party_unresolvable_imports(code)
        assert ("gnn.execute.does_not_exist", 1) in findings

    def test_checker_accepts_resolvable_first_party_import(self) -> None:
        from gnn.render.emitted_artifact_checks import (
            first_party_unresolvable_imports,
        )

        code = "from gnn.execute.pymdp import execute_pymdp_simulation\n"
        assert first_party_unresolvable_imports(code) == []

    def test_corpus_pymdp_runner_imports_resolve(self, tmp_path: Path) -> None:
        from gnn import parse_gnn_file
        from gnn.render.emitted_artifact_checks import (
            first_party_unresolvable_imports,
        )
        from gnn.render.processor import render_gnn_spec

        success, message, paths = render_gnn_spec(
            parse_gnn_file(SAMPLE_GNN), "pymdp", tmp_path
        )
        assert success, message
        primary = next(Path(path) for path in paths if Path(path).suffix == ".py")
        assert first_party_unresolvable_imports(primary.read_text()) == []


class TestMatrixShapeParity:
    """Cross-backend matrix shapes must agree per model.

    The extractor understands all four maintained Python emission
    conventions (pymdp ``*_data`` lists, jax dict-payload calls,
    pytorch/numpyro ``tensor``/``array`` assigns with ``B_slices`` stacking).
    """

    def test_extractor_skips_star_imports(self) -> None:
        from gnn.render.emitted_artifact_checks import matrix_shapes

        assert matrix_shapes("from discopy import *\nTy('x')\n") is None

    def test_extractor_skips_ragged_literals(self) -> None:
        from gnn.render.emitted_artifact_checks import matrix_shapes

        assert matrix_shapes("A_data = [[1.0, 2.0], [3.0]]\n").get("A") is None

    def test_extractor_flags_shape_difference(self) -> None:
        from gnn.render.emitted_artifact_checks import matrix_shapes

        left = matrix_shapes("A_data = [[1.0, 2.0], [3.0, 4.0]]\n")
        right = matrix_shapes("A_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n")
        assert left["A"] == (2, 2)
        assert right["A"] == (2, 3)

    def test_actinf_parity_across_four_backends(self, tmp_path: Path) -> None:
        from gnn import parse_gnn_file
        from gnn.render.emitted_artifact_checks import matrix_shapes
        from gnn.render.processor import render_gnn_spec

        spec = parse_gnn_file(SAMPLE_GNN)
        collected: dict[str, dict[str, tuple[int, ...]]] = {}
        for framework, extension in (
            ("pymdp", ".py"),
            ("jax", ".py"),
            ("pytorch", ".py"),
            ("numpyro", ".py"),
        ):
            out_dir = tmp_path / framework
            success, message, paths = render_gnn_spec(spec, framework, out_dir)
            assert success, message
            primary = next(
                Path(path) for path in paths if Path(path).suffix == extension
            )
            collected[framework] = matrix_shapes(primary.read_text())
        for letter in "ABCD":
            per_backend = {
                fw: shapes[letter]
                for fw, shapes in collected.items()
                if letter in shapes
            }
            assert len(per_backend) >= 2
            assert len(set(per_backend.values())) == 1, (letter, per_backend)

    def test_julia_dimension_constants_extraction(self) -> None:
        from gnn.render.emitted_artifact_checks import julia_dimension_constants

        code = (
            "const NUM_STATES = 3\nconst NUM_OBSERVATIONS = 16\nconst NUM_ACTIONS = 3\n"
        )
        assert julia_dimension_constants(code) == {
            "NUM_STATES": 3,
            "NUM_OBSERVATIONS": 16,
            "NUM_ACTIONS": 3,
        }
        assert julia_dimension_constants("x = 1") == {}

    def test_julia_hierarchical_render_is_exempt_from_joint_parity(self) -> None:
        """A native per-level hierarchical render declares the FAST level's
        dims; joint-vs-per-level comparison must be skipped, not flagged."""
        from gnn.render.emitted_artifact_checks import julia_render_factorization

        code = 'const MODEL_KIND = "hierarchical"\nconst NUM_OBSERVATIONS = 4\n'
        assert julia_render_factorization(code) == "hierarchical"
        assert julia_render_factorization('const MODEL_KIND = "flat"\n') == "flat"
        assert julia_render_factorization("no marker") is None
