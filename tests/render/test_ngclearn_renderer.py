"""ngc-learn render backend contracts (T1).

Covers the four render-side ngclearn contracts without importing ngclearn
(marker-gated to py3.12, absent from the dev venv):
  1. continuous specs render a standalone script carrying the ngclearn stamp,
     the shared Kalman-filter loop, and NGCLEARN_OUTPUT_DIR routing, with
     Kalman numerics byte-identical to the jax backend
  2. discrete POMDPs are first-class unsupported (continuous-only message)
  3. the registry entry carries the backend contract fields and the derived
     POMDP config follows it
  4. the continuous dispatch table routes ngclearn instead of refusing it
"""

from __future__ import annotations

import py_compile
from pathlib import Path
from typing import Any, Dict, cast

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.continuous_common import extract_continuous_spec
from gnn.render.continuous_script import generate_continuous_script
from gnn.render.framework_registry import (
    FRAMEWORK_REGISTRY,
    get_pomdp_framework_configs,
)
from gnn.render.ngclearn import render_gnn_to_ngclearn
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import render_gnn_spec

T = 8

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_EXEMPLAR = (
    REPO / "input" / "gnn_files" / "continuous" / "damped_oscillator_bias.md"
)

_REGISTRY_FIELDS = {
    "name",
    "description",
    "language",
    "file_extension",
    "supported_features",
    "function",
    "output_format",
    "pomdp_compatible",
    "requires_matrices",
    "optional_matrices",
    "supports_multi_modality",
    "supports_multi_factor",
    "available",
    "supports_execution",
    "supports_continuous",
    "continuous_only",
    "unavailable_reason",
}

_POMDP_UNSUPPORTED_MESSAGE = (
    "discrete POMDP: ngc-learn supports continuous linear-Gaussian models only"
)

_DISCRETE_MESSAGE = (
    "discrete POMDP: ngclearn supports continuous linear-Gaussian models only"
)


def _continuous_spec() -> Dict[str, Any]:
    return {
        "name": "Test Continuous",
        "model_name": "Test Continuous",
        "gnn_section": "ActInfContinuous",
        "model_kind": "continuous",
        "initialparameterization": {
            "F": [[1.0, 0.1], [0.0, 0.9]],
            "H": [[1.0, 0.0], [0.0, 1.0]],
            "Q": [[0.05, 0.0], [0.0, 0.05]],
            "R": [[0.1, 0.0], [0.0, 0.1]],
            "prior_mean": [0.0, 0.0],
            "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
        },
        "model_parameters": {"num_timesteps": T, "dt": 0.1, "random_seed": 7},
    }


def _discrete_spec() -> Dict[str, Any]:
    return {
        "name": "Disc",
        "model_name": "Disc",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
        },
    }


def _file_spec() -> Dict[str, Any]:
    """Spec for the continuous exemplar, built through the real file pipeline."""
    pomdp = extract_pomdp_from_file(CONTINUOUS_EXEMPLAR, strict_validation=True)
    assert pomdp is not None
    return cast(Dict[str, Any], pomdp_to_gnn_spec(pomdp))


def _kalman_tail(script: str) -> str:
    """Everything from the shared Kalman step onward (backend-independent)."""
    return script[script.index("def kalman_step") :]


# ── 1. Continuous rendering (codegen-only; the script is never executed) ───


def test_ngclearn_continuous_renders_with_stamp_and_routing(tmp_path: Path) -> None:
    out = tmp_path / "m_ngclearn.py"
    ok, msg, arts = render_gnn_to_ngclearn(_continuous_spec(), out)
    assert ok, msg
    assert arts == [str(out)]
    script = out.read_text()
    # ngclearn stamp: import guard + framework identity + version keys.
    assert "import ngclearn" in script
    assert 'FRAMEWORK = "ngclearn"' in script
    assert '"ngclearn_version": ngclearn.__version__' in script
    assert '"jax_version": jax.__version__' in script
    assert "ERROR: ngc-learn not installed" in script
    assert "uv sync --extra ngclearn" in script
    # Output routing: results land under NGCLEARN_OUTPUT_DIR.
    assert "OUTPUT_ENV = 'NGCLEARN_OUTPUT_DIR'" in script
    # Shared online Kalman filter loop (Joseph form) is present.
    assert "def kalman_step" in script
    assert "Joseph form" in script
    # The script must be valid Python even though ngclearn is absent locally.
    py_compile.compile(str(out), doraise=True)


def test_ngclearn_kalman_numerics_byte_identical_to_jax() -> None:
    spec = extract_continuous_spec(_continuous_spec())
    ngclearn_script = generate_continuous_script(spec, "ngclearn")
    jax_script = generate_continuous_script(spec, "jax")
    assert _kalman_tail(ngclearn_script) == _kalman_tail(jax_script)
    # x64 must be enabled exactly like the jax backend so results compare.
    assert "jax_enable_x64" in ngclearn_script


def test_ngclearn_renders_continuous_exemplar_file_spec(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_ngclearn(_file_spec(), tmp_path / "damped_ngclearn.py")
    assert ok, msg
    script = Path(arts[0]).read_text()
    assert 'FRAMEWORK = "ngclearn"' in script
    assert "OUTPUT_ENV = 'NGCLEARN_OUTPUT_DIR'" in script
    py_compile.compile(arts[0], doraise=True)


def test_generated_script_exports_continuous_result_schema(tmp_path: Path) -> None:
    """Static schema pin: the emitted results dict carries the continuous
    contract keys (rmse_vs_true, beliefs, posterior_cov, ...)."""
    ok, _msg, arts = render_gnn_to_ngclearn(_continuous_spec(), tmp_path / "m.py")
    assert ok
    script = Path(arts[0]).read_text()
    for key in (
        '"rmse_vs_true": rmse',
        '"beliefs": beliefs',
        '"posterior_cov": covs',
        '"true_states_continuous": true_states',
        '"observations_continuous": observations',
        'results["validation"] = validation',
    ):
        assert key in script


# ── 2. Discrete POMDPs are first-class unsupported ─────────────────────────


def test_ngclearn_discrete_spec_is_first_class_unsupported(tmp_path: Path) -> None:
    out = tmp_path / "d_ngclearn.py"
    ok, msg, arts = render_gnn_to_ngclearn(_discrete_spec(), out)
    assert ok is False
    assert msg == _DISCRETE_MESSAGE
    assert arts == []
    assert not out.exists()


def test_ngclearn_structural_wrapper_spec_is_unsupported(tmp_path: Path) -> None:
    spec = {
        "model_name": "Blanket",
        "variables": [],
        "connections": [],
    }
    ok, msg, arts = render_gnn_to_ngclearn(spec, tmp_path / "s_ngclearn.py")
    assert ok is False
    assert msg == _DISCRETE_MESSAGE
    assert arts == []


# ── 3. Registry contract ────────────────────────────────────────────────────


def test_registry_entry_carries_backend_contract() -> None:
    spec = FRAMEWORK_REGISTRY["ngclearn"]
    assert set(spec) == _REGISTRY_FIELDS
    assert spec["name"] == "ngc-learn"
    assert spec["language"] == "Python"
    assert spec["file_extension"] == ".py"
    assert spec["output_format"] == "python"
    assert spec["function"] == "render_gnn_to_ngclearn"
    assert spec["pomdp_compatible"] is True
    assert spec["requires_matrices"] == []
    assert spec["optional_matrices"] == ["F", "H", "Q", "R", "prior_mean", "prior_cov"]
    assert spec["supports_continuous"] is True
    assert spec["continuous_only"] is True
    assert spec["supports_execution"] is True
    assert spec["available"] is True
    assert spec["unavailable_reason"] is None


def test_registry_position_pomdp_config_and_lite_exclusion() -> None:
    from gnn.frameworks import ALL_FRAMEWORKS, LITE_FRAMEWORKS

    # Registry order must match ALL_FRAMEWORKS minus the execution-only lean;
    # ngclearn follows bnlearn in both, and lean stays last.
    registry = list(FRAMEWORK_REGISTRY)
    assert registry.index("ngclearn") == registry.index("bnlearn") + 1
    assert ALL_FRAMEWORKS.count("ngclearn") == 1
    assert ALL_FRAMEWORKS.index("ngclearn") == ALL_FRAMEWORKS.index("bnlearn") + 1
    assert ALL_FRAMEWORKS[-1] == "lean"
    assert "ngclearn" not in LITE_FRAMEWORKS
    configs = get_pomdp_framework_configs()
    assert "ngclearn" in configs  # pomdp_compatible=True joins the loop
    assert configs["ngclearn"]["output_subdir"] == "ngclearn"
    assert configs["ngclearn"]["supports_continuous"] is True
    assert configs["ngclearn"]["continuous_only"] is True
    assert configs["ngclearn"]["supports_execution"] is True
    # The flag discriminates: discrete-capable backends (incl. stan, which
    # also has requires_matrices=[] + supports_continuous=True) stay False.
    assert configs["stan"]["continuous_only"] is False
    assert configs["jax"]["continuous_only"] is False
    assert configs["bnlearn"]["continuous_only"] is False


# ── 4. Continuous dispatch routes ngclearn ──────────────────────────────────


def test_continuous_dispatch_routes_ngclearn(tmp_path: Path) -> None:
    ok, msg, artifacts = render_gnn_spec(_continuous_spec(), "ngclearn", tmp_path)
    assert ok, msg
    assert artifacts, "expected the rendered script among artifacts"
    assert artifacts[0].endswith("_ngclearn.py")
    assert Path(artifacts[0]).is_file()
    # The unknown-target refusal must not fire for ngclearn.
    assert not msg.startswith(
        "Continuous models are unsupported for target: ngclearn"
    )


# ── 5. POMDPRenderProcessor loop (registry-driven routing) ─────────────────


def test_pomdp_loop_discrete_model_is_first_class_unsupported(tmp_path: Path) -> None:
    discrete = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
    pomdp = extract_pomdp_from_file(discrete, strict_validation=True)
    assert pomdp is not None
    result = POMDPRenderProcessor(tmp_path).process_pomdp_for_all_frameworks(
        pomdp, frameworks=["ngclearn"]
    )
    fr = result["framework_results"]["ngclearn"]
    assert fr["success"] is False
    assert fr["unsupported"] is True
    assert fr["status"] == "unsupported"
    assert fr["message"] == _POMDP_UNSUPPORTED_MESSAGE
    # Unsupported frameworks are excluded from the success denominator, so a
    # single-unsupported run reports overall success instead of failure.
    assert result["overall_success"] is True
    assert not (tmp_path / "ngclearn").exists()


def test_pomdp_loop_continuous_model_renders_into_ngclearn_subdir(
    tmp_path: Path,
) -> None:
    pomdp = extract_pomdp_from_file(CONTINUOUS_EXEMPLAR, strict_validation=True)
    assert pomdp is not None
    result = POMDPRenderProcessor(tmp_path).process_pomdp_for_all_frameworks(
        pomdp, frameworks=["ngclearn"]
    )
    fr = result["framework_results"]["ngclearn"]
    assert fr["success"] is True, fr["message"]
    artifacts = fr["output_files"]
    assert artifacts and Path(artifacts[0]).name.endswith("_ngclearn.py")
    artifact = Path(artifacts[0])
    assert artifact.parent == tmp_path / "ngclearn"
    assert artifact.is_file()
    assert result["overall_success"] is True


