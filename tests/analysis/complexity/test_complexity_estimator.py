"""Contract tests for the static complexity estimator (wave-8 W8-A).

Pure-unit, deterministic, zero-framework: pins the ``gnn.complexity_estimate/v1``
receipt schema, kind applicability gating, driver semantics, horizon honesty,
and deterministic ordering. Committed exemplars are asserted to exist — a
missing exemplar fails the test, it never vacuously passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.complexity import (
    ESTIMATOR_VERSION,
    RECEIPT_TYPE,
    UNBOUNDED_HORIZON,
    estimate_model_complexity,
    to_json_text,
)
from gnn.analysis.complexity.bounds import BACKEND_BOUNDS, BACKEND_ORDER
from gnn.parsers import parse_gnn_file_structured

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "input" / "gnn_files"

EXEMPLARS: dict[str, Path] = {
    "pomdp_gridworld": INPUT_DIR / "pomdp_gridworld" / "pomdp_gridworld_3x3.md",
    "tmaze_epistemic": INPUT_DIR / "discrete" / "tmaze_epistemic.md",
    "continuous_navigation": INPUT_DIR / "continuous" / "continuous_navigation.md",
    "hybrid_discrete_continuous": INPUT_DIR
    / "continuous"
    / "hybrid_discrete_continuous.md",
    "regime_switched_dynamics": INPUT_DIR / "discrete" / "regime_switched_dynamics.md",
    "multi_agent_coordination": INPUT_DIR
    / "multiagent"
    / "multi_agent_coordination.md",
}

RECEIPT_KEYS = {
    "receipt_type",
    "model",
    "structure",
    "model_kinds",
    "per_backend",
    "estimator_version",
}
MODEL_KEYS = {"name", "source_sha256", "path"}
STRUCTURE_KEYS = {
    "variable_count",
    "edge_count",
    "factor_arities",
    "total_state_space_dim",
    "max_variable_dim",
    "discrete_var_count",
    "continuous_var_count",
    "time",
}
TIME_KEYS = {"time_type", "discretization", "horizon", "is_dynamic"}
ROW_KEYS = {
    "framework",
    "applicable",
    "family",
    "asymptotic",
    "complexity_class",
    "drivers",
    "notes",
}

FACTORIZED_DRIVER_KEYS = {
    "horizon",
    "state_space_dim_total",
    "max_variable_dim",
    "max_factor_arity",
    "agents",
    "regimes",
}
LGSSM_DRIVER_KEYS = {"horizon", "joint_state_dim", "agents"}

#: Registry backends: the 10 executor ``_RUNNER_LOADERS`` keys + render-only
#: bnlearn (registry-gated per ``render/framework_registry.py``).
EXPECTED_BACKENDS = frozenset(
    {
        "pymdp",
        "rxinfer",
        "discopy",
        "activeinference_jl",
        "jax",
        "numpyro",
        "pytorch",
        "ngclearn",
        "lean",
        "stan",
        "bnlearn",
    }
)


def _require(name: str) -> Path:
    """Exemplar path asserted to exist: absence FAILS the test."""
    path = EXEMPLARS[name]
    assert path.exists(), f"committed exemplar missing: {path}"
    return path


def _row(receipt: dict[str, Any], framework: str) -> dict[str, Any]:
    rows = [row for row in receipt["per_backend"] if row["framework"] == framework]
    assert len(rows) == 1, f"expected exactly one {framework} row, got {len(rows)}"
    return rows[0]


def _applicable(receipt: dict[str, Any], framework: str) -> bool:
    return _row(receipt, framework)["applicable"]


# ---------------------------------------------------------------------------
# Receipt schema
# ---------------------------------------------------------------------------


def test_receipt_schema_exact_keys_pomdp_gridworld() -> None:
    path = _require("pomdp_gridworld")
    receipt = estimate_model_complexity(path)
    assert set(receipt) == RECEIPT_KEYS
    assert set(receipt["model"]) == MODEL_KEYS
    assert set(receipt["structure"]) == STRUCTURE_KEYS
    assert set(receipt["structure"]["time"]) == TIME_KEYS
    assert receipt["receipt_type"] == RECEIPT_TYPE
    assert receipt["estimator_version"] == ESTIMATOR_VERSION
    assert len(receipt["per_backend"]) == len(BACKEND_ORDER) == len(EXPECTED_BACKENDS)
    for row in receipt["per_backend"]:
        assert set(row) == ROW_KEYS


def test_per_backend_lists_every_registry_backend_in_fixed_order() -> None:
    path = _require("pomdp_gridworld")
    receipt = estimate_model_complexity(path)
    frameworks = [row["framework"] for row in receipt["per_backend"]]
    assert frameworks == list(BACKEND_ORDER)
    assert set(frameworks) == EXPECTED_BACKENDS
    assert {entry.framework for entry in BACKEND_BOUNDS} == EXPECTED_BACKENDS


# ---------------------------------------------------------------------------
# Structure statistics + symbolic-dim resolution (pomdp_gridworld)
# ---------------------------------------------------------------------------


def test_pomdp_structure_stats_exact() -> None:
    path = _require("pomdp_gridworld")
    receipt = estimate_model_complexity(path)
    structure = receipt["structure"]
    assert structure["variable_count"] == 12
    assert structure["edge_count"] == 11
    assert structure["factor_arities"] == [2] * 11
    # A[9,9]=81 + B[9,9,5]=405 + C=9 + D=9 + E=5 + s=9 + s'=9 + o=9 + pi=5
    # + u=1 + t=1 + G[pi]=5 (symbolic dim resolved via the type_checker layer).
    assert structure["total_state_space_dim"] == 548
    assert structure["max_variable_dim"] == 405
    # DataType-faithful: float-typed A/B/C/D/E/G/pi/s params count continuous;
    # int-typed o/u/t count discrete. Kind gating uses detect_model_kinds.
    assert structure["discrete_var_count"] == 3
    assert structure["continuous_var_count"] == 9
    assert structure["time"] == {
        "time_type": "Dynamic",
        "discretization": "",
        "horizon": 15,
        "is_dynamic": True,
    }
    assert receipt["model_kinds"] == ["flat"]


# ---------------------------------------------------------------------------
# Applicability gating per kind
# ---------------------------------------------------------------------------


def test_discrete_kinds_gate_discrete_backends_tmaze() -> None:
    path = _require("tmaze_epistemic")
    receipt = estimate_model_complexity(path)
    assert receipt["model_kinds"] == ["flat"]
    assert _applicable(receipt, "pymdp")
    assert _applicable(receipt, "activeinference_jl")
    assert _applicable(receipt, "jax")
    assert _applicable(receipt, "discopy")
    assert not _applicable(receipt, "rxinfer")

    pymdp = _row(receipt, "pymdp")
    assert pymdp["family"] == "exact-factorized"
    assert pymdp["asymptotic"] == "O(T * prod_f |s_f| * prod_m |o_m| * |a|) [ESTIMATE]"
    assert set(pymdp["drivers"]) == FACTORIZED_DRIVER_KEYS
    assert pymdp["drivers"]["horizon"] == 3
    assert pymdp["drivers"]["agents"] == 1
    assert pymdp["drivers"]["regimes"] == 1

    jax = _row(receipt, "jax")
    assert jax["asymptotic"] == (
        "O(T * sum_f |s_f| * |o| * |a|) [ESTIMATE]"
        " (kronecker-factorized: no joint state materialization)"
    )


def test_continuous_kinds_gate_lgssm_backends() -> None:
    path = _require("continuous_navigation")
    receipt = estimate_model_complexity(path)
    assert receipt["model_kinds"] == ["continuous"]
    assert not _applicable(receipt, "pymdp")
    rxa = _row(receipt, "rxinfer")
    assert rxa["applicable"]
    assert rxa["family"] == "exact-dense-LGSSM"
    assert rxa["asymptotic"] == "O(T * d^3) time, O(d^2) memory [ESTIMATE]"
    assert set(rxa["drivers"]) == LGSSM_DRIVER_KEYS
    assert rxa["drivers"]["horizon"] == 15
    assert rxa["drivers"]["joint_state_dim"] == 2
    jax = _row(receipt, "jax")
    assert jax["applicable"]
    assert jax["family"] == "exact-dense-LGSSM"


def test_hybrid_kind_refuses_discrete_backends() -> None:
    path = _require("hybrid_discrete_continuous")
    receipt = estimate_model_complexity(path)
    assert "hybrid" in receipt["model_kinds"]
    assert "continuous" in receipt["model_kinds"]
    # Discrete machinery is refused for the mixed family composition...
    assert not _applicable(receipt, "pymdp")
    assert not _applicable(receipt, "discopy")
    # ...while the continuous family stays boundable.
    assert _applicable(receipt, "rxinfer")
    assert _row(receipt, "rxinfer")["family"] == "exact-dense-LGSSM"


def test_nonstationary_regime_multiplier() -> None:
    path = _require("regime_switched_dynamics")
    receipt = estimate_model_complexity(path)
    assert receipt["model_kinds"] == ["nonstationary"]
    assert _applicable(receipt, "pymdp")
    drivers = _row(receipt, "pymdp")["drivers"]
    assert drivers["regimes"] == 2  # B_regime[2,3,3,2]: one tensor per regime
    assert drivers["horizon"] == 8


def test_multi_agent_agent_count_driver() -> None:
    path = _require("multi_agent_coordination")
    receipt = estimate_model_complexity(path)
    assert receipt["model_kinds"] == ["multi_agent"]
    assert _applicable(receipt, "pymdp")
    # num_agents: 2 declared (ModelParameters) with A_agent1/A_agent2 keys.
    assert _row(receipt, "pymdp")["drivers"]["agents"] == 2
    assert _row(receipt, "numpyro")["drivers"]["agents"] == 2


# ---------------------------------------------------------------------------
# Horizon honesty: Unbounded / symbolic horizons never fabricate totals
# ---------------------------------------------------------------------------

_SYN_DISCRETE = """# GNN Example: Synthetic Discrete
# GNN Version: 1.0

## GNNSection
GNN

## GNNVersionAndFlags
GNN 1.0

## ModelName
SyntheticDiscrete

## StateSpaceBlock
A[2,2,type=float]
B[2,2,1,type=float]
C[2,type=float]
D[2,type=float]
s[2,1,type=float]

## Connections
D>s
s-A
s-B
B>s

## InitialParameterization
A={
  (0.9, 0.1),
  (0.1, 0.9)
}
D=(0.5, 0.5)

## Footer
GNN: SyntheticDiscrete
"""

_SYN_TIME_BLOCK = """
## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon={horizon}
"""


def _write_synthetic(tmp_path: Path, horizon: str | None) -> Path:
    text = _SYN_DISCRETE
    if horizon is not None:
        text += _SYN_TIME_BLOCK.format(horizon=horizon)
    path = tmp_path / "synthetic_discrete.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_unbounded_horizon_no_numeric_total_bound(tmp_path: Path) -> None:
    path = _write_synthetic(tmp_path, horizon=None)
    receipt = estimate_model_complexity(path)
    assert receipt["structure"]["time"]["horizon"] == UNBOUNDED_HORIZON
    assert receipt["model_kinds"] == ["flat"]
    pymdp = _row(receipt, "pymdp")
    assert pymdp["applicable"]
    assert pymdp["asymptotic"] == (
        "O(prod_f |s_f| * prod_m |o_m| * |a|) per timestep [ESTIMATE]"
        " (horizon Unbounded: no numeric total bound)"
    )
    assert pymdp["drivers"]["horizon"] == UNBOUNDED_HORIZON
    # The LGSSM row degrades to per-step form too (inapplicable here: flat).
    rxa = _row(receipt, "rxinfer")
    assert "per step" in rxa["asymptotic"]
    assert "Unbounded" in rxa["asymptotic"]


def test_symbolic_horizon_kept_verbatim(tmp_path: Path) -> None:
    path = _write_synthetic(tmp_path, horizon="T")
    receipt = estimate_model_complexity(path)
    assert receipt["structure"]["time"]["horizon"] == "T"
    pymdp = _row(receipt, "pymdp")
    assert "per timestep" in pymdp["asymptotic"]
    assert pymdp["asymptotic"].startswith("O(prod_f")  # no "O(T ..." total
    assert pymdp["drivers"]["horizon"] == "T"


# ---------------------------------------------------------------------------
# Driver honesty + registry sanity
# ---------------------------------------------------------------------------


def test_driver_and_note_hygiene_pomdp() -> None:
    path = _require("pomdp_gridworld")
    receipt = estimate_model_complexity(path)
    for row in receipt["per_backend"]:
        if row["family"] == "verification":
            assert row["drivers"] == {}
            assert (
                row["asymptotic"]
                == "class-only: proof cost is not numerically estimated"
            )
        elif row["applicable"]:
            assert row["drivers"], f"{row['framework']} applicable without drivers"
        else:
            assert row["drivers"] == {}
        if row["family"] != "verification":
            assert "[ESTIMATE]" in row["asymptotic"]
        assert row["notes"]
    sampling = _row(receipt, "stan")
    assert sampling["family"] == "sampling"
    assert "per_sample_cost" in sampling["asymptotic"]
    assert "bound reported per-sample" in sampling["asymptotic"]
    assert "runner knob" in sampling["asymptotic"]
    bnlearn = _row(receipt, "bnlearn")
    assert bnlearn["family"] == "structure-learning"
    assert "score-based" in bnlearn["asymptotic"]


# ---------------------------------------------------------------------------
# Input forms, hashing, determinism, serialization
# ---------------------------------------------------------------------------


def test_path_and_object_form_hashing() -> None:
    path = _require("tmaze_epistemic")
    from_path = estimate_model_complexity(path)
    parsed = parse_gnn_file_structured(path)
    assert parsed.model is not None
    from_object = estimate_model_complexity(parsed.model)
    assert from_path["model"]["path"] == str(path)
    assert from_object["model"]["path"] == ""
    assert (
        from_path["model"]["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    canonical = json.dumps(
        parsed.model.to_dict(), sort_keys=True, separators=(",", ":"), default=str
    )
    assert (
        from_object["model"]["source_sha256"]
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    assert from_object["model"]["source_sha256"] != from_path["model"]["source_sha256"]
    assert from_object["model"]["name"] == from_path["model"]["name"]
    assert from_object["structure"] == from_path["structure"]


def test_deterministic_repeat_serialization_and_kind_sort() -> None:
    path = _require("hybrid_discrete_continuous")
    first = estimate_model_complexity(path)
    second = estimate_model_complexity(path)
    assert first == second
    assert to_json_text(first) == to_json_text(second)
    assert first["model_kinds"] == sorted(first["model_kinds"])


def test_to_json_text_roundtrip() -> None:
    path = _require("continuous_navigation")
    receipt = estimate_model_complexity(path)
    text = to_json_text(receipt)
    assert json.loads(text) == receipt
    assert to_json_text(json.loads(text)) == text


def test_invalid_inputs_raise() -> None:
    with pytest.raises(TypeError):
        estimate_model_complexity(123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        estimate_model_complexity(INPUT_DIR / "does_not_exist.md")
