"""Contract tests for the W8-CD complexity benchmark harness + CLI subcommands.

Pure-unit, deterministic, zero-framework: pins the static
``gnn.complexity_estimate/v1`` receipt schema against the live estimator, the
benchmark module's pinned receipt-type constants and public surface, and the
``gnn complexity`` / ``gnn benchmark`` CLI wiring — without running any
pipeline step, subprocess, or backend.

Cross-slice note (edit-only lane): the benchmark module
(``gnn.analysis.complexity.benchmark``), the ``handlers_complexity`` module,
and the parser/dispatch wiring land from parallel slices. The pinned contract
fixes the receipt-type STRINGS and the two public callables
(``run_complexity_benchmark``, ``estimate_path_complexity``) but not the
constant NAMES, so this file discovers schema constants by value; a missing
module or renamed symbol fails loudly at fold instead of passing vacuously.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gnn.analysis.complexity import (
    ESTIMATOR_VERSION,
    benchmark,
    estimate_model_complexity,
    to_json_text,
)
from gnn.analysis.complexity import (
    RECEIPT_TYPE as STATIC_RECEIPT_TYPE,
)
from gnn.cli import COMMAND_HANDLERS
from gnn.cli.parser import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
EXEMPLAR = REPO_ROOT / "input" / "gnn_files" / "discrete" / "tmaze_epistemic.md"

#: Pinned receipt-type strings (cross-lane contract).
MEASUREMENT_RECEIPT_TYPE = "gnn.complexity_benchmark/v1"
CALIBRATION_RECEIPT_TYPE = "gnn.complexity_calibration/v1"

RECEIPT_KEYS = {
    "receipt_type",
    "model",
    "structure",
    "model_kinds",
    "per_backend",
    "estimator_version",
}
MODEL_KEYS = {"name", "source_sha256", "path"}
ROW_KEYS = {
    "framework",
    "applicable",
    "family",
    "asymptotic",
    "complexity_class",
    "drivers",
    "notes",
}

#: Pinned calibration-row schema (``gnn.complexity_calibration/v1`` rows) and
#: the receipt-level environment block; discovered by value because the
#: benchmark module's constant names are not part of the pinned contract.
CALIBRATION_ROW_KEYS = frozenset(
    {
        "model_name",
        "source_sha256",
        "framework",
        "applicable",
        "asymptotic",
        "complexity_class",
        "wall_median_seconds",
        "peak_rss_mb",
        "repeats",
        "calibration_note",
    }
)
ENVIRONMENT_KEYS = frozenset(
    {
        "python_version",
        "platform",
        "accelerator_type",
        "backend_versions",
        "repeats",
        "sandbox_mode",
        "generated_at",
    }
)


def _require_exemplar() -> Path:
    """Exemplar path asserted to exist: absence FAILS the test."""
    assert EXEMPLAR.exists(), f"committed exemplar missing: {EXEMPLAR}"
    return EXEMPLAR


def _receipt_type_constants(module: Any) -> dict[str, str]:
    """Public ALL-CAPS string constants whose value is a ``gnn.complexity`` type."""
    return {
        name: value
        for name, value in vars(module).items()
        if name.isupper()
        and isinstance(value, str)
        and value.startswith("gnn.complexity")
    }


# ---------------------------------------------------------------------------
# Static receipt schema (``gnn.complexity_estimate/v1``)
# ---------------------------------------------------------------------------


def test_static_receipt_schema_on_committed_exemplar() -> None:
    path = _require_exemplar()
    receipt = estimate_model_complexity(path)
    assert set(receipt) == RECEIPT_KEYS
    assert receipt["receipt_type"] == STATIC_RECEIPT_TYPE
    assert receipt["estimator_version"] == ESTIMATOR_VERSION
    assert set(receipt["model"]) == MODEL_KEYS
    assert (
        receipt["model"]["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    rows = receipt["per_backend"]
    assert isinstance(rows, list)
    assert rows, "per_backend must carry one row per registry backend"
    for row in rows:
        assert set(row) == ROW_KEYS


def test_to_json_text_roundtrip_is_stable_on_exemplar() -> None:
    path = _require_exemplar()
    text = to_json_text(estimate_model_complexity(path))
    assert (
        json.loads(text)["model"]["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert to_json_text(json.loads(text)) == text


# ---------------------------------------------------------------------------
# Benchmark module public surface (W8-CD harness slice)
# ---------------------------------------------------------------------------


def test_benchmark_module_exposes_pinned_callables() -> None:
    assert callable(benchmark.run_complexity_benchmark)
    assert callable(benchmark.estimate_path_complexity)


def test_benchmark_receipt_type_constants_pinned_by_value() -> None:
    values = set(_receipt_type_constants(benchmark).values())
    assert MEASUREMENT_RECEIPT_TYPE in values
    assert CALIBRATION_RECEIPT_TYPE in values


def test_estimate_path_complexity_matches_estimator_receipt() -> None:
    path = _require_exemplar()
    expected = estimate_model_complexity(path)
    receipts = benchmark.estimate_path_complexity(path)
    assert isinstance(receipts, list) and len(receipts) == 1
    receipt = receipts[0]
    assert receipt["receipt_type"] == STATIC_RECEIPT_TYPE
    assert receipt["model"]["source_sha256"] == expected["model"]["source_sha256"]
    assert receipt["per_backend"] == expected["per_backend"]


def test_benchmark_module_pins_calibration_schema_by_value() -> None:
    assert benchmark.CALIBRATION_ROW_KEYS == CALIBRATION_ROW_KEYS
    assert benchmark.ENVIRONMENT_KEYS == ENVIRONMENT_KEYS


def _synthetic_receipt(
    sha256: str, name: str, backends: tuple[str, ...]
) -> dict[str, Any]:
    """Minimal static receipt in the observed estimator shape (tmp-free)."""
    return {
        "receipt_type": STATIC_RECEIPT_TYPE,
        "model": {"name": name, "source_sha256": sha256, "path": f"{name}.md"},
        "structure": {},
        "model_kinds": [],
        "per_backend": [
            {
                "framework": framework,
                "applicable": True,
                "family": "exact-factorized",
                "asymptotic": "O(1) [ESTIMATE]",
                "complexity_class": "exact",
                "drivers": {},
                "notes": "",
            }
            for framework in backends
        ],
        "estimator_version": ESTIMATOR_VERSION,
    }


def _measurement_row(
    sha256: str, framework: str, *, execution_time: float | None
) -> dict[str, Any]:
    """Measurement row in the observed ``gnn.complexity_benchmark/v1`` shape."""
    measured = execution_time is not None
    return {
        "model_name": "synthetic",
        "source_sha256": sha256,
        "framework": framework,
        "available": measured,
        "success": measured,
        "applicable": True,
        "execution_time": execution_time,
        "execution_time_mean": None,
        "execution_time_std": None,
        "execution_time_samples": None,
        "child_peak_rss_mb": None,
        "rss_sample_interval_seconds": None,
        "rss_samples_count": None,
        "cancelled": False,
        "return_code": None,
        "error": None,
    }


def test_calibration_join_different_shas_same_framework_independent() -> None:
    receipts = [
        _synthetic_receipt("a" * 64, "ModelA", ("pymdp", "jax")),
        _synthetic_receipt("b" * 64, "ModelB", ("pymdp", "jax")),
    ]
    measured = [
        _measurement_row("a" * 64, "pymdp", execution_time=0.5),
        _measurement_row("b" * 64, "pymdp", execution_time=0.7),
    ]
    rows = benchmark._build_calibration_rows(receipts, measured, repeats=3)
    joined = {(row["source_sha256"], row["framework"]): row for row in rows}
    # Join key is (source_sha256, framework): same framework, different shas
    # join independently.
    assert joined[("a" * 64, "pymdp")]["wall_median_seconds"] == 0.5
    assert joined[("b" * 64, "pymdp")]["wall_median_seconds"] == 0.7


def test_calibration_join_same_sha_two_frameworks_two_rows() -> None:
    receipts = [_synthetic_receipt("c" * 64, "ModelC", ("pymdp", "jax"))]
    measured = [
        _measurement_row("c" * 64, "pymdp", execution_time=0.5),
        _measurement_row("c" * 64, "jax", execution_time=0.6),
    ]
    rows = benchmark._build_calibration_rows(receipts, measured, repeats=3)
    for row in rows:
        assert set(row) == CALIBRATION_ROW_KEYS
        assert row["source_sha256"] == "c" * 64
    by_framework = {row["framework"]: row for row in rows}
    assert by_framework["pymdp"]["wall_median_seconds"] == 0.5
    assert by_framework["jax"]["wall_median_seconds"] == 0.6


def test_calibration_unavailable_backend_note_and_nulls() -> None:
    receipts = [_synthetic_receipt("d" * 64, "ModelD", ("pymdp",))]
    rows = benchmark._build_calibration_rows(receipts, [], repeats=3)
    assert len(rows) == 1
    row = rows[0]
    assert row["calibration_note"] == "backend unavailable; no measurement"
    assert row["wall_median_seconds"] is None
    assert row["peak_rss_mb"] is None
    assert row["repeats"] == 3


def test_environment_block_keys_pinned() -> None:
    block = benchmark._environment_block([], ["pymdp", "jax"], repeats=3)
    assert set(block) == ENVIRONMENT_KEYS
    assert block["repeats"] == 3
    assert set(block["backend_versions"]) == {"pymdp", "jax"}
    assert block["backend_versions"]["pymdp"] is None
    assert block["backend_versions"]["jax"] is None


# ---------------------------------------------------------------------------
# CLI wiring (``gnn complexity`` / ``gnn benchmark``)
# ---------------------------------------------------------------------------


def test_build_parser_has_complexity_and_benchmark_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["complexity", str(EXEMPLAR)])
    assert args.command == "complexity"
    assert args.path == EXEMPLAR

    corpus_dir = EXEMPLAR.parent
    args = parser.parse_args(["benchmark", str(corpus_dir)])
    assert args.command == "benchmark"
    assert args.target_dir == corpus_dir


def test_benchmark_parser_flags_and_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", str(EXEMPLAR.parent)])
    assert args.frameworks == "all"
    assert args.repeats == 3

    args = parser.parse_args(
        [
            "benchmark",
            str(EXEMPLAR.parent),
            "--frameworks",
            "pymdp,rxinfer",
            "--repeats",
            "5",
        ]
    )
    assert args.frameworks == "pymdp,rxinfer"
    assert args.repeats == 5


def test_command_handlers_map_new_commands() -> None:
    assert COMMAND_HANDLERS["complexity"] == "_cmd_complexity"
    assert COMMAND_HANDLERS["benchmark"] == "_cmd_benchmark"


def test_handlers_complexity_module_exposes_callables() -> None:
    from gnn.cli import handlers_complexity

    assert callable(handlers_complexity._cmd_complexity)
    assert callable(handlers_complexity._cmd_benchmark)
