#!/usr/bin/env python3
"""Tests for gnn.rxinfer_bridge - the daf-jev GraphSpec <-> RxInfer.jl bridge.

Covers: GraphSpec JSON load/validation matrix, JSON round-trip losslessness,
the minimal .gnn markdown subset parser (round-trip + rejection cases), and
the deterministic RxInfer.jl emitter (golden == the committed example file),
and the downstream marginal round-trip (printed block -> gnn.marginals/1
JSON). NO network, NO Julia required to run this file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gnn.rxinfer_bridge import (
    GRAPH_SPEC_FORMAT,
    MARGINALS_FORMAT,
    GraphEdge,
    GraphSpec,
    GraphVariable,
    emit_rxinfer_jl,
    load_graphspec,
    load_graphspec_file,
    parse_gnn_subset,
    parse_marginals,
    render_gnn_subset,
    write_marginals,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "rxinfer"

# Canonical Asia fixture (Lauritzen-Spiegelhalter), contract section 4 values.
asia_spec_dict: dict = {
    "format": GRAPH_SPEC_FORMAT,
    "variables": [
        {
            "key": "asia",
            "description": "Recently visited Asia?",
            "states": ["false", "true"],
        },
        {
            "key": "dysp",
            "description": "Has dyspnoea (shortness of breath)?",
            "states": ["false", "true"],
        },
        {"key": "smoke", "description": "Is a smoker?", "states": ["false", "true"]},
        {"key": "tub", "description": "Has tuberculosis?", "states": ["false", "true"]},
        {"key": "lung", "description": "Has lung cancer?", "states": ["false", "true"]},
        {"key": "bronc", "description": "Has bronchitis?", "states": ["false", "true"]},
        {
            "key": "either",
            "description": "Has tuberculosis or lung cancer?",
            "states": ["false", "true"],
        },
        {
            "key": "xray",
            "description": "Abnormal X-ray result?",
            "states": ["false", "true"],
        },
    ],
    "edges": [
        {"parent": "either", "child": "dysp"},
        {"parent": "bronc", "child": "dysp"},
        {"parent": "either", "child": "xray"},
        {"parent": "asia", "child": "tub"},
        {"parent": "smoke", "child": "lung"},
        {"parent": "smoke", "child": "bronc"},
        {"parent": "lung", "child": "either"},
        {"parent": "tub", "child": "either"},
    ],
    "cpts": {
        "asia": {
            "child": "asia",
            "parents": [],
            "rows": [{"assignment": {}, "probabilities": [0.99, 0.01]}],
        },
        "dysp": {
            "child": "dysp",
            "parents": ["either", "bronc"],
            "rows": [
                {
                    "assignment": {"either": "false", "bronc": "false"},
                    "probabilities": [0.9, 0.1],
                },
                {
                    "assignment": {"either": "false", "bronc": "true"},
                    "probabilities": [0.2, 0.8],
                },
                {
                    "assignment": {"either": "true", "bronc": "false"},
                    "probabilities": [0.3, 0.7],
                },
                {
                    "assignment": {"either": "true", "bronc": "true"},
                    "probabilities": [0.1, 0.9],
                },
            ],
        },
        "smoke": {
            "child": "smoke",
            "parents": [],
            "rows": [{"assignment": {}, "probabilities": [0.5, 0.5]}],
        },
        "tub": {
            "child": "tub",
            "parents": ["asia"],
            "rows": [
                {"assignment": {"asia": "false"}, "probabilities": [0.99, 0.01]},
                {"assignment": {"asia": "true"}, "probabilities": [0.95, 0.05]},
            ],
        },
        "lung": {
            "child": "lung",
            "parents": ["smoke"],
            "rows": [
                {"assignment": {"smoke": "false"}, "probabilities": [0.99, 0.01]},
                {"assignment": {"smoke": "true"}, "probabilities": [0.9, 0.1]},
            ],
        },
        "bronc": {
            "child": "bronc",
            "parents": ["smoke"],
            "rows": [
                {"assignment": {"smoke": "false"}, "probabilities": [0.7, 0.3]},
                {"assignment": {"smoke": "true"}, "probabilities": [0.4, 0.6]},
            ],
        },
        "either": {
            "child": "either",
            "parents": ["lung", "tub"],
            "rows": [
                {
                    "assignment": {"lung": "false", "tub": "false"},
                    "probabilities": [1.0, 0.0],
                },
                {
                    "assignment": {"lung": "false", "tub": "true"},
                    "probabilities": [0.0, 1.0],
                },
                {
                    "assignment": {"lung": "true", "tub": "false"},
                    "probabilities": [0.0, 1.0],
                },
                {
                    "assignment": {"lung": "true", "tub": "true"},
                    "probabilities": [0.0, 1.0],
                },
            ],
        },
        "xray": {
            "child": "xray",
            "parents": ["either"],
            "rows": [
                {"assignment": {"either": "false"}, "probabilities": [0.95, 0.05]},
                {"assignment": {"either": "true"}, "probabilities": [0.02, 0.98]},
            ],
        },
    },
}


def _mini_spec_dict() -> dict:
    return {
        "format": GRAPH_SPEC_FORMAT,
        "variables": [
            {"key": "rain", "description": "Is it raining?", "states": ["no", "yes"]},
            {
                "key": "grass",
                "description": "Is the grass wet?",
                "states": ["dry", "wet"],
            },
        ],
        "edges": [{"parent": "rain", "child": "grass"}],
        "cpts": {
            "rain": {
                "child": "rain",
                "parents": [],
                "rows": [{"assignment": {}, "probabilities": [0.8, 0.2]}],
            },
            "grass": {
                "child": "grass",
                "parents": ["rain"],
                "rows": [
                    {"assignment": {"rain": "no"}, "probabilities": [0.9, 0.1]},
                    {"assignment": {"rain": "yes"}, "probabilities": [0.1, 0.9]},
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. GraphSpec JSON loading + validation matrix
# ---------------------------------------------------------------------------


def test_load_valid_and_topological_order():
    spec = load_graphspec(copy.deepcopy(asia_spec_dict))
    assert spec.topological_order() == (
        "asia",
        "smoke",
        "tub",
        "lung",
        "bronc",
        "either",
        "dysp",
        "xray",
    )
    # Round-trip is lossless (deterministic canonical JSON).
    assert load_graphspec(spec.to_json()).to_json() == spec.to_json()


def test_top_level_format_rejected():
    data = _mini_spec_dict()
    data["format"] = "dafjev.bayesnet/2"
    with pytest.raises(ValueError, match=r"GraphSpec format: expected"):
        load_graphspec(data)


def test_unknown_top_level_key_rejected():
    data = _mini_spec_dict()
    data["what"] = 1
    with pytest.raises(ValueError, match=r"unknown top-level keys"):
        load_graphspec(data)


def test_variable_shape_rejected():
    data = _mini_spec_dict()
    data["variables"][0]["states"] = ["only"]
    with pytest.raises(ValueError, match=r"at least 2"):
        load_graphspec(data)


def test_duplicate_variable_key_rejected():
    data = _mini_spec_dict()
    data["variables"][1]["key"] = "rain"
    with pytest.raises(ValueError, match=r"duplicate key 'rain'"):
        load_graphspec(data)


def test_duplicate_state_labels_rejected():
    data = _mini_spec_dict()
    data["variables"][0]["states"] = ["yes", "yes"]
    with pytest.raises(ValueError, match=r"duplicate state"):
        load_graphspec(data)


def test_unknown_edge_endpoint_rejected():
    data = _mini_spec_dict()
    data["edges"].append({"parent": "ghost", "child": "tub"})
    with pytest.raises(ValueError, match=r"unknown parent 'ghost'"):
        load_graphspec(data)


def test_self_loop_rejected():
    data = _mini_spec_dict()
    data["edges"].append({"parent": "rain", "child": "rain"})
    with pytest.raises(ValueError, match=r"self-loop on 'rain'"):
        load_graphspec(data)


def test_duplicate_edge_rejected():
    data = _mini_spec_dict()
    data["edges"].append({"parent": "rain", "child": "grass"})
    with pytest.raises(ValueError, match=r"duplicate edge 'rain' -> 'grass'"):
        load_graphspec(data)


def test_cycle_rejected():
    data = _mini_spec_dict()
    data["edges"].append({"parent": "grass", "child": "rain"})
    with pytest.raises(ValueError, match=r"cycle detected involving"):
        load_graphspec(data)


def test_missing_cpt_rejected():
    data = _mini_spec_dict()
    del data["cpts"]["grass"]
    with pytest.raises(ValueError, match=r"missing or non-object CPT entry"):
        load_graphspec(data)


def test_cpt_parents_mismatch_rejected():
    data = _mini_spec_dict()
    data["cpts"]["grass"]["parents"] = []
    with pytest.raises(ValueError, match=r"parents must equal the graph parents"):
        load_graphspec(data)


def test_cpt_parents_order_mismatch_rejected():
    data = copy.deepcopy(asia_spec_dict)
    data["cpts"]["dysp"]["parents"] = ["bronc", "either"]
    with pytest.raises(ValueError, match=r"parents must equal the graph parents"):
        load_graphspec(data)


def test_row_count_wrong():
    data = _mini_spec_dict()
    data["cpts"]["grass"]["rows"] = data["cpts"]["grass"]["rows"][:1]
    with pytest.raises(ValueError, match=r"expected 2 rows"):
        load_graphspec(data)


def test_non_canonical_row_order_rejected():
    data = copy.deepcopy(asia_spec_dict)
    data["cpts"]["dysp"]["rows"] = list(reversed(data["cpts"]["dysp"]["rows"]))
    with pytest.raises(ValueError, match=r"canonical order"):
        load_graphspec(data)


def test_unknown_assignment_label_rejected():
    data = _mini_spec_dict()
    data["cpts"]["grass"]["rows"][0]["assignment"] = {"rain": "maybe"}
    with pytest.raises(ValueError, match=r"violates the canonical order"):
        load_graphspec(data)


def test_negative_probability_rejected():
    data = _mini_spec_dict()
    data["cpts"]["grass"]["rows"][0]["probabilities"] = [-0.1, 1.1]
    with pytest.raises(ValueError, match=r"probability 0 is negative"):
        load_graphspec(data)


def test_jev_factors_wrong_shape_rejected():
    data = _mini_spec_dict()
    data["jev_factors"] = ["flat-string"]
    with pytest.raises(
        ValueError, match=r"jev_factors \(RESERVED field\): entries must be"
    ):
        load_graphspec(data)


def test_jev_factors_passthrough():
    data = _mini_spec_dict()
    data["jev_factors"] = [{"kind": "reserved-demo"}]
    spec = load_graphspec(data)
    assert spec.jev_factors == ({"kind": "reserved-demo"},)
    assert spec.to_json()["jev_factors"] == [{"kind": "reserved-demo"}]


def test_load_graphspec_file_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=r"invalid JSON"):
        load_graphspec_file(str(path))


# ---------------------------------------------------------------------------
# 2. .gnn markdown subset parser
# ---------------------------------------------------------------------------


def test_gnn_parse_roundtrip_asia():
    spec = load_graphspec(copy.deepcopy(asia_spec_dict))
    text = render_gnn_subset(spec)
    assert parse_gnn_subset(text) == spec


def test_parse_rejects_unbalanced():
    with pytest.raises(ValueError, match=r"missing '### States'"):
        parse_gnn_subset("## asia\n### States\n\n## tub\n")


def test_parse_rejects_bad_edge():
    text = "\n".join(
        [
            "## rain",
            "### States",
            "[Discrete] no, yes",
            "",
            "## grass",
            "### States",
            "[Discrete] dry, wet",
            "",
            "## Connections",
            "rain ~~ grass",
        ]
    )
    with pytest.raises(ValueError, match=r"unparsable edge"):
        parse_gnn_subset(text)


def test_parse_rejects_missing_states():
    text = "\n".join(
        [
            "## asia",
            "### Description",
            "Visited Asia?",
            "",
            "## Connections",
        ]
    )
    with pytest.raises(ValueError, match=r"missing '### States'"):
        parse_gnn_subset(text)


def test_parse_rejects_bad_cpt_row():
    text = "\n".join(
        [
            "## rain",
            "### States",
            "[Discrete] no, yes",
            "",
            "## grass",
            "### States",
            "[Discrete] dry, wet",
            "",
            "## Connections",
            "rain -> grass",
            "",
            "## InitialParameterization",
            "rain={",
            "() = (0.8, 0.2)",
            "}",
            "grass={",
            "(rain=no) = (0.5)",
            "}",
        ]
    )
    with pytest.raises(ValueError, match=r"expected 2 rows"):
        parse_gnn_subset(text)


def test_parse_rejects_undeclared_edge_endpoint():
    text = "\n".join(
        [
            "## rain",
            "### States",
            "[Discrete] no, yes",
            "",
            "## Connections",
            "rain -> grass",
        ]
    )
    with pytest.raises(ValueError, match=r"undeclared child 'grass'"):
        parse_gnn_subset(text)


def test_parse_rejects_unknown_section():
    text = "\n".join(
        [
            "## GNNSection",
            "actInfPOMDP",
        ]
    )
    with pytest.raises(ValueError, match=r"missing '### States'"):
        parse_gnn_subset(text)


def test_parse_rejects_undeclared_cpt_state():
    text = "\n".join(
        [
            "## a",
            "### States",
            "[Discrete] banana, hi",
            "",
            "## b",
            "### States",
            "[Discrete] banana, hi",
            "",
            "## Connections",
            "a -> b",
            "",
            "## InitialParameterization",
            "a={",
            "() = (0.5, 0.5)",
            "}",
            "b={",
            "(a=banana) = (0.5, 0.5)",
            "(a=watermelon) = (0.5, 0.5)",
            "}",
        ]
    )
    with pytest.raises(ValueError, match=r"unknown state 'watermelon'"):
        parse_gnn_subset(text)


def test_gnn_accepts_comments_and_blank_lines():
    text = "\n".join(
        [
            "# a comment line",
            "",
            "## asia",
            "### Description",
            "Visited Asia?",
            "### States",
            "[Discrete] false, true",
            "",
            "## tub",
            "### Description",
            "Has tuberculosis?",
            "### States",
            "[Discrete] false, true",
            "",
            "## Connections",
            "asia -> tub",
            "",
            "## InitialParameterization",
            "asia={",
            "() = (0.99, 0.01)",
            "}",
            "tub={",
            "(asia=false) = (0.99, 0.01)",
            "(asia=true) = (0.95, 0.05)",
            "}",
        ]
    )
    spec = parse_gnn_subset(text)
    parents_of = {e.child: e.parent for e in spec.edges}
    assert parents_of["tub"] == "asia"
    assert spec.cpts["tub"].rows[0][1] == (0.99, 0.01)


def test_gnn_parse_handwritten_fixture():
    text = "\n".join(
        [
            "## asia",
            "### Description",
            "Visited Asia?",
            "### States",
            "[Discrete] false, true",
            "",
            "## tub",
            "### Description",
            "Has tuberculosis?",
            "### States",
            "[Discrete] false, true",
            "",
            "## Connections",
            "asia -> tub",
            "",
            "## InitialParameterization",
            "asia={",
            "() = (0.99, 0.01)",
            "}",
            "tub={",
            "(asia=false) = (0.99, 0.01)",
            "(asia=true) = (0.95, 0.05)",
            "}",
        ]
    )
    spec = parse_gnn_subset(text)
    assert spec.parents_of("tub") == ("asia",)
    assert spec.cpts["tub"].rows[1][1] == (0.95, 0.05)
    assert load_graphspec(spec.to_json()) == spec


def test_parse_accepts_arrow_syntax_and_variables_container():
    text = "\n".join(
        [
            "## Variables",
            "",
            "### asia",
            "### Description",
            "Visited Asia?",
            "### States",
            "[Discrete] false, true",
            "",
            "### tub",
            "### Description",
            "Has tuberculosis?",
            "### States",
            "[Discrete] false, true",
            "",
            "## Connections",
            "asia -> tub",
            "",
            "## InitialParameterization",
            "asia={",
            "() = (0.99, 0.01)",
            "}",
            "tub={",
            "(asia=false) = (0.99, 0.01)",
            "(asia=true) = (0.95, 0.05)",
            "}",
        ]
    )
    spec = parse_gnn_subset(text)
    assert [v.key for v in spec.variables] == ["asia", "tub"]
    assert spec.parents_of("tub") == ("asia",)


# ---------------------------------------------------------------------------
# 3. Emitter (golden == committed example)
# ---------------------------------------------------------------------------


def test_emit_golden_matches_example_file():
    spec = load_graphspec_file(str(EXAMPLES / "asia_graphspec.json"))
    expected = (EXAMPLES / "asia_model.jl").read_text(encoding="utf-8")
    assert emit_rxinfer_jl(spec, model_name="asia_model") == expected


def test_emit_deterministic():
    spec = load_graphspec(copy.deepcopy(asia_spec_dict))
    a = emit_rxinfer_jl(spec, model_name="asia_model")
    b = emit_rxinfer_jl(spec, model_name="asia_model")
    assert a == b


def test_emit_model_name_validation():
    spec = load_graphspec(copy.deepcopy(asia_spec_dict))
    for bad in ("", "asia model", "1model", "asia-model"):
        with pytest.raises(ValueError, match=r"must be a Julia identifier"):
            emit_rxinfer_jl(spec, model_name=bad)


def test_emit_rejects_hyphen_collision():
    data = _mini_spec_dict()
    data["variables"][0]["key"] = "dry-grass"
    data["variables"][1]["key"] = "dry_grass"
    data["edges"][0] = {"parent": "dry-grass", "child": "dry_grass"}
    rain_cpt = data["cpts"].pop("rain")
    rain_cpt["child"] = "dry-grass"
    data["cpts"]["dry-grass"] = rain_cpt
    grass_cpt = data["cpts"].pop("grass")
    grass_cpt["child"] = "dry_grass"
    grass_cpt["parents"] = ["dry-grass"]
    grass_cpt["rows"] = [
        {"assignment": {"dry-grass": "no"}, "probabilities": [0.9, 0.1]},
        {"assignment": {"dry-grass": "yes"}, "probabilities": [0.1, 0.9]},
    ]
    data["cpts"]["dry_grass"] = grass_cpt
    spec = load_graphspec(data)
    with pytest.raises(ValueError, match=r"collide"):
        emit_rxinfer_jl(spec, model_name="m")


def test_emit_prefixes_reserved_word():
    data = _mini_spec_dict()
    data["variables"][0]["key"] = "function"
    data["edges"][0]["parent"] = "function"
    rain_cpt = data["cpts"].pop("rain")
    rain_cpt["child"] = "function"
    data["cpts"]["function"] = rain_cpt
    data["cpts"]["grass"]["parents"] = ["function"]
    data["cpts"]["grass"]["rows"] = [
        {"assignment": {"function": "no"}, "probabilities": [0.9, 0.1]},
        {"assignment": {"function": "yes"}, "probabilities": [0.1, 0.9]},
    ]
    spec = load_graphspec(data)
    script = emit_rxinfer_jl(spec, model_name="m")
    assert "x_function" in script


def test_emit_jev_factors_comment_when_present():
    data = _mini_spec_dict()
    data["jev_factors"] = [{"kind": "reserved-demo"}]
    spec = load_graphspec(data)
    script = emit_rxinfer_jl(spec, model_name="m")
    assert "jev_factors present in this GraphSpec" in script


def test_emit_jev_factors_comment_when_absent():
    spec = load_graphspec(_mini_spec_dict())
    script = emit_rxinfer_jl(spec, model_name="m")
    assert "(absent in this input)" in script


def test_emit_structure_and_learning_markers():
    spec = load_graphspec(copy.deepcopy(asia_spec_dict))
    script = emit_rxinfer_jl(spec, model_name="asia_model")
    # Header + contract reference + reserved-field note.
    assert script.startswith("#!/usr/bin/env julia")
    assert "Dict{Symbol,Any}(Symbol(:e_, k) => v for (k, v) in evidence)" in script
    assert "dafjev.bayesnet/1" in script
    # Two-parent tensor line + CHILD_PARENTS const.
    assert "dysp ~ DiscreteTransition(either, A_dysp, bronc)" in script
    assert "const CHILD_PARENTS = Dict{Symbol,Vector{Symbol}}(" in script
    # Learning variant markers (Dirichlet placeholders).
    assert "A_tub ~ DirichletCollection(alpha_A_tub)" in script
    assert "p_asia ~ Dirichlet(alpha_p_asia)" in script
    # Evidence keys are state labels resolved against the RAW names.
    assert "evidence = Dict{Symbol,Any}()" in script
    assert 'RESERVED: the optional GraphSpec "jev_factors" field' in script


# ---------------------------------------------------------------------------
# 4. Downstream round-trip (printed marginals -> gnn.marginals/1 JSON)
# ---------------------------------------------------------------------------


# Canned stdout of the emitted script's _run_inference println block, in the
# EXACT printed format — the minimal single-parent smoke case recorded in
# examples/rxinfer/README.md (P(a | b=true) with
# p=[0.6923076923, 0.3076923077], as printed by Julia's
# round(...; digits=6)).
GOLDEN_MARGINAL_STDOUT = "\n".join(
    [
        "Observed evidence:",
        "  b = true",
        "Posteriors (marginal P(key)):",
        "  a: false=0.692308  true=0.307692",
    ]
)


def test_parse_marginals_golden_print_block():
    assert parse_marginals(GOLDEN_MARGINAL_STDOUT) == {
        "a": {"false": 0.692308, "true": 0.307692},
    }


def test_parse_marginals_skips_unknown_lines():
    stdout = "\n".join(
        [
            "# RxInfer.jl Bayes-net inference — generated by gnn.rxinfer_bridge",
            "Observed evidence:",
            "  rain = yes",
            "Posteriors (marginal P(key)):",
            "",
            "  grass: dry=0.0667  wet=0.9333",
            "Learning inference completed (iterations = 25).",
            "Final variational free energy: -1.25",
            "Wrote posteriors sidecar: /tmp/posteriors.json",
        ]
    )
    assert parse_marginals(stdout) == {"grass": {"dry": 0.0667, "wet": 0.9333}}


def test_parse_marginals_malformed_tokens_fail_closed():
    bad_lines = [
        "  a: false=0.69abc  true=0.307692",  # token number not a literal
        "  a: false=-0.5  true=1.5",  # negative (non-marginal) number
        "  a: false  true=0.5",  # pair without '='
        "  a: no  rain=0.5  rain=0.5",  # state label with a two-space run
    ]
    for line in bad_lines:
        with pytest.raises(ValueError, match=r"unparsable marginal token"):
            parse_marginals(line)
    # 1e999 matches the number grammar but overflows float() to inf.
    with pytest.raises(ValueError, match=r"is not finite"):
        parse_marginals("  a: false=1e999  true=0.0")


def test_parse_marginals_duplicate_key_fails_closed():
    text = "  a: false=0.5  true=0.5\n  a: false=0.9  true=0.1"
    with pytest.raises(ValueError, match=r"duplicate marginal key 'a'"):
        parse_marginals(text)


def test_parse_marginals_duplicate_state_fails_closed():
    with pytest.raises(ValueError, match=r"duplicate state 'false'"):
        parse_marginals("  a: false=0.5  false=0.5")


def test_parse_marginals_no_marginal_lines_returns_empty():
    assert parse_marginals("") == {}
    assert parse_marginals("Posteriors (marginal P(key)):\nObserved evidence:\n") == {}


def test_parse_marginals_multi_state_and_exponent():
    text = "\n".join(
        [
            "  c: low=1.0e-7  mid=0.35  high=0.649999",
            "  a: false=0.25  true=0.75",
        ]
    )
    assert parse_marginals(text) == {
        "c": {"low": 1e-7, "mid": 0.35, "high": 0.649999},
        "a": {"false": 0.25, "true": 0.75},
    }


def test_write_marginals_round_trip_preserves_order(tmp_path):
    marginals = parse_marginals(
        "  c: low=1.0e-7  mid=0.35  high=0.649999\n  a: false=0.25  true=0.75"
    )
    out = write_marginals(
        marginals, tmp_path / "marginals.json", source_model="asia_model"
    )
    assert isinstance(out, Path) and out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith('{\n  "format":')  # indent=2, Julia-sidecar style
    assert text.endswith("\n")
    doc = json.loads(text)
    assert doc["format"] == MARGINALS_FORMAT == "dafjev.bayesnet-posteriors/1"
    # Printed topological order is preserved in the JSON key order.
    assert list(doc["marginals"]) == ["c", "a"]
    assert doc["marginals"] == marginals
    assert doc["source_model"] == "asia_model"


def test_write_marginals_omitted_source_model_is_null(tmp_path):
    out = write_marginals({"a": {"x": 1.0}}, tmp_path / "m.json")
    assert json.loads(out.read_text(encoding="utf-8"))["source_model"] is None


def test_write_marginals_validation_matrix(tmp_path):
    bad_cases = [
        ("not a mapping", r"non-empty mapping"),
        ({}, r"non-empty mapping"),
        ({"a": []}, r"state -> probability"),
        ({"a": {}}, r"state -> probability"),
        ({"a": {"x": 0.5, "y": float("nan")}}, r"is not finite"),
        ({"a": {"x": 0.5, "y": float("inf")}}, r"is not finite"),
        ({"a": {"x": 0.5, "y": True}}, r"must map to a number"),
        ({"a": {"x": 0.5, "y": "0.5"}}, r"must map to a number"),
        ({"a": {"x": 0.5, "y": -0.5}}, r"is negative"),
        ({"a": {"x": 0.3}}, r"sum to"),
        ({"bad key!": {"x": 1.0}}, r"invalid marginal key"),
        ({"a": {"": 1.0}}, r"state must be a non-empty string"),
    ]
    for bad, pattern in bad_cases:
        with pytest.raises(ValueError, match=pattern):
            write_marginals(bad, tmp_path / "marginals.json")


def test_write_marginals_row_sum_tolerance(tmp_path):
    # digits=6 rounding budget scales with state count: 38 x 0.025 +
    # 2 x 0.025006 sums to 1.000012 — beyond a fixed 1e-5 cap, inside the
    # 40-state budget of 1e-6 + 40 * 5e-7 = 2.1e-5.
    ok = {
        "m": {
            **{f"s{i}": 0.025 for i in range(38)},
            "x": 0.025006,
            "y": 0.025006,
        }
    }
    out = write_marginals(ok, tmp_path / "ok.json")
    assert json.loads(out.read_text(encoding="utf-8"))["marginals"]["m"] == ok["m"]
    with pytest.raises(ValueError, match=r"sum to 0\.8999999999999999, expected 1\.0"):
        write_marginals({"c": {"low": 0.3, "mid": 0.3, "high": 0.3}}, tmp_path / "x")


def test_write_marginals_source_model_validation(tmp_path):
    bad_source_models: list = ["", 5]
    for bad in bad_source_models:
        with pytest.raises(ValueError, match=r"source_model"):
            write_marginals({"a": {"x": 1.0}}, tmp_path / "m.json", source_model=bad)
