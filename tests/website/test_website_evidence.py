"""Recorded evidence stays inspectable without becoming a frontend proof claim."""

import hashlib
import json
import re
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from gnn.website import WebsiteGenerator
from gnn.website.evidence import PROOF_BOUNDARY, render_evidence_section


def _receipt() -> dict:
    """Presentation fixture for the declared v2 artifact-only contract."""
    return {
        "schema": "gnn-semantic-witness-v2",
        "status": "refused",
        "source_contract": {
            "model": "gaussian",
            "source_sha256": "a" * 64,
            "question": "filtering",
            "dtype": "float64",
            "axes": {"H": ["observation", "state"]},
            "state_ids": ["position", "velocity"],
            "observation_ids": ["sensor"],
            "equations": ["o_t = H s_t + noise"],
        },
        "reference": {
            "algorithm": "joint_gaussian_prefix_conditioning",
            "quantities": {"beliefs": [[1.0, 0.0]]},
        },
        "candidate": {"quantities": {"beliefs": [[1.2, 0.0]]}},
        "comparisons": {
            "identity:initial_time": {
                "status": "refused",
                "reason": "incompatible scientific identity",
                "expected": "first_observation_uses_prior",
                "actual": "extra_predict",
            },
            "beliefs": {
                "status": "refused",
                "numeric_status": "failed",
                "reason": "semantic admission did not pass",
                "max_abs_error": 0.2,
                "max_rel_error": 0.2,
                "atol": 1e-12,
                "rtol": 1e-10,
            },
            "posterior_cov": {
                "status": "unavailable",
                "reason": "required native quantity is missing",
            },
            "parameter:H": {
                "status": "passed",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 0.0,
                "rtol": 0.0,
            },
        },
        "assumptions": {
            "initial_time": "first_observation_uses_prior",
            "conditioning_horizon": [1],
        },
        "artifacts": {
            "source": {"sha256": "a" * 64, "utf8": "<img src=x>"},
            "result": {"sha256": "b" * 64, "utf8": "{}"},
            "program": None,
        },
        "formal_evidence": {"status": "unavailable", "universal_equivalence": False},
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
            "reported_framework": "jax",
            "reported_versions": {"jax_version": "reported-version"},
        },
    }


def _trace_receipt() -> dict:
    """A complete recorded display alignment with one diagnostic mean mismatch."""
    receipt = _receipt()
    receipt["status"] = "failed"
    contract = receipt["source_contract"]
    contract.update(
        model_parameters={"num_timesteps": 2, "random_seed": 0},
        axes={"F": ["next_state", "previous_state"], "H": ["observation", "state"]},
        initial_time="first_observation_uses_prior",
    )
    contract["state_ids"] = ["position </script><x>", "velocity"]
    identities = {
        "dtype": "float64",
        "initial_time": contract["initial_time"],
        "axes": contract["axes"],
        "conditioning_horizon": [1, 2],
        "num_timesteps": 2,
        "num_states": 2,
        "num_observations": 1,
        "random_seed": 0,
        "inference_contract": "gnn_lgssm_filter_v1",
        "inference_mode": "gaussian_filtering",
        "control_mode": "passive",
    }
    receipt["comparisons"].update(
        {"identity:" + key: {"status": "passed"} for key in identities}
    )
    receipt["comparisons"]["beliefs"] = {
        "status": "failed",
        "reason": "outside declared tolerance",
        "max_abs_error": 0.25,
        "atol": 1e-12,
        "rtol": 1e-10,
    }
    receipt["comparisons"]["posterior_cov"] = {"status": "passed"}
    receipt["reference"].update(
        conditioning_horizon=[1, 2],
        quantities={
            "beliefs": [[1.0, 2.0], [3.0, 4.0]],
            "posterior_cov": [[[0.8, 0.1], [0.1, 0.6]], [[0.4, 0.05], [0.05, 0.3]]],
        },
    )
    receipt["candidate"] = {
        **identities,
        "quantities": {
            "beliefs": [[1.25, 2.0], [3.0, 4.0]],
            "posterior_cov": deepcopy(
                receipt["reference"]["quantities"]["posterior_cov"]
            ),
        },
    }
    receipt["assumptions"] = {
        key: contract[key]
        for key in ("question", "initial_time", "state_ids", "observation_ids", "axes")
    }
    receipt["assumptions"]["conditioning_horizon"] = [1, 2]
    return receipt


def test_trace_explorer_retains_bounded_arrays_controls_and_full_table(
    tmp_path: Path,
) -> None:
    receipt = _trace_receipt()
    original = deepcopy(receipt)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [receipt]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "site/analysis.html").read_text()
    assert receipt == original
    for control in (
        "data-trace-quantity",
        "data-trace-state",
        "data-trace-mode",
        "data-trace-status",
    ):
        assert control in html
    assert (
        'viewBox="0 0 640 250" role="img" aria-labelledby="trace-title-0 trace-desc-0"'
        in html
    )
    assert '<title id="trace-title-0">' in html and '<desc id="trace-desc-0">' in html
    assert "Posterior diagonal variance" in html and "Gaussian posterior mean" in html
    assert "No confidence bands or verification claims are inferred." in html
    assert "Unverified recorded arrays and display-only residuals" in html
    assert "position &lt;/script&gt;&lt;x&gt;" in html
    encoded = re.search(
        r'<script type="application/json" data-trace-data>(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert encoded is not None and "</" not in encoded.group(1)
    payload = json.loads(encoded.group(1))
    assert payload["prefixes"] == [1, 2]
    assert payload["series"]["beliefs"]["reference"] == [[1.0, 2.0], [3.0, 4.0]]
    assert payload["series"]["beliefs"]["candidate"] == [[1.25, 2.0], [3.0, 4.0]]
    assert payload["series"]["beliefs"]["residual"] == [[0.25, 0.0], [0.0, 0.0]]
    assert payload["series"]["beliefs"]["status"] == "failed"
    assert payload["series"]["posterior_variance"]["reference"] == [
        [0.8, 0.6],
        [0.4, 0.3],
    ]
    wrapped = render_evidence_section(
        [
            {
                "schema": "fep-semantic-witness-audit-v1",
                "status": "failed",
                "gnn_witness": receipt,
            }
        ]
    )
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        in wrapped
    )


def test_trace_explorer_refuses_unaligned_unavailable_or_malformed_display_data() -> (
    None
):
    for mutation in (
        "identity",
        "horizon",
        "axes",
        "assumptions",
        "boolean",
        "oversize",
        "overflow",
        "unavailable",
        "status",
    ):
        receipt = _trace_receipt()
        receipt["comparisons"]["posterior_cov"]["status"] = "unavailable"
        if mutation == "identity":
            receipt["comparisons"]["identity:dtype"]["status"] = "refused"
        elif mutation == "horizon":
            receipt["candidate"]["conditioning_horizon"] = [2, 2]
        elif mutation == "axes":
            receipt["candidate"]["axes"] = {"H": ["state", "observation"]}
        elif mutation == "assumptions":
            receipt["assumptions"]["initial_time"] = "extra_predict"
        elif mutation == "boolean":
            receipt["candidate"]["quantities"]["beliefs"][0][0] = True
        elif mutation == "oversize":
            receipt["source_contract"]["model_parameters"]["num_timesteps"] = 9
        elif mutation == "overflow":
            receipt["reference"]["quantities"]["beliefs"][0][0] = -1e308
            receipt["candidate"]["quantities"]["beliefs"][0][0] = 1e308
        elif mutation == "unavailable":
            receipt["comparisons"]["beliefs"]["status"] = "unavailable"
        else:
            receipt["status"] = ["passed"]
        html = render_evidence_section([receipt])
        assert (
            '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
            not in html
        ), mutation
        assert '<svg viewBox="0 0 640 250"' not in html, mutation
    assert '<svg viewBox="0 0 640 250"' not in render_evidence_section([_receipt()])


def test_v2_shows_individual_outcomes_assumptions_and_unverified_boundary() -> None:
    receipt = _receipt()
    receipt["artifacts"]["program"] = {
        "sha256": "c" * 64,
        "utf8": "unique retained program: λ <script>source only</script>",
    }
    receipt["comparisons"]["diagnostic"] = {
        "status": "failed",
        "reason": "<script>bad</script>",
    }
    original = deepcopy(receipt)
    html = render_evidence_section([receipt])
    assert receipt == original
    for value in (
        PROOF_BOUNDARY,
        "Reported status: refused",
        "Reported passed",
        "Reported failed",
        "Reported refused",
        "Reported unavailable",
        "Maximum absolute error",
        "Maximum relative error",
        "Absolute tolerance",
        "Relative tolerance",
        "0.2",
        "first_observation_uses_prior",
        "extra_predict",
        "conditioning_horizon",
        "joint_gaussian_prefix_conditioning",
        "artifact_only",
        "reported-version",
        "a" * 64,
    ):
        assert value in html
    assert "&lt;img src=x&gt;" in html and "<img src=x>" not in html
    assert html.count("unique retained program") == 1
    assert html.count("&lt;img src=x&gt;") == 1
    assert '"utf8_bytes": 56' in html
    assert "Complete saved receipt (unverified) below" in html
    assert "&lt;script&gt;source only&lt;/script&gt;" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert '<label for="evidence-query">' in html
    assert '<label for="evidence-status">' in html
    assert 'role="status" aria-live="polite"' in html
    assert "fetch(" not in html

    # Raw JSON remains complete and safely copyable from a text node. Text-node
    # quotes need no attribute escaping, and metadata summarization cannot
    # mutate the caller's receipt or discard malformed optional records.
    pres = []
    parser = HTMLParser()
    active = False

    def start(tag, attrs):
        nonlocal active
        if tag == "pre":
            pres.append("")
            active = True

    def end(tag):
        nonlocal active
        if tag == "pre":
            active = False

    def content(value):
        if active:
            pres[-1] += value

    parser.handle_starttag = start
    parser.handle_endtag = end
    parser.handle_data = content
    parser.feed(html)
    assert json.loads(pres[-1]) == receipt
    receipt["artifacts"]["source"] = {"utf8": {"malformed": True}}
    assert '"malformed": true' in render_evidence_section([receipt])


def test_fep_audit_keeps_replay_references_compilation_execution_and_pair_separate() -> (
    None
):
    audit = {
        "schema": "fep-semantic-witness-audit-v1",
        "status": "failed",
        "gnn_witness": _receipt(),
        "independent_replay": {
            "algorithm": "gaussian_kalman_joseph",
            "status": "failed",
            "comparisons": {
                "posterior_cov": {
                    "status": "failed",
                    "reason": "covariance mismatch",
                    "max_abs_error": 0.3,
                    "atol": 1e-12,
                    "rtol": 1e-10,
                }
            },
        },
        "formal_references": [
            {
                "declaration": "GaussianFilter.condition",
                "alignment": "structural_analogue",
                "source_sha256": "c" * 64,
                "assumptions": ["positive definite covariance"],
                "unmatched_assumptions": ["native label provenance"],
            }
        ],
        "formal_evidence": {
            "status": "referenced_not_compiled",
            "universal_equivalence": False,
            "native_claim_ready": False,
        },
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
        },
        "pair_evidence": {"status": "unavailable", "current_pair_synchronized": False},
        "live_sources": {"fep": {"revision": "current-fep-source"}},
    }
    html = render_evidence_section([audit])
    assert "FEP recorded replay comparisons" in html
    assert "covariance mismatch" in html
    assert "GaussianFilter.condition" in html
    assert "structural_analogue" in html
    assert "unmatched_assumptions" in html
    assert "referenced_not_compiled" in html
    assert "<dt>Lean compilation</dt><dd>unavailable</dd>" in html
    assert "<dt>Execution</dt><dd>artifact_only</dd>" in html
    assert "<dt>Pair custody</dt><dd>unavailable</dd>" in html
    assert "current-fep-source" in html
    assert PROOF_BOUNDARY in html


def test_legacy_and_cross_framework_admission_do_not_hide_quantity_failures() -> None:
    legacy = {
        "schema": "gnn-independent-numerical-witness-v1",
        "source_contract": {"model": "finite"},
        "native": {
            "script_sha256": "b" * 64,
            "script_utf8": "unique historical script <x>",
            "result_utf8": "unique historical result",
        },
        "comparisons": {
            "beliefs": {
                "status": "passed",
                "max_abs_error": 0,
                "rtol": 1e-10,
                "atol": 1e-12,
            }
        },
    }
    comparison = {
        "comparisons": {
            "metric_agreement": {
                "jax_vs_numpy": {
                    "status": "comparable",
                    "model_id": "source-bound-model",
                    "source_sha256": "a" * 64,
                    "inference_mode": "gaussian_filtering",
                    "tolerance": {"atol": 1e-10, "rtol": 1e-8},
                    "views": {
                        "joint": {
                            "within_tolerance": True,
                            "max_absolute_difference": 0,
                            "covariance_within_tolerance": False,
                            "covariance_max_absolute_difference": 0.25,
                        }
                    },
                    "unavailable_metrics": {
                        "expected_free_energy": "missing declared convention"
                    },
                }
            },
            "scientific_exclusions": {"other[0]": ["no source-bound counterpart"]},
        }
    }
    html = render_evidence_section([legacy, comparison])
    assert 'data-evidence-status="comparable"' in html
    assert "joint / covariance</th><td>Reported failed" in html
    assert "0.25" in html
    assert "missing declared convention" in html
    assert "no source-bound counterpart" in html
    assert "Reported status: comparable" in html
    assert "Admitted for comparison" in html
    assert "4 of 4" not in html  # one legacy receipt, one pair, one exclusion
    assert "3 of 3 recorded evidence entries shown." in html
    assert html.count("unique historical script &lt;x&gt;") == 1
    assert html.count("unique historical result") == 1
    assert "b" * 64 in html


def test_native_formal_receipt_distinguishes_positive_compilation_from_rejected_controls() -> (
    None
):
    receipt = {
        "schema": "fep-gnn-semantic-native-v1",
        "status": "passed",
        "source_path": "formal/GNNSemanticWitness.lean",
        "source_sha256": "c" * 64,
        "toolchain": {
            "lean_toolchain": "leanprover/lean4:v4.24.0",
            "mathlib_tag": "v4.24.0",
            "mathlib_revision": "mathlib-source-revision",
        },
        "results": {
            "positive": {
                "compiles": True,
                "has_sorry": False,
                "accepted": True,
                "stdout": "theorem uses axioms: [propext, Classical.choice]",
                "stderr": "",
                "errors": [],
                "warnings": [],
                "source_sha256": "d" * 64,
            },
            "zero-normalizer": {
                "compiles": False,
                "has_sorry": False,
                "accepted": True,
                "stdout": "",
                "stderr": "false normalization claim rejected",
                "errors": ["type mismatch"],
                "warnings": [],
            },
            "omitted-observation-noise": {
                "compiles": False,
                "has_sorry": False,
                "accepted": True,
                "stdout": "",
                "stderr": "false covariance claim rejected",
                "errors": ["unsolved goals"],
                "warnings": [],
            },
        },
        "execution_source_verified": False,
        "universal_equivalence": False,
    }
    html = render_evidence_section([receipt])
    assert "Retained native Lean compilation" in html
    assert (
        "Global formal artifact; no source-model numerical binding is claimed." in html
    )
    assert "Reported positive compilation: compiled" in html
    assert html.count("Reported negative control: false claim rejected") == 2
    assert "Reported failed" not in html
    assert "Recorded numerical comparisons" not in html
    assert "theorem uses axioms: [propext, Classical.choice]" in html
    assert "mathlib-source-revision" in html
    assert (
        "Finite-weight normalization and scalar arbitrary-H prior measurement algebra only"
        in html
    )
    assert (
        "no GNN parser/runner, matrix Gaussian conditioning or universal translation proof."
        in html
    )
    assert PROOF_BOUNDARY in html
    receipt["results"]["zero-normalizer"].update(compiles=True, accepted=False)
    unexpected = render_evidence_section([receipt])
    assert "false claim compiled unexpectedly" in unexpected


def test_pure_dict_site_renders_evidence_without_collectors(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem collector invoked")

    monkeypatch.setattr(WebsiteGenerator, "_collect_all_data", forbidden)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [_receipt()]}, filesystem=False
    )
    assert result["success"], result["errors"]
    assert result["pages_created"] == 7
    html = (tmp_path / "site/analysis.html").read_text()
    assert "Source-bound numerical evidence" in html
    scripts = []
    parser = HTMLParser()
    parser.handle_starttag = lambda tag, attrs: (
        scripts.append(dict(attrs)) if tag == "script" else None
    )
    parser.feed(html)
    assert scripts and all("src" not in script for script in scripts)


def test_bound_collection_excludes_unowned_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import collect_website_data

    source = tmp_path / "input"
    (source / "selected").mkdir(parents=True)
    (source / "other").mkdir()
    source_bytes = b"# Selected source model\r\n"
    other_bytes = b"# Other source model\n"
    (source / "selected/source.md").write_bytes(source_bytes)
    (source / "other/source.md").write_bytes(other_bytes)
    output = tmp_path / "output"
    evidence = output / "16_analysis_output/semantic_witnesses"
    evidence.mkdir(parents=True)
    current = _receipt()
    current["source_contract"]["model"] = "current-evidence"
    current["source_contract"]["source_sha256"] = hashlib.sha256(
        source_bytes
    ).hexdigest()
    current["artifacts"]["source"] = {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "utf8": source_bytes.decode("utf-8"),
    }
    old = _receipt()
    old["source_contract"]["model"] = "stale-evidence"
    (evidence / "stale.json").write_text(json.dumps(old))
    unselected = deepcopy(current)
    unselected["source_contract"]["model"] = "owned-unselected-evidence"
    unselected["source_contract"]["source_sha256"] = hashlib.sha256(
        other_bytes
    ).hexdigest()
    unselected["artifacts"]["source"] = {
        "sha256": hashlib.sha256(other_bytes).hexdigest(),
        "utf8": other_bytes.decode("utf-8"),
    }
    tampered = deepcopy(current)
    tampered["source_contract"]["model"] = "tampered-embedded-source"
    tampered["artifacts"]["source"]["utf8"] = "changed source bytes"
    missing = deepcopy(current)
    missing["source_contract"]["model"] = "missing-embedded-source"
    missing["artifacts"].pop("source")
    invalid = deepcopy(current)
    invalid["source_contract"]["model"] = "invalid-source-identity"
    invalid["source_contract"]["source_sha256"] = "not-a-digest"
    legacy = {
        "schema": "gnn-independent-numerical-witness-v1",
        "source_contract": {
            "model": "current-legacy-evidence",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "comparisons": {},
    }
    legacy_other = deepcopy(legacy)
    legacy_other["source_contract"] = {
        "model": "unselected-legacy-evidence",
        "source_sha256": hashlib.sha256(other_bytes).hexdigest(),
    }
    records = {
        "current.json": current,
        "other.json": unselected,
        "tampered.json": tampered,
        "missing.json": missing,
        "invalid.json": invalid,
        "legacy.json": legacy,
        "legacy-other.json": legacy_other,
        "fep.json": {"schema": "fep-semantic-witness-audit-v1", "gnn_witness": current},
        "fep-other.json": {
            "schema": "fep-semantic-witness-audit-v1",
            "gnn_witness": unselected,
        },
        "generic.json": {"file_name": "generic analysis remains available"},
        "native-proof.json": {
            "schema": "fep-gnn-semantic-native-v1",
            "source_sha256": "c" * 64,
            "status": "passed",
            "results": {},
        },
    }
    for name, record in records.items():
        (evidence / name).write_text(json.dumps(record))
    context = build_run_context(
        source,
        output,
        "evidence-view",
        [16, 20],
        {
            "testing_matrix": {
                "enabled": True,
                "folders": {"selected": [16, 20], "other": [16]},
            }
        },
    )
    assert len(context.selected_models(20)) == 1
    context_file = context.write(tmp_path / "context.json")
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_file))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    summaries = output / "00_pipeline_summary"
    summaries.mkdir()
    (summaries / "current_summary.json").write_text(
        json.dumps(
            {
                "run_id": context.run_id,
                "steps": [
                    {
                        "artifacts": [
                            {"path": f"16_analysis_output/semantic_witnesses/{name}"}
                            for name in records
                        ]
                    }
                ],
            }
        )
    )
    data = collect_website_data(output, source, output / "20_website_output/assets")
    html = render_evidence_section(data["analysis"])
    assert "current-evidence" in html
    assert "current-legacy-evidence" in html
    assert "stale-evidence" not in html
    for excluded in (
        "owned-unselected-evidence",
        "tampered-embedded-source",
        "missing-embedded-source",
        "invalid-source-identity",
        "unselected-legacy-evidence",
    ):
        assert excluded not in html
    assert len(data["analysis"]) == 5
    assert (
        "Global formal artifact; no source-model numerical binding is claimed." in html
    )
    assert any(
        row.get("file_name") == "generic analysis remains available"
        for row in data["analysis"]
    )


def test_malformed_rows_remain_visible_without_page_failure(tmp_path: Path) -> None:
    malformed = {
        "schema": "gnn-semantic-witness-v2",
        "status": True,
        "source_contract": [],
        "comparisons": {"beliefs": ["bad row"]},
        "formal_evidence": "not a mapping",
    }
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path / "site", "analysis": [None, "bad", malformed]},
        filesystem=False,
    )
    assert result["success"], result["errors"]
    html = (tmp_path / "site/analysis.html").read_text()
    assert "Reported status: unavailable" in html
    assert "Unusable comparison record; no outcome established." in html
    assert "Reported passed" not in html
    assert "No recognized numerical evidence receipts" in render_evidence_section(
        [None, {"schema": "unsupported"}]
    )


def _composition_receipt() -> dict:
    """Presentation-only asymmetric components, with no execution claim."""
    child = _trace_receipt()
    components, references, candidates, comparisons = {}, {}, {}, {}
    for agent in ("agent1", "agent2"):
        component = deepcopy(child["source_contract"])
        reference = deepcopy(child["reference"])
        candidate = deepcopy(child["candidate"])
        if agent == "agent2":
            component["state_ids"] = ["temperature"]
            candidate["num_states"] = 1
            for record in (reference, candidate):
                record["quantities"]["beliefs"] = [
                    [row[0]] for row in record["quantities"]["beliefs"]
                ]
                record["quantities"]["posterior_cov"] = [
                    [[matrix[0][0]]] for matrix in record["quantities"]["posterior_cov"]
                ]
        component["schema"] = "gnn-independent-gaussian-composition-v1"
        component["parameters"] = (
            {
                "F": [[0.9, 0.1], [0.0, 0.8]],
                "H": [[1.0, 0.0]],
                "Q": [[0.1, 0.01], [0.01, 0.2]],
                "R": [[0.3]],
                "prior_mean": [0.0, 0.0],
                "prior_cov": [[1.0, 0.1], [0.1, 0.8]],
            }
            if agent == "agent1"
            else {
                "F": [[0.7]],
                "H": [[1.3]],
                "Q": [[0.2]],
                "R": [[0.4]],
                "prior_mean": [1.5],
                "prior_cov": [[0.8]],
            }
        )
        reference["observations"] = [[0.25], [0.5]]
        candidate.update(
            schema_version="gnn_lgssm_filter_v1",
            model_kind="continuous",
            framework="jax",
            parameters=deepcopy(component["parameters"]),
            controls=[[0.0] * len(component["state_ids"]) for _ in range(2)],
            observations_continuous=deepcopy(reference["observations"]),
            validation={"all_valid": True},
        )
        components[agent] = component
        references[agent] = {
            **reference,
            "state_ids": component["state_ids"],
            "observation_ids": component["observation_ids"],
        }
        candidates[agent] = {**candidate, **candidate["quantities"]}
        for key, value in child["comparisons"].items():
            comparisons[agent + ":" + key] = deepcopy(value)
        required = (
            "identity:framework",
            "identity:schema_version",
            "identity:model_kind",
            "identity:validation.all_valid",
            "identity:parameter_keys",
            "parameter:F",
            "parameter:H",
            "parameter:Q",
            "parameter:R",
            "parameter:prior_mean",
            "parameter:prior_cov",
            "controls",
            "observations",
            "projection:beliefs",
            "projection:posterior_cov",
            "projection:controls",
            "projection:observations_continuous",
            "validity:beliefs",
            "validity:posterior_cov",
        )
        comparisons.update(
            {agent + ":" + key: {"status": "passed"} for key in required}
        )
    comparisons.update(
        {
            "identity:" + key: {"status": "passed"}
            for key in (
                "schema_version",
                "model_kind",
                "framework",
                "agents",
                "agent_coupling",
                "num_timesteps",
                "validation.all_valid",
                "agent_results",
                "beliefs_projection_agents",
                "posterior_cov_projection_agents",
                "controls_projection_agents",
                "observations_continuous_projection_agents",
            )
        }
    )
    contract = {
        "schema": "gnn-independent-gaussian-composition-v1",
        "model": "gaussian_product",
        "agents": ["agent1", "agent2"],
        "question": "filtering",
        "dtype": "float64",
        "initial_time": child["source_contract"]["initial_time"],
        "source_sha256": "a" * 64,
        "axes": child["source_contract"]["axes"],
        "components": components,
        "model_parameters": {
            "num_timesteps": 2,
            "nr_agents": 2,
            "agent_coupling": "independent",
            "random_seed_agent1": 0,
            "random_seed_agent2": 0,
        },
        "parameters": {
            f"{key}_{agent}": value
            for agent, component in components.items()
            for key, value in component["parameters"].items()
        },
    }
    for key in ("state_ids", "observation_ids"):
        contract[key] = [
            f"{agent}:{label}"
            for agent, component in components.items()
            for label in component[key]
        ]
    assumptions = {
        key: contract[key]
        for key in (
            "state_ids",
            "observation_ids",
            "axes",
            "initial_time",
            "question",
        )
    }
    assumptions.update(
        agent_coupling="independent",
        control_mode="passive",
        conditioning_horizon=[1, 2],
    )
    return {
        "schema": "gnn-semantic-witness-v3",
        "status": "failed",
        "source_contract": contract,
        "reference": {"conditioning_horizon": [1, 2], "components": references},
        "candidate": {"agents": ["agent1", "agent2"], "components": candidates},
        "comparisons": comparisons,
        "assumptions": assumptions,
        "formal_evidence": {"status": "unavailable", "universal_equivalence": False},
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
        },
    }


def test_composition_explorer_preserves_agents_full_covariance_and_reported_mismatch(
    tmp_path: Path,
) -> None:
    receipt = _composition_receipt()
    original = deepcopy(receipt)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path, "analysis": [receipt]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "analysis.html").read_text()
    assert receipt == original
    assert 'id="trace-title-0-agent1"' in html
    assert 'id="trace-title-0-agent2"' in html
    encoded = re.findall(
        r'<script type="application/json" data-trace-data>(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert len(encoded) == 2
    payloads = [json.loads(value) for value in encoded]
    assert (
        payloads[0]["states"]
        == original["source_contract"]["components"]["agent1"]["state_ids"]
    )
    assert payloads[1]["states"] == ["temperature"]
    assert payloads[0]["series"]["beliefs"]["status"] == "failed"
    assert payloads[0]["covariance_matrices"]["reference"][0][0][1] == 0.1
    assert "including off-diagonal entries" in html
    assert "not a measured native joint covariance or a proof of independence" in html
    assert "position &lt;/script&gt;&lt;x&gt;" in html
    assert all("</" not in value for value in encoded)
    assert PROOF_BOUNDARY in html


@pytest.mark.parametrize(
    "mutation",
    [
        "null_labels",
        "bad_labels",
        "dimension_budget",
        "wrong_agents",
        "root_axis_identity",
        "child_seed",
        "child_parameter",
        "future_prefix",
        "boolean_array",
        "missing_child",
        "shared_noise",
        "refused_status",
    ],
)
def test_composition_explorer_does_not_draw_incompatible_or_malformed_records(
    mutation: str,
) -> None:
    receipt = _composition_receipt()
    contract = receipt["source_contract"]
    if mutation == "null_labels":
        contract["components"]["agent1"]["state_ids"] = None
    elif mutation == "bad_labels":
        contract["components"]["agent1"]["state_ids"] = {"axis": "wrong"}
    elif mutation == "dimension_budget":
        contract["components"]["agent1"]["state_ids"] = ["a", "b", "c", "d"]
    elif mutation == "wrong_agents":
        receipt["candidate"]["agents"] = ["agent2", "agent1"]
    elif mutation == "root_axis_identity":
        receipt["comparisons"]["identity:agents"]["status"] = "refused"
    elif mutation == "child_seed":
        receipt["candidate"]["components"]["agent2"]["random_seed"] = 999
    elif mutation == "child_parameter":
        receipt["comparisons"]["agent1:parameter:H"]["status"] = "refused"
    elif mutation == "future_prefix":
        receipt["reference"]["conditioning_horizon"] = [2, 3]
    elif mutation == "boolean_array":
        receipt["candidate"]["components"]["agent1"]["beliefs"] = [
            [True, False],
            [True, False],
        ]
        receipt["candidate"]["components"]["agent1"]["posterior_cov"] = None
    elif mutation == "missing_child":
        del receipt["candidate"]["components"]["agent2"]
    elif mutation == "shared_noise":
        receipt["assumptions"]["agent_coupling"] = "shared_noise"
    elif mutation == "refused_status":
        receipt["status"] = "refused"
    html = render_evidence_section([receipt])
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        not in html
    )
    assert PROOF_BOUNDARY in html


def test_v3_and_wrapped_v3_collection_bind_original_source_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import _selected_semantic_evidence

    source = tmp_path / "input"
    source.mkdir()
    raw = b"# Selected composition\r\n"
    (source / "product.md").write_bytes(raw)
    context = build_run_context(source, tmp_path / "output", "v3-evidence", [20], {})
    monkeypatch.setattr("gnn.pipeline.run_context.current_run_context", lambda: context)
    receipt = _composition_receipt()
    digest = hashlib.sha256(raw).hexdigest()
    receipt["source_contract"]["source_sha256"] = digest
    receipt["artifacts"] = {"source": {"sha256": digest, "utf8": raw.decode()}}
    for record in (
        receipt,
        {"schema": "fep-semantic-witness-audit-v1", "gnn_witness": receipt},
    ):
        assert _selected_semantic_evidence(record)
        corrupted = deepcopy(record)
        target = corrupted.get("gnn_witness", corrupted)
        target["artifacts"]["source"]["utf8"] += "changed"
        assert not _selected_semantic_evidence(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        "status_list",
        "status_dict",
        "coherent_wrong_axes",
        "root_dtype",
        "root_coupling",
        "root_seed_float",
        "child_model",
        "missing_parameter_row",
        "changed_native_parameter",
        "extra_native_parameter",
        "nonzero_controls",
        "native_failed_validation",
        "missing_observations",
        "extra_refused_identity",
        "boolean_means",
        "boolean_covariance",
        "nan_mean",
        "oversized_integer",
        "missing_source_hash",
        "missing_projection_row",
        "malformed_child_options",
    ],
)
def test_composition_guard_checks_raw_alignment_and_malformed_optional_fields(mutation):
    receipt = _composition_receipt()
    contract = receipt["source_contract"]
    native = receipt["candidate"]["components"]["agent1"]
    if mutation == "status_list":
        receipt["status"] = ["passed"]
    elif mutation == "status_dict":
        receipt["status"] = {"outcome": "passed"}
    elif mutation == "coherent_wrong_axes":
        wrong = {"F": ["previous_state", "next_state"], "H": ["state", "observation"]}
        contract["axes"] = wrong
        receipt["assumptions"]["axes"] = deepcopy(wrong)
        for agent in ("agent1", "agent2"):
            contract["components"][agent]["axes"] = deepcopy(wrong)
            receipt["candidate"]["components"][agent]["axes"] = deepcopy(wrong)
    elif mutation == "root_dtype":
        contract["dtype"] = "float32"
    elif mutation == "root_coupling":
        contract["model_parameters"]["agent_coupling"] = "shared_noise"
    elif mutation == "root_seed_float":
        contract["model_parameters"]["random_seed_agent1"] = 0.0
    elif mutation == "child_model":
        contract["components"]["agent1"]["model"] = "finite"
    elif mutation == "missing_parameter_row":
        del receipt["comparisons"]["agent1:parameter:Q"]
    elif mutation == "changed_native_parameter":
        native["parameters"]["H"][0][0] = 1.5
    elif mutation == "extra_native_parameter":
        native["parameters"]["goal_mean"] = [0.0, 0.0]
    elif mutation == "nonzero_controls":
        native["controls"][0][0] = 0.1
    elif mutation == "native_failed_validation":
        native["validation"]["all_valid"] = False
    elif mutation == "missing_observations":
        del receipt["reference"]["components"]["agent1"]["observations"]
    elif mutation == "extra_refused_identity":
        receipt["comparisons"]["identity:additional_runtime_claim"] = {
            "status": "refused"
        }
    elif mutation == "boolean_means":
        native["beliefs"][0][0] = True
    elif mutation == "boolean_covariance":
        native["posterior_cov"][0][0][0] = True
    elif mutation == "nan_mean":
        native["beliefs"][0][0] = float("nan")
    elif mutation == "oversized_integer":
        native["beliefs"][0][0] = 10**400
    elif mutation == "missing_source_hash":
        contract["source_sha256"] = None
        for component in contract["components"].values():
            component["source_sha256"] = None
    elif mutation == "missing_projection_row":
        del receipt["comparisons"]["identity:beliefs_projection_agents"]
    else:
        contract["components"]["agent1"]["model_parameters"] = []
    html = render_evidence_section([receipt])
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        not in html
    )
    assert PROOF_BOUNDARY in html
    assert "Complete saved receipt (unverified)" in html


def test_failed_recorded_covariance_keeps_full_matrix_fallback_and_display_overflow():
    receipt = _composition_receipt()
    receipt["comparisons"]["agent1:posterior_cov"]["status"] = "failed"
    receipt["reference"]["components"]["agent1"]["quantities"]["posterior_cov"][0][0][
        1
    ] = -1e308
    receipt["candidate"]["components"]["agent1"]["posterior_cov"][0][0][1] = 1e308
    html = render_evidence_section([receipt])
    encoded = re.findall(
        r'<script type="application/json" data-trace-data>(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert len(encoded) == 2
    payload = json.loads(encoded[0])
    assert payload["series"]["posterior_variance"]["status"] == "failed"
    assert payload["covariance_matrices"]["candidate"][0][0][1] == 1e308
    assert "unavailable (display overflow)" in html
    assert "Recorded full covariance; display only" in html
    assert "not a measured native joint covariance" in html


def test_bound_selection_quarantines_unknown_semantic_schemas_and_malformed_wrappers(
    tmp_path, monkeypatch
):
    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import _selected_semantic_evidence

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.md").write_text("# Selected model")
    context = build_run_context(
        source, tmp_path / "output", "schema-quarantine", [20], {}
    )
    monkeypatch.setattr("gnn.pipeline.run_context.current_run_context", lambda: context)
    for schema in (
        "gnn-semantic-witness-v99",
        "gnn-independent-numerical-witness-v99",
        "fep-semantic-witness-audit-v99",
        None,
        [],
        {},
    ):
        assert not _selected_semantic_evidence({"schema": schema})
    for schema in (None, [], {}, "gnn-semantic-witness-v99"):
        assert not _selected_semantic_evidence(
            {
                "schema": "fep-semantic-witness-audit-v1",
                "gnn_witness": {"schema": schema},
            }
        )
    assert _selected_semantic_evidence({"comparisons": {"metric_agreement": {}}})
    assert _selected_semantic_evidence({"schema": "domain-analysis-v1"})
    assert _selected_semantic_evidence({"schema": "fep-gnn-semantic-native-v1"})


def test_lone_json_surrogates_remain_visible_as_escapes_without_failing_site(tmp_path):
    receipt = _composition_receipt()
    receipt["source_contract"]["components"]["agent1"]["state_ids"][0] = "\ud800"
    receipt["notes"] = {"diagnostic": "\udfff"}
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path, "analysis": [receipt]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "analysis.html").read_text()
    assert "\\ud800" in html and "\\udfff" in html
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        not in html
    )
    assert PROOF_BOUNDARY in html


def _fep_product_audit_receipt():
    comparisons = {
        f"{agent}:{target}:{quantity}": {
            "status": "passed",
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "atol": 1e-12,
            "rtol": 1e-10,
            "reason": "",
        }
        for agent in ("agent1", "agent2")
        for target in ("reference", "candidate")
        for quantity in ("beliefs", "posterior_cov")
    }
    comparisons.update(
        {
            "joint_reference:" + quantity: {"status": "passed", "max_abs_error": 0.0}
            for quantity in ("beliefs", "posterior_cov")
        }
    )
    return {
        "schema": "fep-gaussian-product-witness-audit-v1",
        "status": "passed",
        "source_sha256": "a" * 64,
        "input": {"path": "retained-witness.json", "sha256": "b" * 64},
        "input_bookends_equal": True,
        "independent_replay": {
            "algorithm": "separate_agent_kalman_joseph",
            "comparisons": comparisons,
            "components": {
                "agent1": {"beliefs": [[1.25]], "posterior_cov": [[[0.4]]]},
                "agent2": {
                    "beliefs": [[-0.5, 0.2]],
                    "posterior_cov": [[[0.5, 0.1], [0.1, 0.7]]],
                },
                "retained_marker": "unique retained FEP component payload",
            },
        },
        "formal_evidence": {
            "status": "referenced_not_compiled",
            "alignment": "finite_weight_algebra_only",
            "native_claim_ready": False,
            "universal_equivalence": False,
            "source_path": "specs/gnn-semantic-witness-v1/composition_laws.lean",
            "source_sha256": "c" * 64,
            "scope": "Finite nonnegative weight products; not a Gaussian conditioning proof.",
        },
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
        },
        "cross_agent_covariance": "zero_by_declared_product_model; not a measured native quantity",
        "errors": [],
    }


def _finite_product_native_receipt():
    return {
        "schema": "fep-gnn-finite-product-native-v1",
        "status": "passed",
        "source_path": "specs/gnn-semantic-witness-v1/composition_laws.lean",
        "source_sha256": "c" * 64,
        "toolchain": {"lean_toolchain": "leanprover/lean4:v4.35.0"},
        "results": {
            "positive": {
                "compiles": True,
                "accepted": True,
                "has_sorry": False,
                "stdout": "'FEPSemanticComposition.normalized_product' depends on axioms: [propext, Classical.choice]",
                "source_sha256": "c" * 64,
            },
            "zero-product-normalizer": {
                "compiles": False,
                "accepted": True,
                "has_sorry": False,
                "stdout": "error: unsolved goals",
                "source_sha256": "d" * 64,
            },
            "omitted-product-assumption": {
                "compiles": False,
                "accepted": True,
                "has_sorry": False,
                "stdout": "error: unsolved goals",
                "source_sha256": "e" * 64,
            },
        },
        "scope": "An untrusted reported scope cannot replace the fixed profile.",
        "execution_source_verified": False,
        "universal_equivalence": False,
    }


def test_fep_product_audit_displays_all_ten_recorded_comparisons_without_wrapping(
    tmp_path,
):
    receipt = _fep_product_audit_receipt()
    original = deepcopy(receipt)
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path, "analysis": [receipt]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "analysis.html").read_text()
    table = html.split(
        "<caption>FEP recorded independent Gaussian product replay comparisons</caption>",
        1,
    )[1].split("</table>", 1)[0]
    assert table.count("<tr>") == 11
    for name in receipt["independent_replay"]["comparisons"]:
        assert name in table
    assert table.count("Reported passed") == 10
    assert html.count("unique retained FEP component payload") == 1
    assert "FEP recorded independent Gaussian product replay" in html
    assert "separate_agent_kalman_joseph" in html
    assert "referenced_not_compiled" in html and "finite_weight_algebra_only" in html
    assert "not supplied by this audit schema" in html
    assert "Source selection matches the declared digest only" in html
    assert "no retained source bytes" in html
    assert (
        "No measured native joint covariance or Gaussian conditioning proof is inferred"
        in html
    )
    assert "Complete saved FEP product audit (unverified)" in html
    assert "gnn_witness" not in html and "gnn-semantic-witness-v3" not in html
    assert (
        '<div class="details-body evidence-trace-explorer" data-trace-explorer>'
        not in html
    )
    assert receipt == original
    assert PROOF_BOUNDARY in html


def test_fep_product_replay_retains_reported_mismatch_and_refusal():
    receipt = _fep_product_audit_receipt()
    receipt["status"] = "failed"
    row = receipt["independent_replay"]["comparisons"]["agent2:candidate:beliefs"]
    row.update(status="failed", max_abs_error=0.5, reason="independent replay mismatch")
    html = render_evidence_section([receipt])
    assert "Reported audit status: failed" in html
    assert "Reported failed" in html and "independent replay mismatch" in html
    assert "0.5" in html
    receipt["status"] = "refused"
    receipt["errors"] = ["source identity mismatch"]
    assert "source identity mismatch" in render_evidence_section([receipt])


def test_finite_product_native_profile_uses_its_fixed_scope_and_false_controls():
    receipt = _finite_product_native_receipt()
    html = render_evidence_section([receipt])
    assert "Retained finite product algebra compiler controls" in html
    assert "Finite post-evidence product-weight normalization/nonnegativity" in html
    assert (
        "no Gaussian measure/kernel, matrix-conditioning, parser/runner or general composition proof"
        in html
    )
    assert "scalar arbitrary-H" not in html
    assert html.count("Reported negative control: false claim rejected") == 2
    assert "Reported positive compilation: compiled" in html
    assert "Reported failed" not in html
    assert "FEPSemanticComposition.normalized_product" in html
    assert (
        "Global formal artifact; no source-model numerical binding is claimed" in html
    )
    assert "Complete saved native compiler receipt (unverified)" in html
    assert PROOF_BOUNDARY in html
    receipt["results"]["omitted-product-assumption"]["compiles"] = True
    assert "false claim compiled unexpectedly" in render_evidence_section([receipt])


@pytest.mark.parametrize(
    "mutation",
    [
        "audit_status",
        "replay",
        "components",
        "comparisons",
        "formal",
        "execution",
        "native_results",
        "native_compiles",
        "native_hole",
    ],
)
def test_new_fep_optional_fields_remain_unverified_without_page_failure(
    tmp_path, mutation
):
    audit, native = _fep_product_audit_receipt(), _finite_product_native_receipt()
    if mutation == "audit_status":
        audit["status"] = {"outcome": "passed"}
    elif mutation == "replay":
        audit["independent_replay"] = ["malformed"]
    elif mutation == "components":
        audit["independent_replay"]["components"] = None
    elif mutation == "comparisons":
        audit["independent_replay"]["comparisons"] = []
    elif mutation == "formal":
        audit["formal_evidence"] = "malformed"
    elif mutation == "execution":
        audit["execution_evidence"] = []
    elif mutation == "native_results":
        native["results"] = ["malformed"]
    elif mutation == "native_compiles":
        native["results"]["zero-product-normalizer"]["compiles"] = 0
    else:
        native["results"]["zero-product-normalizer"]["has_sorry"] = ["unknown"]
    result = WebsiteGenerator().generate_website(
        {"output_dir": tmp_path, "analysis": [audit, native]}, filesystem=False
    )
    assert result["success"] and result["pages_created"] == 7
    html = (tmp_path / "analysis.html").read_text()
    assert "Complete saved FEP product audit (unverified)" in html
    assert "Complete saved native compiler receipt (unverified)" in html
    assert PROOF_BOUNDARY in html
    if mutation in {"native_compiles", "native_hole"}:
        assert html.count("Reported negative control: false claim rejected") == 1
    if mutation == "audit_status":
        assert "Reported audit status: unavailable" in html


def test_product_audit_selection_uses_only_declared_source_identity_and_known_versions(
    tmp_path, monkeypatch
):
    from gnn.pipeline.run_context import build_run_context
    from gnn.website.collection import _selected_semantic_evidence

    source = tmp_path / "source"
    source.mkdir()
    raw = b"# Selected Gaussian product\r\n"
    (source / "product.md").write_bytes(raw)
    context = build_run_context(source, tmp_path / "output", "product-audit", [20], {})
    monkeypatch.setattr("gnn.pipeline.run_context.current_run_context", lambda: context)
    audit = _fep_product_audit_receipt()
    audit["source_sha256"] = hashlib.sha256(raw).hexdigest()
    assert "artifacts" not in audit and "gnn_witness" not in audit
    assert _selected_semantic_evidence(audit)
    for digest in (
        None,
        True,
        [],
        {},
        "invalid",
        "0" * 64,
        hashlib.sha256(b"# unselected model").hexdigest(),
    ):
        assert not _selected_semantic_evidence({**audit, "source_sha256": digest})
    assert _selected_semantic_evidence(_finite_product_native_receipt())
    for schema in (
        "fep-gaussian-product-witness-audit-v99",
        "fep-gnn-finite-product-native-v99",
        "fep-gnn-semantic-native-v99",
    ):
        assert not _selected_semantic_evidence({**audit, "schema": schema})
        html = render_evidence_section([{**audit, "schema": schema}])
        assert "No recognized numerical evidence receipts" in html
