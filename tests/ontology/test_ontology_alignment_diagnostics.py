"""Candidate-only alignment diagnostics for unresolved declared ontology labels.

The pinned bridge stays authoritative: unresolved labels never gain a
``term-*`` identifier, candidates are copied verbatim from published release
terms, and the pin/copy bytes are untouched by every helper.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from gnn.ontology.bridge import (
    ALIGNMENT_DIAGNOSTICS_SCHEMA,
    ALIGNMENT_FAMILIES,
    ALIGNMENT_UNCLASSIFIED,
    DECLARED_STATUS_RESOLVED,
    DECLARED_STATUS_UNRESOLVED,
    alignment_tokens,
    classify_declared_label,
    load_ontology_bridge,
    ontology_bridge_path,
    ontology_pin_path,
    pinned_declared_index,
    resolve_declared_label,
    sha256_file,
    suggest_declared_label_alignments,
    summarize_ontology_coverage,
    verify_ontology_pin,
)

PINNED_SHA256 = "b19e3d4c70cb7d47276978f4b1f21777089243f6a8514e86656b9e15f639656a"


@pytest.fixture(scope="module")
def bridge():
    return load_ontology_bridge(str(ontology_bridge_path()))


@pytest.fixture(scope="module")
def report():
    return suggest_declared_label_alignments()


def unresolved_labels(bridge) -> list[str]:
    return sorted(
        label
        for label, entry in bridge["declared"].items()
        if entry["status"] == DECLARED_STATUS_UNRESOLVED
    )


def test_pinned_counts_and_bytes_are_unchanged_by_diagnostics(bridge, report):
    pin_before = ontology_pin_path().read_bytes()
    copy_before = ontology_bridge_path().read_bytes()
    summary = summarize_ontology_coverage()
    assert ontology_pin_path().read_bytes() == pin_before
    assert ontology_bridge_path().read_bytes() == copy_before
    assert sha256_file(ontology_bridge_path()) == PINNED_SHA256
    assert verify_ontology_pin()["verified"] is True
    assert (
        summary["declared_labels"],
        summary["resolved_labels"],
        summary["unresolved_labels"],
        summary["terms"],
    ) == (64, 12, 52, 429)
    assert summary["bridge"] == {"origin": "pinned_copy", "sha256": PINNED_SHA256}
    assert summary["release"] == "v5"
    assert summary["resolved_fraction"] == 0.1875
    assert report["schema"] == ALIGNMENT_DIAGNOSTICS_SCHEMA


def test_every_unresolved_label_stays_unresolved_without_identifier(bridge, report):
    labels = unresolved_labels(bridge)
    assert [row["label"] for row in report["labels"]] == labels
    index, error = pinned_declared_index()
    assert error is None
    for row in report["labels"]:
        assert row["status"] == DECLARED_STATUS_UNRESOLVED
        assert row["term_id"] is None
        resolution = resolve_declared_label(row["label"], index)
        assert resolution.status == DECLARED_STATUS_UNRESOLVED
        assert resolution.identifier is None
    assert all(bridge["declared"][label]["id"] is None for label in labels)


def test_candidates_are_published_terms_only_and_never_obo(bridge, report):
    published = {entry["id"]: label for label, entry in bridge["terms"].items()}
    text = json.dumps(report)
    assert "obo:" not in text and "ACTO_" not in text and "TEMP_" not in text
    assert report["candidate_only"] is True
    for row in report["labels"]:
        assert row["candidates"], row["label"]
        assert len(row["candidates"]) <= report["limit"]
        scores = [(-c["score"], c["ontology_term"]) for c in row["candidates"]]
        assert scores == sorted(scores)
        for candidate in row["candidates"]:
            assert candidate["candidate_only"] is True
            assert candidate["term_id"].startswith("term-")
            assert published[candidate["term_id"]] == candidate["ontology_term"]
            assert candidate["score"] > 0 and candidate["basis"]


def test_family_anchors_exist_in_the_pinned_release(bridge):
    for _family, _rules, anchors in ALIGNMENT_FAMILIES:
        for anchor in anchors:
            assert anchor in bridge["terms"], anchor


def test_unresolved_labels_are_classified_into_semantic_families(report):
    families = {row["label"]: row["family"] for row in report["labels"]}
    assert ALIGNMENT_UNCLASSIFIED not in families.values()
    expected = {
        "LikelihoodMatrixModality1": "likelihood_mapping",
        "ObservationMatrix": "likelihood_mapping",
        "RecognitionMatrix": "likelihood_mapping",
        "TransitionMatrixFactor0": "state_transition",
        "NextHiddenState": "state_transition",
        "ControlInputMatrix": "state_transition",
        "LogPreferenceVector": "preference",
        "Preferences": "preference",
        "PolicyVectorFactor1": "policy_action",
        "ExpectedFreeEnergyPerPolicy": "policy_action",
        "PriorOnAction": "policy_action",
        "Habit": "policy_action",
        "GammaParameter": "precision_parameter",
        "SoftminTemperature": "precision_parameter",
        "PriorOverHiddenStatesFactor1": "prior_posterior_belief",
        "PosteriorBelief": "prior_posterior_belief",
        "HiddenStateFactor0": "state_or_observation",
        "ObservationModality2": "state_or_observation",
        "TimeStep": "temporal_structure",
        "InferenceIterations": "temporal_structure",
        "NumberOfAgents": "population",
        "ControlVariance": "control_constraint",
    }
    assert {label: families[label] for label in expected} == expected
    summary = summarize_ontology_coverage()
    counted = sum(len(rows) for rows in summary["unresolved_by_family"].values())
    assert counted == 52
    assert summary["unresolved_with_candidates"] == 52
    assert summary["unresolved_without_candidates"] == 0


def test_structural_qualifiers_do_not_change_concept_or_candidates(report):
    rows = {row["label"]: row for row in report["labels"]}
    base, indexed = rows["LikelihoodMatrix"], rows["LikelihoodMatrixModality2"]
    assert indexed["structural_qualifiers"] == {"modality": 2}
    assert base["structural_qualifiers"] == {}
    assert indexed["concept_tokens"] == base["concept_tokens"]
    assert indexed["candidates"] == base["candidates"]
    assert rows["ActionFactor1"]["structural_qualifiers"] == {"factor": 1}
    assert rows["HiddenStateFactor0"]["candidates"][0]["ontology_term"] == (
        "Hidden State"
    )
    assert rows["TransitionMatrix"]["candidates"][0]["ontology_term"] == "transition"


def test_tokenizer_handles_camel_case_acronyms_and_punctuation():
    assert alignment_tokens("ExpectedFreeEnergyPerPolicy") == (
        "expected",
        "free",
        "energy",
        "per",
        "policy",
    )
    assert alignment_tokens("Control (states)") == ("control", "state")
    assert alignment_tokens("HiddenStateFactor0") == ("hidden", "state", "factor", "0")
    assert alignment_tokens("EFEValue") == ("efe", "value")
    assert classify_declared_label("Unrelated") == ALIGNMENT_UNCLASSIFIED


def test_report_is_deterministic_and_resolved_labels_get_no_candidates():
    first = suggest_declared_label_alignments()
    assert json.dumps(first, sort_keys=True) == json.dumps(
        suggest_declared_label_alignments(), sort_keys=True
    )
    mixed = suggest_declared_label_alignments(
        labels=["Policy", "PolicyVector", "NotAGnnLabel", "PolicyVector"], limit=2
    )
    rows = {row["label"]: row for row in mixed["labels"]}
    assert list(rows) == ["Policy", "PolicyVector", "NotAGnnLabel"]
    assert rows["Policy"]["status"] == DECLARED_STATUS_RESOLVED
    assert rows["Policy"]["term_id"] == "term-policy"
    assert rows["Policy"]["candidates"] == []
    assert rows["NotAGnnLabel"]["status"] == "not_declared"
    assert rows["NotAGnnLabel"]["candidates"] == []
    assert len(rows["PolicyVector"]["candidates"]) == 2


def test_caller_bridge_without_terms_yields_no_invented_candidates(bridge):
    document = deepcopy(bridge)
    document["terms"] = {"Unrelated Concept": {"id": "term-unrelated-concept"}}
    report = suggest_declared_label_alignments(document)
    assert report["bridge"] == {"origin": "caller", "sha256": None}
    assert all(row["candidates"] == [] for row in report["labels"])
    document["terms"] = {"Likelihood": {"id": "obo:ACTO_000001"}}
    assert all(
        row["candidates"] == []
        for row in suggest_declared_label_alignments(document)["labels"]
    )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_is_refused(limit):
    with pytest.raises(ValueError, match="limit"):
        suggest_declared_label_alignments(limit=limit)
