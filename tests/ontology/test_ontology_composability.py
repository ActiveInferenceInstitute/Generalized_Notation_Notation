#!/usr/bin/env python3
"""Composability and contract tests for the refactored ontology module (Step 10).

Pins the real behaviour introduced/exposed by the v1.7.0 refactor:
  - ``ParsedAnnotation`` NamedTuple (still a 3-tuple) and ``parse_annotation``
  - pure helpers ``_build_term_lookup`` / ``_term_matches`` / ``TermClassification``
  - public ``suggest_terms`` (nearest-term ranking, case-insensitive, deterministic)
  - public ``analyze_ontology_content`` (single parse->load->validate entry point)
  - public ``summarise_coverage`` (human-readable coverage line)
  - public ``build_ontology_terms`` (in-memory vocabulary builder)
  - ``act_inf_ontology_terms.json`` dedup invariant (64 unique canonical terms)
  - ``extract_ontology_annotations_mcp`` validating against the real vocabulary

Deterministic, isolated, no network.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VOCAB = SRC / "src" / "gnn" / "ontology" / "act_inf_ontology_terms.json"


# ── ParsedAnnotation / parse_annotation ────────────────────────────────────


class TestParsedAnnotation:
    def test_returns_namedtuple_with_named_fields(self) -> None:
        from gnn.ontology import ParsedAnnotation, parse_annotation

        result = parse_annotation("o=Observation # sensory")
        assert isinstance(result, tuple)  # still a 3-tuple
        assert isinstance(result, ParsedAnnotation)
        assert result.key == "o"
        assert result.value == "Observation"
        assert result.comment == "sensory"

    def test_positional_unpacking_still_works(self) -> None:
        from gnn.ontology import parse_annotation

        key, value, comment = parse_annotation("A=LikelihoodMatrix")
        assert key == "A"
        assert value == "LikelihoodMatrix"
        assert comment is None

    def test_bare_value_has_none_key(self) -> None:
        from gnn.ontology import parse_annotation

        key, value, comment = parse_annotation("HiddenState")
        assert key is None
        assert value == "HiddenState"
        assert comment is None

    def test_comment_only_split(self) -> None:
        from gnn.ontology import parse_annotation

        key, value, comment = parse_annotation("s=HiddenState # prior")
        assert (key, value, comment) == ("s", "HiddenState", "prior")

    def test_str_result_contains_key_and_value(self) -> None:
        # Regression guard for the test_ontology_annotations contract that
        # only checks str(result) recoverability.
        from gnn.ontology import parse_annotation

        result = parse_annotation("s=HiddenState")
        as_text = str(result)
        assert "s" in as_text and "HiddenState" in as_text


# ── suggest_terms ──────────────────────────────────────────────────────────


class TestSuggestTerms:
    def test_ranks_nearest_term_first(self) -> None:
        from gnn.ontology import suggest_terms

        suggestions = suggest_terms(["x=HidenState"])  # typo of HiddenState
        assert suggestions, "expected at least one suggestion for a 1-char typo"
        first = suggestions[0]
        assert first["suggested_term"] == "HiddenState"
        assert first["distance"] == 1

    def test_results_are_deterministic_across_calls(self) -> None:
        from gnn.ontology import suggest_terms

        first = suggest_terms(["A=LiklyhoodMatrix"])
        second = suggest_terms(["A=LiklyhoodMatrix"])
        assert first == second

    def test_explicit_ontology_terms_avoids_file_load(self) -> None:
        from gnn.ontology import suggest_terms

        vocab = {"Foo": {"description": "a foo term", "uri": "obo:X"}}
        # Substring containment both ways should match.
        results = suggest_terms(["x=FooBar"], vocab)
        names = {r["suggested_term"] for r in results}
        assert "Foo" in names
        assert all("description" in r for r in results)

    def test_empty_value_yields_no_suggestions(self) -> None:
        from gnn.ontology import suggest_terms

        assert suggest_terms(["= "], {"Foo": {"description": ""}}) == []

    def test_distance_zero_for_substring_match(self) -> None:
        from gnn.ontology import suggest_terms

        vocab = {"HiddenState": {"description": "latent state"}}
        results = suggest_terms(["x=Hidden"], vocab)
        assert results
        assert results[0]["distance"] == 0
        assert results[0]["suggested_term"] == "HiddenState"


# ── analyze_ontology_content ────────────────────────────────────────────────


class TestAnalyzeOntologyContent:
    def test_returns_three_keys(self) -> None:
        from gnn.ontology import analyze_ontology_content

        result = analyze_ontology_content(
            "## ActInfOntologyAnnotation\ns=HiddenState\n"
        )
        assert set(result) == {"ontology_data", "validation_result", "ontology_terms"}
        assert result["ontology_data"]["annotations"] == ["s=HiddenState"]
        assert result["validation_result"]["valid_annotations"] == ["s=HiddenState"]
        assert isinstance(result["ontology_terms"], dict)
        assert result["ontology_terms"]  # non-empty default vocabulary loaded

    def test_explicit_terms_passed_through(self) -> None:
        from gnn.ontology import analyze_ontology_content

        vocab = {"HiddenState": {"description": "x"}}
        result = analyze_ontology_content("## Ontology\ns=HiddenState\n", vocab)
        assert result["ontology_terms"] is vocab

    def test_consistent_with_process_gnn_ontology(self, tmp_path: Path) -> None:
        # The file-based entry point should produce the same analysis as the
        # content-based one for the same content.
        from gnn.ontology import analyze_ontology_content, process_gnn_ontology

        content = "## ActInfOntologyAnnotation\ns=HiddenState\no=Observation\n"
        f = tmp_path / "model.md"
        f.write_text(content, encoding="utf-8")

        analysis = analyze_ontology_content(content)
        file_result = process_gnn_ontology(str(f))

        assert file_result["success"] is True
        assert file_result["ontology_data"] == analysis["ontology_data"]
        assert (
            file_result["validation_result"]["valid_annotations"]
            == analysis["validation_result"]["valid_annotations"]
        )
        # process_gnn_ontology surfaces the resolved terms it used.
        assert file_result["ontology_terms"] == analysis["ontology_terms"]


# ── summarise_coverage ──────────────────────────────────────────────────────


class TestSummariseCoverage:
    def test_full_coverage_no_suggestions(self) -> None:
        from gnn.ontology import summarise_coverage

        result = {
            "valid_annotations": ["s=HiddenState"],
            "invalid_annotations": [],
            "suggestions": [],
            "coverage_score": 1.0,
        }
        assert summarise_coverage(result) == "1/1 annotations valid (coverage 100.0%)"

    def test_partial_coverage_with_suggestion(self) -> None:
        from gnn.ontology import summarise_coverage

        result = {
            "valid_annotations": ["s=HiddenState"],
            "invalid_annotations": ["x=Nope"],
            "suggestions": [{"annotation": "x=Nope", "suggested_term": "HiddenState"}],
            "coverage_score": 0.5,
        }
        line = summarise_coverage(result)
        assert line == "1/2 annotations valid (coverage 50.0%); 1 suggestion"

    def test_pluralisation_of_suggestion(self) -> None:
        from gnn.ontology import summarise_coverage

        result = {
            "valid_annotations": [],
            "invalid_annotations": ["a", "b"],
            "suggestions": [{}, {}],
            "coverage_score": 0.0,
        }
        assert "2 suggestions" in summarise_coverage(result)

    def test_works_on_real_validate_result(self) -> None:
        from gnn.ontology import analyze_ontology_content, summarise_coverage

        analysis = analyze_ontology_content(
            "## ActInfOntologyAnnotation\ns=HiddenState\nbad=Nope\n"
        )
        line = summarise_coverage(analysis["validation_result"])
        assert "1/2 annotations valid" in line
        assert "50.0%" in line


# ── build_ontology_terms ────────────────────────────────────────────────────


class TestBuildOntologyTerms:
    def test_builds_validate_compatible_shape(self) -> None:
        from gnn.ontology import build_ontology_terms, validate_annotations

        vocab = build_ontology_terms(
            ["Foo", "Bar"], descriptions={"Foo": "a foo"}, uris={"Bar": "obo:X"}
        )
        assert vocab == {
            "Foo": {"description": "a foo"},
            "Bar": {"description": "", "uri": "obo:X"},
        }
        result = validate_annotations(["f=Foo", "z=Missing"], vocab)
        assert result["valid_annotations"] == ["f=Foo"]
        assert result["invalid_annotations"] == ["z=Missing"]
        assert result["coverage_score"] == 0.5
        assert result["matched_terms"]["f"]["description"] == "a foo"

    def test_rejects_empty_name(self) -> None:
        from gnn.ontology import build_ontology_terms

        with pytest.raises(ValueError, match="non-empty strings"):
            build_ontology_terms(["", "Foo"])

    def test_rejects_duplicate_name(self) -> None:
        from gnn.ontology import build_ontology_terms

        with pytest.raises(ValueError, match="duplicate ontology term"):
            build_ontology_terms(["Foo", "Foo"])

    def test_case_insensitive_validation_with_built_vocab(self) -> None:
        from gnn.ontology import build_ontology_terms, validate_annotations

        vocab = build_ontology_terms(["HiddenState"])
        # validate_annotations matches case-insensitively.
        result = validate_annotations(["s=hiddenstate"], vocab)
        assert result["valid_annotations"] == ["s=hiddenstate"]
        assert result["matched_terms"]["s"]["term_name"] == "HiddenState"


# ── Internal helpers (pure, typed) ─────────────────────────────────────────


class TestInternalHelpers:
    def test_term_classification_is_namedtuple(self) -> None:
        from gnn.ontology.processor import (
            TermClassification,
            _build_term_lookup,
            _term_matches,
        )

        vocab = {"HiddenState": {"description": "latent"}}
        tc = _term_matches("s=HiddenState", _build_term_lookup(vocab))
        assert isinstance(tc, tuple)
        assert isinstance(tc, TermClassification)
        assert tc.matched is True
        assert tc.reason == ""
        assert tc.match_info is not None
        assert tc.match_info["name"] == "HiddenState"

    def test_term_classification_incomplete_mapping(self) -> None:
        from gnn.ontology.processor import (
            _build_term_lookup,
            _term_matches,
        )

        tc = _term_matches("=HiddenState", _build_term_lookup({"HiddenState": {}}))
        assert tc.matched is False
        assert tc.reason == "mapping annotations require a key and a value"

    def test_term_classification_unknown_term(self) -> None:
        from gnn.ontology.processor import (
            _build_term_lookup,
            _term_matches,
        )

        tc = _term_matches("x=Nope", _build_term_lookup({"HiddenState": {}}))
        assert tc.matched is False
        assert tc.reason == "ontology term is not defined"

    def test_build_term_lookup_rejects_bad_names(self) -> None:
        from gnn.ontology.processor import _build_term_lookup

        with pytest.raises(ValueError, match="non-empty strings"):
            _build_term_lookup({"": {}})


# ── Vocabulary file dedup invariant ─────────────────────────────────────────


class TestVocabularyDedup:
    def test_no_duplicate_keys_in_shipped_file(self) -> None:
        # json.load silently keeps the last occurrence of a duplicate key;
        # detect top-level duplicates explicitly so the vocabulary file never
        # re-acquires shadows that override canonical URIs. The pairs hook is
        # invoked bottom-up, so the outermost (root) call holds the top-level
        # keys — capture only that final call.
        root_keys: list[str] = []

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            root_keys.clear()
            root_keys.extend(key for key, _value in pairs)
            return dict(pairs)

        json.loads(VOCAB.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)
        seen: dict[str, int] = {}
        for key in root_keys:
            seen[key] = seen.get(key, 0) + 1
        duplicates = {k: v for k, v in seen.items() if v > 1}
        assert not duplicates, f"duplicate top-level keys in vocabulary: {duplicates}"

    def test_canonical_uris_preserved(self) -> None:
        terms = json.loads(VOCAB.read_text(encoding="utf-8"))
        # VariationalFreeEnergy's canonical ACTO URI must not be shadowed by a
        # TEMP duplicate (the pre-dedup regression).
        assert terms["VariationalFreeEnergy"]["uri"] == "obo:ACTO_000012"
        # Time keeps its earlier TEMP URI (the duplicate TEMP_000066 was removed).
        assert terms["Time"]["uri"] == "obo:TEMP_000019"

    def test_load_returns_64_unique_terms(self) -> None:
        from gnn.ontology import load_defined_ontology_terms

        terms = load_defined_ontology_terms()
        assert len(terms) == 64
        # Core Active Inference concepts that must always be present.
        assert {
            "HiddenState",
            "Observation",
            "Action",
            "VariationalFreeEnergy",
        }.issubset(terms)


# ── extract_ontology_annotations_mcp real-vocabulary behaviour ─────────────


class TestMcpExtractionVocabulary:
    def test_validates_real_vocabulary_terms(self) -> None:
        from gnn.ontology.mcp import extract_ontology_annotations_mcp

        content = (
            "## ActInfOntologyAnnotation\n"
            "s=HiddenState\n"
            "o=Observation\n"
            "p=Policy\n"  # a real vocabulary term the old hard-coded set omitted
            "weird=NotAStandardTerm\n"
            "## SomeOtherSection\n"
            "x=HiddenState\n"
        )
        result = extract_ontology_annotations_mcp(content)
        assert result["success"] is True
        assert result["annotations"]["s"] == "HiddenState"
        assert result["annotations"]["p"] == "Policy"
        # Policy is a real term -> validated, not unknown.
        assert "p" not in result["unknown_terms"]
        assert "weird" in result["unknown_terms"]
        # s, o, p validate; x is after the section break so not counted.
        assert result["valid_count"] == 3

    def test_case_insensitive_validation(self) -> None:
        from gnn.ontology.mcp import extract_ontology_annotations_mcp

        content = "## ActInfOntologyAnnotation\ns=hiddenstate\n"
        result = extract_ontology_annotations_mcp(content)
        assert result["success"] is True
        assert result["valid_count"] == 1
        assert "s" in result["validated_mappings"]

    def test_unknown_term_lands_in_unknown(self) -> None:
        from gnn.ontology.mcp import extract_ontology_annotations_mcp

        content = "## ActInfOntologyAnnotation\nq=DefinitelyNotAnOntologyTerm\n"
        result = extract_ontology_annotations_mcp(content)
        assert result["unknown_terms"] == {"q": "DefinitelyNotAnOntologyTerm"}
        assert result["valid_count"] == 0

    def test_no_annotation_section_yields_empty(self) -> None:
        from gnn.ontology.mcp import extract_ontology_annotations_mcp

        result = extract_ontology_annotations_mcp("# no annotations\n")
        assert result["annotations"] == {}

    def test_spaced_heading_is_recognized(self) -> None:
        from gnn.ontology.mcp import extract_ontology_annotations_mcp

        content = "## ActInf Ontology Annotation\ns=HiddenState\n"
        result = extract_ontology_annotations_mcp(content)
        assert result["success"] is True
        assert result["annotations"] == {"s": "HiddenState"}
        assert result["valid_count"] == 1


# ─__init__ public surface ───────────────────────────────────────────────────


class TestPublicSurface:
    def test_new_exports_present(self) -> None:
        import gnn.ontology as ontology

        for name in (
            "ParsedAnnotation",
            "SUGGESTION_MAX_DISTANCE",
            "analyze_ontology_content",
            "suggest_terms",
            "summarise_coverage",
            "build_ontology_terms",
        ):
            assert name in ontology.__all__, f"{name} missing from __all__"
            assert hasattr(ontology, name), f"{name} missing from module"

    def test_version_tracks_package(self) -> None:
        import gnn
        import gnn.ontology as ontology

        # SC-16: subpackage versions re-export the canonical package version.
        assert ontology.__version__ == gnn.__version__

    def test_module_info_version_synced(self) -> None:
        from gnn.ontology import __version__, get_module_info

        assert get_module_info()["version"] == __version__

    def test_suggestion_max_distance_constant(self) -> None:
        from gnn.ontology import SUGGESTION_MAX_DISTANCE

        assert SUGGESTION_MAX_DISTANCE == 3


# ── OntologyTermIndex (batch-friendly prebuilt index) ──────────────────────


class TestOntologyTermIndex:
    def test_construction_and_len(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex({"HiddenState": {"description": "latent"}})
        assert len(index) == 1
        assert index.terms["HiddenState"]["description"] == "latent"

    def test_from_names_applies_build_rules(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(
            ["Foo", "Bar"], descriptions={"Foo": "a foo"}, uris={"Bar": "obo:X"}
        )
        assert len(index) == 2
        with pytest.raises(ValueError, match="case-folded"):
            OntologyTermIndex.from_names(["A", "a"])

    def test_lookup_is_case_insensitive(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(
            ["HiddenState"], descriptions={"HiddenState": "latent"}
        )
        match = index.lookup("hiddenstate")
        assert match is not None
        assert match["name"] == "HiddenState"
        assert match["description"] == "latent"

    def test_lookup_miss_returns_none(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(["HiddenState"])
        assert index.lookup("Nope") is None

    def test_contains_operator(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(["HiddenState"])
        assert "HIDDENSTATE" in index
        assert "Nope" not in index
        assert 42 not in index

    def test_known_terms_sorted_canonical(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(["bTerm", "ATerm", "cTerm"])
        assert index.known_terms() == ["ATerm", "bTerm", "cTerm"]

    def test_validate_matches_validate_annotations_contract(self) -> None:
        from gnn.ontology import OntologyTermIndex, validate_annotations

        vocab = {"HiddenState": {"description": "latent"}}
        index = OntologyTermIndex(vocab)
        annotations = ["s=HiddenState", "x=Nope"]
        assert index.validate(annotations) == validate_annotations(annotations, vocab)

    def test_suggest_delegates(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_names(["HiddenState"])
        results = index.suggest(["x=HidenState"])
        assert results[0]["suggested_term"] == "HiddenState"

    def test_from_file_uses_real_vocabulary(self) -> None:
        from gnn.ontology import OntologyTermIndex

        index = OntologyTermIndex.from_file()
        assert len(index) == 64
        assert "HiddenState" in index


# ── load_defined_ontology_terms search_paths DI ────────────────────────────


class TestLoadSearchPathsDI:
    def test_custom_search_paths_resolve_vocabulary(self, tmp_path: Path) -> None:
        from gnn.ontology import load_defined_ontology_terms

        custom = tmp_path / "vocab.json"
        custom.write_text(
            json.dumps({"CustomTerm": {"description": "custom", "uri": "obo:C1"}}),
            encoding="utf-8",
        )
        terms = load_defined_ontology_terms(search_paths=[custom])
        assert "CustomTerm" in terms
        assert terms["CustomTerm"]["uri"] == "obo:C1"

    def test_explicit_file_leads_over_search_paths(self, tmp_path: Path) -> None:
        from gnn.ontology import load_defined_ontology_terms

        explicit = tmp_path / "explicit.json"
        explicit.write_text(
            json.dumps({"ExplicitTerm": {"description": "explicit"}}),
            encoding="utf-8",
        )
        other = tmp_path / "other.json"
        other.write_text(
            json.dumps({"OtherTerm": {"description": "other"}}), encoding="utf-8"
        )
        terms = load_defined_ontology_terms(explicit, search_paths=[other])
        assert "ExplicitTerm" in terms
        assert "OtherTerm" not in terms

    def test_missing_search_path_warns_and_falls_back_to_defaults(
        self, tmp_path: Path
    ) -> None:
        from gnn.ontology import load_defined_ontology_terms

        terms = load_defined_ontology_terms(search_paths=[tmp_path / "missing.json"])
        # No explicit file => no fail-closed raise; falls back to the
        # built-in default term set.
        assert "HiddenState" in terms


# ── build_ontology_terms case-folded duplicate rejection ───────────────────


class TestBuildCasefoldedDuplicates:
    def test_rejects_case_folded_duplicates(self) -> None:
        from gnn.ontology import build_ontology_terms

        with pytest.raises(ValueError, match="case-folded"):
            build_ontology_terms(["A", "a"])

    def test_stripped_duplicate_rejected(self) -> None:
        from gnn.ontology import build_ontology_terms

        with pytest.raises(ValueError, match="duplicate ontology term"):
            build_ontology_terms(["Foo", "  Foo  "])


# ── list_standard_ontology_terms_mcp derives from real vocabulary ──────────


class TestMcpCanonicalTermsList:
    def test_derived_from_real_vocabulary(self) -> None:
        from gnn.ontology import load_defined_ontology_terms
        from gnn.ontology.mcp import list_standard_ontology_terms_mcp

        result = list_standard_ontology_terms_mcp()
        assert result["success"] is True
        vocabulary = load_defined_ontology_terms()
        assert result["count"] == len(vocabulary) == 64
        assert set(result["terms"]) == set(vocabulary)

    def test_descriptions_are_nonempty_strings(self) -> None:
        from gnn.ontology.mcp import list_standard_ontology_terms_mcp

        result = list_standard_ontology_terms_mcp()
        assert "HiddenState" in result["terms"]
        for name, desc in result["terms"].items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert desc  # non-empty (pins the intelligent_analysis contract)
