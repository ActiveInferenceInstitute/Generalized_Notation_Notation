#!/usr/bin/env python3
"""
Ontology processor module for GNN Processing Pipeline.

Provides pure, composable primitives for parsing GNN ``ActInfOntologyAnnotation``
sections and validating them against an Active Inference ontology term set.
The thin orchestrator ``10_ontology.py`` calls ``process_ontology``; downstream
steps (render, LLM, analysis) consume the JSON reports written here.
"""

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, cast

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
)

logger = logging.getLogger("ontology")

# Maximum Levenshtein distance at which a candidate term is offered as a
# suggestion for an unknown annotation value.
SUGGESTION_MAX_DISTANCE = 3


class ParsedAnnotation(NamedTuple):
    """A parsed ``KEY=VALUE`` (or bare ``VALUE``) annotation.

    ``parse_annotation`` returns this 3-tuple so callers can unpack it
    positionally (``key, value, comment = parse_annotation(line)``) while
    typed callers can use attribute access. ``key`` and ``comment`` are
    ``None`` when absent.
    """

    key: Optional[str]
    value: str
    comment: Optional[str]


TermLookup = Dict[str, Dict[str, Any]]
"""Case-folded name -> ``{"name": canonical, "data": term_metadata}`` mapping."""


def _build_term_lookup(ontology_terms: Dict[str, Any]) -> TermLookup:
    """Build a case-insensitive lookup of ontology term names.

    Pure function: no I/O, no logging. Terms are sorted by case-folded name
    so the suggestion scan and the resulting lookup are deterministic.

    Raises ``ValueError`` when a term name is empty or non-string — callers
    catch this and surface it as a validation error.
    """
    lookup: TermLookup = {}
    for term_name in sorted(
        ontology_terms, key=lambda candidate: str(candidate).casefold()
    ):
        if not isinstance(term_name, str) or not term_name.strip():
            raise ValueError("ontology term names must be non-empty strings")
        term_data = ontology_terms[term_name]
        lookup[term_name.casefold()] = {
            "name": term_name,
            "data": (
                term_data
                if isinstance(term_data, dict)
                else {"description": str(term_data)}
            ),
        }
    return lookup


class TermClassification(NamedTuple):
    """Result of classifying one annotation against the term lookup.

    ``match_info`` is the matched ``{"name","data"}`` dict (or ``None`` when
    no term matched); ``reason`` is empty on success or a short failure
    reason string on rejection.
    """

    matched: bool
    match_info: Optional[Dict[str, Any]]
    key: Optional[str]
    value: str
    comment: Optional[str]
    reason: str


def _term_matches(annotation: str, term_lookup: TermLookup) -> TermClassification:
    """Classify one annotation against the term lookup.

    Pure: no side effects. Returns a :class:`TermClassification` so the
    caller can unpack typed fields instead of a positional 6-tuple.
    """
    key, value, comment = parse_annotation(annotation)

    if "=" in annotation and (not key or not value):
        return TermClassification(
            False,
            None,
            key,
            value,
            comment,
            "mapping annotations require a key and a value",
        )

    value_lower = value.casefold() if value else ""
    match = term_lookup.get(value_lower)
    if match is None:
        return TermClassification(
            False, None, key, value, comment, "ontology term is not defined"
        )
    return TermClassification(True, match, key, value, comment, "")


def suggest_terms(
    annotations: List[str],
    ontology_terms: Optional[Dict[str, Any]] = None,
    *,
    max_distance: int = SUGGESTION_MAX_DISTANCE,
) -> List[Dict[str, Any]]:
    """Return nearest-ontology-term suggestions for unknown annotations.

    For each annotation whose value is not a known ontology term, candidate
    terms are scored by case-folded substring overlap and Levenshtein
    distance. Each result is a dict with keys ``annotation``,
    ``suggested_term``, ``description``, and ``distance`` (the edit distance;
    ``0`` indicates a substring match). Pure aside from the lazy load of
    ``ontology_terms`` when the caller omits it.
    """
    if ontology_terms is None:
        ontology_terms = load_defined_ontology_terms()
    lookup = _build_term_lookup(ontology_terms)

    suggestions: List[Dict[str, Any]] = []
    for annotation in annotations:
        _key, value, _comment = parse_annotation(annotation)
        value_lower = value.casefold() if value else ""
        if not value_lower:
            continue
        for term_name_lower, term_info in lookup.items():
            distance: Optional[int] = None
            if value_lower in term_name_lower or term_name_lower in value_lower:
                distance = 0
            else:
                edit = _levenshtein_distance(value_lower, term_name_lower)
                if edit <= max_distance:
                    distance = edit
            if distance is not None:
                suggestions.append(
                    {
                        "annotation": annotation,
                        "suggested_term": term_info["name"],
                        "description": term_info["data"].get("description", ""),
                        "distance": distance,
                    }
                )
    return suggestions


def analyze_ontology_content(
    content: str,
    ontology_terms: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse GNN content and validate its ontology annotations in one call.

    Convenience wrapper used by both ``process_gnn_ontology`` (file path
    input) and ``OntologyProcessor.process_ontology`` (dict/str input) so
    the parse → load → validate pipeline exists in exactly one place.
    Returns the ``{"ontology_data", "validation_result", "ontology_terms"}``
    triple callers historically assembled by hand.
    """
    ontology_data = parse_gnn_ontology_section(content)
    resolved_terms = (
        load_defined_ontology_terms() if ontology_terms is None else ontology_terms
    )
    validation_result = validate_annotations(
        ontology_data.get("annotations", []), resolved_terms
    )
    return {
        "ontology_data": ontology_data,
        "validation_result": validation_result,
        "ontology_terms": resolved_terms,
    }


def summarise_coverage(validation_result: Dict[str, Any]) -> str:
    """Render a :func:`validate_annotations` result as a compact summary line.

    Pure: no I/O, no logging. Intended for report/LLM consumers that want a
    one-line human-readable coverage statement (e.g. ``"3/4 annotations valid
    (coverage 75.0%); 1 suggestion"``) without re-deriving counts from the
    result dict.

    Args:
        validation_result: The dict returned by :func:`validate_annotations`.
    """
    total = len(validation_result.get("valid_annotations", [])) + len(
        validation_result.get("invalid_annotations", [])
    )
    valid = len(validation_result.get("valid_annotations", []))
    coverage = validation_result.get("coverage_score", 0.0)
    suggestions = len(validation_result.get("suggestions", []))
    note = (
        f"; {suggestions} suggestion{'s' if suggestions != 1 else ''}"
        if suggestions
        else ""
    )
    return f"{valid}/{total} annotations valid (coverage {coverage * 100:.1f}%){note}"


def build_ontology_terms(
    terms: List[str],
    *,
    descriptions: Optional[Dict[str, str]] = None,
    uris: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a normalized ontology-terms dictionary from name lists.

    The complement of :func:`_normalise_ontology_terms` (which reads JSON):
    this lets callers assemble a custom vocabulary in memory — for tests,
    narrow validation scopes, or programmatic term sets — without writing a
    JSON file. Each term maps to ``{"description": ..., "uri": ...}``, the
    shape :func:`validate_annotations` and :func:`load_defined_ontology_terms`
    consume. Rejects empty names, exact duplicates, and case-folded
    duplicates (e.g. ``["A", "a"]``) so the built vocabulary upholds the same
    case-insensitive lookup invariant :func:`_build_term_lookup` enforces.

    Args:
        terms: Ordered list of term names.
        descriptions: Optional name -> description mapping.
        uris: Optional name -> URI mapping.

    Raises:
        ValueError: On a non-string/empty name, an exact duplicate, or two
            names that collide when case-folded.
    """
    descriptions = descriptions or {}
    uris = uris or {}
    built: Dict[str, Any] = {}
    folded_names: Dict[str, str] = {}
    for name in terms:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ontology term names must be non-empty strings")
        canonical = name.strip()
        if canonical in built:
            raise ValueError(f"duplicate ontology term: {canonical!r}")
        folded = canonical.casefold()
        previous = folded_names.get(folded)
        if previous is not None:
            raise ValueError(
                f"ontology terms are ambiguous when case-folded: {previous!r} "
                f"and {canonical!r}"
            )
        folded_names[folded] = canonical
        entry: Dict[str, Any] = {"description": descriptions.get(canonical, "")}
        uri = uris.get(canonical)
        if uri is not None:
            entry["uri"] = uri
        built[canonical] = entry
    return built


class OntologyTermIndex:
    """Prebuilt case-insensitive index over an ontology vocabulary.

    Composability/performance convenience for batch callers: constructing the
    index once and reusing it across many annotations or files avoids
    rebuilding the case-folded lookup on every :func:`validate_annotations`
    call. The index is immutable after construction; all methods delegate to
    the module-level pure functions so behaviour stays in lock-step with the
    functional API.

    Example:
        >>> index = OntologyTermIndex.from_names(["HiddenState", "Observation"])
        >>> index.lookup("hiddenstate")["name"]
        'HiddenState'
        >>> result = index.validate(["s=HiddenState", "x=Nope"])
        >>> result["valid_annotations"]
        ['s=HiddenState']
    """

    def __init__(self, ontology_terms: Dict[str, Any]) -> None:
        """Build the index from a term-name -> metadata dictionary."""
        self.terms: Dict[str, Any] = dict(ontology_terms)
        self._lookup: TermLookup = _build_term_lookup(ontology_terms)

    @classmethod
    def from_file(cls, ontology_terms_file: Path | None = None) -> "OntologyTermIndex":
        """Build the index from a vocabulary file (default: bundled JSON)."""
        return cls(load_defined_ontology_terms(ontology_terms_file))

    @classmethod
    def from_names(
        cls,
        names: List[str],
        *,
        descriptions: Optional[Dict[str, str]] = None,
        uris: Optional[Dict[str, str]] = None,
    ) -> "OntologyTermIndex":
        """Build the index from an in-memory name list (see
        :func:`build_ontology_terms` for the rejection rules)."""
        return cls(build_ontology_terms(names, descriptions=descriptions, uris=uris))

    def lookup(self, value: str) -> Optional[Dict[str, Any]]:
        """Return the canonical entry for ``value`` (case-insensitive).

        Returns ``{"name": canonical_name, **term_metadata}`` or ``None``
        when the value is not a known term.
        """
        match = self._lookup.get(value.casefold())
        if match is None:
            return None
        return {"name": match["name"], **dict(match["data"])}

    def known_terms(self) -> List[str]:
        """Canonical term names, sorted case-insensitively."""
        return [info["name"] for info in self._lookup.values()]

    def validate(self, annotations: List[str]) -> Dict[str, Any]:
        """Validate annotations against this vocabulary.

        Convenience equivalent of
        ``validate_annotations(annotations, self.terms)`` — the output
        contract is identical. The prebuilt lookup genuinely pays off in
        :meth:`lookup` / :meth:`known_terms` / ``in`` checks (O(1)
        case-insensitive membership); ``validate``/``suggest`` delegate to
        the module functions, which rebuild their own lookup per call.
        """
        return validate_annotations(annotations, self.terms)

    def suggest(
        self,
        annotations: List[str],
        *,
        max_distance: int = SUGGESTION_MAX_DISTANCE,
    ) -> List[Dict[str, Any]]:
        """Suggest nearest terms for unknown annotations (see
        :func:`suggest_terms`)."""
        return suggest_terms(annotations, self.terms, max_distance=max_distance)

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and value.casefold() in self._lookup


def process_ontology(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process ontology for GNN files.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    local_logger = logging.getLogger("ontology")
    strict_validation = kwargs.get("strict_validation", False)
    recursive = kwargs.get("recursive", True)
    ontology_terms_file = kwargs.get("ontology_terms_file")

    try:
        log_step_start(local_logger, "Processing ontology")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "processed_files": 0,
            "reports": [],
            "success": True,
            "errors": [],
            "configuration": {
                "strict_validation": strict_validation,
                "recursive": recursive,
                "ontology_terms_file": str(ontology_terms_file)
                if ontology_terms_file is not None
                else None,
            },
        }

        if not isinstance(strict_validation, bool) or not isinstance(recursive, bool):
            results["success"] = False
            results["errors"].append(
                {
                    "error": "strict_validation and recursive must be booleans",
                    "error_type": "invalid_configuration",
                }
            )

        ontology_terms: dict[str, Any] | None = None
        if results["success"]:
            try:
                ontology_terms = load_defined_ontology_terms(
                    Path(ontology_terms_file)
                    if ontology_terms_file is not None
                    else None
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                results["success"] = False
                results["errors"].append(
                    {
                        "file": str(ontology_terms_file),
                        "error": str(exc),
                        "error_type": "ontology_terms_load_error",
                    }
                )

        # Process each .md file and generate a per-file ontology report.
        discovery = (
            Path(target_dir).rglob("*.md")
            if recursive is True
            else Path(target_dir).glob("*.md")
        )
        gnn_files = sorted(discovery) if results["success"] else []
        results["processed_files"] = len(gnn_files)
        for gnn_file in gnn_files:
            relative_file = gnn_file.relative_to(Path(target_dir))
            report_dir = results_dir / relative_file.parent
            file_report = generate_ontology_report_for_file(
                Path(gnn_file), report_dir, ontology_terms=ontology_terms
            )
            if not file_report.get("success", False):
                results["success"] = False
                results["errors"].append(
                    {
                        "file": str(gnn_file),
                        "error": file_report.get("error", "unknown"),
                    }
                )
            else:
                # Preserve the established report-file API (the path form
                # follows the caller's output_dir) while nested output
                # directories prevent same-stem models from overwriting.
                results["reports"].append(file_report["report_file"])
                invalid = file_report["report"]["validation_result"].get(
                    "invalid_annotations", []
                )
                if strict_validation and invalid:
                    results["success"] = False
                    results["errors"].append(
                        {
                            "file": str(gnn_file),
                            "error": "strict ontology validation failed",
                            "invalid_annotations": invalid,
                        }
                    )
        # Save aggregate results
        results_file = results_dir / "ontology_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if results["success"]:
            log_step_success(local_logger, "Ontology processing completed successfully")
        else:
            log_step_error(local_logger, "Ontology processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(local_logger, f"Ontology processing failed: {e}")
        return False


def parse_gnn_ontology_section(content: str) -> Dict[str, Any]:
    """
    Parse GNN ontology section from content.

    Args:
        content: GNN file content

    Returns:
        Dictionary with parsed ontology information
    """
    try:
        if not content.strip():
            return {}

        # Basic ontology parsing
        ontology_data: dict[str, Any] = {
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": [],
        }

        lines = content.splitlines()
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Markdown section names are case-insensitive for GNN processing,
            # but only an exact level-two ontology heading opens this block.
            if line.startswith("##"):
                heading = line[2:].strip().casefold()
                current_section = (
                    "ontology"
                    if heading in {"ontology", "actinfontologyannotation"}
                    else None
                )
                continue

            if current_section == "ontology":
                if line.startswith("#") or line.startswith("```"):
                    continue
                line = re.sub(r"^[-*+]\s+", "", line)
                # Recognized labels own their complete value, including '='.
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key.lower() in ["concept", "concepts"]:
                        ontology_data["concepts"].append(value)
                        continue
                    elif key.lower() in ["relation", "relations"]:
                        ontology_data["relations"].append(value)
                        continue
                    elif key.lower() in ["property", "properties"]:
                        ontology_data["properties"].append(value)
                        continue
                    elif key.lower() in ["annotation", "annotations"]:
                        ontology_data["annotations"].append(value)
                        continue

                if "=" in line:
                    # Handle unlabeled A=LikelihoodMatrix style annotations.
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    ontology_data["annotations"].append(f"{key}={value}")

        return ontology_data

    except Exception as e:
        return {
            "error": str(e),
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": [],
        }


def process_gnn_ontology(
    gnn_file: str, ontology_terms: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Process ontology annotations for a single GNN file path.

    Reads the file, parses its ``## ActInfOntologyAnnotation`` section, and
    validates the annotations against the supplied (or default) ontology
    term set. Delegates the parse→load→validate pipeline to
    ``analyze_ontology_content`` so that path exists in exactly one place.

    Args:
        gnn_file: Path to the GNN file.
        ontology_terms: Optional pre-resolved term dictionary (loaded by
            default).

    Returns:
        ``{"success", "file_path", "ontology_data", "validation_result",
        "ontology_terms"}`` on success, or ``{"success": False, "error"}``
        on a read/parse failure.
    """
    try:
        file_path = Path(gnn_file)

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {gnn_file}"}

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        analysis = analyze_ontology_content(content, ontology_terms)
        return {
            "success": True,
            "file_path": str(file_path),
            "ontology_data": analysis["ontology_data"],
            "validation_result": analysis["validation_result"],
            "ontology_terms": analysis["ontology_terms"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _normalise_ontology_terms(data: Any) -> Dict[str, Any]:
    """Normalize supported ontology JSON shapes into a term-keyed dictionary."""
    if isinstance(data, dict) and "terms" in data:
        data = data["terms"]

    terms: dict[str, Any] = {}
    casefold_names: dict[str, str] = {}

    def add_term(name: Any, metadata: Any, *, category: str | None = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ontology term names must be non-empty strings")
        normalized_name = name.strip()
        folded_name = normalized_name.casefold()
        previous = casefold_names.get(folded_name)
        if previous is not None and previous != normalized_name:
            raise ValueError(
                f"ontology terms are ambiguous when case-folded: {previous!r} and "
                f"{normalized_name!r}"
            )

        if isinstance(metadata, dict):
            normalized_metadata = dict(metadata)
        elif isinstance(metadata, str):
            normalized_metadata = {"description": metadata}
        elif metadata is None:
            normalized_metadata = {"description": ""}
        else:
            raise ValueError(
                f"ontology metadata for {normalized_name!r} must be an object or string"
            )
        if category is not None:
            normalized_metadata.setdefault("category", category)
        casefold_names[folded_name] = normalized_name
        terms[normalized_name] = normalized_metadata

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                add_term(entry, None)
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("term") or entry.get("id")
                metadata = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"name", "term", "id"}
                }
                add_term(name, metadata)
            else:
                raise ValueError(
                    "ontology term lists may contain only strings or objects"
                )
    elif isinstance(data, dict):
        category_format = any(isinstance(value, list) for value in data.values())
        if category_format:
            if not all(isinstance(value, list) for value in data.values()):
                raise ValueError("ontology category mappings must contain lists only")
            for category, entries in data.items():
                for entry in entries:
                    if isinstance(entry, str):
                        add_term(entry, None, category=str(category))
                    elif isinstance(entry, dict):
                        name = entry.get("name") or entry.get("term") or entry.get("id")
                        metadata = {
                            key: value
                            for key, value in entry.items()
                            if key not in {"name", "term", "id"}
                        }
                        add_term(name, metadata, category=str(category))
                    else:
                        raise ValueError(
                            "ontology category lists may contain only strings or objects"
                        )
        else:
            for name, metadata in data.items():
                add_term(name, metadata)
    else:
        raise ValueError("ontology JSON must be an object or list")

    if not terms:
        raise ValueError("ontology contains no terms")
    return {
        name: terms[name]
        for name in sorted(terms, key=lambda term_name: term_name.casefold())
    }


def load_defined_ontology_terms(
    ontology_terms_file: Path | None = None,
    *,
    search_paths: Sequence[Path] | None = None,
) -> Dict[str, Any]:
    """
    Load defined ontology terms from the Active Inference ontology terms file.

    Dependency-injection hook for the file lookup: pass ``search_paths`` to
    control where the vocabulary is resolved from (tests, alternate installs)
    instead of relying on the hard-coded module-relative paths. Precedence:
    an explicit ``ontology_terms_file`` is authoritative and fails closed
    (``FileNotFoundError``) when missing; ``search_paths`` (when given) are
    tried next, warning-and-continuing on each miss; without either, the
    built-in module-relative paths are used and a total miss falls back to
    the default built-in term set.

    Args:
        ontology_terms_file: Explicit vocabulary file; authoritative.
        search_paths: Optional caller-supplied candidate paths, tried in
            order after ``ontology_terms_file``.

    Returns:
        Dictionary mapping term names to their definitions (including description and URI)
    """
    explicit_file = (
        Path(ontology_terms_file) if ontology_terms_file is not None else None
    )
    if search_paths is not None:
        candidates: list[Path] = list(search_paths)
        if explicit_file is not None:
            candidates.insert(0, explicit_file)
    else:
        # Priority order for ontology files. An explicit file is authoritative
        # and fails closed instead of silently falling back to the built-in
        # vocabulary.
        candidates = (
            [explicit_file]
            if explicit_file is not None
            else [
                Path(__file__).parent / "act_inf_ontology_terms.json",
                Path("src/gnn/ontology/act_inf_ontology_terms.json"),
            ]
        )

    for ontology_file in candidates:
        if not ontology_file.exists():
            if explicit_file is not None:
                raise FileNotFoundError(
                    f"Ontology terms file not found: {ontology_file}"
                )
            continue
        try:
            with open(ontology_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            terms = _normalise_ontology_terms(data)
            logger.info(
                "Loaded %d Active Inference ontology terms from %s",
                len(terms),
                ontology_file,
            )
            return terms
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if explicit_file is not None:
                raise ValueError(
                    f"Failed to load ontology from {ontology_file}: {exc}"
                ) from exc
            logger.warning("Failed to load ontology from %s: %s", ontology_file, exc)
            continue

    # Return default Active Inference terms if no file found
    logger.warning("No ontology terms file found, using defaults")
    return {
        "HiddenState": {
            "description": "A state of the environment or agent that is not directly observable.",
            "uri": "obo:ACTO_000001",
        },
        "Observation": {
            "description": "Data received from the environment through sensory input.",
            "uri": "obo:ACTO_000003",
        },
        "Action": {
            "description": "An output of the agent that can affect the environment.",
            "uri": "obo:ACTO_000004",
        },
        "LikelihoodMatrix": {
            "description": "A probabilistic mapping from hidden states to observations.",
            "uri": "obo:TEMP_000061",
        },
        "TransitionMatrix": {
            "description": "A probabilistic mapping defining the dynamics of hidden states.",
            "uri": "obo:ACTO_000009",
        },
        "VariationalFreeEnergy": {
            "description": "A bound on Bayesian model evidence.",
            "uri": "obo:ACTO_000012",
        },
        "ExpectedFreeEnergy": {
            "description": "A quantity minimized by the agent to select policies.",
            "uri": "obo:ACTO_000011",
        },
    }


def parse_annotation(annotation: str) -> ParsedAnnotation:
    """Parse a ``KEY=VALUE`` annotation into its components.

    Returns a :class:`ParsedAnnotation` (a 3-tuple, so existing positional
    unpacking ``key, value, comment = parse_annotation(line)`` keeps working)
    where ``key`` and ``comment`` are ``None`` when absent. A leading ``#``
    introduces a trailing comment that is stripped and returned separately.

    Args:
        annotation: Raw annotation string (e.g., ``"A=LikelihoodMatrix"``
            or ``"o=Observation # sensory value"``).
    """
    comment: Optional[str] = None
    if "#" in annotation:
        annotation, comment = annotation.split("#", 1)
        comment = comment.strip()
        annotation = annotation.strip()

    if "=" in annotation:
        key, value = annotation.split("=", 1)
        return ParsedAnnotation(key.strip(), value.strip(), comment)

    return ParsedAnnotation(None, annotation.strip(), comment)


def validate_annotations(
    annotations: List[str], ontology_terms: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate annotations against ontology terms.

    Supports ``KEY=VALUE`` format where ``VALUE`` is matched (case-insensitively)
    against ontology term names. The returned dict preserves the established
    contract consumed by ``src/llm/processor.py`` (verbatim JSON injection into
    LLM prompts): ``valid_annotations``, ``invalid_annotations``,
    ``matched_terms`` (``key -> {annotation, term_name, description, uri, key,
    value, comment}``), ``suggestions`` (each
    ``{annotation, suggested_term, description}``), ``coverage_score``, and
    ``invalid_details``.

    The suggestion scan and term-lookup construction are delegated to the pure
    helpers :func:`_build_term_lookup`, :func:`_term_matches`, and
    :func:`suggest_terms` so callers can reuse them independently.

    Args:
        annotations: Annotation strings (e.g., ``["A=LikelihoodMatrix"]``).
        ontology_terms: Term dictionary; loaded by default.
    """
    try:
        if ontology_terms is None:
            ontology_terms = load_defined_ontology_terms()

        term_lookup = _build_term_lookup(ontology_terms)

        validation_result: dict[str, Any] = {
            "valid_annotations": [],
            "invalid_annotations": [],
            "matched_terms": {},  # key -> {term_name, description, uri}
            "suggestions": [],
            "coverage_score": 0.0,
            "invalid_details": [],
        }

        for annotation in annotations:
            classification = _term_matches(annotation, term_lookup)

            if not classification.matched:
                validation_result["invalid_annotations"].append(annotation)
                validation_result["invalid_details"].append(
                    {"annotation": annotation, "reason": classification.reason}
                )
                if classification.reason == "ontology term is not defined":
                    # Preserve the documented suggestion shape
                    # {annotation, suggested_term, description}.
                    nearby = suggest_terms(
                        [annotation],
                        ontology_terms,
                        max_distance=SUGGESTION_MAX_DISTANCE,
                    )
                    for suggestion in nearby:
                        validation_result["suggestions"].append(
                            {
                                "annotation": suggestion["annotation"],
                                "suggested_term": suggestion["suggested_term"],
                                "description": suggestion["description"],
                            }
                        )
                continue

            # Cross-annotation consistency: the same key must not map to two
            # different ontology terms (per-annotation _term_matches cannot see
            # previously matched keys, so this check lives here).
            match_info = classification.match_info
            assert match_info is not None  # matched implies match_info present
            mapping_key = classification.key or classification.value
            existing = validation_result["matched_terms"].get(mapping_key)
            if existing is not None and (
                str(existing.get("term_name", "")).casefold()
                != str(match_info["name"]).casefold()
            ):
                validation_result["invalid_annotations"].append(annotation)
                validation_result["invalid_details"].append(
                    {
                        "annotation": annotation,
                        "reason": "annotation key maps to multiple ontology terms",
                        "previous_term": existing.get("term_name"),
                    }
                )
                continue
            validation_result["valid_annotations"].append(annotation)
            validation_result["matched_terms"][mapping_key] = {
                "annotation": annotation,
                "term_name": match_info["name"],
                "description": match_info["data"].get("description", ""),
                "uri": match_info["data"].get("uri", ""),
                "key": classification.key,
                "value": classification.value,
                "comment": classification.comment,
            }

        # Calculate coverage score
        total_annotations = len(annotations)
        if total_annotations > 0:
            validation_result["coverage_score"] = (
                len(validation_result["valid_annotations"]) / total_annotations
            )

        logger.info(
            "Validated %d annotations: %d valid, %d invalid",
            len(annotations),
            len(validation_result["valid_annotations"]),
            len(validation_result["invalid_annotations"]),
        )

        return validation_result

    except Exception as e:
        logger.error("Validation failed: %s", e)
        return {
            "error": str(e),
            "valid_annotations": [],
            "invalid_annotations": list(annotations),
            "matched_terms": {},
            "suggestions": [],
            "coverage_score": 0.0,
            "invalid_details": [
                {"annotation": annotation, "reason": "validation failed"}
                for annotation in annotations
            ],
        }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: list[Any] = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def generate_ontology_report_for_file(
    gnn_file: Path,
    output_dir: Path,
    *,
    ontology_terms: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate ontology report for a single GNN file.

    Args:
        gnn_file: Path to the GNN file
        output_dir: Output directory for reports

    Returns:
        Dictionary with report generation results
    """
    try:
        # Process ontology for the file
        ontology_result = process_gnn_ontology(str(gnn_file), ontology_terms)

        if not ontology_result["success"]:
            return ontology_result

        # Create report
        report: dict[str, Any] = {
            "file_path": str(gnn_file),
            "file_name": gnn_file.name,
            "ontology_data": ontology_result["ontology_data"],
            "validation_result": ontology_result["validation_result"],
            "summary": {
                "total_concepts": len(ontology_result["ontology_data"]["concepts"]),
                "total_relations": len(ontology_result["ontology_data"]["relations"]),
                "total_properties": len(ontology_result["ontology_data"]["properties"]),
                "total_annotations": len(
                    ontology_result["ontology_data"]["annotations"]
                ),
                "valid_annotations": len(
                    ontology_result["validation_result"]["valid_annotations"]
                ),
                "invalid_annotations": len(
                    ontology_result["validation_result"]["invalid_annotations"]
                ),
                "coverage_score": ontology_result["validation_result"][
                    "coverage_score"
                ],
            },
        }

        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{gnn_file.stem}_ontology_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return {"success": True, "report_file": str(report_file), "report": report}

    except Exception as e:
        return {"success": False, "error": str(e)}
