#!/usr/bin/env python3
"""
Research Processor module for GNN Processing Pipeline.

Generates experimental hypotheses via rule-based static analysis (default)
and optional LLM-powered semantic analysis when an LLM provider is configured.

The FEATURES['fallback_mode'] flag indicates the module operates without
LLM dependencies — this is expected behavior, not a limitation.

Composability notes:
- All static analysis (`detect_model_family`, `extract_state_space_dims`,
  `count_connections`, `analyze_gnn`) is pure: content in, data out.
- Report rendering (`render_research_report`) and results summarization
  (`summarize_hypotheses`) are pure as well; only `write_research_outputs`
  and `process_research` touch the filesystem.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
)

logger = logging.getLogger(__name__)

FEATURES: dict[str, Any] = {
    "fallback_mode": True,  # Rule-based mode; LLM used opportunistically when available
    "model_family_detection": True,
    "dimension_aware_analysis": True,
    "llm_hypothesis_generation": False,  # Set True if LLM provider configured
}

#: Every value ``detect_model_family`` may return, for consumer dispatch.
MODEL_FAMILIES: tuple[str, ...] = (
    "pomdp",
    "hmm",
    "hierarchical",
    "continuous",
    "factor_graph",
    "unknown",
)


def _section_name(line: str) -> str | None:
    """Return a normalized level-two Markdown section name, if present."""
    match = re.match(r"^##(?!#)\s*(.*?)\s*$", line.strip())
    return match.group(1).casefold() if match else None


def _iter_section_lines(content: str, section: str) -> Iterator[str]:
    """Yield stripped, non-comment body lines of the named level-two section.

    The section body starts at the ``## <section>`` heading (case-insensitive)
    and ends at the next level-two heading. Repeated occurrences of the same
    heading re-enter the body, matching the original per-consumer scanners.
    """
    expected = section.casefold()
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        heading = _section_name(stripped)
        if heading == expected:
            in_section = True
            continue
        if heading is not None:
            in_section = False
            continue
        if in_section and stripped and not stripped.startswith("#"):
            yield stripped


def _first_section_value(content: str, section_name: str) -> str | None:
    """Return the first non-comment value in a named GNN section."""
    in_section = False
    for line in content.splitlines():
        heading = _section_name(line)
        if heading is not None:
            in_section = heading == section_name.casefold()
            continue
        stripped = line.strip()
        if in_section and stripped and not stripped.startswith("#"):
            return stripped
    return None


def _has_section(content: str, section_name: str) -> bool:
    """Return whether content contains the exact named level-two section."""
    expected = section_name.casefold()
    return any(_section_name(line) == expected for line in content.splitlines())


def detect_model_family(content: str) -> str:
    """
    Detect the Active Inference model family from GNN content.

    Returns one of :data:`MODEL_FAMILIES`:
    'pomdp', 'hmm', 'hierarchical', 'continuous', 'factor_graph', 'unknown'
    """
    content_lower = content.lower()

    # Check GNNSection first. More specific families must precede POMDP:
    # ``ActInfPOMDP_Hierarchical`` is hierarchical, not a plain POMDP.
    section_value = _first_section_value(content, "gnnsection")
    if section_value:
        section = re.sub(r"[^a-z0-9]+", "", section_value.casefold())
        if "hierarchical" in section:
            return "hierarchical"
        elif "hiddenmarkov" in section or section == "hmm":
            return "hmm"
        elif "continuous" in section:
            return "continuous"
        elif "factor" in section:
            return "factor_graph"
        elif "pomdp" in section:
            return "pomdp"

    # Detect from state space variables
    has_A = bool(re.search(r"^\s*A\s*\[", content, re.MULTILINE))
    has_B = bool(re.search(r"^\s*B\s*\[", content, re.MULTILINE))
    has_C = bool(re.search(r"^\s*C\s*\[", content, re.MULTILINE))
    has_D = bool(re.search(r"^\s*D\s*\[", content, re.MULTILINE))
    has_pi = bool(re.search(r"^\s*(?:π|pi)\s*\[", content, re.MULTILINE))

    # Hierarchical: multiple levels of A/B or explicit nesting
    level_count = len(re.findall(r"level\d|layer\d|hierarchical", content_lower))
    if level_count > 1:
        return "hierarchical"

    # Full POMDP: has A, B, C, D and policy
    if has_A and has_B and has_C and has_D and has_pi:
        return "pomdp"

    # HMM: A and B but no policy/action selection
    if has_A and has_B and not has_pi:
        return "hmm"

    # Continuous: no discrete B matrix, uses continuous dynamics
    if "continuous" in content_lower or "gaussian" in content_lower:
        return "continuous"

    # Factor graph
    if "factor" in content_lower and has_A:
        return "factor_graph"

    return "unknown"


def extract_state_space_dims(content: str) -> dict[str, list[int]]:
    """
    Extract variable dimensions from GNN StateSpaceBlock.
    Only extracts integer dimensions (not symbolic like pi).
    """
    dims: dict[str, list[int]] = {}
    pattern = r"^([^\W\d]\w*\'?)\s*\[([^\]]+)\]"

    for stripped in _iter_section_lines(content, "StateSpaceBlock"):
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        match = re.match(pattern, stripped)
        if match:
            var_name = match.group(1)
            dim_str = match.group(2)
            var_dims: list[int] = []
            for part in dim_str.split(","):
                part = part.strip()
                if part.startswith("type="):
                    continue
                try:
                    dimension = int(part)
                    if dimension <= 0:
                        logger.debug(
                            "Skipping non-positive dimension for %s: %s",
                            var_name,
                            part,
                        )
                        var_dims = []
                        break
                    var_dims.append(dimension)
                except ValueError:
                    logger.debug("Skipping non-integer dimension part: %s", part)
            if var_dims:
                dims[var_name] = var_dims

    return dims


def count_connections(content: str) -> dict[str, int]:
    """Count directed and undirected connections in the Connections section."""
    directed = 0
    undirected = 0

    for stripped in _iter_section_lines(content, "Connections"):
        directed += len(re.findall(r">", stripped))
        undirected += len(re.findall(r"(?<![>])-(?![>])", stripped))

    return {
        "directed": directed,
        "undirected": undirected,
        "total": directed + undirected,
    }


@dataclass(frozen=True)
class ModelAnalysis:
    """Pure static analysis of a single GNN model specification."""

    model_family: str
    dimensions: dict[str, list[int]]
    connections: dict[str, int]


def analyze_gnn(content: str) -> ModelAnalysis:
    """Run all pure static analyses over GNN content in one call."""
    return ModelAnalysis(
        model_family=detect_model_family(content),
        dimensions=extract_state_space_dims(content),
        connections=count_connections(content),
    )


def generate_rule_based_hypotheses(
    content: str,
    model_family: str,
    dims: Mapping[str, list[int]],
    connections: Mapping[str, int],
) -> list[dict[str, Any]]:
    """
    Generate research hypotheses via rule-based static analysis.

    Rules are domain-specific to Active Inference / generative model research.
    """
    hypotheses: list[dict[str, Any]] = []

    # Rule 1: High-Dimensionality -- only flag actual matrix variables, not arbitrary integers
    max_dim = 0
    large_vars: list[tuple[str, list[int], int]] = []
    for name, var_dims in dims.items():
        total = 1
        for d in var_dims:
            total *= d
        if total > 100:
            large_vars.append((name, var_dims, total))
            max_dim = max(max_dim, total)

    if max_dim > 1000:
        hypotheses.append(
            {
                "type": "dimensionality_reduction",
                "description": f"Apply structured mean-field or amortized inference for large variables: {[v[0] for v in large_vars]}",
                "rationale": f"Variables {[f'{v[0]}{v[1]}' for v in large_vars[:3]]} have high parameter counts. "
                f"Structured approximations can maintain accuracy while reducing computational cost.",
                "priority": "high",
            }
        )
    elif max_dim > 100:
        hypotheses.append(
            {
                "type": "dimensionality_check",
                "description": "Consider whether full joint inference is necessary for all state factors",
                "rationale": f"Moderate dimensionality detected (max {max_dim} elements). "
                f"Factored inference may be more efficient.",
                "priority": "medium",
            }
        )

    # Rule 2: Sparse Connectivity
    # Use structured dims when available; fall back to `name:` counting for non-standard GNN formats
    total_vars = len(dims)
    if total_vars == 0:
        total_vars = len(re.findall(r"name:", content))
    total_conns = connections.get("total", 0)
    if total_conns == 0:
        total_conns = len(re.findall(r"->", content))
    if total_vars > 2 and total_conns > 0:
        density = total_conns / (total_vars * (total_vars - 1))
        if density < 0.3:
            hypotheses.append(
                {
                    "type": "connectivity_enrichment",
                    "description": "Investigate potential missing causal links between model components",
                    "rationale": f"Graph density is {density:.2f} ({total_conns} connections, {total_vars} variables). "
                    f"Sparse graphs may miss important dependencies.",
                    "priority": "medium",
                }
            )

    # Rule 3: Model-family-specific suggestions
    if model_family == "pomdp":
        # Check for precision parameters
        has_precision = bool(
            re.search(r"precision|omega|gamma|alpha", content, re.IGNORECASE)
        )
        if not has_precision:
            hypotheses.append(
                {
                    "type": "precision_modulation",
                    "description": "Add precision parameters to modulate sensory and policy uncertainty",
                    "rationale": "The model lacks precision parameters. Active Inference with precision weighting "
                    "better captures attentional modulation and epistemic confidence.",
                    "priority": "high",
                }
            )

        # Check for planning horizon
        horizon_match = re.search(
            r"ModelTimeHorizon\s*=\s*(\w+)", content, re.IGNORECASE
        )
        if horizon_match and horizon_match.group(1).casefold() == "unbounded":
            hypotheses.append(
                {
                    "type": "planning_horizon",
                    "description": "Define explicit planning horizon T for tractable policy inference",
                    "rationale": "Unbounded time horizon requires truncation for policy selection. "
                    "Setting T=3-5 enables efficient Expected Free Energy computation.",
                    "priority": "medium",
                }
            )

        # Check for learning (parameter updating)
        has_learning = bool(
            re.search(
                r"learning|update.*param|concentration|dirichlet",
                content,
                re.IGNORECASE,
            )
        )
        if not has_learning:
            hypotheses.append(
                {
                    "type": "parameter_learning",
                    "description": "Add Dirichlet concentration parameters for online model learning",
                    "rationale": "Static A and B matrices cannot adapt. Adding concentration parameters "
                    "(a, b) enables Bayesian learning from experience.",
                    "priority": "medium",
                }
            )

    elif model_family == "hmm":
        hypotheses.append(
            {
                "type": "upgrade_to_pomdp",
                "description": "Consider extending HMM to full POMDP with action-dependent transitions",
                "rationale": "HMMs have no action selection. Adding a B[states,states,actions] tensor "
                "and preference vector C enables Active Inference policy optimization.",
                "priority": "low",
            }
        )

    elif model_family == "hierarchical":
        hypotheses.append(
            {
                "type": "message_passing",
                "description": "Verify belief propagation schedule across hierarchical levels",
                "rationale": "Hierarchical models require careful message-passing ordering. "
                "Top-down predictions must precede bottom-up updates.",
                "priority": "high",
            }
        )

    # Rule 4: Missing ontology annotation
    has_ontology = _has_section(content, "ActInfOntologyAnnotation")
    if not has_ontology:
        hypotheses.append(
            {
                "type": "ontology_annotation",
                "description": "Add ActInfOntologyAnnotation section for variable semantic labeling",
                "rationale": "Ontology annotations enable automatic cross-framework compatibility "
                "checks and improve model documentation.",
                "priority": "low",
            }
        )

    # Rule 5: Missing initial parameterization
    has_params = _has_section(content, "InitialParameterization")
    if not has_params and dims:
        hypotheses.append(
            {
                "type": "parameterization",
                "description": "Add InitialParameterization section with concrete matrix values",
                "rationale": "Without initial parameters, rendering steps cannot produce executable code. "
                "Define A, B, C, D values for complete model specification.",
                "priority": "high",
            }
        )

    return hypotheses


def summarize_hypotheses(
    hypotheses: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize hypotheses by priority and type for quick triage.

    Pure: accepts any iterable of hypothesis mappings (rule-based, LLM, or
    merged) and returns deterministic counts. Hypotheses with a priority
    outside {high, medium, low} or with a missing type are counted in
    ``total`` but omitted from the respective sub-counts.
    """
    by_priority: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    by_type: dict[str, int] = {}
    total = 0
    for hypothesis in hypotheses:
        total += 1
        priority = hypothesis.get("priority")
        if isinstance(priority, str) and priority in by_priority:
            by_priority[priority] += 1
        hypothesis_type = hypothesis.get("type")
        if isinstance(hypothesis_type, str):
            by_type[hypothesis_type] = by_type.get(hypothesis_type, 0) + 1
    return {
        "total": total,
        "by_priority": by_priority,
        "by_type": dict(sorted(by_type.items())),
    }


def _validate_llm_hypotheses(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize prospective LLM hypotheses before publication."""
    if not isinstance(value, list):
        return []

    validated: list[dict[str, Any]] = []
    for candidate in value[:3]:
        if not isinstance(candidate, dict):
            continue
        hypothesis_type = candidate.get("type")
        description = candidate.get("description")
        rationale = candidate.get("rationale")
        priority = candidate.get("priority")
        if not isinstance(hypothesis_type, str) or not hypothesis_type.strip():
            continue
        if not isinstance(description, str) or not description.strip():
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        if not isinstance(priority, str) or not priority.strip():
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", hypothesis_type):
            continue
        if priority not in {"high", "medium", "low"}:
            continue
        if len(description.split()) > 20 or len(rationale.split()) > 50:
            continue
        validated.append(
            {
                "type": hypothesis_type,
                "description": description.strip(),
                "rationale": rationale.strip(),
                "priority": priority,
                "source": "llm_generated",
                "claim_scope": "prospective_unvalidated_hypothesis",
            }
        )
    return validated


async def _generate_llm_hypotheses(
    model_family: str,
    dims: Mapping[str, list[int]],
    logger: logging.Logger,
) -> list[dict[str, Any]] | None:
    """Generate schema-validated hypotheses when an LLM is available."""
    try:
        from gnn.llm.llm_processor import initialize_global_processor
        from gnn.llm.providers.base_provider import LLMMessage

        processor = await initialize_global_processor()
        if not processor:
            return None
    except Exception:
        return None

    dim_summary = ", ".join(f"{k}{v}" for k, v in list(dims.items())[:6])
    prompt = f"""You are an Active Inference research assistant analyzing a GNN model specification.

Model family: {model_family}
Key variables: {dim_summary}

Based on this Active Inference generative model, generate 2-3 specific, actionable research hypotheses.
Focus on: model extensions, inference improvements, or experimental validations.

Respond with a JSON array of hypothesis objects, each with:
- type: short identifier (snake_case)
- description: one sentence action (20 words or less)
- rationale: explanation referencing specific model structure (50 words or less)
- priority: "high", "medium", or "low"

JSON only, no prose:"""

    try:
        from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL

        model_name = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        messages: list[Any] = [LLMMessage(role="user", content=prompt)]
        response = await processor.get_response(
            messages=messages, model_name=model_name, max_tokens=800
        )
        # Parse JSON response
        json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
        if json_match:
            hypotheses = _validate_llm_hypotheses(json.loads(json_match.group(0)))
            return hypotheses or None
        return None
    except Exception as e:
        logger.debug(f"LLM hypothesis generation failed: {e}")
        return None


def merge_llm_hypotheses(
    llm_hypotheses: list[dict[str, Any]],
    rule_hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge LLM and rule-based hypotheses, preferring the LLM ones.

    LLM hypotheses come first (more specific); rule-based hypotheses whose
    ``type`` is already covered by an LLM hypothesis are dropped.
    """
    covered = {h.get("type") for h in llm_hypotheses}
    return llm_hypotheses + [h for h in rule_hypotheses if h.get("type") not in covered]


def discover_gnn_files(target_dir: Path, recursive: bool) -> list[Path]:
    """Return the sorted GNN markdown files under ``target_dir``.

    Non-recursive discovery scans only the top level; recursive discovery
    walks the whole tree. A missing directory yields an empty list.
    """
    if not target_dir.is_dir():
        return []
    from gnn.processing.discovery import is_model_source_path

    pattern = target_dir.rglob("*.md") if recursive else target_dir.glob("*.md")
    return sorted(path for path in pattern if is_model_source_path(path))


def render_research_report(results: Mapping[str, Any]) -> str:
    """Render the markdown research report from a results payload.

    Pure: no filesystem access. ``write_research_outputs`` writes exactly
    this rendering to ``research_report.md``.
    """
    lines: list[str] = ["# Research Hypotheses Report\n"]
    lines.append(f"**Analysis mode**: {results['analysis_mode']}\n\n")
    lines.append(
        "All items below are prospective, unvalidated hypotheses generated "
        "from static model structure; they are not experimental findings.\n\n"
    )

    for entry in results["hypotheses_generated"]:
        lines.append(f"## {entry['file']} ({entry['model_family']} model)\n")

        # Group by priority
        high = [h for h in entry["hypotheses"] if h.get("priority") == "high"]
        medium = [h for h in entry["hypotheses"] if h.get("priority") == "medium"]
        low = [h for h in entry["hypotheses"] if h.get("priority") == "low"]

        for priority_label, hyps in [
            ("High Priority", high),
            ("Medium Priority", medium),
            ("Low Priority", low),
        ]:
            if hyps:
                lines.append(f"### {priority_label}\n")
                for h in hyps:
                    lines.append(f"- **{h['type']}**: {h['description']}\n")
                    lines.append(f"  - *Rationale*: {h['rationale']}\n")
                    lines.append(f"  - *Source*: {h['source']}\n")
        lines.append("\n")

    return "".join(lines)


def write_research_outputs(results_dir: Path, results: dict[str, Any]) -> None:
    """Write the JSON summaries and the markdown report for a research run.

    The three JSON artifacts share one payload; the report is written
    atomically (temp file + ``os.replace``) so partial reports are never
    observed by downstream steps.
    """
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    for name in (
        "research_results.json",
        "research_summary.json",
        "research_processing_summary.json",
    ):
        (results_dir / name).write_text(payload, encoding="utf-8")

    report_path = results_dir / "research_report.md"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=results_dir, delete=False
    ) as tmp:
        tmp.write(render_research_report(results))
    os.replace(tmp.name, str(report_path))


def process_research(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process research for GNN files.

    Generates deterministic experimental hypotheses based on static analysis rules.
    LLM-powered hypotheses are added opportunistically when an LLM provider is available.
    """
    logger = logging.getLogger("research")

    try:
        log_step_start(logger, "Processing research")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "processed_files": 0,
            "success": True,
            "hypotheses_generated": [],
            "model_families_detected": {},
            "errors": [],
            "analysis_mode": "rule_based",
            "claim_scope": "prospective_unvalidated_hypotheses",
        }

        if not target_dir.is_dir():
            results["success"] = False
            results["errors"].append(
                {"file": str(target_dir), "error": "target directory not found"}
            )

        recursive = kwargs.get("recursive", False)
        if not isinstance(recursive, bool):
            results["success"] = False
            results["errors"].append(
                {
                    "file": str(target_dir),
                    "error": "recursive must be a boolean",
                    "error_type": "invalid_configuration",
                }
            )

        gnn_files: list[Path] = []
        if isinstance(recursive, bool):
            gnn_files = discover_gnn_files(target_dir, recursive)
        results["processed_files"] = len(gnn_files)

        for gnn_file in gnn_files:
            try:
                content = gnn_file.read_text(encoding="utf-8")
                relative_file = gnn_file.relative_to(target_dir).as_posix()

                # Detect model family and structural evidence in one pass
                analysis = analyze_gnn(content)
                results["model_families_detected"][relative_file] = (
                    analysis.model_family
                )

                if verbose:
                    logger.info(
                        f"{relative_file}: detected as '{analysis.model_family}' model"
                    )

                # Rule-based hypotheses (always available)
                hypotheses = [
                    {
                        **hypothesis,
                        "source": "rule_based_static_analysis",
                        "claim_scope": "prospective_unvalidated_hypothesis",
                    }
                    for hypothesis in generate_rule_based_hypotheses(
                        content,
                        analysis.model_family,
                        analysis.dimensions,
                        analysis.connections,
                    )
                ]

                # Attempt LLM-powered hypotheses (opportunistic)
                if FEATURES.get("llm_hypothesis_generation"):
                    try:
                        llm_hypotheses = asyncio.run(
                            _generate_llm_hypotheses(
                                analysis.model_family,
                                analysis.dimensions,
                                logger,
                            )
                        )
                        if llm_hypotheses:
                            results["analysis_mode"] = "llm_enhanced"
                            hypotheses = merge_llm_hypotheses(
                                llm_hypotheses, hypotheses
                            )
                    except Exception as e:
                        logger.debug(
                            f"LLM hypotheses unavailable for {gnn_file.name}: {e}"
                        )

                if hypotheses:
                    results["hypotheses_generated"].append(
                        {
                            "file": relative_file,
                            "model_family": analysis.model_family,
                            "dimension_count": len(analysis.dimensions),
                            "analysis_evidence": {
                                "dimensions": analysis.dimensions,
                                "connections": analysis.connections,
                            },
                            "hypotheses": hypotheses,
                        }
                    )

            except Exception as e:
                logger.warning(f"Could not generate hypotheses for {gnn_file}: {e}")
                results["errors"].append({"file": str(gnn_file), "error": str(e)})

        if results["errors"]:
            results["success"] = False

        # Save results
        write_research_outputs(results_dir, results)

        if results["success"]:
            log_step_success(logger, "Research processing completed successfully")
        else:
            log_step_error(logger, "Research processing completed with errors")
        return bool(results["success"])

    except Exception as e:
        log_step_error(logger, "Research processing failed", error=str(e))
        return False
