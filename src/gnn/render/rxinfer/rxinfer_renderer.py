#!/usr/bin/env python3
"""
RxInfer.jl Renderer

Renders GNN specifications to RxInfer.jl simulation code using probabilistic programming.
This renderer creates executable RxInfer.jl simulations configured from parsed GNN POMDP specifications.

Features:
- GNN-to-RxInfer parameter extraction
- Julia probabilistic programming code generation
- Bayesian Active Inference model specification
- Pipeline integration support

Author: GNN RxInfer Integration
Date: 2024
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gnn.render.pomdp_contract import (
    ModelKind,
    build_canonical_pomdp_spec,
    detect_model_kind,
    detect_model_kinds,
    unsupported_composition_reason,
    unsupported_nonstationary_reason,
)
from gnn.render.rxinfer.model_strategies import get_model_strategy


class RxInferRenderer:
    """
    RxInfer.jl renderer for generating Julia probabilistic programming code from GNN specifications.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize RxInfer renderer.

        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)

    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to RxInfer.jl simulation code.

        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output RxInfer script

        Returns:
            Tuple of (success, message)
        """
        try:
            from gnn.extract.pomdp_extractor import extract_pomdp_from_file
            from gnn.render.pomdp_processor import POMDPRenderProcessor

            pomdp_space = extract_pomdp_from_file(gnn_file_path, strict_validation=True)
            if pomdp_space is None:
                raise ValueError(f"No valid POMDP matrices found in {gnn_file_path}")
            gnn_spec = POMDPRenderProcessor(output_path.parent)._pomdp_to_gnn_spec(
                pomdp_space, native_agents=True
            )
            rxinfer_code = self._generate_rxinfer_simulation_code(
                gnn_spec, gnn_file_path.stem
            )

            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rxinfer_code)
            _write_rxinfer_execution_metadata(output_path, gnn_spec)

            self.logger.info(f"Generated RxInfer.jl simulation: {output_path}")
            return True, "Successfully generated RxInfer.jl simulation code"

        except Exception as e:
            error_msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def _parse_gnn_content(self, content: str, model_name: str) -> Dict[str, Any]:
        """Parse GNN content into a structured dictionary (simplified parser)."""
        gnn_spec: dict[str, Any] = {
            "model_name": model_name,
            "variables": [],
            "model_parameters": {},
            "initial_parameterization": {},
        }

        # Simple parser for key sections
        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
            elif current_section == "ModelParameters" and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        gnn_spec["model_parameters"][key] = float(value)
                    else:
                        gnn_spec["model_parameters"][key] = int(value)
                except ValueError:
                    gnn_spec["model_parameters"][key] = value

        return gnn_spec

    def _generate_rxinfer_simulation_code_simple(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """Require the canonical renderer for file-based RxInfer rendering."""
        raise ValueError(
            f"RxInfer rendering for {model_name} requires explicit POMDP matrices"
        )

    def _generate_rxinfer_simulation_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """
        Generate executable RxInfer.jl simulation code from GNN specification.

        Args:
            gnn_spec: Parsed GNN specification
            model_name: Name of the model

        Returns:
            Generated Julia code string
        """
        # A composed spec declares more than one render family: rendering the
        # single winner (e.g. the discrete multi-agent strategy against an
        # F/H/Q/R parameterization) would fabricate the wrong family or fail
        # with a misleading missing-matrices error. Refuse with the explicit
        # unsupported-composition receipt instead.
        kinds = detect_model_kinds(gnn_spec)
        if kinds == frozenset({ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}):
            from gnn.render.multi_agent_continuous import (
                generate_multi_agent_continuous_script,
            )

            return generate_multi_agent_continuous_script(gnn_spec, "rxinfer")
        if ModelKind.CONTINUOUS in kinds and len(kinds) > 1:
            raise ValueError(unsupported_composition_reason(kinds))
        if ModelKind.NONSTATIONARY in kinds:
            # A B_t/B_regime parameterization cannot be expressed as one
            # static @model transition; refuse with the explicit receipt
            # instead of silently rendering static dynamics.
            raise ValueError(unsupported_nonstationary_reason(kinds))
        if detect_model_kind(gnn_spec) == ModelKind.CONTINUOUS:
            # Linear-Gaussian models carry F/H/Q/R + prior, never A/B/C/D;
            # canonicalising would demand categorical matrices that do not
            # exist. The continuous strategy validates its own contract.
            return self._generate_canonical_rxinfer_code(gnn_spec, model_name)
        from gnn.render.multi_agent_common import has_native_multi_agent_structure

        canonical_spec = (
            gnn_spec
            if has_native_multi_agent_structure(gnn_spec)
            else build_canonical_pomdp_spec(gnn_spec)
        )
        # Renderer options may set inference_mode ("batch" | "online"); a GNN
        # file's own ModelParameters declaration wins over the option so the
        # spec stays authoritative.
        mode = self.options.get("inference_mode")
        if mode is not None:
            canonical_spec.setdefault("model_parameters", {}).setdefault(
                "inference_mode", str(mode)
            )
        return self._generate_canonical_rxinfer_code(canonical_spec, model_name)

    def _generate_canonical_rxinfer_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """Dispatch code generation to the strategy for this detected model kind.

        The flat POMDP generator is preserved as ``FlatStrategy``. Every
        registered model kind supplies a concrete generator.
        """
        model_kind = detect_model_kind(gnn_spec)
        strategy = get_model_strategy(model_kind)
        return strategy.generate_model_code(gnn_spec, model_name)


def render_gnn_to_rxinfer(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to RxInfer.jl simulation script.

    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_path: Path for output RxInfer script
        options: Optional rendering options

    Returns:
        Tuple of (success, message, warnings: List[str])
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "rxinfer")
    if refusal:
        return False, refusal, []

    logger = logging.getLogger(__name__)

    try:
        # Validate input
        spec_obj: object = gnn_spec
        if not isinstance(spec_obj, dict):
            return False, "Invalid GNN specification: must be a dictionary", []

        renderer = RxInferRenderer(options)

        # Get model name safely
        model_name = spec_obj.get("name") or spec_obj.get("model_name", "GNN_Model")

        # Generate simulation code directly from spec (using simplified working version)
        try:
            # Use the full generator with updated syntax
            rxinfer_code = renderer._generate_rxinfer_simulation_code(
                spec_obj, model_name
            )
        except Exception as gen_error:
            logger.error(f"Code generation failed: {gen_error}")
            return False, f"Error generating RxInfer.jl code: {gen_error}", []

        # Write output file
        try:
            metadata = build_rxinfer_execution_metadata(gnn_spec)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rxinfer_code)
            if metadata:
                _write_rxinfer_execution_metadata(output_path, gnn_spec, metadata)
        except Exception as write_error:
            logger.error(f"Failed to write output file: {write_error}")
            return False, f"Error writing RxInfer.jl script: {write_error}", []

        message = f"Generated RxInfer.jl simulation script: {output_path}"
        warnings: list[Any] = []

        # Check for potential issues
        if not (
            gnn_spec.get("initial_parameterization")
            or gnn_spec.get("initialparameterization")
        ):
            warnings.append("No initial parameterization found - using defaults")

        if not gnn_spec.get("model_parameters"):
            warnings.append("No model parameters found - using inferred dimensions")

        logger.info(f"Successfully generated RxInfer.jl script for {model_name}")
        return True, message, warnings

    except Exception as e:
        logger.error(f"Unexpected error in render_gnn_to_rxinfer: {e}", exc_info=True)
        return False, f"Error generating RxInfer.jl script: {e}", []


def build_rxinfer_execution_metadata(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build Step 12 execution metadata for declared RxInfer agent populations.

    Self-contained — does not import from the retired toml_generator.py.
    """
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        return {}

    agents = _extract_declared_rxinfer_agents(initial)
    if not agents:
        return {}

    topology = _extract_agent_topology_inline(initial, agents)
    return {
        "schema": "gnn_rxinfer_execution_metadata_v1",
        "agent_count": len(agents),
        "agents": agents,
        "topology": topology,
    }


def _coerce_positive_int_inline(value: Any) -> int:
    """Coerce a value to a positive int, returning 0 for missing/invalid values."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)


def _as_list_inline(value: Any) -> Optional[list]:
    """Return value as a list when it is list-like enough."""
    return value if isinstance(value, list) else None


def _extract_compact_agents_inline(
    params: Dict[str, Any], nr_agents: int
) -> Optional[List[Dict[str, Any]]]:
    """Extract agents from compact vectorized InitialParameterization keys."""
    agent_ids = _as_list_inline(params.get("agent_ids"))
    initial_positions = _as_list_inline(params.get("agent_initial_positions"))
    target_positions = _as_list_inline(params.get("agent_target_positions"))
    if agent_ids is None and initial_positions is None and target_positions is None:
        return None
    radii = _as_list_inline(params.get("agent_radii")) or _as_list_inline(
        params.get("agent_radius")
    )
    default_radius = params.get("agent_default_radius", 1.0)
    required = {
        "agent_ids": agent_ids,
        "agent_initial_positions": initial_positions,
        "agent_target_positions": target_positions,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing compact multi-agent keys: {', '.join(missing)}")
    assert agent_ids is not None
    assert initial_positions is not None
    assert target_positions is not None
    lengths = {
        "agent_ids": len(agent_ids),
        "agent_initial_positions": len(initial_positions),
        "agent_target_positions": len(target_positions),
    }
    if any(length != nr_agents for length in lengths.values()):
        raise ValueError(
            f"Compact multi-agent lengths must match nr_agents={nr_agents}: {lengths}"
        )
    if radii is not None and len(radii) != nr_agents:
        raise ValueError(
            f"agent_radii length {len(radii)} must match nr_agents={nr_agents}"
        )
    return [
        {
            "id": agent_ids[index],
            "radius": radii[index] if radii is not None else default_radius,
            "initial_position": initial_positions[index],
            "target_position": target_positions[index],
        }
        for index in range(nr_agents)
    ]


def _extract_indexed_agents_inline(
    params: Dict[str, Any], nr_agents: int
) -> List[Dict[str, Any]]:
    """Extract indexed agent{i}_... agent definitions."""
    agents: List[Dict[str, Any]] = []
    for i in range(1, nr_agents + 1):
        agent_id = params.get(f"agent{i}_id")
        radius = params.get(f"agent{i}_radius", params.get("agent_default_radius", 1.0))
        initial_pos = params.get(f"agent{i}_initial_position")
        target_pos = params.get(f"agent{i}_target_position")
        if all(v is not None for v in [agent_id, radius, initial_pos, target_pos]):
            agents.append(
                {
                    "id": agent_id,
                    "radius": radius,
                    "initial_position": initial_pos,
                    "target_position": target_pos,
                }
            )
    return agents


def _infer_indexed_agent_count(params: Dict[str, Any]) -> int:
    """Infer agent count from agent{i}_... keys when nr_agents is omitted."""
    agent_indices: set[int] = set()
    for key in params:
        match = re.match(r"agent(\d+)_", str(key))
        if match:
            agent_indices.add(int(match.group(1)))
    return max(agent_indices) if agent_indices else 0


def _extract_declared_rxinfer_agents(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract explicitly declared agents without inventing default agents.

    Self-contained — does not import from the retired toml_generator.py.
    """
    nr_agents = _coerce_positive_int_inline(params.get("nr_agents"))
    if nr_agents > 0:
        compact_agents = _extract_compact_agents_inline(params, nr_agents)
        if compact_agents is not None:
            return compact_agents
        indexed_agents = _extract_indexed_agents_inline(params, nr_agents)
        if len(indexed_agents) == nr_agents:
            return indexed_agents
        raise ValueError(
            "nr_agents was provided but agent configuration is incomplete. "
            "Provide compact agent_ids/agent_initial_positions/agent_target_positions "
            "or complete agent{i}_id/agent{i}_initial_position/agent{i}_target_position keys."
        )

    indexed_count = _infer_indexed_agent_count(params)
    if indexed_count <= 0:
        return []
    indexed_agents = _extract_indexed_agents_inline(params, indexed_count)
    if len(indexed_agents) != indexed_count:
        raise ValueError(
            "Indexed agent configuration is incomplete. Provide complete "
            "agent{i}_id/agent{i}_initial_position/agent{i}_target_position keys."
        )
    return indexed_agents


def _extract_agent_topology_inline(
    params: Dict[str, Any], agents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract explicit multi-agent topology metadata for execution.

    Self-contained — does not import from the retired toml_generator.py.
    """
    agent_ids = [agent["id"] for agent in agents if "id" in agent]

    # Edges
    raw_edges = (
        params.get("agent_edges")
        or params.get("topology_edges")
        or params.get("edges", [])
    )
    edges: List[Dict[str, Any]] = []
    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if isinstance(raw_edge, dict):
                source = raw_edge.get("source", raw_edge.get("from"))
                target = raw_edge.get("target", raw_edge.get("to"))
            elif isinstance(raw_edge, (list, tuple)) and len(raw_edge) >= 2:
                source, target = raw_edge[0], raw_edge[1]
            else:
                continue
            if source is not None and target is not None:
                edges.append({"source": source, "target": target})

    # Clusters
    raw_clusters = (
        params.get("agent_clusters")
        or params.get("topology_clusters")
        or params.get("clusters", {})
    )
    clusters: List[Dict[str, Any]] = []
    if isinstance(raw_clusters, dict):
        for name, agent_ids_list in raw_clusters.items():
            if isinstance(agent_ids_list, list):
                clusters.append({"name": str(name), "agent_ids": agent_ids_list})
    elif isinstance(raw_clusters, list):
        for index, raw_cluster in enumerate(raw_clusters, start=1):
            if isinstance(raw_cluster, dict):
                ids = raw_cluster.get("agent_ids") or raw_cluster.get("agents")
                if isinstance(ids, list):
                    clusters.append(
                        {
                            "name": str(raw_cluster.get("name", f"cluster_{index}")),
                            "agent_ids": ids,
                        }
                    )

    # Topology type
    topology_type = params.get("agent_topology_type") or params.get("topology_type")
    if topology_type is None:
        if clusters:
            topology_type = "clustered"
        elif edges:
            topology_type = "network"
        else:
            topology_type = "agent_population"

    topology: Dict[str, Any] = {
        "type": str(topology_type),
        "agent_ids": agent_ids,
        "edges": edges,
        "clusters": clusters,
    }
    message_passing = params.get("message_passing") or params.get(
        "agent_message_passing"
    )
    if message_passing:
        topology["message_passing"] = str(message_passing)
    return topology


def _write_rxinfer_execution_metadata(
    output_path: Path,
    gnn_spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write a sibling execution metadata JSON artifact when metadata exists."""
    metadata = (
        metadata if metadata is not None else build_rxinfer_execution_metadata(gnn_spec)
    )
    if not metadata:
        return None
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = dict(metadata)
    metadata["script_path"] = str(output_path)
    metadata["script_sha256"] = _sha256_file(output_path)
    metadata["metadata_provenance"] = "rendered_rxinfer_sidecar"
    topology = dict(metadata.get("topology") or {})
    topology.setdefault("source", str(metadata_path))
    metadata["topology"] = topology
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a rendered script."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
