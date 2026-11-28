#!/usr/bin/env python3
"""
Export formatters module for GNN Processing Pipeline.

This module provides format-specific export functionality.
"""

import json
import logging
import pickle  # nosec B403
import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict

from defusedxml.minidom import parseString as _xml_parseString

logger = logging.getLogger(__name__)


def _pretty_xml(root: ET.Element) -> str:
    """Pretty-print an ElementTree element to an XML string."""
    return str(_xml_parseString(ET.tostring(root)).toprettyxml(indent="  "))  # nosec B318


def _write_pretty_xml(root: ET.Element, output_file: Path) -> None:
    """Serialize ``root`` to a pretty-printed XML file at ``output_file``."""
    xml_str = _pretty_xml(root)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_str)


def _dump_pickle(data: Any, output_file: Path) -> None:
    """Serialize ``data`` to a Python pickle file at ``output_file``."""
    with open(output_file, "wb") as f:
        pickle.dump(data, f)  # nosec B301


def _connection_pairs(parsed_content: Dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Expand legacy single endpoints and structured GNN connection groups."""
    for connection in parsed_content.get("connections", []):
        sources = connection.get(
            "source_variables", connection.get("sources", [connection.get("source")])
        )
        targets = connection.get(
            "target_variables", connection.get("targets", [connection.get("target")])
        )
        sources = [sources] if isinstance(sources, str) else sources or []
        targets = [targets] if isinstance(targets, str) else targets or []
        for source in sources:
            for target in targets:
                if (
                    isinstance(source, str)
                    and source
                    and isinstance(target, str)
                    and target
                ):
                    yield source, target


def _graph_structure(
    parsed_content: Dict[str, Any],
) -> tuple[Dict[str, str], list[tuple[str, str]]]:
    """Resolve connection names to emitted graph IDs in stable source order."""
    names = dict.fromkeys(
        var.get("name") for var in parsed_content.get("variables", [])
    )
    connections = list(_connection_pairs(parsed_content))
    for source, target in connections:
        names.setdefault(source, None)
        names.setdefault(target, None)
    nodes = {
        name: f"n{index}"
        for index, name in enumerate(
            name for name in names if isinstance(name, str) and name
        )
    }
    return nodes, connections


def export_to_json(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export parsed content to JSON format."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_content, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning("export_to_json failed: %s", e)
        return False


def export_to_xml(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export parsed content to XML format."""
    try:
        # Create root element
        root = ET.Element("gnn_model")

        # Add sections
        sections_elem = ET.SubElement(root, "sections")
        sections = parsed_content.get("sections") or parsed_content.get(
            "raw_sections", {}
        )
        for section_name, section_content in sections.items():
            section_elem = ET.SubElement(sections_elem, "section")
            section_elem.set("name", section_name)
            lines = (
                section_content.splitlines()
                if isinstance(section_content, str)
                else section_content
            )
            for line in lines:
                line_elem = ET.SubElement(section_elem, "line")
                line_elem.text = line

        # Add variables
        variables_elem = ET.SubElement(root, "variables")
        for var in parsed_content.get("variables", []):
            var_elem = ET.SubElement(variables_elem, "variable")
            var_elem.set("name", var.get("name", ""))
            var_elem.set("type", var.get("type", var.get("data_type", "")))

        # Add connections
        connections_elem = ET.SubElement(root, "connections")
        for source, target in _connection_pairs(parsed_content):
            conn_elem = ET.SubElement(connections_elem, "connection")
            conn_elem.set("source", source)
            conn_elem.set("target", target)

        # Write XML
        xml_str = _xml_parseString(ET.tostring(root)).toprettyxml(indent="  ")  # nosec B318
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_str)

        return True
    except Exception as e:
        logger.warning("export_to_xml failed: %s", e)
        return False


def export_to_graphml(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export parsed content to GraphML format."""
    try:
        # Create GraphML structure
        node_ids, connections = _graph_structure(parsed_content)
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")

        # Add key definitions
        key_id = ET.SubElement(root, "key")
        key_id.set("id", "d0")
        key_id.set("for", "node")
        key_id.set("attr.name", "label")
        key_id.set("attr.type", "string")

        key_weight = ET.SubElement(root, "key")
        key_weight.set("id", "d1")
        key_weight.set("for", "edge")
        key_weight.set("attr.name", "weight")
        key_weight.set("attr.type", "double")

        # Create graph
        graph = ET.SubElement(root, "graph")
        graph.set("id", "G")
        graph.set("edgedefault", "directed")

        # Add nodes (variables)
        for name, node_id in node_ids.items():
            node = ET.SubElement(graph, "node")
            node.set("id", node_id)
            data = ET.SubElement(node, "data")
            data.set("key", "d0")
            data.text = name

        # Add edges (connections)
        for i, (source, target) in enumerate(connections):
            edge = ET.SubElement(graph, "edge")
            edge.set("id", f"e{i}")
            edge.set("source", node_ids[source])
            edge.set("target", node_ids[target])

        # Write GraphML
        xml_str = _xml_parseString(ET.tostring(root)).toprettyxml(indent="  ")  # nosec B318
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_str)

        return True
    except Exception as e:
        logger.warning("export_to_graphml failed: %s", e)
        return False


def export_to_gexf(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export parsed content to GEXF format."""
    try:
        # Create GEXF structure
        node_ids, connections = _graph_structure(parsed_content)
        root = ET.Element("gexf")
        root.set("xmlns", "http://www.gexf.net/1.2draft")
        root.set("version", "1.2")

        # Add meta
        meta = ET.SubElement(root, "meta")
        meta.set("lastmodifieddate", "2024-01-01")
        creator = ET.SubElement(meta, "creator")
        creator.text = "GNN Pipeline"
        description = ET.SubElement(meta, "description")
        description.text = "GNN Model Export"

        # Create graph
        graph = ET.SubElement(root, "graph")
        graph.set("mode", "static")
        graph.set("defaultedgetype", "directed")

        # Add nodes
        nodes = ET.SubElement(graph, "nodes")
        for name, node_id in node_ids.items():
            node = ET.SubElement(nodes, "node")
            node.set("id", node_id)
            node.set("label", name)

        # Add edges
        edges = ET.SubElement(graph, "edges")
        for i, (source, target) in enumerate(connections):
            edge = ET.SubElement(edges, "edge")
            edge.set("id", f"e{i}")
            edge.set("source", node_ids[source])
            edge.set("target", node_ids[target])

        # Write GEXF
        xml_str = _xml_parseString(ET.tostring(root)).toprettyxml(indent="  ")  # nosec B318
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_str)

        return True
    except Exception as e:
        logger.warning("export_to_gexf failed: %s", e)
        return False


def export_to_pickle(parsed_content: Any, output_file: Path) -> bool:
    """Export parsed content to pickle format."""
    try:
        _dump_pickle(parsed_content, output_file)
        return True
    except Exception as e:
        logger.warning("export_to_pickle failed: %s", e)
        return False


def export_to_json_gnn(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export GNN model data to JSON format."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning("export_to_json_gnn failed: %s", e)
        return False


def export_to_xml_gnn(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export GNN model data to XML format."""
    try:
        # Create root element
        root = ET.Element("gnn_model")
        root.set("version", "1.0")

        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        for key, value in model_data.get("metadata", {}).items():
            meta_elem = ET.SubElement(metadata, key)
            meta_elem.text = str(value)

        # Add model type
        model_type = ET.SubElement(root, "model_type")
        model_type.text = model_data.get("model_type", "gnn")

        # Add sections
        sections_elem = ET.SubElement(root, "sections")
        for section_name, section_content in model_data.get("sections", {}).items():
            section_elem = ET.SubElement(sections_elem, "section")
            section_elem.set("name", section_name)
            for line in section_content:
                line_elem = ET.SubElement(section_elem, "line")
                line_elem.text = str(line)

        # Add variables
        variables_elem = ET.SubElement(root, "variables")
        for var in model_data.get("variables", []):
            var_elem = ET.SubElement(variables_elem, "variable")
            var_elem.set("name", var.get("name", ""))
            var_elem.set("type", var.get("type", ""))

        # Add connections
        connections_elem = ET.SubElement(root, "connections")
        for conn in model_data.get("connections", []):
            conn_elem = ET.SubElement(connections_elem, "connection")
            conn_elem.set("source", conn.get("source", ""))
            conn_elem.set("target", conn.get("target", ""))

        # Write XML
        _write_pretty_xml(root, output_file)

        return True
    except Exception as e:
        logger.warning("export_to_xml_gnn failed: %s", e)
        return False


def export_to_python_pickle(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export GNN model data to Python pickle format."""
    try:
        _dump_pickle(model_data, output_file)
        return True
    except Exception as e:
        logger.warning("export_to_python_pickle failed: %s", e)
        return False


def export_to_plaintext_summary(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export GNN model data to plaintext summary format."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("GNN Model Summary\n")
            f.write("=" * 50 + "\n\n")

            # Model type
            f.write(f"Model Type: {model_data.get('model_type', 'gnn')}\n\n")

            # Variables
            variables = model_data.get("variables", [])
            f.write(f"Variables ({len(variables)}):\n")
            for var in variables:
                f.write(f"  - {var.get('name', '')}: {var.get('type', '')}\n")
            f.write("\n")

            # Connections
            connections = model_data.get("connections", [])
            f.write(f"Connections ({len(connections)}):\n")
            for conn in connections:
                f.write(f"  - {conn.get('source', '')} -> {conn.get('target', '')}\n")
            f.write("\n")

            # Sections
            sections = model_data.get("sections", {})
            f.write(f"Sections ({len(sections)}):\n")
            for section_name, section_content in sections.items():
                f.write(f"  - {section_name}: {len(section_content)} lines\n")

        return True
    except Exception as e:
        logger.warning("export_to_plaintext_summary failed: %s", e)
        return False


def export_to_plaintext_dsl(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export GNN model data to plaintext DSL format."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Write model header
            f.write("# GNN Model\n\n")

            # Write variables
            variables = model_data.get("variables", [])
            if variables:
                f.write("## Variables\n\n")
                for var in variables:
                    f.write(f"{var.get('name', '')}: {var.get('type', '')}\n")
                f.write("\n")

            # Write connections
            connections = model_data.get("connections", [])
            if connections:
                f.write("## Connections\n\n")
                for conn in connections:
                    f.write(f"{conn.get('source', '')} -> {conn.get('target', '')}\n")
                f.write("\n")

            # Write sections
            sections = model_data.get("sections", {})
            for section_name, section_content in sections.items():
                f.write(f"## {section_name}\n\n")
                for line in section_content:
                    f.write(f"{line}\n")
                f.write("\n")

        return True
    except Exception as e:
        logger.warning("export_to_plaintext_dsl failed: %s", e)
        return False
