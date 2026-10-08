#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio UI
Low-dependency visual design experience for Active Inference models
"""

import html
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def build_design_studio(
    markdown_text: str, export_path: Path, logger: logging.Logger
) -> "gr.Blocks":
    """
    Build the State Space Design Studio GUI interface

    Features:
    - Visual state space designer
    - Ontology term editor
    - Connection graph interface
    - Parameter tuning controls
    - Low-dependency HTML/CSS design
    """

    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for Design Studio functionality")

    logger.info("🎨 Building State Space Design Studio...")

    # Parse GNN content for design studio
    design_data = _parse_gnn_for_design(markdown_text)
    initial_state_spaces = design_data.get("state_spaces") or [
        ["D", "3", "Prior over hidden states"],
        ["s", "3", "Hidden state"],
        ["A", "3,3", "Likelihood matrix"],
        ["o", "3", "Observation"],
    ]
    initial_connections = design_data.get("connections_text") or "D>s\ns-A\nA-o"
    parameters = design_data.get("parameters", {})

    demo: gr.Blocks = gr.Blocks(
        title="State Space Design Studio", theme=gr.themes.Base()
    )
    with demo:
        gr.Markdown("# 🎨 State Space Design Studio")
        gr.Markdown(
            "**Low-dependency visual design experience for Active Inference models**"
        )

        with gr.Tabs():
            # Tab 1: State Space Designer
            with gr.TabItem("🏗️ State Space"):
                gr.Markdown("### Visual State Space Architecture")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Current State Spaces:**")
                        state_spaces = gr.Dataframe(
                            value=initial_state_spaces,
                            headers=["Variable", "Dimensions", "Description"],
                            label="State Space Variables",
                        )

                        with gr.Row():
                            add_variable_btn = gr.Button(
                                "➕ Add Variable", variant="primary"
                            )
                            remove_variable_btn = gr.Button(
                                "➖ Remove Last Variable", variant="secondary"
                            )

                    with gr.Column():
                        gr.Markdown("**Visual Designer:**")
                        visual_designer = gr.HTML(
                            value=_generate_visual_designer_html(initial_state_spaces)
                        )

            # Tab 2: Ontology Editor
            with gr.TabItem("📚 Ontology"):
                gr.Markdown("### Active Inference Ontology Terms")

                ontology_editor = gr.Dataframe(
                    value=_format_ontology_data(design_data.get("ontology", {})),
                    headers=["Variable", "Ontology Term", "Description"],
                    label="Ontology Mappings",
                )

                with gr.Row():
                    ontology_variable = gr.Textbox(label="Variable", value="A")
                    ontology_term = gr.Textbox(
                        label="Ontology Term", value="LikelihoodMatrix"
                    )
                    add_mapping_btn = gr.Button("Add Mapping", variant="primary")

            # Tab 3: Connections
            with gr.TabItem("🔗 Connections"):
                gr.Markdown("### Model Connections Graph")

                connections_text = gr.Textbox(
                    value=initial_connections,
                    label="Connections (one per line)",
                    lines=8,
                )

                connections_graph = gr.HTML(
                    value=_generate_connections_html(initial_connections)
                )
                connections_status = gr.Textbox(
                    label="Connection Status", interactive=False
                )

                with gr.Row():
                    validate_connections_btn = gr.Button(
                        "✓ Validate", variant="primary"
                    )
                    auto_layout_btn = gr.Button("🔄 Auto Layout", variant="secondary")

            # Tab 4: Parameters
            with gr.TabItem("⚙️ Parameters"):
                gr.Markdown("### Model Parameters")

                with gr.Row():
                    with gr.Column():
                        num_states = gr.Slider(
                            1,
                            10,
                            value=_bounded_int(
                                parameters.get("num_hidden_states"), 3, 1, 10
                            ),
                            label="Hidden States",
                        )
                        num_obs = gr.Slider(
                            1,
                            10,
                            value=_bounded_int(parameters.get("num_obs"), 3, 1, 10),
                            label="Observations",
                        )
                        num_actions = gr.Slider(
                            1,
                            10,
                            value=_bounded_int(parameters.get("num_actions"), 3, 1, 10),
                            label="Actions",
                        )

                    with gr.Column():
                        planning_horizon = gr.Slider(
                            1,
                            5,
                            value=_bounded_int(
                                parameters.get("planning_horizon"), 1, 1, 5
                            ),
                            label="Planning Horizon",
                        )
                        time_horizon = gr.Dropdown(
                            ["Bounded", "Unbounded"],
                            value=_time_horizon_value(
                                parameters.get("time_horizon", "Unbounded")
                            ),
                            label="Time Horizon",
                        )

        # Export Section
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                export_btn = gr.Button(
                    "💾 Export GNN Model", variant="primary", size="lg"
                )
                export_status = gr.Textbox(label="Export Status", lines=2)

            with gr.Column():
                preview_btn = gr.Button("👁️ Preview Model", variant="secondary")
                model_preview = gr.Code(language="markdown", label="GNN Preview")

        # Event Handlers
        def export_design(
            spaces: Any,
            ontology: Any,
            connections: Any,
            states: Any,
            obs: Any,
            actions: Any,
            horizon: Any,
            time_h: Any,
        ) -> Any:
            """Export current design to GNN format"""
            try:
                gnn_content = _generate_gnn_from_design(
                    spaces, ontology, connections, states, obs, actions, horizon, time_h
                )

                export_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=export_path.parent, delete=False
                ) as tmp_f:
                    tmp_f.write(gnn_content)
                os.replace(tmp_f.name, str(export_path))
                return f"✅ Model exported to {export_path.name}"

            except Exception as e:
                logger.error(f"Export failed: {e}")
                return f"❌ Export failed: {str(e)}"

        def preview_design(
            spaces: Any,
            ontology: Any,
            connections: Any,
            states: Any,
            obs: Any,
            actions: Any,
            horizon: Any,
            time_h: Any,
        ) -> Any:
            """Preview the current design as GNN"""
            try:
                return _generate_gnn_from_design(
                    spaces, ontology, connections, states, obs, actions, horizon, time_h
                )
            except Exception as e:
                return f"Error generating preview: {e}"

        # Wire up event handlers
        add_variable_btn.click(
            _add_state_space_row,
            inputs=[state_spaces],
            outputs=[state_spaces],
        )
        remove_variable_btn.click(
            _remove_last_state_space_row,
            inputs=[state_spaces],
            outputs=[state_spaces],
        )
        state_spaces.change(
            _generate_visual_designer_html,
            inputs=[state_spaces],
            outputs=[visual_designer],
        )
        add_mapping_btn.click(
            _add_ontology_mapping,
            inputs=[ontology_editor, ontology_variable, ontology_term],
            outputs=[ontology_editor],
        )
        validate_connections_btn.click(
            _validate_connections,
            inputs=[connections_text],
            outputs=[connections_status],
        )
        auto_layout_btn.click(
            _generate_connections_html,
            inputs=[connections_text],
            outputs=[connections_graph],
        )
        connections_text.change(
            _generate_connections_html,
            inputs=[connections_text],
            outputs=[connections_graph],
        )
        export_btn.click(
            export_design,
            inputs=[
                state_spaces,
                ontology_editor,
                connections_text,
                num_states,
                num_obs,
                num_actions,
                planning_horizon,
                time_horizon,
            ],
            outputs=[export_status],
        )

        preview_btn.click(
            preview_design,
            inputs=[
                state_spaces,
                ontology_editor,
                connections_text,
                num_states,
                num_obs,
                num_actions,
                planning_horizon,
                time_horizon,
            ],
            outputs=[model_preview],
        )

    logger.info("✅ State Space Design Studio built successfully")
    return demo


def _table_rows(value: Any) -> List[List[Any]]:
    """Normalize Gradio/Pandas/list table values into mutable rows."""
    if value is None:
        return []
    raw_rows = getattr(value, "values", value)
    if hasattr(raw_rows, "tolist"):
        raw_rows = raw_rows.tolist()
    if not isinstance(raw_rows, (list, tuple)):
        return []

    rows: List[List[Any]] = []
    for row in raw_rows:
        if isinstance(row, (list, tuple)):
            rows.append(list(row))
    return rows


def _add_state_space_row(spaces: Any) -> List[List[Any]]:
    """Append a unique editable state-space row."""
    rows = _table_rows(spaces)
    existing = {str(row[0]).strip() for row in rows if row}
    index = 1
    while f"x{index}" in existing:
        index += 1
    rows.append([f"x{index}", "1", "New state-space variable"])
    return rows


def _remove_last_state_space_row(spaces: Any) -> List[List[Any]]:
    """Remove the final state-space row without crashing on an empty table."""
    rows = _table_rows(spaces)
    return rows[:-1] if rows else []


_ONTOLOGY_DESCRIPTIONS = {
    "LikelihoodMatrix": "Maps states to observations",
    "TransitionMatrix": "State transitions given actions",
    "LogPreferenceVector": "Preferences over observations",
    "PriorOverHiddenStates": "Initial state beliefs",
}


def _add_ontology_mapping(ontology: Any, variable: Any, term: Any) -> List[List[Any]]:
    """Add or replace one ontology mapping from GUI text inputs."""
    rows = _table_rows(ontology)
    variable_name = str(variable or "").strip()
    ontology_term = str(term or "").strip()
    if not variable_name or not ontology_term:
        return rows

    new_row = [
        variable_name,
        ontology_term,
        _ONTOLOGY_DESCRIPTIONS.get(ontology_term, "Active Inference component"),
    ]
    for index, row in enumerate(rows):
        if row and str(row[0]).strip() == variable_name:
            rows[index] = new_row
            break
    else:
        rows.append(new_row)
    return rows


_CONNECTION_RE = re.compile(r"^\s*([^\s><*~\-]+)\s*([>\-*~])\s*([^\s><*~\-]+)\s*$")
_VARIABLE_RE = re.compile(r"^[^\s\[\],=#><*~\-]+$")
_DIMENSION_RE = re.compile(r"^[^\s,\[\]=#]+$")


def _normalize_dimensions(value: Any) -> str:
    """Strip embedded metadata and validate a comma-separated dimension list."""
    raw_parts = [part.strip() for part in str(value or "").split(",")]
    dimensions = [part for part in raw_parts if part and "=" not in part]
    if not dimensions or any(
        _DIMENSION_RE.fullmatch(part) is None for part in dimensions
    ):
        raise ValueError(f"Invalid dimensions: {value!r}")
    return ",".join(dimensions)


def _single_line(value: Any) -> str:
    """Collapse user-authored labels so they cannot inject GNN sections."""
    return " ".join(str(value or "").split())


def _connection_records(
    connections: Any,
) -> tuple[List[tuple[str, str, str]], List[str]]:
    """Parse connection text into records plus line-oriented diagnostics."""
    records: List[tuple[str, str, str]] = []
    errors: List[str] = []
    for line_number, raw_line in enumerate(str(connections or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _CONNECTION_RE.match(line)
        if match is None:
            errors.append(f"Line {line_number}: invalid connection {line!r}")
            continue
        records.append((match.group(1), match.group(2), match.group(3)))
    if not records and not errors:
        errors.append("Add at least one connection")
    return records, errors


def _validate_connections(connections: Any) -> str:
    """Return a user-facing validation result for connection text."""
    records, errors = _connection_records(connections)
    if errors:
        return "❌ " + "; ".join(errors)
    duplicates = len(records) - len(set(records))
    if duplicates:
        return f"⚠️ {len(records)} valid connections; {duplicates} duplicate(s)"
    return f"✅ {len(records)} valid connection(s)"


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """Coerce a parsed parameter into a bounded integer slider value."""
    try:
        parsed = int(float(str(value)))
    except (OverflowError, TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _time_horizon_value(value: Any) -> str:
    """Normalize a parsed time-horizon value to a dropdown option."""
    return "Bounded" if str(value).strip().lower() == "bounded" else "Unbounded"


def _parse_gnn_for_design(gnn_content: str) -> Dict[str, Any]:
    """Parse GNN content for design studio"""

    design_data: dict[str, Any] = {
        "state_spaces": [],
        "ontology": {},
        "connections_text": "",
        "parameters": {},
        "parse_errors": [],
    }

    lines = gnn_content.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("## "):
            current_section = line[3:]
            continue

        if current_section == "StateSpaceBlock":
            if "[" in line and "]" in line and not line.startswith("#"):
                var_name = line.split("[")[0].strip()
                try:
                    dimensions = _normalize_dimensions(
                        line.split("[")[1].split("]")[0].strip()
                    )
                except ValueError as exc:
                    design_data["parse_errors"].append(str(exc))
                    continue
                desc = line.split("#")[1].strip() if "#" in line else ""
                design_data["state_spaces"].append([var_name, dimensions, desc])

        elif current_section == "ActInfOntologyAnnotation":
            if "=" in line and not line.startswith("#"):
                var, concept = line.split("=", 1)
                variable = var.strip()
                term = concept.split("#", 1)[0].strip()
                if variable and term:
                    design_data["ontology"][variable] = term

        elif current_section == "Connections":
            if line and not line.startswith("#"):
                if design_data["connections_text"]:
                    design_data["connections_text"] += "\n"
                design_data["connections_text"] += line

        elif current_section == "ModelParameters":
            if ":" in line and not line.startswith("#"):
                parameter, value = line.split(":", 1)
                parameter = parameter.strip()
                value = value.split("#", 1)[0].strip()
                if parameter and value:
                    design_data["parameters"][parameter] = value

    return design_data


def _format_ontology_data(ontology_dict: Dict[str, str]) -> List[List[str]]:
    """Format ontology data for dataframe"""
    if not ontology_dict:
        return [["A", "LikelihoodMatrix", "Maps states to observations"]]

    formatted: list[Any] = []
    for var, term in ontology_dict.items():
        desc = _ONTOLOGY_DESCRIPTIONS.get(term, "Active Inference component")
        formatted.append([var, term, desc])

    return formatted


def _generate_visual_designer_html(spaces: Any = None) -> str:
    """Generate a live HTML summary of the editable state-space table."""
    rows = _table_rows(spaces)
    if not rows:
        cards = '<em style="color:#666">No state-space variables defined.</em>'
    else:
        rendered_cards: List[str] = []
        for row in rows:
            variable = html.escape(str(row[0] if row else "Unnamed"))
            dimensions = html.escape(str(row[1] if len(row) > 1 else "?"))
            description = html.escape(str(row[2] if len(row) > 2 else ""))
            rendered_cards.append(
                '<div style="padding:12px;background:#e3f2fd;border:1px solid #90caf9;'
                'border-radius:6px;text-align:center;min-width:110px">'
                f"<strong>{variable}</strong><br><span>{dimensions}</span>"
                f'<br><small style="color:#555">{description}</small></div>'
            )
        cards = "".join(rendered_cards)

    return (
        '<div style="border:1px solid #ddd;padding:20px;border-radius:8px;'
        'background:#f9f9f9"><h4>🎯 Visual State Space</h4>'
        '<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:12px;'
        f'margin:16px 0">{cards}</div></div>'
    )


def _generate_connections_html(connections: Any = None) -> str:
    """Generate a safe SVG auto-layout for the current connections."""
    records, errors = _connection_records(connections)
    if errors:
        message = html.escape("; ".join(errors))
        return (
            '<div style="border:1px solid #ef9a9a;padding:16px;border-radius:8px;'
            f'background:#ffebee;color:#b71c1c">{message}</div>'
        )

    nodes: List[str] = []
    for source, _operator, target in records:
        for node in (source, target):
            if node not in nodes:
                nodes.append(node)

    width, height = 560, 340
    center_x, center_y = width / 2, height / 2
    radius = min(130.0, 45.0 * max(2, len(nodes)))
    positions: Dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        angle = -math.pi / 2 + (2 * math.pi * index / max(1, len(nodes)))
        positions[node] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )

    edge_svg: List[str] = []
    for source, operator, target in records:
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        midpoint_x = (source_x + target_x) / 2
        midpoint_y = (source_y + target_y) / 2
        edge_svg.append(
            f'<line x1="{source_x:.1f}" y1="{source_y:.1f}" '
            f'x2="{target_x:.1f}" y2="{target_y:.1f}" '
            'stroke="#607d8b" stroke-width="2" marker-end="url(#gnn-arrow)"/>'
            f'<text x="{midpoint_x:.1f}" y="{midpoint_y - 5:.1f}" '
            f'text-anchor="middle" font-size="13">{html.escape(operator)}</text>'
        )

    node_svg: List[str] = []
    for node, (x_pos, y_pos) in positions.items():
        node_svg.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="24" '
            'fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>'
            f'<text x="{x_pos:.1f}" y="{y_pos + 4:.1f}" text-anchor="middle" '
            f'font-size="12">{html.escape(node)}</text>'
        )

    return (
        '<div style="border:1px solid #ddd;padding:16px;border-radius:8px;'
        'background:#f9f9f9"><h4>🔗 Connection Graph</h4>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'style="max-height:360px;border:1px solid #ccc;background:white">'
        '<defs><marker id="gnn-arrow" markerWidth="10" markerHeight="7" '
        'refX="29" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" '
        'fill="#607d8b"/></marker></defs>'
        + "".join(edge_svg)
        + "".join(node_svg)
        + f"</svg><small>{len(records)} connection(s), {len(nodes)} variable(s)</small></div>"
    )


def _generate_gnn_from_design(
    spaces: Any,
    ontology: Any,
    connections: Any,
    states: Any,
    obs: Any,
    actions: Any,
    horizon: Any,
    time_h: Any,
) -> Any:
    """Generate GNN content from design studio inputs"""

    gnn_lines: list[Any] = [
        "# GNN: State Space Design Studio Export",
        "# Generated by Design Studio GUI",
        "",
        "## ModelName",
        "Active Inference Model - Design Studio Export",
        "",
        "## StateSpaceBlock",
    ]

    # Add state spaces
    variable_names: set[str] = set()
    for row in _table_rows(spaces):
        if len(row) < 2:
            continue
        variable = str(row[0] or "").strip()
        dimensions = _normalize_dimensions(row[1])
        description = _single_line(row[2]) if len(row) > 2 else ""
        if _VARIABLE_RE.fullmatch(variable) is None:
            raise ValueError(f"Invalid state-space variable name: {variable!r}")
        if variable in variable_names:
            raise ValueError(f"Duplicate state-space variable: {variable}")
        variable_names.add(variable)
        comment = f"   # {description}" if description else ""
        gnn_lines.append(f"{variable}[{dimensions},type=float]{comment}")

    if not variable_names:
        raise ValueError("Add at least one state-space variable")

    gnn_lines.extend(["", "## Connections"])

    # Add connections
    connection_records, connection_errors = _connection_records(connections)
    if connection_errors:
        raise ValueError("; ".join(connection_errors))
    for source, operator, target in connection_records:
        missing = [name for name in (source, target) if name not in variable_names]
        if missing:
            raise ValueError(
                f"Connection {source}{operator}{target} references undefined "
                f"variable(s): {', '.join(missing)}"
            )
        gnn_lines.append(f"{source}{operator}{target}")

    gnn_lines.extend(["", "## ActInfOntologyAnnotation"])

    # Add ontology
    for row in _table_rows(ontology):
        if len(row) < 2:
            continue
        variable = str(row[0] or "").strip()
        term = str(row[1] or "").strip()
        if variable and term:
            if variable not in variable_names:
                raise ValueError(
                    f"Ontology mapping references undefined variable: {variable}"
                )
            if _VARIABLE_RE.fullmatch(term) is None:
                raise ValueError(f"Invalid ontology term: {term!r}")
            gnn_lines.append(f"{variable}={term}")

    gnn_lines.extend(
        [
            "",
            "## ModelParameters",
            f"num_hidden_states: {_bounded_int(states, 3, 1, 10)}",
            f"num_obs: {_bounded_int(obs, 3, 1, 10)}",
            f"num_actions: {_bounded_int(actions, 3, 1, 10)}",
            f"planning_horizon: {_bounded_int(horizon, 1, 1, 5)}",
            f"time_horizon: {_time_horizon_value(time_h)}",
            "",
            "## Footer",
            "Generated by State Space Design Studio - Low Dependency GUI",
        ]
    )

    return "\n".join(gnn_lines)
