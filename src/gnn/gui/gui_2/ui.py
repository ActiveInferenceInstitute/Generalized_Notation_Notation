"""
UI for GUI 2: Visual Matrix Editor

Provides an interactive tabbed interface for editing matrix structures
and real-time GNN markdown generation.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from pathlib import Path
from typing import Any, List, cast

try:
    import gradio as gr
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    gr = cast(Any, None)
    np = cast(Any, None)
    go = cast(Any, None)
    px = cast(Any, None)

from .matrix_editor import (
    create_matrix_from_gnn,
    get_pomdp_template,
    update_gnn_from_matrix,
    validate_visual_matrix_dimensions,
)

_MAX_EDITOR_DIMENSION = 64
DEBOUNCE_SECONDS: float = 2.0


def _table_rows(value: Any) -> List[List[Any]]:
    """Normalize Gradio, pandas, NumPy, and ordinary table values."""
    raw_rows = getattr(value, "values", value)
    if hasattr(raw_rows, "tolist"):
        raw_rows = raw_rows.tolist()
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, (list, tuple)):
        return []
    rows: List[List[Any]] = []
    for row in raw_rows:
        if isinstance(row, (list, tuple)):
            rows.append(list(row))
        else:
            rows.append([row])
    return rows


def _finite_table(value: Any, name: str) -> List[List[float]]:
    """Return a rectangular finite numeric table or a precise validation error."""
    rows = _table_rows(value)
    if not rows or not rows[0]:
        raise ValueError(f"{name} cannot be empty")
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError(f"{name} must be rectangular")
    converted: List[List[float]] = []
    for row in rows:
        converted_row: List[float] = []
        for raw_value in row:
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} contains a non-numeric value: {raw_value!r}"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} contains a non-finite value")
            converted_row.append(number)
        converted.append(converted_row)
    return converted


def _initial_matrix_state(markdown_text: str) -> dict[str, Any]:
    """Build editable state from a real GNN model, with a valid empty fallback."""
    visual_data = create_matrix_from_gnn(markdown_text)
    matrices = visual_data.get("visual_matrices", {})
    if not any(
        isinstance(matrix, dict) and matrix.get("declared")
        for matrix in matrices.values()
    ) or validate_visual_matrix_dimensions(visual_data):
        visual_data = create_matrix_from_gnn(get_pomdp_template())
        matrices = visual_data["visual_matrices"]

    state: dict[str, Any] = copy.deepcopy(matrices)
    b_matrix = state.get("B")
    if not isinstance(b_matrix, dict):
        b_matrix = create_matrix_from_gnn(get_pomdp_template())["visual_matrices"]["B"]
    if b_matrix.get("type") == "matrix":
        b_matrix = {
            **b_matrix,
            "type": "tensor",
            "source_type": "matrix",
            "depth": 1,
            "current_slice": 0,
            "values": [copy.deepcopy(b_matrix.get("values", []))],
        }
    else:
        b_matrix["source_type"] = "tensor"
        b_matrix["current_slice"] = 0
    state["B"] = b_matrix
    state["_state_spaces"] = copy.deepcopy(visual_data.get("state_spaces", {}))
    return state


def _coerce_editor_state(current_state: Any) -> dict[str, Any]:
    """Return an isolated valid editor state even for stale callback payloads."""
    if not isinstance(current_state, dict):
        return _initial_matrix_state(get_pomdp_template())

    expected_dimensions = {
        "A": ("rows", "cols"),
        "B": ("depth", "rows", "cols"),
        "C": ("size",),
        "D": ("size",),
    }
    for name, dimensions in expected_dimensions.items():
        matrix = current_state.get(name)
        if not isinstance(matrix, dict) or not isinstance(matrix.get("values"), list):
            return _initial_matrix_state(get_pomdp_template())
        for dimension in dimensions:
            try:
                size = int(matrix[dimension])
            except (KeyError, OverflowError, TypeError, ValueError):
                return _initial_matrix_state(get_pomdp_template())
            if size < 1:
                return _initial_matrix_state(get_pomdp_template())
    return copy.deepcopy(current_state)


def _bounded_index(value: Any, upper_bound: int) -> int:
    try:
        parsed = int(float(value))
    except (OverflowError, TypeError, ValueError):
        parsed = 0
    return max(0, min(max(0, upper_bound), parsed))


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(float(value))
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_delta(value: Any) -> int:
    """Coerce callback deltas without allowing oversized UI mutations."""
    try:
        parsed = int(float(value))
    except (OverflowError, TypeError, ValueError):
        return 0
    return max(-_MAX_EDITOR_DIMENSION, min(_MAX_EDITOR_DIMENSION, parsed))


def _editor_visual_data(state: dict[str, Any]) -> dict[str, Any]:
    """Convert the slice-oriented UI state back to matrix-editor model data."""
    matrices = {
        name: copy.deepcopy(state[name])
        for name in ("A", "B", "C", "D")
        if isinstance(state.get(name), dict)
    }
    b_matrix = matrices.get("B")
    if (
        b_matrix
        and b_matrix.get("source_type") == "matrix"
        and b_matrix.get("depth") == 1
    ):
        b_matrix["type"] = "matrix"
        b_matrix["values"] = copy.deepcopy(b_matrix.get("values", [[]])[0])
        b_matrix.pop("depth", None)
        b_matrix.pop("current_slice", None)
    if b_matrix:
        b_matrix.pop("source_type", None)
    return {
        "visual_matrices": matrices,
        "state_spaces": copy.deepcopy(state.get("_state_spaces", {})),
    }


def _state_from_tables(
    current_state: Any,
    a_data: Any,
    b_data: Any,
    c_data: Any,
    d_data: Any,
    b_slice: Any = 0,
) -> dict[str, Any]:
    """Synchronize all editable tables into state without losing B slices."""
    state = _coerce_editor_state(current_state)
    a_values = _finite_table(a_data, "A matrix")
    b_values = _finite_table(b_data, "B matrix")
    c_rows = _finite_table(c_data, "C vector")
    d_rows = _finite_table(d_data, "D vector")
    if any(len(row) != 1 for row in c_rows):
        raise ValueError("C vector must have exactly one column")
    if any(len(row) != 1 for row in d_rows):
        raise ValueError("D vector must have exactly one column")

    state["A"].update(rows=len(a_values), cols=len(a_values[0]), values=a_values)
    state["C"].update(size=len(c_rows), values=[row[0] for row in c_rows])
    state["D"].update(size=len(d_rows), values=[row[0] for row in d_rows])

    b_info = state["B"]
    depth = _positive_int(b_info.get("depth", 1))
    slice_index = _bounded_index(b_slice, depth - 1)
    slices = b_info.get("values")
    if not isinstance(slices, list) or len(slices) != depth:
        slices = [copy.deepcopy(b_values) for _ in range(depth)]
    else:
        slices = copy.deepcopy(slices)
        slices[slice_index] = b_values
    b_info.update(
        rows=len(b_values),
        cols=len(b_values[0]),
        current_slice=slice_index,
        values=slices,
    )
    return state


def _generate_editor_gnn(
    current_state: Any,
    a_data: Any,
    b_data: Any,
    c_data: Any,
    d_data: Any,
    b_slice: Any,
    template: str,
) -> tuple[dict[str, Any], str]:
    """Total model-generation core used by manual and auto-update callbacks."""
    state = _state_from_tables(current_state, a_data, b_data, c_data, d_data, b_slice)
    visual_data = _editor_visual_data(state)
    errors = validate_visual_matrix_dimensions(visual_data)
    if errors:
        raise ValueError("; ".join(errors))
    return state, update_gnn_from_matrix(visual_data, template)


def _validate_editor_tables(
    current_state: Any,
    a_data: Any,
    b_data: Any,
    c_data: Any,
    d_data: Any,
    b_slice: Any = 0,
) -> str:
    """Return a user-facing result for every possible callback payload."""
    try:
        state = _state_from_tables(
            current_state, a_data, b_data, c_data, d_data, b_slice
        )
        errors = validate_visual_matrix_dimensions(_editor_visual_data(state))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        return "❌ **Validation Errors:**\n" + "\n".join(
            f"- {error}" for error in errors
        )
    return "✅ **Validation Passed:** All matrix dimensions and values are consistent"


def _select_b_slice(
    current_state: Any, current_values: Any, requested_slice: Any
) -> tuple[dict[str, Any], List[List[float]]]:
    """Persist the visible B slice and return the requested slice safely."""
    state = _coerce_editor_state(current_state)
    b_info = state["B"]
    depth = _positive_int(b_info.get("depth", 1))
    old_index = _bounded_index(b_info.get("current_slice", 0), depth - 1)
    try:
        visible_values = _finite_table(current_values, "B matrix")
        slices = copy.deepcopy(b_info.get("values", []))
        if len(slices) == depth:
            slices[old_index] = visible_values
            b_info["values"] = slices
    except (TypeError, ValueError):
        pass
    new_index = _bounded_index(requested_slice, depth - 1)
    b_info["current_slice"] = new_index
    slices = b_info.get("values", [])
    if not isinstance(slices, list):
        slices = []
    fallback = [
        [0.0 for _ in range(_positive_int(b_info.get("cols")))]
        for _ in range(_positive_int(b_info.get("rows")))
    ]
    normalized_slices = [
        copy.deepcopy(slice_data)
        if isinstance(slice_data, list)
        else copy.deepcopy(fallback)
        for slice_data in slices[:depth]
    ]
    while len(normalized_slices) < depth:
        normalized_slices.append(copy.deepcopy(fallback))
    b_info["values"] = normalized_slices
    slices = normalized_slices
    return state, copy.deepcopy(slices[new_index])


def build_visual_gui(
    markdown_text: str, export_path: Path, logger: logging.Logger
) -> "gr.Blocks":
    """Build the enhanced visual matrix editor interface with real-time heatmaps and dimension controls"""
    if gr is None:
        raise RuntimeError("Gradio not available")

    # Initialize from the supplied model. Direct calls with blank or unusual
    # content still receive a valid, editable POMDP rather than a crashing UI.
    supplied_data = create_matrix_from_gnn(markdown_text)
    if not any(
        matrix.get("declared")
        for matrix in supplied_data.get("visual_matrices", {}).values()
    ):
        markdown_text = get_pomdp_template()
    initial_state = _initial_matrix_state(markdown_text)
    a_initial = initial_state["A"]
    b_initial = initial_state["B"]
    c_initial = initial_state["C"]
    d_initial = initial_state["D"]

    demo: gr.Blocks = gr.Blocks(title="GNN Visual Matrix Editor", theme=gr.themes.Soft())
    with demo:
        gr.Markdown("# 🎯 GNN Visual Matrix Editor")
        gr.Markdown(
            "🚀 **Interactive matrix editing with real-time heatmap visualization and dimension controls**"
        )

        # State variables for matrix dimensions (using Gradio state)
        matrix_state = gr.State(initial_state)

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 📊 **Interactive Visual Matrix Editor**")

                with gr.Tab("🔵 Matrix A (Likelihood)"):
                    with gr.Row():
                        gr.Markdown("#### **Current Size Display**")

                    # Matrix size display and controls
                    with gr.Row():
                        with gr.Column(scale=1):
                            a_size_display = gr.Markdown(
                                f"**Matrix A**: `{a_initial['rows']}×{a_initial['cols']}` "
                                "(Observations × States)"
                            )

                            # Dimension controls
                            with gr.Row():
                                gr.Markdown("**Rows (Obs):**")
                                a_rows_minus = gr.Button("➖", size="sm")
                                a_rows_display = gr.Markdown(
                                    f"**{a_initial['rows']}**",
                                    elem_classes=["dimension-display"],
                                )
                                a_rows_plus = gr.Button("➕", size="sm")

                            with gr.Row():
                                gr.Markdown("**Cols (States):**")
                                a_cols_minus = gr.Button("➖", size="sm")
                                a_cols_display = gr.Markdown(
                                    f"**{a_initial['cols']}**",
                                    elem_classes=["dimension-display"],
                                )
                                a_cols_plus = gr.Button("➕", size="sm")

                    # Large heatmap visualization
                    matrix_a_plot = gr.Plot(
                        label="A Matrix Heatmap Visualization", scale=2
                    )

                    # Matrix values editor
                    a_values = gr.Dataframe(
                        headers=[
                            f"State_{index}" for index in range(a_initial["cols"])
                        ],
                        value=a_initial["values"],
                        interactive=True,
                        label="A Matrix Values - Edit cells directly",
                        row_count=a_initial["rows"],
                    )

                with gr.Tab("🟠 Matrix B (Transitions)"):
                    with gr.Row():
                        gr.Markdown("#### **Current Size Display**")

                    # Matrix size display and controls
                    with gr.Row():
                        with gr.Column(scale=1):
                            b_size_display = gr.Markdown(
                                f"**Matrix B**: `{b_initial['rows']}×{b_initial['cols']}×"
                                f"{b_initial['depth']}` (States × States × Actions)"
                            )

                            # Dimension controls
                            with gr.Row():
                                gr.Markdown("**States:**")
                                b_states_minus = gr.Button("➖", size="sm")
                                b_states_display = gr.Markdown(
                                    f"**{b_initial['rows']}**",
                                    elem_classes=["dimension-display"],
                                )
                                b_states_plus = gr.Button("➕", size="sm")

                            with gr.Row():
                                gr.Markdown("**Actions (Depth):**")
                                b_actions_minus = gr.Button("➖", size="sm")
                                b_actions_display = gr.Markdown(
                                    f"**{b_initial['depth']}**",
                                    elem_classes=["dimension-display"],
                                )
                                b_actions_plus = gr.Button("➕", size="sm")

                    # Action slice selector
                    with gr.Row():
                        b_slice_selector = gr.Slider(
                            0,
                            max(0, b_initial["depth"] - 1),
                            value=0,
                            step=1,
                            label="Action Slice - Select which action transition matrix to view",
                        )

                    # Large heatmap visualization
                    matrix_b_plot = gr.Plot(
                        label="B Matrix Heatmap Visualization (Current Slice)", scale=2
                    )

                    # Matrix values editor
                    b_values = gr.Dataframe(
                        headers=[
                            f"State_{index}" for index in range(b_initial["cols"])
                        ],
                        value=b_initial["values"][0],
                        interactive=True,
                        label="B Matrix Values - Current Action Slice",
                        row_count=b_initial["rows"],
                    )

                with gr.Tab("🔴 Vectors C & D"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### **C Vector (Preferences)**")

                            # Size display and controls for C
                            with gr.Row():
                                c_size_display = gr.Markdown(
                                    f"**Vector C**: `{c_initial['size']}` "
                                    "(Observation Preferences)"
                                )
                                c_size_minus = gr.Button("➖", size="sm")
                                c_size_display_num = gr.Markdown(
                                    f"**{c_initial['size']}**",
                                    elem_classes=["dimension-display"],
                                )
                                c_size_plus = gr.Button("➕", size="sm")

                            c_plot = gr.Plot(label="C Vector Visualization")
                            c_values = gr.Dataframe(
                                headers=["Preference"],
                                value=[[value] for value in c_initial["values"]],
                                interactive=True,
                                label="C Values",
                                row_count=c_initial["size"],
                            )

                        with gr.Column():
                            gr.Markdown("#### **D Vector (Prior)**")

                            # Size display and controls for D
                            with gr.Row():
                                d_size_display = gr.Markdown(
                                    f"**Vector D**: `{d_initial['size']}` (State Prior)"
                                )
                                d_size_minus = gr.Button("➖", size="sm")
                                d_size_display_num = gr.Markdown(
                                    f"**{d_initial['size']}**",
                                    elem_classes=["dimension-display"],
                                )
                                d_size_plus = gr.Button("➕", size="sm")

                            d_plot = gr.Plot(label="D Vector Visualization")
                            d_values = gr.Dataframe(
                                headers=["Prior"],
                                value=[[value] for value in d_initial["values"]],
                                interactive=True,
                                label="D Values",
                                row_count=d_initial["size"],
                            )

                # Control buttons
                with gr.Row():
                    auto_update_checkbox = gr.Checkbox(
                        value=True, label="🔄 Auto-update visualizations"
                    )
                    manual_update_btn = gr.Button("🔄 Manual Update", variant="primary")
                    reset_btn = gr.Button(
                        "🔄 Reset to POMDP Template", variant="secondary"
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 📝 **GNN Markdown Output**")
                gnn_output = gr.Code(
                    value=markdown_text,
                    language="markdown",
                    label="Generated GNN",
                    lines=30,
                )

                with gr.Row():
                    save_btn = gr.Button("💾 Save GNN", variant="primary")
                    validate_btn = gr.Button("✅ Validate", variant="secondary")

                validation_output = gr.Markdown("")
                save_status = gr.Markdown("")

                # Matrix statistics
                gr.Markdown("### 📈 **Matrix Statistics**")
                stats_output = gr.Markdown("Click 'Update' to see matrix statistics")

        # Enhanced Event Handlers for Interactive Matrix Editing

        def create_enhanced_heatmap(
            matrix_data: Any,
            title: str,
            colorscale: str = "Viridis",
            show_values: bool = True,
        ) -> "go.Figure":
            """Create an enhanced matrix heatmap with better visualization"""
            if go is None:
                return None

            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(matrix_data, "values"):
                    # It's a pandas/Gradio DataFrame
                    matrix_list = matrix_data.values.tolist()
                elif isinstance(matrix_data, list):
                    matrix_list = matrix_data
                else:
                    # Try to convert to list
                    matrix_list = list(matrix_data) if matrix_data is not None else []

                # Ensure we have valid data
                if not matrix_list or len(matrix_list) == 0:
                    return go.Figure().add_annotation(text="No data", x=0.5, y=0.5)

                # Convert strings to floats if needed
                cleaned_data: list[Any] = []
                for row in matrix_list:
                    if isinstance(row, list):
                        cleaned_row: list[Any] = []
                        for val in row:
                            try:
                                cleaned_row.append(
                                    float(val) if val is not None else 0.0
                                )
                            except (ValueError, TypeError):
                                cleaned_row.append(0.0)
                        cleaned_data.append(cleaned_row)
                    else:
                        # Single value row
                        try:
                            cleaned_data.append(
                                [float(row) if row is not None else 0.0]
                            )
                        except (ValueError, TypeError):
                            cleaned_data.append([0.0])

                if not cleaned_data or not cleaned_data[0]:
                    return go.Figure().add_annotation(
                        text="No valid data", x=0.5, y=0.5
                    )

                z_data = np.array(cleaned_data) if np is not None else cleaned_data

                fig = go.Figure(
                    data=go.Heatmap(
                        z=z_data,
                        colorscale=colorscale,
                        showscale=True,
                        text=[[f"{val:.3f}" for val in row] for row in cleaned_data]
                        if show_values
                        else None,
                        texttemplate="%{text}" if show_values else None,
                        textfont={"size": 12, "color": "white"},
                        hoverongaps=False,
                        colorbar={"title": "Value"},
                    )
                )

                rows, cols = (
                    len(cleaned_data),
                    len(cleaned_data[0]) if cleaned_data else (0, 0),
                )

                fig.update_layout(
                    title={"text": f"<b>{title}</b><br>Size: {rows}×{cols}", "x": 0.5},
                    width=500,
                    height=400,
                    xaxis_title="<b>Columns</b>",
                    yaxis_title="<b>Rows</b>",
                    font={"size": 12},
                    margin={"l": 80, "r": 80, "t": 100, "b": 80},
                )

                # Add grid lines for better readability
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

                return fig
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Error creating heatmap: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)

        def create_enhanced_vector_plot(
            vector_data: Any, title: str, color: str = "blue"
        ) -> "go.Figure":
            """Create an enhanced vector visualization"""
            if go is None:
                return None

            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(vector_data, "values"):
                    # It's a pandas/Gradio DataFrame
                    vector_list = vector_data.values.tolist()
                elif isinstance(vector_data, list):
                    vector_list = vector_data
                else:
                    # Try to convert to list
                    vector_list = list(vector_data) if vector_data is not None else []

                # Extract values and convert to floats
                values: list[Any] = []
                for item in vector_list:
                    if isinstance(item, list):
                        # Take first element if it's a list
                        try:
                            val = (
                                float(item[0])
                                if len(item) > 0 and item[0] is not None
                                else 0.0
                            )
                        except (ValueError, TypeError):
                            val = 0.0
                        values.append(val)
                    else:
                        # Single value
                        try:
                            val = float(item) if item is not None else 0.0
                        except (ValueError, TypeError):
                            val = 0.0
                        values.append(val)

                if not values:
                    return go.Figure().add_annotation(
                        text="No vector data", x=0.5, y=0.5
                    )

                indices = list(range(len(values)))

                fig = go.Figure()

                # Add bar plot
                fig.add_trace(
                    go.Bar(
                        x=indices,
                        y=values,
                        marker_color=color,
                        text=[f"{val:.3f}" for val in values],
                        textposition="outside",
                        textfont={"size": 11},
                        name=title,
                    )
                )

                # Add line plot for trend
                fig.add_trace(
                    go.Scatter(
                        x=indices,
                        y=values,
                        mode="lines+markers",
                        line={"color": "red", "width": 2},
                        marker={"size": 6},
                        name="Trend",
                        yaxis="y2",
                    )
                )

                fig.update_layout(
                    title={"text": f"<b>{title}</b><br>Size: {len(values)}", "x": 0.5},
                    width=400,
                    height=300,
                    xaxis_title="<b>Index</b>",
                    yaxis_title="<b>Value</b>",
                    font={"size": 11},
                    showlegend=False,
                    margin={"l": 60, "r": 60, "t": 80, "b": 60},
                )

                # Add secondary y-axis for trend line
                fig.update_layout(
                    yaxis2={"overlaying": "y", "side": "right", "showticklabels": False}
                )

                return fig
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Error creating vector plot: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)

        def resize_matrix(
            matrix_data: List[List[float]],
            new_rows: int,
            new_cols: int,
            fill_value: float = 0.0,
        ) -> List[List[float]]:
            """Resize matrix to new dimensions, preserving existing values"""
            if not matrix_data:
                return [[fill_value for _ in range(new_cols)] for _ in range(new_rows)]

            current_rows, current_cols = (
                len(matrix_data),
                len(matrix_data[0]) if matrix_data else 0,
            )
            new_matrix: list[Any] = []

            for i in range(new_rows):
                row: list[Any] = []
                for j in range(new_cols):
                    if i < current_rows and j < current_cols:
                        # Preserve existing value
                        row.append(matrix_data[i][j])
                    elif i < current_rows and j >= current_cols:
                        # New column, use normalized probability
                        row.append(0.1)
                    elif i >= current_rows and j < current_cols:
                        # New row, use identity-like pattern
                        row.append(1.0 if i == j else 0.1)
                    else:
                        # New row and column
                        row.append(fill_value)
                new_matrix.append(row)

            return new_matrix

        def resize_vector(
            vector_data: List, new_size: int, fill_value: float = 0.33
        ) -> List[List[float]]:
            """Resize vector to new size, preserving existing values"""
            current_values = [
                row[0] if isinstance(row, list) else row for row in vector_data
            ]
            current_size = len(current_values)

            if new_size == current_size:
                return [[val] for val in current_values]
            elif new_size > current_size:
                # Add new elements
                new_values = current_values + [fill_value] * (new_size - current_size)
            else:
                # Truncate
                new_values = current_values[:new_size]

            return [[val] for val in new_values]

        def calculate_matrix_stats(matrix_data: Any, name: str) -> str:
            """Calculate and format matrix statistics"""
            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(matrix_data, "values"):
                    # It's a pandas/Gradio DataFrame
                    matrix_list = matrix_data.values.tolist()
                elif isinstance(matrix_data, list):
                    matrix_list = matrix_data
                else:
                    matrix_list = []

                if not matrix_list:
                    return f"**{name}**: No data"

                # Convert to floats and flatten
                flat_values: list[Any] = []
                for row in matrix_list:
                    if isinstance(row, list):
                        for val in row:
                            try:
                                flat_values.append(
                                    float(val) if val is not None else 0.0
                                )
                            except (ValueError, TypeError):
                                flat_values.append(0.0)
                    else:
                        try:
                            flat_values.append(float(row) if row is not None else 0.0)
                        except (ValueError, TypeError):
                            flat_values.append(0.0)

                if not flat_values:
                    return f"**{name}**: No valid data"

                # Calculate shape
                if matrix_list and isinstance(matrix_list[0], list):
                    shape = f"{len(matrix_list)}×{len(matrix_list[0])}"
                else:
                    shape = f"{len(matrix_list)}"

                stats: dict[str, Any] = {
                    "Shape": shape,
                    "Min": f"{min(flat_values):.3f}",
                    "Max": f"{max(flat_values):.3f}",
                    "Mean": f"{sum(flat_values) / len(flat_values):.3f}",
                    "Sum": f"{sum(flat_values):.3f}",
                }

                stats_text = f"**{name}**:\n"
                for key, value in stats.items():
                    stats_text += f"- {key}: `{value}`\n"

                return stats_text
            except (ValueError, TypeError, ZeroDivisionError) as e:
                return f"**{name}**: Error calculating stats - {e}"

        def generate_gnn_from_matrices(
            current_state: Any,
            a_data: Any,
            b_data: Any,
            c_data: Any,
            d_data: Any,
            b_slice: Any,
        ) -> tuple[dict[str, Any], str]:
            """Generate GNN markdown while preserving every transition slice."""
            return _generate_editor_gnn(
                current_state,
                a_data,
                b_data,
                c_data,
                d_data,
                b_slice,
                markdown_text,
            )

        def save_gnn(gnn_text: Any) -> Any:
            """Save GNN markdown to file"""
            try:
                import tempfile

                if not isinstance(gnn_text, str):
                    raise TypeError("generated GNN content must be text")
                export_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".tmp",
                    dir=export_path.parent,
                    delete=False,
                ) as tmp_f:
                    tmp_f.write(gnn_text)
                os.replace(tmp_f.name, str(export_path))
                return f"✅ Saved to {export_path.name}"
            except (OSError, TypeError, ValueError) as e:
                return f"❌ Save failed: {e}"

        def validate_editor_matrices(
            current_state: Any,
            a_data: Any,
            b_data: Any,
            c_data: Any,
            d_data: Any,
            b_slice: Any,
        ) -> Any:
            """Validate matrix dimensions and consistency"""
            return _validate_editor_tables(
                current_state, a_data, b_data, c_data, d_data, b_slice
            )

        # === ENHANCED INTERACTIVE EVENT HANDLERS ===

        # Matrix A dimension control handlers
        def update_a_dimensions(
            current_state: Any,
            visible_values: Any = None,
            delta_rows: Any = 0,
            delta_cols: Any = 0,
        ) -> Any:
            """Update Matrix A dimensions and resize data"""
            state = _coerce_editor_state(current_state)
            a_info = state["A"]

            new_rows = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(a_info.get("rows", 1)) + _coerce_delta(delta_rows),
                ),
            )
            new_cols = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(a_info.get("cols", 1)) + _coerce_delta(delta_cols),
                ),
            )

            if new_rows != a_info["rows"] or new_cols != a_info["cols"]:
                # Resize matrix
                try:
                    current_values = _finite_table(
                        visible_values
                        if visible_values is not None
                        else a_info.get("values"),
                        "A matrix",
                    )
                except (TypeError, ValueError):
                    current_values = []
                new_values = resize_matrix(current_values, new_rows, new_cols, 0.1)
                a_info.update(rows=new_rows, cols=new_cols, values=new_values)

                # Create new headers
                headers = [f"State_{i}" for i in range(new_cols)]

                return (
                    state,
                    f"**Matrix A**: `{new_rows}×{new_cols}` (Observations × States)",
                    f"**{new_rows}**",
                    f"**{new_cols}**",
                    gr.Dataframe(
                        value=new_values,
                        headers=headers,
                        interactive=True,
                        row_count=new_rows,
                    ),
                )

            return (state, gr.update(), gr.update(), gr.update(), gr.update())

        # Matrix B dimension control handlers
        def update_b_dimensions(
            current_state: Any,
            visible_values: Any = None,
            visible_slice: Any = None,
            delta_states: Any = 0,
            delta_actions: Any = 0,
        ) -> Any:
            """Update Matrix B dimensions and resize data"""
            state = _coerce_editor_state(current_state)
            if visible_values is not None:
                state, _ = _select_b_slice(state, visible_values, visible_slice)
            b_info = state["B"]

            new_states = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(b_info.get("rows", 1)) + _coerce_delta(delta_states),
                ),
            )
            new_actions = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(b_info.get("depth", 1))
                    + _coerce_delta(delta_actions),
                ),
            )

            if new_states != b_info["rows"] or new_actions != b_info["depth"]:
                current_slice = _bounded_index(
                    b_info.get("current_slice", 0), new_actions - 1
                )
                old_slices = b_info.get("values", [])
                resized_slices: List[List[List[float]]] = []
                if isinstance(old_slices, list):
                    for old_slice in old_slices[:new_actions]:
                        try:
                            clean_slice = _finite_table(old_slice, "B matrix")
                        except (TypeError, ValueError):
                            clean_slice = []
                        resized_slices.append(
                            resize_matrix(clean_slice, new_states, new_states, 0.0)
                        )
                while len(resized_slices) < new_actions:
                    resized_slices.append(
                        [
                            [
                                1.0 if row == column else 0.0
                                for column in range(new_states)
                            ]
                            for row in range(new_states)
                        ]
                    )
                b_info.update(
                    depth=new_actions,
                    rows=new_states,
                    cols=new_states,
                    current_slice=current_slice,
                    values=resized_slices,
                )
                if new_actions > 1:
                    b_info["source_type"] = "tensor"
                new_slice_values = resized_slices[current_slice]

                headers = [f"State_{i}" for i in range(new_states)]

                return (
                    state,
                    f"**Matrix B**: `{new_states}×{new_states}×{new_actions}` (States × States × Actions)",
                    f"**{new_states}**",
                    f"**{new_actions}**",
                    gr.Slider(maximum=new_actions - 1, value=current_slice),
                    gr.Dataframe(
                        value=new_slice_values,
                        headers=headers,
                        interactive=True,
                        row_count=new_states,
                    ),
                )

            return (
                state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        # Vector size control handlers
        def update_c_size(
            current_state: Any, visible_values: Any = None, delta_size: Any = 0
        ) -> Any:
            """Update C vector size"""
            state = _coerce_editor_state(current_state)
            c_info = state["C"]
            new_size = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(c_info.get("size", 1)) + _coerce_delta(delta_size),
                ),
            )

            if new_size != c_info["size"]:
                raw_values = c_info.get("values", [])
                if visible_values is not None:
                    try:
                        visible_rows = _finite_table(visible_values, "C vector")
                        if all(len(row) == 1 for row in visible_rows):
                            raw_values = [row[0] for row in visible_rows]
                    except (TypeError, ValueError):
                        pass
                if not isinstance(raw_values, list):
                    raw_values = []
                new_values = resize_vector(
                    [[value] for value in raw_values], new_size, 0.1
                )
                c_info.update(size=new_size, values=[row[0] for row in new_values])

                return (
                    state,
                    f"**Vector C**: `{new_size}` (Observation Preferences)",
                    f"**{new_size}**",
                    gr.Dataframe(
                        value=new_values,
                        headers=["Preference"],
                        interactive=True,
                        row_count=new_size,
                    ),
                )

            return (state, gr.update(), gr.update(), gr.update())

        def update_d_size(
            current_state: Any, visible_values: Any = None, delta_size: Any = 0
        ) -> Any:
            """Update D vector size"""
            state = _coerce_editor_state(current_state)
            d_info = state["D"]
            new_size = max(
                1,
                min(
                    _MAX_EDITOR_DIMENSION,
                    _positive_int(d_info.get("size", 1)) + _coerce_delta(delta_size),
                ),
            )

            if new_size != d_info["size"]:
                raw_values = d_info.get("values", [])
                if visible_values is not None:
                    try:
                        visible_rows = _finite_table(visible_values, "D vector")
                        if all(len(row) == 1 for row in visible_rows):
                            raw_values = [row[0] for row in visible_rows]
                    except (TypeError, ValueError):
                        pass
                if not isinstance(raw_values, list):
                    raw_values = []
                new_values = resize_vector(
                    [[value] for value in raw_values], new_size, 0.33
                )
                d_info.update(size=new_size, values=[row[0] for row in new_values])

                return (
                    state,
                    f"**Vector D**: `{new_size}` (State Prior)",
                    f"**{new_size}**",
                    gr.Dataframe(
                        value=new_values,
                        headers=["Prior"],
                        interactive=True,
                        row_count=new_size,
                    ),
                )

            return (state, gr.update(), gr.update(), gr.update())

        # Comprehensive update function
        def update_all_with_state(
            current_state: Any,
            a_data: Any,
            b_data: Any,
            c_data: Any,
            d_data: Any,
            b_slice: Any,
        ) -> Any:
            """Update all visualizations and generate statistics"""
            try:
                # Update plots
                a_plot = create_enhanced_heatmap(
                    a_data, "A Matrix (Likelihood)", "Blues"
                )
                b_plot = create_enhanced_heatmap(
                    b_data, "B Matrix (Transitions)", "Oranges"
                )
                c_plot = create_enhanced_vector_plot(
                    c_data, "C Vector (Preferences)", "red"
                )
                d_plot = create_enhanced_vector_plot(
                    d_data, "D Vector (Prior)", "green"
                )

                # Generate statistics
                stats_text = "### 📊 **Real-time Matrix Statistics**\n\n"
                stats_text += calculate_matrix_stats(a_data, "Matrix A") + "\n"
                stats_text += calculate_matrix_stats(b_data, "Matrix B") + "\n"
                stats_text += calculate_matrix_stats(c_data, "Vector C") + "\n"
                stats_text += calculate_matrix_stats(d_data, "Vector D") + "\n"

                # Generate GNN
                updated_state, gnn_text = generate_gnn_from_matrices(
                    current_state,
                    a_data,
                    b_data,
                    c_data,
                    d_data,
                    b_slice,
                )

                return (
                    updated_state,
                    a_plot,
                    b_plot,
                    c_plot,
                    d_plot,
                    stats_text,
                    gnn_text,
                    "",
                )

            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.error(f"Error in update_all_with_state: {e}")
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    f"❌ **Validation failed**: {e}",
                )

        def reset_to_pomdp() -> Any:
            """Reset all matrices to POMDP template values"""
            default_state = _initial_matrix_state(get_pomdp_template())

            return (
                default_state,
                "**Matrix A**: `3×3` (Observations × States)",
                "**3**",
                "**3**",
                gr.Dataframe(value=default_state["A"]["values"], interactive=True),
                "**Matrix B**: `3×3×3` (States × States × Actions)",
                "**3**",
                "**3**",
                gr.Slider(maximum=2, value=0),
                gr.Dataframe(value=default_state["B"]["values"][0], interactive=True),
                "**Vector C**: `3` (Observation Preferences)",
                "**3**",
                gr.Dataframe(
                    value=[[v] for v in default_state["C"]["values"]], interactive=True
                ),
                "**Vector D**: `3` (State Prior)",
                "**3**",
                gr.Dataframe(
                    value=[[v] for v in default_state["D"]["values"]], interactive=True
                ),
            )

        # === WIRE UP ALL INTERACTIVE EVENTS ===

        # Matrix A dimension controls
        a_rows_plus.click(
            lambda s, values: update_a_dimensions(s, values, delta_rows=1),
            inputs=[matrix_state, a_values],
            outputs=[
                matrix_state,
                a_size_display,
                a_rows_display,
                a_cols_display,
                a_values,
            ],
        )

        a_rows_minus.click(
            lambda s, values: update_a_dimensions(s, values, delta_rows=-1),
            inputs=[matrix_state, a_values],
            outputs=[
                matrix_state,
                a_size_display,
                a_rows_display,
                a_cols_display,
                a_values,
            ],
        )

        a_cols_plus.click(
            lambda s, values: update_a_dimensions(s, values, delta_cols=1),
            inputs=[matrix_state, a_values],
            outputs=[
                matrix_state,
                a_size_display,
                a_rows_display,
                a_cols_display,
                a_values,
            ],
        )

        a_cols_minus.click(
            lambda s, values: update_a_dimensions(s, values, delta_cols=-1),
            inputs=[matrix_state, a_values],
            outputs=[
                matrix_state,
                a_size_display,
                a_rows_display,
                a_cols_display,
                a_values,
            ],
        )

        # Matrix B dimension controls
        b_states_plus.click(
            lambda s, values, selected: update_b_dimensions(
                s, values, selected, delta_states=1
            ),
            inputs=[matrix_state, b_values, b_slice_selector],
            outputs=[
                matrix_state,
                b_size_display,
                b_states_display,
                b_actions_display,
                b_slice_selector,
                b_values,
            ],
        )

        b_states_minus.click(
            lambda s, values, selected: update_b_dimensions(
                s, values, selected, delta_states=-1
            ),
            inputs=[matrix_state, b_values, b_slice_selector],
            outputs=[
                matrix_state,
                b_size_display,
                b_states_display,
                b_actions_display,
                b_slice_selector,
                b_values,
            ],
        )

        b_actions_plus.click(
            lambda s, values, selected: update_b_dimensions(
                s, values, selected, delta_actions=1
            ),
            inputs=[matrix_state, b_values, b_slice_selector],
            outputs=[
                matrix_state,
                b_size_display,
                b_states_display,
                b_actions_display,
                b_slice_selector,
                b_values,
            ],
        )

        b_actions_minus.click(
            lambda s, values, selected: update_b_dimensions(
                s, values, selected, delta_actions=-1
            ),
            inputs=[matrix_state, b_values, b_slice_selector],
            outputs=[
                matrix_state,
                b_size_display,
                b_states_display,
                b_actions_display,
                b_slice_selector,
                b_values,
            ],
        )

        def switch_b_slice(
            state: Any, visible_values: Any, requested_slice: Any
        ) -> Any:
            """Switch slices while retaining edits made to the previous slice."""
            updated_state, selected_values = _select_b_slice(
                state, visible_values, requested_slice
            )
            columns = len(selected_values[0]) if selected_values else 1
            return updated_state, gr.Dataframe(
                value=selected_values,
                headers=[f"State_{index}" for index in range(columns)],
                interactive=True,
                row_count=max(1, len(selected_values)),
            )

        b_slice_selector.change(
            switch_b_slice,
            inputs=[matrix_state, b_values, b_slice_selector],
            outputs=[matrix_state, b_values],
        )

        # Vector size controls
        c_size_plus.click(
            lambda s, values: update_c_size(s, values, delta_size=1),
            inputs=[matrix_state, c_values],
            outputs=[matrix_state, c_size_display, c_size_display_num, c_values],
        )

        c_size_minus.click(
            lambda s, values: update_c_size(s, values, delta_size=-1),
            inputs=[matrix_state, c_values],
            outputs=[matrix_state, c_size_display, c_size_display_num, c_values],
        )

        d_size_plus.click(
            lambda s, values: update_d_size(s, values, delta_size=1),
            inputs=[matrix_state, d_values],
            outputs=[matrix_state, d_size_display, d_size_display_num, d_values],
        )

        d_size_minus.click(
            lambda s, values: update_d_size(s, values, delta_size=-1),
            inputs=[matrix_state, d_values],
            outputs=[matrix_state, d_size_display, d_size_display_num, d_values],
        )

        # Auto-update functionality
        def maybe_auto_update(
            auto_enabled: Any,
            dirty: Any,
            state: Any,
            a_data: Any,
            b_data: Any,
            c_data: Any,
            d_data: Any,
            b_slice: Any,
        ) -> Any:
            """Regenerate visualizations on debounced timer ticks when enabled"""
            if not (auto_enabled and dirty):
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            return (
                *update_all_with_state(state, a_data, b_data, c_data, d_data, b_slice),
                False,
            )

        def manual_update(
            state: Any,
            a_data: Any,
            b_data: Any,
            c_data: Any,
            d_data: Any,
            b_slice: Any,
        ) -> Any:
            """Run a full regeneration and clear the debounce flag"""
            return (
                *update_all_with_state(state, a_data, b_data, c_data, d_data, b_slice),
                False,
            )

        # Debounced auto-update: dataframe edits only flip a dirty flag;
        # the timer performs the heavy regeneration at most once per interval.
        auto_dirty = gr.State(False)
        regen_timer = gr.Timer(DEBOUNCE_SECONDS)

        for matrix_input in [a_values, b_values, c_values, d_values]:
            matrix_input.change(lambda: True, outputs=[auto_dirty])

        regen_timer.tick(
            maybe_auto_update,
            inputs=[
                auto_update_checkbox,
                auto_dirty,
                matrix_state,
                a_values,
                b_values,
                c_values,
                d_values,
                b_slice_selector,
            ],
            outputs=[
                matrix_state,
                matrix_a_plot,
                matrix_b_plot,
                c_plot,
                d_plot,
                stats_output,
                gnn_output,
                validation_output,
                auto_dirty,
            ],
        )

        # Manual update button
        manual_update_btn.click(
            manual_update,
            inputs=[
                matrix_state,
                a_values,
                b_values,
                c_values,
                d_values,
                b_slice_selector,
            ],
            outputs=[
                matrix_state,
                matrix_a_plot,
                matrix_b_plot,
                c_plot,
                d_plot,
                stats_output,
                gnn_output,
                validation_output,
                auto_dirty,
            ],
        )

        # Reset button
        reset_btn.click(
            reset_to_pomdp,
            outputs=[
                matrix_state,
                a_size_display,
                a_rows_display,
                a_cols_display,
                a_values,
                b_size_display,
                b_states_display,
                b_actions_display,
                b_slice_selector,
                b_values,
                c_size_display,
                c_size_display_num,
                c_values,
                d_size_display,
                d_size_display_num,
                d_values,
            ],
        )

        # File operations
        save_btn.click(save_gnn, inputs=[gnn_output], outputs=[save_status])
        validate_btn.click(
            validate_editor_matrices,
            inputs=[
                matrix_state,
                a_values,
                b_values,
                c_values,
                d_values,
                b_slice_selector,
            ],
            outputs=[validation_output],
        )

        # Initialize on load
        demo.load(
            update_all_with_state,
            inputs=[
                matrix_state,
                a_values,
                b_values,
                c_values,
                d_values,
                b_slice_selector,
            ],
            outputs=[
                matrix_state,
                matrix_a_plot,
                matrix_b_plot,
                c_plot,
                d_plot,
                stats_output,
                gnn_output,
                validation_output,
            ],
        )

    return demo
