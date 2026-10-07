"""Provides helper functions: _dimension_count, build_gui, parse_dims_csv.

Public functions: _dimension_count, build_gui, parse_dims_csv
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

try:
    import gradio as gr
except Exception:
    gr = cast(Any, None)

from .markdown import (
    add_component_to_markdown,
    add_state_space_entry,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    remove_component_from_markdown,
    remove_state_space_entry,
    update_component_states,
    update_state_space_entry,
)


def _dimension_count(item: dict[str, object]) -> int:
    """Handle dimension count for internal callers."""
    dims = item.get("dims", [])
    return len(dims) if isinstance(dims, list) else 0


def parse_dims_csv(raw: str) -> tuple[list[int], str | None]:
    """Parse a comma-separated dimensions string into integers.

    Returns ``(dims, None)`` on success. Empty or whitespace-only input yields
    ``([], None)``. If any token is not a non-empty decimal integer, returns
    ``([], message)`` where the message starts with "❌", repeats the raw
    input verbatim, and names every offending token.
    """
    if not raw or not raw.strip():
        return [], None
    dims: list[int] = []
    invalid: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdecimal():
            dims.append(int(token))
        else:
            invalid.append(token)
    if invalid:
        offenders = ", ".join(repr(t) for t in invalid)
        message = (
            f"❌ Invalid dimensions input: '{raw}' — invalid token(s): {offenders}"
        )
        return [], message
    return dims, None


def build_gui(
    markdown_text: str, export_path: Path, logger: Optional[logging.Logger] = None
) -> "gr.Blocks":
    """Build gui."""
    if gr is None:
        raise RuntimeError("Gradio not available")

    # Compute initial state-space options and selected entry
    initial_items = parse_state_space_from_markdown(markdown_text)
    initial_names = [str(i.get("name", "")) for i in initial_items if i.get("name")]
    initial_selected = initial_names[0] if initial_names else ""

    def _vals_for(name: str) -> Any:
        """Handle vals for for internal callers."""
        for e in initial_items:
            if str(e.get("name")) == name:
                dims_value = e.get("dims", [])
                dims = dims_value if isinstance(dims_value, list) else []
                dims_csv = ", ".join(str(d) for d in dims)
                typ = str(e.get("type", ""))
                cmt = str(e.get("comment", ""))
                return str(e.get("name", "")), dims_csv, typ, cmt
        return "", "", "", ""

    init_name_val, init_dims_val, init_type_val, init_comment_val = _vals_for(
        initial_selected
    )

    with gr.Blocks(
        title="🔧 GNN Form-based Constructor", theme=gr.themes.Base()
    ) as demo:
        gr.Markdown("""# 🔧 GNN Form-based Constructor
        
        **Interactive two-pane editor for systematic GNN model construction**
        
        - **Left Panel**: Component and state space management controls
        - **Right Panel**: Live-updating GNN markdown editor
        - **Real-time Sync**: Changes are immediately reflected in both panels
        """)

        # Status indicator
        status_display = gr.Markdown("🟢 **Status**: Ready for component editing")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Construction Controls")
                with gr.Tab("📋 Components"):
                    gr.Markdown("**Add or modify GNN model components**")
                    component_name = gr.Textbox(
                        label="Component name",
                        value="example_component",
                    )
                    component_type = gr.Dropdown(
                        ["observation", "hidden", "action", "policy"],
                        value="observation",
                        label="Component Type (Select Active Inference component type)",
                    )
                    state_list = gr.Textbox(
                        label="States (comma-separated)",
                    )
                    add_button = gr.Button("➕ Add Component", variant="primary")
                    replace_states_button = gr.Button("🔄 Replace States")
                    append_states_button = gr.Button("📄 Append States")
                    remove_button = gr.Button("🗑️ Remove Component", variant="stop")
                with gr.Tab("🏗️ State Space"):
                    gr.Markdown("**Manage state space entries and dimensions**")
                    state_entries = gr.Dropdown(
                        choices=initial_names,
                        value=initial_selected,
                        label="Select State Entry (Choose entry to edit)",
                        interactive=True,
                    )
                    st_name = gr.Textbox(
                        label="Name",
                        value=init_name_val,
                    )
                    st_dims = gr.Textbox(
                        label="Dimensions (comma-separated)",
                        value=init_dims_val,
                    )
                    st_type = gr.Textbox(
                        label="Type (optional)",
                        value=init_type_val,
                    )
                    st_comment = gr.Textbox(
                        label="Comment (optional)",
                        value=init_comment_val,
                    )
                    with gr.Row():
                        st_refresh = gr.Button("🔄 Refresh State List")
                        st_add = gr.Button("➕ Add State Entry", variant="primary")
                    with gr.Row():
                        st_update = gr.Button("✏️ Update Selected")
                        st_remove = gr.Button("🗑️ Remove Selected", variant="stop")

                # Action buttons
                with gr.Row():
                    save_button = gr.Button(
                        "💾 Save Markdown", variant="primary", scale=2
                    )
                    export_button = gr.Button("📄 Export Model", variant="secondary")
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Live GNN Markdown Editor")
                markdown_editor = gr.Code(
                    value=markdown_text,
                    language="markdown",
                    label="GNN Markdown - Real-time synchronized model specification",
                    lines=25,
                )

                # Model statistics and validation
                model_stats = gr.JSON(
                    label="📊 Model Statistics",
                    value={
                        "components": len(
                            parse_components_from_markdown(markdown_text)
                        ),
                        "state_entries": len(initial_items),
                        "total_states": sum(
                            _dimension_count(item) for item in initial_items
                        ),
                    },
                )

        def _split_states(s: str) -> list[str]:
            """Handle split states for internal callers."""
            return [x.strip() for x in s.split(",") if x.strip()]

        def add_component(
            md: str, name: str, ctype: str, states_csv: str
        ) -> tuple[str, str]:
            """Provide add component behavior."""
            try:
                new_md = add_component_to_markdown(
                    md, name, ctype, _split_states(states_csv)
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def replace_states(md: str, name: str, states_csv: str) -> tuple[str, str]:
            """Provide replace states behavior."""
            try:
                new_md = update_component_states(
                    md, name, _split_states(states_csv), mode="replace"
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def append_states(md: str, name: str, states_csv: str) -> tuple[str, str]:
            """Provide append states behavior."""
            try:
                new_md = update_component_states(
                    md, name, _split_states(states_csv), mode="append"
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def remove_component(md: str, name: str) -> tuple[str, str]:
            """Provide remove component behavior."""
            try:
                new_md = remove_component_from_markdown(md, name)
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def save_md(md: str) -> Any:
            """Save md."""
            try:
                # Admit and parse before writing, including the editor size limit.
                items = parse_state_space_from_markdown(md)
                components = parse_components_from_markdown(md)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=export_path.parent, delete=False
                ) as tmp_f:
                    tmp_f.write(md)
                os.replace(tmp_f.name, str(export_path))
                if logger:
                    logger.info(f"📄 Model saved to {export_path}")
                save_message = (
                    f"✅ **Saved Successfully**\n\n"
                    f"File: `{export_path}`\nSize: {len(md)} characters"
                )
                # Calculate updated statistics
                stats: dict[str, Any] = {
                    "components": len(components),
                    "state_entries": len(items),
                    "total_states": sum(_dimension_count(item) for item in items),
                    "file_size_chars": len(md),
                    "last_saved": datetime.now().strftime("%H:%M:%S"),
                }
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return gr.update(), gr.update(), gr.update(), f"❌ {e}"
            return save_message, "🟢 **Status**: Model saved successfully", stats, ""

        # Status messages and save functionality
        save_status = gr.Markdown(
            "💾 **Ready to save**: Use the save button to export your model"
        )

        validation_output = gr.Markdown("")

        # Component management event handlers
        add_button.click(
            add_component,
            inputs=[markdown_editor, component_name, component_type, state_list],
            outputs=[markdown_editor, validation_output],
        ).then(
            lambda: "🟡 **Status**: Component added - remember to save your changes",
            outputs=[status_display],
        )

        replace_states_button.click(
            replace_states,
            inputs=[markdown_editor, component_name, state_list],
            outputs=[markdown_editor, validation_output],
        ).then(
            lambda: "🟡 **Status**: States replaced - remember to save your changes",
            outputs=[status_display],
        )

        append_states_button.click(
            append_states,
            inputs=[markdown_editor, component_name, state_list],
            outputs=[markdown_editor, validation_output],
        ).then(
            lambda: "🟡 **Status**: States appended - remember to save your changes",
            outputs=[status_display],
        )

        remove_button.click(
            remove_component,
            inputs=[markdown_editor, component_name],
            outputs=[markdown_editor, validation_output],
        ).then(
            lambda: "🟡 **Status**: Component removed - remember to save your changes",
            outputs=[status_display],
        )

        # Save functionality with enhanced feedback
        save_button.click(
            save_md,
            inputs=[markdown_editor],
            outputs=[save_status, status_display, model_stats, validation_output],
        )

        # Export shares the same save path as the Save button
        export_button.click(
            save_md,
            inputs=[markdown_editor],
            outputs=[save_status, status_display, model_stats, validation_output],
        )

        # State Space actions
        def _compute_state_choices(md: str) -> list[str]:
            """Compute state choices."""
            items = parse_state_space_from_markdown(md)
            return [str(i.get("name", "")) for i in items if i.get("name")]

        def refresh_states(md: str) -> Any:
            """Provide refresh states behavior."""
            try:
                return gr.update(choices=_compute_state_choices(md)), ""
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return gr.update(), f"❌ {e}"

        def add_state(
            md: str, name: str, dims_csv: str, typ: str, comment: str
        ) -> tuple[str, str]:
            """Provide add state behavior."""
            try:
                dims, dims_error = parse_dims_csv(dims_csv)
                if dims_error:
                    return md, dims_error
                new_md = add_state_space_entry(
                    md, name, dims, typ or None, comment or None
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def update_state(
            md: str, selected: str, name: str, dims_csv: str, typ: str, comment: str
        ) -> tuple[str, str]:
            """Update state."""
            try:
                dims, dims_error = parse_dims_csv(dims_csv)
                if dims_error:
                    return md, dims_error
                new_md = update_state_space_entry(
                    md,
                    selected or name,
                    name or (selected or ""),
                    dims,
                    typ or None,
                    comment or None,
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"
            return new_md, ""

        def remove_state(md: str, selected: str) -> tuple[str, str]:
            """Provide remove state behavior."""
            try:
                if not selected:
                    return md, ""
                return remove_state_space_entry(md, selected), ""
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, f"❌ {e}"

        st_refresh.click(
            refresh_states,
            inputs=[markdown_editor],
            outputs=[state_entries, validation_output],
        )
        markdown_editor.change(
            refresh_states,
            inputs=[markdown_editor],
            outputs=[state_entries, validation_output],
        )
        st_add.click(
            add_state,
            inputs=[markdown_editor, st_name, st_dims, st_type, st_comment],
            outputs=[markdown_editor, validation_output],
        )
        st_update.click(
            update_state,
            inputs=[
                markdown_editor,
                state_entries,
                st_name,
                st_dims,
                st_type,
                st_comment,
            ],
            outputs=[markdown_editor, validation_output],
        )
        st_remove.click(
            remove_state,
            inputs=[markdown_editor, state_entries],
            outputs=[markdown_editor, validation_output],
        )

        def populate_fields(md: str, selected: str) -> Any:
            """Provide populate fields behavior."""
            entries = parse_state_space_from_markdown(md)
            for e in entries:
                if str(e.get("name")) == selected:
                    dims_value = e.get("dims", [])
                    dims_items = dims_value if isinstance(dims_value, list) else []
                    dims = ", ".join(str(d) for d in dims_items)
                    typ = str(e.get("type", ""))
                    cmt = str(e.get("comment", ""))
                    return [e.get("name", ""), dims, typ, cmt]
            return ["", "", "", ""]

        state_entries.change(
            populate_fields,
            inputs=[markdown_editor, state_entries],
            outputs=[st_name, st_dims, st_type, st_comment],
        )

        # Real-time editing: typing in any field applies update and keeps dropdown synced
        def update_state_live(
            md: str, selected: str, name: str, dims_csv: str, typ: str, comment: str
        ) -> Any:
            """Update state live."""
            try:
                dims, dims_error = parse_dims_csv(dims_csv)
                if dims_error:
                    return md, gr.update(), dims_error
                new_md = update_state_space_entry(
                    md,
                    selected or name,
                    name or (selected or ""),
                    dims,
                    typ or None,
                    comment or None,
                )
                choices = _compute_state_choices(new_md)
                new_selected = name or (
                    selected if selected in choices else (choices[0] if choices else "")
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                return md, gr.update(), f"❌ {e}"
            return new_md, gr.update(choices=choices, value=new_selected), ""

        st_name.change(
            update_state_live,
            inputs=[
                markdown_editor,
                state_entries,
                st_name,
                st_dims,
                st_type,
                st_comment,
            ],
            outputs=[markdown_editor, state_entries, validation_output],
        )
        st_dims.change(
            update_state_live,
            inputs=[
                markdown_editor,
                state_entries,
                st_name,
                st_dims,
                st_type,
                st_comment,
            ],
            outputs=[markdown_editor, state_entries, validation_output],
        )
        st_type.change(
            update_state_live,
            inputs=[
                markdown_editor,
                state_entries,
                st_name,
                st_dims,
                st_type,
                st_comment,
            ],
            outputs=[markdown_editor, state_entries, validation_output],
        )
        st_comment.change(
            update_state_live,
            inputs=[
                markdown_editor,
                state_entries,
                st_name,
                st_dims,
                st_type,
                st_comment,
            ],
            outputs=[markdown_editor, state_entries, validation_output],
        )

    return demo
