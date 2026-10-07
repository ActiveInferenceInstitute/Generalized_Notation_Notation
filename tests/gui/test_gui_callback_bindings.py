#!/usr/bin/env python3
"""Callback-binding and UX regression tests for GUI 1 and GUI 2 (Step 22 wave 2).

Covers scope findings G11/G12 (every visible control must be bound to a real
callback, proven both by source inspection and by executing both builders
under a functional fake ``gradio``), G2 (``parse_dims_csv`` must exist and
surface precise dimension-parsing errors), G4 (the gui_1 validation pane),
G5 (the gui_2 validation output must be routed through the core update
paths), and G6 (gui_2 auto-update must go through a debounced
``gr.Timer`` instead of heavy per-keystroke work). Findings are defined in
``scratch/scope-gui-gui1-gui2.md``.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

_BIND_METHODS = ("click", "change", "load", "then", "input", "select", "tick", "blur")

_CONTAINER_NAMES = ("Blocks", "Row", "Column", "Tab", "Accordion", "Group")

_LEAF_NAMES = (
    "Markdown",
    "Code",
    "Button",
    "Textbox",
    "Dropdown",
    "Dataframe",
    "Slider",
    "Checkbox",
    "State",
    "Timer",
    "JSON",
    "HTML",
    "Plot",
    "Radio",
    "Number",
    "File",
)

_EventRecord = tuple[str, Any, Any, Any]


def _make_functional_gradio(log: list[_EventRecord]) -> types.ModuleType:
    """Build a fake ``gradio`` module whose builders run and wire events.

    Every component accepts arbitrary constructor arguments (so
    ``gr.Timer(DEBOUNCE_SECONDS)`` and keyword-heavy leaf constructors work),
    supports the context-manager protocol Gradio layout contexts rely on,
    and records each ``.click``/``.change``/... binding as
    ``(method, fn, inputs, outputs)`` in the shared ``log``.
    """

    class StubComponent:
        """Universal fake Gradio component that records event bindings."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.constructor_args = args
            self.constructor_kwargs = kwargs

        def __enter__(self) -> StubComponent:
            return self

        def __exit__(self, *_exc_info: object) -> bool:
            # Never swallow builder exceptions: a failing builder must fail
            # the smoke test, not be silenced by the fake context manager.
            return False

        def _record(
            self, method: str, fn: Any, inputs: Any, outputs: Any
        ) -> StubComponent:
            log.append((method, fn, inputs, outputs))
            return self

    def _make_binder(method: str) -> Any:
        def binder(self: StubComponent, fn: Any = None, **kwargs: Any) -> StubComponent:
            return self._record(method, fn, kwargs.get("inputs"), kwargs.get("outputs"))

        binder.__name__ = method
        return binder

    for method in _BIND_METHODS:
        setattr(StubComponent, method, _make_binder(method))

    def update(**kwargs: Any) -> types.SimpleNamespace:
        """Mirror ``gr.update`` by returning a fresh update payload."""
        return types.SimpleNamespace(**kwargs)

    gradio = types.ModuleType("gradio")
    for name in (*_CONTAINER_NAMES, *_LEAF_NAMES):
        setattr(gradio, name, StubComponent)
    gradio.update = update  # type: ignore[attr-defined]
    gradio.themes = types.SimpleNamespace(Base=StubComponent, Soft=StubComponent)  # type: ignore[attr-defined]
    return gradio


@dataclass
class FunctionalGradio:
    """Handle over the shared event log of the installed functional stub."""

    events: list[_EventRecord] = field(default_factory=list)

    @property
    def clicks(self) -> list[_EventRecord]:
        """Return only ``.click`` bindings, the button-wiring surface."""
        return [record for record in self.events if record[0] == "click"]

    def reset(self) -> None:
        """Clear the log so a second builder can be counted in isolation."""
        self.events.clear()


@pytest.fixture
def functional_gradio() -> Iterator[FunctionalGradio]:
    """Install a functional fake ``gradio`` and reload both UI modules.

    Unlike the bare-module fixture in ``test_step22_headless_default.py``
    (which only exercises the static fallback), this stub lets
    ``build_gui``/``build_visual_gui`` execute their real builder bodies so
    wiring regressions surface at build time. The original module state is
    restored afterwards.
    """
    gui1_ui = importlib.import_module("gnn.gui.gui_1.ui")
    gui2_ui = importlib.import_module("gnn.gui.gui_2.ui")
    original_gradio = sys.modules.get("gradio")
    sys.modules["gradio"] = _make_functional_gradio(log := [])
    importlib.reload(gui1_ui)
    importlib.reload(gui2_ui)
    try:
        yield FunctionalGradio(events=log)
    finally:
        if original_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original_gradio
        importlib.reload(gui1_ui)
        importlib.reload(gui2_ui)


def _test_logger() -> logging.Logger:
    """Return a quiet logger for builder calls."""
    return logging.getLogger("test_gui_callback_bindings")


class TestGui1CallbackBindings:
    """GUI 1: no unbound buttons (G11), validation pane present (G4),
    builder executes under the functional stub with every button wired (G12)."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_every_constructor_button_is_bound(self) -> None:
        """G11/G3: every visible GUI 1 control must have a ``.click`` binding.

        The ``export_button`` entry is the G3 regression: the Export Model
        button used to be constructed but never wired to any callback.
        """
        import inspect

        from gnn.gui.gui_1.ui import build_gui

        source = inspect.getsource(build_gui)
        for button_name in (
            "add_button",
            "replace_states_button",
            "append_states_button",
            "remove_button",
            "save_button",
            "export_button",
            "st_refresh",
            "st_add",
            "st_update",
            "st_remove",
        ):
            assert f"{button_name}.click(" in source

    @pytest.mark.unit
    @pytest.mark.fast
    def test_validation_pane_exists_in_constructor(self) -> None:
        """G4: the constructor must declare a ``validation_output`` pane."""
        import inspect

        from gnn.gui.gui_1.ui import build_gui

        assert "validation_output" in inspect.getsource(build_gui)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_builder_runs_and_binds_every_button(
        self, functional_gradio: FunctionalGradio, tmp_path: Path
    ) -> None:
        """G12: ``build_gui`` executes under the functional stub and records
        at least one ``.click`` binding per visible button (10 after the
        Export Model wiring lands)."""
        from gnn.gui.gui_1.ui import build_gui

        demo = build_gui("# M\n", tmp_path / "model.md")
        assert demo is not None
        assert len(functional_gradio.clicks) >= 10

    @pytest.mark.unit
    @pytest.mark.fast
    def test_hostile_markdown_keeps_live_editor_values_and_save_statistics(
        self, functional_gradio: FunctionalGradio, tmp_path: Path
    ) -> None:
        from gnn.gui.gui_1.markdown import parse_state_space_from_markdown
        from gnn.gui.gui_1.ui import build_gui

        path = tmp_path / "model.md"
        build_gui("# M\n", path)
        callbacks = {fn.__name__: fn for _, fn, _, _ in functional_gradio.events}
        hostile = "A[" + "] " * 65_536 + "!"
        document = f"## State Space\n{hostile}\nB[2,type=int] # retained\n"
        updated, choices, error = callbacks["update_state_live"](
            document, "B", "C", "3, 4", "float", "retained"
        )
        assert error == "" and choices.value == "C" and choices.choices == ["C"]
        assert hostile in updated
        assert parse_state_space_from_markdown(updated) == [
            {"name": "C", "dims": [3, 4], "type": "float", "comment": "retained"}
        ]
        message, status, stats, error = callbacks["save_md"](updated)
        assert error == "" and "Saved" in message and "saved" in status
        assert stats["state_entries"] == 1 and stats["total_states"] == 2
        assert path.read_text() == updated

    @pytest.mark.unit
    @pytest.mark.fast
    def test_oversized_editor_callbacks_refuse_before_save(
        self, functional_gradio: FunctionalGradio, tmp_path: Path
    ) -> None:
        from gnn.gui.gui_1.markdown import MAX_MARKDOWN_CHARS
        from gnn.gui.gui_1.ui import build_gui

        path = tmp_path / "model.md"
        original = "# Preserved\n"
        path.write_text(original)
        build_gui(original, path)
        callbacks = {fn.__name__: fn for _, fn, _, _ in functional_gradio.events}
        oversized = "x" * (MAX_MARKDOWN_CHARS + 1)
        assert "editor limit" in callbacks["save_md"](oversized)[3]
        assert path.read_text() == original
        assert list(tmp_path.iterdir()) == [path]
        choices, error = callbacks["refresh_states"](oversized)
        assert vars(choices) == {} and "editor limit" in error
        unchanged, choices, error = callbacks["update_state_live"](
            oversized, "A", "A", "2", "float", ""
        )
        assert unchanged == oversized and vars(choices) == {}
        assert "editor limit" in error
        unchanged, error = callbacks["add_component"](oversized, "C", "state", "x")
        assert unchanged == oversized and "editor limit" in error


class TestGui2CallbackBindings:
    """GUI 2: no unbound buttons (G11), validation routing through core
    update paths (G5), debounced timer auto-update (G6), and a G12 smoke
    run of the builder under the functional stub."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_every_editor_button_is_bound(self) -> None:
        """G11: every visible GUI 2 control must have a ``.click`` binding."""
        import inspect

        from gnn.gui.gui_2.ui import build_visual_gui

        source = inspect.getsource(build_visual_gui)
        for button_name in (
            "a_rows_minus",
            "a_rows_plus",
            "a_cols_minus",
            "a_cols_plus",
            "b_states_minus",
            "b_states_plus",
            "b_actions_minus",
            "b_actions_plus",
            "c_size_minus",
            "c_size_plus",
            "d_size_minus",
            "d_size_plus",
            "manual_update_btn",
            "reset_btn",
            "save_btn",
            "validate_btn",
        ):
            assert f"{button_name}.click(" in source

    @pytest.mark.unit
    @pytest.mark.fast
    def test_validation_output_routed_through_core_update_paths(self) -> None:
        """G5: ``validation_output`` must appear as an output of the
        auto-update tick, the manual update, and the demo-load refresh."""
        import inspect

        from gnn.gui.gui_2.ui import build_visual_gui

        source = inspect.getsource(build_visual_gui)
        assert source.count("validation_output,") >= 3

    @pytest.mark.unit
    @pytest.mark.fast
    def test_auto_update_uses_debounced_timer(self) -> None:
        """G6: auto-update must fire from a debounced ``gr.Timer`` tick with
        a single ``maybe_auto_update`` implementation, not from heavy work
        attached to each of the four per-edit ``.change`` events."""
        import inspect

        from gnn.gui.gui_2.ui import build_visual_gui

        source = inspect.getsource(build_visual_gui)
        assert "gr.Timer(" in source
        assert "regen_timer.tick(" in source
        # Definition + the single tick reference; a third occurrence would
        # mean the heavy callback is still attached to per-edit events.
        assert source.count("maybe_auto_update") == 2
        assert "DEBOUNCE_SECONDS" in source

    @pytest.mark.unit
    @pytest.mark.fast
    def test_builder_runs_and_binds_every_button(
        self, functional_gradio: FunctionalGradio, tmp_path: Path
    ) -> None:
        """G12: ``build_visual_gui`` executes under the functional stub and
        records a ``.click`` binding for each of its 16 visible buttons."""
        from gnn.gui.gui_2.ui import build_visual_gui

        demo = build_visual_gui("# M\n", tmp_path / "model.md", _test_logger())
        assert demo is not None
        assert len(functional_gradio.clicks) >= 16


class TestParseDimsCsv:
    """G2: ``parse_dims_csv`` must exist and reject invalid dimension input
    with a precise, user-facing message instead of silently dropping tokens."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parses_valid_csv(self) -> None:
        """Valid decimal integers parse to an int list with no error."""
        from gnn.gui.gui_1.ui import parse_dims_csv

        assert parse_dims_csv("3, 4") == ([3, 4], None)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_input_is_not_an_error(self) -> None:
        """Empty and whitespace-only input parse to an empty list, no error."""
        from gnn.gui.gui_1.ui import parse_dims_csv

        assert parse_dims_csv("") == ([], None)
        assert parse_dims_csv("   ") == ([], None)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_invalid_token_reported_verbatim(self) -> None:
        """A non-integer token yields an error message naming the raw input
        and every offending token."""
        from gnn.gui.gui_1.ui import parse_dims_csv

        dims, error = parse_dims_csv("3,a,4")
        assert dims == []
        assert error is not None
        assert error.startswith("❌")
        assert "3,a,4" in error
        assert "a" in error

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_token_reported(self) -> None:
        """An empty slot between separators is an offending token, not a
        silently dropped one."""
        from gnn.gui.gui_1.ui import parse_dims_csv

        dims, error = parse_dims_csv("3,,4")
        assert dims == []
        assert error is not None
        assert "3,,4" in error
