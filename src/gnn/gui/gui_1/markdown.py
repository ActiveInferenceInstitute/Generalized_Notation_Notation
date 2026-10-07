"""Provides helper functions: add_component_to_markdown, update_component_states, remove_component_from_markdown, parse_components_from_markdown, and 5 more.

Public functions: add_component_to_markdown, update_component_states, remove_component_from_markdown, parse_components_from_markdown, parse_state_space_from_markdown, _ensure_state_space_section, add_state_space_entry, update_state_space_entry, remove_state_space_entry
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_MARKDOWN_CHARS = 8 * 1024 * 1024


def _check_editor_size(md_text: str) -> None:
    if len(md_text) > MAX_MARKDOWN_CHARS:
        raise ValueError("GUI Markdown exceeds the 8,388,608-character editor limit")


def _markdown_lines(md_text: str) -> list[str]:
    _check_editor_size(md_text)
    return md_text.splitlines()


# ------------------------------
# Component helpers (public API)
# ------------------------------


def add_component_to_markdown(
    md_text: str, name: str, comp_type: str, states: Optional[list[str]] = None
) -> str:
    """
    Append a new component block to the markdown. Creates components section if missing.
    """
    _check_editor_size(md_text)
    states_list = states or []
    block: list[Any] = [
        "components:",
        f"  - name: {name}",
        f"    type: {comp_type}",
        f"    states: [{', '.join(states_list)}]",
    ]
    prefix = "\n" if (md_text and not md_text.endswith("\n")) else ""
    if "\ncomponents:\n" in f"\n{md_text}\n":
        item: list[Any] = [
            f"  - name: {name}",
            f"    type: {comp_type}",
            f"    states: [{', '.join(states_list)}]",
        ]
        return f"{md_text}{prefix}" + "\n".join(item) + "\n"
    return f"{md_text}{prefix}" + "\n".join(block) + "\n"


def update_component_states(
    md_text: str, name: str, states: list[str], mode: str = "append"
) -> str:
    """
    Update the states for a component by name.
    mode = 'append' to add a comment line, or 'replace' to replace a matching 'states' line.
    """
    if mode not in ("append", "replace"):
        mode = "append"
    lines = _markdown_lines(md_text)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.strip().startswith("- name:") and line.strip() == f"- name: {name}":
            if mode == "append":
                out.append(f"  # states appended: {states}")
            else:
                j = i + 1
                replaced = False
                while j < len(lines) and (
                    lines[j].startswith(" ") or not lines[j].strip()
                ):
                    if lines[j].strip().startswith("states:"):
                        out.pop()
                        out.append(lines[i])
                        for k in range(i + 1, j):
                            out.append(lines[k])
                        out.append(f"    states: [{', '.join(states)}]")
                        i = j
                        replaced = True
                        break
                    j += 1
                if not replaced:
                    out.append(f"    states: [{', '.join(states)}]")
        i += 1
    result = "\n".join(out)
    if not result.endswith("\n"):
        result += "\n"
    return result


def remove_component_from_markdown(md_text: str, name: str) -> str:
    """
    Remove a component block from the components list by name. Leaves other content intact.
    """
    lines = _markdown_lines(md_text)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("- name:") and line.strip() == f"- name: {name}":
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
                if lines[i].strip().startswith("- name:"):
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    result = "\n".join(out)
    if not result.endswith("\n"):
        result += "\n"
    return result


def parse_components_from_markdown(md_text: str) -> list[dict[str, object]]:
    """
    Parse components section into a list of {name, type, states} dicts (best-effort).
    """
    components: list[dict[str, object]] = []
    lines = _markdown_lines(md_text)
    i = 0
    current: dict[str, object] | None = None
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip().startswith("- name:"):
            if current:
                components.append(current)
            current = {"name": line.split(":", 1)[1].strip()}
        elif current and line.strip().startswith("type:"):
            current["type"] = line.split(":", 1)[1].strip()
        elif current and line.strip().startswith("states:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith("[") and raw.endswith("]"):
                inner = raw[1:-1]
                states = [s.strip() for s in inner.split(",") if s.strip()]
                current["states"] = states
            else:
                current["states"] = []
        i += 1
    if current:
        components.append(current)
    return components


# ------------------------------
# State space helpers (public API)
# ------------------------------


def _parse_state_line(line: str) -> tuple[str, str, str] | None:
    """Parse the editor's state-line grammar without delimiter backtracking."""
    text = line.lstrip()
    if not text or not text[0].isascii() or not text[0].isalpha():
        return None
    end = 1
    while (
        end < len(text)
        and text[end].isascii()
        and (text[end].isalnum() or text[end] == "_")
    ):
        end += 1
    if end == len(text) or text[end] != "[":
        return None
    close = text.find("]", end + 1)
    while close != -1:
        tail = close + 1
        while tail < len(text) and text[tail].isspace():
            tail += 1
        if tail == len(text) or text[tail] == "#":
            comment = text[tail + 1 :].strip() if tail < len(text) else ""
            return text[:end], text[end + 1 : close], comment
        # Each whitespace suffix lies after a distinct closing bracket, so
        # failed candidates visit at most a constant multiple of the input.
        close = text.find("]", close + 1)
    return None


def parse_state_space_from_markdown(md_text: str) -> list[dict[str, object]]:
    """
    Parse a 'State Space' section with lines like: name[dim1,dim2,type=int]
    Returns list of dicts: {name, dims: [int,...], type}
    """
    result: list[dict[str, object]] = []
    lines = _markdown_lines(md_text)
    in_section = False
    found_section = False
    for line in lines:
        norm = line.strip().lower()
        if norm.startswith("## "):
            # Detect known headers for state-space listings
            tag = norm[3:].replace(" ", "")
            if tag in {
                "statespace",
                "statespaceblock",
                "statespaces",
                "state_space",
                "state_space_block",
            }:
                in_section = True
                found_section = True
                continue
            else:
                in_section = False
        if norm in {"state space", "statespace", "statespaceblock"}:
            in_section = True
            found_section = True
            continue
        if in_section:
            m = _parse_state_line(line)
            if m:
                name, inside, comment = m
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                dims: list[int] = []
                typ = None
                for p in parts:
                    if p.startswith("type="):
                        typ = p.split("=", 1)[1]
                    else:
                        try:
                            dims.append(int(p))
                        except ValueError:
                            logger.debug("Skipping non-integer dimension token: %s", p)
                entry: dict[str, object] = {
                    "name": name,
                    "dims": dims,
                    "type": typ or "",
                }
                if comment:
                    entry["comment"] = comment
                result.append(entry)

    # Recovery: if no explicit section found, scan whole document for state-like lines
    if not result and not found_section:
        for line in lines:
            m = _parse_state_line(line)
            if m:
                name, inside, comment = m
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                dims = []
                typ = None
                for p in parts:
                    if p.startswith("type="):
                        typ = p.split("=", 1)[1]
                    else:
                        try:
                            dims.append(int(p))
                        except ValueError:
                            logger.debug("Skipping non-integer dimension token: %s", p)
                entry = {
                    "name": name,
                    "dims": dims,
                    "type": typ or "",
                }
                if comment:
                    entry["comment"] = comment
                result.append(entry)
    return result


def _ensure_state_space_section(md_text: str) -> tuple[str, int]:
    """Handle ensure state space section for internal callers."""
    lines = _markdown_lines(md_text)
    for i, line in enumerate(lines):
        if (
            line.strip().lower().startswith("## state space")
            or line.strip().lower() == "state space"
        ):
            return md_text, i + 1
    to_add: list[Any] = []
    if lines and lines[-1].strip() != "":
        to_add.append("")
    to_add.append("## State Space")
    new_text = (
        md_text
        + ("\n" if not md_text.endswith("\n") else "")
        + "\n".join(to_add)
        + "\n"
    )
    return new_text, len(new_text.splitlines())


def add_state_space_entry(
    md_text: str,
    name: str,
    dims: list[int],
    typ: str | None = None,
    comment: str | None = None,
) -> str:
    """Provide add state space entry behavior."""
    md_text, _ = _ensure_state_space_section(md_text)
    suffix = f"  # {comment}" if comment else ""
    line = (
        f"{name}[{', '.join(str(d) for d in dims)}{', type=' + typ if typ else ''}]"
        + suffix
    )
    return md_text + ("\n" if not md_text.endswith("\n") else "") + line + "\n"


def update_state_space_entry(
    md_text: str,
    orig_name: str,
    new_name: str,
    dims: list[int],
    typ: str | None = None,
    comment: str | None = None,
) -> str:
    """Update state space entry."""
    lines = _markdown_lines(md_text)
    out: list[str] = []
    replaced = False
    for line in lines:
        m = _parse_state_line(line)
        if m and m[0] == orig_name:
            suffix = f"  # {comment}" if comment else (f"  # {m[2]}" if m[2] else "")
            new_line = (
                f"{new_name}[{', '.join(str(d) for d in dims)}{', type=' + typ if typ else ''}]"
                + suffix
            )
            out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        return add_state_space_entry(
            "\n".join(out) + "\n", new_name, dims, typ, comment
        )
    return "\n".join(out) + ("\n" if not out or out[-1] != "" else "")


def remove_state_space_entry(md_text: str, name: str) -> str:
    """Provide remove state space entry behavior."""
    lines = _markdown_lines(md_text)
    out: list[str] = []
    for line in lines:
        m = _parse_state_line(line)
        if m and m[0] == name:
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if not out or out[-1] != "" else "")
