"""One token grammar and injection contract for manuscript consumers.

The template implementation is authoritative when importable. The local mirror
keeps standalone audits headless, preserving unknown tokens verbatim.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

_FALLBACK_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_FALLBACK_EXCLUDED = frozenset(
    {"AGENTS.md", "MANUSCRIPT_STATUS.md", "README.md", "SYNTAX.md"}
)


def _template_injector() -> ModuleType | None:
    """Load the explicit template root or an already importable template.

    Only absence of the template permits fallback; a broken dependency inside
    an installed template must remain visible rather than changing semantics.
    """
    root = os.environ.get("TEMPLATE_REPO_ROOT")
    if (
        root
        and (Path(root) / "infrastructure/rendering/manuscript_injection.py").is_file()
    ):
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        return importlib.import_module("infrastructure.rendering.manuscript_injection")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "infrastructure",
            "infrastructure.rendering",
            "infrastructure.rendering.manuscript_injection",
        }:
            raise
        return None


_INJECTOR = _template_injector()
TOKEN_RE = getattr(_INJECTOR, "_TOKEN_RE", _FALLBACK_TOKEN_RE)
EXCLUDED_DOC_FILENAMES = frozenset(
    getattr(_INJECTOR, "EXCLUDED_DOC_FILENAMES", _FALLBACK_EXCLUDED)
)


def substitute_manuscript_text(
    text: str, variables: dict[str, str]
) -> tuple[str, list[str]]:
    """Return resolved text and every unresolved key using the active injector."""
    if _INJECTOR is not None:
        return cast(
            "tuple[str, list[str]]",
            _INJECTOR.substitute_manuscript_text(text, variables),
        )
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in variables:
            return variables[key]
        unresolved.append(key)
        return match.group(0)

    return TOKEN_RE.sub(replace, text), unresolved


def hydrate_text(text: str, variables: dict[str, str]) -> str:
    """Resolve known tokens, leaving unknown tokens visible for the audit."""
    return substitute_manuscript_text(text, variables)[0]


def active_preamble(text: str) -> str:
    """Extract LaTeX fences (or raw LaTeX), removing actual TeX comments.

    Prose outside LaTeX fences cannot declare a package. An escaped percent is
    retained, including the even/odd backslash rule of TeX comment syntax.
    """
    blocks = re.findall(r"```\s*latex\s*\n(.*?)\n\s*```", text, re.DOTALL)
    content = (
        "\n".join(blocks)
        if blocks
        else "\n".join(
            line
            for line in re.sub(r"```.*?```", "", text, flags=re.DOTALL).splitlines()
            if line.lstrip().startswith("\\")
        )
    )
    return "\n".join(_strip_tex_comment(line) for line in content.splitlines())


def _strip_tex_comment(line: str) -> str:
    for index, char in enumerate(line):
        if char != "%":
            continue
        backslashes = len(line[:index]) - len(line[:index].rstrip("\\"))
        if backslashes % 2 == 0:
            return line[:index]
    return line
