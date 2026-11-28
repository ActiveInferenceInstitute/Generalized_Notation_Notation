"""Every command the manuscript prints must be one the repository accepts.

§6 opens by promising it "lists only commands that exist in the repository, so
that a reader with a clean checkout can reproduce the pipeline". It shipped

    uv run python scripts/run_model_family_acceptance.py \\
      --manifest input/model_family_manifest.json --strict

which exits 2 on a clean checkout: ``--output-dir`` is ``required=True``. The
file existed, so no gate noticed; "exists" was being read as "the path is
there", not "the invocation runs".

These tests read the published command back out of the manuscript and check it
against the script's own ``argparse`` declarations: every flag it passes must be
declared, and every declared ``required=True`` option must be passed. That is a
static check — it does not prove the command succeeds, only that the repository
would accept its arguments — but it is exactly the class of defect that
shipped, and unlike running the gates it costs nothing and cannot go stale.

Scope: commands invoking ``scripts/*.py`` (directly or as ``-m scripts.<name>``),
whose arguments are declared by a single ``argparse`` parser in that file. Any
other printed ``.py`` path is still checked to exist in this repository — except
the commands §6.3 places in the docxology template checkout, which are exempted
by name below and whose exemption is itself pinned
to that prose claim.
"""

from __future__ import annotations

import ast
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "docs" / "manuscript"
# Authoring guides, not published sections — single-sourced with the token
# gate and the figure build (scripts/lib/manuscript_exclusions.py).
sys.path.insert(0, str(REPO_ROOT))
from scripts.lib.manuscript_exclusions import (
    AUTHORING_GUIDE_FILENAMES as _SKIP_DOCS,  # noqa: E402
)

# Executable command fences, however tagged. ``sh``/``shell`` are scanned
# beside ``bash`` because a fence is executable by what a reader runs it as,
# not by its tag. Plain ``` fences are NOT scanned: the manuscript's plain
# fences carry LaTeX (``preamble.md``), not shell, and no ``console`` fence
# exists — add a tag here only once a fence's content really is a runnable
# command line (or reword the fence in the manuscript to a non-executable tag).
_COMMAND_BLOCK_RE = re.compile(r"```(?:bash|shell|sh)\n(.*?)```", re.DOTALL)

# The render and validation stages belong to the docxology template. The manuscript
# says so in prose, and test_the_external_commands_are_still_declared_external below
# fails if that sentence ever disappears — so this exemption cannot become a way
# to hide a command this repository really is expected to provide.
_EXTERNAL_SCRIPTS = {
    "scripts/pipeline/stage_03_render.py",
    "scripts/pipeline/stage_04_validate.py",
}


def _published_commands() -> list[tuple[str, str]]:
    """Every shell command the manuscript sections print, as (source file, command)."""
    commands: list[tuple[str, str]] = []
    for md in sorted(MANUSCRIPT.glob("*.md")):
        if md.name in _SKIP_DOCS:
            continue
        for block in _COMMAND_BLOCK_RE.findall(md.read_text(encoding="utf-8")):
            # Join the printed backslash-continuations into single commands.
            joined = block.replace("\\\n", " ")
            for line in joined.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    commands.append((md.name, line))
    return commands


def _script_invocations() -> list[tuple[str, str, Path, list[str]]]:
    """(source file, command, script path, argv-after-script) for scripts/ commands."""
    found: list[tuple[str, str, Path, list[str]]] = []
    for source, command in _published_commands():
        tokens = shlex.split(command)
        for index, token in enumerate(tokens):
            script: Path | None = None
            if token.startswith("scripts/") and token.endswith(".py"):
                script = REPO_ROOT / token
            elif token == "-m" and index + 1 < len(tokens):
                module = tokens[index + 1]
                if module.startswith("scripts."):
                    script = REPO_ROOT / (module.replace(".", "/") + ".py")
            if script is not None:
                relative = script.relative_to(REPO_ROOT).as_posix()
                if relative in _EXTERNAL_SCRIPTS:
                    break
                rest = tokens[index + (2 if token == "-m" else 1) :]
                found.append((source, command, script, rest))
                break
    return found


def _declared_options(script: Path) -> tuple[set[str], set[str]]:
    """(every declared option string, the ones declared ``required=True``)."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    declared: set[str] = set()
    required: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        options = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("-")
        ]
        if not options:
            continue  # a positional argument
        declared.update(options)
        is_required = any(
            kw.arg == "required"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if is_required:
            required.add(options[0])
    return declared, required


def test_the_manuscript_prints_commands_to_check() -> None:
    """Guard against the scan silently matching nothing."""
    invocations = _script_invocations()
    assert invocations, "no scripts/ commands found in the manuscript's bash blocks"
    assert any(
        "run_model_family_acceptance" in str(script) for _, _, script, _ in invocations
    ), "the model-family acceptance command is no longer published"


def test_every_published_script_path_exists() -> None:
    for source, command, script, _rest in _script_invocations():
        assert script.is_file(), (
            f"{source} prints a command against a missing script: {command}"
        )


def test_every_published_flag_is_one_the_script_declares() -> None:
    """A renamed or removed flag must not survive in the published command."""
    problems: list[str] = []
    for source, command, script, rest in _script_invocations():
        declared, _required = _declared_options(script)
        if not declared:
            continue  # no argparse options declared in this file
        for token in rest:
            if not token.startswith("-"):
                continue
            flag = token.split("=", 1)[0]
            if flag not in declared:
                problems.append(f"{source}: {script.name} has no {flag} — {command}")
    assert not problems, "\n".join(problems)


def test_every_required_argument_is_supplied_by_the_published_command() -> None:
    """The regression itself: a published command must not exit 2 on its own arguments.

    ``run_model_family_acceptance.py`` declares ``--output-dir`` as
    ``required=True``; §6.2 printed the command without it, so the published
    reproduction step exited 2 while the section claimed it existed.
    """
    problems: list[str] = []
    for source, command, script, rest in _script_invocations():
        _declared, required = _declared_options(script)
        supplied = {token.split("=", 1)[0] for token in rest if token.startswith("-")}
        for flag in sorted(required - supplied):
            problems.append(
                f"{source}: {script.name} requires {flag}; the published command "
                f"omits it and exits 2 — {command}"
            )
    assert not problems, "\n".join(problems)


def test_the_external_commands_are_still_declared_external() -> None:
    """The manuscript must keep saying where the exempted commands run.

    Without this, ``_EXTERNAL_SCRIPTS`` would be a silent way to exempt any
    command whose script this repository stopped shipping.
    """
    text = (MANUSCRIPT / "05_reproducibility.md").read_text(encoding="utf-8")
    for relative in sorted(_EXTERNAL_SCRIPTS):
        assert relative in text, f"{relative} is exempted but no longer published"
    assert "separate checkout" in text, (
        "the manuscript no longer places the render and validation commands "
        "in the template checkout"
    )
