#!/usr/bin/env python3
"""Fail when the CLI flag surface and maintained docs drift apart.

Two ratcheted metrics keep the argparse-registered flags and the flags that
maintained Markdown mentions in visible tension:

Registered surface = the union of

- dynamic: ``option_strings`` across the gnn main parser's actions
  (``gnn.utils.arguments.arg_parsing.ArgumentParser.create_main_parser()``),
  minus ``-h``/``--help``; and
- static: ``additional_arguments`` keys of every ``src/gnn/[0-9]*_*.py`` call
  site, resolved per the runtime parser-selection semantics in
  ``pipeline_template._parse_step_args``. When every key of a step's dict is
  registered in ``ArgumentParser.ARGUMENT_DEFINITIONS``, the primary parser
  registers those keys under their dash-spelled definition flags (already
  covered by the dynamic scan), so the AST half adds nothing. When any key
  falls outside the definitions, the step deterministically runs the
  recovery parser, which registers each key verbatim as ``--{key}`` or under
  an explicit nested ``"flag"`` override; those spellings are added here.
  Module-level dict variables are resolved within the same file.

Doc surface = maintained Markdown (``README.md`` + ``docs/**/*.md`` +
``src/gnn/**/*.md``) with the same exclusions as
``scripts/check_doc_path_references.py`` (excluded parts, historical files
``CHANGELOG.md``/``VERSION_MAP.md``, historical prefixes), scanned for
``--[a-z0-9][a-z0-9_-]*`` tokens.

Metrics (caps in ``scripts/flag_parity_caps.json``):

- ``phantom_doc_flags``: doc tokens that no registered parser knows. Known
  third-party-CLI mentions in maintained docs (pytest ``--cov``, uv
  ``--extra``, docker ``--mount``, tool flags in framework guides, ...) make
  a naive zero-phantom goal wrong today - the ratchet is the honest
  mechanic until those mentions are tightened or the tooling sections move
  to per-tool reference pages.

Runner portability (batch-7 root-cause receipt): the wave-D phantom drift
("141 on ubuntu CI vs 140 locally") was a measurement confound, not a
varying parser. The full registered surface is a pure function of the
working tree - ``ARGUMENT_DEFINITIONS``/``STEP_ARGUMENTS`` are static
:class:`MappingProxyType` literals and ``create_main_parser`` is a plain
loop over them - so no platform, extras, or Python-version input exists
(every registered flag is identical on py3.10-3.14, bare interpreter or
project venv, and CI logs confirm the same 79/204/141). The measured drift
came from runs at different tree states/docs. Two guards keep the dynamic
import deterministic anyway: (1) the import is pinned to ``ROOT/src`` via
the module ``__file__`` so an installed ``gnn`` wheel can never shadow the
working tree; (2) the AST half is path-scoped under ``ROOT``.
- ``undocumented_registered``: registered flags with zero doc mentions.
  Same runner-invariance applies: with the import pinned to the working
  tree, the registered set cannot drift between runners.

Over-cap failures name the offending doc files/tokens. After fixing docs,
lower the cap in ``scripts/flag_parity_caps.json`` to the new measured value
- the ratchet only bites if it is driven down deliberately.

Fully deterministic: AST + regex over the working tree plus an in-process
argparse build; no network, no clock, no randomness.

Usage: uv run --extra dev python scripts/check_flag_parity.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_GLOB = "src/gnn/[0-9]*_*.py"
FLAG_TOKEN_RE = re.compile(r"(?<![a-zA-Z0-9_/-])--[a-z0-9][a-z0-9_-]*")
CAP_FILE = ROOT / "scripts" / "flag_parity_caps.json"

# Doc-tree exclusions, kept identical to scripts/check_doc_path_references.py
# and scripts/lib/shared.py so the gates scan the same maintained set.
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "output",
    "build",
    "dist",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    ".tox",
    ".eggs",
}
HISTORICAL_DOC_FILES = {"CHANGELOG.md", "VERSION_MAP.md"}
HISTORICAL_DOC_PREFIXES = (
    "docs/development/fleet-logs/",
    "docs/other/",
    "docs/sympy/",
)

CAP_KEYS = ("phantom_doc_flags", "undocumented_registered")


def load_caps() -> dict[str, int]:
    """Read the ratchet caps; a missing/malformed cap file is a gate error."""
    if not CAP_FILE.exists():
        raise SystemExit(
            f"check_flag_parity: missing cap file {CAP_FILE.relative_to(ROOT)} "
            '- create it as {"phantom_doc_flags": <int>, '
            '"undocumented_registered": <int>}.'
        )
    try:
        data = json.loads(CAP_FILE.read_text(encoding="utf-8"))
        caps = {key: int(data[key]) for key in CAP_KEYS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"check_flag_parity: unreadable cap file "
            f"{CAP_FILE.relative_to(ROOT)} ({exc}) - expected "
            '{"phantom_doc_flags": <int>, "undocumented_registered": <int>}.'
        ) from exc
    if any(cap < 0 for cap in caps.values()):
        raise SystemExit(
            f"check_flag_parity: negative cap in {CAP_FILE.relative_to(ROOT)}."
        )
    return caps


def iter_doc_files() -> list[Path]:
    """Maintained Markdown files, mirroring check_doc_path_references.py."""
    found: list[Path] = []
    readme = ROOT / "README.md"
    if readme.exists():
        found.append(readme)
    for base in (ROOT / "docs", ROOT / "src" / "gnn"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            rel = path.relative_to(ROOT).as_posix()
            if path.name in HISTORICAL_DOC_FILES or rel.startswith(
                HISTORICAL_DOC_PREFIXES
            ):
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            found.append(path)
    return found


def collect_doc_tokens() -> dict[str, set[str]]:
    """Map long flags and actual CLI short aliases to maintained mentions."""
    short_flags = sorted(
        flag
        for flag in cli_parser_flags()
        if flag.startswith("-") and not flag.startswith("--")
    )
    short_pattern = (
        re.compile(
            r"(?<![A-Za-z0-9_-])(?:"
            + "|".join(re.escape(flag) for flag in short_flags)
            + r")(?![A-Za-z0-9_-])"
        )
        if short_flags
        else None
    )
    tokens: dict[str, set[str]] = {}
    for path in iter_doc_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        matches = list(FLAG_TOKEN_RE.finditer(text))
        if short_pattern is not None:
            matches.extend(short_pattern.finditer(text))
        for match in matches:
            tokens.setdefault(match.group(0), set()).add(rel)
    return tokens


def _literal_dict_flags(node: ast.Dict) -> list[tuple[str, str | None]]:
    """(key, explicit-flag-or-None) pairs from a dict literal."""
    pairs: list[tuple[str, str | None]] = []
    for key, value in zip(node.keys, node.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        override = None
        if isinstance(value, ast.Dict):
            for k2, v2 in zip(value.keys, value.values):
                if (
                    isinstance(k2, ast.Constant)
                    and k2.value == "flag"
                    and isinstance(v2, ast.Constant)
                    and isinstance(v2.value, str)
                ):
                    override = v2.value
        pairs.append((key.value, override))
    return pairs


def static_registered_flags(arg_def_names: set[str]) -> set[str]:
    """Flags the per-step runtime parsers actually register.

    Mirrors ``pipeline_template._parse_step_args``: all keys registered ->
    the primary parser wins (dash-spelled definition flags, covered by the
    dynamic scan); any unregistered key -> the recovery parser registers
    every key verbatim or under its explicit ``"flag"`` override.
    """
    flags: set[str] = set()
    for path in sorted(ROOT.glob(ORCHESTRATOR_GLOB)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module_dicts: dict[str, list[tuple[str, str | None]]] = {}
        for stmt in tree.body:
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict)):
                continue
            target = next((t for t in stmt.targets if isinstance(t, ast.Name)), None)
            if target is not None:
                module_dicts[target.id] = _literal_dict_flags(stmt.value)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "additional_arguments":
                    continue
                if isinstance(kw.value, ast.Dict):
                    pairs = _literal_dict_flags(kw.value)
                elif isinstance(kw.value, ast.Name) and kw.value.id in module_dicts:
                    pairs = module_dicts[kw.value.id]
                else:
                    continue
                if all(key in arg_def_names for key, _ in pairs):
                    continue  # primary parser: dash-spelled defs, dynamically covered
                for key, override in pairs:
                    flags.add(override if override else f"--{key}")
    return flags


def registered_flags() -> set[str]:
    """Union of the dynamic main parser and the per-step recovery spellings.

    The dynamic half intentionally imports the repo-local source tree, never
    an installed ``gnn`` distribution: the import is pinned via
    ``ArgumentParser.__file__`` and fails closed when it resolves outside
    ``ROOT/src``. A runner with a non-editable ``gnn`` install (or a stale
    wheel on ``sys.path``) would otherwise measure a different, stale flag
    surface; the pin makes the measured set a pure function of the working
    tree, so counts are identical across runners and Python versions.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from gnn.utils.arguments import arg_parsing
        from gnn.utils.arguments.arg_parsing import ArgumentParser
    except ImportError as exc:
        raise SystemExit(
            "check_flag_parity: cannot import gnn.utils.arguments.arg_parsing "
            f"({exc}) - the parser registry moved; update this gate."
        ) from exc
    parser_module = Path(getattr(arg_parsing, "__file__", "") or "")
    try:
        parser_module.relative_to(ROOT / "src")
    except ValueError as exc:
        raise SystemExit(
            "check_flag_parity: imported the argument parser from "
            f"{parser_module}, which is outside {ROOT / 'src'} - an installed "
            "gnn package is shadowing the working tree. Measure the "
            "repository's own parser, not a stale wheel."
        ) from exc
    parser = ArgumentParser.create_main_parser()
    flags: set[str] = set()
    for action in parser._actions:  # noqa: SLF001 - argparse's own registry
        flags.update(action.option_strings)
    flags |= cli_parser_flags()
    flags -= {"-h", "--help"}
    # Maintained first-party gates and generated-script runtimes have their own
    # CLIs. Read their actual argparse registrations without executing them.
    for relative in (
        "scripts/check_manuscript_tokens.py",
        "scripts/check_hydrated_prose.py",
        "scripts/run_comprehensive_native_coverage.py",
        "src/gnn/render/thrml/runtime.py",
        "src/gnn/analysis/performance_benchmark.py",
    ):
        for node in ast.walk(ast.parse((ROOT / relative).read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                flags.update(
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                )
    return flags | static_registered_flags(set(ArgumentParser.ARGUMENT_DEFINITIONS))


def cli_parser_flags() -> set[str]:
    """Flags registered by the real ``gnn`` CLI, including subcommands.

    ``create_main_parser`` is the flat pipeline-arguments parser; the user-
    facing command surface lives in :func:`gnn.cli.build_parser` (serve,
    mcp, health, templates, ... with their own subparsers). Both surfaces
    are real registered spellings; docs may cite either.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from gnn.cli import build_parser
    except ImportError as exc:
        raise SystemExit(
            "check_flag_parity: cannot import gnn.cli.build_parser "
            f"({exc}) - the CLI surface moved; update this gate."
        ) from exc

    flags: set[str] = set()

    def _walk(parser: "argparse.ArgumentParser", seen: set[int]) -> None:
        if id(parser) in seen:
            return
        seen.add(id(parser))
        for action in parser._actions:  # noqa: SL001 - argparse's own registry
            flags.update(action.option_strings)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for sub in choices.values():
                    _walk(sub, seen)

    _walk(build_parser(), set())
    return flags - {"-h", "--help"}


def report_metric(metric: str, cap: int, measured: int) -> bool:
    """Print one metric's verdict; return True when over cap."""
    print(f"check_flag_parity: {metric} = {measured} (cap {cap}).")
    if measured < cap:
        print(
            f"  note: {metric} cap can be lowered to {measured} in "
            f"{CAP_FILE.relative_to(ROOT)} after this run."
        )
    return measured > cap


def main() -> int:
    """Compare both surfaces and apply the ratchets; exit 1 over any cap."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "CLI flag parity gate").splitlines()[0]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; cap breaches fail by default.",
    )
    args = parser.parse_args()
    del args  # the gate fails on any breach, with or without --strict

    caps = load_caps()
    doc_tokens = collect_doc_tokens()
    registered = registered_flags()

    phantom = {t: files for t, files in doc_tokens.items() if t not in registered}
    undocumented = sorted(registered - set(doc_tokens))
    print(
        f"check_flag_parity: {len(iter_doc_files())} maintained doc files, "
        f"{len(registered)} registered flags, {len(doc_tokens)} doc tokens."
    )

    failed = report_metric("phantom_doc_flags", caps["phantom_doc_flags"], len(phantom))
    if failed:
        for token in sorted(phantom):
            for rel in sorted(phantom[token]):
                print(f"  {rel}: {token}")
        print(
            "Fix the docs (correct or remove stale/foreign tokens; document "
            "real flags), then lower phantom_doc_flags in "
            f"{CAP_FILE.relative_to(ROOT)} to the new measured value."
        )
    failed |= report_metric(
        "undocumented_registered", caps["undocumented_registered"], len(undocumented)
    )
    if len(undocumented) > caps["undocumented_registered"]:
        for name in undocumented:
            print(f"  {name}")
        print(
            "Document these flags in maintained docs, then lower "
            "undocumented_registered in "
            f"{CAP_FILE.relative_to(ROOT)} to the new measured value."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
