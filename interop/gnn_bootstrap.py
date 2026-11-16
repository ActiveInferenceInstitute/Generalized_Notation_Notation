"""Downstream importer conformance: read a repository's GNN pin, validate its specs.

A repository that applies Generalized Notation Notation records which upstream
revision it consumes in a pin file, and keeps one or more ``*.gnn.md``
specifications that must parse under that revision. Every importer used to
carry its own script for this; the shapes drifted (``commit`` vs ``revision``,
``tag`` vs ``release``, four different file locations), so nothing could check
the estate as a whole.

This stdlib-only module is also the pre-import bootstrap: execute this exact,
reviewed file with the target interpreter's ``-I`` flag, independently of the
GNN revision that interpreter consumes. A separately trusted source manifest
binds that revision's complete package content, including its importer and
verdict modules. Consumer pins and their recorded parser hashes are not changed.

Canonical pin (``schema``: :data:`PIN_SCHEMA`)::

    {
      "schema": "gnn-importer-pin/v1",
      "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
      "tag": "v5.0.3",
      "commit": "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1",
      "recorded": "2026-11-12",
      "spec_files": ["models/gridworld_3x3.gnn.md"]
    }

Earlier shapes are read, never rewritten: ``revision`` is accepted for
``commit``, ``release`` for ``tag``, ``repository`` may be an ``org/name`` or a
URL, and ``spec_files`` is optional (absent means "discover ``*.gnn.md``").

Two shapes need a hint from the caller:

* **A two-party custody pin.** ``fep_formal`` records ``fep_lean`` and ``gnn``
  as separate sub-objects of ``specs/gnn-bridge-w2-source-custody/source-pin.json``
  with per-file SHA-256 owners instead of a release tag. ``--pin`` names the
  file and ``--pin-party gnn`` names the sub-record; the reader normalizes it
  to the same :class:`ImporterPin`.
* **A differently directed pair pin.** The upstream GNN checkout pins its
  downstreams in ``.github/gnn-pair.json``; discovery would read that as a GNN
  pin and report a fork. ``--pin`` is how a caller with a non-conventional pin
  states which file it means.

``--require-pointer`` additionally checks ``.github/GNN.md``: the pointer must
name the pin file, the pinned revision and — when given — the conformance
workflow and job key. It is opt-in because a repository adopts it deliberately,
in the same change that writes the pointer.

Usage (the reviewed bootstrap and consumed parser need not be the same release)::

    python -I /reviewed/gnn/importers.py source-manifest \
        --package-root /qualified/artifact/gnn --commit FULL_PINNED_COMMIT
    python -I /reviewed/gnn/importers.py check --repo /consumer \
        --source-manifest /trusted/source.json --source-manifest-sha256 TRUSTED_SHA256
    python -m gnn.importers check --repo . --no-validate --json

The manifest producer's source identity and this bootstrap file's own bytes
remain independently owner-qualified inputs. Do not derive the trusted manifest
hash from the installation being checked. ``-m`` imports GNN too early for
pre-import admission; use the standalone file for a real verdict. Metadata-only
inspection remains available without a manifest and never claims qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "IMPORTER_VARIABLES_SCHEMA",
    "PIN_SCHEMA",
    "POINTER_PATH",
    "ImporterPin",
    "PinError",
    "SOURCE_MANIFEST_SCHEMA",
    "SOURCE_QUALIFICATION_SCHEMA",
    "SourceQualificationError",
    "build_source_manifest",
    "check_importer",
    "check_pointer",
    "discover_specs",
    "find_pin",
    "importer_variables",
    "main",
    "normalize_repository",
    "qualified_source",
    "qualify_source",
    "read_pin",
    "source_manifest_bytes",
]

#: Schema identifier a canonical importer pin declares.
PIN_SCHEMA = "gnn-importer-pin/v1"
SOURCE_MANIFEST_SCHEMA = "gnn-importer-source/v1"
SOURCE_QUALIFICATION_SCHEMA = "gnn-importer-source-qualification/v1"
_DISTRIBUTION_NAME = "generalized-notation-notation"
_PACKAGE_PREFIX = "src/gnn/"

#: The upstream a pin is expected to name. A pin that names a fork is a real
#: finding: the consumer is tracking code this distribution does not publish.
UPSTREAM_REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"

#: Pin locations in precedence order. The first file that exists wins; the
#: order is canonical-location-first so a repository that adopts
#: ``.github/gnn-pin.json`` keeps one reader even if an older file remains.
PIN_CANDIDATES: Tuple[str, ...] = (
    ".github/gnn-pin.json",
    "specs/gnn-pin.json",
    "gnn/upstream.json",
    "models/gnn-pin.json",
    "models/gnn.pin.json",
    "NHIM/gnn-pin.json",
    ".github/gnn-pair.json",
)

#: The human-readable pointer every importer publishes beside its pin. It
#: records what the conformance job parses, so a reader can reach the pin, the
#: revision and the job from one file instead of reconstructing them from a
#: workflow definition.
POINTER_PATH = ".github/GNN.md"

#: Directories never searched for specifications.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "site-packages",
        "output",
        "cache",
        "tmp",
    }
)


class PinError(ValueError):
    """A pin file exists but does not record a usable upstream revision."""


class SourceQualificationError(ValueError):
    """Source admission failed before a qualified verdict could be issued."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ImporterPin:
    """A normalized importer pin.

    ``commit`` is the only required identity: a tag can move, a commit cannot.
    ``schema`` and ``recorded`` are ``None`` for pins that predate the
    canonical schema; :func:`check_importer` reports that as a finding rather
    than failing, so an importer can adopt the schema deliberately.
    """

    path: Path
    commit: str
    tag: Optional[str] = None
    repository: Optional[str] = None
    schema: Optional[str] = None
    recorded: Optional[str] = None
    spec_files: Tuple[str, ...] = ()
    #: Sub-record this pin was read from, for a two-party custody pin that
    #: nests one object per repository (``fep_formal`` records ``fep_lean`` and
    #: ``gnn`` in one file). ``None`` for the ordinary flat pin.
    party: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def normalize_repository(value: Any) -> Optional[str]:
    """Return ``org/name`` for a pin's repository field, or ``None``.

    Accepts the two forms in use: a bare ``org/name`` and a GitHub URL with or
    without a ``.git`` suffix.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith(("http://", "https://")):
        text = text.split("://", 1)[1]
        if text.startswith("github.com/"):
            text = text[len("github.com/") :]
    if text.startswith("git@github.com:"):
        text = text[len("git@github.com:") :]
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def find_pin(repo_root: Path) -> Optional[Path]:
    """Return the first pin file that exists under ``repo_root``."""
    for relative in PIN_CANDIDATES:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _relative_source_name(value: Any) -> str:
    """Admit a canonical, portable relative filename, never an escape."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\0" in value
    ):
        raise PinError(f"unsafe relative source path {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in (".", "..") for part in value.split("/"))
        or str(path) != value
    ):
        raise PinError(f"unsafe relative source path {value!r}")
    return value


def _pin_record(pin: ImporterPin) -> Dict[str, Any]:
    source = pin.raw.get(pin.party) if pin.party is not None else pin.raw
    if not isinstance(source, dict):
        raise PinError("pin sub-record is not an object")
    return source


def read_pin(path: Path, *, party: Optional[str] = None) -> ImporterPin:
    """Read and normalize one pin file.

    Args:
        path: The pin file.
        party: Sub-record to read, for a custody pin that nests one object per
            repository. ``None`` reads the file as a flat pin.

    Raises:
        PinError: the file is not JSON, records no commit/revision, or the
            requested ``party`` sub-record is absent.
    """
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_object
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise PinError(f"{path}: not readable JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise PinError(f"{path}: pin must be a JSON object")

    source = payload
    if party is not None:
        nested = payload.get(party)
        if not isinstance(nested, dict):
            raise PinError(
                f"{path}: records no {party!r} sub-record "
                f"(file names {sorted(payload)})"
            )
        source = nested

    commit = source.get("commit") or source.get("revision")
    if not isinstance(commit, str) or not commit.strip():
        raise PinError(f"{path}: pin records no commit/revision")
    if source.get("commit") and source.get("revision"):
        if source["commit"] != source["revision"]:
            raise PinError(f"{path}: commit and revision disagree")
    tag = source.get("tag") or source.get("release")
    if source.get("tag") and source.get("release"):
        if source["tag"] != source["release"]:
            raise PinError(f"{path}: tag and release disagree")
    repository = normalize_repository(source.get("repository"))
    if source.get("repository") is not None and repository is None:
        raise PinError(f"{path}: repository is not a usable repository identity")
    spec_files = source.get("spec_files", [])
    if isinstance(spec_files, str):
        spec_files = [spec_files]
    if not isinstance(spec_files, list):
        raise PinError(f"{path}: spec_files must be a string or list of strings")
    spec_files = [_relative_source_name(item) for item in spec_files]
    if len(set(spec_files)) != len(spec_files):
        raise PinError(f"{path}: spec_files contains duplicate paths")
    parser_files = source.get("parser_files", {})
    if not isinstance(parser_files, dict):
        raise PinError(f"{path}: parser_files must be an object of SHA-256 hashes")
    for relative, digest in parser_files.items():
        _relative_source_name(relative)
        if not relative.startswith((_PACKAGE_PREFIX, "gnn/")):
            raise PinError(f"{path}: parser file is outside the GNN package: {relative}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PinError(f"{path}: invalid parser SHA-256 for {relative}")
    return ImporterPin(
        path=path,
        commit=commit.strip(),
        tag=tag.strip() if isinstance(tag, str) and tag.strip() else None,
        repository=repository,
        schema=source.get("schema") if isinstance(source.get("schema"), str) else None,
        recorded=source.get("recorded")
        if isinstance(source.get("recorded"), str)
        else None,
        spec_files=tuple(spec_files),
        party=party,
        raw=payload,
    )


def _contained_spec(repo_root: Path, relative: str) -> Path:
    relative = _relative_source_name(relative)
    candidate = repo_root / relative
    current = repo_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise PinError(f"specification path contains a symlink: {relative}")
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise PinError(f"specification path escapes its repository: {relative}") from exc
    return candidate


def _walk_specs(repo_root: Path) -> Iterable[Path]:
    def fail(error: OSError) -> None:
        raise PinError(f"cannot discover specifications: {error}") from error

    for directory, directories, filenames in os.walk(repo_root, onerror=fail):
        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            if name.endswith(".gnn.md"):
                path = Path(directory) / name
                yield _contained_spec(repo_root, path.relative_to(repo_root).as_posix())


def discover_specs(repo_root: Path, pin: Optional[ImporterPin] = None) -> List[Path]:
    """Admit declared or discovered repository-local specification paths.

    Explicit ``spec_files: []`` is authoritative; only an absent declaration
    enables discovery. Symlinks and noncanonical/outside paths raise PinError
    before any specification content is read.
    """
    repo_root = repo_root.resolve()
    if pin is not None and (pin.spec_files or "spec_files" in _pin_record(pin)):
        return [_contained_spec(repo_root, relative) for relative in pin.spec_files]
    return sorted(_walk_specs(repo_root))


def _source_error(code: str, message: str) -> SourceQualificationError:
    return SourceQualificationError(code, message)


def _package_files(package_root: Path) -> Dict[str, str]:
    """Hash every regular package file except unused Python bytecode caches."""
    if package_root.is_symlink() or not package_root.is_dir():
        raise _source_error("source-root-invalid", "GNN package root is not an ordinary directory")
    files: Dict[str, str] = {}

    def fail(error: OSError) -> None:
        raise _source_error("source-unreadable", f"cannot inventory GNN source: {error}") from error

    try:
        for directory, directories, filenames in os.walk(package_root, onerror=fail):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            for name in directories:
                if (Path(directory) / name).is_symlink():
                    raise _source_error("source-path-unsafe", "GNN source contains a redirected directory")
            for name in sorted(filenames):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = Path(directory) / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise _source_error("source-path-unsafe", "GNN source contains a nonregular file")
                relative = path.relative_to(package_root).as_posix()
                _relative_source_name(relative)
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                files[_PACKAGE_PREFIX + relative] = digest.hexdigest()
    except (OSError, PinError) as exc:
        raise _source_error("source-unreadable", f"cannot read GNN source: {exc}") from exc
    return dict(sorted(files.items()))


def _module_source(module: str, files: Mapping[str, str]) -> Optional[str]:
    if not isinstance(module, str) or re.fullmatch(r"gnn(?:\.[A-Za-z_][A-Za-z_0-9]*)*", module) is None:
        return None
    stem = "src/" + module.replace(".", "/")
    for relative in (stem + "/__init__.py", stem + ".py"):
        if relative in files:
            return relative
    return None


def _manifest_record(payload: Any) -> Dict[str, Any]:
    fields = {"schema", "repository", "commit", "files", "verdict_modules"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise _source_error("source-manifest-invalid", "source manifest must contain exactly schema/repository/commit/files/verdict_modules")
    if payload["schema"] != SOURCE_MANIFEST_SCHEMA:
        raise _source_error("source-manifest-invalid", "unsupported source manifest schema")
    if payload["repository"] != UPSTREAM_REPOSITORY:
        raise _source_error("source-manifest-invalid", "source manifest names a different producer")
    if not isinstance(payload["commit"], str) or re.fullmatch(r"[0-9a-f]{40}", payload["commit"]) is None:
        raise _source_error("source-manifest-invalid", "source manifest requires a full lowercase commit")
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise _source_error("source-manifest-invalid", "source manifest requires a complete nonempty package inventory")
    try:
        for relative, digest in files.items():
            _relative_source_name(relative)
            if not relative.startswith(_PACKAGE_PREFIX):
                raise PinError("source manifest file is outside src/gnn/")
            parts = PurePosixPath(relative).parts
            if "__pycache__" in parts or relative.endswith((".pyc", ".pyo")):
                raise PinError("bytecode caches are not source manifest inputs")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise PinError(f"invalid source SHA-256 for {relative}")
    except PinError as exc:
        raise _source_error("source-manifest-invalid", str(exc)) from exc
    modules = payload["verdict_modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(module, str) for module in modules)
        or len(set(modules)) != len(modules)
    ):
        raise _source_error("source-manifest-invalid", "verdict_modules must be a nonempty unique string list")
    required = {"gnn", "gnn.parsers.basic"}
    if _PACKAGE_PREFIX + "importers.py" in files:
        required.add("gnn.importers")
    if not required.issubset(modules) or any(_module_source(module, files) is None for module in modules):
        raise _source_error("source-manifest-invalid", "manifest omits the package/importer/verdict source coverage")
    return payload


def build_source_manifest(
    package_root: Path,
    *,
    commit: str,
    verdict_modules: Sequence[str] = ("gnn.parsers.basic",),
) -> Dict[str, Any]:
    """Inventory an independently trusted, exact-revision GNN package.

    This producer does not discover or authenticate a Git revision. The owner
    supplies the already-qualified commit and package artifact. Freeze the
    resulting bytes/hash independently of the target being checked; generating
    a manifest from that target itself does not establish trusted provenance.
    All package files, not only a legacy pin's five parser files, are covered.
    """
    files = _package_files(Path(package_root))
    modules = set(verdict_modules) | {"gnn", "gnn.parsers.basic"}
    if _PACKAGE_PREFIX + "importers.py" in files:
        modules.add("gnn.importers")
    return _manifest_record(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "repository": UPSTREAM_REPOSITORY,
            "commit": commit,
            "files": files,
            "verdict_modules": sorted(modules),
        }
    )


def source_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return canonical ASCII JSON: sorted keys, compact separators, final LF."""
    payload = _manifest_record(dict(manifest))
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _source_origin(
    pin: ImporterPin, package_root: Path, origin_mode: str
) -> Dict[str, Any]:
    if origin_mode == "source":
        return {
            "kind": "explicit-source",
            "package_root": str(package_root),
            "identity_basis": "independently-trusted-manifest",
            "vcs_verified": False,
        }
    if origin_mode != "installed":
        raise _source_error("source-origin-invalid", "origin_mode must be installed or source")
    try:
        distribution = importlib.metadata.distribution(_DISTRIBUTION_NAME)
        if Path(distribution.locate_file("gnn/__init__.py")).resolve().parent != package_root:
            raise _source_error("source-origin-mismatch", "selected gnn package is not the named installed distribution")
        text = distribution.read_text("direct_url.json")
        if text is None:
            raise _source_error("source-origin-unavailable", "installed distribution has no exact VCS direct_url.json")
        origin = json.loads(text, object_pairs_hook=_json_object)
        if not isinstance(origin, dict):
            raise ValueError("direct_url.json is not an object")
        vcs = origin.get("vcs_info")
        if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
            raise _source_error("source-origin-unavailable", "installed origin does not record an exact Git revision")
        if vcs.get("commit_id") != pin.commit:
            raise _source_error("source-origin-mismatch", "installed origin commit differs from the consumer pin")
        if normalize_repository(origin.get("url")) != UPSTREAM_REPOSITORY:
            raise _source_error("source-origin-mismatch", "installed origin repository differs from the GNN producer")
        return {
            "kind": "installed-vcs",
            "distribution": _DISTRIBUTION_NAME,
            "version": distribution.version,
            "package_root": str(package_root),
            "repository": UPSTREAM_REPOSITORY,
            "commit": pin.commit,
            "direct_url_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "vcs_verified": True,
        }
    except (importlib.metadata.PackageNotFoundError, OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, SourceQualificationError):
            raise
        raise _source_error("source-origin-unavailable", f"installed origin is unavailable: {type(exc).__name__}") from exc


@dataclass
class _AdmittedSource:
    pin: ImporterPin
    package_root: Path
    files: Dict[str, str]
    receipt: Dict[str, Any]
    origin_mode: str

    def read(self, path: Path) -> bytes:
        try:
            relative = _PACKAGE_PREFIX + path.relative_to(self.package_root).as_posix()
            expected = self.files[relative]
            if not stat.S_ISREG(path.lstat().st_mode):
                raise OSError("source is not a regular file")
            data = path.read_bytes()
        except (OSError, KeyError, ValueError) as exc:
            raise _source_error("source-path-unsafe", "import requested undeclared or redirected GNN content") from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise _source_error("source-content-mismatch", f"GNN source changed before use: {relative}")
        return data

    def verify_unchanged(self) -> None:
        if _package_files(self.package_root) != self.files:
            raise _source_error("source-content-mismatch", "GNN package inventory changed during qualification")
        if _source_origin(self.pin, self.package_root, self.origin_mode) != self.receipt["origin"]:
            raise _source_error("source-origin-mismatch", "GNN origin changed during qualification")


_ACTIVE_SOURCE: Optional[_AdmittedSource] = None


def _admit_source(
    pin: ImporterPin,
    *,
    manifest: bytes,
    manifest_sha256: str,
    package_root: Optional[Path],
    origin_mode: str,
) -> _AdmittedSource:
    if any(name == "gnn" or name.startswith("gnn.") for name in sys.modules):
        raise _source_error("source-already-imported", "qualification requires a fresh process before importing any gnn module")
    if (
        not isinstance(manifest, bytes)
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
        or hashlib.sha256(manifest).hexdigest() != manifest_sha256
    ):
        raise _source_error("source-manifest-hash-mismatch", "manifest bytes do not match their independently trusted SHA-256")
    try:
        payload = _manifest_record(
            json.loads(manifest.decode("utf-8"), object_pairs_hook=_json_object)
        )
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, SourceQualificationError):
            raise
        raise _source_error("source-manifest-invalid", f"source manifest is not valid JSON: {exc}") from exc
    if payload["commit"] != pin.commit or pin.repository not in (None, UPSTREAM_REPOSITORY):
        raise _source_error("source-pin-mismatch", "source manifest does not bind the consumer's exact repository/commit")
    if origin_mode == "source" and package_root is None:
        raise _source_error("source-root-invalid", "source mode requires an explicit package_root")
    try:
        spec = importlib.util.find_spec("gnn")
        if spec is None or spec.origin is None or spec.submodule_search_locations is None:
            raise _source_error("source-origin-unavailable", "target interpreter cannot locate an ordinary gnn package")
        selected = Path(spec.origin).resolve().parent
        if package_root is not None and Path(package_root).resolve() != selected:
            raise _source_error("source-origin-mismatch", "import resolution selects a different GNN package root")
        files = _package_files(selected)
        if files != payload["files"]:
            missing = sorted(set(payload["files"]) - set(files))
            extra = sorted(set(files) - set(payload["files"]))
            changed = sorted(path for path in files.keys() & payload["files"].keys() if files[path] != payload["files"][path])
            raise _source_error(
                "source-content-mismatch",
                f"GNN package differs from trusted inventory (missing={missing}, extra={extra}, changed={changed})",
            )
        parser_files = _pin_record(pin).get("parser_files", {})
        for relative, expected in parser_files.items():
            canonical = relative if relative.startswith(_PACKAGE_PREFIX) else "src/" + relative
            if files.get(canonical) != expected:
                raise _source_error("source-parser-digest-mismatch", f"recorded parser digest differs: {relative}")
        origin = _source_origin(pin, selected, origin_mode)
        bootstrap = Path(__file__).resolve()
        receipt = {
            "schema": SOURCE_QUALIFICATION_SCHEMA,
            "ok": True,
            "status": "admitted",
            "repository": UPSTREAM_REPOSITORY,
            "commit": pin.commit,
            "manifest_sha256": manifest_sha256,
            "package_root": str(selected),
            "origin": origin,
            "file_count": len(files),
            "verdict_modules": list(payload["verdict_modules"]),
            "pinned_parser_files": dict(parser_files),
            "importer_coverage": "pinned-module" if _PACKAGE_PREFIX + "importers.py" in files else "external-bootstrap",
            "bootstrap": {"path": str(bootstrap), "sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest()},
            "preimport": True,
            "imports_verified": False,
            "loaded_modules": {},
            "execution_verified": False,
            "scope": "GNN package source only; not dependency, semantic-result, publisher, or hostile-filesystem attestation",
        }
        return _AdmittedSource(pin, selected, files, receipt, origin_mode)
    except (OSError, RuntimeError, ImportError) as exc:
        raise _source_error("source-origin-unavailable", f"cannot admit GNN package: {type(exc).__name__}") from exc


def qualify_source(
    pin: ImporterPin,
    *,
    manifest: bytes,
    manifest_sha256: str,
    package_root: Optional[Path] = None,
    origin_mode: str = "installed",
) -> Dict[str, Any]:
    """Admit exact target origin/content without importing GNN.

    Load this reviewed file as a standalone module, not via ``import gnn``.
    The manifest hash must come from independent owner custody, not from a
    manifest generated by the target installation. Installed mode additionally
    requires matching PEP 610 Git origin. Explicit source mode binds the selected
    package root by complete content, without claiming Git checkout verification.
    Use qualified_source for verdict execution, not this observation alone.
    """
    return _admit_source(
        pin,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        package_root=package_root,
        origin_mode=origin_mode,
    ).receipt


class _CheckedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name: str, path: Path, admitted: _AdmittedSource) -> None:
        super().__init__(name, str(path))
        self.admitted = admitted

    def get_data(self, path: str) -> bytes:
        return self.admitted.read(Path(path))

    def get_code(self, fullname: str) -> Any:
        # Compile the checked bytes, never a stale or substituted .pyc.
        return compile(self.get_data(self.path), self.path, "exec", dont_inherit=True)


class _CheckedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, admitted: _AdmittedSource) -> None:
        self.admitted = admitted
        self.directories = {
            parent.as_posix()
            for relative in admitted.files
            for parent in PurePosixPath(relative).parents
        }

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "gnn" and not fullname.startswith("gnn."):
            return None
        relative = _module_source(fullname, self.admitted.files)
        if relative is None:
            directory = "src/" + fullname.replace(".", "/")
            if directory not in self.directories:
                raise _source_error("source-module-undeclared", f"module is outside the admitted GNN source: {fullname}")
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [
                str(self.admitted.package_root / directory[len("src/gnn/"):])
            ]
            return spec
        source = self.admitted.package_root / relative[len(_PACKAGE_PREFIX):]
        loader = _CheckedSourceLoader(fullname, source, self.admitted)
        locations = [str(source.parent)] if source.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(
            fullname, source, loader=loader, submodule_search_locations=locations
        )


@contextmanager
def qualified_source(
    pin: ImporterPin,
    *,
    manifest: bytes,
    manifest_sha256: str,
    package_root: Optional[Path] = None,
    origin_mode: str = "installed",
) -> Iterator[Dict[str, Any]]:
    """Admit source before imports and guard exact bytes for one verdict scope.

    Run in a fresh target process. This supports legacy v5.0.3 packages without
    gnn.importers as well as v5.1.0 packages whose importer is itself covered.
    Import the pinned verdict modules only inside this context. On normal exit
    the inventory/origin and every loaded GNN module are rechecked; only then
    imports_verified becomes true. It is not scientific execution attestation.
    """
    global _ACTIVE_SOURCE
    if _ACTIVE_SOURCE is not None:
        raise _source_error("source-session-active", "a GNN source qualification is already active")
    admitted = _admit_source(
        pin,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        package_root=package_root,
        origin_mode=origin_mode,
    )
    finder = _CheckedSourceFinder(admitted)
    admitted.receipt["ok"] = False
    _ACTIVE_SOURCE = admitted
    sys.meta_path.insert(0, finder)
    try:
        yield admitted.receipt
        loaded: Dict[str, Any] = {}
        for name, module in tuple(sys.modules.items()):
            if name != "gnn" and not name.startswith("gnn."):
                continue
            spec = getattr(module, "__spec__", None)
            loader = getattr(spec, "loader", None)
            if isinstance(loader, _CheckedSourceLoader) and loader.admitted is admitted:
                source = Path(loader.path)
                if (
                    getattr(module, "__file__", None) != loader.path
                    or getattr(spec, "origin", None) != loader.path
                ):
                    raise _source_error(
                        "source-loaded-module-mismatch",
                        f"GNN module changed its admitted origin: {name}",
                    )
                admitted.read(source)
                relative = _PACKAGE_PREFIX + source.relative_to(admitted.package_root).as_posix()
                loaded[name] = {"path": loader.path, "sha256": admitted.files[relative]}
                continue
            directory = "src/" + name.replace(".", "/")
            expected = str(admitted.package_root / directory[len(_PACKAGE_PREFIX):])
            if (
                directory not in finder.directories
                or _module_source(name, admitted.files) is not None
                or spec is None
                or spec.origin is not None
                or list(spec.submodule_search_locations or ()) != [expected]
            ):
                raise _source_error(
                    "source-loaded-module-mismatch",
                    f"GNN module did not use admitted source: {name}",
                )
            loaded[name] = {"path": expected, "sha256": None, "namespace": True}
        admitted.verify_unchanged()
        admitted.receipt.update(
            ok=True,
            status="qualified",
            imports_verified=True,
            loaded_modules=loaded,
        )
    except BaseException:
        admitted.receipt.update(ok=False, status="failed", imports_verified=False)
        raise
    finally:
        sys.meta_path[:] = [item for item in sys.meta_path if item is not finder]
        _ACTIVE_SOURCE = None


def _validate_spec(path: Path) -> List[str]:
    """Validate one specification with the installed parser; return errors."""
    from gnn.parsers.basic import validate_gnn_syntax

    try:
        is_valid, errors = validate_gnn_syntax(path)
    except SourceQualificationError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
    ) as exc:
        # A document from another repository is untrusted input: a validator
        # failure on it is a finding about that document, not a crash here.
        return [f"validator raised {type(exc).__name__}: {exc}"]
    if is_valid:
        return []
    return [str(error) for error in errors] or ["invalid specification"]


def _relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when it is inside it, else the name."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def check_pointer(
    repo_root: Path,
    pin: ImporterPin,
    *,
    workflow: Optional[str] = None,
    job: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Check the importer's ``.github/GNN.md`` pointer against its pin.

    The pointer is what a human reads and what the estate's online re-read
    looks for; a pointer that names a different pin, revision or job is a
    finding, not a documentation preference. Every check is textual: the
    pointer is prose, so the reader asks whether the facts appear, not whether
    the file has a schema.
    """
    findings: List[Dict[str, str]] = []
    path = repo_root / POINTER_PATH
    if not path.is_file():
        return [
            {
                "level": "error",
                "code": "pointer-missing",
                "message": f"{POINTER_PATH} is absent; it must name {_relative(pin.path, repo_root)}, "
                f"{pin.commit[:12]} and the conformance job",
            }
        ]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [
            {
                "level": "error",
                "code": "pointer-unreadable",
                "message": f"{POINTER_PATH} is not readable text ({exc})",
            }
        ]

    pin_name = _relative(pin.path, repo_root)
    if pin_name not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-pin-unnamed",
                "message": f"{POINTER_PATH} does not name the pin file {pin_name}",
            }
        )
    if pin.commit not in text and pin.commit[:12] not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-revision-unnamed",
                "message": f"{POINTER_PATH} does not name the pinned revision {pin.commit[:12]}",
            }
        )
    if workflow and Path(workflow).name not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-workflow-unnamed",
                "message": f"{POINTER_PATH} does not name the workflow {workflow}",
            }
        )
    if job and job not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-job-unnamed",
                "message": f"{POINTER_PATH} does not name the job key {job}",
            }
        )
    return findings


#: Schema identifier for the GNN-derived subset of a manuscript variable map.
IMPORTER_VARIABLES_SCHEMA = "gnn-importer-variables/v1"


def importer_variables(
    pin: ImporterPin,
    *,
    repo_root: Path,
    job: Optional[str] = None,
    job_name: Optional[str] = None,
) -> Dict[str, str]:
    """Return the GNN-derived manuscript variables for one importer.

    A repository whose manuscript states which GNN release it consumes needs
    the same seven facts this reader already normalizes. Producing them here
    means the prose and the conformance job cannot disagree about the pin, and
    a consumer does not re-read the JSON in its own variable script.

    ``gnn_job_key`` and ``gnn_job_name`` are the conformance job the registry
    row records; a caller that has them passes them, because the reader does
    not parse workflow YAML.
    """
    return {
        "schema": IMPORTER_VARIABLES_SCHEMA,
        "gnn_pin_path": _relative(pin.path, repo_root),
        "gnn_pin_schema": pin.schema or "",
        "gnn_repository": pin.repository or "",
        "gnn_tag": pin.tag or "",
        "gnn_commit": pin.commit,
        "gnn_job_key": job or "",
        "gnn_job_name": job_name or "",
    }


def check_importer(
    repo_root: Path,
    *,
    expected_commit: Optional[str] = None,
    expected_tag: Optional[str] = None,
    expected_repository: Optional[str] = UPSTREAM_REPOSITORY,
    validate_specs: bool = True,
    require_pin: bool = True,
    pin_path: Optional[Path] = None,
    pin_party: Optional[str] = None,
    require_pointer: bool = False,
    pointer_workflow: Optional[str] = None,
    pointer_job: Optional[str] = None,
    source_manifest: Optional[Union[bytes, Path]] = None,
    source_manifest_sha256: Optional[str] = None,
    package_root: Optional[Path] = None,
    origin_mode: str = "installed",
) -> Dict[str, Any]:
    """Check one importer checkout and return a JSON-ready report.

    Findings carry ``level`` (``error`` fails the check, ``warning`` does not)
    and ``code`` so a caller can filter without parsing prose.

    Args:
        pin_path: Read this pin instead of discovering one. Needed where a
            repository keeps a differently named custody pin
            (``fep_formal``'s ``source-pin.json``) or where the conventional
            candidate list would pick a *different repository's* pin (the
            upstream GNN checkout pins its downstreams in
            ``.github/gnn-pair.json``).
        pin_party: Sub-record to read inside ``pin_path`` for a two-party pin.
        require_pointer: Also require ``.github/GNN.md`` to name the pin, the
            revision and the declared workflow and job.
        source_manifest: Independently trusted complete-package manifest bytes,
            or a path to those exact bytes. Required before a real verdict.
        source_manifest_sha256: SHA-256 bound by independent producer custody.
        package_root: Explicit expected target package directory; required in
            source mode. Never changes import resolution or sys.path.
        origin_mode: Installed VCS origin plus content, or explicitly selected
            source content. Neither mode changes the consumer pin.
        validate_specs: False is inspection only and never imports GNN here.
    """
    repo_root = repo_root.resolve()
    findings: List[Dict[str, str]] = []
    report: Dict[str, Any] = {
        "repo": str(repo_root),
        "pin": None,
        "specs": {"count": 0, "files": [], "invalid": [], "validated": 0},
        "findings": findings,
        "ok": False,
        "mode": "conformance" if validate_specs else "inspection",
        "qualified": False,
        "verdict_scope": "normative-syntax" if validate_specs else None,
        "source_qualification": {
            "schema": SOURCE_QUALIFICATION_SCHEMA,
            "ok": False,
            "status": "not-requested",
            "preimport": False,
            "imports_verified": False,
            "loaded_modules": {},
            "execution_verified": False,
        },
    }
    if not repo_root.is_dir():
        findings.append(
            {"level": "error", "code": "repo-missing", "message": "importer repository is not an existing directory"}
        )
        return report

    pin_file = find_pin(repo_root)
    if pin_path is not None:
        pin_file = pin_path if pin_path.is_absolute() else repo_root / pin_path
        if not pin_file.is_file():
            findings.append(
                {
                    "level": "error",
                    "code": "pin-missing",
                    "message": f"the declared pin {_relative(pin_file, repo_root)} is absent",
                }
            )
            pin_file = None
    pin: Optional[ImporterPin] = None
    if pin_file is None:
        if pin_path is not None:
            pass  # the absent declared pin is already the finding
        elif require_pin:
            findings.append(
                {
                    "level": "error",
                    "code": "pin-missing",
                    "message": "no GNN pin file found (looked for "
                    + ", ".join(PIN_CANDIDATES)
                    + ")",
                }
            )
        else:
            findings.append(
                {
                    "level": "warning",
                    "code": "pin-missing",
                    "message": "no GNN pin file; exact-pin verdicts are unavailable",
                }
            )
    else:
        try:
            pin = read_pin(pin_file, party=pin_party)
        except PinError as exc:
            findings.append(
                {"level": "error", "code": "pin-unreadable", "message": str(exc)}
            )
        else:
            report["pin"] = {
                "path": _relative(pin.path, repo_root),
                "party": pin.party,
                "commit": pin.commit,
                "tag": pin.tag,
                "repository": pin.repository,
                "schema": pin.schema,
                "recorded": pin.recorded,
                "spec_files": list(pin.spec_files),
            }
            if not re.fullmatch(r"[0-9a-f]{40}", pin.commit):
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-commit-shape",
                        "message": f"pin records {pin.commit!r}, which is not a 40-character commit id",
                    }
                )
            if pin.schema != PIN_SCHEMA:
                findings.append(
                    {
                        "level": "warning",
                        "code": "pin-schema-unversioned",
                        "message": f"pin declares schema {pin.schema!r}, not {PIN_SCHEMA!r}",
                    }
                )
            if expected_commit and pin.commit != expected_commit:
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-commit-mismatch",
                        "message": f"pin records {pin.commit}, expected {expected_commit}",
                    }
                )
            if expected_tag and pin.tag != expected_tag:
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-tag-mismatch",
                        "message": f"pin records tag {pin.tag!r}, expected {expected_tag!r}",
                    }
                )
            if (
                expected_repository
                and pin.repository
                and pin.repository != expected_repository
            ):
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-repository-mismatch",
                        "message": (
                            f"pin names {pin.repository}, expected {expected_repository}; "
                            "a pin that names a fork tracks code this distribution does not publish"
                        ),
                    }
                )

    try:
        specs = discover_specs(repo_root, pin)
    except (PinError, OSError) as exc:
        specs = []
        findings.append(
            {"level": "error", "code": "spec-path-unsafe", "message": str(exc)}
        )
    report["specs"]["count"] = len(specs)
    report["specs"]["files"] = [path.relative_to(repo_root).as_posix() for path in specs]
    available: List[Path] = []
    for spec in specs:
        if not spec.is_file():
            findings.append(
                {
                    "level": "error",
                    "code": "spec-missing",
                    "message": f"{spec.relative_to(repo_root)} is declared by the pin but absent",
                }
            )
        else:
            available.append(spec)

    def validate() -> None:
        for spec in available:
            # Recheck path admission immediately before the validator opens it.
            _contained_spec(repo_root, spec.relative_to(repo_root).as_posix())
            errors = _validate_spec(spec)
            report["specs"]["validated"] += 1
            if errors:
                report["specs"]["invalid"].append(
                    {"file": spec.relative_to(repo_root).as_posix(), "errors": errors}
                )
                findings.append(
                    {
                        "level": "error",
                        "code": "spec-invalid",
                        "message": f"{spec.relative_to(repo_root)}: {errors[0]}",
                    }
                )

    if validate_specs and available and not any(item["level"] == "error" for item in findings):
        try:
            if pin is None:
                raise _source_error("source-pin-required", "exact-pin validation requires a readable consumer pin")
            if _ACTIVE_SOURCE is not None:
                admitted = _ACTIVE_SOURCE
                if admitted.pin.commit != pin.commit or pin.repository not in (None, UPSTREAM_REPOSITORY):
                    raise _source_error("source-pin-mismatch", "active source qualification belongs to a different pin")
                for relative, digest in _pin_record(pin).get("parser_files", {}).items():
                    canonical = relative if relative.startswith(_PACKAGE_PREFIX) else "src/" + relative
                    if admitted.files.get(canonical) != digest:
                        raise _source_error("source-parser-digest-mismatch", f"recorded parser digest differs: {relative}")
                report["source_qualification"] = admitted.receipt
                validate()
            else:
                if source_manifest is None or source_manifest_sha256 is None:
                    raise _source_error(
                        "source-manifest-required",
                        "use the reviewed standalone bootstrap with an independently trusted source manifest/hash",
                    )
                try:
                    manifest_bytes = source_manifest.read_bytes() if isinstance(source_manifest, Path) else source_manifest
                except OSError as exc:
                    raise _source_error("source-manifest-unreadable", "source manifest cannot be read") from exc
                with qualified_source(
                    pin,
                    manifest=manifest_bytes,
                    manifest_sha256=source_manifest_sha256,
                    package_root=package_root,
                    origin_mode=origin_mode,
                ) as qualification:
                    report["source_qualification"] = qualification
                    validate()
        except (SourceQualificationError, PinError, ImportError, OSError) as exc:
            code = exc.code if isinstance(exc, SourceQualificationError) else "source-verdict-unavailable"
            report["source_qualification"].update(ok=False, status="failed")
            findings.append({"level": "error", "code": code, "message": str(exc)})

    if pin is not None and not specs:
        findings.append(
            {
                "level": "warning",
                "code": "spec-absent",
                "message": "pin present but the repository carries no *.gnn.md specification",
            }
        )

    if require_pointer:
        if pin is None:
            findings.append(
                {
                    "level": "error",
                    "code": "pointer-unchecked",
                    "message": f"{POINTER_PATH} cannot be checked without a readable pin",
                }
            )
        else:
            pointer_findings = check_pointer(
                repo_root,
                pin,
                workflow=pointer_workflow,
                job=pointer_job,
            )
            findings.extend(pointer_findings)
            report["pointer"] = {
                "path": POINTER_PATH,
                "ok": not pointer_findings,
            }

    report["ok"] = not any(item["level"] == "error" for item in findings)
    report["qualified"] = (
        report["ok"]
        and validate_specs
        and report["specs"]["validated"] > 0
        and report["source_qualification"]["ok"]
        and report["source_qualification"]["imports_verified"]
    )
    return report


def _format_text(report: Dict[str, Any]) -> str:
    lines = [f"importer: {report['repo']}"]
    pin = report["pin"]
    if pin:
        party = f" [{pin['party']}]" if pin.get("party") else ""
        lines.append(
            f"  pin: {pin['path']}{party} -> {pin['tag'] or '(no tag)'} @ {pin['commit'][:10]}"
            f" (schema {pin['schema'] or 'unversioned'})"
        )
    else:
        lines.append("  pin: none")
    lines.append(f"  specs: {report['specs']['count']}")
    pointer = report.get("pointer")
    if pointer:
        lines.append(
            f"  pointer: {pointer['path']} -> {'ok' if pointer['ok'] else 'FAILED'}"
        )
    for item in report["findings"]:
        lines.append(f"  [{item['level']}] {item['code']}: {item['message']}")
    lines.append("  verdict: " + ("ok" if report["ok"] else "FAILED"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run inspection or qualified checking; this file also runs with ``-I``."""
    parser = argparse.ArgumentParser(
        prog="python -m gnn.importers",
        description="Check a downstream repository's GNN pin and specifications.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser(
        "source-manifest", help="inventory an independently qualified exact-revision package"
    )
    manifest_parser.add_argument("--package-root", type=Path, required=True)
    manifest_parser.add_argument("--commit", required=True)
    manifest_parser.add_argument("--verdict-module", action="append", default=[])
    check = subparsers.add_parser("check", help="check one importer checkout")
    check.add_argument("--repo", default=".", help="repository root to check")
    check.add_argument(
        "--expected-commit", default=None, help="upstream commit the pin must record"
    )
    check.add_argument(
        "--expected-tag", default=None, help="upstream tag the pin must record"
    )
    check.add_argument(
        "--expected-repository",
        default=UPSTREAM_REPOSITORY,
        help="repository the pin must name (empty string disables the check)",
    )
    check.add_argument(
        "--no-validate",
        action="store_true",
        help="read the pin and list specifications without parsing them",
    )
    check.add_argument(
        "--allow-missing-pin",
        action="store_true",
        help="treat a missing pin as a warning instead of an error",
    )
    check.add_argument(
        "--pin",
        default=None,
        help="read this pin instead of discovering one (for a differently named custody pin)",
    )
    check.add_argument(
        "--pin-party",
        default=None,
        help="sub-record to read inside --pin for a two-party custody pin",
    )
    check.add_argument(
        "--require-pointer",
        action="store_true",
        help=f"also require {POINTER_PATH} to name the pin, revision, workflow and job",
    )
    check.add_argument(
        "--workflow",
        default=None,
        help="conformance workflow the pointer must name (with --require-pointer)",
    )
    check.add_argument(
        "--job",
        default=None,
        help="conformance job key the pointer must name (with --require-pointer)",
    )
    check.add_argument("--source-manifest", type=Path, default=None)
    check.add_argument("--source-manifest-sha256", default=None)
    check.add_argument("--package-root", type=Path, default=None)
    check.add_argument("--origin-mode", choices=("installed", "source"), default="installed")
    check.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)
    if args.command == "source-manifest":
        try:
            payload = source_manifest_bytes(
                build_source_manifest(
                    args.package_root,
                    commit=args.commit,
                    verdict_modules=args.verdict_module or ("gnn.parsers.basic",),
                )
            )
        except (SourceQualificationError, OSError) as exc:
            print(json.dumps({"ok": False, "error": {"code": getattr(exc, "code", "source-unreadable"), "message": str(exc)}}))
            return 1
        sys.stdout.buffer.write(payload)
        return 0

    report = check_importer(
        Path(args.repo),
        expected_commit=args.expected_commit,
        expected_tag=args.expected_tag,
        expected_repository=args.expected_repository or None,
        validate_specs=not args.no_validate,
        require_pin=not args.allow_missing_pin,
        pin_path=Path(args.pin) if args.pin else None,
        pin_party=args.pin_party,
        require_pointer=args.require_pointer,
        pointer_workflow=args.workflow,
        pointer_job=args.job,
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        package_root=args.package_root,
        origin_mode=args.origin_mode,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
