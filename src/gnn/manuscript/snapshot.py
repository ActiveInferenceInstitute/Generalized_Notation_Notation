"""Read-only repository snapshot backing every manuscript count.

:class:`RepositorySnapshot` fixes the producer's source of truth to one
commit: the file set comes from ``git ls-tree`` and the bytes from
``git cat-file``, so published counts describe exactly that commit rather
than whatever is on disk, and any checkout of it reproduces them. When git
is unavailable the snapshot degrades to the working tree and the
``GNN_GIT_COMMIT`` token reports the ``unknown`` sentinel the gates fail on.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path

# Directories that are NOT counted as authored source when walking ``src/``.
_EXCLUDED_DIR_PARTS = {
    "__pycache__",
    ".venv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in path.parts)


class RepositorySnapshot:
    """Read-only view of the repository at one commit.

    Every published count is computed from a snapshot, so the numbers the
    manuscript prints describe exactly the commit ``GNN_GIT_COMMIT`` names
    rather than whatever happens to be on disk. Two properties follow that the
    older ``git ls-files`` + working-tree-read approach did not have:

    * a tracked file with uncommitted edits contributes its *committed* bytes,
      so a dirty tree cannot move a published line count;
    * a tracked file deleted in the working tree is still counted, so a
      half-finished refactor cannot silently shrink one.

    ``commit`` is the resolved short SHA, or ``"unknown"`` when git is not
    available (a source tarball, a vendored copy). In that case the snapshot
    degrades to the working tree and the ``unknown`` commit token is the signal
    that the numbers are checkout-dependent.
    """

    def __init__(self, project_root: Path, revision: str = "HEAD") -> None:
        self.project_root = Path(project_root).resolve()
        self.commit = self._resolve_commit(revision)
        self._paths: frozenset[Path] | None = None
        self._cache: dict[Path, str] = {}
        self._revision = revision if self.commit != "unknown" else ""

    # -- construction helpers ------------------------------------------------
    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes] | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.project_root,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            return None

    def _resolve_commit(self, revision: str) -> str:
        result = self._git("rev-parse", "--short", revision)
        if result is None or result.returncode != 0:
            return "unknown"
        return result.stdout.decode("utf-8", "replace").strip() or "unknown"

    # -- public surface ------------------------------------------------------
    @property
    def from_git(self) -> bool:
        """True when the snapshot reads committed blobs rather than the disk."""
        return bool(self._revision)

    def files(self) -> frozenset[Path]:
        """Repo-relative paths present in the snapshot."""
        if self._paths is not None:
            return self._paths
        paths: set[Path] = set()
        if self._revision:
            result = self._git("ls-tree", "-r", "-z", "--name-only", self._revision)
            if result is not None and result.returncode == 0:
                for entry in result.stdout.decode("utf-8", "replace").split("\0"):
                    if entry:
                        paths.add(Path(entry))
        if not paths:
            # Working-tree fallback: walk the checkout, minus build/vcs noise.
            for path in self.project_root.rglob("*"):
                if not path.is_file() or _is_excluded(path):
                    continue
                paths.add(path.relative_to(self.project_root))
        self._paths = frozenset(paths)
        return self._paths

    def exists(self, rel: Path | str) -> bool:
        """True when *rel* is present in the snapshot."""
        return Path(rel) in self.files()

    def prefetch(self, rels: Sequence[Path]) -> None:
        """Read many blobs in one ``git cat-file --batch`` call.

        Counting the source tree one ``git show`` per file costs a subprocess
        per file; the batch protocol makes the whole snapshot one process.
        """
        if not self._revision:
            return
        wanted = [rel for rel in rels if rel not in self._cache]
        if not wanted:
            return
        payload = "".join(f"{self._revision}:{rel.as_posix()}\n" for rel in wanted)
        try:
            result = subprocess.run(
                ["git", "cat-file", "--batch"],
                cwd=self.project_root,
                input=payload.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            return
        if result.returncode != 0:
            return
        out = result.stdout
        pos = 0
        for rel in wanted:
            newline = out.find(b"\n", pos)
            if newline < 0:
                break
            header = out[pos:newline].decode("utf-8", "replace")
            pos = newline + 1
            parts = header.rsplit(" ", 2)
            if len(parts) != 3 or not parts[2].isdigit():
                # "<name> missing" — record the absence and keep parsing.
                self._cache[rel] = ""
                continue
            size = int(parts[2])
            self._cache[rel] = out[pos : pos + size].decode("utf-8", "replace")
            pos += size + 1

    def read_text(self, rel: Path | str) -> str:
        """Return the snapshot's bytes for *rel* decoded as UTF-8 (lossy)."""
        rel = Path(rel)
        cached = self._cache.get(rel)
        if cached is not None:
            return cached
        data = ""
        if self._revision:
            result = self._git("show", f"{self._revision}:{rel.as_posix()}")
            if result is not None and result.returncode == 0:
                data = result.stdout.decode("utf-8", "replace")
        else:
            candidate = self.project_root / rel
            if candidate.is_file():
                data = candidate.read_text(encoding="utf-8", errors="replace")
        self._cache[rel] = data
        return data

    def read_bytes(self, rel: Path | str) -> bytes:
        """Read exact evidence bytes without text decoding or newline changes."""
        if self._revision:
            result = self._git("show", f"{self._revision}:{Path(rel).as_posix()}")
            if result is None or result.returncode:
                raise OSError(f"cannot read {rel} at {self.commit}")
            return result.stdout
        return (self.project_root / rel).read_bytes()

    def glob(self, prefix: str, pattern: str) -> list[Path]:
        """Snapshot equivalent of ``(project_root / prefix).rglob(pattern)``.

        Only the file *name* is matched against *pattern*, which is exactly what
        every call site needs (``*.py``, ``*.png``, ``test_*.py``, ``mcp.py``).
        """
        base = Path(prefix)
        return sorted(
            rel
            for rel in self.files()
            if base in rel.parents and fnmatch(rel.name, pattern)
        )
