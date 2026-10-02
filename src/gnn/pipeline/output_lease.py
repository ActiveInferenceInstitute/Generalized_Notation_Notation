"""Exclusive cross-process ownership of a pipeline output directory."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO, Any, cast


class OutputLeaseError(RuntimeError):
    """Safe exclusive ownership of this output directory is unavailable."""


def validate_output_tree(output_dir: Path) -> None:
    """Reject preexisting output entries that can redirect writes outside ownership.

    This runs while the invocation holds its lease, before log setup, context
    staging or any producer writes. Directory symlinks are inspected without
    traversal. Contained symlinks are allowed; external links, hardlinked files
    and nonregular entries fail without modifying their targets. The advisory
    lease does not prevent a noncooperating process from changing the tree later.
    """
    root = output_dir.resolve()

    def fail_walk(error: OSError) -> None:
        raise OutputLeaseError(f"Cannot inspect pipeline output ownership: {error}")

    for directory, directories, files in os.walk(
        root, followlinks=False, onerror=fail_walk
    ):
        for name in directories + files:
            path = Path(directory) / name
            entry = path.lstat()
            relative = path.relative_to(root)
            if stat.S_ISLNK(entry.st_mode):
                try:
                    contained = path.resolve().is_relative_to(root)
                except RuntimeError as error:
                    raise OutputLeaseError(
                        f"Pipeline output entry has a cyclic symlink: {relative}"
                    ) from error
                if not contained:
                    raise OutputLeaseError(
                        f"Pipeline output entry escapes its output root: {relative}"
                    )
            elif stat.S_ISREG(entry.st_mode):
                if entry.st_nlink != 1:
                    raise OutputLeaseError(
                        f"Pipeline output file must have a single link: {relative}"
                    )
            elif not stat.S_ISDIR(entry.st_mode):
                raise OutputLeaseError(
                    f"Pipeline output entry must be a regular file or directory: {relative}"
                )


class OutputLease:
    """Hold an advisory OS lock until the invocation and its workers finish."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.handle: IO[str] | None = None

    def __enter__(self) -> OutputLease:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".gnn_run.lock"
        if lock_path.is_symlink():
            raise OutputLeaseError("Pipeline output lock cannot be a symlink")
        descriptor: int | None = None
        acquired = False
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise OutputLeaseError(
                    "Pipeline output lock must be a regular single-link file"
                )
            self.handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            descriptor = None
            if os.name == "posix":
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt

                self.handle.seek(0)
                windows_lock = cast(Any, msvcrt)
                windows_lock.locking(self.handle.fileno(), windows_lock.LK_NBLCK, 1)
            current = lock_path.lstat()
            if (
                stat.S_ISLNK(current.st_mode)
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise OutputLeaseError(
                    "Pipeline output lock changed during acquisition"
                )
            validate_output_tree(self.output_dir)
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(f"{self.run_id} pid={os.getpid()}\n")
            self.handle.flush()
            acquired = True
        except (BlockingIOError, PermissionError):
            raise OutputLeaseError(
                f"Pipeline output directory is owned by another run: {self.output_dir}"
            ) from None
        except OSError as error:
            raise OutputLeaseError(
                f"Cannot acquire safe pipeline output lock: {error}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not acquired and self.handle is not None:
                self.handle.close()
                self.handle = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None
