"""Exclusive cross-process ownership of a pipeline output directory."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, Any, cast

from gnn.utils.runtime_safety.filesystem import (
    create_directory,
    directory_handle,
    is_redirect,
)


class OutputLeaseError(RuntimeError):
    """Safe exclusive ownership of this output directory is unavailable."""


def validate_output_tree(output_dir: Path) -> None:
    """Reject preexisting output entries that can redirect writes outside ownership.

    This runs while the invocation holds its lease, before log setup, context
    staging or any producer writes. Directory symlinks are inspected without
    traversal. Contained ordinary and dangling symlinks are allowed; cycles,
    other resolution failures, external links, hardlinked files and nonregular
    entries fail without modifying their targets. The advisory
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
            if is_redirect(entry):
                # On Windows a junction can look like a directory rather than
                # a symlink to os.walk; never descend into its target.
                if name in directories:
                    directories.remove(name)
                try:
                    try:
                        resolved = path.resolve(strict=True)
                    except FileNotFoundError:
                        # Preserve contained dangling links. Non-strict
                        # normalization can expose another existing link after
                        # a missing/.. prefix, so check that candidate strictly
                        # before treating only an actual missing target as safe.
                        resolved = path.resolve()
                        try:
                            resolved = resolved.resolve(strict=True)
                        except FileNotFoundError:
                            pass
                    contained = resolved.is_relative_to(root)
                except (RuntimeError, OSError) as error:
                    if isinstance(error, RuntimeError) or error.errno == errno.ELOOP:
                        detail = (
                            f"Pipeline output entry has a cyclic symlink: {relative}"
                        )
                    else:
                        detail = f"Cannot resolve pipeline output entry: {relative}"
                    raise OutputLeaseError(detail) from error
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
        self._directory_context: AbstractContextManager[int] | None = None

    def __enter__(self) -> OutputLease:
        lock_path = self.output_dir / ".gnn_run.lock"
        descriptor: int | None = None
        acquired = False
        try:
            root_descriptor = None
            if os.name == "posix":
                requested = self.output_dir.absolute()
                if ".." in requested.parts:
                    raise ValueError("Output path must not contain parent traversal")
                # CLI parents are operator-trusted and may use native aliases
                # such as macOS /var -> /private/var. Resolve only that parent:
                # the output leaf must still pass the descriptor no-follow open.
                # Retain the requested path for the identity checks below.
                try:
                    lease_directory = requested.parent.resolve() / requested.name
                except RuntimeError as error:
                    raise ValueError("Cannot resolve pipeline output parent") from error
                self._directory_context = directory_handle(lease_directory, create=True)
                root_descriptor = self._directory_context.__enter__()
                root_identity = os.fstat(root_descriptor)
            else:
                create_directory(self.output_dir.absolute())
                root_identity = self.output_dir.lstat()
            entry_name = ".gnn_run.lock" if root_descriptor is not None else lock_path
            try:
                lock_entry = os.stat(
                    entry_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                lock_entry = None
            if lock_entry is not None and is_redirect(lock_entry):
                raise OutputLeaseError(
                    "Pipeline output lock cannot be a symlink or reparse point"
                )
            descriptor = os.open(
                entry_name,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_descriptor,
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

            def check_identity() -> None:
                current = os.stat(
                    entry_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (
                    is_redirect(current)
                    or current.st_nlink != 1
                    or (current.st_dev, current.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise OutputLeaseError(
                        "Pipeline output lock changed during acquisition"
                    )
                current_root = self.output_dir.lstat()
                if is_redirect(current_root) or (
                    current_root.st_dev,
                    current_root.st_ino,
                ) != (root_identity.st_dev, root_identity.st_ino):
                    raise OutputLeaseError(
                        "Pipeline output directory changed during acquisition"
                    )

            check_identity()
            validate_output_tree(self.output_dir)
            # Validation takes time: do not certify a replacement introduced
            # while walking the tree. This detects acquisition races, not later
            # mutations by a noncooperating writer after the lease is returned.
            check_identity()
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(f"{self.run_id} pid={os.getpid()}\n")
            self.handle.flush()
            acquired = True
        except (BlockingIOError, PermissionError):
            raise OutputLeaseError(
                f"Pipeline output directory is owned by another run: {self.output_dir}"
            ) from None
        except (OSError, ValueError) as error:
            raise OutputLeaseError(
                f"Cannot acquire safe pipeline output lock: {error}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not acquired and self.handle is not None:
                self.handle.close()
                self.handle = None
            if not acquired and self._directory_context is not None:
                self._directory_context.__exit__(None, None, None)
                self._directory_context = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None
        if self._directory_context is not None:
            self._directory_context.__exit__(None, None, None)
            self._directory_context = None
