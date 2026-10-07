"""Descriptor-relative directory operations, without a pathname sandbox claim.

POSIX handles pin the opened directory objects and reject symlink components.
They cannot prevent another writer renaming those objects or modifying files
inside them. Windows has no stdlib equivalent: its path operations require a
tree whose directory entries cannot be changed by an untrusted writer.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def is_redirect(entry: os.stat_result) -> bool:
    """Recognize POSIX symlinks and Windows reparse points (including junctions)."""
    return stat.S_ISLNK(entry.st_mode) or bool(
        getattr(entry, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


@contextmanager
def directory_handle(path: Path, *, create: bool = False) -> Iterator[int]:
    """Open every POSIX component without following links; optionally mkdirat.

    Creation and subsequent operations use the same parent descriptor. A
    renamed parent therefore cannot redirect them into a replacement symlink.
    Reject ``..`` rather than silently normalizing through an unchecked entry.
    The caller owns the policy on what directory objects may be accessed.
    """
    if os.name != "posix":
        raise NotImplementedError("Descriptor-relative directories require POSIX")
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise ValueError("Directory path must not contain parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    handles = [os.open(absolute.anchor, flags)]
    try:
        for part in absolute.parts[1:]:
            parent = handles[-1]
            try:
                child = os.open(part, flags, dir_fd=parent)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, dir_fd=parent)
                except FileExistsError:
                    pass  # a concurrent creator must still pass no-follow open
                child = os.open(part, flags, dir_fd=parent)
            handles.append(child)
        yield handles[-1]
    finally:
        for descriptor in reversed(handles):
            os.close(descriptor)


def create_directory(path: Path) -> None:
    """Create a no-follow POSIX directory, or a trusted-tree Windows directory."""
    if os.name == "posix":
        with directory_handle(path, create=True) as descriptor:
            opened = os.fstat(descriptor)
            current = path.lstat()
            if is_redirect(current) or (opened.st_dev, opened.st_ino) != (
                current.st_dev, current.st_ino
            ):
                raise ValueError("Directory changed during creation")
            return
    # Reparse validation is admission only, not a concurrent-writer guarantee.
    node = Path(path.anchor)
    for part in path.parts[1:]:
        node /= part
        try:
            node.mkdir()
        except FileExistsError:
            pass
        entry = node.lstat()
        if is_redirect(entry) or not stat.S_ISDIR(entry.st_mode):
            raise ValueError("Directory path must not traverse reparse points")
