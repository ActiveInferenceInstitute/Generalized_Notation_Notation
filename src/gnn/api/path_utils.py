#!/usr/bin/env python3
"""Shared path validation for local-only API execution."""

import os
from pathlib import Path
from typing import Union

from gnn.utils.runtime_safety.filesystem import (
    create_directory,
    directory_handle,
    is_redirect,
)


class PathValidationError(ValueError):
    """Raised when an API path violates repository-local execution policy."""


def get_repo_root() -> Path:
    """Return the operator-configured API workspace, or the checkout default.

    ``GNN_API_ROOT`` is process configuration, never a request parameter. An
    ordinary installed package must supply its data workspace explicitly rather
    than permitting API callers to write into the installed package directory.
    """
    configured = os.environ.get("GNN_API_ROOT")
    if configured is None:
        owner = Path(__file__).resolve()
        checkout = owner.parent.parent.parent.parent
        if (checkout / "pyproject.toml").is_file() and (
            checkout / "src" / "gnn" / "api" / "path_utils.py"
        ).resolve() == owner:
            return checkout
        raise PathValidationError(
            "Installed API services must configure an absolute existing GNN_API_ROOT"
        )
    root = Path(configured)
    if not configured.strip() or not root.is_absolute():
        raise PathValidationError("GNN_API_ROOT must be an absolute existing directory")
    try:
        if os.name == "posix":
            with directory_handle(root):
                pass
        else:
            node = Path(root.anchor)
            for part in root.parts[1:]:
                if part == "..":
                    raise ValueError("API root must not contain parent traversal")
                node /= part
                if is_redirect(node.lstat()) or not node.is_dir():
                    raise ValueError("API root must not traverse reparse points")
    except (OSError, ValueError) as exc:
        raise PathValidationError(f"GNN_API_ROOT is not a safe directory: {exc}") from exc
    return root


def resolve_repo_path(
    path_value: Union[str, Path],
    *,
    purpose: str,
    must_exist: bool = False,
    must_be_dir: bool = True,
    create: bool = False,
) -> Path:
    """Resolve a caller-provided path and enforce repository-local boundaries."""
    if isinstance(path_value, str) and not path_value.strip():
        raise PathValidationError(f"{purpose} path must not be empty")
    raw = Path(path_value).expanduser()

    repo_root = get_repo_root()
    candidate = raw if raw.is_absolute() else repo_root / raw

    # This admits a pathname, not a handle that remains safe for later readers.
    # Creation below is descriptor-relative on POSIX; execution still requires
    # trusted directory entries throughout the run (see filesystem boundaries).
    try:
        candidate_relative = candidate.relative_to(repo_root)
    except ValueError as exc:
        raise PathValidationError(
            f"{purpose} must be within the repository root: {path_value}"
        ) from exc

    node = repo_root
    for part in candidate_relative.parts:
        if part == "..":
            raise PathValidationError(
                f"{purpose} must not contain parent traversal: {path_value}"
            )
        node = node / part
        try:
            redirected = is_redirect(node.lstat())
        except FileNotFoundError:
            redirected = False
        except OSError as exc:
            raise PathValidationError(
                f"{purpose} could not be inspected: {path_value}: {exc}"
            ) from exc
        if redirected:
            raise PathValidationError(
                f"{purpose} must not traverse symlinks: {path_value}"
            )

    resolved = candidate.resolve(strict=False)

    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PathValidationError(
            f"{purpose} must be within the repository root: {path_value}"
        ) from exc

    if must_exist and not resolved.exists():
        raise PathValidationError(f"{purpose} not found: {path_value}")

    if must_be_dir and resolved.exists() and not resolved.is_dir():
        raise PathValidationError(f"{purpose} must be a directory: {path_value}")

    if create:
        try:
            create_directory(candidate)
        except (OSError, ValueError) as exc:
            raise PathValidationError(
                f"{purpose} could not be created: {path_value}: {exc}"
            ) from exc

    return resolved


def resolve_request_paths(target_dir: str, output_dir: str) -> tuple[Path, Path]:
    """Resolve the standard API request path pair in one call.

    Both FastAPI surfaces validate requests identically: the target
    directory must exist inside the repository, the output directory is
    created on demand. Raises :class:`PathValidationError` when either path
    is rejected; HTTP surfaces map that to a 400 response.
    """
    target_path = resolve_repo_path(
        target_dir,
        purpose="Target directory",
        must_exist=True,
    )
    output_path = resolve_repo_path(
        output_dir,
        purpose="Output directory",
        create=True,
    )
    return target_path, output_path
