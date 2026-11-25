#!/usr/bin/env python3
"""API import path for the shared workspace path policy.

The implementation lives in ``gnn.utils.runtime_safety.workspace_paths`` so
that step packages apply the same rules without importing this transport.
The names below are the same objects. To replace the workspace root in a
test, set ``get_repo_root`` on the implementation module: that is the name
``resolve_repo_path`` reads.
"""

from gnn.utils.runtime_safety.workspace_paths import (
    PathValidationError,
    get_repo_root,
    resolve_repo_path,
    resolve_request_paths,
)

__all__ = [
    "PathValidationError",
    "get_repo_root",
    "resolve_repo_path",
    "resolve_request_paths",
]
