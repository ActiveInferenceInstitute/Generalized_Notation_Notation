"""Native child-query boundaries without inspection of unrelated host PIDs."""

from __future__ import annotations

import ctypes
import errno
from pathlib import Path

import pytest

from gnn.utils.runtime_safety import process_tree


def test_macos_child_query_returns_pid_count_and_retries_truncated_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def query(pid: int, buffer: object, size: int) -> int:
        assert pid == 1200
        calls.append(size)
        capacity = size // ctypes.sizeof(ctypes.c_int)
        count = min(70, capacity)
        for index in range(count):
            buffer[index] = 2000 + index  # type: ignore[index]
        return count  # libproc returns PID count, not buffer byte count

    monkeypatch.setattr(process_tree.sys, "platform", "darwin")
    monkeypatch.setattr(process_tree, "_macos_child_query", lambda: query)
    assert process_tree._child_pids(1200) == list(range(2000, 2070))
    assert len(calls) == 2


def test_macos_child_query_does_not_turn_denied_observation_into_empty_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(*args: object) -> int:
        ctypes.set_errno(errno.EPERM)
        return 0

    monkeypatch.setattr(process_tree.sys, "platform", "darwin")
    monkeypatch.setattr(process_tree, "_macos_child_query", lambda: query)
    with pytest.raises(PermissionError):
        process_tree._child_pids(1200)


def test_linux_child_query_includes_children_spawned_by_other_parent_threads(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "1200" / "task"
    for thread, children in [(1200, "2000 2001"), (1201, "2001 2002")]:
        task = tasks / str(thread)
        task.mkdir(parents=True)
        (task / "children").write_text(children)
    (tasks / "1202").mkdir()  # a thread disappeared during observation
    unrelated = tmp_path / "9999" / "task" / "9999"
    unrelated.mkdir(parents=True)
    (unrelated / "children").write_text("8888")
    assert process_tree._linux_child_pids(1200, tmp_path) == [2000, 2001, 2002]
    assert process_tree._linux_child_pids(7777, tmp_path) == []


def test_linux_child_query_rejects_unbounded_process_listing(tmp_path: Path) -> None:
    task = tmp_path / "1200" / "task" / "1200"
    task.mkdir(parents=True)
    (task / "children").write_bytes(b"1 " * (512 * 1024 + 1))
    with pytest.raises(RuntimeError, match="exceeded"):
        process_tree._linux_child_pids(1200, tmp_path)


def test_linux_kernel_without_child_listing_has_no_observation_claim(
    tmp_path: Path,
) -> None:
    task = tmp_path / "1200" / "task" / "1200"
    task.mkdir(parents=True)
    assert process_tree._linux_child_pids(1200, tmp_path) is None
