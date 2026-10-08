"""Bounded cleanup for a supervised worker and its observed descendants.

Process groups contain ordinary descendants. Birth-time-bound observations also
retain backend workers that start new sessions before the leader exits. This is
observed process-tree containment, not an OS sandbox for hostile daemonization.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _macos_child_query() -> Any:
    """Load the kernel's parent-filtered query, without walking host PIDs."""
    import ctypes

    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    query = library.proc_listchildpids
    query.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    query.restype = ctypes.c_int
    return query


def _linux_child_pids(pid: int, proc_root: Path = Path("/proc")) -> list[int] | None:
    """Read children for every task of the selected process, with bounded input."""
    children: set[int] = set()
    read_any = False
    tasks_root = proc_root / str(pid) / "task"
    try:
        tasks = tasks_root.iterdir()
        for task in tasks:
            try:
                with (task / "children").open("rb") as stream:
                    content = stream.read(1024 * 1024 + 1)
                read_any = True
                if len(content) > 1024 * 1024:
                    raise RuntimeError("Owned-child observation exceeded 1 MiB")
                children.update(int(value) for value in content.split())
            except FileNotFoundError:
                continue  # this particular thread exited during the query
    except FileNotFoundError:
        pass  # the owned process has exited
    # Some kernels disable CONFIG_PROC_CHILDREN. A live task tree without
    # readable children files does not establish descendant observation.
    if not read_any and tasks_root.is_dir():
        return None
    return sorted(children)


def _child_pids(pid: int) -> list[int] | None:
    """Query immediate children of one owned process on macOS or Linux.

    These are racing observations, not a guarantee against an instantaneous
    detach. Unsupported platforms explicitly retain process-group containment.
    No path enumerates or inspects unrelated host processes.
    """
    platform = sys.platform
    if platform == "darwin":
        import ctypes

        query = _macos_child_query()
        capacity = 64
        while capacity <= 65536:
            buffer = (ctypes.c_int * capacity)()
            ctypes.set_errno(0)
            count = query(pid, buffer, ctypes.sizeof(buffer))
            error = ctypes.get_errno()
            if count < 0 or (count == 0 and error):
                raise OSError(error, "Owned-child process query failed")
            # proc_listchildpids returns a PID count, unlike proc_listpids
            # which returns bytes. A full buffer may have been truncated.
            if count < capacity:
                return [int(buffer[index]) for index in range(count) if buffer[index]]
            capacity *= 2
        raise RuntimeError("Owned-child observation exceeded 65536 processes")
    if platform.startswith("linux"):
        # A process's other threads can also spawn children.
        return _linux_child_pids(pid)
    return None


class DescendantTracker:
    """Observe descendants while a leader lives, retaining process identities."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._stop = threading.Event()
        self._tracked: dict[tuple[int, float], Any] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.boundary = (
            "process_group_only" if os.name == "posix" else "direct_worker_only"
        )
        self.errors: list[str] = []

    def start(self) -> None:
        """Start monitoring immediately after worker creation."""
        if self._thread is not None:
            raise RuntimeError("Descendant tracker already started")
        self._thread = threading.Thread(target=self._observe, daemon=True)
        self._thread.start()

    def _observe(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        try:
            leader = psutil.Process(self.pid)
            leader.create_time()  # bind its identity before any query
            while not self._stop.is_set():
                parents = [leader, *self.observed_processes]
                visited: set[tuple[int, float]] = set()
                for parent in parents:
                    if self._stop.is_set():
                        break
                    try:
                        identity = (parent.pid, parent.create_time())
                        if identity in visited or not parent.is_running():
                            continue
                        visited.add(identity)
                        child_pids = _child_pids(parent.pid)
                        if child_pids is None:
                            return
                        self.boundary = "observed_descendants"
                        # Discard a snapshot if its parent identity disappeared
                        # or was recycled while the kernel query ran.
                        if not parent.is_running():
                            continue
                        for pid in child_pids:
                            if self._stop.is_set():
                                break
                            try:
                                child = psutil.Process(pid)
                                created = child.create_time()
                                if child.ppid() != parent.pid:
                                    continue
                                with self._lock:
                                    self._tracked[(pid, created)] = child
                                parents.append(child)
                            except psutil.NoSuchProcess:
                                continue
                    except (psutil.NoSuchProcess, ProcessLookupError):
                        continue
                self._stop.wait(0.05)
        except psutil.NoSuchProcess:
            return
        except (
            psutil.Error,
            OSError,
            RuntimeError,
            ValueError,
            AttributeError,
        ) as error:
            self.errors.append(str(error))

    def stop(self, timeout: float = 0.1) -> list[Any]:
        """Stop observing and return identities retained before parent reaping."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))
            if self._thread.is_alive():
                raise RuntimeError(
                    "Descendant tracker did not stop within cleanup budget"
                )
        return self.observed_processes

    @property
    def observed_processes(self) -> list[Any]:
        """Snapshot retained identities even if stopping observation failed."""
        with self._lock:
            return list(self._tracked.values())

    @property
    def observed_count(self) -> int:
        """Number of retained birth-time-bound identities, including exited ones."""
        with self._lock:
            return len(self._tracked)


def terminate_process_tree(
    process: subprocess.Popen[Any],
    tracked: list[Any] | None = None,
    cleanup_timeout: float = 1.0,
) -> None:
    """Kill and verify observed workers within one shared cleanup ceiling.

    ``tracked`` must be collected while the leader is alive. A successful return
    verifies the direct worker and observed descendants have stopped; unavailable
    psutil leaves only the POSIX process-group boundary. Windows verifies the
    direct worker only: CREATE_NEW_PROCESS_GROUP does not enable tree killing.
    Surviving processes or denied
    termination raise so callers cannot report successful cleanup.
    """
    deadline = time.monotonic() + max(0.0, cleanup_timeout)
    descendants: list[Any] = list(tracked or [])
    errors: list[str] = []
    try:
        import psutil
    except ImportError:
        psutil = None
    try:
        # Kill the ordinary group before any per-identity work. The group
        # identity survives a leader exit; do not resolve a recycled leader PID.
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except PermissionError:
        errors.append("Process-group termination was denied")
        if process.poll() is None:
            process.kill()
    except (ProcessLookupError, AttributeError):
        if process.poll() is None:
            process.kill()
    if psutil is not None and tracked is None and time.monotonic() < deadline:
        # Compatibility callers without an active tracker get a scoped final
        # snapshot. Envelope callers always supply their observed identities,
        # including an empty set; never repeat discovery after their deadline.
        try:
            parents = [psutil.Process(process.pid)]
            visited: set[int] = set()
            for parent in parents:
                if time.monotonic() >= deadline:
                    errors.append("Descendant observation exceeded cleanup budget")
                    break
                if parent.pid in visited or not parent.is_running():
                    continue
                visited.add(parent.pid)
                for pid in _child_pids(parent.pid) or []:
                    try:
                        child = psutil.Process(pid)
                        child.create_time()
                        if child.ppid() == parent.pid and parent.is_running():
                            descendants.append(child)
                            parents.append(child)
                    except psutil.NoSuchProcess:
                        continue
        except psutil.NoSuchProcess:
            pass
        except (psutil.Error, OSError, RuntimeError, ValueError):
            errors.append("Descendant enumeration could not be verified")
    if psutil is not None:
        identities = {}
        for child in descendants:
            if time.monotonic() >= deadline:
                errors.append(
                    "Descendant identity verification exceeded cleanup budget"
                )
                break
            try:
                if child.is_running():
                    identities[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                errors.append("Descendant identity could not be verified")
        descendants = list(identities.values())
        for descendant in descendants:
            if time.monotonic() >= deadline:
                errors.append("Descendant termination exceeded cleanup budget")
                break
            try:
                descendant.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                errors.append("Descendant termination was denied")
    process.wait(timeout=max(0.001, deadline - time.monotonic()))
    if psutil is not None and descendants:
        # Observed descendants may be reparented, so this caller cannot reap
        # their zombies. wait_procs would spend the whole allowance waiting for
        # an unrelated parent to reap them. Verify live identities directly,
        # reserving the remaining shared allowance for the caller's pipe drain.
        while descendants:
            alive = []
            for child in descendants:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Descendant verification exceeded cleanup budget"
                    )
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        alive.append(child)
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error:
                    errors.append("Descendant termination could not be verified")
            if not alive:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Worker descendants survived termination")
            time.sleep(min(0.01, remaining))
            descendants = alive
    if errors:
        raise RuntimeError("; ".join(dict.fromkeys(errors)))
