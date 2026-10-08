"""Real filesystem replacements at the public creation, lease and deletion boundaries.

Threads perform actual rename/symlink operations. The interception only selects
the syscall window deterministically; it never substitutes a filesystem result.
Native Windows tests do not borrow POSIX assertions or fake platform values.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable

import pytest

from gnn.api.path_utils import PathValidationError, get_repo_root, resolve_repo_path
from gnn.api.processor import delete_run
from gnn.pipeline.output_lease import OutputLease, OutputLeaseError
from gnn.utils.runtime_safety import filesystem


@pytest.mark.parametrize("create", [False, True])
def test_native_directory_handle_support_is_explicit(
    tmp_path: Path, create: bool
) -> None:
    sentinel = tmp_path / "retained.txt"
    sentinel.write_bytes(b"unrelated native source")
    directory = tmp_path / "owned-directory"
    if os.name != "posix":
        with pytest.raises(OSError) as error:
            with filesystem.directory_handle(directory, create=create):
                pytest.fail("Unsupported descriptor operation yielded a handle")
        assert error.value.errno == errno.ENOTSUP
        assert not directory.exists()
    else:
        if not create:
            directory.mkdir()
        with filesystem.directory_handle(directory, create=create) as descriptor:
            opened = os.fstat(descriptor)
            actual = directory.lstat()
            assert (opened.st_dev, opened.st_ino) == (actual.st_dev, actual.st_ino)
        assert directory.is_dir()
    assert sentinel.read_bytes() == b"unrelated native source"


def test_explicit_api_workspace_is_operator_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_API_ROOT", str(tmp_path))
    (tmp_path / "models").mkdir()
    assert get_repo_root() == tmp_path
    assert resolve_repo_path("models", purpose="Target", must_exist=True) == tmp_path / "models"
    assert resolve_repo_path("runs/new", purpose="Output", create=True).is_dir()
    with pytest.raises(PathValidationError):
        resolve_repo_path(str(tmp_path.parent), purpose="Output", create=True)
    with pytest.raises(PathValidationError):
        resolve_repo_path("models/../runs", purpose="Output", create=True)


@pytest.mark.parametrize("root", ["", "relative-workspace"])
def test_api_workspace_configuration_rejects_invalid_roots(
    root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_API_ROOT", root)
    with pytest.raises(PathValidationError, match="absolute existing"):
        get_repo_root()


@pytest.mark.parametrize("remove_ancestor", [False, True])
def test_delete_run_retains_workspace_and_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove_ancestor: bool
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    sentinel = root / "source.gnn"
    sentinel.write_text("retain source")
    monkeypatch.setenv("GNN_API_ROOT", str(root))
    output = tmp_path if remove_ancestor else root
    store = {"run": {"status": "completed", "request": {"output_dir": str(output)}}}
    result = delete_run("run", runs_store=store)
    assert result["artifacts_removed"] is False
    assert "workspace or ancestor" in result["artifacts_note"]
    assert sentinel.read_text() == "retain source" and not store


def test_delete_run_with_unverified_cleanup_retains_record_and_artifacts(tmp_path: Path) -> None:
    sentinel = tmp_path / "artifact"
    sentinel.write_text("worker may still be using this")
    store = {"run": {
        "status": "failed", "request": {"output_dir": str(tmp_path)},
        "process_cleanup": {"cleanup_verified": False},
    }}
    with pytest.raises(RuntimeError, match="cleanup could not be verified"):
        delete_run("run", runs_store=store)
    assert "run" in store and sentinel.read_text() == "worker may still be using this"


def test_native_link_or_junction_is_refused_without_touching_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = tmp_path / "workspace", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    sentinel = outside / "retain.txt"
    sentinel.write_text("external")
    linked = root / "linked"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
            capture_output=True, timeout=10,
        )
        assert created.returncode == 0, created.stderr.decode(errors="replace")
    else:
        linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("GNN_API_ROOT", str(root))
    try:
        with pytest.raises(PathValidationError, match="symlinks"):
            resolve_repo_path("linked/new", purpose="Output", create=True)
        with pytest.raises(OutputLeaseError):
            with OutputLease(linked, "linked-output"):
                pytest.fail("A redirected output root was admitted")
        with pytest.raises(OutputLeaseError, match="escapes"):
            with OutputLease(root, "nested-link-output"):
                pytest.fail("An external link/junction inside output was admitted")
        store = {"run": {"status": "completed", "request": {"output_dir": str(linked)}}}
        result = delete_run("run", runs_store=store)
        assert result["artifacts_removed"] is False
        assert "reparse point" in result["artifacts_note"]
        assert sentinel.read_text() == "external"
        assert not (outside / "new").exists() and not (outside / ".gnn_run.lock").exists()
    finally:
        if os.name == "nt":
            linked.rmdir()  # remove the junction, never its target
        else:
            linked.unlink()


@pytest.mark.needs_posix
def test_api_workspace_configuration_refuses_linked_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "linked"
    link.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv("GNN_API_ROOT", str(link))
    with pytest.raises(PathValidationError, match="safe directory"):
        get_repo_root()


def _replacement_thread(action: Callable[[], None]) -> tuple[threading.Event, threading.Event, threading.Thread, list[BaseException]]:
    ready, replaced = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def replace() -> None:
        try:
            if not ready.wait(3):
                raise AssertionError("Operation never reached its real race window")
            action()
        except BaseException as error:
            errors.append(error)
        finally:
            replaced.set()

    worker = threading.Thread(target=replace)
    worker.start()
    return ready, replaced, worker, errors


@pytest.mark.needs_posix
def test_api_creation_parent_rename_cannot_redirect_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = tmp_path / "workspace", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    parent = root / "parent"
    parent.mkdir()
    parked = root / "parked"
    monkeypatch.setenv("GNN_API_ROOT", str(root))

    def replace() -> None:
        parent.rename(parked)
        parent.symlink_to(outside, target_is_directory=True)

    ready, replaced, worker, errors = _replacement_thread(replace)
    mkdir = os.mkdir

    def pause_mkdir(path: object, *args: object, **kwargs: object) -> None:
        if path == "created" and kwargs.get("dir_fd") is not None:
            ready.set()
            assert replaced.wait(3)
        mkdir(path, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "mkdir", pause_mkdir)
    try:
        with pytest.raises(PathValidationError, match="could not be created"):
            resolve_repo_path("parent/created", purpose="Output", create=True)
    finally:
        ready.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert (parked / "created").is_dir()  # the descriptor retained its directory
    assert not (outside / "created").exists()


@pytest.mark.needs_posix
def test_output_directory_replacement_cannot_create_external_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, outside = tmp_path / "output", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    sentinel = outside / ".gnn_run.lock"
    sentinel.write_text("outside-owner")

    def replace() -> None:
        output.rename(tmp_path / "parked")
        output.symlink_to(outside, target_is_directory=True)

    ready, replaced, worker, errors = _replacement_thread(replace)
    open_file = os.open

    def pause_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == ".gnn_run.lock" and kwargs.get("dir_fd") is not None:
            ready.set()
            assert replaced.wait(3)
        return open_file(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", pause_open)
    try:
        with pytest.raises(OutputLeaseError, match="directory changed"):
            with OutputLease(output, "replaced-run"):
                pytest.fail("Replacement was admitted")
    finally:
        ready.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert sentinel.read_text() == "outside-owner"
    assert (tmp_path / "parked" / ".gnn_run.lock").read_text() == ""


@pytest.mark.needs_posix
def test_native_cli_output_parent_alias_preserves_run_identity(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    model = project / "input/gnn_files/discrete/two_state_bistable.md"
    with tempfile.TemporaryDirectory() as temporary:
        native_parent = Path(temporary)
        canonical_parent = native_parent.resolve()
        # Use the actual OS temp alias where supplied (macOS /var); otherwise
        # exercise the same public contract through an operator-owned alias.
        alias = native_parent
        if native_parent == canonical_parent:
            alias = tmp_path / "trusted-temp-alias"
            alias.symlink_to(canonical_parent, target_is_directory=True)
        assert alias != canonical_parent and alias.resolve() == canonical_parent
        source = canonical_parent / "models"
        source.mkdir()
        shutil.copyfile(model, source / model.name)
        output = alias / "output"
        canonical_output = canonical_parent / "output"
        receipts = []
        contexts = []
        lock_identities = []
        for selected_output in (output, canonical_output):
            environment = dict(os.environ)
            invocation = uuid.uuid4().hex
            environment["GNN_RUN_ID"] = invocation
            environment.pop("GNN_RUN_CONTEXT_FILE", None)
            result = subprocess.run(
                [sys.executable, "-m", "gnn.main", "--only-steps", "3",
                 "--target-dir", str(source), "--output-dir", str(selected_output)],
                cwd=project, env=environment, capture_output=True, text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            receipt = json.loads((canonical_output / "00_pipeline_summary" /
                                  "pipeline_execution_summary.json").read_text())
            context = json.loads((canonical_output / "00_pipeline_summary" /
                                  "run_context.json").read_text())
            assert receipt["run_id"] == invocation == context["run_id"]
            assert context["output_root"] == str(canonical_output)
            assert receipt["steps"] and all(
                step["status"].startswith("SUCCESS") for step in receipt["steps"]
            )
            selected = receipt["model_selection"]
            assert len(selected) == 1
            assert selected[0]["source_path"] == str(source / model.name)
            assert selected[0]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
            assert receipt["steps"][0]["artifacts"]
            for artifact in receipt["steps"][0]["artifacts"]:
                assert (canonical_output / artifact["path"]).is_file()
            lock = (canonical_output / ".gnn_run.lock").stat()
            lock_identities.append((lock.st_dev, lock.st_ino))
            receipts.append(receipt)
            contexts.append(context)
        assert lock_identities[0] == lock_identities[1]
        assert receipts[0]["run_id"] != receipts[1]["run_id"]
        assert receipts[0]["run_hash"] == receipts[1]["run_hash"]
        assert receipts[0]["file_hashes"] == receipts[1]["file_hashes"]
        assert receipts[0]["model_selection"] == receipts[1]["model_selection"]
        assert contexts[0]["models"] == contexts[1]["models"]


@pytest.mark.needs_posix
@pytest.mark.parametrize("redirect", ["output", "lock"])
def test_native_cli_trusted_parent_alias_still_refuses_redirected_leaf(
    tmp_path: Path, redirect: str
) -> None:
    project = Path(__file__).resolve().parents[2]
    parent, outside = tmp_path / "parent", tmp_path / "outside"
    parent.mkdir()
    outside.mkdir()
    sentinel = outside / ".gnn_run.lock"
    sentinel.write_text("external owner")
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    output = alias / "output"
    if redirect == "output":
        output.symlink_to(outside, target_is_directory=True)
    else:
        output.mkdir()
        (output / ".gnn_run.lock").symlink_to(sentinel)
    result = subprocess.run(
        [sys.executable, "-m", "gnn.main", "--only-steps", "3",
         "--target-dir", str(project / "input/gnn_files/discrete"),
         "--output-dir", str(output)],
        cwd=project, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1 and "Pipeline ownership failure" in result.stderr
    assert sentinel.read_text() == "external owner"
    assert not (outside / "00_pipeline_summary").exists()
    assert not (parent / "output/00_pipeline_summary").exists()


@pytest.mark.needs_posix
def test_trusted_parent_alias_replacement_cannot_redirect_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, outside = tmp_path / "parent", tmp_path / "outside"
    parent.mkdir()
    outside.mkdir()
    (parent / "output").mkdir()
    (outside / "output").mkdir()
    sentinel = outside / "output/.gnn_run.lock"
    sentinel.write_text("external owner")
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)

    def replace() -> None:
        alias.unlink()
        alias.symlink_to(outside, target_is_directory=True)

    ready, replaced, worker, errors = _replacement_thread(replace)
    open_file = os.open

    def pause_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == ".gnn_run.lock" and kwargs.get("dir_fd") is not None:
            ready.set()
            assert replaced.wait(3)
        return open_file(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", pause_open)
    try:
        with pytest.raises(OutputLeaseError, match="directory changed"):
            with OutputLease(alias / "output", "replaced-alias"):
                pytest.fail("Replaced parent alias was admitted")
    finally:
        ready.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert sentinel.read_text() == "external owner"
    assert (parent / "output/.gnn_run.lock").read_text() == ""


@pytest.mark.needs_posix
@pytest.mark.parametrize("parent_kind", ["traversal", "cycle"])
def test_lease_parent_resolution_failure_preserves_typed_refusal(
    tmp_path: Path, parent_kind: str
) -> None:
    if parent_kind == "traversal":
        output = tmp_path / "absent/../output"
    else:
        alias = tmp_path / "cycle"
        alias.symlink_to(alias, target_is_directory=True)
        output = alias / "output"
    with pytest.raises(OutputLeaseError):
        with OutputLease(output, "invalid-parent"):
            pytest.fail("Unsafe parent was admitted")
    assert not (tmp_path / "output").exists() and not (tmp_path / "absent").exists()


@pytest.mark.needs_posix
def test_delete_run_never_removes_a_link_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_API_ROOT", str(tmp_path.resolve()))
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "artifact"
    sentinel.write_text("retain me")
    linked = tmp_path / "run"
    linked.symlink_to(outside, target_is_directory=True)
    store = {"run": {"status": "completed", "request": {"output_dir": str(linked)}}}
    result = delete_run("run", runs_store=store)
    assert result["artifacts_removed"] is False
    assert "symlink" in result["artifacts_note"]
    assert sentinel.read_text() == "retain me" and linked.is_symlink()
    assert not store


@pytest.mark.needs_posix
def test_delete_run_parent_replacement_cannot_remove_external_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_API_ROOT", str(tmp_path.resolve()))
    parent, outside = tmp_path / "parent", tmp_path / "outside"
    parent.mkdir()
    (parent / "run").mkdir()
    outside.mkdir()
    (outside / "run").mkdir()
    sentinel = outside / "run" / "artifact"
    sentinel.write_text("retain me")
    parked = tmp_path / "parked"

    def replace() -> None:
        parent.rename(parked)
        parent.symlink_to(outside, target_is_directory=True)

    ready, replaced, worker, errors = _replacement_thread(replace)
    rmtree = shutil.rmtree

    def pause_rmtree(path: object, *args: object, **kwargs: object) -> None:
        ready.set()
        assert replaced.wait(3)
        rmtree(path, *args, **kwargs)

    pause_rmtree.avoids_symlink_attacks = True  # retain the native capability
    monkeypatch.setattr("gnn.api.processor.shutil.rmtree", pause_rmtree)
    store = {"run": {"status": "completed", "request": {"output_dir": str(parent / "run")}}}
    try:
        result = delete_run("run", runs_store=store)
    finally:
        ready.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert result["artifacts_removed"] is True
    assert not (parked / "run").exists()
    assert sentinel.read_text() == "retain me"


def test_output_lease_is_exclusive_in_another_native_process(tmp_path: Path) -> None:
    command = [sys.executable, "-c", (
        "import sys; from pathlib import Path; "
        "from gnn.pipeline.output_lease import OutputLease,OutputLeaseError; "
        "\ntry:\n with OutputLease(Path(sys.argv[1]), 'contender'): pass"
        "\nexcept OutputLeaseError: sys.exit(23)"
    ), str(tmp_path)]
    with OutputLease(tmp_path, "owner") as owner:
        held = subprocess.run(command, capture_output=True, timeout=10)
        assert held.returncode == 23, held.stderr.decode()
        # Windows byte-range locks also deny reads through a second handle.
        assert owner.handle is not None
        owner.handle.seek(0)
        assert owner.handle.read().startswith("owner pid=")
    released = subprocess.run(command, capture_output=True, timeout=10)
    assert released.returncode == 0, released.stderr.decode()
    assert (tmp_path / ".gnn_run.lock").read_text().startswith("contender pid=")


def test_native_process_cleanup_reports_its_actual_boundary(tmp_path: Path) -> None:
    from gnn.execute.subprocess_envelope import run_subprocess_envelope

    result = run_subprocess_envelope(
        [sys.executable, "-c", "import time; print('native-worker',flush=True); time.sleep(.3)"],
        timeout=3, sandbox=False,
    )
    assert result["success"] and result["cleanup_verified"] and result["streams_drained"]
    assert result["stdout"].strip() == "native-worker"
    assert result["containment"] == (
        "direct_worker_only" if os.name == "nt" else "observed_descendants"
    )


@pytest.mark.parametrize("requirement", ["require_descendant_containment", "require_descendant_resource_accounting"])
def test_native_descendant_requirements_are_supported_or_refused_before_launch(
    tmp_path: Path, requirement: str
) -> None:
    from gnn.execute.subprocess_envelope import run_subprocess_envelope

    marker = tmp_path / "launched"
    result = run_subprocess_envelope(
        [sys.executable, "-c", f"from pathlib import Path; import time; Path({str(marker)!r}).write_text('started'); time.sleep(.3)"],
        timeout=3, sandbox=False, **{requirement: True},
    )
    if os.name == "nt":
        assert not result["success"] and result["error_type"] == "UnsupportedContainment"
        assert result["containment"] == "not_started" and result["cleanup_verified"] is None
        assert not marker.exists()
    else:
        assert result["success"] and result["containment"] == "observed_descendants"
        assert marker.read_text() == "started"


def test_native_cancellation_stops_direct_worker_within_budget(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import psutil

    from gnn.execute.subprocess_envelope import CancelToken, run_subprocess_envelope

    marker = tmp_path / "pid"
    token = CancelToken()
    errors: list[BaseException] = []
    native_handle: int | None = None
    supervised_handle: int | None = None
    diagnostic: dict[str, object] = {
        "platform": sys.platform,
        "executable": sys.executable,
        "before_cancel": None,
        "after_return": None,
    }

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessId.argtypes = [wintypes.HANDLE]
        kernel32.GetProcessId.restype = wintypes.DWORD
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            *[ctypes.POINTER(wintypes.FILETIME)] * 4,
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        def native_snapshot(handle: int) -> dict[str, int]:
            pid = kernel32.GetProcessId(handle)
            if not pid:
                raise ctypes.WinError(ctypes.get_last_error())
            creation, exit_time, kernel_time, user_time = (
                wintypes.FILETIME() for _ in range(4)
            )
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            wait_state = kernel32.WaitForSingleObject(handle, 0)
            if wait_state == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return {
                "pid": pid,
                "creation_filetime": creation.dwLowDateTime
                | (creation.dwHighDateTime << 32),
                "wait_state": wait_state,
                "exit_code": exit_code.value,
            }

    actual_popen = subprocess.Popen

    def observe_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal supervised_handle
        process = actual_popen(*args, **kwargs)
        diagnostic["supervised_pid"] = process.pid
        if os.name == "nt":
            try:
                supervised_handle = kernel32.OpenProcess(0x00101000, False, process.pid)
                if not supervised_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                diagnostic["supervised_before_return"] = native_snapshot(supervised_handle)
            except OSError as error:
                # Return the actual object even if observation fails, so the
                # envelope still owns cleanup of its newly launched worker.
                errors.append(error)
        return process

    monkeypatch.setattr(subprocess, "Popen", observe_popen)

    def cancel() -> None:
        nonlocal native_handle
        try:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists()
            if os.name == "nt":
                # The writer opens before publishing bytes; stay inside the same
                # original readiness budget until its authored PID is available.
                payload = marker.read_text()
                while not payload and time.monotonic() < deadline:
                    time.sleep(.01)
                    payload = marker.read_text()
                payload_pid = int(payload)
                diagnostic["payload_pid"] = payload_pid
                # Query and synchronize only this authored PID. Retaining its
                # handle binds creation identity and prevents PID recycling.
                # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION; no kill rights.
                native_handle = kernel32.OpenProcess(0x00101000, False, payload_pid)
                if not native_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                before = native_snapshot(native_handle)
                diagnostic["before_cancel"] = before
                assert before["pid"] == payload_pid
                assert before["wait_state"] == 258 and before["exit_code"] == 259
            token.cancel("native acceptance")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=cancel)
    worker.start()
    started = time.monotonic()
    try:
        try:
            result = run_subprocess_envelope(
                [sys.executable, "-c", f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"],
                timeout=4, sandbox=False, cancel_token=token,
            )
        finally:
            worker.join(4)
        diagnostic["result"] = result
        diagnostic["elapsed_seconds"] = time.monotonic() - started
        diagnostic["cancellation_errors"] = [repr(error) for error in errors]
        try:
            if os.name == "nt" and supervised_handle:
                diagnostic["supervised_after_return"] = native_snapshot(supervised_handle)
                handle_to_close, supervised_handle = supervised_handle, None
                if not kernel32.CloseHandle(handle_to_close):
                    raise ctypes.WinError(ctypes.get_last_error())
                diagnostic["supervised_handle_closed_before_pid_assertion"] = True
            if os.name == "nt" and native_handle:
                # A zero-time native wait observes exit without extending cleanup.
                diagnostic["after_return"] = native_snapshot(native_handle)
                # Release this probe before the original PID membership assertion:
                # an observation handle must not prolong the process object there.
                handle_to_close, native_handle = native_handle, None
                if not kernel32.CloseHandle(handle_to_close):
                    raise ctypes.WinError(ctypes.get_last_error())
                diagnostic["probe_handle_closed_before_pid_assertion"] = True
        except OSError as error:
            diagnostic["native_observation_error"] = repr(error)
            raise
        finally:
            detail = json.dumps(diagnostic, sort_keys=True)
            request.node.user_properties.append(("native_cancellation_diagnostic", detail))
        assert not worker.is_alive() and not errors, detail
        assert time.monotonic() - started < 5, detail
        assert result["cancelled"] and not result["success"] and result["error_type"] == "Cancelled", detail
        assert result["cleanup_verified"] and result["streams_drained"], detail
        assert not psutil.pid_exists(int(marker.read_text())), detail
    finally:
        if os.name == "nt":
            for handle in (supervised_handle, native_handle):
                if handle and not kernel32.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())


@pytest.mark.needs_posix
@pytest.mark.asyncio
async def test_api_job_cancel_kills_a_real_worker_ignoring_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import time

    import psutil

    from gnn.api import processor

    (tmp_path / "models").mkdir()
    monkeypatch.setenv("GNN_API_ROOT", str(tmp_path))
    child_pid = tmp_path / "child.pid"
    ready = tmp_path / "ready"
    child = (
        "import signal,time,os; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    command = [sys.executable, "-c", (
        "import subprocess,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"subprocess.Popen([{sys.executable!r},'-c',{child!r}]); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
    )]
    monkeypatch.setattr(processor, "build_pipeline_command", lambda *args, **kwargs: command)
    job_id = processor.create_job("models", "job-output")
    task = asyncio.create_task(processor.execute_job_async(job_id))
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        assert child_pid.exists() and ready.exists()
        child_identity = psutil.Process(int(child_pid.read_text()))
        child_identity.create_time()
        await asyncio.sleep(.15)  # allow the live observer to retain the child
        started = time.monotonic()
        assert processor.cancel_job(job_id)
        await asyncio.wait_for(task, timeout=2)
        assert time.monotonic() - started < 2
        job = processor.get_job(job_id)
        assert job is not None and job["status"] == "cancelled"
        assert job["process_cleanup"]["cleanup_verified"] and job["process_cleanup"]["streams_drained"]
        assert not child_identity.is_running() or child_identity.status() == psutil.STATUS_ZOMBIE
    finally:
        if not task.done():
            processor.cancel_job(job_id)
            await asyncio.wait_for(task, timeout=2)
        processor._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_api_normal_leader_exit_has_bounded_native_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import time

    import psutil

    from gnn.api import processor

    (tmp_path / "models").mkdir()
    monkeypatch.setenv("GNN_API_ROOT", str(tmp_path))
    child_pid = tmp_path / "child.pid"
    child = f"import os,time; from pathlib import Path; Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(30)"
    command = [sys.executable, "-c", (
        "import subprocess,time; "
        f"subprocess.Popen([{sys.executable!r},'-c',{child!r}]); time.sleep(.3)"
    )]
    monkeypatch.setattr(processor, "build_pipeline_command", lambda *args, **kwargs: command)
    job_id = processor.create_job("models", "job-output")
    started = time.monotonic()
    child_identity = None
    try:
        await asyncio.wait_for(processor.execute_job_async(job_id), timeout=3)
        assert time.monotonic() - started < 3
        assert child_pid.exists()
        try:
            child_identity = psutil.Process(int(child_pid.read_text()))
            child_identity.create_time()
        except psutil.NoSuchProcess:
            pass
        job = processor.get_job(job_id)
        assert job is not None
        cleanup = job["process_cleanup"]
        if os.name == "nt":
            assert job["status"] == "failed" and cleanup["cleanup_verified"] is False
            assert cleanup["containment"] == "direct_worker_only" and cleanup["streams_drained"] is False
        else:
            assert job["status"] == "completed" and cleanup["cleanup_verified"]
            assert cleanup["streams_drained"]
            assert child_identity is None or not child_identity.is_running() or child_identity.status() == psutil.STATUS_ZOMBIE
    finally:
        # Windows's deliberately unsupported child remains caller-owned; stop
        # only that published birth-time-bound identity, never a host PID scan.
        if child_identity is not None and child_identity.is_running():
            child_identity.kill()
        processor._JOBS.pop(job_id, None)
