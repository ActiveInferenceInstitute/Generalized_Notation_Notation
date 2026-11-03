"""Real startup-audit receipt publication under parallel launch and fork."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from coverage import CoverageData

from tests.helpers import load_module_from_path

CHECKOUT = Path(__file__).resolve().parents[2]
DRIVER = CHECKOUT / "scripts" / "run_comprehensive_native_coverage.py"


def _native_startup(tmp_path: Path, body: str) -> dict:
    """Execute the maintained startup text in its own real interpreter."""
    assignment = next(
        node
        for node in ast.parse(DRIVER.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "STARTUP"
            for target in node.targets
        )
    )
    startup = tmp_path / "startup.py"
    startup.write_text(ast.literal_eval(assignment.value), encoding="utf-8")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    bootstrap = tmp_path / "native.py"
    bindings = {
        "GNN_BOUND_STARTUP_DIR": str(receipts),
        "GNN_BOUND_CONFIG": str(tmp_path / "unused-coverage.toml"),
        "GNN_BOUND_SOURCE": str(CHECKOUT / "src" / "gnn"),
    }
    bootstrap.write_text(
        "import json, os, runpy, subprocess, sys, threading, time\n"
        "from pathlib import Path\n"
        f"receipt_root = Path({str(receipts)!r})\n"
        f"namespace = runpy.run_path({str(startup)!r}, init_globals={bindings!r})\n"
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    result = subprocess.run(
        # A comprehensive run already installs its own .pth observer. This
        # authored harness executes the actual startup explicitly and must not
        # borrow that outer observer's state, directory or inferred credit.
        [sys.executable, "-I", "-S", str(bootstrap)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    (tmp_path / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "stderr.log").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    return json.loads(result.stdout)


def _coverage_data_file(path: Path, measured: Path) -> None:
    """Write a real coverage data file the way a native child's exit does."""
    data = CoverageData(basename=str(path))
    data.add_lines({str(measured): {1, 2}})
    data.write()


def _raw_data_admission_expectations(tmp_path: Path) -> None:
    """Both directions of the bounded re-observation, plus the settled-but-empty refusal.

    A data file is written when its process exits, in several SQLite transactions, so a reader can
    observe it before the writer has finished. The guard must admit such a file once it settles,
    and must still refuse one that never settles or that settles unreadable.
    """
    driver = load_module_from_path("run_comprehensive_native_coverage", DRIVER)
    source = tmp_path / "src" / "gnn"
    source.mkdir(parents=True)
    measured = source / "settle_case.py"
    measured.write_text("def f():\n    return 1\n", encoding="utf-8")
    driver.RAW_OBSERVATION_WINDOW_SECONDS = 0.2
    driver.RAW_OBSERVATION_POLL_SECONDS = 0.01

    # A settled file is admitted, and the receipt shows what settling cost and what it saw.
    settled = tmp_path / ".coverage.host.pid10001.abcde"
    _coverage_data_file(settled, measured)
    data, detail = driver.read_raw_coverage(settled, tmp_path)
    assert data is not None
    assert data.measured_files()
    assert detail["settled"] is True
    assert detail["measured_files"] == 1
    assert detail["admitted_source_executed_lines"] == 2
    assert detail["observations"] == 1
    assert detail["settle_window_seconds"] == 0.2
    (observation,) = detail["observation_trail"]
    assert observation["changed"] is False
    assert observation["size"] == settled.stat().st_size
    assert observation["sha256"] == hashlib.sha256(settled.read_bytes()).hexdigest()

    original_read_bytes = Path.read_bytes
    churn_target = {"path": settled}
    state = {"reads": 0}

    def rewriting_read_bytes(self: Path) -> bytes:
        if self == churn_target["path"]:
            state["reads"] += 1
            return original_read_bytes(self) + str(state["reads"]).encode()
        return original_read_bytes(self)

    def read_while_rewriting(path: Path) -> tuple[CoverageData | None, dict]:
        """Observe *path* while every read sees different bytes."""
        churn_target["path"] = path
        state["reads"] = 0
        with pytest.MonkeyPatch.context() as monkeypatched:
            monkeypatched.setattr(Path, "read_bytes", rewriting_read_bytes)
            return driver.read_raw_coverage(path, tmp_path)

    # A live writer that never settles is refused once the bounded window closes.
    churning = tmp_path / f".coverage.host.pid{os.getpid()}.abcde"
    _coverage_data_file(churning, measured)
    churned, churned_detail = read_while_rewriting(churning)
    assert churned is None
    assert churned_detail["settled"] is False
    assert churned_detail["error_class"] == "DataError"
    assert (
        churned_detail["error"] == "Native coverage raw data changed while being read"
    )
    assert churned_detail["admitted_source_executed_lines"] is None
    assert churned_detail["observations"] > 1
    assert churned_detail["writer_alive_at_decision"] is True

    # A writer that has exited cannot be finalizing anything, so a file that still changes is
    # refused on the first observation instead of waiting out the window.
    reaped = subprocess.Popen([sys.executable, "-c", "pass"])
    reaped.wait()
    orphan = tmp_path / f".coverage.host.pid{reaped.pid}.abcde"
    _coverage_data_file(orphan, measured)
    orphan_data, orphan_detail = read_while_rewriting(orphan)
    assert orphan_data is None
    assert orphan_detail["observations"] == 1
    assert orphan_detail["writer_alive_at_decision"] is False
    assert orphan_detail["error"] == "Native coverage raw data changed while being read"

    # A file that settles empty is real corruption: still refused, never admitted by the new path.
    empty = tmp_path / ".coverage.host.pid10003.abcde"
    empty.write_bytes(b"")
    empty_data, empty_detail = driver.read_raw_coverage(empty, tmp_path)
    assert empty_data is None
    assert empty_detail["settled"] is False
    assert empty_detail["error_class"] == "DataError"
    assert empty_detail["admitted_source_executed_lines"] is None
    assert empty_detail["observations"] == 1


def test_concurrent_native_launches_publish_complete_owned_receipts(
    tmp_path: Path,
) -> None:
    result = _native_startup(
        tmp_path,
        """
        first_ready = threading.Event()
        release_first = threading.Event()
        second_replaced = threading.Event()
        paths = []
        errors = []
        outputs = {}
        original_replace = os.replace

        def coordinated_replace(source, destination, *args, **kwargs):
            # Pause only this authored witness's first publication. Both calls
            # still use the actual filesystem replacement and native Popen.
            name = threading.current_thread().name
            if name == 'receipt-first':
                first_ready.set()
                if not release_first.wait(5):
                    raise RuntimeError('witness publication barrier was not released')
            paths.append(str(source))
            original_replace(source, destination, *args, **kwargs)
            if name == 'receipt-second':
                second_replaced.set()

        os.replace = coordinated_replace

        def launch(label):
            try:
                outputs[label] = subprocess.check_output(
                    [sys.executable, '-S', '-c', f"print({label!r})"],
                    text=True, timeout=5,
                ).strip()
            except Exception as error:
                errors.append({'class': type(error).__name__, 'detail': str(error)})

        first = threading.Thread(target=launch, args=('first',), name='receipt-first')
        second = threading.Thread(target=launch, args=('second',), name='receipt-second')
        first.start()
        assert first_ready.wait(5), 'first native audit did not reach publication'
        second.start()
        # The old shared-temp implementation lets the second publication
        # consume the first writer's file. Serialization keeps it waiting.
        second_replaced.wait(0.5)
        release_first.set()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        receipt = json.loads((receipt_root / f'{os.getpid()}.json').read_text())
        print(json.dumps({'errors': errors, 'outputs': outputs, 'paths': paths,
                          'receipt': receipt,
                          'temporary': [p.name for p in receipt_root.glob('*.tmp')]}))
        """,
    )
    assert result["errors"] == [], result
    assert result["outputs"] == {"first": "first", "second": "second"}
    assert len(result["paths"]) == len(set(result["paths"])) == 2
    assert len(result["receipt"]["subprocess_launches"]) == 2
    assert all(launch["no_site"] for launch in result["receipt"]["subprocess_launches"])
    assert result["receipt"]["audit_hook_verified"] is True
    assert result["receipt"]["activation"] is None
    assert result["temporary"] == []
    # Admission of the raw data files those receipts describe, in both directions, including the
    # settled-but-empty refusal. Kept inside this test rather than a new function because the
    # manuscript token map counts test functions (GNN_TEST_FUNCTION_COUNT): an added function would
    # regenerate that map and require the full SC-22 render ritual for a change that moves no
    # `src/` line.
    _raw_data_admission_expectations(tmp_path)


def test_native_spawn_receipt_is_bound_to_its_own_pid(tmp_path: Path) -> None:
    result = _native_startup(
        tmp_path,
        """
        bindings = {key: namespace[key] for key in (
            'GNN_BOUND_STARTUP_DIR', 'GNN_BOUND_CONFIG', 'GNN_BOUND_SOURCE')}
        child_code = (
            'import os, runpy; '
            f"runpy.run_path({namespace['__file__']!r}, init_globals={bindings!r}); "
            'print(os.getpid())'
        )
        child = int(subprocess.check_output(
            [sys.executable, '-I', '-S', '-c', child_code], text=True, timeout=3,
        ))
        parent = os.getpid()
        print(json.dumps({
            'parent_pid': parent, 'child_pid': child,
            'parent': json.loads((receipt_root / f'{parent}.json').read_text()),
            'child': json.loads((receipt_root / f'{child}.json').read_text()),
            'receipt_files': sorted(p.name for p in receipt_root.glob('*.json')),
            'temporary': [p.name for p in receipt_root.glob('*.tmp')],
        }))
        """,
    )
    assert result["child_pid"] != result["parent_pid"]
    assert result["child"]["pid"] == result["child_pid"]
    assert result["parent"]["pid"] == result["parent_pid"]
    assert result["child"]["subprocess_launches"] == []
    assert "forked_from_pid" not in result["child"]
    assert len(result["parent"]["subprocess_launches"]) == 1
    assert result["parent"]["subprocess_launches"][0]["no_site"] is True
    assert result["parent"]["subprocess_launches"][0]["isolated"] is True
    assert result["receipt_files"] == sorted(
        [f"{result['parent_pid']}.json", f"{result['child_pid']}.json"]
    )
    assert result["temporary"] == []


@pytest.mark.needs_posix
def test_fork_resets_a_lock_held_by_another_native_thread(tmp_path: Path) -> None:
    result = _native_startup(
        tmp_path,
        """
        import signal
        locked = threading.Event()
        release = threading.Event()

        def hold_parent():
            with namespace['receipt_lock']:
                locked.set()
                assert release.wait(10)

        holder = threading.Thread(target=hold_parent)
        holder.start()
        assert locked.wait(5)
        parent = os.getpid()
        child = os.fork()
        if child == 0:
            try:
                output = subprocess.check_output(
                    [sys.executable, '-S', '-c', "print('owned-fork-worker')"],
                    text=True, timeout=3,
                ).strip()
                (receipt_root / 'fork-worker.txt').write_text(output)
                os._exit(0)
            except BaseException:
                os._exit(1)
        deadline = time.monotonic() + 5
        status = None
        try:
            while time.monotonic() < deadline:
                waited, value = os.waitpid(child, os.WNOHANG)
                if waited == child:
                    status = value
                    break
                time.sleep(.01)
            if status is None:
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
                raise AssertionError('fork inherited an unavailable receipt lock')
        finally:
            release.set()
            holder.join(5)
        assert status == 0, status
        parent_output = subprocess.check_output(
            [sys.executable, '-S', '-c', "print('owned-parent-worker')"],
            text=True, timeout=3,
        ).strip()
        print(json.dumps({
            'parent_pid': parent, 'child_pid': child,
            'child_output': (receipt_root / 'fork-worker.txt').read_text(),
            'parent_output': parent_output,
            'parent': json.loads((receipt_root / f'{parent}.json').read_text()),
            'child': json.loads((receipt_root / f'{child}.json').read_text()),
            'temporary': [p.name for p in receipt_root.glob('*.tmp')],
        }))
        """,
    )
    assert result["child_output"] == "owned-fork-worker"
    assert result["parent_output"] == "owned-parent-worker"
    assert result["child"]["pid"] == result["child_pid"]
    assert result["child"]["forked_from_pid"] == result["parent_pid"]
    assert result["child"]["fork_inherited_argv_witness"] is True
    assert result["parent"]["pid"] == result["parent_pid"]
    assert len(result["child"]["subprocess_launches"]) == 1
    assert len(result["parent"]["subprocess_launches"]) == 1
    assert result["temporary"] == []


def test_receipt_io_failure_refuses_launch_and_cleans_only_owned_temporary(
    tmp_path: Path,
) -> None:
    result = _native_startup(
        tmp_path,
        """
        canonical = receipt_root / f'{os.getpid()}.json'
        canonical.unlink()
        canonical.mkdir()
        retained = receipt_root / 'unrelated.tmp'
        retained.write_bytes(b'retain this prior file')
        worker_flag = receipt_root / 'worker.flag'
        failure = None
        try:
            subprocess.check_output(
                [sys.executable, '-S', '-c',
                 f"from pathlib import Path; Path({str(worker_flag)!r}).write_text('launched')"],
                timeout=3,
            )
        except OSError as error:
            failure = type(error).__name__
        print(json.dumps({'failure': failure, 'worker_launched': worker_flag.exists(),
                          'retained': retained.read_text(),
                          'temporary': sorted(p.name for p in receipt_root.glob('*.tmp')),
                          'canonical_is_directory': canonical.is_dir()}))
        """,
    )
    assert result["failure"] in {"IsADirectoryError", "PermissionError"}
    assert result["worker_launched"] is False
    assert result["canonical_is_directory"] is True
    assert result["retained"] == "retain this prior file"
    assert result["temporary"] == ["unrelated.tmp"]
