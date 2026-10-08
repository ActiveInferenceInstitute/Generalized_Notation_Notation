"""Real startup-audit receipt publication under parallel launch and fork."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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
        [sys.executable, "-I", str(bootstrap)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    (tmp_path / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "stderr.log").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    return json.loads(result.stdout)


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
    assert result["receipt"]["audit_hook_verified"] is True
    assert result["receipt"]["activation"] is None
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
