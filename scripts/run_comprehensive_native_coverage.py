"""Report comprehensive coverage with native interpreter and source receipts.

Run with the provisioned environment's own Python, from a frozen checkout.
This observer leaves pyproject/lock/workflows and the existing core gate intact.
Its comprehensive report adds real child execution; it is a separate report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import sysconfig
import time
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from coverage import Coverage, CoverageData
from coverage.exceptions import DataError
from defusedxml import ElementTree

LANES = {
    "core": (
        ["tests"],
        "not pipeline and not mcp and not ollama and not env_heavy and not toolchain",
    ),
    "mcp": (["tests/mcp"], None),
    "pipeline": (
        ["tests"],
        "pipeline and not ollama and not env_heavy and (not toolchain or needs_posix)",
    ),
}

PILOT = (
    [
        "tests/execute/test_executor_lazy_imports.py::test_hardware_probe_reports_actual_devices_without_parent_import",
        "tests/main/test_module_entry_configuration.py::test_entrypoint_configuration_and_selection",
        "tests/pipeline/test_step_executor.py::TestConsolidatedEquivalence::test_step3_in_process_matches_subprocess_artifacts",
    ],
    None,
)

STARTUP = '''"""Observation-only audit handoff prototype; no GNN imports or signal changes."""
import importlib.util
import json
import os
import sys
import time

if not hasattr(sys, "_gnn_native_coverage_observer"):
    started = time.perf_counter()
    root = globals()["GNN_BOUND_STARTUP_DIR"]
    config_file = globals()["GNN_BOUND_CONFIG"]
    source_root = os.path.realpath(globals()["GNN_BOUND_SOURCE"])
    source_prefix = source_root + os.sep
    spec = importlib.util.find_spec("gnn")
    state = {
        "pid": os.getpid(),
        "python": list(sys.version_info[:3]),
        "executable": os.path.realpath(sys.executable),
        "environment": os.path.realpath(sys.prefix),
        "base_environment": os.path.realpath(sys.base_prefix),
        "gnn_origin": os.path.realpath(spec.origin) if spec and spec.origin else None,
        "startup_sys_path": list(sys.path),
        "argv": globals().get("GNN_BOUND_ORIGINAL_ARGV", list(sys.argv)),
        "activation": None,
        "subprocess_launches": [],
        "metadata_provenance": "Immutable parent inventory before/after lane; child records environment/path/source witnesses",
    }
    sys._gnn_native_coverage_observer = state
    state["preexisting_original_modules"] = sorted(
        name for name, module in list(sys.modules.items())
        if isinstance(getattr(module, "__dict__", {}).get("__file__"), str)
        and os.path.realpath(module.__dict__["__file__"]).startswith(source_prefix)
    )

    def persist():
        path = os.path.join(root, str(os.getpid()) + ".json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(state, stream)
            stream.write("\\n")
        os.replace(temporary, path)

    def activate(filename, kind="before_original_source_exec"):
        if state["activation"] is not None:
            return
        state["activation"] = {"kind": kind, "first_source": filename,
                               "sys_path": list(sys.path), "cwd": os.getcwd()}
        try:
            from coverage import Coverage
            current = Coverage.current()
            if current is None:
                current = Coverage(config_file=config_file, auto_data=True, data_suffix=True)
                state["activation"]["original_source_options"] = current.get_option("run:source")
                current.set_option("run:source", [source_root])
                state["activation"]["resolved_source_options"] = [source_root]
                current.start()
                # A public initial save establishes a valid conservative database.
                current.save()
                state["activation"]["coverage_owner"] = "native_observer"
            else:
                state["activation"]["coverage_owner"] = "existing_coverage"
            state["activation"]["data_file"] = current.get_data().data_filename()
            allowed = globals().get("GNN_BOUND_ADMITTED_PATHS", [])
            roots = [os.path.realpath(path) for path in allowed]
            roots.extend([os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix),
                          os.path.dirname(source_root)])
            extra = sorted({os.path.realpath(path or os.getcwd()) for path in sys.path
                            if not any(os.path.realpath(path or os.getcwd()) == root
                                       or os.path.realpath(path or os.getcwd()).startswith(root + os.sep)
                                       for root in roots)})
            state["activation"]["extra_sys_path_entries"] = extra
            from importlib.metadata import distributions
            import hashlib
            overlays = []
            for distribution in distributions(path=extra):
                overlays.append({
                    "name": distribution.metadata["Name"], "version": distribution.version,
                    "location": os.path.realpath(distribution.locate_file("")),
                    "metadata_sha256": hashlib.sha256(
                        (distribution.read_text("METADATA") or "").encode()
                    ).hexdigest(),
                })
            state["activation"]["extra_metadata_overlays"] = sorted(
                overlays, key=lambda item: (item["name"].lower(), item["version"], item["location"]))
        except Exception as error:
            state["activation"]["error_class"] = type(error).__name__
            state["activation"]["error"] = str(error)
            persist()
            raise
        persist()

    def audit(event, args):
        if event == "gnn.native.coverage.probe":
            state["audit_hook_verified"] = True
            return
        if event == "subprocess.Popen":
            arguments = args[1] if isinstance(args[1], (list, tuple)) else []
            state["subprocess_launches"].append({
                "executable": str(args[0]),
                "no_site": "-S" in arguments,
                "isolated": "-I" in arguments,
                "qualification": "-S bypasses .pth startup and receives no inferred child-source credit"
                if "-S" in arguments else None,
            })
            persist()
            return
        if event != "exec" or state["activation"] is not None:
            return
        filename = args[0].co_filename
        if not filename or filename.startswith("<"):
            return
        filename = os.path.realpath(filename)
        if filename.startswith(source_prefix):
            activate(filename)

    def fork_child():
        previous = state["activation"]
        parent_pid = state["pid"]
        state["pid"] = os.getpid()
        state["forked_from_pid"] = parent_pid
        state["fork_inherited_argv_witness"] = True
        state["startup_sys_path"] = list(sys.path)
        state["subprocess_launches"] = []
        state["activation"] = None
        if previous is not None:
            from coverage import Coverage
            current = Coverage.current()
            if current is not None:
                current.stop()
            activate(previous["first_source"], "fork_after_original_source_activation")
        else:
            persist()

    sys.addaudithook(audit)
    sys.audit("gnn.native.coverage.probe")
    if not state.get("audit_hook_verified"):
        state["observer_error"] = "Native audit hook registration did not receive its verification event"
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=fork_child)
    state["observer_startup_seconds"] = time.perf_counter() - started
    persist()
'''

OBSERVER = '''"""Record actual selected node identities and outcomes without altering tests."""
import json
import os
from pathlib import Path

def pytest_collection_finish(session):
    target = Path(os.environ["GNN_NATIVE_COVERAGE_COLLECTION"])
    target.mkdir(exist_ok=True)
    (target / (str(os.getpid()) + ".json")).write_text(json.dumps([
        {"nodeid": item.nodeid, "markers": sorted({mark.name for mark in item.iter_markers()})}
        for item in session.items
    ], indent=2) + "\\n")
'''


def run(
    command: list[str], checkout: Path, env: dict[str, str], log: Path, timeout: int
) -> int:
    """Preserve native command output and return its actual terminal status."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stream.write(
                "\nNative coverage lane exceeded its explicit outer deadline.\n"
            )
            stream.flush()
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=5)
            return 124


@contextmanager
def observation_hook(
    support: Path,
    startups: Path,
    owned_environment: Path,
    audit: dict,
    config: Path,
    source_root: Path,
    checkout: Path,
) -> Iterator[None]:
    """Observe same-environment isolated children without changing their tests."""
    prefix = Path(sys.prefix).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if (
        prefix != owned_environment.resolve()
        or prefix == Path(sys.base_prefix).resolve()
        or not purelib.is_relative_to(prefix)
        or not (prefix / "pyvenv.cfg").is_file()
    ):
        raise RuntimeError(
            "Observation hook requires the explicitly owned active virtualenv purelib"
        )
    existing = sorted((path.name for path in purelib.glob("*.pth")), key=str.casefold)
    filename = (
        (existing[-1] if existing else "zzzz")
        + ".gnn_native_cov_"
        + uuid.uuid4().hex
        + ".pth"
    )
    path = purelib / filename
    content = (
        "import sys, runpy; runpy.run_path("
        + repr(str(support / "startup.py"))
        + ", init_globals="
        + repr(
            {
                "GNN_BOUND_STARTUP_DIR": str(startups),
                "GNN_BOUND_CONFIG": str(config),
                "GNN_BOUND_SOURCE": str(source_root),
                "GNN_BOUND_ADMITTED_PATHS": [str(checkout), str(support)],
            }
        )[:-1]
        + ", 'GNN_BOUND_ORIGINAL_ARGV': list(sys.argv)}"
        + ")\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(content)
    audit.update(
        {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "environment": str(prefix),
            "purelib": str(purelib),
            "sorts_after_existing_pth": all(
                name.casefold() < filename.casefold() for name in existing
            ),
            "created_exclusively": True,
            "restored": False,
        }
    )
    try:
        yield
    finally:
        if not path.is_file() or path.read_bytes() != content:
            audit["restoration_error"] = (
                "Owned hook missing or modified; no replacement/deletion attempted"
            )
            raise RuntimeError(audit["restoration_error"])
        path.unlink()
        audit["restored"] = True


def identity(checkout: Path) -> dict:
    """Fingerprint tracked content plus pending code, fixtures and configuration."""
    manifest = {}
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=checkout)
        .decode()
        .split("\0")
    )
    candidates = {checkout / name for name in tracked if name}
    for directory in ("src", "tests", "input", "scripts"):
        for path in sorted((checkout / directory).rglob("*")):
            if not path.is_file() or any(
                part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                for part in path.parts
            ):
                continue
            candidates.add(path)
    for name in (
        "pyproject.toml",
        "uv.lock",
        "pytest.ini",
        "setup.cfg",
        "conftest.py",
        ".coveragerc",
    ):
        path = checkout / name
        if path.is_file():
            candidates.add(path)
    for path in sorted(candidates):
        relative = path.relative_to(checkout).as_posix()
        manifest[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "MISSING"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=checkout,
        text=True,
    ).splitlines()
    return {
        "commit": revision,
        "working_tree_changes": dirty,
        "identity_claim": "commit plus working tree" if dirty else "committed checkout",
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files": manifest,
    }


def write_config(checkout: Path, target: Path, data_file: Path) -> None:
    """Copy every original coverage option and add child tracing only."""
    original = tomllib.loads((checkout / "pyproject.toml").read_text())["tool"][
        "coverage"
    ]
    original["run"].update(parallel=True, data_file=str(data_file))
    target.write_text(
        "".join(
            "[tool.coverage."
            + section
            + "]\n"
            + "".join(
                key + " = " + json.dumps(value) + "\n" for key, value in values.items()
            )
            + "\n"
            for section, values in original.items()
        ),
        encoding="utf-8",
    )


def junit_receipt(path: Path) -> dict:
    """Count actual executed cases and preserve every failure/skip reason."""
    cases = list(ElementTree.parse(path).getroot().iter("testcase"))
    outcomes = {kind: [] for kind in ("failure", "error", "skipped")}
    for case in cases:
        for kind in outcomes:
            node = case.find(kind)
            if node is not None:
                outcomes[kind].append(
                    {
                        "identity": case.get("classname", "")
                        + "::"
                        + case.get("name", ""),
                        "message": node.get("message", ""),
                        "detail": node.text or "",
                    }
                )
    return {
        "testcases": len(cases),
        "executed": len(cases) - len(outcomes["skipped"]),
        "passed": len(cases) - sum(map(len, outcomes.values())),
        "failures": outcomes["failure"],
        "errors": outcomes["error"],
        "skips": outcomes["skipped"],
    }


def read_raw_coverage(path: Path, checkout: Path) -> tuple[CoverageData | None, dict]:
    """Retain unreadable native data as unknown, never as zero contribution."""
    match = re.search(r"(?:\.pid|\.)(\d+)\.", path.name)
    detail = {"data_file": path.name, "pid": int(match.group(1)) if match else None}
    try:
        detail["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = CoverageData(basename=str(path))
        data.read()
        executed_original = {
            filename: len(data.lines(filename) or [])
            for filename in data.measured_files()
            if Path(filename).resolve().is_relative_to(checkout / "src/gnn")
            and data.lines(filename)
        }
        detail.update(
            measured_files=len(data.measured_files()),
            admitted_source_executed_lines=sum(executed_original.values()),
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != detail["sha256"]:
            raise DataError("Native coverage raw data changed while being read")
        return data, detail
    except (DataError, OSError) as error:
        detail.update(
            error_class=type(error).__name__,
            error=str(error),
            admitted_source_executed_lines=None,
            qualification="Unreadable raw data retained; original-source execution is unknown; never admitted",
        )
        return None, detail


def installed_inventory() -> list[dict]:
    """Snapshot native installed metadata outside measured child startup."""
    return sorted(
        [
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "location": str(Path(distribution.locate_file("")).resolve()),
                "metadata_sha256": hashlib.sha256(
                    (distribution.read_text("METADATA") or "").encode()
                ).hexdigest(),
            }
            for distribution in importlib.metadata.distributions()
        ],
        key=lambda item: (item["name"].lower(), item["version"], item["location"]),
    )


def line_sets(data: CoverageData) -> dict[str, list[int]]:
    """Return exact measured line sets, including zero-contribution files."""
    return {
        name: sorted(data.lines(name) or []) for name in sorted(data.measured_files())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python-version", choices=("3.11", "3.12", "3.13"), required=True
    )
    parser.add_argument("--lane", choices=tuple(LANES), action="append")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Only the five existing isolation/subprocess observer controls; never a full report",
    )
    parser.add_argument(
        "--owned-environment",
        type=Path,
        help="Declared owned active venv; defaults to checkout/.venv",
    )
    parser.add_argument(
        "--lane-timeout",
        type=int,
        default=1800,
        help="Outer seconds allowed per native lane",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Explicit bounded core worker count; 0 keeps core serial. MCP/pipeline remain serial.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be a nonnegative integer")
    if args.lane_timeout <= 0:
        parser.error("--lane-timeout must be positive")
    if args.pilot and args.lane:
        parser.error("--pilot and --lane are mutually exclusive")
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != args.python_version:
        raise SystemExit(
            f"Expected native Python{args.python_version}, running{actual}"
        )
    checkout = args.checkout.resolve()
    original_coverage = tomllib.loads((checkout / "pyproject.toml").read_text())[
        "tool"
    ]["coverage"]
    if original_coverage["run"].get("source") != ["src/gnn"]:
        parser.error(
            "The observer requires the declared full original source scope ['src/gnn']"
        )
    owned_environment = args.owned_environment or checkout / ".venv"
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(
            "Use a new output directory; existing evidence is never overwritten"
        )
    output.mkdir(parents=True)
    # Original source paths are relative to checkout in both execution and reports.
    os.chdir(checkout)
    before = identity(checkout)
    (output / "source-identity.json").write_text(json.dumps(before, indent=2) + "\n")
    support = output / "observer"
    support.mkdir()
    (support / "startup.py").write_text(STARTUP)
    (support / "gnn_native_coverage_observer.py").write_text(OBSERVER)
    selected = ["observer-pilot"] if args.pilot else args.lane or list(LANES)
    receipt = {
        "schema_version": 2,
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "environment": str(Path(sys.prefix).resolve()),
        "base_environment": str(Path(sys.base_prefix).resolve()),
        "installed_distributions": installed_inventory(),
        "metadata_provenance": "Parent immutable inventory before/after each lane; native child prefix/path/source and extra-metadata witnesses",
        "source_resolution": {
            "original": ["src/gnn"],
            "child_absolute": [str(checkout / "src/gnn")],
        },
        "child_tracing": "Public exec audit before first original-source frame; stdlib-only children remain untraced and receipted",
        "platform": platform.platform(),
        "source_identity": before["manifest_sha256"],
        "commit": before["commit"],
        "lanes": [],
        "selection_overlaps": {},
        "complete": False,
        "original_core_gate_unchanged": True,
        "coverage_scope": "Full original denominator, per-native-environment union; no cross-Python union",
        "original_coverage_config": original_coverage,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_workers": args.workers,
        "lane_timeout_seconds": args.lane_timeout,
        "deadline_cleanup_scope": "POSIX pytest process group or Windows pytest process; detached descendant cleanup is not certified by the observer",
        "observer_pilot": args.pilot,
        "integrity_errors": [],
    }
    receipt["installed_distributions_sha256"] = hashlib.sha256(
        json.dumps(receipt["installed_distributions"], sort_keys=True).encode()
    ).hexdigest()
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def reject(message: str) -> None:
        receipt["integrity_errors"].append(message)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise SystemExit(message)

    union = CoverageData(basename=str(output / ".coverage-comprehensive"))
    expected_union_lines = {}
    expected_report_lines = {}
    selections = {}
    denominator = None
    for name in selected:
        if identity(checkout) != before:
            reject("Source/test/corpus identity changed before a native lane")
        lane = output / name
        lane.mkdir()
        startups = lane / "startups"
        startups.mkdir()
        config = lane / "coverage.toml"
        write_config(checkout, config, lane / ".coverage")
        env = dict(os.environ)
        env.pop("COVERAGE_PROCESS_CONFIG", None)
        env.pop("COVERAGE_PROCESS_START", None)
        env.update(
            MPLBACKEND="Agg",
            PYTHONHASHSEED="0",
            PYTHONPATH=os.pathsep.join((str(support), str(checkout / "src"))),
            GNN_NATIVE_COVERAGE_STARTUP_DIR=str(startups),
            GNN_NATIVE_COVERAGE_COLLECTION=str(lane / "collections"),
        )
        paths, marker = PILOT if name == "observer-pilot" else LANES[name]
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
            *paths,
            "-p",
            "gnn_native_coverage_observer",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "-rsx",
            "--junitxml=" + str(lane / "junit.xml"),
            "--basetemp=" + str(lane / "tmp"),
        ]
        if marker:
            command += ["-m", marker]
        if name in {"core", "observer-pilot"} and args.workers:
            command += ["-n", str(args.workers), "--dist", "worksteal"]
        started = time.monotonic()
        hook_audit = {}
        try:
            with observation_hook(
                support,
                startups,
                owned_environment,
                hook_audit,
                config,
                checkout / "src/gnn",
                checkout,
            ):
                status = run(
                    command, checkout, env, lane / "run.log", args.lane_timeout
                )
        finally:
            receipt.setdefault("observation_hooks", {})[name] = hook_audit
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        lane_receipt = {
            "name": name,
            "paths": paths,
            "marker": marker,
            "command": command,
            "returncode": status,
            "seconds": time.monotonic() - started,
            "junit": junit_receipt(lane / "junit.xml")
            if (lane / "junit.xml").is_file()
            else None,
            "processes": [],
            "out_of_scope_processes": [],
            "invalid_raw_files": [],
        }
        receipt["lanes"].append(lane_receipt)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if status == 124:
            receipt["integrity_errors"].append("Native lane deadline exceeded: " + name)
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return status
        if identity(checkout) != before:
            reject("Source/test/corpus identity changed during a native lane")
        after_inventory = installed_inventory()
        lane_receipt["installed_distributions_after_sha256"] = hashlib.sha256(
            json.dumps(after_inventory, sort_keys=True).encode()
        ).hexdigest()
        if after_inventory != receipt["installed_distributions"]:
            reject("Native installed metadata changed during a coverage lane")
        records = {
            int(path.stem): json.loads(path.read_text())
            for path in startups.glob("*.json")
        }
        lane_receipt["startup_processes"] = records
        if any(record.get("observer_error") for record in records.values()):
            reject("Native process could not establish its exec audit hook")
        if any(
            record.get("preexisting_original_modules") for record in records.values()
        ):
            reject("Original-source modules loaded before the native observation hook")
        if any(
            (record.get("activation") or {}).get("error_class")
            for record in records.values()
        ):
            reject("Native original-source coverage activation failed")
        lane_receipt["zero_source_execution_processes"] = sorted(
            pid for pid, record in records.items() if record["activation"] is None
        )
        measured = lane_receipt["processes"]
        out_of_scope = lane_receipt["out_of_scope_processes"]
        lane_union = CoverageData(basename=str(lane / ".coverage"))
        expected_lane_lines = {}
        for data_path in sorted(lane.glob(".coverage.*")):
            data, raw_receipt = read_raw_coverage(data_path, checkout)
            if data is None:
                lane_receipt["invalid_raw_files"].append(raw_receipt)
                continue
            match = re.search(r"(?:\.pid|\.)(\d+)\.", data_path.name)
            if not match or int(match.group(1)) not in records:
                if raw_receipt["admitted_source_executed_lines"] == 0:
                    out_of_scope.append(
                        {
                            "data_file": data_path.name,
                            "sha256": hashlib.sha256(
                                data_path.read_bytes()
                            ).hexdigest(),
                            "reason": "No executed admitted source and no native startup receipt; never unioned",
                            "admitted_source_executed_lines": 0,
                            "measured_files": len(data.measured_files()),
                        }
                    )
                    continue
                reject(
                    f"Coverage process has no native identity receipt: {data_path.name}"
                )
            process = records[int(match.group(1))]
            if process["python"][:2] != [
                sys.version_info.major,
                sys.version_info.minor,
            ]:
                reject(f"Refusing cross-Python union: {data_path.name}")
            if (
                process["executable"] != receipt["executable"]
                or process["environment"] != receipt["environment"]
                or process["base_environment"] != receipt["base_environment"]
                or process["gnn_origin"] != str(checkout / "src/gnn/__init__.py")
                or (process.get("activation") or {}).get("extra_metadata_overlays")
            ):
                if raw_receipt["admitted_source_executed_lines"] == 0:
                    raw_receipt["reason"] = (
                        "Different native source/environment/metadata overlay with zero executed original-source lines; retained and never unioned"
                    )
                    out_of_scope.append(raw_receipt)
                    continue
                reject(
                    f"Coverage process resolved a different interpreter/environment/source: {data_path.name}"
                )
            union.update(data)
            lane_union.update(data)
            for filename, lines in line_sets(data).items():
                expected_lane_lines.setdefault(filename, set()).update(lines)
                expected_union_lines.setdefault(filename, set()).update(lines)
            measured.append(raw_receipt)
        if lane_receipt["invalid_raw_files"]:
            reject(
                "Unreadable native coverage data retained; original-source contribution unknown: "
                + ", ".join(
                    item["data_file"] for item in lane_receipt["invalid_raw_files"]
                )
            )
        if not measured:
            reject("Native coverage lane produced no identifiable process data")
        cov = Coverage(config_file=str(config))
        # Combine only admitted native data. Foreign/unknown data remains separate.
        admitted = lane / "admitted"
        admitted.mkdir()
        for process in measured:
            (admitted / process["data_file"]).hardlink_to(lane / process["data_file"])
        # Public initial saves can leave equal filename hashes on unequal data.
        # Update from admitted data directly; never dedupe by that filename hint.
        lane_union.write()
        lane_loaded = CoverageData(basename=str(lane / ".coverage"))
        lane_loaded.read()
        expected_lane = {
            name: sorted(lines) for name, lines in sorted(expected_lane_lines.items())
        }
        if line_sets(lane_loaded) != expected_lane:
            reject(
                "Native lane data differs from the exact admitted raw line-set union"
            )
        lane_receipt["raw_line_set_union_sha256"] = hashlib.sha256(
            json.dumps(expected_lane, sort_keys=True).encode()
        ).hexdigest()
        cov.load()
        cov.json_report(outfile=str(lane / "coverage.json"))
        collections = {
            path.stem: json.loads(path.read_text())
            for path in sorted((lane / "collections").glob("*.json"))
        }
        actual_collections = [items for items in collections.values() if items]
        if not actual_collections:
            reject("Native coverage lane collected no test identities")
        selected_items = actual_collections[0]
        if any(items != selected_items for items in actual_collections):
            reject(
                "Native pytest workers reported different selected node/marker identities"
            )
        (lane / "collection.json").write_text(
            json.dumps(selected_items, indent=2) + "\n"
        )
        selections[name] = {item["nodeid"] for item in selected_items}
        report = json.loads((lane / "coverage.json").read_text())
        for path, item in report["files"].items():
            expected_report_lines.setdefault(path, set()).update(item["executed_lines"])
        actual_denominator = {
            path: {
                "statements": data["summary"]["num_statements"],
                "excluded_lines": data["excluded_lines"],
            }
            for path, data in report["files"].items()
        }
        if denominator is not None and actual_denominator != denominator:
            reject(
                "Native coverage lanes report different source/statement/exclusion denominators"
            )
        denominator = actual_denominator
        summary = report["totals"]
        lane_receipt.update(
            {
                "collected_nodes": len(selected_items),
                "collection_processes": {
                    pid: len(items) for pid, items in collections.items()
                },
                "coverage": summary,
            }
        )
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if status:
            return status
    union.write()
    union_loaded = CoverageData(basename=str(output / ".coverage-comprehensive"))
    union_loaded.read()
    expected_union = {
        name: sorted(lines) for name, lines in sorted(expected_union_lines.items())
    }
    if line_sets(union_loaded) != expected_union:
        reject(
            "Comprehensive data differs from the exact admitted native raw line-set union"
        )
    receipt["raw_line_set_union_sha256"] = hashlib.sha256(
        json.dumps(expected_union, sort_keys=True).encode()
    ).hexdigest()
    combined_config = output / "coverage-comprehensive.toml"
    write_config(checkout, combined_config, output / ".coverage-comprehensive")
    combined = Coverage(config_file=str(combined_config))
    combined.load()
    combined.json_report(outfile=str(output / "coverage-comprehensive.json"))
    combined_report = json.loads((output / "coverage-comprehensive.json").read_text())
    combined_denominator = {
        path: {
            "statements": data["summary"]["num_statements"],
            "excluded_lines": data["excluded_lines"],
        }
        for path, data in combined_report["files"].items()
    }
    if combined_denominator != denominator:
        reject("Combined coverage report changed the native lane denominator")
    actual_report_lines = {
        path: item["executed_lines"] for path, item in combined_report["files"].items()
    }
    if actual_report_lines != {
        path: sorted(lines) for path, lines in expected_report_lines.items()
    }:
        reject(
            "Comprehensive report differs from the exact native lane executed-statement union"
        )
    receipt["coverage"] = combined_report["totals"]
    receipt["source_files"] = len(combined_report["files"])
    receipt["selection_overlaps"] = {
        first + "/" + second: len(selections[first] & selections[second])
        for first in selections
        for second in selections
        if first < second
    }
    receipt["unique_selected_nodes"] = len(set().union(*selections.values()))
    receipt["complete"] = True
    receipt["complete_declared_core_mcp_pipeline_selection"] = set(selected) == set(
        LANES
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "complete",
                    "coverage",
                    "source_files",
                    "unique_selected_nodes",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
