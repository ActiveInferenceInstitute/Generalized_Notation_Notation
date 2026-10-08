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

STARTUP = '''"""Record native interpreter/source resolution without importing GNN."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path(os.environ.get("GNN_NATIVE_COVERAGE_STARTUP_DIR") or globals()["GNN_BOUND_STARTUP_DIR"])
spec = importlib.util.find_spec("gnn")
distributions = sorted(
    [{"name": distribution.metadata["Name"], "version": distribution.version}
     for distribution in importlib.metadata.distributions()],
    key=lambda item: (item["name"].lower(), item["version"]),
)
record = {
    "pid": os.getpid(),
    "python": list(sys.version_info[:3]),
    "executable": str(Path(sys.executable).resolve()),
    "environment": str(Path(sys.prefix).resolve()),
    "installed_distributions_sha256": hashlib.sha256(
        json.dumps(distributions, sort_keys=True).encode()
    ).hexdigest(),
    "gnn_origin": str(Path(spec.origin).resolve()) if spec and spec.origin else None,
}
(root / (str(os.getpid()) + ".json")).write_text(json.dumps(record) + "\\n")
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
    support: Path, startups: Path, owned_environment: Path, audit: dict
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
        "import runpy; runpy.run_path("
        + repr(str(support / "startup.py"))
        + ", init_globals={'GNN_BOUND_STARTUP_DIR': "
        + repr(str(startups))
        + "})\n"
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
    original["run"].update(
        patch=["subprocess"], parallel=True, data_file=str(data_file)
    )
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
        "schema_version": 1,
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "environment": str(Path(sys.prefix).resolve()),
        "installed_distributions": sorted(
            [
                {"name": distribution.metadata["Name"], "version": distribution.version}
                for distribution in importlib.metadata.distributions()
            ],
            key=lambda item: (item["name"].lower(), item["version"]),
        ),
        "platform": platform.platform(),
        "source_identity": before["manifest_sha256"],
        "commit": before["commit"],
        "lanes": [],
        "selection_overlaps": {},
        "complete": False,
        "original_core_gate_unchanged": True,
        "coverage_scope": "Full original denominator, per-native-environment union; no cross-Python union",
        "original_coverage_config": tomllib.loads(
            (checkout / "pyproject.toml").read_text()
        )["tool"]["coverage"],
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
            with observation_hook(support, startups, owned_environment, hook_audit):
                status = run(
                    command, checkout, env, lane / "run.log", args.lane_timeout
                )
        finally:
            receipt.setdefault("observation_hooks", {})[name] = hook_audit
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if status == 124:
            receipt["integrity_errors"].append("Native lane deadline exceeded: " + name)
            receipt["lanes"].append(
                {"name": name, "command": command, "returncode": status}
            )
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return status
        if identity(checkout) != before:
            reject("Source/test/corpus identity changed during a native lane")
        records = {
            int(path.stem): json.loads(path.read_text())
            for path in startups.glob("*.json")
        }
        measured = []
        out_of_scope = []
        for data_path in sorted(lane.glob(".coverage.*")):
            data = CoverageData(basename=str(data_path))
            data.read()
            executed_original = {
                filename: len(data.lines(filename) or [])
                for filename in data.measured_files()
                if Path(filename).resolve().is_relative_to(checkout / "src/gnn")
                and data.lines(filename)
            }
            match = re.search(r"(?:\.pid|\.)(\d+)\.", data_path.name)
            if not match or int(match.group(1)) not in records:
                if not executed_original:
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
                or process["installed_distributions_sha256"]
                != receipt["installed_distributions_sha256"]
                or process["gnn_origin"] != str(checkout / "src/gnn/__init__.py")
            ):
                reject(
                    f"Coverage process resolved a different interpreter/environment/source: {data_path.name}"
                )
            union.update(data)
            measured.append(
                {
                    "data_file": data_path.name,
                    "pid": process["pid"],
                    "measured_files": len(data.measured_files()),
                    "admitted_source_executed_lines": sum(executed_original.values()),
                }
            )
        if not measured:
            reject("Native coverage lane produced no identifiable process data")
        cov = Coverage(config_file=str(config))
        # Combine only admitted native data. Foreign/unknown data remains separate.
        admitted = lane / "admitted"
        admitted.mkdir()
        for process in measured:
            (admitted / process["data_file"]).hardlink_to(lane / process["data_file"])
        cov.combine(data_paths=[str(admitted)], keep=True)
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
        receipt["lanes"].append(
            {
                "name": name,
                "paths": paths,
                "marker": marker,
                "command": command,
                "returncode": status,
                "seconds": time.monotonic() - started,
                "collected_nodes": len(selected_items),
                "collection_processes": {
                    pid: len(items) for pid, items in collections.items()
                },
                "junit": junit_receipt(lane / "junit.xml"),
                "coverage": summary,
                "processes": measured,
                "out_of_scope_processes": out_of_scope,
            }
        )
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if status:
            return status
    union.write()
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
