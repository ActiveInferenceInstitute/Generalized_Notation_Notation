"""Run the read-only FEP bridge in the inspected matrix interpreter.

Receipts bind live source bytes, installed environment identity and actual CLI
execution. They are workflow custody evidence, never native Lean evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import re
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

STAGES = ("installed", "status", "finite", "continuous", "final")
Q7_STAGE = "q7-source"
Q7_CONTRACT = "specs/gnn-bridge-q7-continuous-ou-proof/verify_native.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object receipt: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    # Never replace a prior success, partial result or failed receipt.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.lstat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "link": os.readlink(path) if path.is_symlink() else None,
        "sha256": digest(path),
    }


def runtime(expected: str, environment: Path) -> dict[str, Any]:
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected:
        raise ValueError(f"interpreter mismatch: requested {expected}, actual {actual}")
    executable = Path(sys.executable).absolute()  # Preserve the venv symlink.
    if Path(sys.prefix).resolve() != environment.resolve():
        raise ValueError(f"virtual environment mismatch: {sys.prefix}")
    if executable.parent.parent.resolve() != environment.resolve():
        raise ValueError(f"executable outside installed environment: {executable}")
    stat = environment.stat()
    return {
        "requested_python": expected,
        "actual_python": actual,
        "version": sys.version,
        "executable": str(executable),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "environment_device": stat.st_dev,
        "environment_inode": stat.st_ino,
        "executable_identity": file_identity(executable),
        "base_executable_identity": file_identity(executable.resolve()),
        "pyvenv_cfg_identity": file_identity(environment / "pyvenv.cfg"),
    }


def source(root: Path, expected_revision: str) -> dict[str, Any]:
    command = ["git", "-c", "core.fsmonitor=false", "-C", str(root)]
    revision = subprocess.check_output(
        command + ["rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected_revision:
        raise ValueError(f"source revision mismatch: {root}: {revision}")
    subprocess.run(
        command + ["diff", "--quiet", "--no-ext-diff", "HEAD", "--"], check=True
    )
    names = subprocess.check_output(command + ["ls-files", "-z"]).split(b"\0")
    tree = hashlib.sha256()
    for name in sorted(filter(None, names)):
        path = root / os.fsdecode(name)
        content = (
            os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        )
        tree.update(name + b"\0" + hashlib.sha256(content).digest())
    return {
        "root": str(root),
        "revision": revision,
        "tracked_tree_sha256": tree.hexdigest(),
        "lock_sha256": digest(root / "uv.lock"),
    }


def fep_namespace(root: Path) -> str:
    """Prefer the canonical source layout; retain pinned legacy checkouts."""
    return "fep_formal" if (root / "src/fep_formal").is_dir() else "fep_lean"


def fep_module_path(root: Path, suffix: str) -> Path:
    """Require an exact source file, refusing symlinked package escapes."""
    expected = root / "src" / fep_namespace(root) / (suffix.replace(".", "/") + ".py")
    if expected.resolve() != expected or not expected.is_file():
        raise ValueError(f"invalid FEP module source: {expected}")
    return expected


def import_fep_module(root: Path, suffix: str) -> ModuleType:
    """Import only the module belonging to the selected checkout namespace."""
    expected = fep_module_path(root, suffix)
    module = importlib.import_module(f"{fep_namespace(root)}.{suffix}")
    actual = Path(module.__file__).resolve() if module.__file__ else None
    if actual != expected:
        raise ValueError(
            f"installed FEP module source mismatch: {actual}; expected {expected}"
        )
    return module


def release_binding(args: argparse.Namespace) -> dict[str, Any]:
    """Admit only a reviewed version/lock delta from the fixed scientific pair."""
    release = args.gnn_release_root
    policy = read(release / ".github/fep-release-baseline.json")
    required = {
        "schema_version",
        "mode",
        "fep_revision",
        "gnn_revision",
        "release_version",
        "approved_lock_sha256",
        "resynchronize",
    }
    if (
        set(policy) != required
        or policy["schema_version"] != 1
        or policy["mode"] != "fixed-release-baseline"
        or policy["resynchronize"] != "next-major-release"
        or policy["gnn_revision"] != args.gnn_revision
        or policy["fep_revision"] != args.fep_revision
    ):
        raise ValueError("invalid fixed release baseline")
    owner_roster = import_fep_module(args.fep_root, "bridge.operations").owner_roster

    pin = read(args.fep_root / "specs/gnn-bridge-w2-source-custody/source-pin.json")
    owners = pin["gnn"]["owners"]
    if pin["gnn"]["commit"] != args.gnn_revision:
        raise ValueError("baseline does not match the companion source pin")
    if owner_roster(release, "gnn") != sorted(owners):
        raise ValueError("release scientific owner roster changed")
    exceptions = {"pyproject.toml", "src/gnn/__init__.py", "uv.lock"}
    changed = {}
    for name, expected in owners.items():
        if digest(args.gnn_root / name) != expected:
            raise ValueError(f"fixed baseline owner changed: {name}")
        actual = digest(release / name)
        if actual != expected:
            if name not in exceptions:
                raise ValueError(f"release scientific owner changed: {name}")
            changed[name] = {"baseline_sha256": expected, "release_sha256": actual}
    old = tomllib.loads((args.gnn_root / "pyproject.toml").read_text())
    new = tomllib.loads((release / "pyproject.toml").read_text())
    if new["project"]["version"] != policy["release_version"]:
        raise ValueError("release version does not match baseline policy")
    old["project"]["version"] = new["project"]["version"]
    if old != new:
        raise ValueError("project metadata changed beyond the release version")

    def without_version(root: Path) -> str:
        text = (root / "src/gnn/__init__.py").read_text()
        version = tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "version"
        ]
        pattern = r'^__version__ = "' + re.escape(version) + r'"$'
        normalized, count = re.subn(
            pattern, '__version__ = "RELEASE"', text, flags=re.M
        )
        if count != 1:
            raise ValueError("package version is not canonical")
        return normalized

    if without_version(args.gnn_root) != without_version(release):
        raise ValueError("package initializer changed beyond the release version")
    if digest(release / "uv.lock") != policy["approved_lock_sha256"]:
        raise ValueError("release dependency lock is not the reviewed frozen lock")
    return {
        "mode": policy["mode"],
        "policy_sha256": digest(release / ".github/fep-release-baseline.json"),
        "current_release": source(release, args.gnn_release_revision),
        "changed_owners": changed,
        "current_pair_synchronized": False,
        "native_claim_ready": False,
        "resynchronize": policy["resynchronize"],
    }


def custody(args: argparse.Namespace) -> dict[str, Any]:
    pin_root = args.gnn_release_root or args.gnn_root
    pin = read(pin_root / ".github/fep-lean-pair.json")
    if pin != {
        "repository": "ActiveInferenceInstitute/fep_formal",
        "revision": args.fep_revision,
    }:
        raise ValueError("companion pin does not match requested source")
    binding = release_binding(args) if args.gnn_release_root else None
    return {
        "release_binding": binding,
        "runtime": runtime(args.expected_python, args.environment),
        "gnn": source(args.gnn_root, args.gnn_revision),
        "fep": source(args.fep_root, args.fep_revision),
        "pin": pin,
    }


def bridge_command(fep: str, gnn: str, stage: str) -> list[str]:
    operation = (
        ["status"] if stage == "status" else ["emit", "--check", "--model", stage]
    )
    return ["--project-root", fep, "bridge", *operation, "--gnn-root", gnn]


def accepted(path: Path, stage: str, baseline: dict[str, Any]) -> dict[str, Any]:
    result = read(path)
    if (
        result.get("stage") != stage
        or result.get("status") != "ok"
        or result.get("native_claim_ready") is not False
        or result.get("before") != baseline
        or result.get("after") != baseline
    ):
        raise ValueError(f"missing or incompatible successful runtime receipt: {path}")
    if stage in ("status", "finite", "continuous"):
        if (
            result.get("exit_code") != 0
            or result.get("cli_source")
            != str(fep_module_path(Path(baseline["fep"]["root"]), "cli"))
            or result.get("command")
            != bridge_command(baseline["fep"]["root"], baseline["gnn"]["root"], stage)
        ):
            raise ValueError("wrong command or installed CLI receipt")
        output = path.parent / f"bridge-{stage}.json"
        payload = read(output)
        if (
            digest(output) != result.get("output_sha256")
            or payload.get("status") != "ok"
        ):
            raise ValueError(f"bridge output changed or failed: {output}")
        if stage == "status" and payload.get("native_claim_ready") is not False:
            raise ValueError("bridge status cannot promote a native claim")
        if stage != "status" and (
            payload.get("model") != stage or payload.get("read_only") is not True
        ):
            raise ValueError("incorrect emit-check output")
    if stage == Q7_STAGE:
        output = path.parent / "q7-current-source.json"
        payload = read(output)
        if (
            result.get("exit_code") != 0
            or result.get("verifier_source")
            != str(Path(baseline["fep"]["root"]) / Q7_CONTRACT)
            or result.get("command") != ["source_snapshot", baseline["gnn"]["root"]]
            or digest(output) != result.get("output_sha256")
            or payload.get("status") != "ok"
            or payload.get("native_claim_ready") is not False
            or not isinstance(payload.get("source_snapshot"), dict)
        ):
            raise ValueError("missing or incompatible current Q7 source evidence")
    return result


def execute(args: argparse.Namespace) -> None:
    receipt: dict[str, Any] = {
        "stage": args.stage,
        "status": "error",
        "native_claim_ready": False,
    }
    path = args.receipts / f"runtime-{args.stage}.json"
    try:
        if args.stage == Q7_STAGE and args.expected_python != "3.14":
            raise ValueError("current Q7 source checks require CPython 3.14")
        before = custody(args)
        receipt["before"] = before
        if args.stage == "installed":
            receipt["uv_version"] = subprocess.check_output(
                ["uv", "--version"], text=True
            ).strip()
        else:
            baseline = read(args.receipts / "runtime-installed.json")["before"]
            if before != baseline:
                raise ValueError(
                    "installed interpreter, environment or source identity changed"
                )
            prior = (
                STAGES if args.stage == Q7_STAGE else STAGES[: STAGES.index(args.stage)]
            )
            for stage in prior:
                accepted(args.receipts / f"runtime-{stage}.json", stage, baseline)
            if args.stage != "final":
                cli = import_fep_module(args.fep_root, "cli")
                module = fep_module_path(args.fep_root, "cli")
                if args.stage == Q7_STAGE:
                    verifier = args.fep_root / Q7_CONTRACT
                    contract = runpy.run_path(str(verifier), run_name="fep_current_q7")
                    snapshot = contract["VERIFICATION"].source_snapshot(args.gnn_root)
                    output = args.receipts / "q7-current-source.json"
                    write(
                        output,
                        {
                            "status": "ok",
                            "native_claim_ready": False,
                            "source_snapshot": snapshot,
                        },
                    )
                    receipt.update(
                        verifier_source=str(verifier),
                        command=["source_snapshot", str(args.gnn_root)],
                        exit_code=0,
                        output_sha256=digest(output),
                    )
                else:
                    command = bridge_command(
                        str(args.fep_root), str(args.gnn_root), args.stage
                    )
                    output = args.receipts / f"bridge-{args.stage}.json"
                    with (
                        output.open("x", encoding="utf-8") as stream,
                        contextlib.redirect_stdout(stream),
                    ):
                        code = cli.main(command)
                    receipt.update(
                        cli_source=str(module),
                        command=command,
                        exit_code=code,
                        output_sha256=digest(output),
                    )
                    if code != 0:
                        raise ValueError(f"bridge {args.stage} exited {code}")
        receipt["after"] = custody(args)
        if receipt["after"] != before:
            raise ValueError("runtime or source changed during bridge execution")
        receipt["status"] = "ok"
        write(path, receipt)
        accepted(path, args.stage, before)
    except Exception as exc:
        receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
        # Preserve any already-written receipt too; failures are additional evidence.
        write(args.receipts / f"runtime-{args.stage}-failed.json", receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, Q7_STAGE))
    parser.add_argument(
        "--expected-python", choices=("3.11", "3.12", "3.14"), required=True
    )
    for flag in ("environment", "gnn-root", "fep-root", "receipts"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--gnn-release-root", type=Path)
    parser.add_argument("--gnn-release-revision")
    parser.add_argument("--gnn-revision", required=True)
    parser.add_argument("--fep-revision", required=True)
    args = parser.parse_args()
    for name in ("environment", "gnn_root", "fep_root", "receipts"):
        setattr(args, name, getattr(args, name).resolve())
    if bool(args.gnn_release_root) != bool(args.gnn_release_revision):
        parser.error("release root and revision must be supplied together")
    if args.gnn_release_root:
        args.gnn_release_root = args.gnn_release_root.resolve()
    try:
        execute(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"paired runtime rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
