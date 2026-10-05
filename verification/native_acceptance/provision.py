"""Frozen ordinary-wheel provisioning, distinct from native acceptance.

The bootstrap uses bounded POSIX process groups before the installed public
process envelope is available. Its receipts claim direct-child reaping and
process-group disappearance, not universal detached-descendant containment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpyro", "rxinfer"), required=True)
    parser.add_argument("--rebuild-diagnostic", action="store_true")
    args = parser.parse_args()
    workspace = Path(os.environ["GNN_NATIVE_VERIFY_WORKSPACE"]).absolute()
    source_root = Path(os.environ["GNN_NATIVE_VERIFY_SOURCE_ROOT"]).absolute()
    require(workspace != source_root and not workspace.is_relative_to(source_root), "Isolated workspace required")
    require(all(not p.is_symlink() for p in (workspace, *workspace.parents, source_root, *source_root.parents)),
            "Symlink workspace or checkout refused")
    require(sys.platform == "linux" and sys.version_info[:3] == (3, 11, 15), "Frozen Linux Python 3.11.15 required")
    workspace.mkdir(parents=True, exist_ok=False)
    out = workspace / "evidence/provisioning"
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + 1200
    work_deadline = deadline - 15
    minimum = shutil.disk_usage(workspace).free
    record = {"accepted": False, "backend": args.backend, "commands": [],
              "inclusive_budget_seconds": 1200, "cleanup_reservation_seconds": 15,
              "scope": "Provisioning only; no native acceptance or inference execution."}
    try:
        require(minimum >= 6 * 1024**3, "RESOURCE_HOLD: provisioning requires six GiB")
        binding = json.loads((CODE / "source-binding.json").read_text())
        require(binding["release_binding_complete"] is True
                and binding["content_commit"] == "8d202439eaa386d13a9b93888a7d560657579a4b"
                and isinstance(binding["artifact_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", binding["artifact_commit"])
                and isinstance(binding["wheel_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", binding["wheel_sha256"]),
                "Unsealed release artifact/wheel binding refused")
        for name, expected in binding["release_owner_inventory"].items():
            path = source_root / name
            require(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    "Frozen release owner changed before provisioning: " + name)
        env = {name: value for name, value in os.environ.items()
               if name in {"PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TMPDIR",
                           "GNN_NATIVE_VERIFY_WORKSPACE", "GNN_NATIVE_VERIFY_SOURCE_ROOT"}}
        env.update(UV_DEFAULT_INDEX="https://pypi.org/simple", UV_NO_PROGRESS="1",
                   JULIA_PKG_PRECOMPILE_AUTO="0", GKSwstype="100", SOURCE_DATE_EPOCH="1580601600")

        def group_alive(pid):
            try:
                os.killpg(pid, 0)
                return True
            except ProcessLookupError:
                return False

        def execute(command, name):
            nonlocal minimum
            require(time.monotonic() < work_deadline, "Provisioning work deadline exhausted")
            row = {"name": name, "command": command, "accepted": False}
            record["commands"].append(row)
            child = None
            row_started = time.monotonic()
            try:
                with (out / (name + ".stdout.log")).open("wb") as stdout, (out / (name + ".stderr.log")).open("wb") as stderr:
                    child = subprocess.Popen(command, cwd=source_root, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                    row["pid"] = child.pid
                    # Observe completion without reaping the leader. Its retained
                    # PID keeps killpg authority bound to this owned session until
                    # all process-group signaling is finished in finally.
                    while True:
                        status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        if status is not None:
                            row["observed_return_code"] = status.si_status if status.si_code == os.CLD_EXITED else -status.si_status
                            break
                        minimum = min(minimum, shutil.disk_usage(workspace).free)
                        require(minimum > 4 * 1024**3, "Authoritative four GiB provisioning floor exhausted")
                        require(time.monotonic() < work_deadline, "Inclusive provisioning deadline exhausted")
                        time.sleep(.1)
                    require(row["observed_return_code"] == 0, "Provisioning command failed: " + name)
            finally:
                if child is not None:
                    if group_alive(child.pid):
                        os.killpg(child.pid, signal.SIGTERM)
                        until = min(deadline, time.monotonic() + 1)
                        while group_alive(child.pid) and time.monotonic() < until:
                            time.sleep(.05)
                        if group_alive(child.pid):
                            os.killpg(child.pid, signal.SIGKILL)
                    try:
                        row["return_code"] = child.wait(timeout=max(.001, min(5, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        row["direct_child_reaped"] = False
                    else:
                        row["direct_child_reaped"] = True
                    # No more signaling occurs after this reap; subsequent numeric
                    # group probes are read-only and may fail closed on reuse.
                    until = min(deadline, time.monotonic() + 1)
                    while group_alive(child.pid) and time.monotonic() < until:
                        time.sleep(.05)
                    row["process_group_disappeared"] = not group_alive(child.pid)
                row["elapsed_seconds"] = time.monotonic() - row_started
                (out / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            require(row.get("direct_child_reaped") and row.get("process_group_disappeared"), "Bootstrap cleanup unverified")
            minimum = min(minimum, shutil.disk_usage(workspace).free)
            require(minimum > 4 * 1024**3 and time.monotonic() < deadline, "Provisioning postflight exhausted")
            row["accepted"] = True

        requirements = workspace / "frozen-requirements.txt"
        environment = workspace / "py311"
        python = environment / "bin/python"
        execute(["uv", "export", "--frozen", "--no-dev", "--extra", "ml-ai", "--extra", "thrml",
                 "--extra", "cpomdp", "--no-emit-project", "--output-file", str(requirements)], "export-frozen-dependencies")
        execute(["uv", "venv", "--python", sys.executable, str(environment)], "create-isolated-environment")
        # Consume the complete pinned/hashed export without re-reading ambient
        # project constraints as additional unpinned requirements.
        execute(["uv", "pip", "install", "--no-config", "--python", str(python), "--require-hashes", "-r", str(requirements)], "install-frozen-dependencies")
        carried = CODE / binding["wheel_name"]
        require(carried.is_file() and not carried.is_symlink()
                and hashlib.sha256(carried.read_bytes()).hexdigest() == binding["wheel_sha256"],
                "Carried public-source ordinary archive differs from admitted W")
        (workspace / "wheels").mkdir(exist_ok=False)
        wheel = workspace / "wheels" / binding["wheel_name"]
        shutil.copyfile(carried, wheel)
        require(hashlib.sha256(wheel.read_bytes()).hexdigest() == binding["wheel_sha256"], "Carried wheel copy changed")
        execute(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], "install-ordinary-wheel")
        if args.rebuild_diagnostic:
            # Diagnostic only: native admission always uses the carried W archive.
            diagnostic = workspace / "rebuild-diagnostic"
            execute(["uv", "build", "--wheel", "--python", sys.executable,
                     "--build-constraints", str(CODE / "build-constraints.txt"),
                     "--out-dir", str(diagnostic)], "optional-build-diagnostic")
            rebuilt = list(diagnostic.glob("*.whl"))
            require(len(rebuilt) == 1, "Optional diagnostic produced ambiguous archives")
            rebuilt_sha = hashlib.sha256(rebuilt[0].read_bytes()).hexdigest()
            record["optional_build_diagnostic"] = {"archive_sha256": rebuilt_sha,
                "equals_carried_W": rebuilt_sha == binding["wheel_sha256"],
                "source_date_epoch": 1580601600, "used_for_native_install_or_admission": False}
        if args.backend == "rxinfer":
            # Same relative site-package location used by the installed executor.
            project = environment / "lib/python3.11/site-packages/gnn/execute/rxinfer"
            require(project.is_dir(), "Installed committed Julia project missing")
            execute(["julia", "--project=" + str(project), "-e",
                     'using Pkg; VERSION == v"1.12.7" || error("Julia version differs"); Pkg.instantiate();'], "instantiate-frozen-julia-project")
            # Dependency warmup includes the declared local Julia module. Its
            # PrecompileTools sample inference is prerequisite compilation only,
            # never acceptance of the four source-bound authored cases. The same
            # inclusive 1200-second prerequisite deadline governs this work.
            execute(["julia", "--project=" + str(project), "-e",
                     'using RxInfer, Distributions, JSON, StatsBase, Plots, GnnRxInferModels; VERSION == v"1.12.7" || error("Julia version differs"); Base.pkgversion(RxInfer) == v"5.5.0" || error("RxInfer version differs"); println(JSON.json(Dict("julia_version" => string(VERSION), "rxinfer_version" => string(Base.pkgversion(RxInfer)), "scope" => "dependency and local module warmup; PrecompileTools sample inference is not authored-case acceptance", "reported_authored_cases_executed" => 0)));'],
                    "warmup-frozen-julia-dependencies-and-local-module")
        for name, expected in binding["release_owner_inventory"].items():
            require(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == expected, "Provisioning mutated a release owner")
        require(time.monotonic() < deadline, "Provisioning final deadline exhausted")
        record["ordinary_wheel_sha256"] = binding["wheel_sha256"]
        record["frozen_requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        minimum = min(minimum, shutil.disk_usage(workspace).free)
        require(minimum > 4 * 1024**3 and time.monotonic() < deadline,
                "Final provisioning storage or inclusive deadline admission failed")
        record["accepted"] = True
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["minimum_free_storage_bytes"] = minimum
    record["storage_floor_bytes"] = 4 * 1024**3
    record["elapsed_seconds"] = time.monotonic() - started
    (out / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"accepted": record["accepted"], "receipt": str(out / "receipt.json"),
                      "error": record.get("error")}, sort_keys=True))
    return 0 if record["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
