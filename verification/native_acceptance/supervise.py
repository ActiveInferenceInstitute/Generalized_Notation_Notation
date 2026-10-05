"""Supervise controls and complete reported-native scans in killable processes.

The 3,600-second scientific work budget includes admission, controls, readiness,
all four cases, validation and their retrieval. Two seconds are reserved outside
that work budget for the public envelope's bounded cleanup and receipt retention.
Provisioning is a separately identified prerequisite and is never native proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import types
import uuid

CODE = Path(__file__).resolve().parent
HASHES = {
    "runner.py": "5c8e09d68e0e37db2249ce3fb09b1d2419155013dca661ec2d052c7f951dce0c",
    "common.py": "74d061a42bad33a5c53d3be42f797a888acb19003bb2e5e75cb1f0bcbd85935e",
    "artifacts.py": "4b75739ce163eecf5cbbdd9439db0848bf811d20c84829264ad2f26bc985b45c",
    "controls.py": "6a815b2f718cf99442e7c43ffa57eb137d80681aad9c3be357b71018c145d5b8",
    "source-binding.json": "bed9d6e8e4725d6838a1fb6a9cd8b27601c7f3e20a8d9374bc3089565d4f31d7",
    "generalized_notation_notation-4.0.0-py3-none-any.whl": "36e71bebe855711b12751d5d58feaeac5d45f53c12e21eb5481e80405c41b907",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def regular_bytes(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "Nonregular or symlink verification source refused")
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpyro", "rxinfer"), required=True)
    parser.add_argument("--expected-root-head", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_root_head), "Invalid expected commit")
    require(sys.flags.isolated == 1, "Ordinary installed isolated Python required")
    workspace = Path(os.environ["GNN_NATIVE_VERIFY_WORKSPACE"]).absolute()
    source_root = Path(os.environ["GNN_NATIVE_VERIFY_SOURCE_ROOT"]).absolute()
    require(workspace != source_root and not workspace.is_relative_to(source_root),
            "Evidence and installed environment must be outside the checkout")
    (workspace / "evidence").mkdir(parents=True, exist_ok=True)
    out = workspace / "evidence" / ("supervisor-" + args.backend + "-" + uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    work_deadline = started + 3600
    cleanup_deadline = work_deadline + 2
    record = {"accepted": False, "backend": args.backend,
              "expected_root_head": args.expected_root_head,
              "work_budget_seconds": 3600, "cleanup_reservation_seconds": 2,
              "inclusive_budget_seconds": 3602, "children": [],
              "scope": "Eight reported categorical cases; each group contains four unchanged authored models."}
    token = None
    try:
        require(shutil.disk_usage(workspace).free >= 6 * 1024**3,
                "RESOURCE_HOLD: less than six GiB before group admission")
        sources = {}
        for name, expected in HASHES.items():
            raw = regular_bytes(CODE / name)
            require(hashlib.sha256(raw).hexdigest() == expected, "Reviewed helper changed: " + name)
            sources[name] = raw
        common = types.ModuleType("native_acceptance_common")
        common.__file__ = str(CODE / "common.py")
        exec(compile(sources["common.py"], common.__file__, "exec", dont_inherit=True), common.__dict__)
        token = common.StorageToken()
        from gnn.execute.subprocess_envelope import run_subprocess_envelope

        def launch(command, name, work_limit):
            require(not token.cancelled and time.monotonic() < work_deadline,
                    "Group budget or resource exhausted before dispatch")
            allowance = min(work_limit, work_deadline - time.monotonic())
            envelope = run_subprocess_envelope(
                command, cwd=workspace, env={"GNN_SANDBOX": "off", "GKSwstype": "100",
                    "JULIA_PKG_OFFLINE": "true", "JULIA_PKG_PRECOMPILE_AUTO": "0"},
                timeout=allowance, deadline_monotonic=cleanup_deadline,
                cancel_token=token, sandbox=False)
            (out / (name + ".stdout.log")).write_text(envelope.pop("stdout"))
            (out / (name + ".stderr.log")).write_text(envelope.pop("stderr"))
            record["children"].append({"name": name, "command": command, "envelope": envelope})
            (out / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            require(envelope.get("success") is True and envelope.get("return_code") == 0
                    and envelope.get("cleanup_verified") is True
                    and envelope.get("streams_drained") is True and not token.cancelled,
                    "Required supervised process failed, exhausted budget or lacks cleanup: " + name)

        launch([sys.executable, "-I", str(CODE / "controls.py"), "--resource-go",
                "root-reviewed-lightweight-controls"], "controls", 60)
        control_paths = sorted((workspace / "evidence").glob("reported-native-controls-*/receipt.json"))
        require(len(control_paths) == 1, "Missing or duplicate current controls receipt")
        controls = json.loads(regular_bytes(control_paths[0]))
        require(controls.get("accepted") is True and controls["controls"]
                and all(row["passed"] is True for row in controls["controls"])
                and controls.get("native_inference_executed") is False,
                "Actual controls failed or are mislabeled as native acceptance")
        record["controls_receipt_sha256"] = hashlib.sha256(regular_bytes(control_paths[0])).hexdigest()
        remaining = work_deadline - time.monotonic()
        require(remaining > 60, "Insufficient remaining native budget")
        launch([sys.executable, "-I", str(CODE / "runner.py"), "--backend", args.backend,
                "--split", "311", "--expected-root-head", args.expected_root_head,
                "--aggregate-budget", str(remaining), "--case-budget", "900",
                "--worker-sha256", HASHES["artifacts.py"]], "native-group", remaining)
        paths = sorted((workspace / "evidence").glob("reported-native-" + args.backend + "-311-*/receipt.json"))
        require(len(paths) == 1, "Missing or duplicate current native group receipt")
        native = json.loads(regular_bytes(paths[0]))
        require(native.get("accepted") is True and native["expected_root_head"] == args.expected_root_head
                and len(native["cases"]) == 4 and all(case["accepted"] is True for case in native["cases"]),
                "Required native case remains unfinished or unverified")
        record["native_receipt_sha256"] = hashlib.sha256(regular_bytes(paths[0])).hexdigest()
        require(all(hashlib.sha256(regular_bytes(CODE / name)).hexdigest() == expected
                    for name, expected in HASHES.items()), "Helper bytes changed during verification")
        require(time.monotonic() < work_deadline and not token.cancelled, "Group exhausted during final verification")
        record["accepted"] = True
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    if token is not None:
        record["storage"] = token.snapshot()
        if token.cancelled:
            record["accepted"] = False
    record["elapsed_seconds"] = time.monotonic() - started
    if record["elapsed_seconds"] >= 3602:
        record["accepted"] = False
    record["helper_hashes"] = HASHES
    (out / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"accepted": record["accepted"], "backend": args.backend,
                      "receipt": str(out / "receipt.json"), "error": record.get("error")}, sort_keys=True))
    return 0 if record["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
