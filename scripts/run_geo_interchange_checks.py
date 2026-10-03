#!/usr/bin/env python3
"""Run the paired GNN/GEO-INFER interchange checks and write receipts.

GNN-side mirror (roadmap item GNN-04) of GEO-INFER's paired-revision
mechanism, in the same spirit as the fep-lean bridge steps: every input is
explicit, receipts are written even on failure, and the exit code is
faithful. Given an explicitly selected pinned GEO-INFER checkout and an
explicitly selected GNN environment, the script

1. validates the committed pin ``.github/gnn-pair.json`` (known repository
   slug, 40-hex revision, no extra keys),
2. verifies the GEO checkout ``HEAD`` equals the pinned revision,
3. runs GEO's read-only validator
   ``GEO-INFER-TEST/validate_gnn_interchange.py``, which exports the tracked
   gridworld, a compiled H3 stay/diffuse model, a rectangular Gaussian and
   the explicit factored fixture inside the GNN environment and consumes +
   replays them inside the GEO environment, and
4. writes the receipt directory: ``gnn-revision.txt``,
   ``geo-infer-revision.txt``, ``pair.json``, ``interchange.json`` (the
   validator's full JSON trace with artifact digests) and
   ``digest-manifest.json`` (the digest cross-check).

No retries, no fallbacks, no silent downgrades: a failure of any check exits
non-zero and says which receipt to read. The validator prefixes its JSON
receipt with its own "GNN layout:" probe lines, so the receipt is extracted
with ``raw_decode`` from the first brace rather than ``json.loads`` on the
whole stdout.

Exit codes: 0 all checks green; 1 interchange or receipt failure; 2 usage or
pin/revision mismatch (nothing ran).

Usage:
    python scripts/run_geo_interchange_checks.py \
        --geo-root ../GEO-INFER --receipts-dir receipts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_GEO_REPOSITORY = "ActiveInferenceInstitute/GEO_INFER"
VALIDATOR_RELATIVE = "GEO-INFER-TEST/validate_gnn_interchange.py"

_DIGEST_KEYS = (
    "artifact_sha256",
    "h3_artifact_sha256",
    "gaussian_artifact_sha256",
    "factored_artifact_sha256",
    "source_sha256",
)


def _fail_usage(message: str) -> NoReturn:
    """Abort with the documented usage exit code 2 and a stderr diagnosis."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _python_path(value: Path) -> Path:
    """Anchor relative interpreter paths without resolving symlinks.

    Resolving an absolute venv interpreter path (e.g. ``.venv/bin/python``,
    a symlink to the uv-managed base interpreter) would strip the venv
    context and break module discovery in the launched interpreter; absolute
    paths are therefore passed verbatim, while relative paths are anchored to
    the working directory.
    """
    if value.is_absolute():
        return value
    return Path.cwd() / value


def _read_pin(pin_file: Path) -> dict[str, str]:
    """Load and validate the committed pair pin; exit code 2 on any defect."""
    try:
        pin = json.loads(pin_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _fail_usage(f"pin file unreadable ({pin_file}): {error}")
    if set(pin) != {"repository", "revision"}:
        _fail_usage(
            f"pin file must have exactly repository/revision keys, got {sorted(pin)}"
        )
    if pin["repository"] != EXPECTED_GEO_REPOSITORY:
        _fail_usage(
            f"pin repository {pin['repository']!r} is not {EXPECTED_GEO_REPOSITORY!r}"
        )
    revision = pin["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        _fail_usage("pin revision must be a 40-hex SHA string")
    if not all(character in "0123456789abcdef" for character in revision):
        _fail_usage("pin revision must be hexadecimal")
    return dict(repository=pin["repository"], revision=revision)


def _git_revision(root: Path) -> str:
    """Return the checkout HEAD, or exit code 2 when git cannot identify it."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _fail_usage(f"cannot resolve HEAD of {root}: {error}")
    return completed.stdout.strip()


def _write_receipt(receipts_dir: Path, name: str, payload: str) -> None:
    """Record one receipt file inside the receipt directory."""
    (receipts_dir / name).write_text(payload, encoding="utf-8")


def _validator_receipt(
    geo_root: Path,
    geo_python: Path,
    gnn_root: Path,
    gnn_python: Path,
    receipts_dir: Path,
    *,
    timeout: float = 1800,
) -> dict[str, Any]:
    """Run the pinned GEO validator; write its receipt; return parsed JSON.

    The validator is read-only on both checkouts: it writes artifacts only to
    its own temporary directory and prints one JSON receipt. Its stdout is
    passed through so a hosted run shows the probe lines, and its stderr is
    passed through so a hosted failure shows the traceback.
    """
    validator = geo_root / VALIDATOR_RELATIVE
    if not validator.is_file():
        _fail_usage(f"validator missing in pinned checkout: {validator}")
    from gnn.execute.subprocess_envelope import run_subprocess_envelope

    completed = run_subprocess_envelope(
        [
            str(geo_python),
            str(validator),
            "--gnn-repo",
            str(gnn_root),
            "--gnn-python",
            str(gnn_python),
        ],
        env=os.environ.copy(),
        timeout=timeout,
        sandbox=False,
    )
    if completed["stdout"]:
        sys.stdout.write(completed["stdout"])
    if completed["stderr"]:
        sys.stderr.write(completed["stderr"])
    if not completed["success"]:
        _write_receipt(
            receipts_dir,
            "validator-failed.json",
            json.dumps(
                {
                    "returncode": completed["return_code"],
                    "stderr_tail": completed["stderr"][-4000:],
                    "stdout_tail": completed["stdout"][-4000:],
                    "error_type": completed.get("error_type"),
                    "execution_error_type": completed.get("execution_error_type"),
                    "cleanup_verified": completed.get("cleanup_verified"),
                    "streams_drained": completed.get("streams_drained"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        raise SystemExit(
            f"interchange validator failed (exit {completed['return_code']}); "
            f"see {receipts_dir / 'validator-failed.json'}"
        )
    try:
        receipt, _ = json.JSONDecoder().raw_decode(
            completed["stdout"], completed["stdout"].index("{")
        )
    except ValueError as error:
        _write_receipt(
            receipts_dir,
            "validator-unparseable.txt",
            completed["stdout"][-8000:],
        )
        raise SystemExit(f"validator printed unparseable JSON receipt: {error}")
    _write_receipt(
        receipts_dir,
        "interchange.json",
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return receipt


def _digest_manifest(
    receipts_dir: Path,
    pin: dict[str, str],
    gnn_revision: str,
    geo_revision: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the validator's digests and write digest-manifest.json."""
    missing = [key for key in _DIGEST_KEYS if not receipt.get(key)]
    if missing:
        raise SystemExit(f"validator receipt lacks digests: {', '.join(missing)}")
    if receipt.get("deterministic_replay") is not True:
        raise SystemExit("validator receipt does not certify deterministic replay")
    manifest = {
        "pair": pin,
        "gnn_revision": gnn_revision,
        "geo_infer_revision": geo_revision,
        "contract": receipt["contract"],
        "digests": {key: receipt[key] for key in _DIGEST_KEYS},
        "h3_state_count": receipt["h3_state_count"],
        "steps": receipt["steps"],
        "deterministic_replay": True,
    }
    _write_receipt(
        receipts_dir,
        "digest-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Declare the explicit argument surface (no environmental discovery)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin-file",
        type=Path,
        default=REPO_ROOT / ".github/gnn-pair.json",
        help="Committed pair pin (default: <repo>/.github/gnn-pair.json).",
    )
    parser.add_argument(
        "--gnn-root",
        type=Path,
        default=REPO_ROOT,
        help="This GNN checkout (default: the repository containing this script).",
    )
    parser.add_argument(
        "--geo-root",
        type=Path,
        required=True,
        help="Pinned GEO-INFER checkout (must be exactly at the pinned revision).",
    )
    parser.add_argument(
        "--gnn-python",
        type=Path,
        default=Path(sys.executable),
        help="GNN environment interpreter (default: the running interpreter).",
    )
    parser.add_argument(
        "--geo-python",
        type=Path,
        default=None,
        help="GEO environment interpreter (default: <geo-root>/.venv/bin/python).",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=None,
        help="Receipt destination (default: a fresh temporary directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the pin, run the round trip, write receipts, return the code."""
    args = parse_args(argv)
    pin = _read_pin(args.pin_file)
    gnn_root = args.gnn_root.resolve()
    geo_root = args.geo_root.resolve()
    # Pin/checkout identity first: it is the most specific diagnosis and it
    # must hold before any interpreter or validator is even looked for.
    gnn_revision = _git_revision(gnn_root)
    geo_revision = _git_revision(geo_root)
    if geo_revision != pin["revision"]:
        _fail_usage(
            f"GEO checkout {geo_root} is at {geo_revision}, pin requires "
            f"{pin['revision']}: re-pin or re-checkout before running"
        )
    gnn_python = _python_path(args.gnn_python)
    if not gnn_python.is_file():
        _fail_usage(f"GNN interpreter is not an existing file: {gnn_python}")
    geo_python = (
        _python_path(args.geo_python)
        if args.geo_python is not None
        else _python_path(geo_root / ".venv/bin/python")
    )
    if not geo_python.is_file():
        _fail_usage(
            f"GEO interpreter is not an existing file: {geo_python} "
            "(sync the pinned checkout first, e.g. uv sync --project "
            f"{geo_root} --locked --all-packages --all-extras)"
        )
    receipts_dir = (
        args.receipts_dir.resolve()
        if args.receipts_dir is not None
        else Path(tempfile.mkdtemp(prefix="geo-interchange-receipts-"))
    )
    receipts_dir.mkdir(parents=True, exist_ok=True)
    _write_receipt(receipts_dir, "gnn-revision.txt", gnn_revision + "\n")
    _write_receipt(receipts_dir, "geo-infer-revision.txt", geo_revision + "\n")
    _write_receipt(
        receipts_dir, "pair.json", json.dumps(pin, indent=2, sort_keys=True) + "\n"
    )
    receipt = _validator_receipt(
        geo_root, geo_python, gnn_root, gnn_python, receipts_dir
    )
    manifest = _digest_manifest(receipts_dir, pin, gnn_revision, geo_revision, receipt)
    digests = manifest["digests"]
    print("geo interchange checks: green")
    print(f"receipts_dir:             {receipts_dir}")
    print(f"pin:                      {pin['repository']}@{pin['revision']}")
    print(f"contract:                 {manifest['contract']}")
    print(f"artifact_sha256:          {digests['artifact_sha256']}")
    print(f"h3_artifact_sha256:       {digests['h3_artifact_sha256']}")
    print(f"gaussian_artifact_sha256: {digests['gaussian_artifact_sha256']}")
    print(f"factored_artifact_sha256: {digests['factored_artifact_sha256']}")
    print("deterministic_replay:     True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
