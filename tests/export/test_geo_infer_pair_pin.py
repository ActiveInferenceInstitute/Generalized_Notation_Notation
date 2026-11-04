"""Paired GEO-INFER revision pin and runner contract (GNN-side, GNN-04).

The committed pin ``.github/gnn-pair.json`` is consumed by the hosted
workflow (``.github/workflows/geo-infer-interchange.yml``) and by
``scripts/run_geo_interchange_checks.py``. A malformed or silently retargeted
pin must fail fast here in the export suite instead of surfacing as a
confusing hosted-CI failure, and the runner must refuse a checkout that is
not at the pinned revision before running any round trip.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = ROOT / ".github/gnn-pair.json"
RUNNER = ROOT / "scripts/run_geo_interchange_checks.py"
EXPECTED_REPOSITORY = "ActiveInferenceInstitute/GEO_INFER"


def _read_pin() -> dict[str, str]:
    return json.loads(PIN_FILE.read_text(encoding="utf-8"))


def test_pin_is_well_formed_and_targets_the_known_geo_repository() -> None:
    pin = _read_pin()
    assert set(pin) == {"repository", "revision"}
    assert pin["repository"] == EXPECTED_REPOSITORY
    revision = pin["revision"]
    assert isinstance(revision, str)
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def _init_repo_with_commit(directory: Path, message: str) -> str:
    directory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
    subprocess.run(
        ["git", "-C", str(directory), "config", "user.email", "pin-test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(directory), "config", "user.name", "pin test"], check=True
    )
    (directory / "README.md").write_text("pin test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(directory), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(directory), "commit", "-q", "-m", message], check=True
    )
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_runner_refuses_checkout_that_is_not_at_the_pinned_revision(
    tmp_path: Path,
) -> None:
    """Exit code 2, no receipts, before any validator invocation."""
    mismatched = _init_repo_with_commit(tmp_path / "geo", "not the pinned revision")
    pin = _read_pin()
    assert mismatched != pin["revision"], "test fixture must not equal the pin"
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(pin), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--pin-file",
            str(pin_file),
            "--geo-root",
            str(tmp_path / "geo"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
    assert pin["revision"] in completed.stderr
    assert mismatched in completed.stderr
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    "broken",
    [
        {"repository": "example-org/geo-infer", "revision": "a" * 40},
        {"repository": EXPECTED_REPOSITORY, "revision": "short"},
        {"repository": EXPECTED_REPOSITORY, "revision": "z" * 40},
        {"repository": EXPECTED_REPOSITORY},
    ],
)
def test_runner_rejects_malformed_pins(tmp_path: Path, broken: dict[str, str]) -> None:
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(broken), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--pin-file",
            str(pin_file),
            "--geo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2


def _satisfies(version: str, requirement: str) -> bool:
    """Whether ``version`` satisfies the comma-joined specifier ``uv.toml`` declares.

    The repository declares a floor with a minor ceiling (``>=0.13.6,<0.14``) so a
    host package-manager update cannot make the checkout unusable; each hosted pin
    stays exact and is checked against that requirement here.
    """
    parts = tuple(int(part) for part in version.split("."))
    for clause in requirement.split(","):
        clause = clause.strip()
        operator = next(
            (
                candidate
                for candidate in (">=", "<=", "==", "!=", ">", "<")
                if clause.startswith(candidate)
            ),
            None,
        )
        assert operator is not None, f"unsupported uv requirement: {clause!r}"
        bound = tuple(int(part) for part in clause[len(operator) :].split("."))
        width = max(len(parts), len(bound))
        left = parts + (0,) * (width - len(parts))
        right = bound + (0,) * (width - len(bound))
        if (
            (operator == ">=" and left < right)
            or (operator == "<=" and left > right)
            or (operator == "==" and left != right)
            or (operator == "!=" and left == right)
            or (operator == ">" and left <= right)
            or (operator == "<" and left >= right)
        ):
            return False
    return True


def test_workflow_installs_each_repository_with_its_exact_uv_toolchain() -> None:
    """Track the active tool through both installs and the later GNN check."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/geo-infer-interchange.yml").read_text(
            encoding="utf-8"
        )
    )
    job = workflow["jobs"]["geo-infer-interchange"]
    assert job["strategy"]["matrix"]["python"] == ["3.11", "3.12"]
    uv_requirement = tomllib.loads((ROOT / "uv.toml").read_text())[
        "required-version"
    ]
    # uv.toml declares a floor with a minor ceiling (>=0.13.6,<0.14); the hosted
    # pin stays exact and must satisfy it.
    expected_versions = {"geo-infer": "0.13.1", "gnn": "0.13.6"}
    active_uv = None
    installs = {}
    setup_versions = []
    qualification_checked = False
    for step in job["steps"]:
        action = step.get("uses", "")
        if action.startswith("astral-sh/setup-uv@"):
            assert re.fullmatch(r"astral-sh/setup-uv@[0-9a-f]{40}", action)
            active_uv = step["with"]["version"]
            setup_versions.append(active_uv)
        for line in step.get("run", "").replace("\\\n", " ").splitlines():
            command = shlex.split(line)
            if command[:2] == ["uv", "sync"]:
                project = command[command.index("--project") + 1]
                assert project not in installs
                assert active_uv == expected_versions[project]
                installs[project] = command
            if command[:2] == ["uv", "run"]:
                assert step["working-directory"] == "gnn"
                assert active_uv == expected_versions["gnn"]
                assert _satisfies(active_uv, uv_requirement)
                assert command[2:5] == ["--frozen", "--no-sync", "--python"]
                qualification_checked = True
    assert setup_versions == ["0.13.1", expected_versions["gnn"]]
    assert list(installs) == ["geo-infer", "gnn"]
    assert installs["gnn"] == [
        "uv",
        "sync",
        "--project",
        "gnn",
        "--locked",
        "--python",
        "3.11",
        "--extra",
        "dev",
        "--extra",
        "geo-infer",
    ]
    assert installs["geo-infer"] == [
        "uv",
        "sync",
        "--project",
        "geo-infer",
        "--locked",
        "--python",
        "$GEO_PYTHON",
        "--all-packages",
        "--all-extras",
        "--no-install-package",
        "cupy",
        "--no-install-package",
        "mayavi",
        "--no-install-package",
        "vaex",
        "--no-install-package",
        "vaex-core",
    ]
    assert qualification_checked
