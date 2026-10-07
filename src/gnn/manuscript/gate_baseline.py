"""Read-only, content-bound comparison of manuscript findings against a PR base.

Both sides use the HEAD audit implementation. This compares evidence, not the
historical behavior of an older gate. Main and scheduled audits supply no base
and remain strict. Missing git history never permits an inherited-drift waiver.
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tarfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

EVIDENCE_ROOTS = (
    "manuscript",
    "output/data",
    "output/manuscript",
    "output/pdf",
    "output/figures",
    "input",
    "src",
    "STEP_INDEX.md",
)


@dataclass(frozen=True)
class GateComparison:
    """New failures and byte-identical inherited diagnostics, with exact base."""

    failures: tuple[str, ...]
    inherited: tuple[str, ...] = ()
    base_sha: str | None = None


def resolve_base(project_root: Path, base_ref: str) -> str:
    """Resolve the merge base against HEAD; reject unresolved or unrelated refs."""
    resolved = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "rev-parse",
            "--verify",
            f"{base_ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if resolved.returncode:
        raise ValueError(
            f"cannot resolve PR base {base_ref!r}; fetch the target history"
        )
    result = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "HEAD", resolved.stdout.strip()],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    sha = result.stdout.strip()
    if result.returncode or re.fullmatch(r"[0-9a-f]{40,64}", sha) is None:
        raise ValueError(f"cannot resolve merge base for {base_ref!r}")
    return sha


@contextmanager
def base_evidence(project_root: Path, base_sha: str) -> Iterator[Path]:
    """Materialize only immutable gate evidence, without a checkout or Git writes."""
    names = (
        subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                base_sha,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    present = tuple(
        prefix
        for prefix in EVIDENCE_ROOTS
        if any(name == prefix or name.startswith(prefix + "/") for name in names)
    )
    archive = (
        subprocess.run(
            ["git", "-C", str(project_root), "archive", base_sha, "--", *present],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
        if present
        else b""
    )
    with TemporaryDirectory(prefix="gnn-manuscript-base-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive or bytes(10240))) as files:
            for member in files:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsafe gate evidence entry: {member.name}")
                path = root / relative
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    source = files.extractfile(member)
                    assert source is not None
                    path.write_bytes(source.read())
        yield root


def finding_fingerprint(project_root: Path, issue: str) -> str:
    """Bind a diagnostic to exact supporting bytes, including missing evidence.

    The message carries computed values/digests; referenced source/output files
    supply byte identity. Hydration findings additionally bind their source and
    token map. This cannot grandfather altered evidence with the same wording.
    """
    paths = set(re.findall(r"(?:output|manuscript|input|src)/[A-Za-z0-9_./-]+", issue))
    for name in re.findall(r"\b[A-Za-z0-9_-]+\.(?:md|yaml|bib)\b", issue):
        if (project_root / name).is_file():
            paths.add(name)
        if (project_root / "manuscript" / name).is_file():
            paths.add(f"manuscript/{name}")
    if issue.startswith("fig:"):
        paths.add("output/figures/figure_registry.json")
        for name in re.findall(r"\b[A-Za-z0-9_-]+\.png\b", issue):
            paths.add(f"output/figures/{name}")
    for path in tuple(paths):
        if path.startswith("output/manuscript/"):
            paths.add("manuscript/" + path.rsplit("/", 1)[-1])
            paths.add("output/data/manuscript_variables.json")
        if path.startswith("output/pdf/") and "commit" in issue:
            paths.add("output/data/manuscript_variables.json")
    digest = hashlib.sha256(issue.encode("utf-8"))
    for relative in sorted(paths):
        path = project_root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def compare_findings(
    project_root: Path,
    findings: list[str],
    base_ref: str | None,
    audit_base: Callable[[Path, str], list[str]],
) -> GateComparison:
    """Tolerate only findings with the same message and evidence fingerprint."""
    if not base_ref:
        return GateComparison(tuple(findings))
    try:
        sha = resolve_base(project_root, base_ref)
        with base_evidence(project_root, sha) as root:
            inherited = {
                finding_fingerprint(root, issue) for issue in audit_base(root, sha)
            }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return GateComparison((*findings, f"PR base comparison failed: {exc}"))
    failures: list[str] = []
    unchanged: list[str] = []
    for issue in findings:
        (
            unchanged
            if finding_fingerprint(project_root, issue) in inherited
            else failures
        ).append(issue)
    return GateComparison(tuple(failures), tuple(unchanged), sha)
