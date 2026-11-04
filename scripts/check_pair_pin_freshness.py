#!/usr/bin/env python3
"""Pair-pin freshness gate: pins and bound owner content must stay current.

GNN-side freshness gate for the two committed pair pins
(``.github/fep-lean-pair.json`` pinning fep_lean and ``.github/gnn-pair.json``
pinning GEO-INFER), per the BC-12 backlog item and scope-comp-consumers
section 5.3. The paired-revision workflows
(``fep-lean-paired-revision.yml``, ``geo-infer-interchange.yml``) check out
the companions at their *pinned* revisions and prove that the bridge and
interchange surfaces still agree with this repository; they cannot notice
that a companion default branch has moved ahead of its pin. This gate
closes exactly that gap.

The FEP bridge pair is content-checked, not just ancestry-checked. The FEP
source pin (``specs/gnn-bridge-w2-source-custody/source-pin.json`` in the
FEP companion checkout) records SHA-256 digests for GNN's declared
notation-contract owners. The gate recomputes those digests at the GNN
checkout and exits 2 when any owner differs. Ancestry remains a cheap
pre-check for every pair; it cannot produce a green verdict over stale
owner content.

Contract:

1. every pair file must carry exactly ``repository``/``revision`` keys,
   with a ``owner/repo`` slug and a 40-hex revision,
2. the companion checkout must resolve a ``HEAD`` (the caller provides the
   checkout; this script performs no network access and no fetching),
3. ``git merge-base --is-ancestor <pin> <head>`` is the ancestry pre-check:
   the pin may trail the companion default branch by any number of commits
   (the measured lag is reported, not enforced), but it must never leave
   the companion default branch's history,
4. for the FEP bridge pair, every ``gnn.owners`` digest in the FEP source
   pin must match the GNN checkout byte-for-byte; a mismatch names the
   owner, the recorded digest and the actual digest,
5. receipts are written even on failure, and the exit code is faithful:
   no retries, no fallbacks, no silent downgrades.

Exit codes: 0 every checked pin is fresh and every bound owner matches;
1 operational failure (companion checkout missing or not a git work tree,
receipts directory not writable); 2 at least one stale pin or owner-content
drift -- "re-pin required", naming the stale pair file, the pinned
revision, the companion tip, and every drifting owner -- or a pair-file or
source-pin usage defect.

Usage:
    python scripts/check_pair_pin_freshness.py \
        --check gnn/.github/fep-lean-pair.json fep-lean \
        --check gnn/.github/gnn-pair.json geo-infer \
        --source-pin fep-lean/specs/gnn-bridge-w2-source-custody/source-pin.json \
        --gnn-root gnn \
        --receipts-dir receipts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

EXIT_FRESH = 0
EXIT_OPERATIONAL = 1
EXIT_STALE = 2

_REPOSITORY_SLUG = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _git(companion_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run git inside the companion checkout without any host-side fallback."""
    return subprocess.run(
        ["git", "-C", str(companion_dir), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _load_pair(pair_file: Path) -> dict[str, str]:
    """Load and validate one committed pair pin.

    Raises ``ValueError`` on any defect so the caller records the failing
    pair file in the receipts before the process exits with code 2.
    """
    try:
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"pair file unreadable ({pair_file}): {error}") from error
    if not isinstance(pair, dict) or set(pair) != {"repository", "revision"}:
        raise ValueError(
            f"pair file must have exactly repository/revision keys, "
            f"got {sorted(pair) if isinstance(pair, dict) else pair!r} ({pair_file})"
        )
    if not _REPOSITORY_SLUG.fullmatch(pair["repository"]):
        raise ValueError(f"pair file has malformed repository slug ({pair_file})")
    if not _REVISION.fullmatch(pair["revision"]):
        raise ValueError(f"pair file revision must be 40 hex characters ({pair_file})")
    return dict(repository=pair["repository"], revision=pair["revision"])


def _sha256(path: Path) -> str:
    """Return the SHA-256 of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_owner_content(
    gnn_root: Path, source_pin_path: Path
) -> tuple[dict[str, Any], str | None]:
    """Compare the FEP source pin's GNN owners with the GNN checkout bytes."""
    try:
        payload = json.loads(source_pin_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {}, f"source pin unreadable ({source_pin_path}): {error}"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {}, f"source pin has an unsupported schema ({source_pin_path})"
    gnn = payload.get("gnn")
    if not isinstance(gnn, dict):
        return {}, f"source pin has no gnn binding ({source_pin_path})"
    owners = gnn.get("owners")
    if not isinstance(owners, dict) or not owners:
        return {}, f"source pin has no gnn.owners mapping ({source_pin_path})"

    mismatches: list[dict[str, str]] = []
    matched = 0
    for owner, recorded in sorted(owners.items()):
        if (
            not isinstance(owner, str)
            or not isinstance(recorded, str)
            or _DIGEST.fullmatch(recorded) is None
        ):
            return {}, f"source pin has a malformed owner entry ({owner!r})"
        try:
            actual = _sha256(gnn_root / owner)
        except OSError as error:
            mismatches.append(
                {
                    "owner": owner,
                    "recorded": recorded,
                    "actual": f"unreadable ({error})",
                }
            )
            continue
        if actual == recorded:
            matched += 1
        else:
            mismatches.append({"owner": owner, "recorded": recorded, "actual": actual})
    return (
        {
            "source_pin": str(source_pin_path),
            "checked": len(owners),
            "matched": matched,
            "mismatches": mismatches,
        },
        None,
    )


def _check_pair(
    pair_file: Path,
    companion_dir: Path,
    *,
    source_pin_path: Path | None = None,
    gnn_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one pin against its companion checkout HEAD and bound owners."""
    entry: dict[str, Any] = {
        "pair_file": str(pair_file),
        "companion_dir": str(companion_dir),
        "repository": None,
        "revision": None,
        "companion_head": None,
        "behind": None,
        "status": "operational",
        "note": None,
        "content": None,
    }
    try:
        pair = _load_pair(pair_file)
    except ValueError as error:
        entry["status"] = "usage"
        entry["note"] = str(error)
        return entry
    entry["repository"] = pair["repository"]
    entry["revision"] = pair["revision"]

    head = _git(companion_dir, "rev-parse", "HEAD")
    if head.returncode != 0:
        entry["note"] = (
            f"companion checkout unusable ({companion_dir}): "
            f"{(head.stderr or head.stdout).strip()}"
        )
        return entry
    entry["companion_head"] = head.stdout.strip()

    ancestry = _git(
        companion_dir, "merge-base", "--is-ancestor", pair["revision"], "HEAD"
    )
    if ancestry.returncode == 0:
        entry["status"] = "fresh"
    elif ancestry.returncode == 1:
        entry["status"] = "stale"
        entry["note"] = "pinned revision is not an ancestor of companion HEAD"
    else:
        entry["status"] = "stale"
        entry["note"] = (
            f"ancestry undecidable against companion HEAD "
            f"({(ancestry.stderr or ancestry.stdout).strip()})"
        )

    if entry["status"] == "fresh" or ancestry.returncode in (0, 1):
        count = _git(companion_dir, "rev-list", "--count", f"{pair['revision']}..HEAD")
        if count.returncode == 0:
            entry["behind"] = int(count.stdout.strip())

    if (
        pair["repository"] == "ActiveInferenceInstitute/fep_formal"
        and source_pin_path is not None
        and gnn_root is not None
    ):
        content, error = _check_owner_content(gnn_root, source_pin_path)
        if error is not None:
            entry["status"] = "usage"
            entry["note"] = error
        else:
            entry["content"] = content
            if content["mismatches"]:
                entry["status"] = "stale"
                entry["note"] = (
                    f"owner content drift: {len(content['mismatches'])} of "
                    f"{content['checked']} GNN owners differ from the FEP "
                    f"source pin"
                )
            elif entry["status"] == "stale":
                entry["note"] = (
                    "pinned revision is not an ancestor of companion HEAD; "
                    "bound owner content matches"
                )
    return entry


def _write_receipts(receipts_dir: Path, receipt: dict[str, Any]) -> str | None:
    """Persist the freshness receipt; report the error path on failure."""
    try:
        receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipts_dir / "pair-pin-freshness.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return None
    except OSError as error:
        return str(error)


def _print_entry(entry: dict[str, Any]) -> None:
    """Emit one human-readable verdict line for a checked pair."""
    label = "STALE" if entry["status"] == "stale" else entry["status"].upper()
    behind = entry["behind"]
    print(
        f"{label:12s} {entry['pair_file']} pin={entry['revision']} "
        f"tip={entry['companion_head']} "
        f"behind={'n/a' if behind is None else behind}"
    )
    if entry["note"]:
        print(f"             note: {entry['note']}")
    content = entry.get("content")
    if content:
        print(
            f"             content: {content['matched']}/{content['checked']} "
            f"bound owners match"
        )
        for mismatch in content["mismatches"]:
            print(
                f"             owner drift: {mismatch['owner']} "
                f"recorded={mismatch['recorded']} actual={mismatch['actual']}"
            )


def _parse_args() -> argparse.Namespace:
    """Parse the gate's arguments: repeatable --check pairs plus receipts."""
    parser = argparse.ArgumentParser(
        description=(
            "Assert committed pair pins stay ancestor-or-equal of their "
            "companion default-branch HEAD (BC-12 freshness gate)."
        )
    )
    parser.add_argument(
        "--check",
        dest="checks",
        nargs=2,
        action="append",
        metavar=("PAIR_FILE", "COMPANION_DIR"),
        required=True,
        help=(
            "pair file to verify and the companion checkout directory "
            "holding its default-branch HEAD; repeat for every pair file"
        ),
    )
    parser.add_argument(
        "--source-pin",
        type=Path,
        default=None,
        help=(
            "FEP source-pin JSON whose gnn.owners digests are compared with "
            "the GNN checkout; applies to the fep_formal pair"
        ),
    )
    parser.add_argument(
        "--gnn-root",
        type=Path,
        default=None,
        help=(
            "GNN checkout root used for owner digest comparison "
            "(default: current directory)"
        ),
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=None,
        help="directory receiving pair-pin-freshness.json (written on every outcome)",
    )
    return parser.parse_args()


def main() -> int:
    """Check every requested pair and exit 0/1/2 per the documented contract."""
    options = _parse_args()
    checks = [
        (Path(pair_file), Path(companion)) for pair_file, companion in options.checks
    ]

    entries = [
        _check_pair(
            pair_file,
            companion,
            source_pin_path=options.source_pin,
            gnn_root=options.gnn_root or Path.cwd(),
        )
        for pair_file, companion in checks
    ]

    stale = [entry for entry in entries if entry["status"] == "stale"]
    usage = [entry for entry in entries if entry["status"] == "usage"]
    operational = [entry for entry in entries if entry["status"] == "operational"]

    exit_two = bool(stale or usage)
    receipt = {
        "result": "stale" if exit_two else ("operational" if operational else "fresh"),
        "stale_count": len(stale),
        "checked": len(entries),
        "checks": entries,
    }
    receipt_error = None
    if options.receipts_dir is not None:
        receipt_error = _write_receipts(options.receipts_dir, receipt)

    for entry in entries:
        _print_entry(entry)

    if stale:
        for entry in stale:
            content = entry.get("content")
            if content and content["mismatches"]:
                print(
                    f"RE-FREEZE REQUIRED: {entry['pair_file']} owner content "
                    f"disagrees with the FEP source pin "
                    f"({len(content['mismatches'])} of {content['checked']} "
                    f"GNN owners)",
                    file=sys.stderr,
                )
                continue
            behind = entry["behind"]
            lag = (
                f"{behind} commits ahead of the pin"
                if behind is not None
                else "commit count unavailable"
            )
            print(
                f"RE-PIN REQUIRED: {entry['pair_file']} pins {entry['revision']} "
                f"but {entry['repository']} default branch is at "
                f"{entry['companion_head']} ({lag})",
                file=sys.stderr,
            )
    if receipt_error is not None:
        print(f"receipt write failed: {receipt_error}", file=sys.stderr)

    if stale or usage:
        print(
            f"exit 2: re-pin required for {len(stale) + len(usage)} of "
            f"{len(entries)} checked pair file(s)"
        )
        return EXIT_STALE
    if operational:
        print(
            f"exit 1: operational failure on {len(operational)} of "
            f"{len(entries)} checked pair file(s)"
        )
        return EXIT_OPERATIONAL
    if receipt_error is not None:
        return EXIT_OPERATIONAL
    print(
        f"exit 0: all {len(entries)} checked pair pin(s) are fresh and every "
        f"bound owner matches"
    )
    return EXIT_FRESH


if __name__ == "__main__":
    raise SystemExit(main())
