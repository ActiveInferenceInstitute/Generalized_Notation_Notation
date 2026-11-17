# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: uv run python docs/other/geo_infer/qualify_export.py CONDITION --step-seconds 21600
"""Reproduce one retained Cascadia export using this checkout's real producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias

from gnn.export.geo_infer import MAX_SOURCE_BYTES, build_geo_infer_artifact

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
STEM: Final = "smith_river_homing"


class ExportMismatch(ValueError):
    """A retained export differs from the current producer's output."""


@dataclass(frozen=True, slots=True)
class Qualification:
    """Producer reproduction evidence, separate from simulation and native proof."""

    source_sha256: str
    state_ids_sha256: str
    artifact_sha256: str
    producer_revision: str
    qualifier_sha256: str
    producer_source_sha256: tuple[tuple[str, str], ...]
    schema_version: str = "gnn/cascadia-export-qualification/2"
    export_reproduced: bool = True
    native_claim_ready: bool = False


def unique_json(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    """Reject duplicate keys rather than accepting the last occurrence."""
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            message = f"duplicate JSON key: {key}"
            raise ExportMismatch(message)
        result[key] = value
    return result


def read_bounded(path: Path) -> bytes:
    """Read the producer's input budget without allocating an unbounded file."""
    with path.open("rb") as stream:
        content = stream.read(MAX_SOURCE_BYTES + 1)
    if len(content) > MAX_SOURCE_BYTES:
        message = f"input exceeds four MiB: {path.name}"
        raise ExportMismatch(message)
    return content


def producer_fingerprint(root: Path) -> tuple[tuple[str, str], ...]:
    """Record current checkout bytes, including additions to the Python roster."""
    fixed = (
        "pyproject.toml",
        "uv.lock",
        "docs/gnn/gnn_syntax.md",
        "docs/other/fep_lean/bridge-contract.md",
    )
    owners = set(fixed) | {
        path.relative_to(root).as_posix() for path in root.glob("src/gnn/**/*.py")
    }
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest())
        for name in sorted(owners)
    )


def qualify_export(condition: Path, step_seconds: float) -> Qualification:
    """Re-export source plus ordered IDs and compare the complete artifact."""
    root = Path(__file__).resolve().parents[3]
    before = producer_fingerprint(root)
    source = read_bounded(condition / f"{STEM}.gnn.md")
    ids = read_bounded(condition / f"{STEM}.state_ids.json")
    artifact = read_bounded(condition / f"{STEM}.geo-infer.json")
    labels = json.loads(ids, object_pairs_hook=unique_json)
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        message = "state IDs must be a JSON array of strings"
        raise ExportMismatch(message)
    produced = build_geo_infer_artifact(
        source.decode("utf-8"),
        step_seconds=step_seconds,
        state_ids=labels,
        space_kind="h3",
    )
    retained = json.loads(artifact, object_pairs_hook=unique_json)
    if json.dumps(produced, sort_keys=True, allow_nan=False) != json.dumps(
        retained, sort_keys=True, allow_nan=False
    ):
        message = "retained artifact differs from the current GNN producer"
        raise ExportMismatch(message)
    if producer_fingerprint(root) != before:
        message = "GNN source bytes changed during export qualification"
        raise ExportMismatch(message)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=60
    ).strip()
    return Qualification(
        source_sha256=hashlib.sha256(source).hexdigest(),
        state_ids_sha256=hashlib.sha256(ids).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        producer_revision=revision,
        qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_source_sha256=before,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Qualify without modifying inputs; optional receipts are create-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", type=Path)
    parser.add_argument("--step-seconds", type=float, required=True)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--check-receipt", type=Path)
    args = parser.parse_args(argv)
    result = qualify_export(args.condition, args.step_seconds)
    content = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.check_receipt is not None:
        previous = json.loads(
            read_bounded(args.check_receipt), object_pairs_hook=unique_json
        )
        if json.dumps(previous, sort_keys=True, allow_nan=False) != json.dumps(
            asdict(result), sort_keys=True, allow_nan=False
        ):
            message = "receipt differs from current export qualification"
            raise ExportMismatch(message)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
