"""Verify the source distribution and the two independently built wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import tarfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

METADATA_KEYS = (
    "Name",
    "Version",
    "Requires-Python",
    "Requires-Dist",
    "Provides-Extra",
)
TYPING_MARKER = "gnn/py.typed"


def metadata_fields(contents: bytes) -> dict[str, list[str]]:
    message = BytesParser().parsebytes(contents)
    return {key: sorted(message.get_all(key, [])) for key in METADATA_KEYS}


def wheel_inventory(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    with ZipFile(path) as archive:
        filenames = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"{path.name} contains duplicate archive members")
        for name in filenames:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or str(member) != name:
                raise ValueError(f"{path.name} contains an invalid member path: {name}")
        (record_name,) = [
            name for name in filenames if name.endswith(".dist-info/RECORD")
        ]
        reader = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        records: dict[str, tuple[str, str]] = {}
        for name, digest, size in reader:
            if name in records:
                raise ValueError(f"{path.name} RECORD repeats {name}")
            records[name] = (digest, size)
        if records.keys() != set(filenames) or records[record_name] != ("", ""):
            raise ValueError(
                f"{path.name} RECORD does not own exactly the wheel members"
            )

        inventory = {}
        for name in filenames:
            if name == record_name:
                continue
            contents = archive.read(name)
            digest_bytes = hashlib.sha256(contents).digest()
            digest_b64 = base64.urlsafe_b64encode(digest_bytes).rstrip(b"=").decode()
            if records[name] != (f"sha256={digest_b64}", str(len(contents))):
                raise ValueError(
                    f"{path.name} RECORD digest or size differs for {name}"
                )
            inventory[name] = digest_bytes.hex()
        metadata_name = record_name.removesuffix("RECORD") + "METADATA"
        return inventory, metadata_fields(archive.read(metadata_name))


def verify_distributions(direct: Path, rebuilt: Path, source: Path) -> dict:
    direct_contents, direct_metadata = wheel_inventory(direct)
    rebuilt_contents, rebuilt_metadata = wheel_inventory(rebuilt)
    if direct_contents != rebuilt_contents or direct_metadata != rebuilt_metadata:
        different = sorted(
            name
            for name in direct_contents.keys() | rebuilt_contents.keys()
            if direct_contents.get(name) != rebuilt_contents.get(name)
        )
        raise ValueError(f"direct and sdist wheels differ: {different}")
    if TYPING_MARKER not in direct_contents:
        raise ValueError(f"both wheels must contain {TYPING_MARKER}")
    with tarfile.open(source) as archive:
        (package_info,) = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        stream = archive.extractfile(package_info)
        if stream is None or metadata_fields(stream.read()) != direct_metadata:
            raise ValueError("sdist PKG-INFO differs from wheel metadata")

    digests = {}
    for role, path in (
        ("wheel", direct),
        ("rebuilt_wheel", rebuilt),
        ("sdist", source),
    ):
        with path.open("rb") as artifact:
            digests[role] = hashlib.file_digest(artifact, "sha256").hexdigest()
    return {
        "metadata": direct_metadata,
        "wheel_members_verified": len(direct_contents) + 1,
        "pep561_marker": TYPING_MARKER,
        "artifacts_sha256": digests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--rebuilt-wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify_distributions(args.wheel, args.rebuilt_wheel, args.sdist)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
