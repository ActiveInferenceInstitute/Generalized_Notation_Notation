"""Maintained local template library for the ``gnn`` CLI."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_NAME = __package__ or "cli"
TEMPLATE_INDEX_RESOURCE = "template_index.json"


@dataclass(frozen=True)
class TemplateRecord:
    """One maintained template entry exposed through ``gnn pull``."""

    name: str
    description: str
    source: str
    filename: str

    @property
    def source_resource(self) -> Traversable:
        """Return the packaged template source resource."""
        resource = resources.files(PACKAGE_NAME)
        for part in self.source.split("/"):
            resource = resource.joinpath(part)
        return resource

    @property
    def checksum(self) -> str:
        """Return the SHA256 checksum for the template source."""
        return _sha256_resource(self.source_resource)

    def as_dict(self) -> Dict[str, str]:
        """Return a JSON-serializable description."""
        return {
            "name": self.name,
            "description": self.description,
            "source": f"package://{PACKAGE_NAME}/{self.source}",
            "filename": self.filename,
            "sha256": self.checksum,
        }


def _load_template_index() -> Dict[str, TemplateRecord]:
    """Load maintained template records from the JSON index."""
    index_text = (
        resources.files(PACKAGE_NAME)
        .joinpath(TEMPLATE_INDEX_RESOURCE)
        .read_text(encoding="utf-8")
    )
    raw_records = json.loads(index_text)
    if not isinstance(raw_records, list):
        raise ValueError("Template index must be a list of records")
    records: Dict[str, TemplateRecord] = {}
    for raw in raw_records:
        record = _template_record_from_raw(raw)
        if record.name in records:
            raise ValueError(f"Duplicate template name: {record.name}")
        records[record.name] = record
    return records


def _template_record_from_raw(raw: Any) -> TemplateRecord:
    """Validate and construct one maintained template index record."""
    if not isinstance(raw, dict):
        raise ValueError("Template index entries must be objects")
    record = TemplateRecord(
        name=str(raw["name"]),
        description=str(raw["description"]),
        source=str(raw["source"]),
        filename=str(raw["filename"]),
    )
    _validate_template_record(record)
    return record


def _validate_template_record(record: TemplateRecord) -> None:
    """Reject template records that could escape package/output boundaries."""
    filename_path = Path(record.filename)
    if (
        filename_path.is_absolute()
        or filename_path.name != record.filename
        or ".." in filename_path.parts
    ):
        raise ValueError(f"Template filename must be a basename: {record.filename}")

    if "\\" in record.source:
        raise ValueError(
            f"Template source must use package-relative POSIX paths: {record.source}"
        )
    source_path = PurePosixPath(record.source)
    if (
        source_path.is_absolute()
        or ".." in source_path.parts
        or len(source_path.parts) < 2
        or source_path.parts[0] != "template_assets"
        or source_path.suffix != ".md"
    ):
        raise ValueError(
            "Template source must stay under package template_assets/*.md: "
            f"{record.source}"
        )


TEMPLATE_INDEX: Dict[str, TemplateRecord] = _load_template_index()


def _sha256_chunks(chunks: Iterable[bytes]) -> str:
    """Return the SHA256 hex digest of a byte-chunk iterable."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_chunks(iter(lambda: handle.read(1024 * 1024), b""))


def _sha256_resource(resource: Traversable) -> str:
    with resource.open("rb") as handle:
        return _sha256_chunks(iter(lambda: handle.read(1024 * 1024), b""))


def list_templates() -> List[Dict[str, str]]:
    """List available templates with checksums."""
    return [
        record.as_dict()
        for record in sorted(TEMPLATE_INDEX.values(), key=lambda r: r.name)
    ]


def show_template(name: str) -> Dict[str, str]:
    """Return one template record with checksum metadata."""
    if name not in TEMPLATE_INDEX:
        available = ", ".join(sorted(TEMPLATE_INDEX))
        raise KeyError(f"Unknown template '{name}'. Available templates: {available}")
    return TEMPLATE_INDEX[name].as_dict()


def pull_template(
    name: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Copy a template into ``output_dir`` with collision and checksum handling."""
    if name not in TEMPLATE_INDEX:
        available = ", ".join(sorted(TEMPLATE_INDEX))
        raise KeyError(f"Unknown template '{name}'. Available templates: {available}")

    record = TEMPLATE_INDEX[name]
    source_resource = record.source_resource
    if not source_resource.is_file():
        raise FileNotFoundError(
            f"Template source does not exist: {record.as_dict()['source']}"
        )

    destination = output_dir / record.filename
    source_checksum = record.checksum
    result: Dict[str, Any] = {
        "template": name,
        "source": record.as_dict()["source"],
        "destination": str(destination),
        "sha256": source_checksum,
        "dry_run": dry_run,
        "overwritten": False,
        "copied": False,
    }

    if destination.is_symlink():
        raise FileExistsError(f"Refusing to write through symlink: {destination}")

    if destination.exists():
        existing_checksum = _sha256(destination)
        result["existing_sha256"] = existing_checksum
        if existing_checksum == source_checksum:
            result["message"] = "Template already present with matching checksum"
            return result
        if not overwrite:
            raise FileExistsError(
                f"Destination exists with different checksum: {destination}. "
                "Pass --overwrite to replace it."
            )
        result["overwritten"] = True

    if dry_run:
        result["message"] = "Dry run: no files copied"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise FileExistsError(f"Refusing to write through symlink: {destination}")
    temp_destination = output_dir / f".{record.filename}.tmp"
    if temp_destination.is_symlink():
        raise FileExistsError(f"Refusing to use symlink temp path: {temp_destination}")
    if temp_destination.exists():
        temp_destination.unlink()
    with resources.as_file(source_resource) as source_path:
        shutil.copy2(source_path, temp_destination)
    copied_checksum = _sha256(temp_destination)
    if copied_checksum != source_checksum:
        temp_destination.unlink(missing_ok=True)
        raise OSError(
            f"Checksum mismatch after copy: expected {source_checksum}, got {copied_checksum}"
        )
    temp_destination.replace(destination)
    result["copied"] = True
    result["message"] = "Template copied"
    return result
