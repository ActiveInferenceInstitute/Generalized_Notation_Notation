"""Build miniature Stage 04 evidence for manuscript custody unit tests."""

import hashlib
import json
from pathlib import Path

from gnn.manuscript.publication_artifacts import delivery_digests


def write_template_validation(root: Path) -> None:
    """Model the template's inventory, checks, and evidence file bindings."""
    config = root / "docs/manuscript/config.yaml"
    if not config.exists():
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("render:\n  formats:\n    pdf: true\n", encoding="utf-8")
        (root / "output/manuscript/config.yaml").write_bytes(config.read_bytes())

    report_dir = root / "output/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = root / "output/web"
    combined_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "validation_report": report_dir / "validation_report.json",
        "artifact_manifest": report_dir / "artifact_manifest.json",
        "composition_manifest": report_dir / "manuscript_composition.json",
        "combined_manuscript": combined_dir / "_combined_manuscript.md",
    }
    files["validation_report"].write_text(
        json.dumps(
            {"checks": {"PDF validation": True}, "summary": {"all_passed": True}}
        ),
        encoding="utf-8",
    )
    files["artifact_manifest"].write_text(
        json.dumps(
            {
                "entries": [
                    {"path": rel, "contract_match": True, **digest}
                    for rel, digest in delivery_digests(root).items()
                ]
            }
        ),
        encoding="utf-8",
    )
    files["composition_manifest"].write_text("{}", encoding="utf-8")
    files["combined_manuscript"].write_text("# Combined manuscript\n", encoding="utf-8")
    evidence = {
        kind: {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for kind, path in files.items()
    }
    (report_dir / "rendered_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "template-rendered-provenance-v2",
                "evidence_files": evidence,
            }
        ),
        encoding="utf-8",
    )
