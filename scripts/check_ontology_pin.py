#!/usr/bin/env python3
"""Fail when the pinned ontology vocabulary bridge drifts from its pin record.

GNN validates ``ActInfOntologyAnnotation`` labels against the Active Inference
Ontology vocabulary bridge published by
``ActiveInferenceInstitute/Active_Inference_Ontology`` (schema
``active-inference-ontology/gnn-terms/v1``). The repository keeps a byte-exact
copy at ``src/gnn/ontology/active_inference_ontology_terms.json`` beside a
provenance record at ``src/gnn/ontology/ontology-pin.json``.

This gate verifies, deterministically and offline:

1. the copy exists and its SHA-256 equals the pin's ``sha256``;
2. the pin declares repository, commit, release, path and both embedded digests;
3. the bridge's own ``source.sha256`` / ``export.sha256`` match the pin;
4. the bridge's ``counts`` block agrees with the document it carries;
5. resolved declared labels carry ``term-*`` identifiers and unresolved ones
   carry none.

Point 5 is the substance of the pin: the bridge maps the 64 declared GNN labels
to release-v5 identifiers where a counterpart exists (12 today) and records the
remainder as unresolved. The ``obo:ACTO_*`` / ``obo:TEMP_*`` values in the
bundled vocabulary stay declared metadata and are never used as identifiers.

Verification lives in ``gnn.ontology.bridge.verify_ontology_pin`` so this gate,
the runtime status record and the tests share one implementation.

Usage: uv run --extra dev python scripts/check_ontology_pin.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _verify() -> Dict[str, Any]:
    """Delegate verification to the package (single implementation)."""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from gnn.ontology.bridge import (
        ontology_bridge_path,
        ontology_pin_path,
        verify_ontology_pin,
    )

    summary = verify_ontology_pin()
    summary["copyPath"] = str(ontology_bridge_path())
    summary["pinPath"] = str(ontology_pin_path())
    return summary


def main(argv: list[str] | None = None) -> int:
    """Verify the pinned copy; exit 1 when it or its pin is non-compliant."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; mismatches fail by default.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full summary as JSON."
    )
    args = parser.parse_args(argv)

    summary = _verify()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["verified"]:
        counts = (summary.get("bridge") or {}).get("counts") or {}
        print(
            "check_ontology_pin: verified {copy} (sha256 {sha}) against {pin} "
            "at {commit}".format(
                copy=summary.get("copyPath"),
                sha=summary.get("sha256"),
                pin=summary.get("pinPath"),
                commit=(summary.get("pin") or {}).get("commit"),
            )
        )
        print(
            "check_ontology_pin: bridge release {release}, {declared} declared "
            "labels ({resolved} resolved, {unresolved} unresolved), {terms} terms".format(
                release=(summary.get("bridge") or {}).get("release"),
                declared=counts.get("declaredLabels"),
                resolved=counts.get("resolvedDeclaredLabels"),
                unresolved=counts.get("unresolvedDeclaredLabels"),
                terms=counts.get("terms"),
            )
        )
        return 0

    print(f"check_ontology_pin: {len(summary['errors'])} failure(s):")
    for error in summary["errors"]:
        print(f"  {error}")
    print(
        "Re-pin by copying GNN/act_inf_ontology_terms.json from the ontology "
        "repository at the reviewed commit and updating "
        "src/gnn/ontology/ontology-pin.json with the new sha256 and embedded "
        "source/export digests."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
