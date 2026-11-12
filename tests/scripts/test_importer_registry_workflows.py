"""The workflow-group block and the contract page must stay one artifact.

``docs/workflows/gnn-fep.md`` records the group's members and the canonical pin
reader; ``docs/downstream/importers.json`` records the same rows for the gate.
Two hand-edited artifacts that can disagree are worse than one, so these tests
drive ``check_workflow_groups`` against the live data and against a mutated copy,
requiring the specific finding that each divergence must produce.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "check_importer_registry.py"
_spec = importlib.util.spec_from_file_location("check_importer_registry_wg", _SCRIPT)
assert _spec is not None and _spec.loader is not None
registry_gate = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_importer_registry_wg", registry_gate)
_spec.loader.exec_module(registry_gate)

CANONICAL = registry_gate.CANONICAL_IMPORTER_ARM
UPSTREAM = "Generalized_Notation_Notation"


def _live() -> Dict[str, Any]:
    return json.loads(
        (REPO_ROOT / "docs" / "downstream" / "importers.json").read_text()
    )


def _group(registry: Dict[str, Any]) -> Dict[str, Any]:
    return registry["workflow_groups"]["gnn-fep"]


def _codes(registry: Dict[str, Any]) -> List[str]:
    findings = registry_gate.Findings()
    registry_gate.check_workflow_groups(registry, findings)
    return [item["code"] for item in findings.items]


def _member(registry: Dict[str, Any], repo: str) -> Dict[str, Any]:
    for member in _group(registry)["members"]:
        if member["repo"] == repo:
            return member
    raise AssertionError(f"{repo} not in the live group")


def _rooted(tmp_path: Path, monkeypatch: Any) -> None:
    """Point the gate's contract lookup at a writable copy of the contract."""
    relative = Path(_group(_live())["contract"])
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        (REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(registry_gate, "ROOT", tmp_path)


def test_live_registry_and_contract_agree() -> None:
    """The shipped pair is the evidence that the shipped data is coherent."""
    findings = registry_gate.Findings()
    registry_gate.check_workflow_groups(_live(), findings)

    assert findings.items == []


def test_every_member_records_the_canonical_arm() -> None:
    registry = _live()
    arms = {member["repo"]: member["arm"] for member in _group(registry)["members"]}

    assert arms, "the group must not be empty"
    assert set(arms.values()) == {CANONICAL}


def test_group_lists_the_upstream_and_twenty_one_importers() -> None:
    registry = _live()
    members = _group(registry)["members"]

    roles = {member["role"] for member in members}
    assert len(members) == 22
    assert roles == {"upstream", "importer"}
    assert [m["repo"] for m in members if m["role"] == "upstream"] == [UPSTREAM]


def test_a_member_with_its_own_pin_reader_fails_the_arm_check() -> None:
    """A member that keeps a private script is the drift this contract forbids."""
    registry = _live()
    _member(registry, "pymcp")["arm"] = "python scripts/gnn_conformance.py"

    assert "group-member-arm" in _codes(registry)


def test_a_member_missing_from_the_json_fails() -> None:
    """A contract row the JSON block does not register."""
    registry = _live()
    group = _group(registry)
    group["members"] = [m for m in group["members"] if m["repo"] != "Biofirm"]

    assert "group-member-unregistered" in _codes(registry)


def test_a_member_missing_from_the_contract_table_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A JSON member the contract page does not document."""
    _rooted(tmp_path, monkeypatch)
    contract = tmp_path / Path(_group(_live())["contract"])
    text = contract.read_text(encoding="utf-8")
    assert "| [AEOS]" in text
    contract.write_text(
        text.replace("| [AEOS]", "| [Absent_Repository]"), encoding="utf-8"
    )

    assert "group-member-undocumented" in _codes(_live())


def test_a_pin_that_drifted_from_the_registry_row_fails() -> None:
    registry = _live()
    _member(registry, "GEO_INFER")["pin"] = "gnn/upstream.json"

    assert "group-member-pin" in _codes(registry)


def test_a_wrapper_without_a_reason_fails() -> None:
    registry = _live()
    wrapper = _member(registry, "active_inference_battleship")
    wrapper.pop("reason")

    assert "group-member-reason" in _codes(registry)


def test_an_unknown_member_fails() -> None:
    registry = _live()
    _group(registry)["members"].append(
        {
            "repo": "Not_A_Repository",
            "role": "importer",
            "install": "direct",
            "arm": CANONICAL,
            "pin": None,
            "pointer": None,
            "docs_index": None,
            "docs_flat_index": None,
            "manuscript": None,
            "gate": None,
        }
    )

    assert "group-member-unknown" in _codes(registry)


def test_a_duplicated_member_fails() -> None:
    registry = _live()
    _group(registry)["members"].append(copy.deepcopy(_member(registry, "AEOS")))

    assert "group-member-duplicate" in _codes(registry)


def test_the_upstream_member_carries_no_pin_or_pointer() -> None:
    registry = _live()
    _member(registry, UPSTREAM)["pin"] = ".github/gnn-pair.json"

    assert "group-upstream-pin" in _codes(registry)


def test_an_unknown_docs_hub_fails() -> None:
    registry = _live()
    _member(registry, "AEOS")["docs_index"] = "docs/hub.md"

    assert "group-member-docs" in _codes(registry)
