"""URL-boundary units, not installed/native source qualification."""

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SHA256 = "e04e9114c860a622f26ee8d323ac44b77c6484f5a29bdc7654fb298a9b58c3b2"
CASES = json.loads(Path(__file__).with_name("bootstrap_origin_cases.json").read_text())


@pytest.fixture(scope="module")
def strict_bootstrap() -> Iterator[ModuleType]:
    path = ROOT / "interop" / "gnn_bootstrap.py"
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CANDIDATE_SHA256
    module = ModuleType("_reviewed_strict_origin_bootstrap")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(payload, str(path), "exec", dont_inherit=True), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("case", CASES)
def test_pin_identity_and_actual_origin_url_have_distinct_boundaries(
    strict_bootstrap: ModuleType, case: dict
) -> None:
    assert strict_bootstrap.normalize_repository(case["input"]) == case["pin_expected"]
    assert (
        strict_bootstrap._normalize_repository(case["input"], allow_bare=False)
        == case["installed_url_expected"]
    )
