"""Contracts for the category routing table (``tests.categories``)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import tests.categories as categories_mod
from tests.categories import (
    MODULAR_TEST_CATEGORIES,
    TestCategory,
    get_all_test_files,
    get_category,
    get_category_files,
    get_category_names,
    missing_category_files,
)

if TYPE_CHECKING:
    from tests.test_runner_modular import _ModularTestRunner


pytestmark = pytest.mark.fast


def test_every_category_defines_the_full_config_shape() -> None:
    """Each category carries name/description/files/markers/timeout/failure caps."""
    for name, config in MODULAR_TEST_CATEGORIES.items():
        assert config.get("name"), f"{name}: missing name"
        assert config.get("description"), f"{name}: missing description"
        assert config.get("files"), f"{name}: no test files"
        assert config.get("timeout_seconds", 0) > 0, f"{name}: bad timeout"
        assert config.get("max_failures", 0) > 0, f"{name}: bad max_failures"
        assert isinstance(config.get("parallel"), bool), f"{name}: parallel not bool"


def test_accessors_round_trip() -> None:
    names = get_category_names()
    assert names == list(MODULAR_TEST_CATEGORIES.keys())
    first = names[0]
    assert get_category(first) is MODULAR_TEST_CATEGORIES[first]
    assert get_category("definitely_not_a_category") == {}
    assert get_category_files("definitely_not_a_category") == []
    for name in names:
        for entry in get_category_files(name):
            assert entry.endswith(".py"), (name, entry)


def test_all_test_files_sorted_and_deduplicated() -> None:
    files = get_all_test_files()
    assert files == sorted(files)
    assert len(files) == len(set(files))
    raw: set[str] = set()
    for config in MODULAR_TEST_CATEGORIES.values():
        raw.update(config.get("files", []))
    assert set(files) == raw


def test_missing_category_files_empty_for_real_tree() -> None:
    """The routing table must reference files that exist under tests/."""
    assert missing_category_files() == {}


def test_missing_category_files_with_isolated_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an isolated routing table, absent files are reported per category."""
    synthetic: TestCategory = {
        "name": "Synthetic",
        "description": "Synthetic category for the drift detector",
        "files": ["present.py", "absent.py"],
        "markers": [],
        "timeout_seconds": 10,
        "max_failures": 1,
        "parallel": False,
    }
    (tmp_path / "present.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(categories_mod, "MODULAR_TEST_CATEGORIES", {"syn": synthetic})

    assert missing_category_files(tmp_path) == {"syn": ["absent.py"]}
    assert missing_category_files(str(tmp_path)) == {"syn": ["absent.py"]}


def _make_runner() -> _ModularTestRunner:
    """Minimal _ModularTestRunner for warning-contract tests."""
    from tests.test_runner_modular import _ModularTestRunner

    return _ModularTestRunner(
        SimpleNamespace(output_dir="unused"), logging.getLogger("contract-test")
    )


def test_discovery_warns_on_unmatched_entry(caplog: pytest.LogCaptureFixture) -> None:
    """A category entry matching no file is logged, not silently skipped."""
    runner = _make_runner()
    with caplog.at_level(logging.WARNING, logger="contract-test"):
        matched = runner.discover_test_files("syn", {"files": ["nope/absent_test.py"]})
    assert matched == []
    assert any(
        "nope/absent_test.py" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_startup_drift_warning_reports_missing_entries(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_warn_stale_category_files surfaces routing-table drift at startup."""
    import tests.test_runner_modular as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "missing_category_files",
        lambda *a, **k: {"syn": ["zzz/absent.py"]},
    )
    runner = _make_runner()
    with caplog.at_level(logging.WARNING, logger="contract-test"):
        missing = runner._warn_stale_category_files()
    assert missing == {"syn": ["zzz/absent.py"]}
    assert any(
        "1 category file entries match no file across 1 categories" in rec.message
        and "syn: ['zzz/absent.py']" in rec.message
        for rec in caplog.records
    )
