"""Concurrent native strategy calls keep their real source artifacts separate."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gnn.parsers import GNNParsingSystem
from gnn.testing import RoundTripTestStrategy, round_trip_config


def test_concurrent_native_sources_retain_distinct_saved_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    ).read_text(encoding="utf-8")
    sources = []
    names = {"First supplied model", "Second supplied model"}
    for index, name in enumerate(sorted(names)):
        directory = tmp_path / str(index)
        directory.mkdir()
        source = directory / "same_name.md"
        source.write_text(original.replace("Simple MDP Agent", name), encoding="utf-8")
        sources.append(source)
    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    strategy = RoundTripTestStrategy()
    template = strategy.round_trip_tester
    reference_before = template.reference_file
    formats_before = list(template.supported_formats)
    template.temp_dir = tmp_path / "saved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda source: strategy.test([source]), sources))

    assert all(outcome["success"] for outcome in outcomes)
    for source, outcome in zip(sources, outcomes):
        assert set(outcome["file_results"]) == {str(source)}
        assert outcome["file_results"][str(source)]["total_formats_tested"] == 20
    artifacts = list(template.temp_dir.glob("source-*/test_model.json"))
    assert len(artifacts) == 2
    saved_names = set()
    for artifact in artifacts:
        result = GNNParsingSystem().parse_file(artifact)
        assert result.success, result.errors
        saved_names.add(result.model.model_name)
        assert len(list(artifact.parent.glob("test_model.*"))) == 20
    assert saved_names == names
    assert strategy.round_trip_tester is template
    assert template.reference_file == reference_before
    assert template.supported_formats == formats_before
