"""Protect the paired CPU runner's disk budget before either runtime sync."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_paired_runtime_disk_budget():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/geo-infer-interchange.yml").read_text()
    )
    steps = workflow["jobs"]["geo-infer-interchange"]["steps"]
    names = [step.get("name", "") for step in steps]
    geo = names.index("Install GEO locked runtime")
    gnn = names.index("Install GNN locked runtime")
    reclaim = names.index("Reclaim unused hosted runner toolchains")
    prune = names.index("Prune GEO wheel archives before GNN installation")
    assert reclaim < geo < prune < gnn
    for toolchain in ("/usr/share/dotnet", "/usr/local/lib/android", "/opt/ghc"):
        assert toolchain in steps[reclaim]["run"]
    assert steps[prune]["run"] == "uv cache prune"
    for index in (geo, gnn):
        assert "uv sync --no-cache --project " in steps[index]["run"]
        assert "--locked" in steps[index]["run"]
    assert "uv cache prune" in steps[gnn]["run"]
    setups = [step for step in steps if "astral-sh/setup-uv@" in step.get("uses", "")]
    assert len(setups) == 2
    assert all(step["with"]["enable-cache"] is False for step in setups)
