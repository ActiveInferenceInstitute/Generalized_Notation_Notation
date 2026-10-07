"""Portable interactive bundles preserve dependencies and producer collisions."""

from pathlib import Path

from gnn.website.collection import _collect_visualizations


def test_interactive_sidecars_keep_relative_layout_without_collision(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "8_visualization_output", tmp_path / "9_advanced_viz_output"]
    for index, root in enumerate(roots):
        (root / "models").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "models/view.html").write_text(
            '<script src="../data/scene.js"></script>'
        )
        (root / "data/scene.js").write_text(f"const producer = {index};")
    assets = tmp_path / "20_website_output/assets"
    assets.mkdir(parents=True)
    entries, warnings = _collect_visualizations(roots, assets)
    assert not warnings
    assert len(entries) == 2
    assert entries[0]["path"] != entries[1]["path"]
    for index, entry in enumerate(entries):
        html = assets / entry["path"]
        assert html.read_text() == '<script src="../data/scene.js"></script>'
        assert (
            html.parent / "../data/scene.js"
        ).read_text() == f"const producer = {index};"
