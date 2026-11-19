"""Validate the authoring templates and the README example as GNN documents."""

from pathlib import Path

import pytest

from gnn.parsers.basic import validate_gnn_syntax

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "docs" / "templates"
TEMPLATE_NAMES = {
    "basic_gnn_template.md",
    "hierarchical_template.md",
    "multiagent_template.md",
    "pomdp_template.md",
}
TEMPLATE_FILES = sorted(TEMPLATES_DIR.glob("*_template.md"))


@pytest.mark.unit
@pytest.mark.fast
def test_documented_templates_are_present() -> None:
    assert TEMPLATE_NAMES <= {path.name for path in TEMPLATE_FILES}


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("path", TEMPLATE_FILES, ids=lambda path: path.name)
def test_template_passes_strict_gnn_validation(path: Path) -> None:
    valid, errors = validate_gnn_syntax(path, "strict")
    assert valid, errors


@pytest.mark.unit
@pytest.mark.fast
def test_readme_example_passes_strict_gnn_validation() -> None:
    readme = (TEMPLATES_DIR / "README.md").read_text(encoding="utf-8")
    _, opening, fenced = readme.partition("```gnn\n")
    example, closing, _ = fenced.partition("\n```")
    assert opening and closing, "README.md must contain a fenced GNN example"
    valid, errors = validate_gnn_syntax(example, "strict", is_content=True)
    assert valid, errors
