"""DisCoPy diagnostics and explicit missing-dependency behavior.

These are pytest assertions: import or translation failures must not be reduced
to a returned Boolean, which pytest would accept regardless of its value.
"""

from pathlib import Path

import pytest

from gnn.render.discopy import file_translation
from gnn.render.discopy.bootstrap import (
    check_discopy_availability,
    create_discopy_error_report,
    generate_setup_report,
)
from gnn.render.discopy.translator import (
    gnn_file_to_discopy_diagram,
    gnn_file_to_discopy_matrix_diagram,
)


def test_discopy_improvements(tmp_path: Path) -> None:
    availability = check_discopy_availability()
    assert availability and all(
        isinstance(value, bool) for value in availability.values()
    )
    assert availability["tensor_components"]
    setup_report = generate_setup_report()
    assert "DisCoPy" in setup_report and "JAX" in setup_report

    source = tmp_path / "model.md"
    source.write_text(
        "## ModelName\nTest Model\n\n## StateSpaceBlock\nA[2]\nB[3]\n\n## Connections\nA > B\n",
        encoding="utf-8",
    )
    report = create_discopy_error_report(source, "unavailable")
    assert report["success"] is False
    assert report["error_type"] == "unavailable"
    assert report["gnn_file"] == str(source)
    assert report["setup_instructions"] == setup_report
    diagram = gnn_file_to_discopy_diagram(source)
    assert diagram is not None
    assert len(diagram.boxes) == 1
    assert source.read_text(encoding="utf-8").endswith("A > B\n")
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("matrix", [False, True])
def test_file_translation_refuses_missing_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix: bool
) -> None:
    source = tmp_path / "model.md"
    source.write_text("## ModelName\nUnavailable\n", encoding="utf-8")
    flag = "JAX_AVAILABLE" if matrix else "TENSOR_COMPONENTS_AVAILABLE"
    monkeypatch.setattr(file_translation, flag, False)
    translator = (
        gnn_file_to_discopy_matrix_diagram if matrix else gnn_file_to_discopy_diagram
    )
    assert translator(source) is None
    assert list(tmp_path.iterdir()) == [source]
