"""Saved Gaussian figures bind control identities to unchanged values."""

from copy import deepcopy
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest
from matplotlib.ticker import MaxNLocator
from PIL import Image

from gnn.analysis.rxinfer.family_visuals import continuous_png


@pytest.mark.parametrize("vfe", [[], [2.0, 1.0]])
@pytest.mark.parametrize("units", [{}, {"state": "m", "control": "N", "vfe": "arb"}])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("convention", [None, "native objective"])
def test_saved_gaussian_controls_and_unreported_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vfe: list[float],
    units: dict,
    nested: bool,
    convention: str | None,
) -> None:
    data = {
        "model_kind": "continuous",
        "num_timesteps": 2,
        "beliefs": [[-2.0, 3.0], [-1.0, 2.0]],
        "posterior_cov": [[[0.4, 0.1], [0.1, 0.3]], [[0.2, 0.0], [0.0, 0.1]]],
        "controls": [[0.1, 0.2], [0.2, 0.3]],
        "vfe_per_iteration": vfe,
        "variational_free_energy_convention": convention,
        "model_parameters" if nested else "units": {"units": units}
        if nested
        else units,
    }
    original = deepcopy(data)
    captured = {}
    savefig = matplotlib.figure.Figure.savefig

    def capture(figure, *args, **kwargs):
        states, controls, convergence = figure.axes
        captured["labels"] = controls.get_legend_handles_labels()[1]
        captured["traces"] = [line.get_ydata().tolist() for line in controls.lines]
        captured["x"] = [line.get_xdata().tolist() for line in controls.lines]
        captured["styles"] = {
            (line.get_marker(), line.get_linestyle()) for line in controls.lines
        }
        for axis, quantity in ((states, "state"), (controls, "control")):
            assert isinstance(axis.xaxis.get_major_locator(), MaxNLocator)
            assert units.get(quantity, "units unspecified") in axis.get_ylabel()
            assert "zero-based" in axis.get_xlabel()
        captured["vfe_axis"] = convergence.axison
        if vfe:
            np.testing.assert_array_equal(convergence.lines[0].get_ydata(), vfe)
            assert isinstance(convergence.xaxis.get_major_locator(), MaxNLocator)
            assert "zero-based" in convergence.get_xlabel()
            assert (convention or "convention unspecified") in convergence.get_ylabel()
            assert units.get("vfe", "units unspecified") in convergence.get_ylabel()
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    path = tmp_path / "gaussian.png"
    assert continuous_png(data, path, "Gaussian") == str(path)
    assert captured["labels"] == ["Control 0", "Control 1"]
    assert captured["traces"] == [[0.1, 0.2], [0.2, 0.3]]
    assert captured["x"] == [[0, 1], [0, 1]]
    assert len(captured["styles"]) == 2
    assert captured["vfe_axis"] is bool(vfe)
    assert data == original
    with Image.open(path) as image:
        assert image.format == "PNG" and image.width > 100
