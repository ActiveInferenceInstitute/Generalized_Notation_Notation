"""Observable scientific presentation contracts, using real PNG/GIF exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from gnn.analysis.viz_animations import (
    animate_belief_evolution,
    animate_cross_framework_gridworld_trajectories,
    animate_gridworld_trajectory,
)
from gnn.analysis.viz_plots import generate_vfe_vs_efe_plot
from gnn.visualization.matrix.visualizer import MatrixVisualizer


def test_signed_heatmap_text_tracks_rendered_color_and_preserves_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed ranges must not determine text contrast from a fixed value threshold."""
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args: None)
    matrix = np.array([[-100.0, 0.0], [50.0, 100.0]])
    original = matrix.copy()
    path = tmp_path / "signed.png"
    try:
        assert MatrixVisualizer().generate_matrix_heatmap("signed", matrix, path)
        fig = plt.gcf()
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.images[0].get_array(), original)
        np.testing.assert_array_equal(matrix, original)
        assert ax.texts[0].get_color() == "white"
        assert ax.texts[-1].get_color() == "black"
        assert "0-based" in ax.get_xlabel() and "0-based" in ax.get_ylabel()
        with Image.open(path) as image:
            assert image.width > 100 and image.height > 100
        assert path.with_suffix(".csv").exists()
    finally:
        real_close("all")


def test_natural_log_transition_entropy_is_labeled_nats() -> None:
    tensor = np.full((2, 2, 1), 0.5)
    text = MatrixVisualizer()._generate_tensor_statistics(tensor, "B", "transition")
    assert "0.693 nats" in text
    assert "bits" not in text


def test_nonprobability_tensor_does_not_display_nan_as_entropy() -> None:
    tensor = np.array([[[-1.0], [1.0]], [[2.0], [0.0]]])
    with np.errstate(invalid="raise"):
        text = MatrixVisualizer()._generate_tensor_statistics(
            tensor, "raw", "transition"
        )
    assert "unavailable: not normalized nonnegative probabilities" in text
    assert "nan" not in text


@pytest.mark.parametrize(
    "beliefs", [[[2.0, -1.0]], [[0.4, 0.4]], [[float("nan"), 0.5]]]
)
def test_categorical_animation_refuses_nonprobability_and_gaussian_values(
    tmp_path: Path, beliefs: list[list[float]]
) -> None:
    path = tmp_path / "invalid.gif"
    with pytest.raises(ValueError, match="finite|categorical"):
        animate_belief_evolution(beliefs, path)
    assert not path.exists()


def test_categorical_animation_binds_real_frames_to_trace_and_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "posterior.gif"
    beliefs = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
    original = [row[:] for row in beliefs]
    assert animate_belief_evolution(beliefs, path, title="Bound posterior") == str(path)
    assert beliefs == original
    receipt = json.loads(path.with_suffix(".manifest.json").read_text())
    assert receipt["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(beliefs, separators=(",", ":"))
    assert receipt["trace_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert receipt["index_base"] == 0
    with Image.open(path) as image:
        assert image.n_frames == receipt["frame_count"] == len(beliefs)


@pytest.mark.parametrize("states", [[-1], [4], [True], [1.5], []])
def test_grid_trajectory_does_not_clamp_or_round_invalid_state_indices(
    tmp_path: Path, states: list[int]
) -> None:
    path = tmp_path / "invalid.gif"
    with pytest.raises(ValueError, match="indices"):
        animate_gridworld_trajectory(states, path, "invalid", state_count=4)
    assert not path.exists()


def test_cross_framework_panels_keep_each_grid_and_ended_trace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args: None)
    path = tmp_path / "separate.gif"
    items = [
        {
            "framework": "pymdp",
            "model_name": "four",
            "state_count": 4,
            "states": [0, 3],
        },
        {
            "framework": "rxinfer",
            "model_name": "nine",
            "state_count": 9,
            "states": [0, 5, 8],
        },
    ]
    try:
        animate_cross_framework_gridworld_trajectories(items, path)
        left, right = plt.gcf().axes
        assert left.images[0].get_array().shape == (2, 2)
        assert right.images[0].get_array().shape == (3, 3)
        assert "pymdp: four" in left.get_title()
        assert "trace ended" in left.get_title()
        assert "Timestep 2" in right.get_title()
        assert left.lines[1].get_xdata()[0] == 1
        assert right.lines[1].get_xdata()[0] == 2
        with Image.open(path) as image:
            assert image.n_frames == 3
    finally:
        real_close("all")


@pytest.mark.parametrize(
    "vfe,efe", [([1.0], [[]]), ([1.0], [[float("inf")]]), ([float("nan")], [1.0])]
)
def test_free_energy_plot_refuses_missing_or_nonfinite_costs(
    tmp_path: Path, vfe: list[float], efe: list
) -> None:
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="finite|nonempty"):
        generate_vfe_vs_efe_plot(vfe, efe, path)
    assert not path.exists()


def test_energy_axes_retain_domains_units_and_complete_declared_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.analysis import viz_plots

    real_save = viz_plots.safe_savefig
    figures = []

    def capture(path: Path, **kwargs):
        figures.append(plt.gcf())
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_plots, "safe_savefig", capture)
    convention = "Native signed utility and information gain; retain this exact declared convention."
    path = tmp_path / "energies.png"
    generate_vfe_vs_efe_plot(
        [3.0, 2.0, 1.0],
        [[0.8, 1.0], [0.7, 0.9]],
        path,
        vfe_per_iteration=True,
        efe_convention=convention,
        efe_units="reported score",
    )
    fig = figures[0]
    vfe_ax, efe_ax = fig.axes
    assert "1-based" in vfe_ax.get_xlabel()
    assert "0-based" in efe_ax.get_xlabel()
    assert "units unspecified" in vfe_ax.get_ylabel()
    assert "reported score" in efe_ax.get_ylabel()
    assert "Minimum reported policy score" in efe_ax.get_ylabel()
    assert all(float(tick).is_integer() for tick in vfe_ax.get_xticks())
    assert all(float(tick).is_integer() for tick in efe_ax.get_xticks())
    note = " ".join(" ".join(t.get_text().split()) for t in fig.texts)
    assert convention in note
    assert "VFE: convention/units unspecified." in note
    np.testing.assert_array_equal(vfe_ax.lines[0].get_ydata(), [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(efe_ax.lines[0].get_ydata(), [0.8, 0.7])
    with Image.open(path) as image:
        assert image.width > 100 and image.height > 100


def test_long_native_convention_has_readable_caption_and_exact_bound_detail(
    tmp_path: Path,
) -> None:
    from gnn.execute.pymdp.simulation import EFE_CONVENTION_PYMDP

    path = tmp_path / "native.png"
    generate_vfe_vs_efe_plot(
        [2.0, 1.0],
        [[0.2, 0.4], [0.1, 0.3]],
        path,
        efe_convention=EFE_CONVENTION_PYMDP,
    )
    receipt = json.loads(path.with_suffix(".conventions.json").read_text())
    assert receipt["declarations"][1]["convention"] == EFE_CONVENTION_PYMDP
    assert receipt["declarations"][1]["units"] is None
    assert receipt["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(receipt["declarations"], sort_keys=True)
    assert (
        receipt["declarations_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    )


def test_failed_save_does_not_bind_new_declarations_to_an_old_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.analysis import viz_plots

    path = tmp_path / "old.png"
    old_bytes = b"prior artifact"
    path.write_bytes(old_bytes)
    monkeypatch.setattr(viz_plots, "safe_savefig", lambda *args, **kwargs: None)
    try:
        generate_vfe_vs_efe_plot(
            [2.0, 1.0], [0.2, 0.1], path, efe_convention="x " * 150
        )
        assert path.read_bytes() == old_bytes
        assert not path.with_suffix(".conventions.json").exists()
    finally:
        plt.close("all")


@pytest.mark.parametrize("count", [0, False, -1, 1.5, "9"])
def test_cross_framework_explicit_invalid_count_refused_before_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "invalid declaration must be refused before figure allocation"
        )

    monkeypatch.setattr(plt, "subplots", forbidden)
    with pytest.raises(ValueError, match="state_count"):
        animate_cross_framework_gridworld_trajectories(
            [{"states": [0], "state_count": count}, {"states": [0], "state_count": 9}],
            tmp_path / "invalid.gif",
        )
    assert not (tmp_path / "invalid.gif").exists()


def test_deterministic_transition_entropy_uses_exact_zero_limit() -> None:
    tensor = np.eye(2)[:, :, None]
    with np.errstate(divide="raise", invalid="raise"):
        text = MatrixVisualizer()._generate_tensor_statistics(tensor, "B", "transition")
    assert "Mean Transition Entropy: 0.000 nats" in text
    assert "-0.000" not in text


def test_generic_3d_likelihood_keeps_axes_without_claiming_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.visualization.core.process import render_matrix_artifacts

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args: None)
    tensor = np.arange(12, dtype=float).reshape(2, 3, 2) - 4
    original = tensor.copy()
    mv = MatrixVisualizer()
    try:
        overview = tmp_path / "overview.png"
        assert mv.generate_matrix_analysis(
            [{"name": "A", "value": tensor.tolist()}], overview
        )
        ax = plt.gcf().axes[0]
        values = (
            ax.collections[0].get_array()
            if ax.collections
            else ax.images[0].get_array()
        )
        np.testing.assert_array_equal(np.asarray(values).reshape(3, 2), tensor[0])
        assert "axis 0 slice 0" in ax.get_title() and "action" not in ax.get_title()
        assert "Full tensor" in ax.texts[-1].get_text()
        assert "Mean: 1.500" in ax.texts[-1].get_text()
        assert ax.get_xlabel() == "Axis 2 Index (0-based)"
        assert ax.get_ylabel() == "Axis 1 Index (0-based)"
        assert plt.gcf().axes[1].get_ylabel() == "Value (units unspecified)"
        csv = next(tmp_path.glob("overview_matrix_*.csv")).read_text()
        assert "Axis 2 slice" in csv and "Action" not in csv
        failures: list[str] = []
        paths = render_matrix_artifacts(
            {"A": tensor}, tmp_path, "generic", mv, failures=failures
        )
        assert not failures
        assert (tmp_path / "generic_A_tensor.png").is_file()
        assert not any("analysis.png" in path for path in paths)
        fig = plt.gcf()
        assert all("Action" not in axis.get_title() for axis in fig.axes)
        planes = [image for axis in fig.axes for image in axis.images]
        assert len(planes) == 2
        assert all(image.get_clim() == (-4.0, 7.0) for image in planes)
        np.testing.assert_array_equal(planes[0].get_array(), original[:, :, 0])
        np.testing.assert_array_equal(planes[1].get_array(), original[:, :, 1])
        np.testing.assert_array_equal(tensor, original)
        payload = json.loads((tmp_path / "generic_A_threejs.json").read_text())
        np.testing.assert_array_equal(payload["values"], original)
    finally:
        real_close("all")


def test_recent_variation_is_descriptive_not_inference_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.analysis import viz_plots

    figures = []
    real_save = viz_plots.safe_savefig

    def capture(path, **kwargs):
        figures.append(plt.gcf())
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_plots, "safe_savefig", capture)
    values = [2.0] * 15
    viz_plots.generate_free_energy_plots(values, tmp_path / "variation.png")
    fig = figures[0]
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "Not an inference convergence test" in text and "Converged" not in text
    assert "Recent Variation" in fig.axes[3].get_title()
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), values)
    np.testing.assert_array_equal(fig.axes[3].lines[0].get_ydata(), np.zeros(15))


def test_compact_energy_caption_requires_explicit_source_bound_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.analysis import viz_plots
    from gnn.execute.pymdp.simulation import (
        EFE_CONVENTION_PYMDP,
        EFE_CONVENTION_PYMDP_SUMMARY,
    )

    figures = []
    real_save = viz_plots.safe_savefig

    def capture(path, **kwargs):
        figures.append(plt.gcf())
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_plots, "safe_savefig", capture)
    path = tmp_path / "declared.png"
    generate_vfe_vs_efe_plot(
        [2.0, 1.0],
        [[0.2], [0.1]],
        path,
        efe_convention=EFE_CONVENTION_PYMDP,
        efe_convention_summary=EFE_CONVENTION_PYMDP_SUMMARY,
    )
    assert "neg_efe = -EFE = expected utility" in " ".join(
        t.get_text() for t in figures[0].texts
    )
    right_axis = figures[0].axes[1]
    assert "Minimum reported policy score" in right_axis.get_ylabel()
    assert right_axis.lines[0].get_label() == "Minimum reported policy score"
    np.testing.assert_array_equal(right_axis.lines[0].get_ydata(), [0.2, 0.1])
    receipt = json.loads(path.with_suffix(".conventions.json").read_text())
    assert receipt["declarations"][1]["display_summary"] == EFE_CONVENTION_PYMDP_SUMMARY
    assert receipt["declarations"][1]["convention"] == EFE_CONVENTION_PYMDP
    arbitrary = "pymdp neg_efe linear payoff " + "source retains its own meaning " * 15
    generate_vfe_vs_efe_plot(
        [2.0], [0.2], tmp_path / "unclassified.png", efe_convention=arbitrary
    )
    caption = " ".join(t.get_text() for t in figures[1].texts)
    assert "source excerpt" in caption and "expected utility +" not in caption


def test_minimum_reported_policy_score_does_not_claim_agent_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.analysis import viz_plots

    figures = []
    real_save = viz_plots.safe_savefig

    def capture(path, **kwargs):
        figures.append(plt.gcf())
        return real_save(path, **kwargs)

    monkeypatch.setattr(viz_plots, "safe_savefig", capture)
    scores = [[1.0, 3.0], [2.0, 4.0]]
    viz_plots.generate_free_energy_plots(scores, tmp_path / "scores.png")
    ax, distribution = figures[0].axes[:2]
    minimum = next(
        line for line in ax.lines if line.get_label() == "Minimum reported policy score"
    )
    np.testing.assert_array_equal(minimum.get_ydata(), [1.0, 2.0])
    labels = " ".join(
        [line.get_label() for line in ax.lines]
        + [distribution.get_title(), distribution.get_xlabel()]
    )
    assert "Selected" not in labels and "Minimum reported policy score" in labels
    assert ax.get_legend().get_window_extent().y0 > ax.get_window_extent().y1
