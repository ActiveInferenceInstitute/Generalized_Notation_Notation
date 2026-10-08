"""The public advanced POMDP processor retains canonical saved matrix axes."""

import json

import numpy as np
import pytest

from tests.advanced_visualization.test_native_network_consumers_410 import (
    figures as figures,
)
from tests.advanced_visualization.test_native_network_consumers_410 import (
    run_saved_model,
    saved_model,
)
from tests.visualization.test_native_matrix_views_410 import native_png, plot_values


@pytest.mark.parametrize("passive", [False, True])
def test_public_advanced_pomdp_artifact_preserves_authored_action_planes_and_axes(
    tmp_path, figures, passive
):
    _, parsed, inventory = saved_model(
        tmp_path, [("sensor", "belief"), ("belief", "prior")]
    )
    # Each column is a next-state probability distribution. Four distinct
    # actions distinguish the action axis from the two state dimensions.
    transition = np.array(
        [
            [[0.9, 0.4, 0.2, 0.75], [0.2, 0.7, 0.1, 0.55]],
            [[0.1, 0.6, 0.8, 0.25], [0.8, 0.3, 0.9, 0.45]],
        ]
    )
    authored = transition[:, :, 0] if passive else transition
    model = json.loads(parsed.read_text())
    model["parameters"] = [
        {"name": "B", "value": authored.tolist()},
        {"name": "E", "value": [0.1, 0.2, 0.3, 0.4]},
    ]
    parsed.write_text(json.dumps(model))
    result, _, summary = run_saved_model(tmp_path, parsed, inventory, "pomdp")
    assert result is True
    assert (summary["successful"], summary["failed"], summary["skipped"]) == (2, 0, 0)
    paths = {
        attempt["viz_type"]: attempt["output_files"][0]
        for attempt in summary["attempts"]
    }
    for path in paths.values():
        native_png(path)
    transition_figure = figures[paths["pomdp_transitions"].split("/")[-1]]
    panels = [axis for axis in transition_figure.axes if axis.get_title()]
    assert len(panels) == (1 if passive else 4)
    for action, panel in enumerate(panels):
        expected = authored if passive else transition[:, :, action]
        np.testing.assert_array_equal(plot_values(panel).reshape(2, 2), expected)
        assert panel.get_xlabel() == "Previous State"
        assert panel.get_ylabel() == "Next State"
        if not passive:
            assert panel.get_title() == f"Transition Matrix (Action {action})"
    policy_figure = figures[paths["policy"].split("/")[-1]]
    assert [bar.get_height() for bar in policy_figure.axes[0].patches] == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]


def test_public_advanced_pomdp_without_b_or_policy_reports_explicit_absence(tmp_path):
    _, parsed, inventory = saved_model(tmp_path, [("sensor", "belief")])
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "pomdp")
    assert result == 2 and result is not True
    assert (summary["successful"], summary["failed"], summary["skipped"]) == (0, 0, 2)
    assert {attempt["error_message"] for attempt in summary["attempts"]} == {
        "Not applicable to this model type (B matrix not found)",
        "No policy data found",
    }
    assert summary["output_files"] == [] and not list(output.glob("*.png"))
