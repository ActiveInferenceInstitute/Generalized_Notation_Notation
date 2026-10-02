"""JAX Gaussian families preserve independent posterior views in Step 16."""

from pathlib import Path

from gnn.analysis.jax.analyzer import create_visualizations_from_structured_data


def test_jax_multi_agent_gaussian_views_are_plotted_without_flat_beliefs(
    tmp_path: Path,
) -> None:
    payload = {
        "model_kind": "multi_agent_continuous",
        "agents": ["agent1", "agent2"],
        "beliefs_by_agent": {"agent1": [[2.0], [3.0]], "agent2": [[-1.0], [0.0]]},
        "posterior_cov_by_agent": {
            "agent1": [[[0.5]], [[0.25]]],
            "agent2": [[[0.75]], [[0.5]]],
        },
    }
    files = create_visualizations_from_structured_data(
        payload, tmp_path, "original-model"
    )
    assert len(files) == 2
    assert len(set(files)) == 2
    assert all(
        Path(file).is_file() and Path(file).stat().st_size > 1000 for file in files
    )
    assert all("gaussian" in Path(file).name for file in files)
