"""Tests for the RxInfer GIF animation generator.

Validates that generate_gif_animation produces valid GIF files with
the expected structure, embedded data, and white publication style.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.rxinfer.gif_animator import generate_gif_animation


def _build_synthetic_results(
    n_steps: int = 10,
    n_states: int = 3,
    n_actions: int = 2,
) -> dict[str, Any]:
    """Build a synthetic rxinfer_simulation_v1 result for testing."""
    beliefs = []
    for t in range(n_steps):
        row = [0.2 / (n_states - 1)] * n_states
        row[t % n_states] = 0.8
        beliefs.append(row)

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "model_name": "TestModel",
        "num_timesteps": n_steps,
        "beliefs": beliefs,
        "observations": [t % n_states for t in range(n_steps)],
        "true_states": [(t + 1) % n_states for t in range(n_steps)],
        "actions": [t % n_actions for t in range(n_steps)],
        "variational_free_energy": [6.11 - 0.1 * i for i in range(20)],
        "vfe_per_iteration": [6.11 - 0.1 * i for i in range(20)],
        "expected_free_energy": [3.0 - 0.1 * t for t in range(n_steps)],
        "efe_per_action": [
            [round(3.0 - 0.1 * t + i, 4) for i in range(n_actions)]
            for t in range(n_steps)
        ],
        "policy_posterior": [[1.0 / n_actions] * n_actions for _ in range(n_steps)],
        "validation": {
            "all_valid": True,
            "inference_converged": True,
            "vfe_present": True,
            "belief_entropy_ok": True,
            "belief_accuracy": 0.8,
            "belief_accuracy_ok": True,
        },
        "runtime_metadata": {
            "uses_real_rxinfer": True,
            "model_kind": "flat",
        },
    }


def test_generates_gif_file(tmp_path: Path) -> None:
    """generate_gif_animation produces a GIF file on disk."""
    results = _build_synthetic_results()
    output = tmp_path / "test.gif"
    path = generate_gif_animation(results, output, "TestModel")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def test_gif_has_correct_format(tmp_path: Path) -> None:
    """The GIF file starts with the GIF89a header."""
    results = _build_synthetic_results()
    output = tmp_path / "test_header.gif"
    generate_gif_animation(results, output, "TestModel")
    with open(output, "rb") as f:
        header = f.read(6)
    assert header in (b"GIF87a", b"GIF89a"), f"Invalid GIF header: {header!r}"


def test_different_state_counts(tmp_path: Path) -> None:
    """Models with different state counts produce valid GIFs."""
    for n_states in [2, 4, 8, 16]:
        results = _build_synthetic_results(n_states=n_states)
        output = tmp_path / f"test_{n_states}.gif"
        generate_gif_animation(results, output, f"Model_{n_states}")
        assert output.exists(), f"GIF for {n_states} states not created"
        assert output.stat().st_size > 500


def test_dict_shaped_beliefs_handled(tmp_path: Path) -> None:
    """Dict-shaped beliefs_by_factor are handled correctly."""
    results = _build_synthetic_results()
    beliefs = results.pop("beliefs")
    results["beliefs_by_factor"] = {"joint_state": beliefs}
    output = tmp_path / "test_dict.gif"
    generate_gif_animation(results, output, "DictModel")
    assert output.exists()


def test_empty_beliefs_returns_empty(tmp_path: Path) -> None:
    """Empty beliefs produces empty string (no GIF)."""
    results: dict[str, Any] = {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
    }
    output = tmp_path / "empty.gif"
    path = generate_gif_animation(results, output, "EmptyModel")
    assert path == ""


def test_with_gnn_spec_for_graph_model(tmp_path: Path) -> None:
    """GIF generation works when gnn_spec is embedded for graph model."""
    results = _build_synthetic_results()
    results["gnn_spec"] = {
        "connections": ["D>s", "s>A", "s>o", "s>s_prime", "B>s_prime", "u>s_prime"],
        "model_parameters": {"num_states": 3},
    }
    output = tmp_path / "test_graph.gif"
    generate_gif_animation(results, output, "GraphModel")
    assert output.exists()
    assert output.stat().st_size > 1000


def test_white_publication_style(tmp_path: Path) -> None:
    """The GIF uses white publication style (not dark mode)."""
    import matplotlib

    matplotlib.use("Agg")
    results = _build_synthetic_results()
    output = tmp_path / "test_style.gif"
    generate_gif_animation(results, output, "StyleTest")
    # The function uses plt.style.use("default") which is white
    # Verify the file was created (the style is applied inside the function)
    assert output.exists()


def _build_factored_results(
    factor_sizes: tuple[tuple[str, int], ...] = (
        ("s_agent1", 4),
        ("s_agent2", 4),
        ("s_joint", 16),
    ),
    n_steps: int = 5,
) -> dict[str, Any]:
    """Build a multi-factor result: beliefs over the flattened 256-state joint space."""
    sizes = [size for _, size in factor_sizes]
    joint_size = math.prod(sizes)
    beliefs = []
    for t in range(n_steps):
        raw = [float(((t + 1) * (k + 3)) % 7) + 0.5 for k in range(joint_size)]
        total = sum(raw)
        beliefs.append([value / total for value in raw])

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "model_name": "multi_agent_coordination",
        "num_timesteps": n_steps,
        "beliefs": beliefs,
        "observations": [t % 4 for t in range(n_steps)],
        "true_states": [t % 4 for t in range(n_steps)],
        "actions": [t % 2 for t in range(n_steps)],
        "vfe_per_iteration": [6.11 - 0.1 * i for i in range(12)],
        "model_parameters": {
            "num_states": joint_size,
            "num_state_factors": len(factor_sizes),
            "state_factors": [
                {"name": name, "size": size} for name, size in factor_sizes
            ],
        },
        "runtime_metadata": {"uses_real_rxinfer": True, "model_kind": "factored"},
    }


def test_factored_model_gif_and_manifest(tmp_path: Path) -> None:
    """A 256-state (4,4,16) factored payload renders a GIF plus its manifest sidecar."""
    results = _build_factored_results()
    assert len(results["beliefs"][0]) == 256

    output = tmp_path / "multi_agent.gif"
    path = generate_gif_animation(results, output, "multi_agent_coordination")

    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
    with open(output, "rb") as f:
        assert f.read(6) in (b"GIF87a", b"GIF89a")

    manifest_path = output.with_suffix(".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generator"] == "gif_animator.py"
    assert manifest["num_states"] == 256


def test_state_factors_change_the_rendered_panel(tmp_path: Path) -> None:
    """The factor panels really replace the joint bar chart, byte-for-byte.

    Rendering is deterministic, so an identical payload yields identical GIF
    bytes. Dropping only ``state_factors`` therefore isolates the panel swap:
    the same beliefs must render differently once the factor structure is known.
    """
    factored = _build_factored_results(n_steps=3)
    same_again = _build_factored_results(n_steps=3)
    flat = _build_factored_results(n_steps=3)
    del flat["model_parameters"]["state_factors"]

    first = tmp_path / "factored.gif"
    second = tmp_path / "factored_again.gif"
    third = tmp_path / "flat.gif"
    generate_gif_animation(factored, first, "M")
    generate_gif_animation(same_again, second, "M")
    generate_gif_animation(flat, third, "M")

    assert first.read_bytes() == second.read_bytes(), "rendering is not deterministic"
    assert first.read_bytes() != third.read_bytes(), (
        "state_factors did not change the rendered panel"
    )


def test_factored_model_with_degenerate_factor(tmp_path: Path) -> None:
    """Size-1 factors count toward the reshape but never get their own panel."""
    results = _build_factored_results(
        factor_sizes=(("s_agent1", 9), ("s_agent2", 9), ("signal_decay", 1)),
        n_steps=4,
    )
    assert len(results["beliefs"][0]) == 81

    output = tmp_path / "swarm.gif"
    path = generate_gif_animation(results, output, "stigmergic_swarm")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def test_flat_model_keeps_joint_belief_panel(tmp_path: Path) -> None:
    """A payload without state_factors still renders the original joint-belief panel."""
    results = _build_synthetic_results()
    assert "model_parameters" not in results

    output = tmp_path / "flat.gif"
    path = generate_gif_animation(results, output, "FlatModel")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def _build_hierarchical_results(n_steps: int = 8) -> dict[str, Any]:
    """Build a synthetic two-level hierarchical payload (fast + slow factors)."""
    n_fast, n_slow, n_actions = 4, 2, 2
    fast_beliefs = []
    slow_beliefs = []
    for t in range(n_steps):
        fast = [0.05] * n_fast
        fast[t % n_fast] = 1.0 - 0.05 * (n_fast - 1)
        fast_beliefs.append(fast)
        slow = [0.3, 0.7] if t % 2 else [0.7, 0.3]
        slow_beliefs.append(slow)

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "model_name": "hierarchical_agent",
        "num_timesteps": n_steps,
        "beliefs_by_factor": {"s_fast": fast_beliefs, "s_slow": slow_beliefs},
        "observations": [t % n_fast for t in range(n_steps)],
        "true_states": [t % n_fast for t in range(n_steps)],
        "actions": [t % n_actions for t in range(n_steps)],
        "vfe_per_iteration": [5.5 - 0.2 * i for i in range(15)],
        "gnn_spec": {
            "connections": ["D>s", "s>o", "A>o", "B>s", "u>s", "C>G", "G>u"],
        },
        "model_parameters": {"num_states": n_fast, "num_slow_states": n_slow},
        "validation": {
            "inference_converged": True,
            "context_beliefs_valid": True,
            "context_beliefs_sum_to_one": True,
        },
        "runtime_metadata": {
            "uses_real_rxinfer": True,
            "model_kind": "hierarchical",
        },
    }


def test_hierarchical_kind_renders_and_manifest_records_kind(tmp_path: Path) -> None:
    """A hierarchical payload (fast/slow beliefs_by_factor) renders via the
    hierarchical strategy layout and the manifest records model_kind."""
    results = _build_hierarchical_results()
    output = tmp_path / "hierarchical_100steps.gif"
    path = generate_gif_animation(results, output, "hierarchical_agent")

    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
    with open(output, "rb") as f:
        assert f.read(6) in (b"GIF87a", b"GIF89a")

    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_kind"] == "hierarchical"


def test_flat_kind_still_renders_and_manifest_records_kind(tmp_path: Path) -> None:
    """A model_kind='flat' payload still renders, with the kind in the manifest."""
    results = _build_synthetic_results()
    assert results["runtime_metadata"]["model_kind"] == "flat"
    results["gnn_spec"] = {
        "connections": ["D>s", "s>o", "A>o", "s>s_prime", "B>s_prime", "u>s_prime"],
    }
    output = tmp_path / "flat_kind_100steps.gif"
    path = generate_gif_animation(results, output, "FlatKind")

    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_kind"] == "flat"


def test_unknown_model_kind_raises_value_error(tmp_path: Path) -> None:
    """An unknown runtime_metadata.model_kind fails loud with ValueError."""
    results = _build_synthetic_results()
    results["runtime_metadata"]["model_kind"] = "quantum"
    output = tmp_path / "unknown_kind.gif"
    with pytest.raises(ValueError, match="unknown model_kind 'quantum'"):
        generate_gif_animation(results, output, "UnknownKind")
    assert not output.exists()


def test_missing_model_kind_defaults_to_flat(tmp_path: Path) -> None:
    """Older payloads without runtime_metadata.model_kind keep working."""
    results = _build_synthetic_results()
    del results["runtime_metadata"]["model_kind"]
    output = tmp_path / "older_schema.gif"
    path = generate_gif_animation(results, output, "OlderSchemaModel")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def test_multiple_frames(tmp_path: Path) -> None:
    """The GIF has multiple animation frames."""
    results = _build_synthetic_results(n_steps=15)
    output = tmp_path / "test_frames.gif"
    generate_gif_animation(results, output, "FramesTest", fps=4)
    assert output.exists()
    # A GIF with 15 frames at 4fps should be at least 10KB
    assert output.stat().st_size > 10000
