"""Covariance-aware analysis for the experimental cpomdp result schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gnn.analysis.result_adapter import continuous_result_metrics
from gnn.pipeline._io import atomic_write_text


def analyze_payload(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Validate a native Gaussian trace and plot means with posterior uncertainty."""
    import matplotlib.pyplot as plt

    metrics = continuous_result_metrics(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    means = np.asarray(payload["beliefs"], dtype=float)
    cov = np.asarray(payload["posterior_cov"], dtype=float)
    sigma = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
    times = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        for index in range(means.shape[1]):
            ax.plot(times, means[:, index], label=f"State {index + 1} mean")
            ax.fill_between(
                times,
                means[:, index] - sigma[:, index],
                means[:, index] + sigma[:, index],
                alpha=0.15,
            )
        ax.set(
            xlabel="Timestep",
            ylabel="Posterior state mean",
            title="cpomdp Gaussian posterior (±1 standard deviation)",
        )
        ax.legend()
        trajectory = output_dir / "cpomdp_posterior.png"
        fig.savefig(trajectory, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    artifacts = [str(trajectory)]
    scores = np.asarray(payload.get("efe_history", []), dtype=float)
    if scores.size:
        if (
            scores.ndim != 2
            or len(scores) != len(means)
            or not np.isfinite(scores).all()
        ):
            raise ValueError(
                "cpomdp EFE scores must be finite and aligned with Gaussian timesteps"
            )
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            ax.plot(times, scores)
            ax.set(
                xlabel="Timestep",
                ylabel="Expected free energy",
                title="Declared finite policy scores",
            )
            score_path = output_dir / "cpomdp_efe.png"
            fig.savefig(score_path, dpi=150, bbox_inches="tight")
            artifacts.append(str(score_path))
        finally:
            plt.close(fig)
    summary = {
        "schema_version": 1,
        "framework": "cpomdp",
        "experimental": True,
        "model_kind": "continuous",
        "model_name": payload.get("model_name"),
        "metrics": metrics,
        "source_identity": payload.get("source_identity"),
        "admission": payload.get("admission"),
        "measured_resources": payload.get("measured_resources"),
        "unavailable_metrics": {
            "categorical_entropy": "Gaussian means are not probabilities",
            "categorical_confidence": "Gaussian uncertainty is reported from covariance",
        },
        "artifacts": artifacts,
    }
    atomic_write_text(
        output_dir / "cpomdp_analysis.json", json.dumps(summary, indent=2)
    )
    return summary


def generate_analysis_from_logs(
    results_dir: Path, output_dir: Path | None = None, verbose: bool = False
) -> list[str]:
    """Analyze each native result once; malformed payloads fail explicitly."""
    generated = []
    paths = sorted(
        set(results_dir.glob("**/cpomdp/**/simulation_results.json"))
        | set(results_dir.glob("**/cpomdp_simulation_results.json"))
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = (
            path.parent.parent.parent.name
            if path.parent.name == "simulation_data"
            else path.parent.name
        )
        destination = (output_dir or results_dir) / model / "cpomdp"
        analyze_payload(payload, destination)
        generated.append(str(destination / "cpomdp_analysis.json"))
    return generated


__all__ = ["analyze_payload", "generate_analysis_from_logs"]
