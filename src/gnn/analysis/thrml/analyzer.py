"""Readable artifacts for validated empirical THRML categorical traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gnn.frameworks import ALL_FRAMEWORKS
from gnn.pipeline._io import atomic_write_text

from .adapter import adapt_result


def analyze_payload(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Publish categorical smoothing and predictive plots with honest metrics."""
    summary = adapt_result(payload)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    components = payload.get("components") or {"joint_state": payload}
    for name, component in components.items():
        traces = [
            (
                "beliefs",
                "THRML empirical Gibbs smoothing posterior",
                component["beliefs"],
            ),
            (
                "predictive_observations",
                "Replicate observation prediction given smoothed states",
                component["predictive_observations"],
            ),
        ]
        for field, title in (
            ("state_factor_marginals", "Empirical Gibbs state factor marginal"),
            (
                "observation_modality_marginals",
                "Replicate observation modality marginal",
            ),
        ):
            for label, values in component.get(field, {}).items():
                traces.append((f"{field}:{label}", f"{title}: {label}", values))
        for field, title, trace in traces:
            prefix = hashlib.sha256(f"{name}:{field}".encode()).hexdigest()[:12]
            values = np.asarray(trace, dtype=float)
            fig, ax = plt.subplots(figsize=(9, 5))
            try:
                for index in range(values.shape[1]):
                    ax.plot(
                        np.arange(len(values)),
                        values[:, index],
                        label=f"Category {index}",
                    )
                ax.set(
                    xlabel="Timestep",
                    ylabel="Probability",
                    title=f"{title}: {name}",
                    ylim=(0, 1),
                )
                ax.legend()
                artifact = output_dir / f"thrml_{prefix}.png"
                fig.savefig(artifact, dpi=150, bbox_inches="tight")
                artifacts.append(str(artifact))
            finally:
                plt.close(fig)
    summary["artifacts"] = artifacts
    atomic_write_text(output_dir / "thrml_analysis.json", json.dumps(summary, indent=2))
    return summary


def generate_analysis_from_logs(
    results_dir: Path, output_dir: Path | None = None, verbose: bool = False
) -> list[str]:
    """Analyze each selected native payload once; malformed artifacts fail."""
    paths = sorted(
        set(results_dir.glob("**/simulation_data/simulation_results.json"))
        | set(results_dir.glob("**/thrml/**/simulation_results.json"))
        | set(results_dir.glob("**/thrml_simulation_results.json"))
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for path in paths:
        # Skip known foreign framework directories before reading their data.
        framework_directory = (
            path.parent.parent.name
            if path.parent.name == "simulation_data"
            else path.parent.name
        )
        if framework_directory in ALL_FRAMEWORKS and framework_directory != "thrml":
            continue
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(results_dir.resolve())
            or not path.is_file()
            or path.stat().st_size > 16 * 1024 * 1024
        ):
            raise ValueError("THRML analysis result is unsafe or exceeds 16 MiB")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("framework") != "thrml":
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("source_identity"), dict
        ):
            raise ValueError(
                "THRML analysis requires a result object with source identity"
            )
        identity = payload.get("source_identity", {})
        # Prefer the stable artifact identity when available, never a display name.
        relative = path.relative_to(results_dir).as_posix()
        fallback = f"{path.parent.parent.name}_{hashlib.sha256(relative.encode()).hexdigest()[:16]}"
        model = identity.get("artifact_stem") or identity.get("model_id") or fallback
        if (
            not isinstance(model, str)
            or Path(model).name != model
            or model in {".", ".."}
        ):
            raise ValueError(
                "THRML analysis artifact identity must be a safe path segment"
            )
        if model in seen:
            raise ValueError(
                "THRML analysis contains duplicate model artifact identity"
            )
        seen.add(model)
        # Preflight every selected scientific witness before publishing plots.
        adapt_result(payload)
        selected.append((model, payload))
    generated: list[str] = []
    for model, payload in selected:
        destination = (output_dir or results_dir) / model / "thrml"
        analyze_payload(payload, destination)
        generated.append(str(destination / "thrml_analysis.json"))
    return generated
