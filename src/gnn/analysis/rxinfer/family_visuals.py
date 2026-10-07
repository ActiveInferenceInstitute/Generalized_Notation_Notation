"""Gaussian and native marginal visualizations without categorical stand-ins."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image
from PIL.GifImagePlugin import GifImageFile

from gnn.analysis.result_adapter import continuous_result_metrics, model_family


def artifact_component(name: str) -> str:
    """Stable file component, including disambiguation after sanitization."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "trace"
    return f"{clean}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"


def _gaussian_figure(data: dict[str, Any], model_name: str) -> tuple[Any, Any]:
    continuous_result_metrics(data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{model_name}: Gaussian posterior")
    return fig, axes


def _unit_label(data: dict[str, Any], quantity: str) -> str:
    """Display declared result units without inferring physical quantities."""
    units = data.get("units", {})
    unit = units.get(quantity) if isinstance(units, dict) else None
    if isinstance(unit, str) and unit.strip():
        return unit.strip()
    return "units unspecified"


def _draw_gaussian(axes: Any, data: dict[str, Any], count: int) -> None:
    for ax in axes:
        ax.clear()
    means = np.asarray(data["beliefs"], dtype=float)[:count]
    cov = np.asarray(data["posterior_cov"], dtype=float)[:count]
    x = np.arange(count)
    std = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
    true = data.get("true_states_continuous")
    for state in range(means.shape[1]):
        (line,) = axes[0].plot(x, means[:, state], label=f"State {state}: mean")
        axes[0].fill_between(
            x,
            means[:, state] - 1.96 * std[:, state],
            means[:, state] + 1.96 * std[:, state],
            alpha=0.2,
            color=line.get_color(),
        )
        if true is not None and len(true):
            axes[0].plot(
                x,
                np.asarray(true)[:count, state],
                "--",
                color=line.get_color(),
                label=f"State {state}: truth",
            )
    axes[0].set(
        title="Means and 95% Gaussian intervals",
        xlabel="Timestep",
        ylabel=f"State value ({_unit_label(data, 'state')})",
    )
    axes[0].legend(fontsize=7)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    controls = data.get("controls", [])
    if len(controls):
        values = np.asarray(controls)[:count]
        if values.ndim == 1:
            values = values[:, None]
        styles = (("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"))
        for column in range(values.shape[1]):
            marker, linestyle = styles[column % len(styles)]
            axes[1].plot(
                np.arange(len(values)),
                values[:, column],
                label=f"Control {column}",
                marker=marker,
                linestyle=linestyle,
                markersize=3,
            )
        axes[1].set(
            title="Reported controls",
            xlabel="Timestep",
            ylabel=f"Control value ({_unit_label(data, 'control')})",
        )
        axes[1].legend(fontsize=7)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[1].text(
            0.5, 0.5, "Controls not reported", ha="center", transform=axes[1].transAxes
        )
    vfe = data.get("vfe_per_iteration", data.get("variational_free_energy", []))
    if len(vfe):
        axes[2].plot(np.arange(len(vfe)), vfe)
        axes[2].set(
            title="Inference convergence", xlabel="Inference iteration", ylabel="VFE"
        )
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[2].text(
            0.5, 0.5, "VFE not reported", ha="center", transform=axes[2].transAxes
        )
        axes[2].set_axis_off()


def continuous_png(data: dict[str, Any], output_path: Path, model_name: str) -> str:
    """Plot the Gaussian trajectory and its actual uncertainty."""
    fig, axes = _gaussian_figure(data, model_name)
    try:
        _draw_gaussian(axes, data, len(data["beliefs"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=100)
    finally:
        plt.close(fig)
    return str(output_path)


def _manifest(data: dict[str, Any], path: Path, views: list[str]) -> None:
    rt = data.get("runtime_metadata", {})
    record = {
        "model_kind": model_family(data),
        "views": views,
        "timesteps": data.get("num_timesteps"),
        "seed": rt.get("random_seed"),
        "julia_version": rt.get("julia_version"),
        "rxinfer_version": rt.get("rxinfer_version"),
        "gnn_spec_sha256": hashlib.sha256(
            json.dumps(data.get("gnn_spec", {}), sort_keys=True).encode()
        ).hexdigest(),
        "generator": "family_visuals.py",
    }
    if model_family(data) == "continuous":
        record["metrics"] = continuous_result_metrics(data)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def continuous_gif(
    data: dict[str, Any], output_path: Path, model_name: str, fps: int, dpi: int
) -> str:
    """Animate Gaussian posterior means and intervals over reported timesteps."""
    fig, axes = _gaussian_figure(data, model_name)
    try:

        def draw(frame: int) -> list[Any]:
            _draw_gaussian(axes, data, frame + 1)
            return []

        anim = animation.FuncAnimation(
            fig, draw, frames=len(data["beliefs"]), repeat=True
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    _manifest(data, output_path, ["continuous"])
    return str(output_path)


def marginal_gif(
    data: dict[str, Any],
    views: dict[str, dict[str, Any]],
    output_path: Path,
    model_name: str,
    fps: int,
    dpi: int,
    generator: Callable[..., str],
) -> str:
    """Synchronize complete per-agent/factor GIFs without a fabricated joint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    with TemporaryDirectory(prefix="gnn-marginals-") as directory:
        images: list[GifImageFile] = []
        try:
            for index, (name, view) in enumerate(views.items()):
                path = Path(directory) / f"view_{index}.gif"
                generator(view, path, f"{model_name}: {name}", fps=fps, dpi=dpi)
                opened = Image.open(path)
                if not isinstance(opened, GifImageFile):
                    opened.close()
                    raise ValueError("Marginal animation must produce a GIF image")
                images.append(opened)
            count = max(image.n_frames for image in images)
            for index in range(count):
                panels = []
                for image in images:
                    image.seek(min(index, image.n_frames - 1))
                    panels.append(image.convert("RGB"))
                frame = Image.new(
                    "RGB",
                    (
                        max(panel.width for panel in panels),
                        sum(panel.height for panel in panels),
                    ),
                    "white",
                )
                top = 0
                for panel in panels:
                    frame.paste(panel, (0, top))
                    top += panel.height
                frames.append(frame)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0,
            )
        finally:
            for image in images:
                image.close()
            for frame in frames:
                frame.close()
    _manifest(data, output_path, list(views))
    return str(output_path)


def continuous_html(data: dict[str, Any], output_path: Path, model_name: str) -> str:
    """Self-contained animated Gaussian report using the same result contract."""
    with TemporaryDirectory(prefix="gnn-gaussian-") as directory:
        path = Path(directory) / "posterior.gif"
        continuous_gif(data, path, model_name, fps=4, dpi=75)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(model_name)}</title><h1>{html.escape(model_name)}: Gaussian posterior</h1><p>Means and 95% Gaussian intervals; VFE is indexed by inference iteration.</p><img alt="Gaussian posterior animation" src="data:image/gif;base64,{encoded}"></html>',
        encoding="utf-8",
    )
    return str(output_path)
