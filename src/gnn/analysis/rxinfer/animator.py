#!/usr/bin/env python3
"""Animated HTML visualization generator for RxInfer simulation results.

Produces self-contained HTML files with JavaScript-driven animations showing
how beliefs, observations, actions, and VFE evolve over time. Each animation
frame represents one timestep (or one inference iteration for VFE), with colors
encoding probability mass, state identity, and energy values.

The generated HTML is fully self-contained — no external dependencies, no
server needed. Open the file in any browser to watch the animation.
"""

import html as html_module
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_beliefs(data: Dict[str, Any]) -> List[List[float]]:
    """Extract beliefs as a list of rows from the results dict."""
    from gnn.analysis.result_adapter import result_views

    views = result_views(data)
    if not views:
        return []
    if len(views) != 1:
        raise ValueError(
            "Multiple native marginals require separate visualization panels"
        )
    return [list(map(float, row)) for row in next(iter(views.values()))["beliefs"]]


def _normalize_observations(data: Dict[str, Any]) -> List[int]:
    obs = data.get("observations")
    if obs is None:
        obm = data.get("observations_by_modality", {})
        if isinstance(obm, dict) and obm:
            obs = next(iter(obm.values()))
    if not obs:
        return []
    return [int(x) for x in obs]


def _normalize_true_states(data: Dict[str, Any]) -> List[int]:
    ts = data.get("true_states")
    if ts is None:
        hsf = data.get("hidden_states_by_factor", {})
        if isinstance(hsf, dict) and hsf:
            ts = next(iter(hsf.values()))
    if not ts:
        return []
    return [int(x) for x in ts]


def _normalize_actions(data: Dict[str, Any]) -> List[int]:
    acts = data.get("actions")
    if acts is None:
        acf = data.get("actions_by_control_factor", {})
        if isinstance(acf, dict) and acf:
            acts = next(iter(acf.values()))
    if not acts:
        return []
    return [int(x) for x in acts]


def _normalize_vfe(data: Dict[str, Any]) -> List[float]:
    vfe = data.get("vfe_per_iteration")
    if vfe is None:
        vfe = data.get("variational_free_energy")
    if vfe is None:
        vfe = data.get("expected_free_energy")
    if not vfe:
        return []
    return [float(x) for x in vfe]


def _normalize_efe(data: Dict[str, Any]) -> List[float]:
    efe = data.get("expected_free_energy")
    if not efe:
        return []
    return [float(x) for x in efe]


def generate_animated_html(
    data: Dict[str, Any],
    output_path: Path,
    model_name: str = "model",
    title: Optional[str] = None,
) -> str:
    """Generate a self-contained animated HTML visualization.

    The animation shows:
    - Belief evolution: bar chart with colors per state, heights = probability
    - True state vs observation: highlighted cells in a grid
    - Action timeline: colored segments per action
    - VFE convergence: line plot with animated draw
    - EFE per action: stacked area with animated reveal

    Args:
        data: Simulation results dict (rxinfer_simulation_v1)
        output_path: Where to write the HTML file
        model_name: Model name for labeling
        title: Optional custom title

    Returns:
        Path to the generated HTML file
    """
    from tempfile import TemporaryDirectory

    from gnn.analysis.result_adapter import model_family, result_views

    from .family_visuals import artifact_component, continuous_html

    views = result_views(data)
    if model_family(data) == "continuous":
        return continuous_html(data, output_path, model_name)
    if len(views) > 1:
        panels = []
        with TemporaryDirectory(prefix="gnn-marginal-html-") as directory:
            for name, view in views.items():
                path = Path(directory) / f"{artifact_component(name)}.html"
                generate_animated_html(view, path, f"{model_name}: {name}")
                panels.append(
                    f'<h2>{html_module.escape(name)}</h2><iframe title="{html_module.escape(name)}" style="width:100%;height:850px;border:0" srcdoc="{html_module.escape(path.read_text())}"></iframe>'
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
            + html_module.escape(model_name)
            + "</title><body>"
            + "".join(panels)
            + "</body></html>",
            encoding="utf-8",
        )
        return str(output_path)
    if views:
        data = next(iter(views.values()))
    beliefs = _normalize_beliefs(data)
    observations = _normalize_observations(data)
    true_states = _normalize_true_states(data)
    actions = _normalize_actions(data)
    vfe = _normalize_vfe(data)
    efe = _normalize_efe(data)

    if not beliefs:
        logger.warning("No beliefs found in results — cannot generate animation")
        return ""

    n_steps = len(beliefs)
    n_states = len(beliefs[0]) if beliefs else 0
    n_actions = max(actions) + 1 if actions else 0
    n_obs = max(observations) + 1 if observations else 0

    # Color palette for states
    state_colors = []
    for i in range(n_states):
        hue = int(360 * i / max(n_states, 1))
        state_colors.append(f"hsl({hue}, 70%, 55%)")

    # Color palette for actions
    action_colors = []
    for i in range(max(n_actions, 1)):
        hue = int(360 * i / max(n_actions, 1))
        action_colors.append(f"hsl({hue}, 60%, 45%)")

    # Embed data as JSON for the animation
    animation_data = {
        "model_name": model_name,
        "n_steps": n_steps,
        "n_states": n_states,
        "n_actions": max(n_actions, 1),
        "n_obs": max(n_obs, 1),
        "beliefs": beliefs,
        "observations": observations,
        "true_states": true_states,
        "actions": actions,
        "vfe": vfe,
        "efe": efe,
        "state_colors": state_colors,
        "action_colors": action_colors,
        "belief_accuracy": data.get("validation", {}).get("belief_accuracy", None),
        "inference_converged": data.get("validation", {}).get(
            "inference_converged", None
        ),
        "uses_real_rxinfer": data.get("runtime_metadata", {}).get(
            "uses_real_rxinfer", None
        ),
        "model_kind": data.get("runtime_metadata", {}).get("model_kind", None),
    }

    html_title = title or f"RxInfer Animation — {model_name}"

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__HTML_TITLE__</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ text-align: center; color: #00d4ff; margin: 20px 0; font-size: 1.8em; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 0.9em; }}
.controls {{
    display: flex; justify-content: center; gap: 12px; margin: 20px 0;
    flex-wrap: wrap;
}}
.controls button {{
    background: #16213e; color: #00d4ff; border: 1px solid #0f3460;
    padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 1em;
    transition: all 0.2s;
}}
.controls button:hover {{ background: #0f3460; }}
.controls button.active {{ background: #0f3460; border-color: #00d4ff; }}
.slider-row {{
    display: flex; align-items: center; justify-content: center; gap: 10px;
    margin: 10px 0;
}}
.slider-row label {{ color: #aaa; font-size: 0.9em; }}
input[type=range] {{
    width: 300px; accent-color: #00d4ff;
}}
.step-display {{
    text-align: center; color: #00d4ff; font-size: 1.2em; font-weight: bold;
    margin: 10px 0;
}}
.panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
.panel {{
    background: #16213e; border: 1px solid #0f3460; border-radius: 10px;
    padding: 15px;
}}
.panel-title {{
    color: #00d4ff; font-size: 0.95em; margin-bottom: 10px;
    text-transform: uppercase; letter-spacing: 1px;
}}
canvas {{ display: block; margin: 0 auto; }}
.stats {{
    display: flex; justify-content: center; gap: 30px; margin: 15px 0;
    flex-wrap: wrap;
}}
.stat {{
    text-align: center; padding: 8px 16px;
    background: #0f3460; border-radius: 8px;
}}
.stat-label {{ color: #888; font-size: 0.75em; text-transform: uppercase; }}
.stat-value {{ color: #00d4ff; font-size: 1.3em; font-weight: bold; }}
.legend {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 8px; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 0.8em; color: #ccc; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
.full-width {{ grid-column: 1 / -1; }}
.footer {{
    text-align: center; color: #555; margin-top: 30px; font-size: 0.8em;
}}
</style>
</head>
<body>
<div class="container">
    <h1>{html_title}</h1>
    <div class="subtitle">
        Offline batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation —
        <span id="model-kind">__MODEL_KIND__</span> model,
        <span id="n-states">__N_STATES__</span> states,
        <span id="n-steps">__N_STEPS__</span> timesteps
    </div>

    <div class="stats">
        <div class="stat"><div class="stat-label">Real RxInfer</div><div class="stat-value" id="stat-rx">__USES_RX__</div></div>
        <div class="stat"><div class="stat-label">Belief Accuracy</div><div class="stat-value" id="stat-acc">__BELIEF_ACC__</div></div>
        <div class="stat"><div class="stat-label">Converged</div><div class="stat-value" id="stat-conv">__CONVERGED__</div></div>
        <div class="stat"><div class="stat-label">VFE Iterations</div><div class="stat-value" id="stat-vfe">__VFE_LEN__</div></div>
    </div>

    <div class="controls">
        <button id="btn-play" onclick="togglePlay()">▶ Play</button>
        <button id="btn-reset" onclick="reset()">⟲ Reset</button>
        <button id="btn-step-prev" onclick="stepBack()">◀ Step</button>
        <button id="btn-step-next" onclick="stepForward()">Step ▶</button>
    </div>
    <div class="slider-row">
        <label>Speed:</label>
        <input type="range" id="speed-slider" min="100" max="2000" value="500" oninput="updateSpeed()">
        <label>Step:</label>
        <input type="range" id="step-slider" min="0" max="__MAX_STEP__" value="0" oninput="manualStep()">
    </div>
    <div class="step-display">Step <span id="current-step">0</span> / <span id="total-steps">__TOTAL_STEPS__</span></div>

    <div class="panels">
        <div class="panel">
            <div class="panel-title">Belief Evolution (smoothed posteriors)</div>
            <canvas id="belief-canvas" width="540" height="300"></canvas>
            <div class="legend" id="belief-legend"></div>
        </div>
        <div class="panel">
            <div class="panel-title">True State vs Observation</div>
            <canvas id="state-canvas" width="540" height="300"></canvas>
        </div>
        <div class="panel">
            <div class="panel-title">Actions Over Time</div>
            <canvas id="action-canvas" width="540" height="250"></canvas>
        </div>
        <div class="panel">
            <div class="panel-title">VFE Convergence (per iteration)</div>
            <canvas id="vfe-canvas" width="540" height="250"></canvas>
        </div>
    </div>

    <div class="footer">
        Generated from rxinfer_simulation_v1 — offline batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation.
        Real RxInfer.jl @model + infer() with free_energy=true.
    </div>
</div>

<script>
const DATA = __DATA_JSON__;

let currentStep = 0;
let playing = false;
let playInterval = null;
let playSpeed = 500;

const beliefCanvas = document.getElementById('belief-canvas');
const beliefCtx = beliefCanvas.getContext('2d');
const stateCanvas = document.getElementById('state-canvas');
const stateCtx = stateCanvas.getContext('2d');
const actionCanvas = document.getElementById('action-canvas');
const actionCtx = actionCanvas.getContext('2d');
const vfeCanvas = document.getElementById('vfe-canvas');
const vfeCtx = vfeCanvas.getContext('2d');

// Initialize legend
function initLegend() {
    const legend = document.getElementById('belief-legend');
    legend.innerHTML = '';
    for (let i = 0; i < DATA.n_states; i++) {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<div class="legend-swatch" style="background:${{DATA.state_colors[i]}}"></div>State ${{i+1}}`;
        legend.appendChild(item);
    }
}

// Draw belief bar chart for current step
function drawBeliefs(step) {
    const ctx = beliefCtx;
    const w = beliefCanvas.width;
    const h = beliefCanvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = (h - 40) * i / 4 + 20;
        ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(w - 20, y); ctx.stroke();
    }

    if (step >= DATA.beliefs.length) return;
    const belief = DATA.beliefs[step];
    const barWidth = Math.min(60, (w - 70) / DATA.n_states - 10);
    const startX = 50 + 10;

    for (let i = 0; i < DATA.n_states; i++) {
        const x = startX + i * (barWidth + 10);
        const barH = (h - 40) * belief[i];
        const y = h - 20 - barH;

        // Bar with gradient
        const grad = ctx.createLinearGradient(x, y, x, h - 20);
        grad.addColorStop(0, DATA.state_colors[i]);
        grad.addColorStop(1, DATA.state_colors[i].replace('55%)', '25%)'));
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, barWidth, barH);

        // Value label
        ctx.fillStyle = '#ccc';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(belief[i].toFixed(3), x + barWidth/2, y - 5);

        // State label
        ctx.fillStyle = '#888';
        ctx.fillText(`S${{i+1}}`, x + barWidth/2, h - 5);
    }

    // Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const y = (h - 40) * (4 - i) / 4 + 20;
        ctx.fillText((i * 0.25).toFixed(2), 45, y + 3);
    }
}

// Draw true state vs observation
function drawStates(step) {
    const ctx = stateCtx;
    const w = stateCanvas.width;
    const h = stateCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const nSteps = DATA.beliefs.length;
    const nStates = DATA.n_states;
    const cellW = (w - 40) / Math.max(nSteps, 1);
    const cellH = (h - 60) / Math.max(nStates, 1);

    // Draw all steps as a heatmap of argmax(belief)
    for (let t = 0; t <= step && t < nSteps; t++) {
        const belief = DATA.beliefs[t];
        const argmax = belief.indexOf(Math.max(...belief));
        const trueState = DATA.true_states[t] || 0;
        const obs = DATA.observations[t] || 0;

        // Argmax belief cell
        const x = 20 + t * cellW;
        ctx.fillStyle = DATA.state_colors[argmax];
        ctx.globalAlpha = t === step ? 1.0 : 0.3;
        ctx.fillRect(x, 20, cellW - 1, cellH - 1);

        // True state cell
        ctx.fillStyle = DATA.state_colors[trueState];
        ctx.fillRect(x, 20 + cellH, cellW - 1, cellH - 1);

        // Observation cell
        const obsColor = DATA.state_colors[obs] || '#444';
        ctx.fillStyle = obsColor;
        ctx.fillRect(x, 20 + 2 * cellH, cellW - 1, cellH - 1);
    }
    ctx.globalAlpha = 1.0;

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Argmax(belief)', 5, 15);
    ctx.fillText('True state', 5, 15 + cellH);
    ctx.fillText('Observation', 5, 15 + 2 * cellH);

    // Highlight current step
    const x = 20 + step * cellW;
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x - 1, 18, cellW + 1, 3 * cellH - 1);

    // Step label
    ctx.fillStyle = '#00d4ff';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`t=${{step}}`, x + cellW/2, h - 5);
}

// Draw actions timeline
function drawActions(step) {
    const ctx = actionCtx;
    const w = actionCanvas.width;
    const h = actionCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const nSteps = DATA.actions.length;
    if (nSteps === 0) return;
    const segW = (w - 40) / Math.max(nSteps, 1);
    const segH = (h - 40) / Math.max(DATA.n_actions, 1);

    for (let t = 0; t <= step && t < nSteps; t++) {
        const action = DATA.actions[t];
        const x = 20 + t * segW;
        const y = 20;
        ctx.fillStyle = DATA.action_colors[action];
        ctx.globalAlpha = t === step ? 1.0 : 0.35;
        ctx.fillRect(x, y, segW - 1, segH - 1);
    }
    ctx.globalAlpha = 1.0;

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    for (let a = 0; a < DATA.n_actions; a++) {
        ctx.fillText(`A${{a+1}}`, 5, 20 + a * segH + segH/2);
    }

    // Current step highlight
    if (step < nSteps) {
        const x = 20 + step * segW;
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, 18, segW + 1, segH + 1);
    }
}

// Draw VFE convergence
function drawVFE(step) {
    const ctx = vfeCtx;
    const w = vfeCanvas.width;
    const h = vfeCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const vfeData = DATA.vfe;
    if (vfeData.length === 0) {
        ctx.fillStyle = '#666';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No VFE data', w/2, h/2);
        return;
    }

    const maxVFE = Math.max(...vfeData);
    const minVFE = Math.min(...vfeData);
    const range = maxVFE - minVFE || 1;
    const padL = 50, padR = 20, padT = 20, padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * i / 4;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    }

    // Draw VFE line up to current step
    const nIter = vfeData.length;
    const stepIdx = Math.min(step, nIter - 1);
    const iterW = plotW / Math.max(nIter - 1, 1);

    // Filled area
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    for (let i = 0; i <= stepIdx; i++) {
        const x = padL + i * iterW;
        const y = padT + plotH * (1 - (vfeData[i] - minVFE) / range);
        ctx.lineTo(x, y);
    }
    if (stepIdx >= 0) {
        ctx.lineTo(padL + stepIdx * iterW, padT + plotH);
    }
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, 'rgba(255, 50, 50, 0.3)');
    grad.addColorStop(1, 'rgba(255, 50, 50, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= stepIdx; i++) {
        const x = padL + i * iterW;
        const y = padT + plotH * (1 - (vfeData[i] - minVFE) / range);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Current point
    if (stepIdx >= 0) {
        const x = padL + stepIdx * iterW;
        const y = padT + plotH * (1 - (vfeData[stepIdx] - minVFE) / range);
        ctx.fillStyle = '#ff6b6b';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Value label
        ctx.fillStyle = '#ff6b6b';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(vfeData[stepIdx].toFixed(3), x, y - 10);
    }

    // Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * (4 - i) / 4;
        const val = minVFE + range * i / 4;
        ctx.fillText(val.toFixed(2), padL - 5, y + 3);
    }

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillText('Inference Iteration', w/2, h - 5);
}

function updateDisplay() {
    document.getElementById('current-step').textContent = currentStep;
    document.getElementById('step-slider').value = currentStep;
    drawBeliefs(currentStep);
    drawStates(currentStep);
    drawActions(currentStep);
    drawVFE(currentStep);
}

function togglePlay() {
    playing = !playing;
    const btn = document.getElementById('btn-play');
    if (playing) {
        btn.textContent = '⏸ Pause';
        playInterval = setInterval(() => {
            currentStep++;
            if (currentStep >= DATA.n_steps) {
                if (DATA.vfe.length > 0 && currentStep < DATA.vfe.length + DATA.n_steps - 1) {
                    // Keep going into VFE iterations
                } else {
                    currentStep = 0;
                }
            }
            updateDisplay();
        }, playSpeed);
    } else {
        btn.textContent = '▶ Play';
        clearInterval(playInterval);
    }
}

function reset() {
    currentStep = 0;
    playing = false;
    clearInterval(playInterval);
    document.getElementById('btn-play').textContent = '▶ Play';
    updateDisplay();
}

function stepForward() {
    currentStep = Math.min(currentStep + 1, DATA.n_steps - 1);
    updateDisplay();
}

function stepBack() {
    currentStep = Math.max(currentStep - 1, 0);
    updateDisplay();
}

function manualStep() {
    currentStep = parseInt(document.getElementById('step-slider').value);
    updateDisplay();
}

function updateSpeed() {
    playSpeed = parseInt(document.getElementById('speed-slider').value);
    if (playing) {
        clearInterval(playInterval);
        playInterval = setInterval(() => {
            currentStep++;
            if (currentStep >= DATA.n_steps) currentStep = 0;
            updateDisplay();
        }, playSpeed);
    }
}

// Initialize
initLegend();
updateDisplay();
</script>
</body>
</html>"""

    # Substitute template tokens with rendered values.
    html = html.replace("__DATA_JSON__", json.dumps(animation_data))
    html = html.replace("__HTML_TITLE__", html_title)
    html = html.replace(
        "__MODEL_KIND__", str(animation_data.get("model_kind", "unknown"))
    )
    html = html.replace("__N_STATES__", str(n_states))
    html = html.replace("__N_STEPS__", str(n_steps))
    html = html.replace("__MAX_STEP__", str(max(n_steps - 1, 0)))
    html = html.replace("__TOTAL_STEPS__", str(n_steps))
    html = html.replace(
        "__USES_RX__", "✓" if animation_data["uses_real_rxinfer"] else "✗"
    )
    html = html.replace(
        "__BELIEF_ACC__",
        f"{animation_data['belief_accuracy']:.1%}"
        if animation_data.get("belief_accuracy") is not None
        else "N/A",
    )
    html = html.replace(
        "__CONVERGED__", "✓" if animation_data["inference_converged"] else "✗"
    )
    html = html.replace("__VFE_LEN__", str(len(vfe)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated animated HTML: {output_path}")
    return str(output_path)


__all__: list[Any] = ["generate_animated_html"]
