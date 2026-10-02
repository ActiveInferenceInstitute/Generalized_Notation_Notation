# Experimental THRML categorical rendering

THRML supplies native categorical factor/block Gibbs sampling for selected
GNN models. The GNN adapter targets released `thrml==0.1.4` and uses local
JAX simulation. It supports positive finite categorical A/B/C/D models,
optional E, independent complete agent/factor groups, bounded exact joint
factor composition, and conditionally independent observation modalities.

```python
from pathlib import Path
from gnn.render.thrml import render_gnn_to_thrml

success, message, artifacts = render_gnn_to_thrml(
    spec,
    Path("model_thrml.py"),
    {
        "num_timesteps": 3,
        "observations": [0, 1, 0],
        "transition_actions": [1, 0],
        "num_samples": 2048,
        "burn_in": 256,
        "thin": 2,
        "seed": 0,
    },
)
assert success, message
```

Use the repository's configured render/execute/analysis surfaces for managed
runs, current-run artifact identities, deadlines, and result validation. The
generated script delegates to the installed `gnn.render.thrml.runtime`:

```bash
uv sync --extra thrml
uv run python model_thrml.py --output-json thrml_results.json
```

Direct script execution records its script hash but has no managed execution
token unless the executor supplies `GNN_THRML_EXECUTION_ID`. A standalone
artifact therefore does not establish verified membership in a new run.

Defaults are 10 timesteps (or the model's declared duration), 2,048 retained
samples, 256 burn-in sweeps, thinning 2, and seed 0. Observations may be
provided; otherwise a native THRML joint Gibbs draw supplies synthetic data
with explicit provenance. Transition actions default to action 0. Actions
have length `T-1`, matching state-transition edges. Multi-component data and
actions require mappings that bind every component ID exactly.

The posterior conditions on the entire observation sequence and fixed
actions. Predictive observations describe a replicate observation drawn
from the empirical smoothed state distribution. Preferences C and habits E
are retained; this implementation performs no EFE search. Gibbs sampling
does not establish exact inference or convergence. Zero-probability support,
unrepresented coupled dynamics, and continuous/nonstationary models are
explicitly unsupported. Malformed probabilities and invalid options fail.

See [SPEC](SPEC.md) for schemas/admission and the
[implementation guide](../../../../docs/gnn/implementations/thrml.md) for
CLI, API, MCP, configuration, and verification. The
[upstream documentation](https://docs.thrml.ai/en/latest/) describes THRML's
native API.
