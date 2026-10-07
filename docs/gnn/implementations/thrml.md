# THRML implementation backend

THRML is an explicitly selected, experimental GNN backend using the ordinary
released `thrml==0.1.4` wheel. GNN maps admitted finite categorical models to
native THRML factor programs and estimates their posterior trajectories with
categorical Gibbs sampling. This implementation runs through JAX on a local
computer. It does not establish execution on Extropic hardware, energy savings,
exact inference, Gibbs convergence, or Active Inference action optimization.

Upstream references: [THRML documentation](https://docs.thrml.ai/en/latest/),
[getting started](https://docs.thrml.ai/en/latest/getting-started.html), and
[released package](https://pypi.org/project/thrml/0.1.4/). GNN's contract is
deliberately narrower than the upstream library's general sampling interface.

## Install and select

```bash
uv sync --extra thrml
uv run --extra thrml gnn render input/gnn_files/thrml/categorical_smoothing.md \
  --framework thrml --output /tmp/gnn-thrml-example.py \
  --options '{"observations":[0,1,0],"transition_actions":[1,0]}' --json
```

To select THRML through the numbered pipeline, pass a directory containing the
models, select the framework explicitly, and use a dedicated output directory:

```bash
uv run --extra thrml python -m gnn.main \
  --target-dir input/gnn_files/thrml --output-dir /tmp/gnn-thrml-run \
  --only-steps "3,11,12,16" --frameworks thrml --skip-llm
```

THRML is excluded from the default framework set. Rendering does not import
THRML or JAX. Planning, doctor, and execution use the shared bounded,
interpreter-specific readiness probe; installation remains an explicit action.
An unavailable package is a readiness diagnosis, whereas an unsupported model
is a scientific mapping refusal. Neither produces verified execution evidence.

## Model and inference contract

The flat model requires finite `A[observations,states]`, canonical
`B[next_state,previous_state,action]`, `D[states]`, and finite log-preferences
`C[observations]`; optional `E[actions]` is retained. A, B, and D must be strictly
positive with valid probability mass. GNN does not clip structural zeros or
repair malformed probability mass through unconditional normalization.
Declared dimensions and B-axis provenance must agree with the values.

For a trajectory of length T, the posterior conditions on all T observations
and the T−1 supplied transition actions. Unary prior, observation likelihood,
and adjacent-state transition factors retain their numerical values. Native
THRML block schedules update alternating hidden-state sites while observations
remain clamped. The reported beliefs are empirical marginals of the retained
state samples. This is full-sequence smoothing, so it must not be compared as
though it were an online filtering trajectory.

If observations are absent, a native THRML joint Gibbs draw supplies a synthetic
observation sequence and records that origin. If transition actions are absent,
the explicitly recorded default is fixed action 0 at each transition. C and E
remain source preferences and habits; they do not silently become an EFE policy
optimizer or an action-selection result.

Supported composition has an explicit binding:

- Complete independent `A_agentN/B_agentN/C_agentN/D_agentN[/E_agentN]` or
  `A_fN/B_fN/C_fN/D_fN[/E_fN]` groups retain separate component identities,
  samples, observations, and marginals.
- Admitted dependent state-factor tables use the bounded canonical joint
  composer and retain the factor descriptors and axis mappings.
- Declared conditionally independent `A_mN` observation modalities over one
  shared state use a bounded Cartesian observation composition with their C_m
  preferences and modality descriptors retained.

Cross-agent coupling, omitted matrix semantics, ambiguous component transition
orientation, structural zeros, continuous/hybrid/nonstationary models, and
action optimization produce an unsupported result. Resource limits apply before
Cartesian materialization and again to the aggregate admitted components.

## Configuration and public interfaces

Resolved pipeline configuration freezes these options with the run selection:

```yaml
render:
  backend_options:
    thrml:
      num_samples: 2048
      burn_in: 256
      thin: 2
      seed: 0
      # num_timesteps: 3
      # observations: [0, 1, 0]
      # transition_actions: [1, 0]
```

| Option | Default | Validation |
|---|---|---|
| `num_timesteps` | Model's declared value, otherwise 10 | Positive integer |
| `num_samples` | 2048 | Positive integer |
| `burn_in` | 256 | Nonnegative integer |
| `thin` | 2 | Positive integer |
| `seed` | 0 | Integer in 0 through 2³²−1 |
| `observations` | Synthetic native joint draw | Exactly T integer observation indices |
| `transition_actions` | Fixed action 0 | Exactly T−1 integer action indices |

Boolean and fractional integer options, unknown options, invalid category
indices, and over-budget requests are rejected. For multiple independent
components, supplied observations and actions must be mappings whose keys bind
every selected component exactly. Direct option values override configured
backend values; `simulation_params.thrml` supplies explicit command-layer
overrides. Requested work is never silently reduced to fit admission limits.

The shared Python rendering entry point is:

```python
from pathlib import Path
from gnn import parse_gnn_file
from gnn.render.processor import render_gnn_spec

spec = parse_gnn_file(Path("input/gnn_files/thrml/categorical_smoothing.md"))
ok, message, artifacts = render_gnn_spec(
    spec,
    "thrml",
    Path("/tmp/gnn-thrml-render"),
    options={
        "observations": [0, 1, 0],
        "transition_actions": [1, 0],
        "num_samples": 2048,
    },
)
```

The direct leaf API exports `render_gnn_to_thrml`, `build_thrml_payload`, and
`UnsupportedTHRMLModel` from `gnn.render.thrml`. Supervised execution uses
`gnn.execute.thrml.execute_thrml_script`; analysis uses
`gnn.analysis.thrml.adapt_result` and `analyze_payload`. Those boundaries separate
pure validation/admission, native sampling, process ownership, and scientific
result interpretation.

`POST /api/v1/render` accepts `framework: "thrml"` and an `options` object with
the same keys as the single-file CLI. The exact-framework MCP tool
`render_spec_to_format` also accepts `framework: "thrml"` and `options`.
The older MCP `render_gnn_to_format` directory wrapper treats its framework
argument as a hint; use `render_spec_to_format` for exact experimental selection.
Framework inventory comes from the live render and executor registries.

## Admission, execution, and artifacts

The current experimental admission caps are 16 components, 4096 state or
observation categories, 1,048,576 factor entries or retained source matrix
entries, 4,194,304 site updates, and 128 MiB estimated tensor allocation.
Retained source admission includes every declared action slice of B, even when
the fixed trajectory selects only a subset of actions. Estimates are recorded as estimates,
without claiming an allocation upper bound. JAX runtime/compilation overhead is
separate; the supervised process watchdog is authoritative for the deadline.
Finite positive execution ceilings, remaining run budget, cancellation,
result retrieval, and cleanup share the execution deadline. A child that exits
successfully after required work becomes unfinished cannot make the run succeed.

Step 12 retains the native JSON at
`<artifact-stem>/thrml/simulation_data/simulation_results.json` inside
its execution output. Direct execution uses the same `simulation_data/` layout
inside its leased output directory. Each result binds a fresh execution ID and
the executed script's SHA-256. Pipeline rendering additionally binds the frozen
model ID, relative source path, and source hash. Historical JSON cannot satisfy
that fresh execution binding.

Schema version 1 retains canonical/source parameters and matrix provenance,
sampling options, dependency versions, semantic/source identity, fixed
transition actions, observation origin, samples, measured resources, admission,
and explicit limitations. The result adapter verifies integer samples and
indices, dimensions, mass, dependency identity, empirical marginals against
`sample_states[num_samples,T]`, and replicate observation predictions against
`beliefs @ A.T`. Independent components retain separate records instead of a
fabricated joint posterior.

Analysis writes readable posterior/predictive PNGs and `thrml_analysis.json`.
Its entropy and maximum probability describe the empirical posterior only.
VFE, EFE, effective sample size, convergence, calibration, and continuous
covariance remain absent with reasons; arbitrary fields claiming those metrics
are rejected by the validated result contract.

## Verification and evidence limits

```bash
uv run --extra dev python -m pytest \
  tests/render/test_thrml_contracts.py tests/render/test_thrml_surfaces.py \
  tests/execute/test_thrml_runner.py tests/execute/test_thrml_step_contract.py \
  tests/analysis/test_thrml_analysis.py -q
uv run --extra dev --extra thrml python -m pytest \
  tests/render/test_thrml_native.py -q
```

Native acceptance must identify the ordinary released wheel, Python/JAX split,
generated script, source parameters, seed, samples, and result validation. An
asymmetric model is compared with independent NumPy log-space
forward/backward smoothing using a stated Monte Carlo tolerance; matching that
fixture does not prove universal equivalence or convergence. Structural-zero
refusals protect against the released stock sampler's inactive-padding and
all-invalid-logit hazards. Installed-wheel acceptance runs outside the checkout
without `PYTHONPATH` and exercises the real pipeline route.

The [scope ledger](../../../SCOPE-2026-10-01.md) and
[verification ledger](../../development/verification_2026_10_01.json) distinguish
library probes, generated/native acceptance, installed-package checks, hosted
CI, and remaining release custody. Consult those receipts before making an
operational claim about an environment or model.
