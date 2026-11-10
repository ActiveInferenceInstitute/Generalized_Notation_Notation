# PyTorch Framework Implementation

The PyTorch backend emits tensor programs for admitted categorical models
and single-block linear-Gaussian models. The runtime is the `torch` extra
declared in [package metadata](../../../pyproject.toml). Step 11 generates
code without importing PyTorch. Step 12 checks runtime readiness before
launching the generated program.

## Categorical filtering and action selection

The renderer embeds the source-derived matrices as float64 tensors and
records matrix provenance. The generated program uses fixed parameters.
At the first observation the prior is `D`. Later updates predict using
the transition slice for the previously selected action:

```text
prior_t = B[:, :, action_(t-1)] @ belief_(t-1)
belief_t = normalize(A[observation_t, :] * prior_t)
```

For each candidate action, the program predicts a next-state distribution
and its observation distribution. It adds expected conditional observation
entropy (ambiguity) to the KL divergence from predicted observations to
`softmax(C)` (risk). It samples the action from the softmax of negative
expected free energy. The result records that convention along with the
beliefs, observations, actions, and per-action scores.

## Continuous (linear-Gaussian) models

The shared continuous generator emits an online Kalman filter over the
declared dynamics, observation model, and covariances. With `goal_mean`
and `control_gain` it applies proportional control to the filtered mean.
Results include posterior means and covariances, simulated states,
observations, and controls. See [backend admission](README.md#backend-admission)
for composition restrictions.

## Installation

```bash
uv sync --frozen --extra torch
```

## Run

From the repository root, render and execute the maintained categorical
gridworld using the prepared environment:

```bash
uv run --frozen --no-sync python src/gnn/main.py --only-steps "3,11,12" \
  --frameworks pytorch --target-dir input/gnn_files/pomdp_gridworld \
  --output-dir output/pytorch_gridworld
```

Inspect the current Step 11 and Step 12 summaries for admission and
execution outcomes. A framework comparison needs the same model contract
and compatible inference and score conventions.

## Source Code Connections

| Stage | Module |
|---|---|
| Categorical rendering | [PyTorch renderer](../../../src/gnn/render/pytorch/pytorch_renderer.py) |
| Continuous rendering | [Shared continuous generator](../../../src/gnn/render/continuous_script.py) |
| Execution | [PyTorch runner](../../../src/gnn/execute/pytorch/pytorch_runner.py) |
| Analysis | [PyTorch analyzer](../../../src/gnn/analysis/pytorch/analyzer.py) |
