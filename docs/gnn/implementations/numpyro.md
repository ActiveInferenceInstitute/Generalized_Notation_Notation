# NumPyro Framework Implementation

The NumPyro backend has separate categorical and single-block linear-Gaussian
routes. NumPyro and JAX are core dependencies in the
[package metadata](../../../pyproject.toml).

## Categorical filtering and action selection

The categorical program embeds source-derived `A/B/C/D` values and records
matrix provenance. It samples observations and transitions with NumPyro
categorical distributions. Its belief update is a JAX tensor calculation
using the observation likelihood and predicted state prior. The declared
matrices remain fixed throughout the run.

For each candidate action it adds expected conditional observation entropy
to the KL divergence from predicted observations to `softmax(C)`. It
samples from the softmax of negative expected free energy. The result
contains `beliefs`, `observations`, `actions`, `efe_history`, and
validation findings.

## Continuous (linear-Gaussian) models

The shared continuous generator first runs an online Kalman filter using
the declared `F/H/Q/R`, Gaussian prior, and any source-defined proportional
controller. The NumPyro route then runs NUTS over the latent state trajectory
conditional on the resulting observations and controls. The system
matrices remain fixed.

The result distinguishes online filter means and covariances from
`mcmc_posterior_means`, which use the full sequence. It also records
`mcmc_r_hat_max`, `mcmc_sampling`, and `mcmc_rmse_vs_kalman`. Smoothing
and filtering condition on different information, so their mean
trajectories can differ.

## Installation

```bash
uv sync --frozen
```

## Run

From the repository root, render and execute the maintained categorical
gridworld using the prepared environment:

```bash
uv run --frozen --no-sync python src/gnn/main.py --only-steps "3,11,12" \
  --frameworks numpyro --target-dir input/gnn_files/pomdp_gridworld \
  --output-dir output/numpyro_gridworld
```

Read the current run's render and execution summaries. For model selection
and composition restrictions, see [backend admission](README.md#backend-admission).

## Source Code Connections

| Stage | Module |
|---|---|
| Categorical rendering | [NumPyro renderer](../../../src/gnn/render/numpyro/numpyro_renderer.py) |
| Continuous rendering | [Shared continuous generator](../../../src/gnn/render/continuous_script.py) |
| Execution | [NumPyro runner](../../../src/gnn/execute/numpyro/numpyro_runner.py) |
| Analysis | [NumPyro analyzer](../../../src/gnn/analysis/numpyro/analyzer.py) |
