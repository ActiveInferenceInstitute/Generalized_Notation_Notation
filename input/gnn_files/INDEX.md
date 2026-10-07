# GNN Examples Index

Cold-start index of the exemplar GNN spec files under `input/gnn_files/`. Each
entry is a runnable Active Inference generative model spec: parse it, render it,
execute it through the 25-step pipeline. For syntax and file-structure rules see
[normative syntax](../../docs/gnn/reference/gnn_syntax.md) and the tutorials in
[docs/gnn/tutorials/](../../docs/gnn/tutorials/).

**Counts (measured 2026-10-07):** 38 model `.md` sources across 11 task
folders, excluding `INDEX.md`, `AGENTS.md` and `README.md` scaffolds. Backend
admission depends on each declared model contract; a registry entry alone does
not establish that every exemplar runs on that backend. The six corrected
scientific examples have source-preserving numerical/native acceptance documented
in [the scientific receipt](../../docs/development/issue250_scientific_acceptance_2026_10_07.json).
The block-reset hierarchy, approximate temporal controller and contingent episodic
T-maze require explicit JAX execution; other backends report them unsupported.
The canonical [family manifest](../model_family_manifest.json) selects these
contracts accordingly. Live per-backend counts come from the current run's
`output/11_render_output/render_processing_summary.json`.

## Choosing an example

| If you want to… | Start with |
| --- | --- |
| Learn GNN syntax from scratch | `basics/static_perception.md` → `basics/dynamic_perception.md` |
| Run a minimal discrete-state agent | `discrete/simple_mdp.md` → `discrete/two_state_bistable.md` |
| See a canonical full Active Inference agent | `discrete/actinf_pomdp_agent.md` |
| Compare render targets / scaling | `pymdp_scaling_study/pymdp_scaling_N4_T100.md` (then N8…N64) |
| Continuous-state (linear-Gaussian) models — passive filtering | `continuous/damped_oscillator_bias.md`, `continuous/ngclearn_lgssm.md`, `continuous/predictive_coding_agent.md`, `continuous/stochastic_dynamics.md` |
| Continuous-state closed-loop control on beliefs | `continuous/continuous_navigation.md` |
| Explicit independent continuous agents (native JAX/RxInfer) | `continuous/independent_gaussian_agents.md` |
| Composed kind set (continuous × multi-agent) | `continuous/multi_agent_lgssm.md` |
| Factored continuous (independent per-factor LGSSM) | `continuous/factored_continuous_lgssm.md` |
| Hybrid kind (discrete POMDP + continuous block) | `continuous/hybrid_discrete_continuous.md` |
| Non-stationary dynamics (`B_t` phases / regime switching) | `discrete/time_varying_dynamics.md`, `discrete/regime_switched_dynamics.md` |
| Multi-agent & stigmergy (v3+ features) | `multiagent/stigmergic_swarm.md` |
| Hierarchical / deep temporal models | `hierarchical/hierarchical_pomdp.md` |
| Parameter learning | `learning/dirichlet_likelihood_learning.md` |
| Precision & curiosity mechanisms | `precision/precision_weighted.md`, `precision/curiosity_driven_agent.md` |
| Causal models (bnlearn export) | `discrete/bnlearn_causal_model.md` |

## Full exemplar set

### basics/
- [dynamic_perception.md](basics/dynamic_perception.md) — corrected normalized likelihood; fresh authored-length JAX filtering accepted in float32 and float64
- [static_perception.md](basics/static_perception.md) — corrected normalized likelihood; fresh authored-length JAX filtering accepted in float32 and float64

### continuous/
- [continuous_navigation.md](continuous/continuous_navigation.md)
- [factored_continuous_lgssm.md](continuous/factored_continuous_lgssm.md) — composed factored × continuous exemplar (two independent 2-dim `F_fN`/`H_fN`/`Q_fN`/`R_fN` factors, `num_factors: 2`); `detect_model_kinds` returns `{FACTORED, CONTINUOUS}` and the JAX backend renders one native LGSSM block per factor while every other continuous backend receipts it `unsupported-factored-continuous` rather than flattening the factors
- [hybrid_discrete_continuous.md](continuous/hybrid_discrete_continuous.md) — composed hybrid × continuous exemplar (a minimal 2-state POMDP declared alongside a passive `F`/`H`/`Q`/`R` block); `detect_model_kinds` returns `{HYBRID, CONTINUOUS}` and every framework receipts it `unsupported-composition` (continuous × hybrid) rather than rendering one family
- [independent_gaussian_agents.md](continuous/independent_gaussian_agents.md) — source-declared independent asymmetric Gaussian agents, rendered natively by JAX and RxInfer
- [multi_agent_lgssm.md](continuous/multi_agent_lgssm.md) — composed continuous × multi-agent exemplar (`nr_agents: 2` declared alongside the `F`/`H`/`Q`/`R` block); `detect_model_kinds` returns `{CONTINUOUS, MULTI_AGENT}` and every framework receipts it `unsupported-composition` rather than rendering one family
- [ngclearn_lgssm.md](continuous/ngclearn_lgssm.md) — passive 2-state damped-rotation linear-Gaussian model; the ngc-learn (ngclearn) backend exemplar of the continuous family
- [predictive_coding_agent.md](continuous/predictive_coding_agent.md)
- [stochastic_dynamics.md](continuous/stochastic_dynamics.md)

### discrete/
- [actinf_pomdp_agent.md](discrete/actinf_pomdp_agent.md)
- [bnlearn_causal_model.md](discrete/bnlearn_causal_model.md)
- [deep_planning_horizon.md](discrete/deep_planning_horizon.md)
- [hmm_baseline.md](discrete/hmm_baseline.md) — corrected normalized emissions; passive 50-step JAX filtering accepted in float32 and float64
- [markov_chain.md](discrete/markov_chain.md)
- [multi_armed_bandit.md](discrete/multi_armed_bandit.md)
- [regime_switched_dynamics.md](discrete/regime_switched_dynamics.md) — regime-switched transitions (`B_regime` + `b_regime_schedule`); the switching exemplar of the NONSTATIONARY kind — pymdp applies the declared schedule per step, every other categorical framework receipts `unsupported-nonstationary`
- [simple_mdp.md](discrete/simple_mdp.md)
- [time_varying_dynamics.md](discrete/time_varying_dynamics.md) — non-stationary time-indexed `B_t` phases (NONSTATIONARY kind; pymdp runs the phase sequence per step, hold-last beyond the declared span)
- [tmaze_epistemic.md](discrete/tmaze_epistemic.md) — approved episodic contingent policy contract; native JAX float64 acceptance with an independent 256-policy oracle
- [two_state_bistable.md](discrete/two_state_bistable.md)

### hierarchical/
- [hierarchical_pomdp.md](hierarchical/hierarchical_pomdp.md) — approved block-reset hierarchy; native JAX float64 acceptance with independent latent-path enumeration
- [temporal_hierarchy.md](hierarchical/temporal_hierarchy.md) — approved approximate 10/100-clock soft-message controller; native JAX float64 acceptance across 200 transitions

### learning/
- [dirichlet_likelihood_learning.md](learning/dirichlet_likelihood_learning.md)

### multiagent/
- [multi_agent_coordination.md](multiagent/multi_agent_coordination.md)
- [multi_agent_coordination_acceptance.md](multiagent/multi_agent_coordination_acceptance.md) — compact 3-agent clustered mean-field acceptance fixture (relocated from `input/multi_agent_models/`); hand-runnable `--target-dir` target for the RxInfer and DisCoPy roadmap acceptance checks, not a manifest-family exemplar
- [stigmergic_swarm.md](multiagent/stigmergic_swarm.md)

### pomdp_gridworld/
- [pomdp_gridworld_3x3.md](pomdp_gridworld/pomdp_gridworld_3x3.md)
- folder docs: [AGENTS.md](pomdp_gridworld/AGENTS.md), [README.md](pomdp_gridworld/README.md)

### recursive/
- reserved directory for bounded `--autonomous` proposal-loop runs — holds no committed models ([README.md](recursive/README.md)); nothing here contributes to the example counts

### precision/
- [curiosity_driven_agent.md](precision/curiosity_driven_agent.md)
- [precision_weighted.md](precision/precision_weighted.md)

### pymdp_scaling_study/
- [pymdp_scaling_N4_T100.md](pymdp_scaling_study/pymdp_scaling_N4_T100.md)
- [pymdp_scaling_N8_T100.md](pymdp_scaling_study/pymdp_scaling_N8_T100.md)
- [pymdp_scaling_N16_T100.md](pymdp_scaling_study/pymdp_scaling_N16_T100.md)
- [pymdp_scaling_N32_T100.md](pymdp_scaling_study/pymdp_scaling_N32_T100.md)
- [pymdp_scaling_N64_T100.md](pymdp_scaling_study/pymdp_scaling_N64_T100.md)
- folder docs: [README.md](pymdp_scaling_study/README.md)

### structured/
- [factorized_posterior.md](structured/factorized_posterior.md)

## Running an example

```bash
uv run python src/gnn/main.py --target-dir input/gnn_files/discrete --output-dir output
```
