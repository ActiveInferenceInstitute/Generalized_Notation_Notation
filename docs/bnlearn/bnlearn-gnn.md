# bnlearn-GNN Integration Architecture

This document defines the architectural integration pathways between the `bnlearn` Python package and the 25-step **GNN (Generalized Notation Notation)** pipeline utilized for processing Active Inference generative models.

## Concept Overview

Active Inference relies natively on Partially Observable Markov Decision Processes (POMDPs) modeled using Generative Graphical structures. `bnlearn` provides a robust Bayesian abstraction that integrates naturally into the GNN pipeline, acting as a mathematical bridge for structure learning and inference on simulated traces. It renders **discrete categorical Bayesian networks** (optional A/B/C/D/E matrices; multi-modality and multi-factor supported — continuous linear-Gaussian models are unsupported). By converting generalized notation payloads into `bnlearn` DAGs and DataFrames, the GNN ecosystem can conduct deep causal validations, parameter tracking, and interactive visualizations.

---

## GNN Pipeline Touchpoints

### Step 3: GNN File Discovery & Parsing
Within the `src/gnn/3_gnn.py` thin orchestrator layer, `bnlearn` can consume structurally serialized model dictionaries. 
*   **Structural Import**: GNN representations of generative models ($A$, $B$, $C$, $D$ matrices) can be directly transformed into an adjacency matrix using `bn.vec2adjmat()` or `bn.dag2adjmat()`.
*   **Validation Check**: Passing the extracted matrices through `bn.check_model()` immediately surfaces disconnected nodes or cyclical violations, ensuring rigorous structural integrity early in the pipeline.

### Step 5 & Step 6: Validation and Consistency Verification
Once the types are correctly deduced in Step 5, the model parameters (CPTs corresponding to standard POMDP matrices) are verified:
*   **Edge Strength**: `bn.independence_test()` is deployed against structural traces to validate expected epistemic separation properties natively assumed by the GNN ontological model.
*   **Topological Scoring**: Metrics exposed by `bn.structure_scores()` evaluate network robustness. 

### Step 8: Visualization
The network visualization framework in `bnlearn` complements GNN's Graph and Matrix Visualization scripts.
*   **Interaction graphs**: Calling `bn.plot(model, interactive=True)` emits a dynamic `D3.js` HTML artifact that GNN packages into the `output/` directory for visual evaluation.
*   **Comparison tracking**: For models moving through registry tracking (Step 4), `bn.compare_networks()` provides visual diffs of active inference networks evolving across versions.

### Step 12: Simulation & Execute

`bnlearn` is itself an execution target, not render-only: rendered `bnlearn` scripts are run at Step 12 by the dedicated runner in [`src/gnn/execute/bnlearn/`](../../src/gnn/execute/bnlearn/) with output written under `BNLEARN_OUTPUT_DIR`; scripts skip when the `bnlearn` extra is absent (`gnn.utils.runtime_safety.framework_availability`). Separately, the Active Inference execution engines (`pymdp`, `rxinfer`) generate trajectory traces as they sample hidden state inferences over time, and those raw simulation traces produced in `Step 12: Execute` act as the raw historical `DataFrame` fed into `bnlearn`.

This creates a powerful recursive loop where Bayesian structures are continuously updated utilizing real-time simulated outputs:
```python
# After simulation yields a trajectory dataframe `df_sim`
learned_structure = bn.structure_learning.fit(df_sim, methodtype="hc", scoretype="bic")
```

### Fixed-action full-sequence smoothing

Run a generated source-bound bnlearn program with `--inference smooth`.
Supply a JSON array of observation indices and a JSON array containing one
action index per transition. For a one-observation run, supply an empty action
array. The program writes `simulation_results.json` under
`BNLEARN_OUTPUT_DIR` and reports one float64 posterior per time, conditioned
on all observations.

```bash
BNLEARN_OUTPUT_DIR="$RESULT_DIR" python "$PROGRAM" --inference smooth \
  --observations-json "$OBSERVATIONS" --transition-actions-json "$ACTIONS"
gnn witness "$SOURCE" --results "$RESULT_DIR/simulation_results.json" \
  --program "$PROGRAM" --output "$RECEIPT"
gnn witness --verify "$RECEIPT"
```

Witness replay requires a source admitted by the [E4 finite smoothing contract](../development/remaining_roadmap_contracts.md#e4-replayable-finite-smoothing-v4).
The native scope is one stationary categorical factor with at most four
states, observations and actions, eight time steps, and 4,096 latent paths.
Observation zero conditions the declared prior. Action `t` selects the
transition from `t` to `t+1`.

### Step 16: Analysis
In this step, advanced statistical analysis utilizes `bnlearn` specifically for **Causal Inference**:
*   Given evidence from a particular time step slice of the simulation, `bn.inference.fit()` runs exact probabilistic counterfactuals to evaluate agent policy consequences over different time horizons. 

---

## Architectural Advantages for GNN

1.  **Unified CPT Generation**: `bn.generate_cpt()` reduces boilerplate in mapping Active Inference matrix initialization properties to valid Probability Distributions within the validation layer.
2.  **Streamlined Matrix Conversions**: Robust internal mappings between `Vector <-> DAG <-> Adjacency Matrix` remove the fragile `NumPy` transpositions often encountered when cross-loading PyMDP objects.
3.  **Functional Edge Tracking**: Exposes specific edge strengths, preventing "dead" states or isolated actions in an Active Inference generative model from stalling the execution loop undetected.

Through this integration, `bnlearn` effectively acts as both a validation sandbox and an analytical powerhouse, securing the epistemic integrity of the Bayesian systems orchestrated across the 25 GNN framework steps.
