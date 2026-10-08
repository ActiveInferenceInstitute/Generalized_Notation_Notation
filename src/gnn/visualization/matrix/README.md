# Matrix Visualization

Heatmap and statistical visualization for model likelihood, transition,
preference, prior, and PyMDP/POMDP tensor matrices.

## Files

- `visualizer.py` — Matrix heatmap engine with 3-D tensor support. POMDP `B`
  and explicitly indexed `B_fN` tensors use `(next_state, previous_state, action)` and validate stochasticity
  by summing over the next-state axis for each previous-state/action column.
- `extract.py` — Matrix extraction from parsed models
- `compat.py` — Shared helper exports

## Outputs

- Pipeline PNG visualizations; direct Python filename/Matplotlib options control other formats.
- Canonical `B`/`B_fN` action planes; other 3-D tensors retain explicit generic axis slices unless the direct caller declares `tensor_type="transition"`. Generic tensors do not receive POMDP transition diagnostics.
- Independent, source-labeled CSV exports for every matrix and every 3-D axis-2 plane, using bounded disambiguated filenames so one matrix cannot overwrite another.
- POMDP transition analysis panels for entropy, stochasticity, and dominant
  next-state structure. Natural-log entropy is in nats with the exact `0*log(0)=0` limit; invalid probabilities receive an unavailable reason.
- Color-normalized black/white annotations preserve signed values. Heatmap axis bases and caller-declared units are explicit; undeclared units remain unspecified.
- Small correlation heatmaps retain their actual column-correlation values,
  annotations, and index labels with a symmetric signed colorbar around zero.
  Explicit color limits avoid upstream deprecated colormap mutation. When the
  existing constant-column handling yields only zeros, the `[-1, 1]` color
  domain is a display fallback; it adds no inferred correlations or numeric
  data. The existing one-row/one-column data passthrough and large-matrix image
  branch remain unchanged.

## See Also

- [Parent: visualization/README.md](../README.md)
