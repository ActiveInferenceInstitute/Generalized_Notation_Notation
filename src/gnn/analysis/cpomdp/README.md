# cpomdp Analysis

`analyze_payload(payload, output_dir)` validates continuous means and posterior covariance, reports Gaussian uncertainty, and writes readable trajectory and EFE plots. Categorical entropy and confidence remain unavailable with reasons.

`generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)` analyzes each selected `simulation_results.json` once and returns artifact paths. Pipeline scheduling supplies only current-run execution artifacts.

Source/dependency identity, search admission and measured resource receipts remain attached to the summary. [Backend usage](../../render/cpomdp/README.md) and [implementation guide](../../../../docs/gnn/implementations/cpomdp.md) describe explicit experimental selection.
