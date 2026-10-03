# cpomdp Analysis Contract

The public API exports `analyze_payload` and `generate_analysis_from_logs`. Inputs are native continuous simulation result dictionaries or directories containing them. Invalid mean/covariance or EFE arrays fail explicitly.

Outputs are a JSON summary and Gaussian trajectory/EFE PNGs. Metrics derive uncertainty from covariance; unavailable categorical quantities include reasons. Source/dependency identity, admission estimates and measured resources remain attached. Analysis does not establish extraction correctness or a Lean proof.
