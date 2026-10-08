# Performance measurement

Accept a baseline before optimizing. Match model sources/hashes, selection,
backend, configuration and LLM mode. Record phase CPU/wall time, sampled peak
RSS, artifact counts, environment and resource admission. Include bounded
nested/large inputs, repeated/concurrent runs and distributed transfers where
relevant. Report variance and measurement limits with the tradeoff.

The [complexity benchmark](../scripts/run_complexity_benchmark.py) and
[PyMDP scaling experiment](../scripts/experiments/run_pymdp_gnn_scaling_analysis.py)
have different contracts. Tensor-allocation/disk estimates do not measure
whole-process RSS, compilation or native runtime. Endpoint RSS difference is
not peak memory; summed parallel case durations are not wall time.

Optimize the measured dominant owner and preserve selected results/artifact
semantics. Cache keys bind actual source bytes, configuration and relevant
environment. Reading a path again after hashing permits drift. Inherited
artifacts cannot become new evidence; do not drop work or substitute backends.

Use shared supervision/deadline/resource/cleanup owners. Avoid competing ad hoc
signal timeouts, unchecked pools or cleanup that only deletes a local variable.
Measure transfer volume/admission for bounded distributed paths.

For CI, compare exact-run job/step timestamps and retained native test and
coverage reports. Include runner queuing and runtime variance. Preserve status
names, markers, artifacts and failure coupling. See
[workflow scheduling](../.github/workflows/README.md#scheduling-and-evidence)
and the [performance measurement report](../src/gnn/analysis/PERFORMANCE.md).
