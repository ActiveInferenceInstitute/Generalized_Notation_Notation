# Benchmarks, fixtures, and provenance

This is the consumer reference for research artifacts produced by the
`generalized-notation-notation` distribution ([package metadata](../../pyproject.toml)).
Package version, model-family manifest schema, execution summary format and analysis
schema version are separate identities. Preserve them when exchanging artifacts.

## Benchmark entry points

| Surface | Exact entry point | Definition and output owner |
| --- | --- | --- |
| PyMDP scaling study | `uv run python scripts/experiments/run_pymdp_gnn_scaling_analysis.py` | [Harness](../../scripts/experiments/run_pymdp_gnn_scaling_analysis.py), [configuration](../../scripts/experiments/pymdp_scaling_config.yaml) and deterministic [spec generator](../../scripts/pymdp_spec_generator.py). The harness generates an N by T grid, delegates to the pipeline, and writes `pymdp_scaling_run_manifest.json` under its configured pipeline output directory. |
| Repeated rendered-script timing | `uv run python src/gnn/main.py --target-dir input/gnn_files/pomdp_gridworld --only-steps 3,11,12 --frameworks pymdp --execution-benchmark-repeats 3` | [Step 12 producer](../../src/gnn/execute/processor/single.py), `execute_single_script`; runs the rendered script sequentially for each repeat and records process wall time in seconds. |
| Parser and validator microbenchmarks | `gnn.testing.performance_benchmarks.benchmark_gnn_files(file_paths)` | [Canonical module](../../src/gnn/testing/performance_benchmarks.py), with `list[Path]` input and a dictionary report. `GNNPerformanceBenchmark(test_iterations=100)` supports explicit iteration counts. |
| Pipeline performance budgets | `uv run pytest tests/pipeline/test_pipeline_performance.py -v -m performance` | [Test definitions and `THRESHOLDS`](../../tests/pipeline/test_pipeline_performance.py). These are absolute budgets from one measurement environment, not regression bands. The `pipeline` marker excludes them from the default gate. |
| Model-family acceptance | `uv run python scripts/run_model_family_acceptance.py --families gridworld --output-dir output/model_family_acceptance --strict` | [CLI](../../scripts/run_model_family_acceptance.py), [implementation](../../src/gnn/pipeline/model_family_acceptance.py) and [input manifest](../../input/model_family_manifest.json). This measures pipeline acceptance and artifact evidence rather than producing benchmark scores. |

Commands run from the repository root. The scaling harness appends its own `scripts/`
directory to the import path, so no `PYTHONPATH` is needed; the committed
configuration carries the resource limits, output locations and output clearing. Read
the [scaling preflight guide](../../scripts/README.md#pymdp-scaling-preflight) before
running a sweep: the default configuration inserts Step 5 (`type_check: true`) after
Step 3, runs steps 3, 5, 11, 12 and 17, and defaults to one timing repeat. A single
repeat does not measure run-to-run uncertainty.

With the source-root import setup described in the [API reference](interop_api.md), the
benchmark types are `gnn.testing.performance_benchmarks.PerformanceMetrics` and
`gnn.integration.meta_analysis.collector.SweepRecord`. The package exports the same
`SweepRecord` object through `gnn.integration.meta_analysis`; analysis consumers import
that declaration, as the statistics, validator, reporter and visualizer modules already
do.

## Existing result schemas

The execution producer and collector are the canonical definitions for execution
records. There is no separately versioned JSON Schema for those dictionary outputs. Do
not infer a new schema generation from the package version.

| Artifact | Producer and current shape |
| --- | --- |
| `pymdp_scaling_run_manifest.json` | Harness `_build_run_manifest`: `status: string`, `updated_at: UTC ISO string`, `configuration: object`, `paths: object` with generated-spec/pipeline/render/execution/integration roots, `grid_pairs: integer`, `planned_count: integer`, `planned_pairs: array<[integer, integer]>`, `skip_lines: array<string>`, `estimated_spec_bytes: integer`, `estimated_spec_mib: number`, `phases: array<object>`. It has no schema/version field and no fixture digests. |
| `12_execute_output/summaries/execution_summary.json` | `process_execute` writes aggregate counters and `execution_details` with `execution_summary_format: "slim_v1"`. [`_slim_execution_detail`](../../src/gnn/execute/metadata.py) keeps timing fields, identities, success/skip/error state and file pointers. `--execution-summary-detail` additionally writes `execution_summary_detail.json` with full detail dictionaries and `execution_summary_format: "detail_v1"`; the aggregate stays slim. The [collector](../../src/gnn/integration/meta_analysis/collector.py) reads the aggregate and fills gaps from per-script logs; timing fields survive slimming, so they are present either way. |
| Execution timing fields | `execution_time: number` is the median of successful repeats, `execution_time_mean: number` is their arithmetic mean, `execution_time_std: number` is their population standard deviation, `execution_time_samples: array<number>` holds seconds, and `execution_benchmark_repeats: integer` is the requested repeat count normalized to at least one. On success all requested samples exist; the standard deviation is zero for one sample. Failures can lack aggregates and samples, and their `execution_time` can describe the failing attempt. Inspect `success`, `skipped`, `error` and `return_code` before using timings. |
| Per-script `execution_logs/*_results.json` | `execute_single_script` writes `framework`, `model_name`, `script_name`, `script_path`, `success`, `return_code`, `execution_time`, `timestamp`, `simulation_data`, `execution_benchmark_repeats`, `execution_metadata`, plus timing aggregate keys when available. Dependency or executor preflight can return before writing this file; aggregate skip accounting is still required. |
| `meta_statistics.json` | [`compute_meta_statistics`](../../src/gnn/integration/meta_analysis/statistics.py) emits `schema_version: "1.0"`, `per_framework: object`, `per_cell_best_framework: array<object>` and `loglog_runtime_vs_n_by_T: object`. Runtime aggregates use successful positive timing records; `sum_median_proxy_s` sums per-record medians. Missing NumPy produces only `schema_version` and `error: "numpy_missing"`. |
| `sweep_validation.json` | [`validate_sweep_records`](../../src/gnn/integration/meta_analysis/validator.py) emits `schema_version: "1.0"`, `issues: array<object>` and severity counts in `summary: {info, warning, error}`. Each issue carries `severity`, `code` and `message` with contextual fields. Findings are non-fatal reporting evidence. |
| `visualizations/data/sweep_data.csv` | [`SweepExportMixin._export_csv`](../../src/gnn/integration/meta_analysis/visualizer_export.py) writes `model_name, framework, num_states, num_timesteps, execution_time_s, execution_time_std_s, execution_benchmark_repeats, time_per_step_ms, success, timed_out, lines_of_code, total_lines, final_accuracy, mean_belief_entropy`. Absent values are empty cells; a zero standard deviation is also serialized as empty. JSON timing samples retain the more precise evidence. |
| `model_family_acceptance_ledger.json` | [`run_model_family_acceptance`](../../src/gnn/pipeline/model_family_acceptance.py) emits `schema: "gnn_model_family_acceptance_ledger_v1"`, `created_at`, `manifest`, `strict`, `only_steps`, `frameworks`, `family_count`, `families`, `status` (`passed` or `failed`) and `failed_families`. Family rows carry step status, evidence, artifact pointers and interpretability summaries. |

The timing contract is tested by
[`test_execute_benchmark_samples.py`](../../tests/execute/test_execute_benchmark_samples.py)
and its consumption by
[`test_integration_meta_analysis_benchmark_fields.py`](../../tests/integration/test_integration_meta_analysis_benchmark_fields.py).
For a focused check:

```bash
uv run --frozen --no-sync python -m pytest tests/execute/test_execute_benchmark_samples.py tests/integration/test_integration_meta_analysis_benchmark_fields.py -q
```
