# Measured visualization performance

The developer benchmark runs the real numbered Step 8 CLI with selected, byte-frozen sources and fresh output. It records configuration, backend, disabled LLM mode, package source and environment identities; each repetition records terminal status, phase wall/process CPU time, sampled process-tree RSS and actual artifact hashes/counts. Before/after source identities must agree for overall acceptance. This does not change runtime admission or model/inference algorithms.

```sh
uv run python scripts/benchmark_visualization.py \
  --target-dir input/gnn_files --output-dir /tmp/gnn-step8-fresh \
  --repetitions 3 --concurrency 1 --timeout-seconds 900 \
  --max-source-files 256 --max-input-bytes 67108864 --profile
```

Use a new output path per batch. Repetitions are limited to 1–5, concurrency to 1–2 and each deadline to at most 1,800 seconds. Count/byte limits are checked before corpus copying or worker launch. Selected symlinks and paths outside the source root are refused. The benchmark retains the existing subprocess envelope's deadline, observed-descendant cleanup and sampled RSS behavior. It does not enable provider calls or train models.

## Production bottleneck and accepted change

A complete 38-source profile generated 668 visualization entries with 552 real PNG saves and no figure errors. Its instrumented total was 684.34 seconds; PNG encoder self time was 175.71 seconds (25.7%). These are profiler observations from a concurrently loaded native host, not an uninstrumented CI forecast or process CPU total.

The accepted optimization changes the shared PNG compression default from level 6 to level 3. Explicit caller `pil_kwargs` override it; non-PNG formats and caller DPI remain authoritative. Two alternating paired rounds re-encoded **all 552 actual production PNGs**:

| Encoder level | Process CPU median ± sample SD | Wall median ± sample SD | Encoded bytes |
| --- | --- | --- | --- |
| 6 | 76.591 ± 0.121 s | 86.938 ± 1.964 s | 107,485,534 |
| 3 | 72.394 ± 0.290 s | 82.052 ± 0.488 s | 110,785,382 |

Level 3 reduced encoder-stage CPU by **5.48%**, with **3.07% more bytes**. Every selected image retained its dimensions, decoded RGBA pixels and original metadata. Input decoding and equality verification were outside the timed stage. Real Matplotlib save tests separately retain artists, resolution/metadata and explicit level-6 byte identity. This finite coverage does not establish unchanged artists for every possible plot. Level 1 was rejected: a 16-production-image sample added approximately 131% bytes for only approximately 5% CPU gain.

## Complete-route and capacity witnesses

Three balanced before/after pairs used identical source bytes, configuration, environment and the same public benchmark harness. All six real CLI phases passed, with unchanged package source within each phase and 76 artifacts (44 PNGs) per run. The nested corpus includes actual static-perception, active-inference and hybrid-model source content; one historical filename `continuous/linear_gaussian.md` is an explicitly recorded alias of hybrid-model bytes, not evidence of a different Gaussian workload.

| Whole selected Step 8 route | Wall median ± sample SD | Process CPU median ± sample SD |
| --- | --- | --- |
| Before | 37.429 ± 20.132 s | 26.326 ± 8.367 s |
| After | 27.274 ± 11.287 s | 22.768 ± 4.205 s |

The large variance and concurrent host activity prevent attributing a whole-route or CI acceleration to the encoder alone. These measured source epochs include presentation changes and precede the final scientific followups; they do not certify the final integrated release commit. Full raw receipts retain per-source, per-configuration, source-epoch and artifact hashes.

A separate nested/large-input witness combined the actual static-perception model with a 2,101,485-byte expanded model containing 160 additional scalar variables, 159 additional connections and a large annotation. Existing production sampling remained intact. Three real runs with concurrency 2 passed, each producing 43 artifacts/16,815,956 bytes; wall median was 29.289 s (sample SD 5.167 s). Source-count and 2-MiB byte-limit refusal controls created no output and launched no worker. These admission bounds do not certify RSS or JIT capacity.

Two caller-owned localhost Dask worker processes transferred two genuine 1-MiB payloads with byte/hash parity in 0.038 s. Real sibling-success/error controls retained both outcomes; a 0.2-second collection budget returned a timeout in 0.208 s. Owned cluster close reaped both workers. This bounded local witness makes no WAN, Ray, throughput-scaling or immediate remote cancellation claim.

## Measurement limits and custody

RSS is sampled every 0.25 seconds by the existing envelope and is not an exact peak or allocation/JIT certificate. Process CPU includes the worker's threads but excludes separate descendants. Fresh processes still use operating-system caches. Artifact census/hash/decoding is outside phase time but included in batch wall time. Instrumented and uninstrumented timings are not interchangeable.

The earlier full after-corpus run failed during shared-volume exhaustion and is retained as unsuccessful evidence. Earlier profile-enabled nested trials and rejected compression-level-1 measurements are excluded from acceptance. No unsuccessful or pending phase is normalized into success.

[The durable measurement index](performance_measurements_410.json) binds complete raw receipt hashes, source/configuration epochs, timing arrays, artifact counts, capacity admissions and bounded distributed outcomes. The complete raw native receipts, source copies and actual production images remain in the task-owned `evidence410-science` area. The final integrated source still requires the repository's full release gates and visual review.
