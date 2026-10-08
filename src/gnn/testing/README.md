# Testing and Benchmarks

This directory holds round-trip tests, integration tests, and benchmarks for the GNN module.

- **`round_trip_tester.py`** — Round-trip harness (`GNNRoundTripTester`, production helper used by `schema_validator`). Default config lists **21** format strings in `FORMAT_TEST_CONFIG['test_formats']` (**`markdown`** plus **20** targets). See **[../SPEC.md](../SPEC.md)** for how that relates to all **23** `GNNFormat` values and **22** serializers. The pytest suite exercising it lives in **`tests/testing/`**.
- **`README_round_trip.md`** — Methodology, configuration, and historical notes.
- **`round_trip_reports/`** — Generated reports when enabled.
- **`performance_benchmarks.py`**, **`test_*.py`** — Other tests as named.

`GNNValidator` uses this harness for its JSON, XML and YAML round-trip checks.
The adapter carries supplied parameter names and exact values into independent
copies, retaining zero, false and nested containers alongside the existing
variable dimensions and model metadata. A passing conversion is checked
against those supplied values rather than an empty parameter list.

The saved-format adapter maps authored Markdown `int` and `bool` aliases to the
existing INTEGER and BINARY datatypes. It retains values, dimensions and
descriptions; `binary` and other existing enum spellings keep their meanings.
Unknown-type recovery remains categorical.

## Round-trip suite status

For the reference model (`input/gnn_files/discrete/actinf_pomdp_agent.md`, with `src/gnn/gnn_examples/` as fallback) and the default `test_formats` list, the suite is configured to report **100%** pass rate.

**Outside the default round-trip list:** `ebnf` (shares BNF / `GrammarSerializer` machinery; not a separate row in `test_formats`), `pnml` (disabled in config; parse-focused). See `FORMAT_TEST_CONFIG` in `round_trip_config.py`.

### Categories (20 conversion targets + markdown reference)

- **Schema (7):** JSON, XML, YAML, PKL, ASN.1, Protobuf, XSD  
- **Languages (6):** Scala, Python, Lean, Coq, Isabelle, Haskell  
- **Formal (5):** Alloy, BNF, Z-notation, TLA+, Agda  
- **Other (2):** Pickle, Maxima  
- **Reference (1):** Markdown  

### Performance (order of magnitude)

Typical full suite run is sub-second on small models; exact numbers depend on hardware and `REFERENCE_CONFIG` / logging flags.
