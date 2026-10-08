# DisCoPy rendering

The supported package entrypoint, `render_gnn_to_discopy`, generates a Python
program that constructs a categorical string diagram and exports model metadata.
Step 12 executes the generated program. Generation alone does not establish
numerical equivalence to a generative model.

## Generate a program

```bash
uv run --frozen --no-sync python -m gnn.cli render input/gnn_files/discrete/actinf_pomdp_agent.md --framework discopy --output /tmp/gnn-discopy-example.py
```

Python callers pass a parsed specification to
`gnn.render.discopy.render_gnn_to_discopy(spec, output_path, options=None)`.
The result is `(success, message, warnings)`. Missing optional sections produce
warnings; generation failures return `False` with the original diagnosis.
See [AGENTS.md](AGENTS.md#api-reference) for the signature and example.

Generated programs preserve model names as Python literals and export their
original values in JSON. Matrix permutation options are validated metadata;
they do not reorder the generated diagram's wires or boxes.

## Translation owners

| Owner | Responsibility |
| --- | --- |
| `discopy_renderer.py` | Supported package generator and saved program metadata |
| `translator.py` | Compatibility imports and standalone diagnostic harness |
| `gnn_parsing.py` | Section, dimension and tensor notation parsing |
| `diagram_builders.py` | Abstract tensor diagram structure |
| `file_translation.py` | File admission and diagram construction orchestration |
| `matrix_builders.py` | Experimental JAX-backed matrix construction |
| `bootstrap.py` | Optional dependency availability and setup diagnostics |
| `code_templates.py` | Compatibility source templates |
| `symmetry.py` | Permutation metadata validation |

The file-level `translator.gnn_file_to_discopy_diagram` route constructs abstract
`discopy.tensor.Diagram` objects. Native topology checks exercise the actual
domains, codomains and boxes; they do not evaluate a probability model.

## Experimental matrix route

`translator.gnn_file_to_discopy_matrix_diagram` remains importable for
compatibility but is unsupported for numerical evaluation. Native controls
found two independent problems: real two-element rows can be interpreted as
complex pairs, and the builder passes tensor dimensions to a matrix constructor
whose API takes an array first. This prevents ordinary authored tables from
establishing a valid evaluated matrix diagram.

Dependency availability flags do not establish semantic readiness. A repair
requires an explicit tensor-versus-matrix contract, domain/codomain axes and
unambiguous real/complex notation under [E4](../../../../TO-DO.md#major-work).
The abstract diagram and supported package generator remain separate routes.

## Verification

```bash
uv run --frozen --no-sync python -m pytest tests/render/test_native_discopy_generated_consumers_410.py tests/render/test_render_cli_targets.py -q
```

Generation uses the Python standard library. Executing generated programs
requires DisCoPy and NumPy; drawing requires the relevant optional visualization
dependencies. Use explicit provisioned native lanes for runtime acceptance.

See [SPEC.md](SPEC.md), [AGENTS.md](AGENTS.md) and the
[parent renderer](../README.md).
