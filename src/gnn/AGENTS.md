# GNN Processing - Agent Scaffolding

## Module Overview

**Purpose**: Core GNN file discovery, parsing, multi-format serialization, and validation for Generalized Notation Notation specifications

**Pipeline Step**: Step 3: GNN file processing (3_gnn.py)

**Category**: Core Processing

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

### Primary Responsibilities

1. Discover GNN specification files in target directories
2. Parse GNN markdown specifications into structured data
3. Serialize parsed models to **22** registered output formats (**23** `GNNFormat` values; PNML is parser-focused — see [SPEC.md](SPEC.md))
4. Validate GNN syntax and semantic correctness

### Key Capabilities

- Multi-format GNN parsing (markdown, JSON, YAML, etc.)
- **22** registered serializers for **23** `GNNFormat` values (PNML: parse-only in `SERIALIZER_REGISTRY` — see [SPEC.md](SPEC.md)); covers Scala, Lean, Coq, Python, BNF, EBNF, Isabelle, Maxima, XML, JSON, Protobuf, YAML, XSD, ASN.1, PKL, Alloy, Z-notation, TLA+, Agda, Haskell, Pickle, Markdown
- `.pkl` is treated as textual PKL DSL by default; binary pickle inputs should use `.pickle`, with previous-format binary `.pkl` routed by content detection and logged as a warning.
- Round-trip validation (parse → serialize → parse)
- Cross-format consistency checking

---

## API Reference

### Pipeline Processing Function

#### `process_gnn_multi_format(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = True, verbose: bool = False, **kwargs: Any) -> bool`

**Description**: Main processing function used by pipeline orchestrator (3_gnn.py). Discovers, parses, and serializes GNN files to all supported formats.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Base output directory (step-specific directory will be created)
- `logger` (logging.Logger): Logger instance
- `recursive` (bool): Whether to recurse into subdirectories (default: True)
- `verbose` (bool): Enable verbose logs (default: False)
- `**kwargs` (Any): Additional processing options

**Returns**: `bool` - True on success, False otherwise

**Location**: `src/gnn/processing/multi_format_processor.py`

**Example**:

```python
from gnn.processing.multi_format_processor import process_gnn_multi_format
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    logger=logger,
    recursive=True,
    verbose=True,
)
```

### Exported Functions from `__init__.py`

#### `process_gnn_directory(directory: Union[str, Path], output_dir: Union[str, Path, None] = None, recursive: bool = True, parallel: bool = False) -> Dict[str, Any]`

**Description**: Process all GNN files in a directory. Returns processing results dictionary.

**Parameters**:

- `directory` (Union[str, Path]): Directory to process
- `output_dir` (Union[str, Path, None]): Optional output directory for results (default: None)
- `recursive` (bool): Whether to process subdirectories (default: True)
- `parallel` (bool): Whether to process discovered files concurrently (default: False)

**Returns**: `Dict[str, Any]` - Dictionary with processing results containing:

- `status` (str): Processing status ("SUCCESS" or "FAILED")
- `files` (List[str]): List of processed file paths
- `processed_files` (List[str]): List of successfully processed files

**Location**: `src/gnn/processing/processor.py`

#### `process_gnn_directory_lightweight(target_dir: Path, output_dir: Path = None, recursive: bool = False) -> Dict[str, Any]`

**Description**: Lightweight GNN directory processing without heavy dependencies and faster execution.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path, optional): Directory to save results (default: None)
- `recursive` (bool): Whether to process subdirectories (default: False)

**Returns**: `Dict[str, Any]` - Dictionary with processing results containing:

- `timestamp` (str): Processing timestamp
- `target_directory` (str): Source directory path
- `files_found` (int): Number of files discovered
- `files_processed` (int): Number of files successfully processed
- `success` (bool): Overall success status
- `errors` (List[Dict]): List of error information
- `parsed_files` (List[Dict]): List of parsed file information
- `validation_results` (List[Dict]): List of validation results

**Location**: `src/gnn/processing/processor.py`

#### `discover_gnn_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]`

**Description**: Discovers candidate files for **lightweight** processing (`process_gnn_directory`, reports, etc.). This is **not** the same discovery policy as pipeline Step 3.

**Parameters**:

- `directory` (Union[str, Path]): Directory to search
- `recursive` (bool): Whether to search subdirectories (default: True)

**Returns**: `List[Path]` - List of Path objects for discovered files

**Glob patterns**: `*.md`, `*.gnn`, `*.txt` only. Excludes `README.md`, `CHANGELOG.md`, `LICENSE.md`, and names matching `*.template.md` / `*.example.md`.

**Pipeline Step 3 (`process_gnn_multi_format`)** uses a **broader** extension list in `multi_format_processor.py` (e.g. `.json`, `.yaml`, `.lean`, …) so interchange artifacts on disk are found and re-processed. See [SPEC.md](SPEC.md) § File discovery.

**Location**: `src/gnn/processing/processor.py`

#### `parse_gnn_file(file_path: Union[str, Path], content: Optional[str] = None) -> Dict[str, Any]`

**Description**: Parse a single GNN file and extract basic information.

**Parameters**:

- `file_path` (Union[str, Path]): Path to the GNN file
- `content` (Optional[str]): Pre-read file content; if None the file is opened and read.

**Returns**: `Dict[str, Any]` - Dictionary with parsed information containing:

- `file_path` (str): Path to the file
- `file_name` (str): Name of the file
- `file_size` (int): Size of the file in bytes
- `sections` (List[str]): List of extracted sections
- `variables` (List[str]): List of extracted variables
- `structure_info` (Dict): Structure analysis information
- `parse_timestamp` (str): Timestamp of parsing

**Location**: `src/gnn/processing/processor.py`

#### `check_gnn_file_structure(file_path: Union[str, Path], content: Optional[str] = None) -> Dict[str, Any]`

**Description**: Validate the structure of a GNN file.

**Parameters**:

- `file_path` (Union[str, Path]): Path to the GNN file
- `content` (Optional[str]): Pre-read file content; if None the file is opened and read.

**Returns**: `Dict[str, Any]` - Dictionary with validation results containing:

- `file_path` (str): Path to the file
- `file_name` (str): Name of the file
- `valid` (bool): Whether the file structure is valid
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of validation warnings
- `validation_timestamp` (str): Timestamp of validation

**Location**: `src/gnn/processing/processor.py`

#### `generate_gnn_report(processing_results: Dict[str, Any], output_path: Union[str, Path] = None) -> str`

**Description**: Generate a report from GNN processing results.

**Parameters**:

- `processing_results` (Dict[str, Any]): Results from GNN processing
- `output_path` (Union[str, Path, None]): Optional path to save the report (default: None)

**Returns**: `str` - Report content as markdown string

**Location**: `src/gnn/processing/processor.py`

#### `get_module_info() -> Dict[str, Any]`

**Description**: Get information about the GNN module.

**Returns**: `Dict[str, Any]` - Dictionary with module information containing:

- `name` (str): Module name
- `version` (str): Module version
- `description` (str): Module description
- `features` (List[str]): List of available features
- `available_validators` (List[str]): List of available validators
- `available_parsers` (List[str]): List of available parsers
- `schema_formats` (List[str]): List of supported schema formats
- `supported_formats` (List[str]): List of supported file formats
- `capabilities` (Dict): Dictionary of capability flags

**Location**: `src/gnn/processing/processor.py`


#### `validate_gnn_source(source: Any, *, is_content: bool = False) -> Dict[str, Any]`

**Description**: Validate a GNN source — file path or content string. When
`source` is an existing path and `is_content` is `False`, the file is read
first; otherwise `source` is treated as raw content.

**Returns**: `Dict[str, Any]` with `is_valid` (bool) and `errors` (List[str])

#### `validate_gnn_syntax(file_path_or_content: str, validation_level: ValidationLevel = ValidationLevel.STANDARD, **kwargs) -> Tuple[bool, List[str]]`

**Description**: Validate a GNN file or content string via the formal
`schema_validator` pipeline (delegates to `GNNValidator.validate_file`;
returned errors are the formal validator's messages, e.g.
`"Required section missing: ModelName"`). An existing path and the same
bytes passed as a content string validate identically; content is
validated as GNN markdown text. Validity means the formal validator's
normative markdown gate (every required section with substantive body
content) — a deliberate tightening over the earlier regex heuristic; the
formal level ladder applies at every level (BASIC included).


**Parameters**:

- `file_path_or_content` (str): Path to a GNN file or GNN content string
- `validation_level` (ValidationLevel): Level of validation to perform
  (default: STANDARD); also accepts level strings (`"strict"`,
  `"STRICT"`) and `None` (the validator's default level, STANDARD).
  Unknown level strings raise `ValueError`.
- `**kwargs`: Ignored (present for call-site compatibility only).

**Returns**: `Tuple[bool, List[str]]` - Tuple of (is_valid, list_of_errors)

**Location**: `src/gnn/parsers/basic.py`

### Headless Extraction (`gnn.extract`)

#### `extract_to_json(path: Union[str, Path], *, strict_validation: bool = True, on_error: str = "lenient", compact: bool = False) -> str`

**Description**: Machine-readable headless extraction entry point. Calls
`gnn.extract.pomdp_extractor.extract_pomdp_from_file` and returns a JSON string — it
never raises. Success: the `POMDPStateSpace.to_dict()` payload (no `status`
key). Failure: the envelope
`{"status": "error", "error": {"code", "message", "line", "section"}}`
(codes from the extractor's structured errors, e.g. `GNN-E002` shape /
orientation contradictions, `GNN-E006` parameter-parse failures; `GNN-E000`
for a bare no-result extraction). `compact=True` emits compact JSON
(`separators=(",", ":")`, no indent) instead of `indent=2`.

#### `main(argv: Optional[List[str]] = None) -> int`

**Description**: CLI wrapper: `python -m gnn.extract FILE [--strict|--no-strict] [--compact]`.
Prints the payload JSON (exit 0) or the error envelope (exit 1).

**Location**: `src/gnn/extract/__init__.py` (stdlib-only at module scope; the
extractor module is imported lazily inside the call path).

### Lazy package `__init__`

`gnn/__init__.py` re-exports its names lazily via PEP 562 module
`__getattr__` backed by an explicit name → submodule map: `import gnn` executes
no submodule at import time, so heavy module-scope dependencies (psutil,
matplotlib) and the pipeline stack are only paid when a name is actually
resolved. Every previously eager re-export (including `__version__` and
`FEATURES`) is preserved; ImportErrors propagate unchanged (no silent
fallback).

### Format Detection and File Conversion

#### `detect_gnn_format_from_content(content: str) -> GNNFormat`

**Description**: Pure content-based format sniffing (public, `unified_parser.py`).
Inspects structural markers (XML/PNML/XSD declarations, JSON braces, GNN markdown
sections, per-language comment/keyword fingerprints). Never raises; unrecognized
content falls back to `GNNFormat.MARKDOWN`. `UnifiedGNNParser._detect_format_from_content`
delegates to it after reading the first 2000 characters of a file.

#### `GNNParsingSystem.convert_file(input_path, output_path, from_format=None, to_format=None) -> Path`

**Description**: One-call file conversion: parse the source (format from
`from_format` or the input extension), serialize to the target (`to_format`
or the output extension), write the output (creating parent directories).
Returns the resolved output path. Raises `FileNotFoundError` (missing input),
`ValueError` (unsupported target format / unknown output extension),
`ParseError` (parse or serialization failure). Targets without a registered
serializer (e.g. PNML) raise `ValueError`.

### Shared Embedded-Model-Data Mechanism (`parsers/common.py`)

`BaseGNNParser` centralizes the per-parser `MODEL_DATA` JSON round-trip machinery:

- `EMBEDDED_JSON_PATTERNS: ClassVar[list[str]]` — comment regexes locating
  embedded `MODEL_DATA` JSON; each concrete parser declares its own prefixes
  (e.g. `//` for Scala, `--` for Lean/Haskell, `(* *)` for Coq/Isabelle).
- `_extract_embedded_json_data(content)` — first pattern whose captured JSON
  parses wins (delegates to `common.extract_embedded_json_data`).
- `_parse_from_embedded_data(data, result)` — strict variant: builds a NEW
  `GNNInternalRepresentation` with direct enum casts (scala, lean, coq,
  isabelle, python, haskell, TLA, agda).
- `EMBEDDED_LENIENT_MODEL_NAME: ClassVar[str]` + `_parse_embedded_data_lenient(data, result)`
  — lenient variant: mutates `result.model` in place with `.get` defaults
  (BNF, EBNF, Maxima).

Parsers with semantically richer embedded handling keep their own implementations
(XML/PNML `_build_model_from_embedded_data`, protobuf, binary).

### Helper Functions (Internal but Exported)

#### `_extract_sections_lightweight(content: str) -> List[str]`

**Description**: Extract sections from GNN content using lightweight parsing.

**Parameters**:

- `content` (str): GNN file content

**Returns**: `List[str]` - List of extracted section names

**Location**: `src/gnn/processing/processor.py`

#### `_extract_variables_lightweight(content: str) -> List[str]`

**Description**: Extract variables from GNN content using lightweight parsing.

**Parameters**:

- `content` (str): GNN file content

**Returns**: `List[str]` - List of extracted variable names

**Location**: `src/gnn/processing/processor.py`

---

### Parser and registry types

#### `GNNParsingSystem` (`parsers/system.py`)

**Description**: Unified registry-backed API — loads parsers/serializers from `PARSER_REGISTRY` / `SERIALIZER_REGISTRY`.

**Typical use**: `GNNParsingSystem().parse_file(path)`, then serialize via the system’s serializer map for a chosen `GNNFormat`.

#### `GNNFormalParser` (`parsers/basic.py`)

**Description**: Formal / section-oriented parsing helpers used with `validate_gnn_syntax`, `parse_gnn_formal`, etc.

#### `GNNParser` (two meanings)

- **`schema_validator.GNNParser`**: Section-level parser used by enhanced validation.
- **`parsers.common.GNNParser`**: **Protocol** implemented by concrete format parsers.

#### `GNNFormat` (Enum)

**Description**: Enumeration of supported GNN formats

**Values**:

- `MARKDOWN`, `JSON`, `XML`, `YAML`, `SCALA`, `PROTOBUF`, `PKL`, `XSD`, `ASN1`, `PNML`, `LEAN`, `COQ`, `PYTHON`, `BNF`, `EBNF`, `ISABELLE`, `MAXIMA`, `ALLOY`, `Z_NOTATION`, `TLA_PLUS`, `AGDA`, `HASKELL`, `PICKLE`

---

## Dependencies

### Required Dependencies

- `pathlib` - File path manipulation
- `typing` - Type annotations
- `re` - Regular expression parsing
- `json` - JSON serialization

### Optional Dependencies

- `yaml` - YAML format support (recovery: skip YAML generation)
- `protobuf` - Protocol buffer support (recovery: skip Protobuf generation)

### Internal Dependencies

- `gnn.utils.pipeline_orchestration.pipeline_template` - Logging and pipeline utilities
- `pipeline.config` - Configuration management

---

## Configuration

Step 3 behavior is controlled by `process_gnn_multi_format` kwargs — notably
`serialize_preset` (default `"full"`), plus `recursive` and `verbose`. There are
no `GNN_*` environment variables or module-level `DEFAULT_*` size/format
constants in this module; do not rely on the ones some older docs listed.

---

## Usage Examples

### Basic Usage

```python
from gnn.processing.multi_format_processor import process_gnn_multi_format
from pathlib import Path

success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/3_gnn_output"),
    logger=logger,
)
```

### With a Serialization Preset

```python
# Restrict serialization to a subset of formats
success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/3_gnn_output"),
    logger=logger,
    serialize_preset="full",
)
```

### Pipeline Integration

```python
# Called from 3_gnn.py
from gnn.processing.multi_format_processor import process_gnn_multi_format

run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN discovery, parsing, and multi-format serialization",
)
```

---

## Input/Output Specification

### Input Requirements

- **File Formats**: `.md` files containing GNN specifications
- **Directory Structure**: Any directory structure (recursive search supported)
- **Prerequisites**: None (first processing step after template/setup)

### Output Products

- **Primary Outputs**:
  - Parsed model JSON files (`*_parsed.json`)
  - One artifact per **serializer-backed** format (**22** registered serializers; **PNML** is parse-only in `SERIALIZER_REGISTRY` — see [SPEC.md](SPEC.md))
- **Metadata Files**:
  - `gnn_processing_results.json` - Processing summary
  - `gnn_processing_summary.json` - Detailed statistics
- **Artifacts**: Format-specific files in subdirectories

### Output Directory Structure

```
output/3_gnn_output/
├── model_name/
│   ├── model_name_parsed.json
│   ├── model_name.scala
│   ├── model_name.lean
│   ├── model_name.coq
│   ├── model_name.py
│   ├── ... (additional serializer outputs)
├── gnn_processing_results.json
└── gnn_processing_summary.json
```

---

## Error Handling

### Error Categories

1. **File Not Found**: Log warning, continue to next file
2. **Parse Errors**: Log error with line number, mark file as failed
3. **Serialization Errors**: Log warning, skip problematic format
4. **Validation Errors**: Log error, optionally continue based on strict mode

### Recovery Strategies

- **Primary**: Parse all formats successfully
- **Recovery 1**: Skip problematic format, continue with others
- **Recovery 2**: Generate minimal JSON representation
- **Final**: Log error, continue pipeline (non-blocking)

### Error Reporting

- **Logging Level**: ERROR for parse failures, WARNING for format skips
- **User Messages**: "Failed to parse {file}: {specific_error}"
- **Recovery Suggestions**: "Check GNN syntax at line {N}" or "Install {dependency} for {format} support"

---

## Integration Points

### Orchestrated By

- **Script**: `3_gnn.py`
- **Function**: `run_script()` wrapper

### Imports From

- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized logging and error handling
- `pipeline.config` - Output directory management

### Imported By

- `5_type_checker.py` - Re-parses target-dir GNN files (no Step 3 artifact read)
- `6_validation.py` - Loads `gnn_processing_results.json` + parsed models from `output/3_gnn_output/`
- `7_export.py` - Loads the Step 3 manifest + parsed models from `output/3_gnn_output/`
- `8_visualization.py` - Prefers `output/3_gnn_output/` parsed JSON; markdown re-parse fallback
- `10_ontology.py` - Re-parses target-dir GNN files (no Step 3 artifact read)
- `11_render.py` - Re-parses target-dir GNN files (no Step 3 artifact read)

### Data Flow

```
input/gnn_files/ (mixed extensions per multi_format_processor) → GNNParsingSystem → Serializers → output/3_gnn_output/
                            ↓
                    Parsed Model JSON
                            ↓
                  [Consumers: Steps 6, 7, 8 (preferred), 9 — all other steps re-parse input]
```

---

## Testing

### Test Files

- `tests/gnn/test_gnn_overall.py` - Module-level coverage and smoke tests
- `tests/gnn/test_gnn_parsing.py` - Parsing-focused tests
- `tests/gnn/test_gnn_parsing_system.py` - `GNNParsingSystem` / registry tests
- `tests/gnn/test_gnn_processing.py` - Directory processing tests
- `tests/gnn/test_gnn_parsers_common.py` - Parser utilities tests
- `tests/gnn/test_gnn_parsers_json.py` - JSON parser tests
- `tests/gnn/test_gnn_parsers_base_serializer.py` - Serializer base tests
- `tests/gnn/test_gnn_xml_parser.py` - XML parser tests
- `tests/gnn/test_gnn_schema.py` - Schema validator tests
- `tests/gnn/test_gnn_cross_format_validator.py` - Cross-format validation tests
- `tests/gnn/test_gnn_validation.py` - Validation tests

### Test Coverage

Measure locally: `uv run --extra dev python -m pytest tests/test_gnn*.py --cov=src/gnn --cov-report=term-missing`. Targets are project-defined (see CI / maintainer notes); do not treat fixed percentages in docs as measured unless cited from a report.

### Key Test Scenarios

1. Parse valid GNN markdown files
2. Handle malformed GNN syntax gracefully
3. Serialize to all **serializer-backed** formats (**22**; **23** enum values — see [SPEC.md](SPEC.md))
4. Round-trip validation (parse → serialize → parse)
5. Cross-format consistency checking

### Test Commands

```bash
# Run GNN-specific tests
uv run --extra dev python -m pytest tests/test_gnn*.py -v

# Run with coverage
uv run --extra dev python -m pytest tests/test_gnn*.py --cov=src/gnn --cov-report=term-missing

# Run only parser tests
uv run --extra dev python -m pytest tests/gnn/test_gnn_parsing.py -v
```

---

## MCP Integration

### Tools Registered (representative)

See **`mcp.py`** `register_tools` for the authoritative list. Examples include:

- `get_gnn_documentation` — load bundled docs / schema / grammar snippets
- `validate_gnn_content` — validate content with level and optional round-trip flags
- `parse_gnn_content` — parse content with format hint
- `validate_cross_format_consistency_content` — cross-format checks
- `process_gnn_directory`, `run_round_trip_tests`, `get_gnn_module_info`, etc.

### MCP File Location

- `src/gnn/mcp/gnn_root.py` — GNN root MCP tool registrations

---

## Performance Characteristics

### Resource Requirements

- **Memory**: ~5MB per GNN file + 2MB per format
- **CPU**: Low (primarily I/O bound)
- **Disk**: Order of magnitude ~150KB per format × **22** serializer outputs (varies by model)

### Execution Time

- **Fast Path**: <100ms for typical GNN file (13 variables, 11 connections)
- **Slow Path**: ~2-3s for large models (>100 variables, >50 connections)
- **Timeout**: None (synchronous processing)

### Scalability

- **Input Size Limits**: 10MB per file (configurable)
- **Parallelization**: Lightweight directory processing can process discovered files concurrently.

---

## Development Guidelines

### Adding New Formats

1. Add a value to **`GNNFormat`** in `src/gnn/parsers/common.py` (if it is a new format id).
2. Implement **`src/gnn/parsers/<name>_parser.py`** and, unless parse-only, **`src/gnn/parsers/<name>_serializer.py`**.
3. Register classes in **`PARSER_REGISTRY`** and, when applicable, **`SERIALIZER_REGISTRY`** in **`src/gnn/parsers/system.py`**.
4. Add tests under `tests/` and extend **`tests/testing/test_round_trip.py`** (participation configured in **`src/gnn/testing/round_trip_config.py`**) if the format should join the default round-trip list.
5. Update **[SPEC.md](SPEC.md)** if canonical counts change.

### Code Style

- Follow PEP 8
- Use type hints for all public functions
- Document all public classes and methods
- Include docstring examples

### Testing Requirements

- New serializers need tests; round-trip tests should cover any format claimed in **[SPEC.md](SPEC.md)** / `round_trip_config.py` config.

---

## Troubleshooting

### Common Issues

#### Issue 1: "Failed to parse GNN file at line X"

**Symptom**: Parser error with line number  
**Cause**: Invalid GNN syntax (missing delimiter, incorrect format)  
**Solution**: Check GNN syntax at specified line, ensure proper markdown formatting

#### Issue 2: "Format {X} not supported"

**Symptom**: Warning about missing format support  
**Cause**: Optional dependency not installed  
**Solution**: Install missing dependency or accept format will be skipped

#### Issue 3: "Round-trip validation failed"

**Symptom**: Parsed model differs after serialize/parse cycle  
**Cause**: Lossy serialization format or parser inconsistency  
**Solution**: Check format specification, report bug if parser issue

### Debug Mode

```bash
# Run with verbose logging
python src/gnn/3_gnn.py --verbose

# Check output directory
ls -la output/3_gnn_output/

# View processing summary
cat output/3_gnn_output/gnn_processing_summary.json | python -m json.tool
```

## Version History

### Current Version: [pyproject.toml](../../pyproject.toml) (canonical)

**Features**:

- **23** `GNNFormat` values; **22** registered serializers (see [SPEC.md](SPEC.md))
- Round-trip validation
- Cross-format consistency checking
- Comprehensive error handling

**Known Issues**:

- Some formats (Protobuf, ASN.1) require optional dependencies
- Large models (>100 variables) may be slow to serialize

### Roadmap

- **Next Version**: Parallel processing for multiple files
- **Future**: Incremental parsing, lazy serialization

---

## References

### Related Documentation

- [GNN Syntax Guide](../../docs/gnn/reference/gnn_syntax.md)
- [Pipeline Overview](../../README.md)
- [.agent_rules](../../.agent_rules)

### External Resources

- [GNN Specification](../../docs/gnn/reference/gnn_dsl_manual.md)
- [Active Inference Papers](https://en.wikipedia.org/wiki/Active_inference)

---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
