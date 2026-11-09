# GNN MCP Tool Quick Reference

Audit-backed quick reference for the GNN MCP server tool surface. Use `tests/mcp/test_mcp_audit.py` and `src/gnn/mcp/validate_tools.py` for the current live count. For full per-domain documentation see **[../modules/21_mcp.md](../modules/21_mcp.md)**.

**165 tools across 36 modules** — see the generated [`src/gnn/mcp/audit_report.json`](../../../src/gnn/mcp/audit_report.json) for the authoritative current count (regenerate with `uv run python src/gnn/mcp/validate_tools.py`).

## Full Tool Table

| Domain | Tool | Description |
|--------|------|-------------|
| advanced_visualization | `check_visualization_capabilities` | Check available advanced visualization capabilities (D2, dashboard, network backends). |
| advanced_visualization | `get_advanced_visualization_module_info` | Return version, feature flags, and tool inventory of the advanced visualization module. |
| advanced_visualization | `list_d2_visualization_types` | Return all D2 diagram types supported for GNN model visualization. |
| advanced_visualization | `process_advanced_visualization` | Process advanced visualization for GNN files: D2 diagrams, dashboards, network visualizations. |
| analysis | `compute_complexity_metrics` | Compute complexity metrics (variables, connections, cyclomatic complexity) for GNN content. |
| analysis | `evaluate_torchlens_fidelity` | Compare declared GNN connections with TorchLens-traced PyTorch dataflow (read-only). |
| analysis | `get_analysis_results` | Read and return saved analysis results from a previous analysis run. |
| analysis | `list_analysis_tools` | Return information about available GNN analysis tools and capabilities. |
| analysis | `process_analysis` | Run statistical and complexity analysis on GNN files in a directory. |
| api | `gnn_cancel_job` | Cancel a GNN pipeline job. |
| api | `gnn_delete_run` | Delete a pipeline run, cancelling it first when active. |
| api | `gnn_get_job_status` | Retrieve the status of a GNN pipeline job. |
| api | `gnn_get_pipeline_tools` | List available pipeline steps. |
| api | `gnn_list_jobs` | List recent GNN pipeline jobs. |
| api | `gnn_submit_job` | Create a GNN pipeline job record (pending; not executed). Execution happens only via POST /api/v1/process or POST /api/v1/tools/{step} on the job/tool API server (gnn.api.server), or a direct execute_job_async call; gnn serve (Runs surface) never consumes these records. |
| audio | `analyze_audio_characteristics` | Analyse characteristics of a GNN-generated audio file (duration, RMS, spectral centroid, etc.). |
| audio | `check_audio_backends` | Check which audio generation backends (scipy, soundfile, pedalboard, wave) are available. |
| audio | `get_audio_generation_options` | Return all configurable audio generation options with defaults and valid ranges. |
| audio | `get_audio_module_info` | Return version, feature flags, supported backends and formats of the GNN audio module. |
| audio | `process_audio` | Run GNN audio processing pipeline: convert GNN models to audio files. |
| audio | `validate_audio_content` | Validate a GNN-generated audio file: header, sample count, amplitude bounds. |
| audio.sapf | `get_sapf_module_info` | Return metadata about the SAPF audio synthesis module (version, formats, capabilities). |
| audio.sapf | `list_audio_artifacts` | List audio and SAPF script artifacts in an output directory. |
| audio.sapf | `process_sapf` | Generate SAPF audio from GNN Active Inference models using SuperCollider synthesis. |
| cli | `cli.health` | Return CLI module health and list of available subcommands |
| cli | `cli.preflight` | Run pipeline preflight checks and return explicit readiness diagnostics |
| execute | `check_execute_dependencies` | Check which execution backend dependencies (pymdp, numpy, scipy, jax) are installed. |
| execute | `execute_gnn_model` | Execute a single GNN model file via GNNExecutor (PyMDP default); timesteps come from the model's Time section. |
| execute | `execute_pymdp_simulation` | Run a PyMDP Active Inference simulation from a GNN model (A/B/C/D matrices -> Agent -> perception-action loop). |
| execute | `get_doctor_report` | Return one structured capability report: per-framework availability plus a Step 12 execution-readiness dry run (no scripts run). |
| execute | `get_execute_module_info` | Return version, feature flags, and API surface of the GNN execute module. |
| execute | `process_execute` | Run trusted Step 11 rendered scripts listed in render_processing_summary.json. |
| execute | `run_cross_framework_comparison` | Render one GNN model to every registered backend, execute each, and write a cross-framework comparison HTML page with per-framework status receipts. |
| export | `export_single_gnn_file` | Export a single GNN file to one or more target formats. |
| export | `list_export_formats` | List all supported GNN export formats and their descriptions. |
| export | `process_export` | Export GNN models to all supported output formats (JSON, YAML, Python, Julia, etc.). |
| export | `validate_export_format` | Check whether a given export format is supported. |
| extract | `extract_pomdp` | Extract the POMDP state space from a GNN specification file as a versioned JSON payload. |
| gnn | `get_gnn_documentation` | Retrieve the content of a GNN documentation file, schema definition, or grammar specification. |
| gnn | `get_gnn_module_info` | Get comprehensive information about the GNN module capabilities and features. |
| gnn | `get_gnn_schema_info` | Get comprehensive information about the GNN schema structure and validation rules. |
| gnn | `parse_gnn_content` | Parse GNN content with enhanced multi-format support and return structured model representation. |
| gnn | `process_gnn_directory` | Process a directory of GNN files with enhanced validation and testing capabilities. |
| gnn | `run_round_trip_tests` | Run comprehensive round-trip tests on GNN files across all supported formats. |
| gnn | `validate_cross_format_consistency_content` | Validate cross-format consistency for GNN content across all supported formats. |
| gnn | `validate_directory_cross_format_consistency` | Validate cross-format consistency for files in a directory with comprehensive analysis. |
| gnn | `validate_gnn_content` | Enhanced validation of GNN file content with comprehensive testing capabilities. |
| gnn | `validate_schema_definitions_consistency` | Validate consistency between GNN schema definition files across different formats. |
| gui | `get_gui_module_info` | Return version, feature flags, GUI types, and tool inventory of the GUI module. |
| gui | `list_available_guis` | List all available GUI implementations (gui_1, gui_2, gui_3, oxdraw) with capabilities. |
| gui | `process_gui` | Process GUI generation for GNN files: form editors, visual constructors, OxDraw diagram tools. |
| gui.oxdraw | `oxdraw.check_installation` | Check if oxdraw CLI is installed and available |
| gui.oxdraw | `oxdraw.convert_from_mermaid` | Convert Mermaid flowchart edited in oxdraw back to GNN format |
| gui.oxdraw | `oxdraw.convert_to_mermaid` | Convert a GNN Active Inference model to Mermaid flowchart format for visual editing in oxdraw |
| gui.oxdraw | `oxdraw.get_info` | Get oxdraw integration module information and capabilities |
| gui.oxdraw | `oxdraw.launch_editor` | Launch interactive oxdraw editor for visual GNN model construction |
| integration | `check_integration_dependencies` | Check which third-party integration dependencies (pymdp, JAX, Julia, etc.) are installed. |
| integration | `get_integration_status` | Check status and inventory of a previous integration run. |
| integration | `list_supported_integrations` | Return all supported third-party integration targets and their availability. |
| integration | `process_integration` | Run GNN integration processing: export to ActiveInference.jl, pymdp, Pyro, Stan, etc. |
| intelligent_analysis | `get_analysis_capabilities` | Get intelligent analysis capabilities, supported types, available tools, and feature flags. |
| intelligent_analysis | `get_intelligent_analysis_module_info` | Return version, feature flags, and tool inventory of the intelligent analysis module. |
| intelligent_analysis | `process_intelligent_analysis` | Process AI-powered intelligent analysis of GNN pipeline results: failure analysis, performance optimization. |
| llm | `analyze_gnn_with_llm` | Run LLM-based analysis on a single GNN model file: summary, complexity, connections. |
| llm | `generate_llm_documentation` | Generate Markdown or text documentation from deterministic GNN structural analysis. |
| llm | `get_llm_module_info` | Return version, analysis types, output formats, and tool list of the GNN LLM module. |
| llm | `get_llm_providers` | Return available LLM providers and their API key / configuration status. |
| llm | `process_llm` | Run LLM analysis pipeline for all GNN files in a directory. |
| mcp | `mcp.list_available_resources` | List all resources currently registered with the MCP registry. |
| mcp | `mcp.list_available_tools` | List all tools currently registered with the MCP registry. |
| mcp.sympy_mcp | `sympy_analyze_stability` | Analyze system stability using eigenvalue analysis |
| mcp.sympy_mcp | `sympy_cleanup` | Clean up SymPy MCP integration and reset state |
| mcp.sympy_mcp | `sympy_get_latex` | Convert a mathematical expression to LaTeX format |
| mcp.sympy_mcp | `sympy_initialize` | Initialize SymPy MCP integration |
| mcp.sympy_mcp | `sympy_simplify_expression` | Simplify a mathematical expression to canonical form |
| mcp.sympy_mcp | `sympy_solve_equation` | Solve an equation algebraically for a specified variable |
| mcp.sympy_mcp | `sympy_validate_equation` | Validate a mathematical equation using SymPy symbolic processing |
| mcp.sympy_mcp | `sympy_validate_matrix` | Validate matrix properties including stochasticity constraints |
| meta | `get_mcp_diagnostics` | Get comprehensive diagnostic information for troubleshooting and monitoring, including health checks and recommendations. |
| meta | `get_mcp_module_info` | Get detailed information about a specific loaded module, including its tools and resources. |
| meta | `get_mcp_performance_metrics` | Get performance metrics and statistics for the MCP server, including execution times and error rates. |
| meta | `get_mcp_server_auth_status` | Describes the current authentication mechanisms and security configuration of the MCP server. |
| meta | `get_mcp_server_capabilities` | Retrieves the full capabilities description of this MCP server, including all tools and resources. |
| meta | `get_mcp_server_encryption_status` | Describes the current encryption status for server transport and data handling with security recommendations. |
| meta | `get_mcp_server_status` | Provides comprehensive operational status of the MCP server, including uptime, modules, and performance metrics. |
| meta | `get_mcp_tool_categories` | Get tools organized by category for easier discovery and navigation. |
| ml_integration | `check_ml_frameworks` | Check available ML frameworks (PyTorch, TensorFlow, JAX, scikit-learn) and their versions. |
| ml_integration | `get_ml_module_info` | Return version, feature flags, and tool inventory of the ML integration module. |
| ml_integration | `list_ml_integration_targets` | Return GNN-compatible ML integration targets and their dependency availability. |
| ml_integration | `process_ml_integration` | Process ML integration for GNN files: model training, inference setup, and framework export. |
| model_registry | `model_registry.get_model` | Get a model from the registry by ID |
| model_registry | `model_registry.list_models` | List all models in the registry |
| model_registry | `model_registry.register_model` | Register a model in the model registry |
| model_registry | `model_registry.search_models` | Search models in the registry by name, description, or tags |
| multimodel | `generate_dependency_graph` | Render the inter-model dependency graph of a GNN file as a Mermaid diagram or text adjacency list. |
| ontology | `extract_ontology_annotations` | Extract ActInfOntologyAnnotation variable-to-term mappings from GNN model content. |
| ontology | `list_standard_ontology_terms` | Return the canonical list of Active Inference Ontology (ActInfO) terms and descriptions. |
| ontology | `process_ontology` | Map GNN variables to Active Inference Ontology terms and produce an ontology report. |
| ontology | `validate_ontology_terms` | Validate ontology term names against the Active Inference Ontology. |
| pipeline | `get_pipeline_config_info` | Get detailed pipeline configuration information and settings. |
| pipeline | `get_pipeline_status` | Get current pipeline execution status, recent logs, and execution statistics. |
| pipeline | `get_pipeline_steps` | Get information about all available pipeline steps, their metadata, and dependencies. |
| pipeline | `get_v3_orchestration_capabilities` | Describe the v3.0.0 long-running orchestration contracts: durable observation streams, resumable run sessions, and auditable container plans (safe-by-design, no live mutation). |
| pipeline | `list_step_artifacts` | List the files each pipeline step wrote under the pipeline output root (registry-complete, per-step file cap, optional step number filter). |
| pipeline | `run_v3_container_security_review` | Run the auditable container-plan static security review on a hardened and an insecure example, proving the review flags privileged/root/unpinned/secret findings. |
| pipeline | `run_v3_orchestration_self_check` | Run in-process checks of all three v3 orchestration contracts (stream manifest tamper detection, session status math, container review) and report pass counts. |
| pipeline | `validate_pipeline_dependencies` | Validate pipeline step dependencies and identify missing or circular dependencies. |
| processing | `processing.check_gnn_file_structure` | Validate the structure of a GNN file with lightweight checks (empty/short content, missing sections, unmatched brackets). |
| processing | `processing.discover_gnn_files` | Discover GNN model files in a directory and return their paths. |
| processing | `processing.parse_gnn_file` | Parse a GNN file and extract basic structural information (sections, variables, counts). |
| render | `get_render_module_info` | Return metadata about the render module: supported frameworks and input/output formats. |
| render | `list_render_frameworks` | Return supported render framework names and availability (best effort). |
| render | `process_render` | Render GNN models in a directory to all supported code frameworks. |
| render | `render_gnn_to_format` | Render a single GNN file to exactly the requested framework with current artifacts. |
| render | `render_spec_to_format` | Render a single GNN file to exactly one framework via render_gnn_spec. |
| report | `generate_report` | Generate a comprehensive pipeline execution report with statistics and summaries. |
| report | `get_report_module_info` | Return metadata about the report module (version, supported formats). |
| report | `list_report_formats` | Return all supported report output formats (JSON, HTML, Markdown, etc.). |
| report | `process_report` | Run the full report pipeline step. |
| report | `read_report` | Read and return the contents of a generated pipeline report file. |
| research | `get_research_module_info` | Return metadata about the research module capabilities. |
| research | `list_research_topics` | Return Active Inference and GNN research topic taxonomy. |
| research | `process_research` | Run GNN research processing: generate experiment metadata and cross-references. |
| research | `read_research_results` | Read and return research output files from a previous research processing run. |
| schema_validator | `schema_validator.parse_syntax` | Parse a GNN specification file with the regex-based syntax parser and return its structural summary. |
| schema_validator | `schema_validator.validate_comprehensive` | Validate a GNN specification file with the full validator pipeline (schema, round-trip, and semantic checks). |
| security | `get_security_report` | Read and return saved security scan reports from a previous security processing run. |
| security | `list_security_checks` | Return the list of security checks performed (CVE scan, injection detection, path traversal, etc.). |
| security | `process_security` | Run security scanning and compliance checks on GNN pipeline files. |
| security | `scan_gnn_file` | Perform a lightweight security scan of a single GNN file for injection patterns. |
| setup | `check_uv_project_status` | Checks the status of a UV project including pyproject.toml, uv.lock, and virtual environment. |
| setup | `ensure_directory_exists` | Ensures a directory exists, creating it if necessary. Returns the absolute path. |
| setup | `find_project_gnn_files` | Finds all GNN (.md) files in a specified directory within the project. |
| setup | `get_standard_output_paths` | Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed. |
| setup | `get_uv_environment_info` | Gets information about the current UV environment including paths and status. |
| setup | `install_uv_dependency` | Installs a dependency using UV with optional extras support. |
| setup | `setup_uv_project_structure` | Sets up a new UV project structure with standard directories and configuration. |
| setup | `sync_uv_dependencies` | Syncs dependencies using UV from pyproject.toml and updates the lock file. |
| template | `template.get_info` | Get information about the template step |
| template | `template.list` | List maintained templates with checksums (CLI `gnn templates list` parity) |
| template | `template.process_directory` | Process all files in a directory using the template processor |
| template | `template.process_file` | Process a file using the template processor |
| template | `template.pull` | Pull a maintained template into an output directory |
| template | `template.show` | Show one maintained template record with checksum metadata (CLI `gnn templates show` parity) |
| type_checker | `validate_gnn_files` | Validate GNN files for syntax and type correctness. |
| type_checker | `validate_single_gnn_file` | Validate a single GNN file for syntax and type correctness. |
| utils | `get_environment_info` | Get environment information including Python packages, environment variables, and paths. |
| utils | `get_file_info` | Get detailed information about a file or directory including size, permissions, and contents. |
| utils | `get_logging_info` | Get current logging configuration and status for all loggers. |
| utils | `get_system_info` | Get comprehensive system information including CPU, memory, disk, and platform details. |
| utils | `validate_dependencies` | Validate system dependencies and required packages for the GNN pipeline. |
| validation | `check_schema_compliance` | Check a GNN model string against canonical GNN schema requirements. |
| validation | `get_validation_report` | Read and return saved validation reports from a previous validation run. |
| validation | `process_validation` | Run full GNN validation pipeline on a directory of GNN files. |
| validation | `validate_gnn_file` | Validate a single GNN file at a given level (basic/standard/strict). |
| visualization | `get_visualization_module_info` | Return metadata about the visualization module (version, backends, output formats). |
| visualization | `get_visualization_options` | Return available visualization types and their configuration options. |
| visualization | `list_visualization_artifacts` | List all visualization artifacts (PNG, SVG, HTML, PDF) in an output directory. |
| visualization | `process_visualization` | Generate static PNGs/SVGs for all GNN models (state-space, connection matrix, parameters). |
| website | `build_website_from_pipeline_output` | Build the full GNN website by auto-discovering all pipeline artifacts from numbered output directories. |
| website | `get_website_module_info` | Return metadata about the website module: version, supported file types, and available MCP tools. |
| website | `get_website_page` | Read one generated website page's HTML content by catalogue page key, with a character cap for large pages. |
| website | `get_website_status` | Inspect an existing generated website: list pages, sizes, and check completeness of key pages. |
| website | `list_generated_website_pages` | List all HTML pages in a generated website directory with sizes and timestamps. |
| website | `process_website` | Generate a premium 7-page static HTML website from GNN pipeline artifacts. |

Use `tests/mcp/test_mcp_audit.py` for the current registered tool/resource contract. The audit verifies module discovery, callable tools, non-empty module/category metadata, canonical JSON schemas, and the parent GUI exposure of nested `oxdraw.*` tools.
