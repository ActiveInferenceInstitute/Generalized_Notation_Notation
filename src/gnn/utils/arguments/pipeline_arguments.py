"""
Pipeline arguments for the GNN processing pipeline.

Provides the PipelineArguments dataclass, the centralized argument
configuration for the entire pipeline, with post-initialization path
resolution, validation, and dictionary conversion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ONTOLOGY_TERMS_FILE = (
    Path(__file__).resolve().parents[2] / "ontology" / "act_inf_ontology_terms.json"
)
"""Ontology terms file packaged inside the ``gnn`` package (src/gnn/ontology/)."""


@dataclass
class PipelineArguments:
    """Centralized argument configuration for the entire pipeline."""

    # Core directories
    target_dir: Any = field(default_factory=lambda: Path("input/gnn_files"))
    output_dir: Any = field(default_factory=lambda: Path("output"))

    # Processing options
    recursive: bool = True
    verbose: bool = False

    # Opt-in Step 7 interchange metadata and format selection.
    formats: Optional[List[str]] = None
    geo_infer_options_file: Any = None

    # Logging options
    log_format: str = "human"

    # Validation options (enabled by default for comprehensive testing)
    enable_round_trip: bool = True  # Enable round-trip testing across all 21 formats
    enable_cross_format: bool = True  # Enable cross-format consistency validation

    # Step control
    skip_steps: Optional[str] = None
    only_steps: Optional[str] = None
    parallel: bool = False
    autonomous: bool = False
    # V4-STAGE (S2-11): opt-in consolidated in-process execution
    # (docs/decisions/0001-consolidated-pipeline-execution.md)
    consolidated_steps: bool = False

    # Type checking options
    strict: bool = False
    estimate_resources: bool = False
    profile: bool = False
    simulate_error: bool = False
    registry_path: Any = None
    query_ontology: Optional[str] = None

    # File references
    ontology_terms_file: Any = None
    pipeline_summary_file: Any = None

    # LLM options
    llm_tasks: str = "all"
    llm_timeout: Optional[int] = None

    # Setup options
    recreate_venv: bool = False  # Virtual environment recreation flag
    dev: bool = False
    # Step 1: skip post-sync JAX/PyMDP self-test; core deps still sync normally
    setup_core_only: bool = False
    # Optional setup groups to install, used by step 1
    install_optional: bool = False
    optional_groups: Optional[str] = None

    # Execution options
    frameworks: str = "all"
    strict_framework_success: bool = False
    render_output_dir: Any = None
    distributed: bool = False
    execution_workers: int = 1
    backend: str = "ray"
    serialize_preset: str = "full"
    execution_benchmark_repeats: int = 1
    execution_summary_detail: bool = False

    # Audio generation options
    duration: float = 30.0
    audio_backend: str = "auto"
    sonification: bool = True
    full_analysis: bool = False

    # Test options (fast pipeline suite is default for step 2 subprocess)
    fast_only: bool = True
    include_performance: bool = False
    comprehensive: bool = False

    # Skip LLM-powered processing where supported.
    skip_llm: bool = False

    # Advanced visualization options
    viz_type: str = "all"
    interactive: bool = False
    export_formats: Optional[List[str]] = None

    # GUI options
    headless: bool = False
    gui_types: str = "gui_1,gui_2"
    open_browser: bool = False

    # Intelligent analysis options
    analysis_model: Optional[str] = None
    bottleneck_threshold: float = 60.0

    # Step 1: uv sync --all-extras (optional heavy install)
    install_all_extras: bool = False

    # MCP/performance mode (used by step 21) and fine-grained overrides
    performance_mode: str = "low"
    mcp_strict_validation: Optional[bool] = None
    mcp_cache_ttl: Optional[float] = None
    mcp_per_module_timeout: Optional[float] = None
    mcp_overall_timeout: Optional[float] = None
    mcp_modules_allowlist: Optional[str] = None

    # Custom pipeline step configs
    timesteps: Optional[int] = None
    simulation_params: str = "{}"
    timeout: int = 300
    advanced_stats: bool = False
    generate_animations: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation and path resolution."""
        # Ensure Path objects
        if isinstance(self.target_dir, str):
            self.target_dir = Path(self.target_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Set defaults for optional paths
        if self.ontology_terms_file is None:
            self.ontology_terms_file = DEFAULT_ONTOLOGY_TERMS_FILE
        elif isinstance(self.ontology_terms_file, str):
            self.ontology_terms_file = Path(self.ontology_terms_file)

        if self.pipeline_summary_file is None:
            self.pipeline_summary_file = (
                self.output_dir
                / "00_pipeline_summary"
                / "pipeline_execution_summary.json"
            )
        elif isinstance(self.pipeline_summary_file, str):
            self.pipeline_summary_file = Path(self.pipeline_summary_file)

        if self.render_output_dir is not None and isinstance(
            self.render_output_dir, str
        ):
            self.render_output_dir = Path(self.render_output_dir)
        if self.registry_path is not None and isinstance(self.registry_path, str):
            self.registry_path = Path(self.registry_path)

    def validate(self) -> List[str]:
        """Validate argument values and return list of errors."""
        errors: list[Any] = []

        # Check that target directory exists when it is a real filesystem path.
        if not str(self.target_dir).startswith("<"):
            if not self.target_dir.exists():
                errors.append(f"Target directory does not exist: {self.target_dir}")

        # Check that ontology terms file exists when specified as a real path.
        if (
            self.ontology_terms_file
            and not str(self.ontology_terms_file).startswith("<")
            and not self.ontology_terms_file.exists()
        ):
            errors.append(
                f"Ontology terms file does not exist: {self.ontology_terms_file}"
            )

        # Validate LLM timeout
        if self.llm_timeout is not None and self.llm_timeout <= 0:
            errors.append(f"LLM timeout must be positive: {self.llm_timeout}")

        # Validate step lists format
        if self.skip_steps:
            try:
                [s.strip() for s in self.skip_steps.split(",")]
            except Exception:
                errors.append(f"Invalid skip_steps format: {self.skip_steps}")

        if self.only_steps:
            try:
                [s.strip() for s in self.only_steps.split(",")]
            except Exception:
                errors.append(f"Invalid only_steps format: {self.only_steps}")

        if self.execution_benchmark_repeats < 1:
            errors.append(
                f"execution_benchmark_repeats must be >= 1: {self.execution_benchmark_repeats}"
            )

        sp = (self.serialize_preset or "full").strip().lower()
        if sp not in {"full", "minimal"}:
            errors.append(
                f"serialize_preset must be 'full' or 'minimal': {self.serialize_preset!r}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string representation of paths."""
        result: dict[Any, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
