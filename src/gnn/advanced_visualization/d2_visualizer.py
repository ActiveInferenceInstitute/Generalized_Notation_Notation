#!/usr/bin/env python3
"""
D2 Visualizer Module for GNN Pipeline

This module provides D2 (Declarative Diagramming) integration for the GNN pipeline,
enabling generation of professional diagrams from GNN model specifications.

Features:
- Convert GNN models to D2 diagram specifications
- Generate pipeline architecture diagrams
- Create Active Inference concept diagrams
- Visualize state spaces and transitions
- Generate framework integration mappings
- Compile D2 files to SVG/PNG/PDF formats
"""

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

D2_COMPILE_TIMEOUT_S = 30
D2_MISSING_MESSAGE = "D2 CLI not available. Install from https://d2lang.com"
VALID_D2_FORMATS = ("svg", "png", "pdf")


def _resolve_d2_binary() -> Optional[str]:
    """Resolve the ``d2`` CLI binary path from PATH (None when absent)."""
    return shutil.which("d2")


@dataclass
class D2DiagramSpec:
    """Specification for a D2 diagram"""

    name: str
    description: str
    d2_content: str
    output_formats: List[str] = field(default_factory=lambda: ["svg"])
    layout_engine: str = "elk"  # dagre, elk, tala
    theme: int = 1
    dark_theme: Optional[int] = None
    sketch_mode: bool = False
    pad: int = 20
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class D2GenerationResult:
    """Result of D2 diagram generation"""

    success: bool
    diagram_name: str
    d2_file: Optional[Path] = None
    output_files: List[Path] = field(default_factory=list)
    compilation_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class D2Visualizer:
    """
    D2 diagram generator for GNN models.

    This class handles conversion of GNN specifications to D2 diagram format
    and manages compilation to various output formats.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize D2 visualizer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.d2_available = self._check_d2_availability()

        if not self.d2_available:
            self.logger.warning(D2_MISSING_MESSAGE)

    def _check_d2_availability(self) -> bool:
        """Check if D2 CLI is available in system PATH"""
        return _resolve_d2_binary() is not None

    def generate_model_structure_diagram(
        self, model_data: Dict[str, Any], output_name: Optional[str] = None
    ) -> D2DiagramSpec:
        """
        Generate D2 diagram for GNN model structure.

        Args:
            model_data: Parsed GNN model data
            output_name: Optional custom name for diagram

        Returns:
            D2DiagramSpec with diagram definition
        """
        model_name = model_data.get("model_name", "Unknown Model")
        safe_name = output_name or self._sanitize_name(model_name)

        # Extract state space and connections
        state_space = model_data.get("state_space", {})
        connections = model_data.get("connections", [])
        annotations = model_data.get("actinf_annotations", {})

        # Build D2 content
        d2_lines: list[Any] = [
            f"# GNN Model: {model_name}",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            f"{safe_name}: {{",
            "  direction: down",
            "",
            "  # State Space Components",
            "  state_space: State Space {",
            "    direction: right",
            "",
        ]

        # Add state space variables
        for var_name, var_info in state_space.items():
            shape = self._get_d2_shape_for_variable(var_name, var_info, annotations)
            label = self._format_variable_label(var_name, var_info, annotations)

            d2_lines.extend(
                [
                    f"    {var_name}: {self._d2_label(label)} {{",
                    f"      shape: {shape}",
                    "    }",
                    "",
                ]
            )

        d2_lines.append("  }")
        d2_lines.append("")

        # Add connections
        if connections:
            d2_lines.append("  # Model Connections")
            for conn in connections:
                source = conn.get("source", "")
                target = conn.get("target", "")
                conn_type = conn.get("type", "->")
                label = conn.get("label", "")

                if source and target:
                    arrow = self._get_d2_arrow(conn_type)
                    if label:
                        d2_lines.append(
                            f"  state_space.{source} {arrow} state_space.{target}: "
                            f"{self._d2_label(label)}"
                        )
                    else:
                        d2_lines.append(
                            f"  state_space.{source} {arrow} state_space.{target}"
                        )
            d2_lines.append("")

        # Add Active Inference annotations
        if annotations:
            d2_lines.append("  # Active Inference Ontology")
            d2_lines.append("  annotations: Ontology Mapping {")
            d2_lines.append("    shape: document")
            d2_lines.append("    label: |md")
            d2_lines.append("      # Active Inference Concepts")
            d2_lines.append("")
            for var, concept in annotations.items():
                d2_lines.append(f"      - **{var}**: {concept}")
            d2_lines.append("    |")
            d2_lines.append("  }")
            d2_lines.append("")

        d2_lines.append("}")

        d2_content = "\n".join(d2_lines)

        return D2DiagramSpec(
            name=f"{safe_name}_structure",
            description=f"GNN Model Structure for {model_name}",
            d2_content=d2_content,
            layout_engine="elk",
            theme=1,
            metadata={"model_name": model_name, "type": "structure"},
        )

    def generate_pomdp_diagram(
        self, model_data: Dict[str, Any], output_name: Optional[str] = None
    ) -> D2DiagramSpec:
        """
        Generate D2 diagram for POMDP (Active Inference) structure.

        Args:
            model_data: Parsed GNN model data with POMDP components
            output_name: Optional custom name for diagram

        Returns:
            D2DiagramSpec with POMDP diagram definition
        """
        model_name = model_data.get("model_name", "POMDP Agent")
        safe_name = output_name or self._sanitize_name(model_name)

        state_space = model_data.get("state_space", {})

        # Identify POMDP components
        matrices: dict[Any, Any] = {}
        vectors: dict[Any, Any] = {}

        for var_name, var_info in state_space.items():
            dims = var_info.get("dimensions", [])
            if len(dims) == 2 and dims[0] > 1 and dims[1] > 1:
                matrices[var_name] = var_info
            elif len(dims) == 1 or (len(dims) == 2 and (dims[0] == 1 or dims[1] == 1)):
                vectors[var_name] = var_info
            elif len(dims) >= 3:
                matrices[var_name] = var_info  # Tensor treated as matrix

        # Build D2 content for POMDP structure
        d2_lines: list[Any] = [
            f"# Active Inference POMDP: {model_name}",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "Active Inference POMDP Agent: {",
            "  direction: down",
            "",
            "  # Generative Model Components",
            "  generative_model: Generative Model {",
            "    direction: right",
            "",
        ]

        # Add matrices (A, B, etc.)
        for var_name, var_info in matrices.items():
            dims = var_info.get("dimensions", [])
            dims_str = "×".join(map(str, dims))
            label = f"{var_name} [{dims_str}]"

            d2_lines.extend(
                [
                    f"    {var_name}: {self._d2_label(label)} {{",
                    "      shape: hexagon",
                    f"      tooltip: {self._d2_label(var_info.get('description', 'POMDP matrix'))}",
                    "    }",
                    "",
                ]
            )

        d2_lines.append("  }")
        d2_lines.append("")

        # Add state inference
        d2_lines.extend(
            [
                "  # Inference Process",
                "  inference: Inference Engine {",
                "    direction: right",
                "",
                "    state_inference: State Inference {",
                "      shape: diamond",
                "      label: infer_states()",
                "    }",
                "",
                "    policy_inference: Policy Selection {",
                "      shape: diamond",
                "      label: infer_policies()",
                "    }",
                "",
                "    action_selection: Action Sampling {",
                "      shape: diamond",
                "      label: sample_action()",
                "    }",
                "  }",
                "",
                "  # Data Flow",
                "  generative_model -> inference: Model-based inference",
                "  inference -> generative_model: Belief updates",
                "}",
            ]
        )

        d2_content = "\n".join(d2_lines)

        return D2DiagramSpec(
            name=f"{safe_name}_pomdp",
            description=f"POMDP Structure for {model_name}",
            d2_content=d2_content,
            layout_engine="elk",
            theme=1,
            metadata={"model_name": model_name, "type": "pomdp"},
        )

    def generate_pipeline_flow_diagram(
        self, include_frameworks: bool = True
    ) -> D2DiagramSpec:
        """
        Generate D2 diagram for GNN pipeline architecture and data flow.

        Args:
            include_frameworks: Include framework execution details

        Returns:
            D2DiagramSpec with pipeline flow diagram
        """
        d2_lines: list[Any] = [
            "# GNN Pipeline Data Flow",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "GNN Pipeline: {",
            "  direction: down",
            "",
            "  # Input Stage",
            "  input: Input {",
            "    direction: right",
            "    gnn_files: GNN Files {",
            "      shape: document",
            "      label: *.md specifications",
            "    }",
            "    config: Configuration {",
            "      shape: document",
            "      label: config.yaml",
            "    }",
            "  }",
            "",
            "  # Core Processing",
            "  processing: Core Processing {",
            "    direction: right",
            "",
            "    parse: GNN Parsing {",
            "      shape: rectangle",
            "      label: Step 3: Parse & Serialize\\n22 formats",
            "    }",
            "",
            "    validate: Validation {",
            "      shape: rectangle",
            "      label: Steps 5-6: Type check &\\nconsistency validation",
            "    }",
            "",
            "    export: Export {",
            "      shape: rectangle",
            "      label: Step 7: Multi-format export\\nJSON, XML, GraphML...",
            "    }",
            "",
            "    visualize: Visualization {",
            "      shape: rectangle",
            "      label: Steps 8-9: Graph & matrix\\nvisualization",
            "    }",
            "",
            "    parse -> validate -> export -> visualize",
            "  }",
            "",
        ]

        if include_frameworks:
            d2_lines.extend(
                [
                    "  # Framework Generation",
                    "  generation: Code Generation {",
                    "    direction: right",
                    "",
                    "    render: Framework Rendering {",
                    "      shape: hexagon",
                    "      label: Step 11: Generate code for\\nPyMDP, RxInfer.jl, etc.",
                    "    }",
                    "",
                    "    execute: Simulation Execution {",
                    "      shape: hexagon",
                    "      label: Step 12: Run simulations\\nwith result capture",
                    "    }",
                    "",
                    "    render -> execute",
                    "  }",
                    "",
                    "  # Analysis",
                    "  analysis: Analysis & Output {",
                    "    direction: right",
                    "",
                    "    llm: LLM Analysis {",
                    "      shape: cloud",
                    "    }",
                    "",
                    "    ml_integration: ML Integration {",
                    "      shape: cloud",
                    "    }",
                    "",
                    "    report: Final Report {",
                    "      shape: document",
                    "    }",
                    "",
                    "    llm -> ml_integration -> report",
                    "  }",
                    "",
                    "  # Main Flow",
                    "  input -> processing: Input data",
                    "  processing -> generation: Parsed models",
                    "  generation -> analysis: Simulation results",
                    "}",
                ]
            )
        else:
            d2_lines.append("  input -> processing")
            d2_lines.append("}")

        d2_content = "\n".join(d2_lines)

        return D2DiagramSpec(
            name="gnn_pipeline_flow",
            description="GNN Pipeline Data Flow and Architecture",
            d2_content=d2_content,
            layout_engine="elk",
            theme=1,
            metadata={"type": "pipeline_architecture"},
        )

    def generate_framework_mapping_diagram(
        self, frameworks: Optional[List[str]] = None
    ) -> D2DiagramSpec:
        """
        Generate D2 diagram showing framework integration mapping.

        Args:
            frameworks: List of frameworks to include (default: all)

        Returns:
            D2DiagramSpec with framework mapping diagram
        """
        if frameworks is None:
            frameworks = [
                "pymdp",
                "rxinfer",
                "activeinference_jl",
                "discopy",
                "jax",
            ]

        d2_lines: list[Any] = [
            "# GNN Framework Integration",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "Framework Integration: {",
            "  direction: down",
            "",
            "  gnn_model: GNN Specification {",
            "    shape: document",
            "    label: Active Inference Model",
            "  }",
            "",
            "  render_step: Code Generation {",
            "    direction: right",
            "",
        ]

        # Framework definitions
        framework_info: dict[str, Any] = {
            "pymdp": ("Python Active Inference", "rectangle"),
            "rxinfer": ("Julia Reactive Inference", "rectangle"),
            "activeinference_jl": ("Julia Active Inference", "rectangle"),
            "discopy": ("Python Categorical Diagrams", "rectangle"),
            "jax": ("Python HPC Simulation", "rectangle"),
        }

        for fw in frameworks:
            if fw in framework_info:
                label, shape = framework_info[fw]
                d2_lines.extend(
                    [
                        f"    {fw}: {fw.upper()} {{",
                        f"      shape: {shape}",
                        f"      label: {self._d2_label(label)}",
                        "    }",
                        "",
                    ]
                )

        d2_lines.extend(
            [
                "  }",
                "",
                "  execution: Simulation Execution {",
                "    direction: right",
                "",
            ]
        )

        for fw in frameworks:
            d2_lines.append(f"    {fw}_exec: {fw.upper()} Simulation")

        d2_lines.extend(
            [
                "  }",
                "",
                "  # Connections",
                "  gnn_model -> render_step: GNN → Code Generation",
                "  render_step -> execution: Generated Code → Execution",
                "}",
            ]
        )

        d2_content = "\n".join(d2_lines)

        return D2DiagramSpec(
            name="framework_integration",
            description="GNN Framework Integration Mapping",
            d2_content=d2_content,
            layout_engine="elk",
            theme=1,
            metadata={"type": "framework_mapping", "frameworks": frameworks},
        )

    def generate_active_inference_concepts_diagram(self) -> D2DiagramSpec:
        """
        Generate D2 diagram explaining Active Inference concepts.

        Returns:
            D2DiagramSpec with Active Inference conceptual diagram
        """
        d2_content = """# Active Inference Free Energy Principle
# Generated conceptual diagram

Active Inference Free Energy Principle: {
  direction: down

  agent: Cognitive Agent {
    shape: person
    label: Active Inference Agent
  }

  world: External World {
    shape: cloud
    label: Environment
  }

  generative_model: Generative Model {
    direction: right

    prior: Prior Beliefs {
      shape: diamond
      label: P(s,π)
    }

    likelihood: Likelihood {
      shape: diamond
      label: P(o|s)
    }

    preferences: Preferences {
      shape: diamond
      label: P(π)
    }
  }

  inference: Inference Process {
    direction: right

    perception: State Inference {
      shape: hexagon
      label: Minimize VFE\\nVariational Free Energy
    }

    action: Policy Selection {
      shape: hexagon
      label: Minimize EFE\\nExpected Free Energy
    }
  }

  # Information Flow
  agent -> generative_model: Internal model
  world -> agent: Observations o
  agent -> world: Actions u
  generative_model -> inference: Model-based inference
  inference -> agent: Belief updates & action selection
  inference -> generative_model: Update beliefs
}
"""

        return D2DiagramSpec(
            name="active_inference_concepts",
            description="Active Inference Free Energy Principle",
            d2_content=d2_content,
            layout_engine="elk",
            theme=1,
            metadata={"type": "conceptual", "domain": "active_inference"},
        )

    def compile_d2_diagram(
        self, spec: D2DiagramSpec, output_dir: Path, formats: Optional[List[str]] = None
    ) -> D2GenerationResult:
        """
        Compile D2 diagram specification to output formats.

        Each format compiles through the shared subprocess envelope
        (``run_subprocess_envelope``) with a wall-clock force-kill timeout;
        failures classify into timeout / non-zero-exit / never-started
        warnings.

        Args:
            spec: D2DiagramSpec to compile
            output_dir: Directory for output files
            formats: List of output formats (svg, png, pdf)

        Returns:
            D2GenerationResult with compilation results
        """
        start_time = time.time()

        if not self.d2_available:
            return D2GenerationResult(
                success=False,
                diagram_name=spec.name,
                error_message=D2_MISSING_MESSAGE,
            )

        formats = formats or spec.output_formats
        # Drop unsupported formats early rather than passing arbitrary suffixes to the CLI.
        formats = [fmt for fmt in formats if fmt in VALID_D2_FORMATS]
        if not formats:
            formats = list(VALID_D2_FORMATS[:2])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write D2 source file (atomic-ish: temp file then os.replace; clean up on failure)
        d2_file = output_dir / f"{spec.name}.d2"
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=d2_file.parent, delete=False
            ) as tmp_f:
                tmp_f.write(spec.d2_content)
                tmp_path = Path(tmp_f.name)
            os.replace(tmp_path, str(d2_file))
        except Exception as e:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return D2GenerationResult(
                success=False,
                diagram_name=spec.name,
                error_message=f"Failed to write D2 file: {e}",
            )

        output_files: list[Any] = []
        warnings: list[Any] = []

        # Lazy import: pulling gnn.execute at module scope would drag its
        # executor/processor/validator package onto this module's import
        # chain; the envelope is only needed once compilation actually runs.
        from gnn.execute.subprocess_envelope import run_subprocess_envelope

        d2_binary = _resolve_d2_binary()
        if d2_binary is None:
            return D2GenerationResult(
                success=False,
                diagram_name=spec.name,
                error_message=D2_MISSING_MESSAGE,
            )

        # Compile to each format
        for fmt in formats:
            output_file = output_dir / f"{spec.name}.{fmt}"

            cmd: list[Any] = [
                d2_binary,
                f"--layout={spec.layout_engine}",
                f"--theme={spec.theme}",
                f"--pad={spec.pad}",
            ]

            if spec.dark_theme is not None:
                cmd.append(f"--dark-theme={spec.dark_theme}")

            if spec.sketch_mode:
                cmd.append("--sketch")

            cmd.extend([str(d2_file), str(output_file)])

            envelope = run_subprocess_envelope(cmd, timeout=D2_COMPILE_TIMEOUT_S)
            if envelope["success"]:
                output_files.append(output_file)
                self.logger.info(f"Generated {fmt}: {output_file}")
            elif envelope.get("error_type") == "TimeoutExpired":
                warning = f"Timeout compiling to {fmt}"
                warnings.append(warning)
                self.logger.warning(warning)
            else:
                detail = (
                    envelope.get("stderr")
                    or envelope.get("error")
                    or f"exit code {envelope.get('return_code')}"
                )
                warning = f"Failed to generate {fmt}: {detail}"
                warnings.append(warning)
                self.logger.warning(warning)

        compilation_time = time.time() - start_time

        return D2GenerationResult(
            success=len(output_files) > 0,
            diagram_name=spec.name,
            d2_file=d2_file,
            output_files=output_files,
            compilation_time=compilation_time,
            warnings=warnings,
        )

    def generate_all_diagrams_for_model(
        self,
        model_data: Dict[str, Any],
        output_dir: Path,
        formats: Optional[List[str]] = None,
    ) -> List[D2GenerationResult]:
        """
        Generate all applicable D2 diagrams for a GNN model.

        Args:
            model_data: Parsed GNN model data
            output_dir: Directory for output files
            formats: List of output formats

        Returns:
            List of D2GenerationResult for each generated diagram
        """
        results: list[Any] = []

        # Generate model structure diagram
        try:
            struct_spec = self.generate_model_structure_diagram(model_data)
            struct_result = self.compile_d2_diagram(struct_spec, output_dir, formats)
            results.append(struct_result)
        except Exception as e:
            self.logger.error(f"Failed to generate structure diagram: {e}")
            results.append(
                D2GenerationResult(
                    success=False, diagram_name="structure", error_message=str(e)
                )
            )

        # Generate POMDP diagram if applicable
        try:
            if self._is_pomdp_model(model_data):
                pomdp_spec = self.generate_pomdp_diagram(model_data)
                pomdp_result = self.compile_d2_diagram(pomdp_spec, output_dir, formats)
                results.append(pomdp_result)
        except Exception as e:
            self.logger.error(f"Failed to generate POMDP diagram: {e}")

        return results

    def _is_pomdp_model(self, model_data: Dict[str, Any]) -> bool:
        """Check if model appears to be a POMDP/Active Inference model"""
        state_space = model_data.get("state_space", {})
        annotations = model_data.get("actinf_annotations", {})

        # Check for typical POMDP matrices
        pomdp_indicators: list[Any] = ["A", "B", "C", "D", "E", "F", "G"]
        has_pomdp_vars = any(var in state_space for var in pomdp_indicators)
        has_actinf_annotations = bool(annotations)

        return has_pomdp_vars or has_actinf_annotations

    def _d2_label(self, text: str) -> str:
        """Quote a D2 label when it carries characters D2 treats as reserved.

        D2 accepts unquoted labels with spaces and most punctuation, but ``["
        and ``]`` delimit arrays (the generator's dimension notation ``A
        [3×3]``) and ``{``/``}`` delimit blocks, so labels containing those
        characters must be double-quoted to render literally. Embedded double
        quotes are escaped; everything else is left verbatim.
        """
        if any(ch in text for ch in "[]{}"):
            return '"' + text.replace('"', '\\"') + '"'
        return text

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use in D2 identifiers"""

        # Replace spaces and special chars with underscores
        sanitized = re.sub(r"[^\w\s-]", "", name)
        sanitized = re.sub(r"[-\s]+", "_", sanitized)
        return sanitized.lower()

    def _get_d2_shape_for_variable(
        self, var_name: str, var_info: Dict[str, Any], annotations: Dict[str, str]
    ) -> str:
        """Determine appropriate D2 shape for a variable"""
        # Check Active Inference ontology
        concept = annotations.get(var_name, "")

        if "Matrix" in concept:
            return "hexagon"
        elif "Vector" in concept or "Preference" in concept:
            return "diamond"
        elif "State" in concept:
            return "cylinder"
        elif "Observation" in concept:
            return "circle"
        elif "Action" in concept:
            return "square"
        elif "Policy" in concept:
            return "parallelogram"

        # Recovery based on dimensions
        dims = var_info.get("dimensions", [])
        if len(dims) == 2 and dims[0] > 1 and dims[1] > 1:
            return "hexagon"  # Matrix
        elif len(dims) == 1 or (len(dims) == 2 and (dims[0] == 1 or dims[1] == 1)):
            return "diamond"  # Vector

        return "rectangle"

    def _format_variable_label(
        self, var_name: str, var_info: Dict[str, Any], annotations: Dict[str, str]
    ) -> str:
        """Format variable label for D2 display"""
        dims = var_info.get("dimensions", [])
        var_info.get("type", "")
        concept = annotations.get(var_name, "")

        dims_str = "×".join(map(str, dims)) if dims else ""

        if concept:
            if dims_str:
                return f"{var_name} [{dims_str}]\\n{concept}"
            return f"{var_name}\\n{concept}"
        else:
            if dims_str:
                return f"{var_name} [{dims_str}]"
            return var_name

    def _get_d2_arrow(self, conn_type: str) -> str:
        """Convert connection type to D2 arrow notation"""
        arrow_map: dict[str, Any] = {
            "->": "->",
            "<-": "<-",
            "<->": "<->",
            "-": "--",
            ">": "->",
            "<": "<-",
        }
        return cast("str", arrow_map.get(conn_type, "->"))


def process_gnn_file_with_d2(
    gnn_file: Path,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    formats: Optional[List[str]] = None,
    parsed_json_dir: Optional[Path] = None,
) -> List[D2GenerationResult]:
    """
    Process a GNN file and generate D2 diagrams.

    Args:
        gnn_file: Path to GNN file
        output_dir: Output directory for diagrams
        logger: Optional logger instance
        formats: List of output formats (svg, png, pdf)
        parsed_json_dir: Directory containing Step 3 parsed-model JSON
            (``<model>/<model>_parsed.json``). When omitted, falls back to the
            historical cwd-relative ``output/3_gnn_output`` lookup.

    Returns:
        List of D2GenerationResult for generated diagrams
    """
    logger = logger or logging.getLogger(__name__)

    # Try to load parsed model data
    model_data = None

    # Look for parsed JSON from GNN processing step
    gnn_output_dir = (
        Path(parsed_json_dir)
        if parsed_json_dir is not None
        else Path("output/3_gnn_output")
    )

    if gnn_output_dir.exists():
        model_name = gnn_file.stem
        parsed_json = gnn_output_dir / model_name / f"{model_name}_parsed.json"

        if parsed_json.exists():
            try:
                with open(parsed_json) as f:
                    model_data = json.load(f)
                logger.info(f"Loaded parsed model data from {parsed_json}")
            except Exception as e:
                logger.warning(f"Failed to load parsed JSON: {e}")

    # Parse GNN file directly when Step 3 artifacts are unavailable.
    if model_data is None:
        try:
            from gnn import parse_gnn_file

            model_data = parse_gnn_file(gnn_file)
            logger.info(f"Parsed GNN file directly: {gnn_file}")
        except Exception as e:
            logger.error(f"Failed to parse GNN file: {e}")
            return [
                D2GenerationResult(
                    success=False,
                    diagram_name=gnn_file.stem,
                    error_message=f"Failed to parse GNN file: {e}",
                )
            ]

    # Generate D2 diagrams
    visualizer = D2Visualizer(logger=logger)
    results = visualizer.generate_all_diagrams_for_model(
        model_data, output_dir, formats
    )

    return results
