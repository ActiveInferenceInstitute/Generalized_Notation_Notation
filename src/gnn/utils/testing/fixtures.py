"""Shared test fixtures: standard args, sample pipeline arguments, sample GNN
content, and file-creation helpers.

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

from gnn.utils.arguments.pipeline_arguments import DEFAULT_ONTOLOGY_TERMS_FILE
from gnn.utils.testing.constants import PROJECT_ROOT, TEST_CONFIG


def get_test_args() -> Dict[str, Any]:
    """Get standard test arguments."""
    return {
        "target_dir": str(PROJECT_ROOT / "input" / "gnn_files"),
        "output_dir": str(PROJECT_ROOT / "output"),
        "verbose": True,
        "recursive": True,
        "strict": False,
        "estimate_resources": True,
        "enable_round_trip": True,
        "enable_cross_format": True,
        "llm_tasks": "all",
        "llm_timeout": 360,
        "recreate_venv": False,
        "dev": False,
        "duration": 30.0,
        "audio_backend": "auto",
        "ontology_terms_file": str(DEFAULT_ONTOLOGY_TERMS_FILE),
        "pipeline_summary_file": str(
            PROJECT_ROOT
            / "output"
            / "00_pipeline_summary"
            / "pipeline_execution_summary.json"
        ),
    }


def get_sample_pipeline_arguments() -> Dict[str, Any]:
    """Get sample pipeline arguments for testing."""
    return {
        "target_dir": "input/gnn_files",
        "output_dir": "output",
        "recursive": True,
        "verbose": False,
        "enable_round_trip": True,
        "enable_cross_format": True,
        "skip_steps": [],
        "only_steps": [],
        "strict": False,
        "estimate_resources": False,
        "ontology_terms_file": str(DEFAULT_ONTOLOGY_TERMS_FILE),
        "pipeline_summary_file": "output/00_pipeline_summary/pipeline_execution_summary.json",
        "llm_tasks": "all",
        "llm_timeout": 360,
        "recreate_venv": False,
        "dev": False,
        "duration": 30.0,
        "audio_backend": "auto",
        "test_data_dir": TEST_CONFIG["test_data_dir"],
    }


def get_step_metadata_dict() -> Dict[str, Any]:
    """Get metadata dictionary for pipeline steps."""
    return {
        "setup": {
            "description": "Environment setup and dependency management",
            "timeout": 300,
        },
        "tests": {"description": "Comprehensive test suite execution", "timeout": 600},
        "gnn": {"description": "GNN file discovery and processing", "timeout": 300},
        "type_checker": {"description": "Type checking and validation", "timeout": 300},
        "export": {"description": "Multi-format export", "timeout": 300},
        "visualization": {
            "description": "Graph and matrix visualization",
            "timeout": 300,
        },
        "mcp": {"description": "Model Context Protocol integration", "timeout": 300},
        "ontology": {"description": "Ontology processing", "timeout": 300},
        "render": {"description": "Code generation", "timeout": 300},
        "execute": {"description": "Simulation execution", "timeout": 300},
        "llm": {"description": "LLM analysis", "timeout": 300},
        "audio": {"description": "Audio generation", "timeout": 300},
        "website": {"description": "Website generation", "timeout": 300},
        "report": {"description": "Report generation", "timeout": 300},
    }


def is_safe_mode() -> bool:
    """Check if tests are running in safe mode."""
    return cast("bool", TEST_CONFIG.get("safe_mode", True))


def create_missing_test_files() -> None:
    """Create missing test files and directories."""
    # Create test directories
    test_dirs: list[Any] = [
        PROJECT_ROOT / "input" / "gnn_files",
        PROJECT_ROOT / "output" / "2_tests_output" / "artifacts",
        PROJECT_ROOT / "output" / "test_reports",
        PROJECT_ROOT / "output" / "test_coverage",
    ]

    for test_dir in test_dirs:
        test_dir.mkdir(parents=True, exist_ok=True)

    # Create sample GNN files if they don't exist
    gnn_dir = PROJECT_ROOT / "input" / "gnn_files"
    if not any(gnn_dir.glob("*.md")):
        create_test_gnn_files(gnn_dir)

    # Create test configuration files
    config_files: list[Any] = [
        (PROJECT_ROOT / "input" / "config.yaml", create_sample_config),
    ]

    for config_path, creator_func in config_files:
        if not config_path.exists():
            creator_func(config_path)


def create_sample_config(config_path: Path) -> None:
    """Create a sample configuration file."""
    config_content = """
pipeline:
  target_dir: "input/gnn_files"
  output_dir: "output"
  verbose: true
  recursive: true

type_checker:
  strict: false
  estimate_resources: true

ontology:
  terms_file: "src/gnn/ontology/act_inf_ontology_terms.json"

llm:
  tasks: "all"
  timeout: 360


setup:
  recreate_venv: false
  dev: false

sapf:
  duration: 30.0
"""
    with open(config_path, "w") as f:
        f.write(config_content)


def create_sample_ontology(ontology_path: Path) -> None:
    """Create a sample ontology terms file."""
    ontology_content: dict[str, Any] = {
        "terms": {
            "state_space": "The set of all possible states of a system",
            "observation_space": "The set of all possible observations",
            "action_space": "The set of all possible actions",
            "generative_model": "A model that describes how observations are generated",
            "recognition_model": "A model that describes how states are inferred",
            "free_energy": "A measure of surprise or prediction error",
            "active_inference": "A framework for understanding behavior and perception",
        }
    }
    with open(ontology_path, "w") as f:
        json.dump(ontology_content, f, indent=2)


def create_test_gnn_files(target_dir: Path) -> List[Path]:
    """Create test GNN files in the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    test_files: list[Any] = []
    gnn_content = create_sample_gnn_content()

    for name, content in gnn_content.items():
        file_path = target_dir / f"{name}.md"
        with open(file_path, "w") as f:
            f.write(content)
        test_files.append(file_path)

    return test_files


def create_test_files(target_dir: Path, num_files: int = 3) -> List[Path]:
    """Create generic test files in the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    test_files: list[Any] = []
    for i in range(num_files):
        file_path = target_dir / f"test_file_{i + 1}.txt"
        content = f"This is test file {i + 1} created for testing purposes.\n"
        content += f"Created at: {datetime.now().isoformat()}\n"
        content += f"File number: {i + 1}\n"

        with open(file_path, "w") as f:
            f.write(content)
        test_files.append(file_path)

    return test_files


def create_sample_gnn_content() -> Dict[str, str]:
    """Create sample GNN content for testing."""
    return {
        "valid_basic": """## ModelName
TestModel

## StateSpaceBlock
s[3,1,type=int]

## Connections
s -> o
""",
        "simple_model": """## GNNVersionAndFlags
Version: 1.0.0

## ModelName
Simple Active Inference Model

## Description
A basic active inference model for testing

## State Space
- **States**: [s1, s2, s3]
- **Observations**: [o1, o2, o3]
- **Actions**: [a1, a2]

## Connections
- s1 -> o1
- s2 -> o2  
- s3 -> o3
- a1 -> s1
- a2 -> s2

## Initial Parameterization
- **Prior**: Uniform
- **Likelihood**: Gaussian
- **Transition**: Deterministic

## Equations
- **Free Energy**: F = -ln p(o|m) + KL[q(s)||p(s|m)]
- **Belief Update**: q(s) = p(s|o,m)
- **Action Selection**: a* = argmin F

## Time Settings
- **Duration**: 100 steps
- **Step Size**: 0.1
- **Integration**: Euler

## Active Inference Ontology
- **Model Type**: Active Inference
- **Framework**: PyMDP
- **Inference**: Variational
- **Control**: Active Inference
""",
        "complex_model": """## GNNVersionAndFlags
Version: 2.0.0

## ModelName
Complex Active Inference Model

## Description
A complex active inference model with multiple modalities

## State Space
- **Visual States**: [v1, v2, v3, v4, v5]
- **Auditory States**: [a1, a2, a3]
- **Proprioceptive States**: [p1, p2]
- **Observations**: [obs_v1, obs_v2, obs_a1, obs_a2, obs_p1]
- **Actions**: [move_forward, move_backward, turn_left, turn_right]

## Connections
- v1 -> obs_v1
- v2 -> obs_v2
- a1 -> obs_a1
- a2 -> obs_a2
- p1 -> obs_p1
- move_forward -> v1
- move_backward -> v2
- turn_left -> a1
- turn_right -> a2

## Initial Parameterization
- **Prior**: Dirichlet
- **Likelihood**: Categorical
- **Transition**: Stochastic

## Equations
- **Free Energy**: F = -ln p(o|m) + KL[q(s)||p(s|m)]
- **Belief Update**: q(s) = p(s|o,m)
- **Action Selection**: a* = argmin F
- **Precision**: γ = 1/σ²

## Time Settings
- **Duration**: 500 steps
- **Step Size**: 0.05
- **Integration**: Runge-Kutta

## Active Inference Ontology
- **Model Type**: Multi-Modal Active Inference
- **Framework**: RxInfer
- **Inference**: Message Passing
- **Control**: Hierarchical Active Inference
""",
        "minimal_model": """## GNNVersionAndFlags
Version: 0.1.0

## ModelName
Minimal Model

## Description
Minimal active inference model

## State Space
- **States**: [s1]
- **Observations**: [o1]
- **Actions**: [a1]

## Connections
- s1 -> o1
- a1 -> s1

## Initial Parameterization
- **Prior**: Uniform
- **Likelihood**: Deterministic
- **Transition**: Deterministic

## Equations
- **Free Energy**: F = -ln p(o|m)
- **Belief Update**: q(s) = p(s|o,m)

## Time Settings
- **Duration**: 10 steps
- **Step Size**: 1.0

## Active Inference Ontology
- **Model Type**: Active Inference
- **Framework**: PyMDP
- **Inference**: Direct
- **Control**: Simple
""",
    }


def get_test_filesystem_structure() -> Dict[str, Dict[str, List[str]]]:
    """Get a test filesystem structure for testing."""
    return {
        "input": {
            "gnn_files": ["model1.md", "model2.md", "model3.md"],
            "config": ["config.yaml"],
        },
        "output": {
            "test_artifacts": ["test1.json", "test2.json"],
            "test_reports": ["report1.md", "report2.md"],
            "test_coverage": ["coverage1.json", "coverage2.json"],
        },
        "src": {
            "utils": [
                "__init__.py",
                "logging_utils.py",
                "argument_utils.py",
                "arg_definitions.py",
                "pipeline_arguments.py",
                "step_config.py",
                "arg_parsing.py",
                "path_conversion.py",
                "safe_eval.py",
            ],
            "tests": ["__init__.py", "conftest.py", "runner.py"],
            "gnn": ["__init__.py", "processor.py", "validator.py"],
        },
    }
