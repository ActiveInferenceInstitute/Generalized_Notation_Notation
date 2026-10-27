"""Test-harness constants: path anchors and category/stage/coverage/configuration tables.

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1). The
``sys.path`` bootstrap is preserved exactly: it is load-bearing behavior that
test code outside the package relies on."""

import sys
from pathlib import Path
from typing import Any

from gnn.utils.arguments.pipeline_arguments import DEFAULT_ONTOLOGY_TERMS_FILE

# Ensure src is in Python path for imports
# parents[3] (not [2]): the leaf sits one level deeper than the old
# testing_utils.py home, so the resolved src/ directory is identical.
SRC_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = SRC_DIR.parent
TEST_DIR = PROJECT_ROOT / "tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Test categories and markers
TEST_CATEGORIES: dict[str, Any] = {
    "fast": "Quick validation tests for core functionality",
    "standard": "Integration tests and moderate complexity",
    "slow": "Complex scenarios and benchmarks",
    "performance": "Resource usage and scalability tests",
    "unit": "Individual component tests",
    "integration": "Multi-component workflow tests",
    "mcp": "Model Context Protocol integration tests",
}

# Test execution stages
TEST_STAGES: dict[str, Any] = {
    "fast": {"timeout": 180, "max_failures": 10, "parallel": True, "coverage": False},
    "standard": {
        "timeout": 600,
        "max_failures": 20,
        "parallel": True,
        "coverage": True,
    },
    "slow": {"timeout": 300, "max_failures": 20, "parallel": False, "coverage": True},
    "performance": {
        "timeout": 600,
        "max_failures": 5,
        "parallel": False,
        "coverage": False,
    },
}

# Test coverage targets
COVERAGE_TARGETS: dict[str, Any] = {
    "overall": 85.0,
    "unit": 90.0,
    "integration": 80.0,
    "performance": 70.0,
}

# Test configuration constants
TEST_CONFIG: dict[str, Any] = {
    "safe_mode": True,
    "verbose": False,
    "strict": False,
    "estimate_resources": False,
    "skip_steps": [],
    "only_steps": [],
    "timeout_seconds": 300,  # 5 minutes default timeout
    "temp_output_dir": PROJECT_ROOT / "output" / "2_tests_output" / "artifacts",
    "max_test_files": 10,  # Maximum number of test files to process
    # Add missing keys that tests expect
    "sample_gnn_dir": PROJECT_ROOT / "input" / "gnn_files",
    "simulate_external_deps": True,
    "temp_dir": PROJECT_ROOT / "output" / "2_tests_output" / "artifacts",
    "recursive": True,
    "enable_round_trip": True,
    "enable_cross_format": True,
    "llm_tasks": "all",
    "llm_timeout": 360,
    "recreate_venv": False,
    "dev": False,
    "duration": 30.0,
    "audio_backend": "auto",
    "ontology_terms_file": DEFAULT_ONTOLOGY_TERMS_FILE,
    "pipeline_summary_file": PROJECT_ROOT
    / "output"
    / "00_pipeline_summary"
    / "pipeline_execution_summary.json",
    "fast_only": False,
    "include_performance": True,
    # Required by tests
    "test_data_dir": str(TEST_DIR / "test_data"),
}
