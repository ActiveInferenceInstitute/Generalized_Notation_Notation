"""
Execution API for registry-declared GNN simulation backends.

The canonical framework registry supplies the declared implementation features
exposed here. These features describe backend support; use
``collect_doctor_report`` for dependency, toolchain, and execution readiness.
The experimental cpomdp and THRML backends require explicit framework selection.
"""

from typing import Any

from gnn import __version__
from gnn.frameworks import ALL_FRAMEWORKS as _ALL_FRAMEWORKS

FEATURES: dict[str, Any] = {
    **{f"{name}_execution": True for name in _ALL_FRAMEWORKS},
    "validation": True,
    "mcp_integration": True,
}

from pathlib import Path
from typing import Any, Dict, List

# Constrained type for supported execution framework names — the canonical
# Literal lives in ``execute.types``; re-exported here for the public name.
# All execute submodules are in-tree — their import must succeed or tests
# catch it. Any ImportError here is a real bug, not a "missing optional dep"
# situation, and should fail loudly.
from .doctor import collect_doctor_report
from .executor import (
    GNNExecutor,
    clear_execution_cache,
    execute_gnn_model,
    execute_script_safely,
    list_frameworks,
    run_simulation,
)
from .planning import plan_execute
from .processor import execute_simulation_from_gnn, process_execute
from .pymdp import (
    PyMDPSimulation,
    execute_pymdp_simulation,
    execute_pymdp_simulation_from_gnn,
    get_pymdp_health_status,
    validate_pymdp_environment,
)

# Constrained type for supported execution framework names — the canonical
# Literal lives in ``execute.types``; re-exported here for the public name.
from .types import ExecutionFrameworkName as FrameworkName
from .validator import (
    check_dependencies,
    check_file_permissions,
    check_network_connectivity,
    check_python_environment,
    check_system_resources,
    log_validation_results,
    validate_execution_environment,
)

__all__: list[Any] = [
    "__version__",
    "FEATURES",
    "FrameworkName",
    # Core classes
    "GNNExecutor",
    "PyMDPSimulation",
    # Execution functions
    "process_execute",
    "execute_simulation_from_gnn",
    "execute_gnn_model",
    "run_simulation",
    "execute_script_safely",
    "clear_execution_cache",
    # Introspection / planning
    "list_frameworks",
    "plan_execute",
    "collect_doctor_report",
    "execute_pymdp_simulation_from_gnn",
    "execute_pymdp_simulation",
    "validate_pymdp_environment",
    "get_pymdp_health_status",
    # Validation
    "validate_execution_environment",
    "log_validation_results",
    "check_python_environment",
    "check_system_resources",
    "check_dependencies",
    "check_file_permissions",
    "check_network_connectivity",
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "execute",
        "version": __version__,
        "description": "Multi-framework execution of rendered simulation scripts",
        "features": FEATURES,
    }
