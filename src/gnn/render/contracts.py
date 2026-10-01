#!/usr/bin/env python3
"""GNN Output Contracts — Framework-specific validation for rendered code.

Each contract pins the MAINTAINED output shape of one framework backend, as
emitted by the current delegated-executor renderers and validated against the
committed corpus artifacts:

- pymdp: delegated runner importing ``gnn.execute.pymdp`` with ``A_data`` /
  ``B_data`` matrix literals.
- rxinfer: genuine ``infer()`` programs using an inline ``@model`` or the
  shared ``GnnRxInferModels`` module (multi-agent per-agent runners).
- jax / pytorch: params-payload or Joseph-form Kalman programs
  (``'A_matrix': jnp.array(...)`` / ``A = torch.tensor(...)`` / ``K @ H``).
- numpyro: ``numpyro.distributions`` sampling sites (discrete
  ``dist.Categorical(...).sample`` or continuous ``numpyro.sample``).
- cpomdp: continuous-only programs building a ``LinearGaussianModel`` and
  an ``EnumeratedEfeSearch`` over a ``FiniteActionSet``.
- activeinference_jl: self-contained runner programs with the agent logic in
  emitted functions (no ``Agent(`` constructor call in the artifact).
- bnlearn: generator-backed ``bn.make_DAG`` + ``bn.parameter_learning.fit``
  programs (executed by ``src/gnn/execute/bnlearn/``).

Validation surface: ``scripts/bench_render_backends.py`` (corpus x framework
conformance benchmark) and ``tests/render/test_render_contracts.py`` (shape
pins against real ``render_gnn_spec`` output). Nothing gates pipeline receipts
on these contracts; re-pin a contract here whenever a renderer's output shape
changes deliberately.

"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContractViolation:
    """A violation of a framework output contract."""

    framework: str
    field: str
    expected: str
    actual: str
    file_path: Optional[str] = None
    line: Optional[int] = None

    def __str__(self) -> str:
        """Return the string representation."""
        loc = f" [{self.file_path}:{self.line}]" if self.file_path else ""
        return f"[{self.framework}] {self.field}: expected {self.expected}, got {self.actual}{loc}"


# ── Framework Contracts ──────────────────────────────────────────────────────────

CONTRACTS: Dict[str, Dict[str, Any]] = {
    "pymdp": {
        "required_imports": ["gnn.execute.pymdp"],
        "required_variables": [],
        "matrix_patterns": [
            r"\bA_data\s*=\s*",
            r"\bB_data\s*=\s*",
            r"execute_pymdp_simulation\s*\(",
        ],
        "optional_variables": [],
    },
    "rxinfer": {
        "required_imports": ["RxInfer"],
        "required_variables": [],
        "matrix_patterns": [
            r"using\s+RxInfer",
            r"(@model|GnnRxInferModels)",
            r"infer\s*\(",
        ],
        "optional_variables": [],
    },
    "jax": {
        "required_imports": ["jax", "jax.numpy"],
        "required_variables": [],
        "matrix_patterns": [
            r"jnp\.\w+",
            r"(A_matrix|B_matrix|K @ H)",
        ],
        "optional_variables": [],
    },
    "pytorch": {
        "required_imports": ["torch"],
        "required_variables": [],
        "matrix_patterns": [
            r"torch\.(tensor|zeros|ones|eye)",
            r"(torch\.tensor|K @ H|torch\.linalg)",
        ],
        "optional_variables": [],
    },
    "numpyro": {
        "required_imports": ["numpyro"],
        "required_variables": [],
        "matrix_patterns": [
            r"dist\.\w+",
            r"\.sample\s*\(",
        ],
        "optional_variables": [],
    },
    "cpomdp": {
        "required_imports": ["cpomdp"],
        "required_variables": [],
        "matrix_patterns": [
            r"LinearGaussianModel\s*\(",
            r"(EnumeratedEfeSearch|KalmanBackend)\s*\(",
        ],
        "optional_variables": [],
    },
    "stan": {
        "required_imports": [],
        "required_variables": [],
        "matrix_patterns": [
            r"data\s*\{",
            r"model\s*\{",
        ],
        "optional_variables": [],
    },
    "activeinference_jl": {
        "required_imports": ["ActiveInference"],
        "required_variables": [],
        "matrix_patterns": [
            r"using\s+ActiveInference",
            r"function\s+\w+\(",
        ],
        "optional_variables": [],
    },
    "discopy": {
        "required_imports": ["discopy"],
        "required_variables": [],
        "matrix_patterns": [
            r"(Ty|Box)\(",
        ],
        "optional_variables": [],
    },
    "bnlearn": {
        "required_imports": ["bnlearn"],
        "required_variables": [],
        "matrix_patterns": [
            r"bn\.make_DAG",
            r"bn\.parameter_learning\.fit",
        ],
        "optional_variables": [],
    },
}


def validate_rendered_output(
    code: str,
    framework: str,
    *,
    file_path: Optional[str] = None,
) -> List[ContractViolation]:
    """
    Validate rendered code against framework-specific contract.

    Args:
        code: Generated source code string.
        framework: Target framework name (pymdp, rxinfer, jax).
        file_path: Optional source file for error reporting.

    Returns:
        List of ContractViolation for any issues found.
    """
    contract = CONTRACTS.get(framework.lower())
    if not contract:
        raise ValueError(
            f"No contract defined for framework '{framework}'. "
            f"Known frameworks: {', '.join(sorted(CONTRACTS))}"
        )

    violations: list[Any] = []

    # Check required imports
    for imp in contract.get("required_imports", []):
        if imp not in code:
            violations.append(
                ContractViolation(
                    framework=framework,
                    field="import",
                    expected=f"import containing '{imp}'",
                    actual="not found",
                    file_path=file_path,
                )
            )

    # Check required variable assignments
    for var in contract.get("required_variables", []):
        pattern = rf"\b{re.escape(var)}\s*="
        if not re.search(pattern, code):
            violations.append(
                ContractViolation(
                    framework=framework,
                    field=f"variable_{var}",
                    expected=f"assignment to '{var}'",
                    actual="not found",
                    file_path=file_path,
                )
            )

    # Check matrix patterns
    for pattern in contract.get("matrix_patterns", []):
        if not re.search(pattern, code):
            violations.append(
                ContractViolation(
                    framework=framework,
                    field="pattern",
                    expected=f"pattern matching '{pattern}'",
                    actual="not found",
                    file_path=file_path,
                )
            )

    if violations:
        logger.warning(f"⚠️ {len(violations)} contract violation(s) for {framework}")
    else:
        logger.debug(f"✅ {framework} contract satisfied")

    return violations
