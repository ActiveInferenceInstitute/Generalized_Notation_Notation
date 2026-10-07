"""
Shared constants for the GNN setup module.

These constants are imported by uv_management.py and uv_package_ops.py to avoid
duplicate definitions and the circular import that would result if either file
imported from the other.
"""

import sys
from pathlib import Path
from typing import Any

# --- Directory and file names ---
VENV_DIR = ".venv"
PYPROJECT_FILE = "pyproject.toml"
LOCK_FILE = "uv.lock"

# --- Resolved paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VENV_PATH = PROJECT_ROOT / VENV_DIR
PYPROJECT_PATH = PROJECT_ROOT / PYPROJECT_FILE
LOCK_PATH = PROJECT_ROOT / LOCK_FILE

if sys.platform == "win32":
    VENV_PYTHON = VENV_PATH / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_PATH / "bin" / "python"

MIN_PYTHON_VERSION = (3, 11)

# Extra ``uv sync --extra …`` groups for step 1 when non-empty. Core Step 12
# backends, LLM clients, and interactive visualization are default dependencies;
# PyTorch, bnlearn, Stan, cpomdp, THRML, and ngc-learn remain explicit optional backends.
SETUP_DEFAULT_PIPELINE_EXTRAS: tuple[str, ...] = ()

OPTIONAL_GROUPS: dict[str, Any] = {
    "dev": "Development tools (pytest, ruff, mypy, sphinx, jupyterlab, etc.)",
    "api": "FastAPI/uvicorn server dependencies",
    "ml-ai": "Machine learning extensions (transformers, scipy, scikit-learn)",
    "audio": "Audio processing (librosa, soundfile, pedalboard, pydub)",
    "gui": "GUI frameworks (gradio)",
    "graphs": "Graphviz bindings for graph rendering workflows",
    "research": "Research tools (jupyterlab, sympy, numba, cython)",
    "scaling": "Scaling (dask, distributed, ray)",
    "torch": "PyTorch tensor simulation backend",
    "stan": "Stan Python driver (CmdStan toolchain installed separately)",
    "bnlearn": "Bayesian network learning backend",
    "cpomdp": "Experimental continuous POMDP inference backend",
    "thrml": "Experimental THRML categorical Gibbs sampling backend",
    "ngclearn": "ngc-learn runtime (Python >= 3.12)",
    "all": "All functionally distinct optional dependency groups combined",
}
