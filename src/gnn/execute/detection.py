#!/usr/bin/env python3
"""
Framework and path detection for GNN Step 12.

Owns rendered-script discovery, framework detection from paths, framework
parameter parsing, render-output directory resolution, script execution
context normalization, and hardware accelerator detection. Extracted from
``execute.processor``.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from gnn.frameworks import ALL_FRAMEWORKS, LITE_FRAMEWORKS

from .types import ScriptExecutionContext

logger = logging.getLogger(__name__)

# Canonical name of the Step 11 render-output directory. Referenced by the
# resolution heuristic and by callers that build sibling paths off it.
RENDER_OUTPUT_DIR_NAME: str = "11_render_output"


def determine_script_framework(
    script_path: Path, render_output_dir: Path, framework_dirs: Dict[str, str]
) -> str:
    """
    Determine the framework for a script based on its directory path.

    Args:
        script_path: Path to the script
        render_output_dir: Base render output directory
        framework_dirs: Mapping of directory names to framework names

    Returns:
        Framework name or 'unknown'
    """
    try:
        # Get relative path from render output directory
        relative_path = script_path.relative_to(render_output_dir)

        # Render outputs use model/framework/script.ext. Match framework
        # directories exactly so model names like "bnlearn_causal_model" do
        # not override the actual framework directory.
        for part in relative_path.parts[:-1]:
            if part.lower() in framework_dirs:
                return framework_dirs[part.lower()]

        script_name = relative_path.name.lower()
        for framework_name in framework_dirs.values():
            if script_name.endswith(f"_{framework_name}.py") or script_name.endswith(
                f"_{framework_name}.jl"
            ):
                return framework_name

        # Default recovery
        return "unknown"

    except Exception as e:
        logging.getLogger(__name__).debug(
            f"Error determining framework for script: {e}"
        )
        return "unknown"


def parse_frameworks_parameter(frameworks: str, logger: Any) -> List[str]:
    """
    Parse the frameworks parameter into a list of framework names.

    Args:
        frameworks: Comma-separated string of framework names or preset
        logger: Logger instance

    Returns:
        List of framework names to include
    """
    if not frameworks or frameworks.lower() == "all":
        return list(ALL_FRAMEWORKS)

    if frameworks.lower() == "lite":
        return list(LITE_FRAMEWORKS)

    # Parse comma-separated list against the canonical enumeration. Inputs
    # with no known framework are rejected upstream by
    # ``validate_frameworks_arg`` (both callers pre-validate), so an empty
    # result here is an honest no-match and is returned as-is.
    framework_list = [f.strip() for f in frameworks.split(",")]
    valid_set = set(ALL_FRAMEWORKS)

    # Filter out invalid frameworks
    valid_list = [f for f in framework_list if f in valid_set]

    if len(valid_list) != len(framework_list):
        invalid = [f for f in framework_list if f not in valid_set]
        logger.warning(
            f"Invalid frameworks specified: {invalid}. "
            f"Valid options: {list(ALL_FRAMEWORKS)}"
        )

    return valid_list


def _resolve_render_output_dir(
    target_dir: Path,
    kwargs: dict,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the render output directory from kwargs and filesystem heuristics.

    Resolution priority:
    1. Explicit ``--render-output-dir`` kwarg.
    2. Sibling of the current step's output dir: when ``output_dir`` is
       ``<base>/12_execute_output``, use ``<base>/11_render_output`` (and nested layout).
    3. target_dir itself if it looks like a render output directory.
    4. Common pipeline and test output locations (searched in order).

    Returns the first existing, non-empty directory found, or None.
    """

    def _if_nonempty(p: Path) -> Optional[Path]:
        """Handle if nonempty for internal callers."""
        if p.exists() and any(p.rglob("*")):
            return p
        return None

    # Priority 1: explicit kwarg
    if kwargs.get("render_output_dir"):
        p = Path(kwargs["render_output_dir"])
        return _if_nonempty(p) or p

    # Priority 2: same pipeline base as step 12 (target often remains GNN input dir)
    if output_dir is not None:
        base = output_dir.parent
        for rel in (
            f"{RENDER_OUTPUT_DIR_NAME}/{RENDER_OUTPUT_DIR_NAME}",
            RENDER_OUTPUT_DIR_NAME,
        ):
            found = _if_nonempty(base / rel)
            if found is not None:
                return found

    # Priority 3: target_dir is already the render output
    if (
        RENDER_OUTPUT_DIR_NAME in str(target_dir)
        or target_dir.name == RENDER_OUTPUT_DIR_NAME
    ):
        return _if_nonempty(target_dir) or target_dir

    # Priority 4: search common cwd-relative locations.
    nested = f"{RENDER_OUTPUT_DIR_NAME}/{RENDER_OUTPUT_DIR_NAME}"
    candidates: List[Path] = [
        target_dir.parent / "output" / RENDER_OUTPUT_DIR_NAME,
        target_dir / RENDER_OUTPUT_DIR_NAME,
        Path(f"output/test_render/{nested}"),
        Path(f"output/test_render_improved/{nested}"),
        *list(Path("output").glob(f"*/{nested}")),
        *list(Path("output").glob(f"**/{RENDER_OUTPUT_DIR_NAME}")),
    ]
    for candidate in candidates:
        found = _if_nonempty(candidate)
        if found is not None:
            return found
    return None


def find_executable_scripts(
    render_output_dir: Path,
    verbose: bool,
    logger: Any,
    requested_frameworks: List[str],
    allowed_scripts: Optional[set[Path]] = None,
) -> List[Dict[str, Any]]:
    """Find executable scripts in the render output directory.

    **Discovery strategy (V-10)**:

    1. Manifest-first: when ``allowed_scripts`` is provided (from a
       ``render_processing_summary.json`` manifest), only those scripts the
       render step actually produced are considered. No blanket file-tree
       walk — stale or un-rendered scripts are ignored.

    2. rglob fallback: when the manifest is missing or corrupt,
       ``allowed_scripts`` is ``None`` and the function performs the
       traditional recursive file walk. A warning is emitted since this may
       pick up stale intermediate files.

    Scripts are filtered by the requested frameworks and excluded if they
    match common non-executable patterns (test files, __init__.py, etc.).

    Args:
        render_output_dir: Directory containing rendered scripts from Step 11.
        verbose: Enable verbose logging of discovered scripts.
        logger: Logger instance for output messages.
        requested_frameworks: List of framework names to include (e.g.,
            ["pymdp", "jax", "discopy"]). Scripts from other frameworks
            will be skipped.
        allowed_scripts: Optional set of resolved Paths from the Step 11
            manifest. When not None, these paths replace the rglob step.

    Returns:
        List of dictionaries, each containing:
            - path: Path to the script file
            - name: Script filename
            - framework: Detected framework name
            - executor: Command to execute the script (python/julia)
            - relative_path: Path relative to render_output_dir
            - size_bytes: File size in bytes
    """
    executable_scripts: list[Any] = []

    # Define supported script types and their executors
    script_types: dict[str, Any] = {
        "*.py": {"executor": sys.executable, "framework": "python"},
        "*.jl": {"executor": "julia", "framework": "julia"},
    }

    # Map framework directories to framework names
    framework_dirs: dict[str, Any] = {
        "pymdp": "pymdp",
        "jax": "jax",
        "discopy": "discopy",
        "rxinfer": "rxinfer",
        "activeinference_jl": "activeinference_jl",
        "activeinference.jl": "activeinference_jl",
        "pytorch": "pytorch",
        "numpyro": "numpyro",
        "stan": "stan",
        "bnlearn": "bnlearn",
        "ngclearn": "ngclearn",
        "cpomdp": "cpomdp",
    }

    # Normalise the base directory for consistent framework detection and
    # relative-path computation across both discovery modes.
    base_dir = render_output_dir.resolve()

    # --- Phase 1: Discover candidate script paths ---
    if allowed_scripts is not None:
        # Manifest-based discovery (V-10): only rendered scripts qualify.
        if verbose:
            logger.info(
                f"Discovering scripts from render manifest "
                f"({len(allowed_scripts)} rendered scripts listed)"
            )
        manifest_paths: list[Path] = []
        for p in allowed_scripts:
            candidate = Path(p).resolve()
            if not candidate.exists():
                # Old rglob only ever surfaced files that exist; a manifest
                # entry whose file is missing is reported downstream as a
                # missing rendered script rather than executed.
                logger.warning(
                    f"Render manifest references missing script: {candidate}"
                )
                continue
            manifest_paths.append(candidate)
        candidates = sorted(manifest_paths)
    else:
        # rglob fallback: crawl the directory tree when no manifest is present.
        logger.warning(
            "No render manifest provided — falling back to recursive rglob "
            "discovery. This may include stale or un-rendered scripts."
        )
        candidates = []
        for pattern, config in script_types.items():
            # rglob on an absolute path yields absolute paths.
            candidates.extend(base_dir.rglob(pattern))

    # --- Phase 2: Build script-info dicts ---
    for script_path in candidates:
        # Skip support modules in test folders without excluding rendered
        # model scripts whose model name naturally starts with "test_".
        script_name = script_path.name.lower()
        path_parts = {part.lower() for part in script_path.parts}
        if (
            script_name == "__init__.py"
            or script_name.startswith("__")
            or script_path.stem.lower().endswith("_test")
            or "tests" in path_parts
        ):
            continue

        # Determine framework from directory path
        framework = determine_script_framework(script_path, base_dir, framework_dirs)

        # Filter by requested frameworks
        if framework not in requested_frameworks:
            if verbose:
                logger.debug(
                    f"Skipping {framework} script: {script_path.name} "
                    f"(not in requested frameworks)"
                )
            continue

        # Resolve executor from the file suffix
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            executor = sys.executable
        elif suffix == ".jl":
            executor = "julia"
        else:
            continue  # not a recognised script type

        # Compute relative path (best-effort — may not be under base_dir
        # when the manifest is a direct pass-through from the render step).
        try:
            rel = script_path.relative_to(base_dir)
        except ValueError:
            rel = script_path

        # Check if script is executable or can be made executable
        script_info: dict[str, Any] = {
            "path": script_path,
            "name": script_path.name,
            "framework": framework,
            "executor": executor,
            "relative_path": rel,
            "size_bytes": script_path.stat().st_size if script_path.exists() else 0,
        }

        executable_scripts.append(script_info)

        if verbose:
            logger.info(f"Found {framework} script: {rel}")

    return executable_scripts


def _build_script_execution_context(
    script_info: Dict[str, Any],
) -> ScriptExecutionContext:
    """Normalize model/framework metadata from a rendered script path."""
    script_path = script_info["path"]
    path_parts = script_path.parts
    if len(path_parts) >= 3:
        model_name = path_parts[-3]
        framework = path_parts[-2]
    else:
        model_name = "unknown_model"
        framework = script_info["framework"]

    return ScriptExecutionContext(
        script_path=script_path,
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
    )


def _detect_accelerator_type() -> str:
    """Best-effort hardware accelerator detection for execution metadata."""
    accelerator_type = "cpu"
    try:
        if shutil.which("nvidia-smi") is not None:
            accelerator_type = "cuda"
        elif sys.platform == "darwin":
            accelerator_type = "mps"
    except Exception as e:
        logger.debug("Accelerator detection failed; falling back to cpu: %s", e)
        accelerator_type = "cpu"
    return accelerator_type
