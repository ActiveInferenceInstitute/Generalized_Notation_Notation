"""
Dependency setup functions for the GNN project.

This module handles JAX installation and testing, Julia environment setup,
optional package group installation, and project structure creation.
"""

import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, List, cast

from .constants import (
    OPTIONAL_GROUPS,
    PROJECT_ROOT,
    SETUP_DEFAULT_PIPELINE_EXTRAS,
    VENV_PYTHON,
)
from .uv_management import (
    check_system_requirements,
    create_uv_environment,
    install_uv_dependencies,
    run_command,
    save_setup_results,
    validate_uv_setup,
)

logger = logging.getLogger(__name__)

try:
    from gnn.utils.runtime_safety.jax_stack_validation import (
        run_jax_stack_probe_subprocess,
    )
except ImportError:
    try:
        from gnn.utils.runtime_safety.jax_stack_validation import (
            run_jax_stack_probe_subprocess,
        )
    except ImportError:
        run_jax_stack_probe_subprocess = cast(Any, None)


def install_jax_and_test(verbose: bool = False) -> bool:
    """
    Ensure JAX, Optax, Flax, and pymdp 1.x work in the project venv.

    Runs :mod:`gnn.utils.runtime_safety.jax_stack_validation` (JIT, vmap, XLA sync, Optax, Flax, pymdp Agent API).
    On failure, attempts ``uv sync`` with configured extras once, then re-probes.
    """

    if not VENV_PYTHON.exists():
        logger.error("Venv Python not found, cannot test JAX stack")
        return False

    probe_fn: Any = run_jax_stack_probe_subprocess
    if probe_fn is None:
        logger.error("jax_stack_validation module not importable; cannot run JAX probe")
        return False

    ok, out = probe_fn(VENV_PYTHON, PROJECT_ROOT)
    if ok:
        logger.info("✅ JAX + Optax + Flax + pymdp stack validated in venv")
        if verbose and out:
            for line in out.splitlines()[:50]:
                logger.info("   %s", line)
        return True

    logger.warning(
        "JAX stack validation failed: %s", out[:2000] if out else "(no output)"
    )

    try:
        logger.info("Attempting repair: uv sync (project extras)...")
        install_cmd: list[Any] = ["uv", "sync", "--verbose"]
        for extra in SETUP_DEFAULT_PIPELINE_EXTRAS:
            install_cmd.extend(["--extra", extra])

        if verbose:
            logger.info("Running: %s", " ".join(install_cmd))

        result = subprocess.run(  # nosec B603
            install_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(
                "uv sync failed: %s", result.stderr[:2000] if result.stderr else ""
            )
            return False

        ok2, out2 = run_jax_stack_probe_subprocess(VENV_PYTHON, PROJECT_ROOT)
        if ok2:
            logger.info("✅ JAX stack validated after uv sync")
            return True
        logger.warning(
            "JAX stack still failing after sync: %s", out2[:2000] if out2 else ""
        )
        return False
    except Exception as e:
        logger.error("JAX stack repair failed: %s", e)
        return False


def setup_julia_environment(verbose: bool = False) -> bool:
    """
    Set up Julia environment for GNN execution frameworks.

    Args:
        verbose: Enable verbose logging

    Returns:
        True if setup succeeded, False otherwise
    """
    logger.info("🔧 Setting up Julia environment for GNN execution")

    from gnn.utils.runtime_safety.julia_probe import julia_executable

    try:
        julia_path = julia_executable()
        if not julia_path:
            logger.error("❌ Julia not found in PATH")
            logger.info("💡 Install Julia from: https://julialang.org/downloads/")
            return False  # nosec B603

        version_result = subprocess.run(  # nosec B603
            [julia_path, "--version"], capture_output=True, text=True, timeout=10
        )

        if version_result.returncode == 0:
            version_line = version_result.stdout.strip().split("\n")[0]
            logger.info(f"✅ Julia found: {version_line}")
        else:
            logger.warning(
                f"⚠️ Could not determine Julia version: {version_result.stderr}"
            )
            logger.info("✅ Julia is available, proceeding with setup")

        setup_scripts: list[Any] = [
            PROJECT_ROOT
            / "src"
            / "gnn"
            / "execute"
            / "rxinfer"
            / "setup_environment.jl",
            PROJECT_ROOT
            / "src"
            / "gnn"
            / "execute"
            / "activeinference_jl"
            / "setup_environment.jl",
        ]

        success_count = 0

        for script_path in setup_scripts:
            if script_path.exists():
                logger.info(f"Running Julia setup script: {script_path}")
                try:
                    result = subprocess.run(  # nosec B603
                        [julia_path, str(script_path)],
                        cwd=script_path.parent,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode == 0:
                        logger.info(f"✅ Julia setup completed for {script_path.name}")
                        success_count += 1
                    else:
                        logger.error(
                            f"❌ Julia setup failed for {script_path.name}: {result.stderr}"
                        )
                except subprocess.TimeoutExpired:
                    logger.error(f"⏰ Julia setup timed out for {script_path.name}")
                except Exception as e:
                    logger.error(
                        f"❌ Error running Julia setup for {script_path.name}: {e}"
                    )
            else:
                logger.warning(f"⚠️ Julia setup script not found: {script_path}")

        if success_count > 0:
            logger.info(
                f"✅ Julia environment setup completed ({success_count} frameworks configured)"
            )
            return True
        else:
            logger.warning("⚠️ No Julia frameworks were successfully configured")
            return False

    except Exception as e:
        logger.error(f"❌ Error setting up Julia environment: {e}")
        return False


def install_optional_package_group(group_name: str, verbose: bool = False) -> bool:
    """
    Install a specific optional package group using UV sync with extras.

    Args:
        group_name: Name of the package group (see OPTIONAL_GROUPS constant)
        verbose: Enable verbose logging

    Returns:
        True if installation succeeded, False otherwise
    """
    if group_name.lower() == "julia":
        return setup_julia_environment(verbose=verbose)

    group_aliases: dict[str, Any] = {
        "ml": "ml-ai",
        "jax": "",
        "pymdp": "",
        "numpyro": "",
        "plotly": "",
        "llm": "",
    }
    normalized_name = group_aliases.get(group_name.lower(), group_name.lower())

    if normalized_name == "":
        logger.info(
            "ℹ️ '%s' is installed by the core dependency set; run `uv sync`.",
            group_name,
        )
        return True

    if normalized_name not in OPTIONAL_GROUPS:
        logger.error(f"❌ Unknown package group: {group_name}")
        logger.info(f"ℹ️ Available groups: {', '.join(OPTIONAL_GROUPS.keys())}")
        return False

    group_description = OPTIONAL_GROUPS[normalized_name]
    logger.info(f"📦 Installing '{normalized_name}' package group: {group_description}")

    try:
        sync_cmd: list[Any] = ["uv", "sync", "--extra", normalized_name]

        if verbose:
            sync_cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(sync_cmd)}")

        result = subprocess.run(  # nosec B603
            sync_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )

        if result.returncode == 0:
            logger.info(f"✅ Successfully installed '{normalized_name}' package group")
            if verbose and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        logger.debug(f"  {line}")
            return True
        else:
            logger.error(f"❌ Failed to install '{normalized_name}' package group")
            logger.error(f"STDERR: {result.stderr}")
            if verbose:
                logger.error(f"STDOUT: {result.stdout}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Installation timed out for '{normalized_name}' package group")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing '{normalized_name}' package group: {e}")
        return False


def install_all_optional_packages(verbose: bool = False) -> dict:
    """
    Install all optional package groups using UV sync with 'all' extra.

    Args:
        verbose: Enable verbose logging

    Returns:
        Dictionary with installation status
    """
    logger.info(
        "🚀 Installing ALL optional package groups via 'uv sync --extra all'..."
    )

    try:
        sync_cmd: list[Any] = ["uv", "sync", "--extra", "all"]

        if verbose:
            sync_cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(sync_cmd)}")

        result = subprocess.run(  # nosec B603
            sync_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )

        if result.returncode == 0:
            logger.info("✅ All optional packages installed successfully")
            if verbose and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        logger.debug(f"  {line}")
            return {"all": True, "success": True}
        else:
            logger.error("❌ Failed to install all optional packages")
            logger.error(f"STDERR: {result.stderr}")
            return {"all": False, "success": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("⏰ Installation timed out for all optional packages")
        return {"all": False, "success": False, "error": "timeout"}
    except Exception as e:
        logger.error(f"❌ Error installing all optional packages: {e}")
        return {"all": False, "success": False, "error": str(e)}


def install_optional_dependencies(
    project_root: Path,
    logger: logging.Logger,
    package_groups: (List[str]) | None = None,
) -> bool:
    """Install optional dependencies for the project using UV sync with extras."""
    try:
        logger.info("Installing optional dependencies using UV sync")

        if not package_groups:
            package_groups = ["dev", "test", "docs"]

        for group in package_groups:
            try:
                logger.info(f"Installing {group} dependencies via UV sync")
                result = run_command(
                    ["uv", "sync", "--extra", group],
                    cwd=project_root,
                    check=False,
                    verbose=True,
                )
                if result.returncode == 0:
                    logger.info(f"✅ {group} dependencies installed successfully")
                else:
                    logger.warning(
                        f"⚠️ {group} dependencies installation failed (exit {result.returncode})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to install {group} dependencies: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to install optional dependencies: {e}")
        return False


def create_project_structure(output_dir: Path, logger: logging.Logger) -> bool:
    """Create the standard project structure."""
    try:
        logger.info("Creating project structure")

        directories: list[Any] = [
            "input/gnn_files",
            "output",
            "output/logs",
            "output/temp",
            "doc",
            "tests",
        ]

        for directory in directories:
            dir_path = output_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

        config_files: dict[str, Any] = {
            "input/config.yaml": "# GNN Pipeline Configuration\n",
            "output/.gitkeep": "",
            "tests/__init__.py": "# Tests package\n",
        }

        for file_path, content in config_files.items():
            full_path = output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            logger.debug(f"Created file: {full_path}")

        logger.info("Project structure created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create project structure: {e}")
        return False


def setup_gnn_project(project_path: str, verbose: bool = False) -> bool:
    """
    Set up a new GNN project at the specified path using UV.

    Args:
        project_path: Path where the project should be set up
        verbose: Enable verbose logging

    Returns:
        True if setup successful, False otherwise
    """
    try:
        project_dir = Path(project_path)
        project_dir.mkdir(parents=True, exist_ok=True)

        (project_dir / "input" / "gnn_files").mkdir(parents=True, exist_ok=True)
        (project_dir / "output").mkdir(parents=True, exist_ok=True)
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        # nosec B607 B603
        try:
            subprocess.run(["uv", "init"], cwd=project_dir, check=True, timeout=30)  # nosec B607 B603
            logger.info(f"UV project initialized at {project_dir}")
        except Exception as e:
            logger.warning(f"Could not initialize UV project: {e}")

        logger.info(f"GNN project structure created at {project_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to set up GNN project with UV: {e}")
        return False


def setup_complete_environment(
    verbose: bool = False,
    recreate: bool = False,
    install_optional: bool = False,
    optional_groups: (list) | None = None,
    output_dir: (Path) | None = None,
) -> bool:
    """
    Complete environment setup with optional dependencies.

    Args:
        verbose: Enable verbose logging
        recreate: Recreate environment if it exists
        install_optional: Install optional dependencies
        optional_groups: Specific groups to install (None = all groups)
        output_dir: Output directory for setup logs

    Returns:
        True if setup successful, False otherwise
    """
    try:
        logger.info("🚀 Starting COMPLETE GNN environment setup from cold start...")

        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            return False

        if not create_uv_environment(verbose, recreate):
            logger.error("❌ UV environment creation failed")
            return False

        logger.info("📦 Installing core dependencies...")
        if not install_uv_dependencies(verbose=verbose):
            logger.warning(
                "⚠️ Core dependency installation had issues, but continuing..."
            )

        if install_optional:
            logger.info("\n🎁 Installing optional dependency groups...")

            if optional_groups:
                for group in optional_groups:
                    install_optional_package_group(group, verbose=verbose)
            else:
                install_all_optional_packages(verbose=verbose)

        logger.info("\n✅ Validating complete environment...")
        validation_results = validate_uv_setup(PROJECT_ROOT, logger)

        if output_dir:
            save_setup_results(
                output_dir, validation_results, optional_groups or [], True
            )

        logger.info("\n🎉 COMPLETE environment setup finished!")
        return True

    except Exception as e:
        logger.error(f"❌ Complete environment setup failed: {e}")
        if verbose:
            import traceback

            logger.error(traceback.format_exc())
        return False
