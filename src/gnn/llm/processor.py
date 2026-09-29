#!/usr/bin/env python3
"""
LLM processor module for GNN analysis.
"""

import inspect
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404
from datetime import datetime
from pathlib import Path
from typing import Any, List, cast

try:
    import yaml
except ImportError:  # PyYAML is a project dependency; keep processor import-safe
    yaml = cast(Any, None)

_logger = logging.getLogger(__name__)
# Step-13 logging convention: all pipeline-facing messages ride the "llm"
# channel (matching the per-run getLogger("llm") in process_llm).
logger = logging.getLogger("llm")

from gnn.pipeline.config import get_pipeline_config


def _get_llm_config() -> dict:
    """Read LLM configuration from input/config.yaml, with pipeline config recovery."""
    try:
        if os.getenv("GNN_TESTING_NO_LLM_CONFIG"):
            return {}

        # Repo root is parents[3] of src/gnn/llm/processor.py; parents[2] is
        # src/, which silently dropped every ``llm:`` setting (model,
        # max_files, timeouts) after the move under src/gnn/.
        config_path = Path(__file__).resolve().parents[3] / "input" / "config.yaml"
        if config_path.exists():
            if yaml is None:
                _logger.debug(
                    "PyYAML not available; skipping input/config.yaml LLM section"
                )
                return {}
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f) or {}
            return cast("dict[Any, Any]", full_config.get("llm", {}))
    except Exception as e:
        _logger.debug("LLM config YAML load failed: %s", e)
    # Recovery to pipeline config system
    try:
        config = get_pipeline_config()
        return cast("dict[Any, Any]", config.get("llm", {}))
    except Exception as e:
        _logger.debug("LLM pipeline config recovery failed: %s", e)
        return {}


_CROSS_FRAMEWORK_EXECUTE_DIR = "12_execute_output"
_MAX_COMPARISON_MATCHES = 5
_MAX_FRAMEWORK_RESULT_FILES = 8


def _collect_cross_framework_summary(
    output_dir: Path, model_stem: str
) -> dict[str, Any] | None:
    """Collect a light cross-framework comparison summary for one model.

    Scans the Step 12 execute output tree next to ``output_dir`` for a prior
    cross-framework comparison HTML and the sibling per-framework
    ``simulation_results.json`` files, reading only their light identity
    fields (``framework`` and ``validation``). Never executes anything; the
    scan is bounded to a flat directory walk plus per-framework subdirs.

    Returns ``None`` when no comparison artifact exists (quiet absence),
    ``{}`` when an artifact exists but cannot be read (one warning logged),
    or ``{"model", "comparison_html", "frameworks"}`` on success.
    """
    execute_root = output_dir.parent / _CROSS_FRAMEWORK_EXECUTE_DIR
    if not execute_root.is_dir():
        return None
    candidates = sorted(execute_root.glob(f"{model_stem}_comparison.html"))
    candidates += sorted(execute_root.glob(f"*/{model_stem}_comparison.html"))
    candidates = candidates[:_MAX_COMPARISON_MATCHES]
    if not candidates:
        logger.debug("No cross-framework comparison HTML found for %s", model_stem)
        return None
    comparison_html = candidates[0]
    frameworks: list[dict[str, str]] = []
    for results_path in sorted(
        comparison_html.parent.glob("*/simulation_results.json")
    )[:_MAX_FRAMEWORK_RESULT_FILES]:
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read cross-framework results %s: %s", results_path, exc
            )
            return {}
        validation = payload.get("validation")
        if isinstance(validation, dict) and validation.get("all_valid") is True:
            status = "success"
        elif isinstance(validation, dict):
            status = "validation_failed"
        else:
            status = "unknown"
        frameworks.append(
            {
                "framework": str(payload.get("framework", results_path.parent.name)),
                "status": status,
            }
        )
    relative = comparison_html.relative_to(output_dir.parent)
    return {
        "model": model_stem,
        "comparison_html": relative.as_posix(),
        "frameworks": frameworks,
    }


_OLLAMA_MODEL_NAME_PATTERN = re.compile(
    r"""^[A-Za-z0-9][A-Za-z0-9._-]*      # model base name (no leading dash/space)
        (?:/[A-Za-z0-9][A-Za-z0-9._-]*)? # optional namespace
        (?::[A-Za-z0-9][A-Za-z0-9._-]*)? # optional tag (e.g. :8b, :latest)
        $""",
    re.VERBOSE,
)


class ModelNameValidationError(ValueError):
    """Raised when a client/config-supplied Ollama model name is rejected."""


def _validate_model_name(model_name: str) -> str:
    """Validate an Ollama model name before it is interpolated into CLI flags.

    Accepts ``name``, ``namespace/name``, and ``name:tag`` forms of strictly
    alphanumeric/``._-`` characters. Rejects whitespace, leading dashes (flag
    injection like ``--help`` or ``-e``), and any shell metacharacters.
    Raises :class:`ModelNameValidationError` on any violation.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelNameValidationError("model name must be a non-empty string")
    name = model_name.strip()
    if not _OLLAMA_MODEL_NAME_PATTERN.match(name):
        raise ModelNameValidationError(
            f"rejected model name {model_name!r}: must be a plain Ollama "
            "model identifier ([ns/]name[:tag], alphanumerics plus . _ -), "
            "with no spaces, leading dashes, or shell metacharacters"
        )
    return name


def _model_is_cached(model_name: str, logger: logging.Logger) -> bool:
    """Check if an Ollama model is already cached locally using 'ollama show'."""
    try:
        validated = _validate_model_name(model_name)
        result = subprocess.run(  # nosec B607 B603
            ["ollama", "show", validated, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"✅ Model '{validated}' is already cached locally")
            return True
        return False
    except ModelNameValidationError as e:
        logger.warning(f"Rejected invalid model name: {e}")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Model cache check failed for '{model_name}': {e}")
        return False


import asyncio

from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

from .defaults import DEFAULT_OLLAMA_MODEL


def _env_flag(name: str) -> bool:
    """Return whether an opt-in environment flag is explicitly enabled."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _start_ollama_if_needed(logger: Any) -> tuple[bool, list[str]]:
    """
    Check if Ollama is available and running with enhanced detection.

    Returns:
        Tuple of (is_available, list_of_models)
    """
    try:
        if _env_flag("OLLAMA_DISABLED"):
            logger.info("Ollama disabled by OLLAMA_DISABLED; using recovery analysis")
            return False, []

        # Check if ollama command exists
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.info("ℹ️ Ollama not found in PATH - LLM analysis will use recovery")
            return False, []

        logger.info(f"🔍 Found Ollama at: {ollama_path}")

        # Check if Ollama is already running by trying 'ollama list'
        try:
            result = subprocess.run(  # nosec B607 B603
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,  # Increased timeout
            )

            if result.returncode == 0:
                logger.info("✅ Ollama is running and ready")
                # Parse available models
                models: list[Any] = []
                if result.stdout:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:  # Skip header
                        for line in lines[1:]:
                            parts = line.split()
                            if parts:
                                model_name = parts[0]
                                models.append(model_name)

                if models:
                    logger.info(
                        f"📦 Available Ollama models ({len(models)}): {', '.join(models[:5])}"
                    )
                    if len(models) > 5:
                        logger.info(f"   ... and {len(models) - 5} more models")
                else:
                    logger.warning("⚠️ Ollama is running but no models are installed")
                    logger.info(
                        f"To install a model, run: ollama pull {DEFAULT_OLLAMA_MODEL}"
                    )

                return True, models

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Ollama list command timed out (>10s)")
        except Exception as e:
            logger.debug(f"Ollama list check failed: {e}")

        # Starting a daemon is an external side effect.  Only do it when the
        # operator has explicitly opted in; ordinary and CI runs recover
        # immediately when an installed daemon is not already available.
        if not _env_flag("OLLAMA_AUTO_START"):
            logger.info(
                "Ollama is installed but unavailable; set OLLAMA_AUTO_START=1 "
                "to allow automatic daemon startup"
            )
            return False, []

        # Try to start Ollama if it's not running and startup was authorized.
        logger.info("🔄 Attempting to start Ollama...")
        try:
            # Try to start Ollama in background using subprocess.Popen for non-blocking
            # Start Ollama serve in background
            ollama_process = subprocess.Popen(  # nosec B607 B603
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
                if hasattr(os, "setsid")
                else None,  # Create new process group on Unix
            )

            logger.info(f"✅ Ollama started with PID {ollama_process.pid}")

            # Give it a moment to start up
            import time

            ollama_startup_wait = float(os.environ.get("OLLAMA_STARTUP_WAIT", "3"))
            if ollama_startup_wait > 0:
                time.sleep(ollama_startup_wait)

            # Try to check again
            try:
                result = subprocess.run(  # nosec B607 B603
                    ["ollama", "list"], capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    models = []
                    if result.stdout:
                        lines = result.stdout.strip().split("\n")
                        if len(lines) > 1:
                            for line in lines[1:]:
                                parts = line.split()
                                if parts:
                                    model_name = parts[0]
                                    models.append(model_name)

                    if models:
                        logger.info(f"📦 Available Ollama models: {', '.join(models)}")
                    else:
                        logger.warning("⚠️ Ollama started but no models are installed")
                        if _env_flag("OLLAMA_AUTO_PULL"):
                            logger.info("📥 Installing default model...")
                            install_result = subprocess.run(  # nosec B607 B603
                                ["ollama", "pull", DEFAULT_OLLAMA_MODEL],
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            if install_result.returncode == 0:
                                logger.info("✅ Default model installed successfully")
                                models = [DEFAULT_OLLAMA_MODEL]
                            else:
                                logger.warning(
                                    f"⚠️ Failed to install default model: {install_result.stderr}"
                                )
                        else:
                            logger.info(
                                "Automatic model pull disabled; set "
                                "OLLAMA_AUTO_PULL=1 to allow downloads"
                            )

                    return True, models

            except Exception as e:
                logger.debug(f"Post-start check failed: {e}")
                logger.info("ℹ️ Ollama may be starting up, but not ready yet")

        except Exception as e:
            logger.debug(f"Failed to start Ollama: {e}")
            logger.warning("⚠️ Could not start Ollama automatically")

        # If 'ollama list' failed, try to check if the service is running
        # by attempting a simple API endpoint check
        logger.info("🔄 Attempting to check Ollama serve status via API...")
        try:
            # Check if Ollama is serving by testing the API endpoint
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            connect_result = sock.connect_ex(
                ("localhost", 11434)
            )  # Default Ollama port
            sock.close()

            if connect_result == 0:
                logger.info("✅ Ollama server is running on localhost:11434")
                logger.warning("⚠️ Could not list models, but server is responsive")
                return True, []
            else:
                logger.info("ℹ️ Ollama server not responding on localhost:11434")
        except Exception as e:
            logger.debug(f"Socket check failed: {e}")

        # Ollama exists but may not be running - provide helpful instructions
        logger.warning("⚠️ Ollama is installed but may not be running")
        logger.info("📝 To start Ollama, run in a separate terminal:")
        logger.info("   $ ollama serve")
        logger.info("📝 To install a lightweight model for testing:")
        logger.info(f"   $ ollama pull {DEFAULT_OLLAMA_MODEL}")
        logger.info("   $ ollama pull tinyllama")
        logger.info(
            "ℹ️ LLM analysis will use recovery mode without live model interaction"
        )
        return False, []

    except Exception as e:
        logger.debug(f"Error checking Ollama availability: {e}")
        return False, []


def _select_best_ollama_model(
    available_models: List[str], logger: logging.Logger
) -> str:
    """
    Select the best available Ollama model for GNN analysis.

    Priority: environment variable > configured model (if installed) >
    preference list > first available > configured/default fallback.
    """
    # 1. Check environment variable override
    env_model = os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_TEST_MODEL")
    if env_model:
        logger.info(f"🎯 Using model from environment: {env_model}")
        return env_model

    # 2. Respect config.yaml when that model is available locally
    llm_config = _get_llm_config()
    config_model = llm_config.get("model")
    if config_model:
        for available in available_models:
            if available.startswith(config_model):
                logger.info(f"🎯 Using model from config.yaml: {available}")
                return available

    # 3. Preference order: prioritize smaller/faster models for reliability
    preferred_models: list[Any] = [
        "smollm2",
        "tinyllama",
        "gemma3:4b",
        "gemma2:2b",
        "ministral-3:3b",
        "mistral:7b",
        "llama2:7b",
        "phi3",
        "llama2",
        "mistral",
    ]

    # Find first available model from preference list
    for preferred in preferred_models:
        for available in available_models:
            if available.startswith(preferred):
                logger.info(f"🎯 Selected model: {available}")
                return available

    # 4. Recovery to first available model
    if available_models:
        model = available_models[0]
        logger.info(f"🎯 Using first available model: {model}")
        return model

    # 5. Ultimate recovery
    default_model = llm_config.get("model", DEFAULT_OLLAMA_MODEL)
    logger.warning(f"⚠️ No models found, defaulting to: {default_model}")
    logger.info(f"   Note: You may need to run: ollama pull {default_model}")
    return cast("str", default_model)


from .analyzer import analyze_gnn_file_with_llm
from .cache import LLMCache
from .generator import (
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary,
    generate_model_insights,
)
from .llm_processor import LLMProcessor, ProviderType
from .prompts import PromptType, get_prompt
from .providers.base_provider import LLMConfig, LLMMessage

_AUTH_ERROR_MARKERS = ("401", "403", "invalid_api_key", "incorrect api key")


def _classify_auth_error(error_str: str) -> str | None:
    """Return the provider name when an LLM error is an auth failure, else None.

    Attribution scans the lowercased error for a known provider name and
    falls back to ``"unknown"`` when the message carries an auth marker but
    no recognizable provider.
    """
    lowered = error_str.lower()
    if not any(marker in lowered for marker in _AUTH_ERROR_MARKERS):
        return None
    for prov_name in ("openai", "anthropic", "openrouter", "perplexity", "ollama"):
        if prov_name in lowered:
            return prov_name
    return "unknown"


def _prompt_fallback_text(label: str, custom: bool) -> str:
    """Recovery text recorded for a prompt that could not be executed."""
    qualifier = f"custom prompt {label}" if custom else label
    return (
        f"LLM analysis for {qualifier} was not available. "
        "Please ensure that Ollama is running and the required model is installed."
    )


async def _execute_prompt(
    processor: LLMProcessor,
    cache: LLMCache,
    *,
    cache_content: str,
    model_name: str,
    messages: list[LLMMessage],
    prompt_text: str,
    label: str,
    custom: bool,
    max_tokens: int,
    max_prompt_timeout: float,
    failed_auth_providers: set[str],
    auth_errors: list[dict[str, Any]],
) -> str:
    """Run one LLM prompt with cache lookup, timeout, and auth fail-fast.

    Shared funnel for the structured ``PromptType`` sequence and free-form
    custom prompts. Returns the text to record for ``label``; never raises.

    Args:
        processor: Initialized multi-provider processor (required non-None).
        cache: Response cache keyed on content + model + prompt text.
        cache_content: Raw GNN content used in the cache key.
        model_name: Model tag passed to the provider and cache key.
        messages: Chat messages for this prompt.
        prompt_text: Prompt text used in the cache key.
        label: Identity used in logs and recovery text.
        custom: Whether this is a free-form custom prompt (affects wording).
        max_tokens: Response token cap.
        max_prompt_timeout: Per-prompt wall-clock budget in seconds.
        failed_auth_providers: Providers that already failed auth (mutated).
        auth_errors: Auth-failure records for the run summary (mutated).
    """
    cached = cache.get(cache_content, model_name, prompt_text)
    if cached is not None:
        logger.info(f"  ⚡ Cache HIT for {label}")
        return cached

    # Fail fast when the provider that would serve this request already
    # failed authentication (mirrors get_response's default routing). The
    # peek is duck-typed so in-memory provider doubles without routing introspection skip it.
    get_default = getattr(processor, "get_default_provider", None)
    provider = get_default() if callable(get_default) else None
    if provider is not None and provider.provider_type.value in failed_auth_providers:
        logger.warning(
            f"  ⏭️ Skipping {label} — provider "
            f"'{provider.provider_type.value}' previously failed auth"
        )
        return _prompt_fallback_text(label, custom)

    try:
        resp = await asyncio.wait_for(
            processor.get_response(
                messages=messages,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=0.2,
                config=LLMConfig(timeout=60),
            ),
            timeout=max_prompt_timeout,
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        if not content or content.strip() == "":
            kind = "custom prompt" if custom else "prompt"
            content = (
                f"No response generated for {kind} {label}. This may indicate "
                "that the LLM provider is not available or not responding."
            )
        cache.put(cache_content, model_name, prompt_text, content)
        logger.debug("  ✅ Prompt completed successfully")
        return content
    except asyncio.TimeoutError:
        error_msg = f"Prompt execution timed out after {max_prompt_timeout} seconds"
        logger.error(f"  ❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_str = str(e)
        auth_provider = _classify_auth_error(error_str)
        if auth_provider is not None:
            if auth_provider not in failed_auth_providers:
                failed_auth_providers.add(auth_provider)
                auth_errors.append(
                    {"provider": auth_provider, "error": error_str[:200]}
                )
                logger.error(
                    f"  🔑 Auth failure for {auth_provider} — skipping future "
                    "calls to this provider"
                )
            logger.error(f"  ❌ Auth error ({auth_provider}): {error_str[:100]}")
        else:
            kind = "Custom prompt" if custom else "Prompt"
            logger.error(f"  ❌ {kind} execution failed: {e}")
        return _prompt_fallback_text(label, custom)


def _optional_positive_int(value: Any) -> int | None:
    """Normalize optional positive integer config values."""
    if value in (None, "", False):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _resolve_llm_budget_seconds(
    kwargs: dict[str, Any], llm_config: dict[str, Any]
) -> int:
    """Resolve the total LLM budget from CLI kwargs, then config, then default."""
    for key in ("total_budget", "llm_timeout"):
        budget = _optional_positive_int(kwargs.get(key))
        if budget is not None:
            return budget
    return _optional_positive_int(llm_config.get("timeout_seconds")) or 600


def _resolve_llm_max_files(
    kwargs: dict[str, Any], llm_config: dict[str, Any]
) -> int | None:
    """Resolve the optional file cap for bounded pipeline LLM runs."""
    for key in ("max_files", "llm_max_files"):
        max_files = _optional_positive_int(kwargs.get(key))
        if max_files is not None:
            return max_files
    return _optional_positive_int(llm_config.get("max_files"))


def _llm_file_sort_key(path: Path) -> tuple[int, str]:
    """Sort deterministically and keep the large scaling corpus behind examples."""
    return (1 if "pymdp_scaling_study" in path.parts else 0, str(path))


def process_llm(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process GNN files with LLM-enhanced analysis.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    import asyncio

    # Run the async implementation in a single event loop
    try:
        return asyncio.run(
            _process_llm_async(target_dir, output_dir, verbose, **kwargs)
        )
    except Exception as e:
        logger = logging.getLogger("llm")
        log_step_error(logger, f"LLM processing failed: {e}")
        return False


async def _process_llm_async(
    target_dir: Path, output_dir: Path, verbose: bool, **kwargs: Any
) -> bool:
    """Async implementation of process_llm."""
    import time as _time

    logger = logging.getLogger("llm")

    # Initialize processor variable for cleanup in finally block
    processor = None

    # Initialize LLM response cache
    cache = LLMCache(cache_dir=output_dir / ".cache")

    # Total budget: CLI overrides config so long pipeline runs can scale the LLM step.
    llm_config = _get_llm_config()
    TOTAL_BUDGET_SECONDS = _resolve_llm_budget_seconds(kwargs, llm_config)
    budget_start = _time.monotonic()

    def _budget_remaining() -> float:
        """Handle budget remaining for internal callers."""
        return max(0.0, TOTAL_BUDGET_SECONDS - (_time.monotonic() - budget_start))

    try:
        log_step_start(logger, "Processing LLM with enhanced Ollama integration")

        # Check if Ollama is available and running with model detection
        ollama_available, ollama_models = _start_ollama_if_needed(logger)

        # Select best model if Ollama is available
        selected_model = None
        if ollama_available and ollama_models:
            selected_model = _select_best_ollama_model(ollama_models, logger)
        elif ollama_available:
            # Ollama running but no models listed - use default
            selected_model = _select_best_ollama_model([], logger)
        else:
            logger.info(
                "ℹ️ Proceeding with recovery LLM analysis (no live model interaction)"
            )

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "auth_errors": [],
            "provider_matrix": {
                "ollama": {
                    "available": ollama_available,
                    "models": ollama_models,
                    "selected_model": selected_model,
                },
                "openai": {
                    "available": bool(os.getenv("OPENAI_API_KEY")),
                    "models": ["gpt-4", "gpt-3.5-turbo"]
                    if os.getenv("OPENAI_API_KEY")
                    else [],
                    "selected_model": None,
                },
                "anthropic": {
                    "available": bool(os.getenv("ANTHROPIC_API_KEY")),
                    "models": ["claude-3", "claude-2"]
                    if os.getenv("ANTHROPIC_API_KEY")
                    else [],
                    "selected_model": None,
                },
            },
            "analysis_results": [],
            "model_insights": [],
            "code_suggestions": [],
            "documentation_generated": [],
        }

        # Track providers with auth failures to fail-fast on subsequent calls
        failed_auth_providers: set[Any] = set()

        # Find GNN files (recursive to handle subdirectory structure)
        # README/AGENTS/INDEX etc. are documentation, not models.
        from gnn.processing.discovery import is_model_source_path

        discovered_gnn_files = sorted(
            (path for path in target_dir.rglob("*.md") if is_model_source_path(path)),
            key=_llm_file_sort_key,
        )
        max_files = _resolve_llm_max_files(kwargs, llm_config)
        gnn_files = (
            discovered_gnn_files[:max_files]
            if max_files is not None
            else discovered_gnn_files
        )
        if max_files is not None and len(discovered_gnn_files) > len(gnn_files):
            logger.info(
                "LLM file selection limited to %s/%s files by config",
                len(gnn_files),
                len(discovered_gnn_files),
            )

        results["total_files_discovered"] = len(discovered_gnn_files)
        results["selected_files"] = len(gnn_files)
        results["skipped_files"] = max(0, len(discovered_gnn_files) - len(gnn_files))
        results["file_selection"] = {
            "max_files": max_files,
            "policy": "sorted, non-scaling fixtures first",
            "selected_paths": [str(path) for path in gnn_files],
        }

        if not gnn_files:
            logger.warning("No GNN files found for LLM processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            # Initialize LLM processor (prioritize Ollama)
            processor_initialized = False
            try:
                # Create processor with Ollama prioritized
                processor = LLMProcessor(
                    preferred_providers=[
                        ProviderType.OLLAMA,
                        ProviderType.OPENAI,
                        ProviderType.OPENROUTER,
                        ProviderType.PERPLEXITY,
                    ]
                )
                processor_initialized = await processor.initialize()

                if not processor_initialized:
                    logger.warning(
                        "LLM processor initialization failed - using recovery analysis"
                    )
                else:
                    logger.info(
                        f"LLM processor initialized with providers: {[p.value for p in processor.get_available_providers()]}"
                    )
            except Exception as e:
                logger.warning(
                    f"LLM processor initialization failed: {e} - using recovery analysis"
                )
                processor_initialized = False
                processor = cast(Any, None)

            # Process each GNN file
            for file_idx, gnn_file in enumerate(gnn_files, 1):
                # Check total budget before starting a new file
                remaining = _budget_remaining()
                if remaining < 30:
                    logger.warning(
                        f"⏱️ Budget exhausted after {file_idx - 1}/{len(gnn_files)} files — skipping remaining"
                    )
                    break
                logger.info(
                    f"📄 File {file_idx}/{len(gnn_files)}: {gnn_file.name} (budget: {remaining:.0f}s remaining)"
                )
                try:
                    # Await the coroutine since we're in an async context
                    resolved_ollama_for_summary = (
                        (selected_model if selected_model else DEFAULT_OLLAMA_MODEL)
                        if ollama_available
                        else None
                    )
                    analysis_candidate = analyze_gnn_file_with_llm(
                        gnn_file,
                        verbose,
                        ollama_model=resolved_ollama_for_summary,
                        attempt_llm=processor_initialized,
                    )
                    file_analysis = (
                        await analysis_candidate
                        if inspect.isawaitable(analysis_candidate)
                        else analysis_candidate
                    )
                    results["analysis_results"].append(file_analysis)

                    # Generate insights
                    insights = generate_model_insights(file_analysis)
                    results["model_insights"].append(insights)

                    # Generate code suggestions
                    suggestions = generate_code_suggestions(file_analysis)
                    results["code_suggestions"].append(suggestions)

                    # Generate documentation
                    docs = generate_documentation(file_analysis)
                    results["documentation_generated"].append(docs)

                    # If LLM processor is available, run structured prompts and save outputs
                    if processor_initialized and processor:
                        with open(gnn_file, "r") as f:
                            gnn_content = f.read()

                        # --- BEGIN ONTOLOGY INJECTION ---
                        ontology_file = (
                            output_dir.parent
                            / "10_ontology_output"
                            / "ontology_results.json"
                        )
                        if ontology_file.exists():
                            try:
                                with open(ontology_file, "r") as fn:
                                    ont_data = json.load(fn)
                                gnn_content += f"\n\n--- INJECTED ACTIVE INFERENCE ONTOLOGY META ---\n{json.dumps(ont_data, indent=2)}\n"
                                logger.info(
                                    "🧠 Injected Neurosymbolic Ontology context into LLM prompt."
                                )
                            except Exception as oe:
                                logger.warning(
                                    f"Could not inject ontology metadata: {oe}"
                                )
                        # --- END ONTOLOGY INJECTION ---
                        # --- BEGIN CROSS-FRAMEWORK INJECTION ---
                        try:
                            comparison_meta = _collect_cross_framework_summary(
                                output_dir, gnn_file.stem
                            )
                        except Exception as comparison_error:
                            logger.warning(
                                f"Could not collect cross-framework metadata: {comparison_error}"
                            )
                            comparison_meta = None
                        if comparison_meta:
                            gnn_content += (
                                "\n\n--- INJECTED CROSS-FRAMEWORK COMPARISON META ---\n"
                                f"{json.dumps(comparison_meta, indent=2)}\n"
                            )
                            logger.info(
                                "🔬 Injected cross-framework comparison context into LLM prompt."
                            )
                        # --- END CROSS-FRAMEWORK INJECTION ---
                        # Build custom prompt sequence including user-requested prompts
                        prompt_sequence: list[Any] = [
                            PromptType.SUMMARIZE_CONTENT,
                            PromptType.EXPLAIN_MODEL,
                            PromptType.IDENTIFY_COMPONENTS,
                            PromptType.ANALYZE_STRUCTURE,
                            PromptType.EXTRACT_PARAMETERS,
                            PromptType.PRACTICAL_APPLICATIONS,
                        ]

                        # Allow custom_prompts to be overridden via kwargs
                        custom_prompts = kwargs.get(
                            "custom_prompts",
                            [
                                (
                                    "technical_description",
                                    "Describe this GNN model comprehensively, in technical detail.",
                                ),
                                (
                                    "nontechnical_description",
                                    "Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.",
                                ),
                                (
                                    "runtime_behavior",
                                    "Describe what happens when this GNN model runs and how it would behave in different settings or domains.",
                                ),
                            ],
                        )

                        per_file_dir = results_dir / f"prompts_{gnn_file.stem}"
                        per_file_dir.mkdir(parents=True, exist_ok=True)

                        prompt_outputs: dict[Any, Any] = {}
                        # Use selected model or recovery
                        ollama_model = (
                            selected_model if selected_model else DEFAULT_OLLAMA_MODEL
                        )
                        try:
                            ollama_model = _validate_model_name(ollama_model)
                        except ModelNameValidationError as e:
                            # Flag-injection guard: never interpolate an
                            # unvalidated model name into an ollama CLI flag.
                            logger.warning(
                                f"⚠️ Invalid model name; falling back to default: {e}"
                            )
                            ollama_model = DEFAULT_OLLAMA_MODEL
                        logger.info(f"🤖 Using model '{ollama_model}' for LLM prompts")
                        # Per-prompt timeout: config-driven, recovery to kwargs.
                        # Loop-invariant, shared by structured and custom prompts.
                        max_prompt_timeout = kwargs.get(
                            "max_prompt_timeout",
                            llm_config.get("prompt_timeout", 45),
                        )

                        # Ensure model is available — use pre-pull guard to skip if cached
                        if (
                            ollama_available
                            and ollama_model not in ollama_models
                            and _env_flag("OLLAMA_AUTO_PULL")
                        ):
                            if _model_is_cached(ollama_model, logger):
                                logger.info(
                                    f"⏭️ Skipping pull — '{ollama_model}' already cached"
                                )
                            else:
                                logger.info(
                                    f"📥 Pulling model '{ollama_model}' (not cached)..."
                                )
                                try:
                                    install_result = subprocess.run(  # nosec B607 B603
                                        ["ollama", "pull", ollama_model],
                                        capture_output=True,
                                        text=True,
                                        timeout=120,
                                    )
                                    if install_result.returncode == 0:
                                        logger.info(
                                            f"✅ Model '{ollama_model}' pulled successfully"
                                        )
                                    else:
                                        logger.warning(
                                            f"⚠️ Failed to pull model '{ollama_model}': {install_result.stderr}"
                                        )
                                except subprocess.TimeoutExpired:
                                    logger.warning(
                                        "⚠️ Model pull timed out after 120s — continuing with recovery"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"⚠️ Could not pull model '{ollama_model}': {e}"
                                    )

                        for idx, ptype in enumerate(prompt_sequence, start=1):
                            # Check budget before each prompt
                            if _budget_remaining() < 30:
                                logger.warning(
                                    f"⏱️ Budget exhausted ({TOTAL_BUDGET_SECONDS}s), skipping remaining prompts for {gnn_file.name}"
                                )
                                break
                            prompt_cfg = get_prompt(ptype, gnn_content)
                            messages: list[Any] = [
                                LLMMessage(
                                    role="system", content=prompt_cfg["system_message"]
                                ),
                                LLMMessage(
                                    role="user", content=prompt_cfg["user_prompt"]
                                ),
                            ]

                            # Log progress
                            logger.info(
                                f"  📝 Running prompt {idx}/{len(prompt_sequence)}: {ptype.value}"
                            )

                            prompt_outputs[ptype.value] = await _execute_prompt(
                                processor,
                                cache,
                                cache_content=gnn_content,
                                model_name=ollama_model,
                                messages=messages,
                                prompt_text=prompt_cfg["user_prompt"],
                                label=ptype.value,
                                custom=False,
                                max_tokens=min(512, prompt_cfg.get("max_tokens", 512)),
                                max_prompt_timeout=max_prompt_timeout,
                                failed_auth_providers=failed_auth_providers,
                                auth_errors=results["auth_errors"],
                            )

                            # Write to file
                            out_path = per_file_dir / f"{ptype.value}.md"
                            with open(out_path, "w") as outf:
                                outf.write(f"# {ptype.name}\n\n")
                                outf.write(prompt_outputs[ptype.value] or "")

                        # Run custom free-form prompts
                        for cust_idx, (key, user_prompt) in enumerate(
                            custom_prompts, start=1
                        ):
                            # Check budget before each custom prompt
                            if _budget_remaining() < 30:
                                logger.warning(
                                    f"⏱️ Budget exhausted ({TOTAL_BUDGET_SECONDS}s), skipping remaining custom prompts for {gnn_file.name}"
                                )
                                break
                            logger.info(
                                f"  📝 Running custom prompt {cust_idx}/{len(custom_prompts)}: {key}"
                            )

                            messages = [
                                LLMMessage(
                                    role="system",
                                    content="You are an expert in Active Inference and GNN specifications.",
                                ),
                                LLMMessage(
                                    role="user",
                                    content=f"{user_prompt}\n\nGNN Model Content:\n{gnn_content}",
                                ),
                            ]

                            prompt_outputs[key] = await _execute_prompt(
                                processor,
                                cache,
                                cache_content=gnn_content,
                                model_name=ollama_model,
                                messages=messages,
                                prompt_text=user_prompt,
                                label=key,
                                custom=True,
                                max_tokens=512,
                                max_prompt_timeout=max_prompt_timeout,
                                failed_auth_providers=failed_auth_providers,
                                auth_errors=results["auth_errors"],
                            )

                            out_path = per_file_dir / f"{key}.md"
                            with open(out_path, "w") as outf:
                                title = key.replace("_", " ").title()
                                outf.write(f"# {title}\n\n")
                                outf.write(f"Prompt:\n\n> {user_prompt}\n\n")
                                outf.write("Response:\n\n")
                                outf.write(prompt_outputs[key] or "")

                        # Attach to file analysis record
                        file_analysis["llm_prompt_outputs"] = prompt_outputs

                except Exception as e:
                    error_info: dict[str, Any] = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
                finally:
                    results["processed_files"] += 1

        # Save detailed results (include cache stats)
        results["cache_stats"] = cache.summary()
        results_file = results_dir / "llm_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Generate summary report
        summary = generate_llm_summary(results)
        summary_file = results_dir / "llm_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary)

        # Log cache summary
        cs = cache.summary()
        logger.info(
            f"📦 Cache: {cs['hits']} hits, {cs['misses']} misses, {cs['writes']} new entries ({cs['hit_ratio_pct']}% hit ratio)"
        )

        # Surface auth errors in final status
        if results["auth_errors"]:
            providers_failed = [e["provider"] for e in results["auth_errors"]]
            log_step_warning(
                logger,
                f"LLM processing completed with auth errors for: {', '.join(providers_failed)}",
            )
            logger.warning(
                "💡 Check your API keys — invalid keys waste time on retries"
            )
            # Mark as not fully successful so pipeline reports SUCCESS_WITH_WARNINGS
            return False
        elif results["success"]:
            log_step_success(logger, "LLM processing completed successfully")
        else:
            log_step_error(logger, "LLM processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(logger, f"LLM processing failed: {e}")
        return False
    finally:
        # Close processor connections
        if processor:
            await processor.close()
