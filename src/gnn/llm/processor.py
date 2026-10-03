#!/usr/bin/env python3
"""
LLM processor module for GNN analysis.
"""

import json
import logging
import math
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Callable, List, cast

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
    """Use the resolved invocation snapshot; never infer a checkout location."""
    from gnn.pipeline.run_context import effective_input_config

    resolved = effective_input_config()
    if resolved is not None:
        return cast("dict[Any, Any]", resolved.get("llm", {}))
    if os.getenv("GNN_TESTING_NO_LLM_CONFIG"):
        return {}
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

from gnn.utils.logging_utils import log_step_error

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


from .analyzer import analyze_gnn_file_with_llm as analyze_gnn_file_with_llm
from .cache import LLMCache
from .generator import (
    generate_code_suggestions as generate_code_suggestions,
)
from .generator import (
    generate_documentation as generate_documentation,
)
from .generator import (
    generate_llm_summary as generate_llm_summary,
)
from .generator import (
    generate_model_insights as generate_model_insights,
)
from .llm_processor import LLMProcessor, ProviderType
from .prompts import PromptType as PromptType
from .prompts import get_prompt as get_prompt
from .providers.base_provider import LLMConfig, LLMMessage

_LIVE_PROCESSOR_CLASS = LLMProcessor

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
    outcomes: dict[str, Any] | None = None,
    provider_type: ProviderType | None = None,
    isolated: bool = False,
    endpoint_url: str | None = None,
    remaining_budget: Callable[[], float] | None = None,
) -> str:
    """Run one LLM prompt with cache lookup, timeout, and auth fail-fast.

    Shared funnel for the structured ``PromptType`` sequence and free-form
    custom prompts. Returns the text to record for ``label``; never raises.

    Args:
        processor: Initialized multi-provider processor (required non-None).
        cache: Response cache bound to content and the full request definition.
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
    if outcomes is not None:
        outcomes[label] = {"status": "failed"}
    import time

    prompt_deadline = time.monotonic() + max_prompt_timeout

    def check_deadline() -> None:
        if time.monotonic() >= prompt_deadline or (
            remaining_budget is not None and remaining_budget() <= 0
        ):
            raise asyncio.TimeoutError("Required prompt work exceeded its deadline")

    if provider_type and provider_type.value in failed_auth_providers:
        return _prompt_fallback_text(label, custom)
    request_definition = json.dumps(
        {
            "schema_version": 2,
            "provider": provider_type.value if provider_type else "default",
            "model": model_name,
            "messages": [
                {"role": message.role, "content": message.content, "name": message.name}
                for message in messages
            ],
            "prompt_text": prompt_text,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        sort_keys=True,
    )
    cached = cache.get(cache_content, model_name, request_definition)
    if isinstance(cached, str) and cached.strip():
        if time.monotonic() >= prompt_deadline or (
            remaining_budget is not None and remaining_budget() <= 0
        ):
            if outcomes is not None:
                outcomes[label] = {"status": "timed_out"}
            return (
                f"Prompt cache retrieval timed out after {max_prompt_timeout} seconds"
            )
        logger.info(f"  ⚡ Cache HIT for {label}")
        if outcomes is not None:
            outcomes[label] = {"status": "success", "cached": True}
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
        check_deadline()
        request_timeout = max(0.0, prompt_deadline - time.monotonic())
        if remaining_budget is not None:
            request_timeout = min(request_timeout, remaining_budget())
        if isolated:
            from .request_worker import isolated_request

            resp = await isolated_request(
                provider=(provider_type or ProviderType.OLLAMA).value,
                model=model_name,
                messages=[
                    {"role": msg.role, "content": msg.content} for msg in messages
                ],
                max_tokens=max_tokens,
                timeout=request_timeout,
                endpoint_url=endpoint_url,
            )
        else:
            resp = await asyncio.wait_for(
                processor.get_response(
                    messages=messages,
                    model_name=model_name,
                    provider_type=provider_type,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    config=LLMConfig(timeout=request_timeout),
                ),
                timeout=request_timeout,
            )
        check_deadline()
        content = resp.content if hasattr(resp, "content") else str(resp)
        if not content or content.strip() == "":
            kind = "custom prompt" if custom else "prompt"
            return (
                f"No response generated for {kind} {label}. This may indicate "
                "that the LLM provider is not available or not responding."
            )
        cache.put(cache_content, model_name, request_definition, content)
        check_deadline()
        if outcomes is not None:
            outcomes[label] = {"status": "success", "cached": False}
        logger.debug("  ✅ Prompt completed successfully")
        return content
    except asyncio.TimeoutError:
        if outcomes is not None:
            outcomes[label] = {"status": "timed_out"}
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
    kwargs: dict[str, Any], llm_config: dict[str, Any], selected_count: int = 1
) -> float:
    """Resolve the total LLM budget from CLI kwargs, then config, then default."""
    for key in ("total_budget", "llm_timeout"):
        value = kwargs.get(key)
        if value is not None:
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
            return float(value)
    value = llm_config.get("timeout_seconds")
    if value is not None:
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("LLM timeout_seconds must be finite and positive")
        return float(value)
    return 600 * max(1, selected_count)


def _resolve_llm_max_files(
    kwargs: dict[str, Any], llm_config: dict[str, Any]
) -> int | None:
    """Resolve the optional file cap for bounded pipeline LLM runs."""
    for key in ("max_files", "llm_max_files"):
        value = kwargs.get(key)
        if value is not None:
            if isinstance(value, bool) or str(value) != str(
                _optional_positive_int(value)
            ):
                raise ValueError(f"{key} must be a positive integer")
            return int(value)
    value = llm_config.get("max_files")
    if value is not None and (
        isinstance(value, bool) or str(value) != str(_optional_positive_int(value))
    ):
        raise ValueError("LLM max_files must be a positive integer")
    return _optional_positive_int(value)


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
    """Delegate corpus scheduling while preserving public helper injection."""
    import sys

    from .corpus_runner import run_corpus

    return await run_corpus(
        target_dir, output_dir, verbose, kwargs, sys.modules[__name__]
    )
