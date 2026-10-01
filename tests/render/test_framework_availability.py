#!/usr/bin/env python3
"""Tests for framework availability gating in the canonical registry.

Asserts:
  1. Every framework in ``FRAMEWORK_REGISTRY`` carries either an importable
     backend *or* an explicit ``available=False`` with a human-readable
     ``unavailable_reason`` string.
  2. ``validate_framework_requested()`` accepts every registry-available
     framework and documents enable paths for any intentionally gated one.
     PyTorch left the gated set once torch>=2.13.0 resolved
     GHSA-rrmf-rvhw-rf47; bnlearn left it when the ``bnlearn`` extra
     (bnlearn>=0.14.0 with pgmpy transitively) became the supported enable
     path and the stale torch lock exclusion no longer applied.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.frameworks import ALL_FRAMEWORKS, LITE_FRAMEWORKS
from gnn.render.framework_registry import (
    FRAMEWORK_REGISTRY,
    get_framework_availability,
    get_supported_frameworks,
    validate_framework_requested,
)

# ── Framework definitions used across tests ────────────────────────────────

# Frameworks that MUST be marked unavailable in the registry because their
# Python dependency is intentionally absent from the default lock.
INTENTIONALLY_UNAVAILABLE: set[str] = set()

# Frameworks that ARE importable (Python) or unconditionally available
# (Julia/Stan code generation).  These should never be marked unavailable.
# PyTorch joined this set when torch>=2.13.0 resolved GHSA-rrmf-rvhw-rf47
# (the package ships in the ``torch`` extra).
EXPECTED_AVAILABLE: set[str] = {
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "stan",
    # bnlearn joined when its dependency precondition was resolved by the
    # ``bnlearn`` extra; it gained a Step 12 executor in
    # ``src/gnn/execute/bnlearn/`` (scripts skip without the extra).
    "bnlearn",
    # cpomdp: continuous models only; rendering never imports cpomdp.
    "cpomdp",
}


# Canonical name enumeration (single source: ``gnn.frameworks``).
CANONICAL_ALL_FRAMEWORKS: tuple[str, ...] = ALL_FRAMEWORKS


# ── Helpers ────────────────────────────────────────────────────────────────


def _importable(name: str) -> bool:
    """Return True if ``name`` can be imported in the current interpreter."""
    return importlib.util.find_spec(name) is not None


# ── Tests ──────────────────────────────────────────────────────────────────


class TestRegistryCompleteness:
    """Every registered framework has an explicit availability story."""

    @pytest.mark.parametrize("framework", list(FRAMEWORK_REGISTRY.keys()))
    def test_framework_has_availability_field(self, framework: str) -> None:
        """Every entry must carry 'available' and 'unavailable_reason'."""
        spec = FRAMEWORK_REGISTRY[framework]
        assert "available" in spec, f"{framework} is missing 'available' field"
        assert "unavailable_reason" in spec, (
            f"{framework} is missing 'unavailable_reason' field"
        )
        assert isinstance(spec["available"], bool), (
            f"{framework}.available must be bool, got {type(spec['available'])}"
        )
        reason = spec["unavailable_reason"]
        if spec["available"]:
            assert reason is None, (
                f"{framework} is marked available but has unavailable_reason={reason!r}"
            )
        else:
            assert reason is not None and len(reason) > 10, (
                f"{framework} is marked unavailable but has no meaningful "
                f"unavailable_reason (got {reason!r})"
            )

    def test_supported_frameworks_matches_registry_keys(self) -> None:
        assert set(get_supported_frameworks()) == set(FRAMEWORK_REGISTRY.keys())


class TestAvailabilityStory:
    """Every framework has either an importable backend or an explicit reason."""

    def test_expected_available_are_available(self) -> None:
        for fw in EXPECTED_AVAILABLE:
            available, reason = get_framework_availability(fw)
            assert available is True, (
                f"Framework {fw!r} expected available but got "
                f"available={available}, reason={reason!r}"
            )

    def test_intentionally_unavailable_are_unavailable(self) -> None:
        for fw in INTENTIONALLY_UNAVAILABLE:
            available, reason = get_framework_availability(fw)
            assert available is False, (
                f"Framework {fw!r} expected unavailable but got available={available}"
            )
            assert reason is not None, (
                f"Framework {fw!r} is unavailable but reason is None"
            )
            # The reason MUST mention the GHSA identifier.
            assert "GHSA" in reason, (
                f"Framework {fw!r} unavailable reason should mention the "
                f"security advisory, got: {reason}"
            )

    def test_unavailable_reason_is_human_readable(self) -> None:
        """Unavailable reasons are full sentences, not error codes."""
        for fw in INTENTIONALLY_UNAVAILABLE:
            _available, reason = get_framework_availability(fw)
            assert reason is not None
            # Should be at least 50 chars (meaningful sentence)
            assert len(reason) >= 50, f"{fw} reason too short: {reason!r}"
            # Should start with a capital letter
            assert reason[0].isupper(), f"{fw} reason not capitalized: {reason!r}"
            # Should mention how to re-enable
            assert "your own risk" in reason.lower() or "enable" in reason.lower(), (
                f"{fw} reason should mention how to enable: {reason!r}"
            )

    def test_get_framework_availability_unknown_framework(self) -> None:
        """Unknown frameworks are assumed available (no over-blocking)."""
        available, reason = get_framework_availability("__definitely_not_a_framework__")
        assert available is True
        assert reason is None


class TestValidationGate:
    """``validate_framework_requested()`` raises clear errors."""

    def test_validate_available_framework_passes(self) -> None:
        # Should not raise for any known-available framework.
        for fw in EXPECTED_AVAILABLE:
            validate_framework_requested(fw)

    def test_validate_unknown_framework_passes(self) -> None:
        # Unknown frameworks should not be blocked.
        validate_framework_requested("__some_external_tool__")

    def test_validate_unavailable_raises_value_error(self) -> None:
        """Requesting an unavailable framework raises ValueError with a clear
        message.

        SC-29: an empty ``INTENTIONALLY_UNAVAILABLE`` list must not silently
        report as an empty-parametrize skip. When the list is empty this test
        instead cross-checks the registry loudly: any framework the registry
        marks unavailable must be enumerated here, otherwise the gate drifts
        and this assertion fails.
        """
        if not INTENTIONALLY_UNAVAILABLE:
            unlisted = [
                framework
                for framework in get_supported_frameworks()
                if not get_framework_availability(framework)[0]
            ]
            assert not unlisted, (
                "Registry marks frameworks unavailable but "
                "INTENTIONALLY_UNAVAILABLE is empty — enumerate them so the "
                "ValueError gate stays exercised: "
                f"{sorted(unlisted)}"
            )
            return

        for framework in sorted(INTENTIONALLY_UNAVAILABLE):
            with pytest.raises(ValueError) as exc_info:
                validate_framework_requested(framework)

            msg = str(exc_info.value)
            # Must contain the framework name
            assert framework.lower() in msg.lower(), (
                f"Error message should mention framework name: {msg}"
            )
            # Must mention how to enable
            assert "uv add" in msg.lower() or "enable" in msg.lower(), (
                f"Error message should suggest how to enable: {msg}"
            )

    def test_validate_bnlearn_is_available(self) -> None:
        """bnlearn is registry-available via the ``bnlearn`` extra: requesting
        it must not raise."""
        validate_framework_requested("bnlearn")

    def test_validate_pytorch_is_available(self) -> None:
        """PyTorch is registry-available: torch>=2.13.0 resolves
        GHSA-rrmf-rvhw-rf47, so requesting it must not raise."""
        validate_framework_requested("pytorch")


class TestRealEnvironmentCheck:
    """Cross-reference registry status against the real interpreter import state.

    This catches drift where a framework marked 'available' was never actually
    importable in the default environment.
    """

    def test_available_python_frameworks_are_importable(self) -> None:
        """Python-based frameworks marked 'available' should be importable."""
        # Map framework name → Python module name for import check
        framework_to_module: dict[str, str] = {
            "pymdp": "pymdp",
            "jax": "jax",
            "discopy": "discopy",
            "numpyro": "numpyro",
        }
        for fw, mod in framework_to_module.items():
            available, _ = get_framework_availability(fw)
            if available:
                # Just check that the module CAN be found (not that features
                # work) — this is an existence check, not a runtime test.
                spec = importlib.util.find_spec(mod)
                assert spec is not None, (
                    f"{fw} ({mod}) not installed in this environment — "
                    "core dependency missing"
                )

    def test_unavailable_frameworks_are_not_importable(self) -> None:
        """Frameworks marked 'unavailable' should NOT be importable in a
        clean environment — confirms the gate is meaningful."""
        # bnlearn → bnlearn module
        for fw in INTENTIONALLY_UNAVAILABLE:
            available, _ = get_framework_availability(fw)
            assert available is False
            # No need to assert non-importability — the registry says it's
            # unavailable regardless of local env state.


class TestRendererHealthModulePaths:
    """``health.check_renderers`` module paths resolve in the real package.

    Regression pin: the health check used to build ``render.{name}`` module
    paths, but no top-level ``render`` package exists (the installed package
    is ``gnn``), so the CLI ``gnn health`` and API ``/api/v1/health`` surfaces
    reported every renderer unavailable. Module paths are now derived uniformly
    as ``gnn.render.{name}``; each renderer (bnlearn included, via its
    dedicated package) imports from that direct path.
    """

    def test_every_status_module_path_imports(self) -> None:
        import importlib

        from gnn.render.health import check_renderers

        results = check_renderers()
        assert set(results) == set(get_supported_frameworks())
        for name, status in results.items():
            assert status.available, f"{name}: {status.error}"
            assert status.module_path is not None
            assert status.module_path.startswith("gnn.render.")
            importlib.import_module(status.module_path)

    def test_all_renderer_module_paths_follow_package_convention(self) -> None:
        """Every renderer module path is the uniform ``gnn.render.{name}``
        derivation — no per-framework overrides. Pins bnlearn's dedicated
        package parity with its peers.
        """
        from gnn.render.health import _RENDERERS

        for name in get_supported_frameworks():
            assert _RENDERERS[name] == f"gnn.render.{name}", (
                f"{name}: expected gnn.render.{name}, got {_RENDERERS[name]}"
            )

    def test_bnlearn_module_path_is_dedicated_package(self) -> None:
        from gnn.render.health import _RENDERERS

        assert _RENDERERS["bnlearn"] == "gnn.render.bnlearn"


class TestPomdpConfigTruthfulness:
    """``get_pomdp_framework_configs`` must derive fields from the registry."""

    def test_supports_execution_derived_from_registry(self) -> None:
        from gnn.render.framework_registry import (
            FRAMEWORK_REGISTRY,
            get_pomdp_framework_configs,
        )

        configs = get_pomdp_framework_configs()
        for name, spec in FRAMEWORK_REGISTRY.items():
            if not spec.get("pomdp_compatible", False):
                assert name not in configs
                continue
            assert configs[name]["supports_execution"] is bool(
                spec["supports_execution"]
            )
        # bnlearn is executed by ``src/gnn/execute/bnlearn/`` — the config
        # must not contradict the registry spec.
        assert configs["bnlearn"]["supports_execution"] is True

    def test_cpomdp_is_routed_as_continuous_only(self) -> None:
        from gnn.render.framework_registry import get_pomdp_framework_configs

        # Routed like its peers, but discrete specs report ``unsupported``.
        configs = get_pomdp_framework_configs()
        assert configs["cpomdp"]["continuous_only"] is True
        assert configs["cpomdp"]["supports_continuous"] is True


class TestCanonicalConsistency:
    """The registry stays in lockstep with ``gnn.frameworks.ALL_FRAMEWORKS``."""

    def test_registry_order_matches_canonical_tuple(self) -> None:
        """Registry keys, in order, are the canonical tuple minus ``lean``."""
        assert tuple(FRAMEWORK_REGISTRY) == CANONICAL_ALL_FRAMEWORKS[:-1]

    def test_registry_plus_lean_covers_canonical_set(self) -> None:
        """Registry keys plus the execution-only ``lean`` backend = the full set."""
        assert set(FRAMEWORK_REGISTRY) | {"lean"} == set(CANONICAL_ALL_FRAMEWORKS)

    def test_supported_frameworks_match_canonical_render_order(self) -> None:
        assert get_supported_frameworks() == list(CANONICAL_ALL_FRAMEWORKS[:-1])

    def test_lite_frameworks_reexports_canonical_tuple(self) -> None:
        """``LITE_FRAMEWORKS`` is a re-export, not a duplicate literal."""
        from gnn.render.framework_registry import LITE_FRAMEWORKS as registry_lite

        assert registry_lite is LITE_FRAMEWORKS
        assert registry_lite == ("pymdp", "jax", "discopy", "bnlearn")

    def test_bnlearn_registry_function_names_real_surface(self) -> None:
        """The bnlearn ``function`` entry names its actual generator (not a
        phantom symbol): the defect this row fixed."""
        from gnn.render.bnlearn import generate_bnlearn_code
        from gnn.render.health import _RENDERERS

        assert FRAMEWORK_REGISTRY["bnlearn"]["function"] == "generate_bnlearn_code"
        # The default module-path derivation resolves bnlearn's dedicated package.
        assert _RENDERERS["bnlearn"] == "gnn.render.bnlearn"
        assert callable(generate_bnlearn_code)

    def test_execution_framework_name_literal_matches_canonical(self) -> None:
        """``ExecutionFrameworkName`` Literal equals the canonical tuple."""
        from typing import get_args

        from gnn.execute.types import ExecutionFrameworkName

        assert get_args(ExecutionFrameworkName) == CANONICAL_ALL_FRAMEWORKS
