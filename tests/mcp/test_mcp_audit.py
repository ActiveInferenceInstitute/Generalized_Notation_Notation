"""
test_mcp_audit.py — Comprehensive MCP Tool Audit Tests

Validates that every registered MCP tool is:
  1. Detectable via discover_modules / initialize()
  2. Real (callable, named function — not a lambda or None)
  3. Documented (non-empty description string)
  4. Has a named backing function (not anonymous)
  5. Exposes non-empty module/category metadata and valid JSON schemas
  6. Callable live via execute_tool with no required arguments (spot-check)
  7. Registered from a module that logs its registration (static check)

All sub-checks are reported as distinct pytest assertions so individual
failures pinpoint the exact problem.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

pytestmark = pytest.mark.mcp


# Single source of truth for the minimum registered MCP tool count.
# Consumers: .github/workflows/ci.yml, .github/workflows/mcp-audit.yml,
# and the justfile mcp-audit gate (tests.mcp.test_mcp_audit.MCP_TOOL_FLOOR).
MCP_TOOL_FLOOR = 140


def count_mcp_tools() -> int:
    """Return registered MCP tool count after initialize and recovery settle.

    Used by GitHub Actions (``.github/workflows/ci.yml``) to guard against
    accidental tool registration regressions. Mirrors the wait loop in
    ``mcp_initialized`` fixture.
    """
    from gnn.mcp import initialize, mcp_instance

    initialize(halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True)
    prev_count = -1
    for _ in range(25):
        current = len(mcp_instance.tools)
        if current == prev_count:
            break
        prev_count = current
        time.sleep(0.2)
    return len(mcp_instance.tools)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mcp_initialized() -> Any:
    """Return a fully initialized MCP instance with all recovery registrations done.

    After ``initialize()`` returns, MCP continues registering timed-out modules
    via background threads.  We wait up to 5 seconds for the tool count to
    stabilise, then return the instance.
    """
    from gnn.mcp import initialize, mcp_instance

    initialize(halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True)

    # Wait for recovery registration threads to complete.
    # Poll until tool count stops growing (max 5 s).
    prev_count = -1
    for _ in range(25):  # 25 × 0.2 s = 5 s max
        current = len(mcp_instance.tools)
        if current == prev_count:
            break
        prev_count = current
        time.sleep(0.2)

    return mcp_instance


@pytest.fixture(scope="module")
def all_tools(mcp_initialized: Any) -> Dict[str, Any]:
    """Return the full tools dictionary from the initialized MCP instance."""
    return dict(mcp_initialized.tools)  # copy snapshot after recovery wait


@pytest.fixture(scope="module")
def all_modules(mcp_initialized: Any) -> Dict[str, Any]:
    """Return the full modules dictionary from the initialized MCP instance."""
    return cast("dict[str, Any]", mcp_initialized.modules)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level checks
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPModuleDiscovery:
    """Verify that MCP discover_modules loads all expected pipeline modules."""

    EXPECTED_MODULES = (
        "advanced_visualization",
        "analysis",
        "api",
        "audio",
        "cli",
        "doc",
        "execute",
        "export",
        "extract",
        "gnn",
        "gui",
        "integration",
        "intelligent_analysis",
        "llm",
        "lsp",
        "mcp",
        "meta_mcp",
        "ml_integration",
        "model_registry",
        "multimodel",
        "ontology",
        "pipeline",
        "processing",
        "render",
        "report",
        "research",
        "schema_validator",
        "security",
        "setup",
        "sympy_mcp",
        "template",
        "type_checker",
        "utils",
        "validation",
        "visualization",
        "website",
    )
    ZERO_TOOL_MODULES = ("doc", "lsp")

    @pytest.mark.parametrize("mod_name", EXPECTED_MODULES)
    def test_expected_module_loaded(
        self, mod_name: str, all_modules: Dict[str, Any], all_tools: Dict[str, Any]
    ) -> None:
        """Each expected runtime MCP module must be discovered and loaded."""
        assert mod_name in all_modules, (
            f"Module '{mod_name}' not discovered. "
            f"Known modules: {sorted(all_modules.keys())}"
        )
        assert all_modules[mod_name].status == "loaded", (
            f"Module '{mod_name}' status is {all_modules[mod_name].status!r}: "
            f"{all_modules[mod_name].error_message}"
        )

    @pytest.mark.parametrize("mod_name", EXPECTED_MODULES)
    def test_expected_module_has_tools(
        self, mod_name: str, all_modules: Dict[str, Any], all_tools: Dict[str, Any]
    ) -> None:
        """Every functional module except documented no-tool modules contributes tools."""
        info = all_modules.get(mod_name)
        if mod_name in self.ZERO_TOOL_MODULES:
            assert info is not None
            assert info.tools_count == 0
            return
        from_info = (info.tools_count >= 1) if info else False
        from_tools = any(
            mod_name in (getattr(t, "module", "") or "") for t in all_tools.values()
        )
        assert from_info or from_tools, (
            f"Module '{mod_name}' registered 0 tools. Check register_tools() implementation."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tool-level checks
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPToolRealness:
    """Verify every registered tool has a real, named, callable function."""

    def test_at_least_50_tools_registered(self, all_tools: Dict[str, Any]) -> None:
        """MCP should register at least 50 tools across all modules."""
        assert len(all_tools) >= 50, (
            f"Only {len(all_tools)} tools registered. Expected ≥50."
        )

    def test_all_tools_have_callable_funcs(self, all_tools: Dict[str, Any]) -> None:
        """Every tool must have a callable backing function (not None)."""
        not_callable: list[Any] = []
        for name, tool in all_tools.items():
            func = getattr(tool, "func", None) or getattr(tool, "function", None)
            if not callable(func):
                not_callable.append(name)
        assert not_callable == [], (
            f"{len(not_callable)} tools have non-callable func: {not_callable}"
        )

    def test_no_lambda_tools(self, all_tools: Dict[str, Any]) -> None:
        """No tool may be backed by an anonymous lambda."""
        lambdas: list[Any] = []
        for name, tool in all_tools.items():
            func = getattr(tool, "func", None) or getattr(tool, "function", None)
            fn = getattr(func, "__name__", "") if func else ""
            if fn == "<lambda>":
                lambdas.append(name)
        assert lambdas == [], f"Tools backed by lambdas: {lambdas}"

    def test_all_tools_have_named_functions(self, all_tools: Dict[str, Any]) -> None:
        """Every tool's backing function must have a proper __name__ attribute."""
        unnamed: list[Any] = []
        for name, tool in all_tools.items():
            func = getattr(tool, "func", None) or getattr(tool, "function", None)
            fn = getattr(func, "__name__", "") if func else ""
            if not fn or fn in ("<lambda>", ""):
                unnamed.append(name)
        assert unnamed == [], f"Tools with unnamed functions: {unnamed}"

    def test_all_tools_have_descriptions(self, all_tools: Dict[str, Any]) -> None:
        """Every tool must have a non-empty description string."""
        undocumented: list[Any] = []
        for name, tool in all_tools.items():
            desc = (getattr(tool, "description", "") or "").strip()
            if not desc:
                undocumented.append(name)
        assert undocumented == [], (
            f"{len(undocumented)} tools have no description: {undocumented}"
        )

    def test_all_tools_have_module_and_category_metadata(
        self, all_tools: Dict[str, Any]
    ) -> None:
        """Every tool must expose usable module/category metadata to clients."""
        missing_module: list[str] = []
        missing_category: list[str] = []
        for name, tool in all_tools.items():
            if not (getattr(tool, "module", "") or "").strip():
                missing_module.append(name)
            if not (getattr(tool, "category", "") or "").strip():
                missing_category.append(name)
        assert missing_module == [], f"Tools missing module metadata: {missing_module}"
        assert missing_category == [], (
            f"Tools missing category metadata: {missing_category}"
        )

    def test_all_tools_have_valid_json_schemas(self, all_tools: Dict[str, Any]) -> None:
        """Every registered tool schema must satisfy MCPTool's schema contract."""
        invalid: dict[str, list[str]] = {}
        for name, tool in all_tools.items():
            issues = tool.validate_schema()
            if issues:
                invalid[name] = issues
        assert invalid == {}, f"Tools with invalid schemas: {invalid}"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-domain tool presence checks
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPDomainTools:
    """
    Verify that each module registers expected domain-specific tools.
    These are real callable functions.
    """

    DOMAIN_TOOLS = (
        # analysis
        "process_analysis",
        "get_analysis_results",
        "compute_complexity_metrics",
        "list_analysis_tools",
        "evaluate_torchlens_fidelity",
        # render
        "process_render",
        "list_render_frameworks",
        "render_gnn_to_format",
        # export
        "process_export",
        "list_export_formats",
        "validate_export_format",
        # validation
        "process_validation",
        "validate_gnn_file",
        "get_validation_report",
        "check_schema_compliance",
        # ontology
        "process_ontology",
        "validate_ontology_terms",
        "extract_ontology_annotations",
        "list_standard_ontology_terms",
        # visualization
        "process_visualization",
        "get_visualization_options",
        "list_visualization_artifacts",
        # report
        "generate_report",
        "list_report_formats",
        "read_report",
        # integration
        "process_integration",
        "list_supported_integrations",
        "check_integration_dependencies",
        # security
        "process_security",
        "scan_gnn_file",
        "list_security_checks",
        # research
        "process_research",
        "list_research_topics",
        # website
        "process_website",
        "build_website_from_pipeline_output",
        "get_website_status",
        "list_generated_website_pages",
        "get_website_page",
        # sapf
        "process_sapf",
        "list_audio_artifacts",
        # audio (new real tools)
        "process_audio",
        "check_audio_backends",
        "get_audio_generation_options",
        "get_audio_module_info",
        # execute (new real tools)
        "process_execute",
        "execute_gnn_model",
        "check_execute_dependencies",
        "get_execute_module_info",
        # llm (new real tools)
        "process_llm",
        "analyze_gnn_with_llm",
        "generate_llm_documentation",
        "get_llm_providers",
        "get_llm_module_info",
        # advanced_visualization
        "process_advanced_visualization",
        "check_visualization_capabilities",
        "list_d2_visualization_types",
        "get_advanced_visualization_module_info",
        # ml_integration
        "process_ml_integration",
        "check_ml_frameworks",
        "list_ml_integration_targets",
        "get_ml_module_info",
        # intelligent_analysis
        "process_intelligent_analysis",
        "get_analysis_capabilities",
        "get_intelligent_analysis_module_info",
        # gui
        "process_gui",
        "list_available_guis",
        "get_gui_module_info",
        # pipeline
        "get_pipeline_steps",
        "get_pipeline_status",
        "list_step_artifacts",
        # extract
        "extract_pomdp",
        # multimodel
        "generate_dependency_graph",
        # template
        "template.pull",
        # nested GUI/oxdraw tools exposed by the parent GUI MCP module
        "oxdraw.convert_to_mermaid",
        "oxdraw.convert_from_mermaid",
        "oxdraw.check_installation",
        "oxdraw.get_info",
        "oxdraw.launch_editor",
        # GNN gold-standard spot-check
        "validate_gnn_content",
        "parse_gnn_content",
    )

    @pytest.mark.parametrize("tool_name", DOMAIN_TOOLS)
    def test_domain_tool_registered(
        self, tool_name: str, all_tools: Dict[str, Any]
    ) -> None:
        """Each expected domain-specific tool must be registered.

        The fixture now waits for all recovery registrations to stabilise.
        A missing tool indicates a genuine registration failure.
        """
        assert tool_name in all_tools, (
            f"Domain tool '{tool_name}' not found in MCP registry.\n"
            f"Registered tools with same prefix: "
            f"{[t for t in all_tools if t.split('_')[0] == tool_name.split('_')[0]]}"
        )

    @pytest.mark.parametrize("tool_name", DOMAIN_TOOLS)
    def test_domain_tool_is_callable(
        self, tool_name: str, all_tools: Dict[str, Any]
    ) -> None:
        """Each domain tool's backing function must be a named callable (not a lambda)."""
        if tool_name not in all_tools:
            raise AssertionError(
                f"Tool '{tool_name}' not registered — see test_domain_tool_registered"
            )
        tool = all_tools[tool_name]
        func = getattr(tool, "func", None) or getattr(tool, "function", None)
        assert callable(func), f"Tool '{tool_name}' func is not callable: {func!r}"
        fn = getattr(func, "__name__", "")
        assert fn not in ("", "<lambda>"), (
            f"Tool '{tool_name}' is backed by an anonymous lambda — replace with a named function"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Live spot-check: zero-arg tools via execute_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPToolExecution:
    """
    Live-execute zero-argument tools and verify they return real results.
    Uses mcp_instance.execute_tool() — the same path clients use.
    """

    ZERO_ARG_TOOLS = (
        "list_analysis_tools",
        "list_render_frameworks",
        "list_export_formats",
        "list_standard_ontology_terms",
        "list_supported_integrations",
        "list_research_topics",
        "list_security_checks",
        "get_website_module_info",
        "get_render_module_info",
        "get_visualization_options",
        "get_report_module_info",
        "get_sapf_module_info",
        "list_step_artifacts",
    )

    @pytest.mark.parametrize("tool_name", ZERO_ARG_TOOLS)
    def test_zero_arg_tool_executes(self, tool_name: str, mcp_initialized: Any) -> None:
        """Each zero-arg tool must execute without exception and return a dict."""
        if tool_name not in mcp_initialized.tools:
            raise AssertionError(f"Tool '{tool_name}' not registered")
        result = mcp_initialized.execute_tool(tool_name, {})
        assert isinstance(result, dict), (
            f"Tool '{tool_name}' returned {type(result)}, expected dict"
        )
        assert result.get("success", False), (
            f"Tool '{tool_name}' returned success=False: {result}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Static: logging coverage in register_tools
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPLoggingCoverage:
    """Statically verify that every mcp.py register_tools() calls logger.info."""

    def _get_mcp_files(self) -> List[Path]:
        src = Path(__file__).parent.parent.parent / "src" / "gnn"
        return [
            f
            for f in src.rglob("mcp.py")
            # Skip core mcp/ directory and nested sub-submodule files (e.g. gui/gui_1/mcp.py)
            if not any(
                part.startswith("_") or part == "__pycache__" for part in f.parts
            )
            and f.parent.name != "mcp"  # skip core mcp/ directory
            and f.parent.parent.name == src.name  # only direct children of src/gnn/
        ]

    def test_all_mcp_files_have_register_tools(self) -> None:
        """Every submodule mcp.py must define register_tools()."""
        files = self._get_mcp_files()
        assert len(files) >= 20, f"Expected ≥20 mcp.py files, found {len(files)}"
        missing = [f for f in files if "def register_tools" not in f.read_text()]
        assert missing == [], (
            f"mcp.py files missing register_tools: {[f.name for f in missing]}"
        )

    def test_all_register_tools_have_logger_info(self) -> None:
        """Every register_tools() implementation must call logger.info."""
        files = self._get_mcp_files()
        no_log: list[Any] = []
        for mcp_file in files:
            src = mcp_file.read_text(encoding="utf-8", errors="replace")
            # Find register_tools body and check for logger.info
            in_register = False
            has_log = False
            for line in src.splitlines():
                if "def register_tools" in line:
                    in_register = True
                if in_register and ("logger.info" in line or "logging.info" in line):
                    has_log = True
                    break
            if not has_log:
                no_log.append(str(mcp_file.relative_to(mcp_file.parent.parent.parent)))
        assert no_log == [], f"register_tools() missing logger.info in: {no_log}"

    def test_all_mcp_files_have_module_logging(self) -> None:
        """Every mcp.py must define a module-level logger."""
        files = self._get_mcp_files()
        no_logger: list[Any] = []
        for mcp_file in files:
            src = mcp_file.read_text(encoding="utf-8", errors="replace")
            if (
                "logger = logging.getLogger" not in src
                and "logger=logging.getLogger" not in src
            ):
                no_logger.append(mcp_file.name)
        assert no_logger == [], f"mcp.py files missing module-level logger: {no_logger}"


# ─────────────────────────────────────────────────────────────────────────────
#  Audit report generation
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPAuditReport:
    """Generate and validate the MCP audit report artifact."""

    def test_generate_audit_report(
        self, all_tools: Dict[str, Any], all_modules: Dict[str, Any], tmp_path: Any
    ) -> None:
        """Generate a full audit JSON report and verify its structure."""
        loaded = [n for n, i in all_modules.items() if i.status == "loaded"]
        errored = [n for n, i in all_modules.items() if i.status == "error"]

        tools_list: list[Any] = []
        for name, tool in sorted(all_tools.items()):
            func = getattr(tool, "func", None) or getattr(tool, "function", None)
            fn = getattr(func, "__name__", "?") if func else "NONE"
            desc = (getattr(tool, "description", "") or "").strip()
            tools_list.append(
                {
                    "name": name,
                    "module": getattr(tool, "module", ""),
                    "category": getattr(tool, "category", ""),
                    "fn": fn,
                    "is_real": callable(func) and fn not in ("", "<lambda>"),
                    "documented": bool(desc),
                    "description": desc[:120],
                }
            )

        report: dict[str, Any] = {
            "generated_at": "2026-02-24T06:51:00",
            "modules_total": len(all_modules),
            "modules_loaded": len(loaded),
            "modules_errored": len(errored),
            "modules_loaded_names": sorted(loaded),
            "modules_errored_names": sorted(errored),
            "tools_total": len(all_tools),
            "tools_real": sum(1 for t in tools_list if t["is_real"]),
            "tools_documented": sum(1 for t in tools_list if t["documented"]),
            "tools": tools_list,
        }

        out = tmp_path / "mcp_audit_report.json"
        out.write_text(json.dumps(report, indent=2))
        assert out.exists()

        # Assertions on the report
        assert report["tools_total"] >= 50, (
            f"Expected ≥50 tools total, got {report['tools_total']}"
        )
        # Module-discovery count is advisory (async recovery timing).
        # The authoritative check is domain-tool presence (TestMCPDomainTools).
        assert report["tools_real"] == report["tools_total"], (
            f"{report['tools_total'] - report['tools_real']} tools are not real"
        )
        assert report["tools_documented"] == report["tools_total"], (
            f"{report['tools_total'] - report['tools_documented']} tools are undocumented"
        )
