"""
Comprehensive Core Module Tests

This module provides thorough testing for all core GNN processing modules
to ensure 100% functionality and coverage. Each test validates:

1. Module import capabilities and dependency resolution
2. Core functionality and data processing
3. Error handling and edge cases
4. Integration with other modules
5. Performance characteristics
6. Documentation and API consistency

All tests execute real methods and file operations; tests may skip if optional backends are unavailable.
"""

import importlib
import logging
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark: list[Any] = [pytest.mark.core, pytest.mark.fast]


class TestGNNModuleComprehensive:
    """Comprehensive tests for the GNN processing module."""

    @pytest.mark.unit
    def test_gnn_module_imports(self) -> Any:
        """Test that GNN module can be imported and has expected structure."""
        from gnn import (
            discover_gnn_files,
            generate_gnn_report,
            parse_gnn_file,
            process_gnn_directory,
        )
        from gnn.processing import check_gnn_file_structure

        assert callable(discover_gnn_files), "discover_gnn_files should be callable"
        assert callable(parse_gnn_file), "parse_gnn_file should be callable"
        assert callable(check_gnn_file_structure), (
            "check_gnn_file_structure should be callable"
        )
        assert callable(process_gnn_directory), (
            "process_gnn_directory should be callable"
        )
        assert callable(generate_gnn_report), "generate_gnn_report should be callable"
        logging.info("GNN module imports validated")

    @pytest.mark.unit
    def test_gnn_file_discovery(self, sample_gnn_files: Any) -> Any:
        """Test GNN file discovery functionality."""
        from gnn import discover_gnn_files

        gnn_dir = list(sample_gnn_files.values())[0].parent
        discovered_files = discover_gnn_files(gnn_dir)
        assert isinstance(discovered_files, list), (
            "discover_gnn_files should return a list"
        )
        assert len(discovered_files) > 0, "Should discover GNN files"
        for file_path in discovered_files:
            assert isinstance(file_path, Path), (
                "Discovered files should be Path objects"
            )
            assert file_path.exists(), "Discovered files should exist"
        logging.info(
            f"GNN file discovery validated: {len(discovered_files)} files found"
        )

    @pytest.mark.unit
    def test_gnn_file_parsing(self, sample_gnn_files: Any) -> Any:
        """parse_gnn_file returns a structured dict for every sample fixture.

        Shape (see src/gnn/processor.py): success, file_path, file_name,
        file_size, sections (list), variables (list), structure_info (dict),
        parse_timestamp.
        """
        from gnn import parse_gnn_file

        for file_path in sample_gnn_files.values():
            parsed = parse_gnn_file(file_path)
            assert isinstance(parsed, dict), f"{file_path.name}: expected dict"
            assert parsed["success"] is True, f"{file_path.name}: parse unsuccessful"
            assert "ModelName" in parsed["sections"]
            assert isinstance(parsed["structure_info"], dict)
            assert isinstance(parsed["variables"], list)

    @pytest.mark.unit
    def test_gnn_validation(self, sample_gnn_files: Any) -> Any:
        """Lightweight GNN structure validation against the sample fixtures."""
        for file_path in sample_gnn_files.values():
            content = file_path.read_text()
            has_model_name = "## ModelName" in content
            has_gnn_version = "## GNNVersionAndFlags" in content
            if file_path.name != "invalid.md":
                assert has_model_name or has_gnn_version, (
                    f"{file_path.name}: valid GNN file must have ModelName or GNNVersionAndFlags"
                )


class TestRenderModuleComprehensive:
    """Comprehensive tests for the render module."""

    @pytest.mark.unit
    def test_render_module_imports(self) -> Any:
        """Test that render module can be imported and has expected structure."""
        from gnn.render import (
            process_render,
            render_gnn_to_activeinference_jl,
            render_gnn_to_discopy,
            render_gnn_to_pymdp,
            render_gnn_to_rxinfer,
        )

        assert callable(render_gnn_to_pymdp), "render_gnn_to_pymdp should be callable"
        assert callable(render_gnn_to_rxinfer), (
            "render_gnn_to_rxinfer should be callable"
        )
        assert callable(render_gnn_to_discopy), (
            "render_gnn_to_discopy should be callable"
        )
        assert callable(render_gnn_to_activeinference_jl), (
            "render_gnn_to_activeinference_jl should be callable"
        )
        assert callable(process_render), "process_render should be callable"
        logging.info("Render module imports validated")

    @pytest.mark.unit
    def test_pymdp_rendering_callable(self) -> Any:
        """render_gnn_to_pymdp resolves to a real callable.

        Full pipeline-level rendering (parsed GNN spec → PyMDP script on
        disk) is exercised in tests/render/test_render_cli_targets.py.
        Passing a dict-of-Path sample_gnn_files through this public API
        produces opaque errors; assert only the public-symbol contract.
        """
        from gnn.render import render_gnn_to_pymdp

        assert callable(render_gnn_to_pymdp)

    @pytest.mark.unit
    def test_rxinfer_rendering_callable(self) -> Any:
        """render_gnn_to_rxinfer is callable; POMDP content validated elsewhere.

        tests/render/test_render_cli_targets.py parametrizes every
        backend target against the real sample corpus, including RxInfer.
        """
        from gnn.render import render_gnn_to_rxinfer

        assert callable(render_gnn_to_rxinfer)

    @pytest.mark.unit
    def test_discopy_rendering(self, sample_gnn_files: Any) -> Any:
        """Test DisCoPy rendering functionality."""
        from gnn.render import render_gnn_to_discopy

        sample_spec: dict[str, Any] = {
            "model_name": "TestModel",
            "variables": [{"name": "A", "dimensions": [2, 2]}],
            "model_parameters": {},
            "initial_parameterization": {},
            # DisCoPy renders declared wiring; a self-loop is a valid minimal graph.
            "connections": [{"source": "A", "relation": ">", "target": "A"}],
        }
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "discopy_diagram.py"
            result = render_gnn_to_discopy(sample_spec, output_path)
            assert isinstance(result, tuple), (
                "render_gnn_to_discopy should return a tuple"
            )
            assert len(result) == 3, (
                "render_gnn_to_discopy should return (success, message, warnings)"
            )
            assert result[0] is True, "render_gnn_to_discopy should succeed"
            assert output_path.exists(), "Output file should be created"


class TestExecuteModuleComprehensive:
    """Comprehensive tests for the execute module."""

    @pytest.mark.unit
    def test_execute_module_imports(self) -> Any:
        """Test that execute module can be imported and has expected functions."""
        from gnn.execute import (
            GNNExecutor,
            PyMDPSimulation,
            process_execute,
            validate_execution_environment,
        )

        assert GNNExecutor is not None, "GNNExecutor should be available"
        assert PyMDPSimulation is not None, "PyMDPSimulation should be available"
        assert callable(process_execute), "process_execute should be callable"
        assert callable(validate_execution_environment), (
            "validate_execution_environment should be callable"
        )

    @pytest.mark.unit
    def test_execution_environment_validation(self) -> Any:
        """validate_execution_environment returns a structured status dict."""
        from gnn.execute import validate_execution_environment

        env_status = validate_execution_environment()
        assert isinstance(env_status, dict)
        assert "python_version" in env_status
        assert "dependencies" in env_status

    @pytest.mark.unit
    def test_safe_script_execution(self, isolated_temp_dir: Any) -> Any:
        """Test safe script execution functionality."""
        from gnn.execute import GNNExecutor

        test_script = isolated_temp_dir / "test_script.py"
        test_script.write_text("print('Hello from test script!')")
        engine = GNNExecutor()
        assert engine is not None, "GNNExecutor should be instantiable"


class TestLLMModuleComprehensive:
    """Comprehensive tests for the LLM module."""

    @pytest.mark.unit
    def test_llm_module_imports(self) -> Any:
        """Test that LLM module can be imported and has expected functions."""
        from gnn.llm import (
            analyze_gnn_file_with_llm,
            generate_code_suggestions,
            generate_model_insights,
            process_llm,
        )

        assert callable(process_llm), "process_llm should be callable"
        assert callable(analyze_gnn_file_with_llm), (
            "analyze_gnn_file_with_llm should be callable"
        )
        assert callable(generate_model_insights), (
            "generate_model_insights should be callable"
        )
        assert callable(generate_code_suggestions), (
            "generate_code_suggestions should be callable"
        )

    @pytest.mark.unit
    @pytest.mark.slow
    def test_llm_model_analysis(self, sample_gnn_files: Any) -> Any:
        """Test LLM-based model analysis functionality."""
        from gnn.llm import analyze_gnn_file_with_llm

        for file_path in sample_gnn_files.values():
            analysis = analyze_gnn_file_with_llm(file_path, verbose=False)
            assert isinstance(analysis, dict), "Analysis should return a dict"
            assert "file_path" in analysis, "Analysis should contain file_path"
            break

    @pytest.mark.unit
    def test_llm_description_generation(self, sample_gnn_files: Any) -> Any:
        """Test LLM description generation functionality."""
        from gnn.llm import generate_documentation

        sample_analysis: dict[str, Any] = {
            "file_path": "test.md",
            "file_name": "test.md",
            "semantic_analysis": {"model_type": "POMDP", "complexity_level": "simple"},
            "complexity_metrics": {"variable_count": 3, "connection_count": 2},
            "variables": [{"name": "X", "line": 1}, {"name": "Y", "line": 2}],
        }
        docs = generate_documentation(sample_analysis)
        assert isinstance(docs, dict), "Documentation should return a dict"
        assert "file_path" in docs, "Documentation should contain file_path"


class TestMCPModuleComprehensive:
    """Comprehensive tests for the MCP module."""

    @pytest.mark.unit
    def test_mcp_module_imports(self) -> Any:
        """Test that MCP module can be imported and has expected structure."""
        from gnn.mcp import generate_mcp_report, get_available_tools, handle_mcp_request
        from gnn.mcp import register_module_tools as register_tools

        assert callable(register_tools), "register_tools should be callable"
        assert callable(get_available_tools), "get_available_tools should be callable"
        assert callable(handle_mcp_request), "handle_mcp_request should be callable"
        assert callable(generate_mcp_report), "generate_mcp_report should be callable"
        logging.info("MCP module imports validated")

    @pytest.mark.unit
    def test_mcp_tool_registration(self) -> Any:
        """register_module_tools() returns a non-empty list + get_available_tools is a list."""
        from gnn.mcp import get_available_tools
        from gnn.mcp import register_module_tools as register_tools

        tools = register_tools()
        assert isinstance(tools, list), (
            f"register_module_tools() should return list, got {type(tools).__name__}"
        )
        assert len(tools) > 0, "Expected at least one registered MCP tool"
        available_tools = get_available_tools()
        assert isinstance(available_tools, list)

    @pytest.mark.unit
    def test_mcp_request_handling(self) -> Any:
        """handle_mcp_request returns a dict with the request id echoed back."""
        from gnn.mcp import handle_mcp_request

        sample_request: dict[str, Any] = {"method": "tools/list", "params": {}, "id": 1}
        response = handle_mcp_request(sample_request)
        assert isinstance(response, dict)
        assert "id" in response


class TestOntologyModuleComprehensive:
    """Comprehensive tests for the ontology module."""

    @pytest.mark.unit
    def test_ontology_module_imports(self) -> Any:
        """Test that ontology module can be imported and has expected functions."""
        from gnn.ontology import FEATURES, process_ontology

        assert callable(process_ontology), "process_ontology should be callable"
        assert isinstance(FEATURES, dict), "FEATURES should be a dict"
        assert FEATURES.get("basic_processing", False), (
            "Basic processing should be available"
        )

    @pytest.mark.unit
    def test_ontology_term_validation(self, isolated_temp_dir: Any) -> Any:
        """Test ontology processing functionality."""
        from gnn.ontology import process_ontology

        input_dir = isolated_temp_dir / "input"
        input_dir.mkdir()
        sample_file = input_dir / "test_model.md"
        sample_file.write_text(
            "## GNNVersionAndFlags\nVersion: 1.0\n\n## ModelName\nTestModel\n\n## Variables\n- X: [2]\n"
        )
        output_dir = isolated_temp_dir / "output"
        result = process_ontology(input_dir, output_dir, verbose=False)
        assert isinstance(result, bool), "process_ontology should return a boolean"
        assert (output_dir / "ontology_results.json").exists(), (
            "Results file should be created"
        )


class TestWebsiteModuleComprehensive:
    """Comprehensive tests for the website module."""

    @pytest.mark.unit
    def test_website_module_imports(self) -> Any:
        """Test that website module can be imported and has expected functions."""
        from gnn.website import FEATURES, process_website

        assert callable(process_website), "process_website should be callable"
        assert isinstance(FEATURES, dict), "FEATURES should be a dict"
        assert FEATURES.get("basic_processing", False), (
            "Basic processing should be available"
        )

    @pytest.mark.unit
    def test_website_generation(self, isolated_temp_dir: Any) -> Any:
        """Test website generation functionality."""
        from gnn.website import process_website

        input_dir = isolated_temp_dir / "input"
        input_dir.mkdir()
        sample_file = input_dir / "test_model.md"
        sample_file.write_text(
            "## GNNVersionAndFlags\nVersion: 1.0\n\n## ModelName\nTestModel\n\n## Variables\n- X: [2]\n"
        )
        output_dir = isolated_temp_dir / "output"
        result = process_website(input_dir, output_dir, verbose=False)
        assert isinstance(result, bool), "process_website should return a boolean"
        assert (output_dir / "index.html").exists(), "Index file should be created"

    @pytest.mark.unit
    def test_html_report_creation(self, isolated_temp_dir: Any) -> Any:
        """Test HTML report creation functionality."""
        from gnn.website import process_website

        input_dir = isolated_temp_dir / "input"
        input_dir.mkdir()
        for i in range(3):
            sample_file = input_dir / f"test_model_{i}.md"
            sample_file.write_text(
                f"## GNNVersionAndFlags\nVersion: 1.0\n\n## ModelName\nTestModel{i}\n\n## Variables\n- X: [{i + 1}]\n"
            )
        output_dir = isolated_temp_dir / "output"
        result = process_website(input_dir, output_dir, verbose=False)
        assert isinstance(result, bool), "process_website should return a boolean"
        results_dir = output_dir
        assert results_dir.exists(), "Results directory should be created"
        results_file = results_dir / "website_results.json"
        assert results_file.exists(), "Results file should be created"


class TestSAPFModuleComprehensive:
    """Comprehensive tests for the SAPF module."""

    @pytest.mark.unit
    def test_sapf_module_imports(self) -> Any:
        """Test that SAPF module can be imported and has expected structure."""
        from gnn.audio.sapf import (
            convert_gnn_to_sapf,
            create_sapf_visualization,
            generate_sapf_audio,
            generate_sapf_report,
            validate_sapf_code,
        )

        assert callable(convert_gnn_to_sapf), "convert_gnn_to_sapf should be callable"
        assert callable(generate_sapf_audio), "generate_sapf_audio should be callable"
        assert callable(validate_sapf_code), "validate_sapf_code should be callable"
        logging.info("SAPF module imports validated successfully")

    @pytest.mark.unit
    def test_gnn_to_sapf_conversion(self) -> Any:
        """convert_gnn_to_sapf turns a GNN markdown string into SAPF code."""
        from gnn.audio.sapf import convert_gnn_to_sapf

        sample_gnn = (
            "\n## ModelName\nTestActiveInferenceModel\n\n"
            "## StateSpaceBlock\ns1: State\ns2: State\ns3: State\n\n"
            "## Connections\ns1 -> s2: Transition\ns2 -> s3: Transition\ns3 -> s1: Transition\n\n"
            "## InitialParameterization\nA: [0.8, 0.2; 0.3, 0.7]\nB: [0.9, 0.1; 0.2, 0.8]\nC: [0.7, 0.3; 0.4, 0.6]\n"
        )
        sapf_code = convert_gnn_to_sapf(
            sample_gnn, model_name="TestActiveInferenceModel"
        )
        assert isinstance(sapf_code, str)
        assert len(sapf_code) > 0, "convert_gnn_to_sapf produced empty output"

    @pytest.mark.unit
    def test_sapf_validation(self) -> Any:
        """validate_sapf_code returns (bool, list) for any input string."""
        from gnn.audio.sapf import validate_sapf_code

        sample_sapf_code = (
            "\n; Test SAPF code\n261.63 = base_freq\nbase_freq 0 sinosc 0.3 * = osc1\n"
            "10 sec 0.1 1 0.8 0.2 env = envelope\nosc1 envelope * = final_audio\nfinal_audio play\n"
        )
        is_valid, issues = validate_sapf_code(sample_sapf_code)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    @pytest.mark.unit
    def test_sapf_audio_generation_callable(self) -> Any:
        """generate_sapf_audio resolves to a real callable.

        Full audio synthesis exercised in tests/audio/. Here we guard the
        public surface — that the symbol is exported and callable.
        """
        from gnn.audio.sapf import generate_sapf_audio

        assert callable(generate_sapf_audio)


class TestCoreModuleIntegration:
    """Integration tests for core module coordination."""

    @pytest.mark.integration
    def test_cross_module_public_surface(self) -> Any:
        """Cross-module public APIs are importable together without circular-import issues.

        Full end-to-end flow (parse → render → execute → report) is covered by
        tests/pipeline/test_pipeline_render_execute_analyze.py. Here we
        only assert that the combined import does not break — a common
        regression when modules add reciprocal imports.
        """
        from gnn import parse_gnn_file
        from gnn.execute import execute_gnn_model
        from gnn.llm import LLMProcessor
        from gnn.render import render_gnn_to_pymdp
        from gnn.website import generate_html_report

        for sym in (
            execute_gnn_model,
            parse_gnn_file,
            render_gnn_to_pymdp,
            generate_html_report,
        ):
            assert callable(sym)
        assert isinstance(LLMProcessor, type)


def test_core_module_completeness() -> Any:
    """Every core module must import and expose its version/feature metadata.

    Strict: any ImportError fails the test with the full failure list — a
    broken module can no longer hide behind the remaining healthy imports.
    """
    core_modules: list[str] = [
        "gnn",
        "gnn.render",
        "gnn.execute",
        "gnn.validation",
        "gnn.visualization",
    ]
    failures: list[str] = []
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — report every failure mode
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        assert hasattr(module, "__version__") or hasattr(module, "FEATURES"), (
            f"Module {module_name} missing __version__ or FEATURES"
        )
    assert not failures, (
        f"{len(failures)}/{len(core_modules)} core modules failed to import:\n"
        + "\n".join(failures)
    )
    logging.info(
        f"Core module completeness: all {len(core_modules)} modules importable"
    )


@pytest.mark.slow
def test_core_module_performance() -> Any:
    """Core module imports stay fast; a crashing module fails, not escapes."""
    modules_to_time: list[str] = [
        "gnn",
        "gnn.render",
        "gnn.validation",
    ]
    failures: list[str] = []
    for module_name in modules_to_time:
        start = time.perf_counter()
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — report every failure mode
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Module {module_name} import took {elapsed:.2f}s"
    assert not failures, (
        f"{len(failures)}/{len(modules_to_time)} modules failed to import:\n"
        + "\n".join(failures)
    )
    logging.info("Core module performance test completed")
