#!/usr/bin/env python3
"""
Comprehensive Render Module Tests

This module tests the render module's ability to generate simulation code for various
target frameworks. All tests use real rendering functions with actual GNN specifications
and verify that output artifacts are correctly created.

Test Coverage:
- PyMDP code generation (test_render_to_pymdp, test_render_to_rxinfer_toml)
- DisCoPy categorical diagram rendering (test_render_to_discopy, test_render_to_discopy_combined)
- Multiple language/framework rendering (test_render_to_activeinference_jl, test_render_to_jax, test_render_to_jax_pomdp)

All tests validate real render function execution.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Real-implementation render target tests
class TestRenderTargets:
    """
    Test rendering to different target frameworks.

    These tests verify that the GNN render module can successfully generate executable code
    for various target frameworks including PyMDP, RxInfer, DisCoPy, ActiveInference.jl,
    and JAX. Each test uses real render function calls with actual GNN specifications
    and verifies output artifacts are created correctly.

    Fixtures:
    - tmp_path: Temporary directory for artifact output
    - sample_gnn_spec: Basic GNN specification dict
    - test_render_module: Real RealRenderModule instance (not a simulated)
    """

    @pytest.mark.unit
    def test_render_to_pymdp(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to PyMDP format."""
        # Call real render function with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "pymdp", tmp_path
        )

        # Verify successful rendering
        assert ok is True, "PyMDP rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"

        # Verify artifacts are created
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"

    @pytest.mark.unit
    def test_render_to_rxinfer_toml(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test that the retired rxinfer_toml target is rejected."""
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "rxinfer_toml", tmp_path
        )

        # The retired TOML target must return False with a helpful message.
        assert ok is False, "rxinfer_toml target should be rejected"
        assert isinstance(msg, str)
        assert "no longer supported" in msg.lower(), msg
        assert artifacts == [], "Retired target should produce no artifacts"

    @pytest.mark.unit
    def test_render_to_discopy(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to DisCoPy format."""
        # Call real render function with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "discopy", tmp_path
        )

        # Verify successful rendering
        assert ok is True, "DisCoPy rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"

        # Verify artifacts are created
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"

    @pytest.mark.unit
    def test_render_to_discopy_combined(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to DisCoPy combined format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "discopy_combined", tmp_path
        )

        assert ok is True, "DisCoPy combined rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"

        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"

    @pytest.mark.unit
    def test_render_to_activeinference_jl(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to ActiveInference.jl format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "activeinference_jl", tmp_path
        )

        assert ok is True, "ActiveInference.jl rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"

        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"

    @pytest.mark.unit
    def test_render_to_jax(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to JAX format using real rendering with content validation."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "jax", tmp_path
        )

        assert ok is True, "JAX rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"

        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"

        # Content validation: verify generated code has Active Inference constructs
        jax_artifacts = [a for a in artifacts if a.endswith(".py")]
        if jax_artifacts:
            content = (tmp_path / jax_artifacts[0]).read_text()
            assert "import jax" in content or "from jax" in content, (
                "Generated JAX code should import jax"
            )
            assert "create_params" in content or "belief_update" in content, (
                "Generated JAX code should contain Active Inference functions"
            )
            # Verify valid Python syntax
            compile(content, jax_artifacts[0], "exec")

    @pytest.mark.unit
    def test_render_to_jax_pomdp(
        self, tmp_path: Any, sample_gnn_spec: Any, test_render_module: Any
    ) -> Any:
        """Test rendering to JAX POMDP format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(
            sample_gnn_spec, "jax_pomdp", tmp_path
        )

        assert ok is True, "JAX POMDP rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"

        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"


class TestRenderDisCoPyTranslator:
    """Smoke tests for render.discopy.translator sub-module."""

    def _import_translator(self) -> Any:
        try:
            from gnn.render.discopy import translator

            return translator
        except Exception:
            raise AssertionError("gnn.render.discopy.translator not importable")

    def test_module_importable(self) -> Any:
        self._import_translator()

    def test_check_discopy_availability(self) -> Any:
        from gnn.render.discopy import bootstrap

        result = bootstrap.check_discopy_availability()
        assert isinstance(result, dict)

    def test_generate_setup_report(self) -> Any:
        from gnn.render.discopy import bootstrap

        result = bootstrap.generate_setup_report()
        assert isinstance(result, str)

    def test_parse_gnn_content_empty(self) -> Any:
        translator = self._import_translator()
        result = translator.parse_gnn_content("")
        assert isinstance(result, dict)
