"""
Test Audio Overall Tests

This file contains comprehensive tests for the audio module functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAudioModuleComprehensive:
    """Comprehensive tests for the audio module."""

    @pytest.mark.unit
    def test_audio_module_imports(self) -> Any:
        """Test that audio module can be imported."""
        import gnn.audio as audio

        assert hasattr(audio, "__version__")
        assert hasattr(audio, "AudioGenerator")
        assert hasattr(audio, "SAPFGNNProcessor")

    @pytest.mark.unit
    def test_audio_generator_instantiation(self) -> Any:
        """Test AudioGenerator class instantiation."""
        from gnn.audio import AudioGenerator

        generator = AudioGenerator()
        assert generator is not None
        assert hasattr(generator, "generate_audio")
        assert hasattr(generator, "process_gnn_to_audio")

    @pytest.mark.unit
    def test_sapf_gnn_processor_instantiation(self) -> Any:
        """Test SAPFGNNProcessor class instantiation."""
        from gnn.audio import SAPFGNNProcessor

        processor = SAPFGNNProcessor()
        assert processor is not None
        assert hasattr(processor, "convert_gnn_to_sapf")
        assert hasattr(processor, "generate_audio")

    @pytest.mark.unit
    def test_audio_module_info(self) -> Any:
        """Test audio module information retrieval."""
        from gnn.audio import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info

    @pytest.mark.unit
    def test_audio_generation_options(self) -> Any:
        """Test audio generation options retrieval."""
        from gnn.audio import get_audio_generation_options

        options = get_audio_generation_options()
        assert isinstance(options, dict)
        assert "formats" in options
        assert "effects" in options
        assert "backends" in options


class TestAudioProcessing:
    """Tests for audio processing functionality."""

    @pytest.mark.unit
    def test_gnn_to_audio_conversion(self, sample_gnn_files: Any) -> Any:
        """Test GNN to audio conversion."""
        from gnn.audio import AudioGenerator

        generator = AudioGenerator()
        gnn_content = "test GNN content"
        result = generator.process_gnn_to_audio(gnn_content)
        assert result is not None

    @pytest.mark.unit
    def test_audio_validation(self) -> Any:
        """Test audio validation functionality."""
        from gnn.audio import validate_audio_content

        result = validate_audio_content("test audio content")
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_audio_generation(self, safe_filesystem: Any) -> Any:
        """Test audio generation functionality."""
        from gnn.audio import generate_audio_from_gnn

        output_dir = safe_filesystem.create_dir("audio_output")
        result = generate_audio_from_gnn(
            "test GNN content\nwith variables", output_dir=output_dir
        )
        assert result is not None
        assert "audio_files" in result


class TestAudioIntegration:
    """Integration tests for audio module."""

    @pytest.mark.integration
    def test_audio_pipeline_integration(
        self, sample_gnn_files: Any, isolated_temp_dir: Any
    ) -> Any:
        """Test audio module integration with pipeline."""
        from gnn.audio import AudioGenerator

        generator = AudioGenerator()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file) as f:
            gnn_content = f.read()
        result = generator.process_gnn_to_audio(gnn_content)
        assert result is not None

    @pytest.mark.integration
    def test_audio_mcp_integration(self) -> Any:
        """Test audio MCP integration."""
        from gnn.audio.mcp import register_tools

        assert callable(register_tools)


def test_audio_module_completeness() -> Any:
    """Test that audio module has all required components."""
    required_components: list[Any] = [
        "AudioGenerator",
        "SAPFGNNProcessor",
        "get_module_info",
        "get_audio_generation_options",
    ]
    try:
        import gnn.audio as audio

        for component in required_components:
            assert hasattr(audio, component), f"Missing component: {component}"
    except ImportError:
        raise AssertionError("Audio module not available")


@pytest.mark.slow
def test_audio_module_performance() -> Any:
    """Test audio module performance characteristics."""
    import time

    from gnn.audio import AudioGenerator

    generator = AudioGenerator()
    start_time = time.time()
    generator.process_gnn_to_audio("test content")
    processing_time = time.time() - start_time
    assert processing_time < 10.0
