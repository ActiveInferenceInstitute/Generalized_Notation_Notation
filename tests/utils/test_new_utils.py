"""
Tests for gnn.utils.logging.logging_utils and gnn.utils.pipeline_orchestration.base_processor

These tests verify the utility modules work correctly.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest


class TestStepLogging:
    """Tests for step_logging module."""

    def test_step_logging_imports(self) -> None:
        """Test that step_logging module imports correctly."""
        from gnn.utils.logging.logging_utils import (
            log_step_error,
            log_step_start,
            log_step_success,
            log_step_warning,
            setup_step_logging,
        )

        assert callable(log_step_start)
        assert callable(log_step_success)
        assert callable(log_step_warning)
        assert callable(log_step_error)
        assert callable(setup_step_logging)

    def test_setup_step_logging_returns_logger(self) -> None:
        """Test that setup_step_logging returns a logger."""
        from gnn.utils.logging_utils import setup_step_logging

        logger = setup_step_logging("test_step")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_step"

    def test_setup_step_logging_verbose(self) -> None:
        """Test verbose mode sets DEBUG level."""
        from gnn.utils.logging_utils import setup_step_logging

        logger = setup_step_logging("test_verbose", verbose=True)
        assert logger.getEffectiveLevel() <= logging.DEBUG

    def test_log_step_functions_no_exception(self) -> None:
        """Test that log functions don't raise exceptions."""
        from gnn.utils.logging.logging_utils import (
            log_step_error,
            log_step_start,
            log_step_success,
            log_step_warning,
            setup_step_logging,
        )

        logger = setup_step_logging("test_funcs")
        log_step_start(logger, "Starting test")
        log_step_success(logger, "Success message")
        log_step_warning(logger, "Warning message")
        log_step_error(logger, "Error message")


class TestBaseProcessor:
    """Tests for base_processor module."""

    def test_base_processor_imports(self) -> None:
        """Test that base_processor module imports correctly."""
        from gnn.utils.pipeline_orchestration.base_processor import (
            BaseProcessor,
            ProcessingResult,
            create_processor,
        )

        assert BaseProcessor is not None
        assert ProcessingResult is not None
        assert callable(create_processor)

    def test_processing_result_dataclass(self) -> None:
        """Test ProcessingResult dataclass."""
        from gnn.utils.pipeline_orchestration.base_processor import ProcessingResult

        result = ProcessingResult(success=True)
        assert result.success is True
        assert result.files_processed == 0
        assert result.files_failed == 0
        assert result.errors == []
        assert result.warnings == []

    def test_processing_result_to_dict(self) -> None:
        """Test ProcessingResult to_dict method."""
        from gnn.utils.pipeline_orchestration.base_processor import ProcessingResult

        result = ProcessingResult(success=True, files_processed=5, files_failed=1)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert d["files_processed"] == 5
        assert d["files_failed"] == 1

    def test_processing_result_save_to_json(self) -> None:
        """Test ProcessingResult save_to_json method."""
        import json

        from gnn.utils.pipeline_orchestration.base_processor import ProcessingResult

        result = ProcessingResult(success=True, files_processed=3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        try:
            result.save_to_json(temp_path)
            with open(temp_path) as f:
                loaded = json.load(f)
            assert loaded["success"] is True
            assert loaded["files_processed"] == 3
        finally:
            temp_path.unlink(missing_ok=True)

    def test_create_processor_function(self) -> None:
        """Test create_processor factory function."""
        from gnn.utils.pipeline_orchestration.base_processor import (
            BaseProcessor,
            create_processor,
        )

        def simple_process(file_path: Any, output_dir: Any) -> bool:
            return True

        processor = create_processor("test_step", simple_process)
        assert isinstance(processor, BaseProcessor)
        assert processor.step_name == "test_step"

    def test_base_processor_find_files(self) -> None:
        """Test BaseProcessor.find_files method."""
        from gnn.utils.pipeline_orchestration.base_processor import create_processor

        def sample_process(file_path: Any, output_dir: Any) -> bool:
            return True

        processor = create_processor("test", sample_process)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "test1.md").write_text("# Test 1")
            (tmppath / "test2.md").write_text("# Test 2")
            (tmppath / "test3.txt").write_text("Not markdown")
            files = processor.find_files(tmppath, extensions=[".md"])
            assert len(files) == 2
            assert all(f.suffix == ".md" for f in files)


class TestModuleIntegration:
    """Integration tests for new modules."""

    def test_utils_package_exports_new_modules(self) -> None:
        """Test that utils package exports new modules."""
        from gnn.utils import BaseProcessor, ProcessingResult, create_processor

        assert BaseProcessor is not None
