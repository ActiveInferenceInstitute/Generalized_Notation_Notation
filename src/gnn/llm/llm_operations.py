#!/usr/bin/env python3
"""
LLM Operations Module for GNN Pipeline

This module provides Large Language Model operations for analyzing
and processing GNN files, including summarization, analysis, and enhancement.

Uses the multi-provider LLM system for provider selection, request handling,
and analysis.
"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

# Import the LLM system directly to avoid circular imports
from .llm_processor import (
    AnalysisType,
    LLMProcessor,
    get_default_provider_configs,
    load_api_keys_from_env,
)
from .llm_processor import get_processor as get_global_processor
from .providers.base_provider import LLMConfig, LLMMessage, ProviderType
from .sync_bridge import run_async

logger = logging.getLogger(__name__)

# Constants for default values
DEFAULT_MODEL: Optional[str] = None  # Let the selected provider resolve its model.
DEFAULT_MAX_TOKENS = 8000


class LLMOperations:
    """
    Main class for LLM operations on GNN content.

    This class uses the multi-provider LLM system internally.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize LLM operations.

        Args:
            api_key: OpenAI API key (if None, will try to get from env)
        """
        # Multi-provider initialization
        try:
            # Try to use global processor first
            self.processor = get_global_processor()
            if self.processor:
                self._initialized = True
                logger.info("Using global LLM processor")
            else:
                # Create new processor
                api_keys = load_api_keys_from_env()
                if api_key:
                    api_keys["openai"] = api_key

                self.processor = LLMProcessor(
                    api_keys=api_keys, provider_configs=get_default_provider_configs()
                )
                self._initialized = (
                    False  # Will be initialized on first use or explicit call
                )
                logger.info("Created new LLM processor")
        except Exception as e:
            logger.error(f"Failed to initialize multi-provider system: {e}")
            raise

    async def _ensure_initialized(self) -> bool:
        """Ensure the processor is initialized."""
        if not self._initialized and self.processor:
            self._initialized = await self.processor.initialize()

        return self._initialized

    def _require_processor(self) -> LLMProcessor:
        """Return the initialized processor or raise a clear runtime error."""
        if self.processor is None:
            raise RuntimeError("LLM processor not initialized")
        return self.processor

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine from synchronous context, handling already-running event loops."""
        return run_async(coro)

    def construct_prompt(self, content_parts: List[str], task_description: str) -> str:
        """
        Construct a well-formatted prompt for LLM processing.

        Args:
            content_parts: List of content pieces to include
            task_description: Description of the task to perform

        Returns:
            Formatted prompt string
        """
        prompt_parts: list[Any] = [
            "You are an expert in Active Inference and GNN (Generalized Notation Notation) specifications.",
            "",
            f"Task: {task_description}",
            "",
            "Content to analyze:",
        ]

        for i, content in enumerate(content_parts, 1):
            prompt_parts.extend([f"--- Content Part {i} ---", content, ""])

        prompt_parts.extend(
            [
                "Please provide a comprehensive and accurate response based on the content above.",
                "Focus on Active Inference concepts, model structure, and practical implications.",
            ]
        )

        return "\n".join(prompt_parts)

    def get_llm_response(
        self,
        prompt: str,
        model: Optional[str] = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Get response from LLM for given prompt.

        Args:
            prompt: Input prompt for the LLM
            model: Model to use for generation
            max_tokens: Maximum tokens in response

        Returns:
            LLM response or error message
        """
        try:
            return cast(
                "str",
                self._run_async(self._get_async_response(prompt, model, max_tokens)),
            )
        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}"

    async def _get_async_response(
        self,
        prompt: str,
        model: Optional[str] = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Async version of get_llm_response."""
        if not await self._ensure_initialized():
            return "Error: LLM processor not initialized"

        try:
            # Use the new processor system
            response = await self._require_processor().get_response(
                messages=[LLMMessage(role="user", content=prompt)],
                model_name=model,
                max_tokens=max_tokens,
                temperature=0.3,
            )

            if response.finish_reason != "stop":
                raise RuntimeError(f"LLM response unfinished: {response.finish_reason}")
            return response.content

        except Exception as e:
            logger.error(f"Multi-provider LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}"

    async def _async_analyze(
        self,
        gnn_content: str,
        analysis_type: AnalysisType,
        provider_type: Optional[ProviderType] = None,
        config: Optional[LLMConfig] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run one GNN analysis on the multi-provider system and return its text.

        Single funnel for every structured operation: ensures the processor is
        initialized, routes through :meth:`LLMProcessor.analyze_gnn`, and
        returns the response content.
        """
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        response = await self._require_processor().analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=analysis_type,
            provider_type=provider_type,
            config=config,
            additional_context=additional_context,
        )
        if response.finish_reason != "stop":
            raise RuntimeError(f"LLM analysis unfinished: {response.finish_reason}")
        return response.content

    def summarize_gnn(
        self,
        gnn_content: str,
        max_length: int = 500,
        ollama_model: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of GNN content.

        Args:
            gnn_content: The GNN file content to summarize
            max_length: Maximum length of summary
            ollama_model: When set (e.g. from pipeline model selection), use Ollama with this tag.

        Returns:
            Summary text
        """
        try:
            return cast(
                "str",
                self._run_async(
                    self._async_summarize_gnn(gnn_content, max_length, ollama_model)
                ),
            )
        except Exception as e:
            logger.error(f"Async summarization failed: {e}")
            return f"Error: Summarization failed - {str(e)}"

    async def _async_summarize_gnn(
        self,
        gnn_content: str,
        max_length: int = 500,
        ollama_model: Optional[str] = None,
    ) -> str:
        """Async GNN summarization (Ollama-pinned when a tag is given)."""
        if ollama_model:
            return await self._async_analyze(
                gnn_content,
                AnalysisType.SUMMARY,
                provider_type=ProviderType.OLLAMA,
                config=LLMConfig(
                    model=ollama_model, max_tokens=max(1024, max_length * 3)
                ),
            )
        return await self._async_analyze(gnn_content, AnalysisType.SUMMARY)

    def analyze_gnn_structure(self, gnn_content: str) -> str:
        """
        Analyze the structure and components of a GNN model.

        Args:
            gnn_content: The GNN file content to analyze

        Returns:
            Structured analysis
        """
        try:
            return cast(
                "str", self._run_async(self._async_analyze_structure(gnn_content))
            )
        except Exception as e:
            logger.error(f"Async structure analysis failed: {e}")
            return f"Error: Structure analysis failed - {str(e)}"

    async def _async_analyze_structure(self, gnn_content: str) -> str:
        """Async structure analysis."""
        return await self._async_analyze(gnn_content, AnalysisType.STRUCTURE)

    def generate_questions(self, gnn_content: str, num_questions: int = 5) -> List[str]:
        """
        Generate relevant questions about a GNN model.

        Args:
            gnn_content: The GNN file content
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        try:
            return cast(
                "list[str]",
                self._run_async(
                    self._async_generate_questions(gnn_content, num_questions)
                ),
            )
        except Exception as e:
            logger.error(f"Async question generation failed: {e}")
            return []

    async def _async_generate_questions(
        self, gnn_content: str, num_questions: int = 5
    ) -> List[str]:
        """Async question generation."""
        response_text = await self._async_analyze(
            gnn_content,
            AnalysisType.QUESTIONS,
            additional_context={"num_questions": num_questions},
        )
        return self._extract_questions_from_response(response_text, num_questions)

    def _extract_questions_from_response(
        self, response: str, num_questions: int
    ) -> List[str]:
        """Extract questions from LLM response."""
        lines = response.split("\n")
        questions: list[Any] = []

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("•")
            ):
                # Remove numbering and clean up
                question = line.split(".", 1)[-1].strip()
                if question:
                    questions.append(question)

        return questions[:num_questions]

    # New methods leveraging the multi-provider system
    def enhance_gnn(self, gnn_content: str) -> str:
        """
        Generate enhancement suggestions for a GNN model.

        Args:
            gnn_content: The GNN file content to enhance

        Returns:
            Enhancement suggestions
        """
        try:
            return cast("str", self._run_async(self._async_enhance_gnn(gnn_content)))
        except Exception as e:
            logger.error(f"Async enhancement failed: {e}")
            return f"Error: Enhancement failed - {str(e)}"

    async def _async_enhance_gnn(self, gnn_content: str) -> str:
        """Async enhancement suggestions."""
        return await self._async_analyze(gnn_content, AnalysisType.ENHANCEMENT)

    def validate_gnn_with_llm(self, gnn_content: str) -> str:
        """
        Validate a GNN model for correctness and completeness.

        Args:
            gnn_content: The GNN file content to validate

        Returns:
            Validation results
        """
        try:
            return cast("str", self._run_async(self._async_validate_gnn(gnn_content)))
        except Exception as e:
            logger.error(f"Async validation failed: {e}")
            return f"Error: Validation failed - {str(e)}"

    async def _async_validate_gnn(self, gnn_content: str) -> str:
        """Async GNN validation."""
        return await self._async_analyze(gnn_content, AnalysisType.VALIDATION)

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        if self.processor:
            return [p.value for p in self.processor.get_available_providers()]

        return []

    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the LLM processor."""
        if self.processor:
            return {
                "mode": "multi-provider",
                "providers": [
                    p.value for p in self.processor.get_available_providers()
                ],
                "initialized": self._initialized,
                "provider_info": self.processor.get_provider_info(),
            }

        return {"mode": "uninitialized"}


# Lazy singleton — instantiated on first use so env vars are read at call time,
# not at import time (which would snapshot env state for the process lifetime).
_llm_ops: Optional["LLMOperations"] = None


def _get_llm_ops() -> "LLMOperations":
    """Return llm ops."""
    global _llm_ops
    if _llm_ops is None:
        _llm_ops = LLMOperations()
    return _llm_ops


# Convenience functions.
def construct_prompt(content_parts: List[str], task_description: str) -> str:
    """Convenience function for prompt construction."""
    return _get_llm_ops().construct_prompt(content_parts, task_description)


def get_llm_response(
    prompt: str,
    model: Optional[str] = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Convenience function for getting LLM response."""
    return _get_llm_ops().get_llm_response(prompt, model, max_tokens)


def load_api_key() -> Optional[str]:
    """
    Load API key from environment or return None.

    Returns:
        API key string or None if not available
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key loaded from environment")
        return api_key
    return None


# Additional convenience functions for new capabilities
def summarize_gnn(
    gnn_content: str, max_length: int = 500, ollama_model: Optional[str] = None
) -> str:
    """Convenience function for GNN summarization."""
    return _get_llm_ops().summarize_gnn(gnn_content, max_length, ollama_model)


def analyze_gnn_structure(gnn_content: str) -> str:
    """Convenience function for GNN structure analysis."""
    return _get_llm_ops().analyze_gnn_structure(gnn_content)


def generate_questions(gnn_content: str, num_questions: int = 5) -> List[str]:
    """Convenience function for question generation."""
    return _get_llm_ops().generate_questions(gnn_content, num_questions)


def enhance_gnn(gnn_content: str) -> str:
    """Convenience function for GNN enhancement."""
    return _get_llm_ops().enhance_gnn(gnn_content)


def validate_gnn_with_llm(gnn_content: str) -> str:
    """Validate a GNN model with the LLM validation pipeline."""
    return _get_llm_ops().validate_gnn_with_llm(gnn_content)


if __name__ == "__main__":
    # Example Usage (requires .env file with OPENAI_API_KEY)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing llm_operations.py...")
    try:
        load_api_key()

        example_gnn_content = """
## ModelName
MyExampleGNN

## StateSpaceBlock
# States
S1: [Location] dimension(3) # {Location_A, Location_B, Location_C}
O1: [Observation] dimension(2) # {Obs_Hot, Obs_Cold}

## Connections
S1 -> O1
        """

        contexts: list[Any] = [
            "Context: This is a GNN file describing a simple model.",
            f"GNN File Content:\n{example_gnn_content}",
        ]

        task_summary = "Provide a concise summary of this GNN model, highlighting its key components."
        prompt_summary = construct_prompt(contexts, task_summary)
        print(f"\n--- Prompt for Summary ---\n{prompt_summary}")

        summary = get_llm_response(prompt_summary)
        print(f"\n--- LLM Summary ---\n{summary}")

        task_explanation = "Explain the purpose of the StateSpaceBlock in this GNN file in simple terms."
        prompt_explanation = construct_prompt(
            [f"GNN File Content:\n{example_gnn_content}"], task_explanation
        )
        print(f"\n--- Prompt for Explanation ---\n{prompt_explanation}")
        explanation = get_llm_response(prompt_explanation)
        print(f"\n--- LLM Explanation ---\n{explanation}")

    except ValueError as ve:
        print(f"Setup Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")
