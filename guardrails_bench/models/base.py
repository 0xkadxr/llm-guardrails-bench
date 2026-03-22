"""
Abstract base class for model backends.

All model implementations must subclass ``BaseModel`` and provide
a ``generate`` method that accepts a user prompt and an optional
system prompt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from guardrails_bench.config import ModelConfig


class BaseModel(ABC):
    """
    Abstract interface for a language model backend.

    Subclasses wrap provider-specific APIs (OpenAI, Anthropic, local
    HuggingFace models, etc.) behind a uniform interface so that the
    benchmark runner can evaluate them interchangeably.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.name = config.name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user message / test prompt.
            system_prompt: Optional system-level instruction prepended
                to the conversation.

        Returns:
            The model's text response.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
