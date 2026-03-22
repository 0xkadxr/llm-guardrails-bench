"""
Anthropic API model backend.

Wraps the ``anthropic`` Python SDK to provide message completions
through Claude 3, Claude 3.5, and other Anthropic models.
"""

from __future__ import annotations

from typing import Optional

from guardrails_bench.config import ModelConfig
from guardrails_bench.models.base import BaseModel


class AnthropicModel(BaseModel):
    """
    Model backend for Anthropic's Messages API.

    Requires the ``anthropic`` package and a valid API key (set via
    ``ANTHROPIC_API_KEY`` environment variable or passed in the config).

    Example::

        config = ModelConfig(
            name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
        )
        model = AnthropicModel(config)
        response = model.generate("Hello!")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicModel. "
                "Install it with: pip install anthropic"
            ) from exc

        api_key = config.resolve_api_key()
        if not api_key:
            raise ValueError(
                "No API key found for Anthropic. Set the ANTHROPIC_API_KEY "
                "environment variable or pass it in the ModelConfig."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a message request to the Anthropic API."""
        kwargs = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        kwargs.update(self.config.extra_params)

        response = self.client.messages.create(**kwargs)

        # Extract text from content blocks
        text_parts = [
            block.text
            for block in response.content
            if hasattr(block, "text")
        ]
        return "".join(text_parts)
