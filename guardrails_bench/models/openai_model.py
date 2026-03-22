"""
OpenAI API model backend.

Wraps the ``openai`` Python SDK to provide chat completions through
GPT-4, GPT-4o, GPT-3.5-turbo, and other OpenAI models.
"""

from __future__ import annotations

from typing import Optional

from guardrails_bench.config import ModelConfig
from guardrails_bench.models.base import BaseModel


class OpenAIModel(BaseModel):
    """
    Model backend for OpenAI's Chat Completions API.

    Requires the ``openai`` package and a valid API key (set via
    ``OPENAI_API_KEY`` environment variable or passed in the config).

    Example::

        config = ModelConfig(
            name="GPT-4o",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
        )
        model = OpenAIModel(config)
        response = model.generate("Hello, how are you?")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIModel. "
                "Install it with: pip install openai"
            ) from exc

        api_key = config.resolve_api_key()
        if not api_key:
            raise ValueError(
                "No API key found for OpenAI. Set the OPENAI_API_KEY "
                "environment variable or pass it in the ModelConfig."
            )

        self.client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a chat completion request to the OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params,
        )

        return response.choices[0].message.content or ""
