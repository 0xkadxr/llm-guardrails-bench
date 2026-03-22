"""Model backend implementations for various LLM providers."""

from guardrails_bench.models.base import BaseModel
from guardrails_bench.models.openai_model import OpenAIModel
from guardrails_bench.models.anthropic_model import AnthropicModel
from guardrails_bench.models.local_model import LocalModel
from guardrails_bench.config import ModelRegistry

# Auto-register built-in model backends
ModelRegistry.register("openai", OpenAIModel)
ModelRegistry.register("anthropic", AnthropicModel)
ModelRegistry.register("local", LocalModel)

__all__ = ["BaseModel", "OpenAIModel", "AnthropicModel", "LocalModel"]
