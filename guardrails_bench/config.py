"""
Configuration and model registry for the benchmark suite.

Provides centralized configuration management and a registry pattern
for dynamically registering and discovering model backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class BenchmarkCategory(str, Enum):
    """Enumeration of supported benchmark categories."""

    JAILBREAK = "jailbreak"
    CONTENT_FILTER = "content_filter"
    INSTRUCTION_HIERARCHY = "instruction_hierarchy"
    DATA_LEAKAGE = "data_leakage"
    REFUSAL_ACCURACY = "refusal_accuracy"


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ModelConfig(BaseModel):
    """Configuration for a single model to benchmark."""

    name: str = Field(..., description="Human-readable model name")
    provider: ModelProvider = Field(..., description="Model provider backend")
    model_id: str = Field(..., description="Provider-specific model identifier")
    api_key: Optional[str] = Field(None, description="API key (reads from env if None)")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> Optional[str]:
        """Resolve the API key from config or environment variables."""
        if self.api_key:
            return self.api_key
        env_map = {
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        env_var = env_map.get(self.provider)
        if env_var:
            return os.environ.get(env_var)
        return None


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    categories: List[BenchmarkCategory] = Field(
        default_factory=lambda: list(BenchmarkCategory),
        description="Benchmark categories to run",
    )
    max_concurrent: int = Field(4, ge=1, le=32, description="Max concurrent requests")
    timeout: int = Field(60, ge=5, description="Per-request timeout in seconds")
    retries: int = Field(2, ge=0, description="Number of retries on failure")
    output_dir: Path = Field(Path("results"), description="Directory for result output")
    dataset_dir: Optional[Path] = Field(None, description="Custom dataset directory")
    seed: int = Field(42, description="Random seed for reproducibility")
    verbose: bool = Field(False, description="Enable verbose logging")

    def get_dataset_dir(self) -> Path:
        """Return the dataset directory, defaulting to the built-in data."""
        if self.dataset_dir:
            return self.dataset_dir
        return Path(__file__).parent / "datasets" / "data"


class ModelRegistry:
    """
    Registry for model backend implementations.

    Provides a singleton-like pattern for registering and looking up
    model classes by their provider identifier.

    Example::

        from guardrails_bench.config import ModelRegistry
        from guardrails_bench.models.openai_model import OpenAIModel

        ModelRegistry.register("openai", OpenAIModel)
        model_cls = ModelRegistry.get("openai")
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, provider: str, model_class: Type) -> None:
        """Register a model class for a given provider name."""
        cls._registry[provider] = model_class

    @classmethod
    def get(cls, provider: str) -> Type:
        """Retrieve a model class by provider name."""
        if provider not in cls._registry:
            raise KeyError(
                f"No model registered for provider '{provider}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider]

    @classmethod
    def available_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (primarily for testing)."""
        cls._registry.clear()
