"""
Local / HuggingFace model backend.

Loads transformer models locally using the ``transformers`` library,
supporting any causal language model available on the HuggingFace Hub.
"""

from __future__ import annotations

from typing import Optional

from guardrails_bench.config import ModelConfig
from guardrails_bench.models.base import BaseModel


class LocalModel(BaseModel):
    """
    Model backend for locally hosted HuggingFace transformer models.

    Loads the model and tokenizer on first instantiation. Supports
    GPU acceleration when ``torch.cuda.is_available()`` is ``True``.

    Example::

        config = ModelConfig(
            name="Llama 3 8B",
            provider=ModelProvider.LOCAL,
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            extra_params={"device_map": "auto", "torch_dtype": "float16"},
        )
        model = LocalModel(config)
        response = model.generate("What is the capital of France?")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for "
                "LocalModel. Install them with: pip install transformers torch"
            ) from exc

        device_map = config.extra_params.get("device_map", "auto")
        torch_dtype_str = config.extra_params.get("torch_dtype", "float16")
        torch_dtype = getattr(torch, torch_dtype_str, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.extra_params.get("trust_remote_code", False),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.extra_params.get("trust_remote_code", False),
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response using the local transformer model."""
        import torch

        # Build a chat-style prompt if the tokenizer supports it
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers without chat templates
            parts = []
            if system_prompt:
                parts.append(f"System: {system_prompt}")
            parts.append(f"User: {prompt}")
            parts.append("Assistant:")
            formatted = "\n".join(parts)

        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=max(self.config.temperature, 1e-7),
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (exclude the input)
        input_length = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
