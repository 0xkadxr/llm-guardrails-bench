"""
Dataset loading utilities.

Handles loading JSON test datasets from the built-in data directory
or a user-specified custom directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from guardrails_bench.config import BenchmarkConfig


def load_dataset(
    filename: str,
    config: Optional[BenchmarkConfig] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load a JSON dataset file.

    Resolution order:
        1. Explicit ``data_dir`` argument.
        2. ``config.get_dataset_dir()`` if a config is provided.
        3. Built-in ``datasets/data/`` directory relative to this file.

    Args:
        filename: Name of the JSON file (e.g., ``"jailbreak_prompts.json"``).
        config: Optional benchmark config for resolving the data path.
        data_dir: Optional explicit directory containing the dataset file.

    Returns:
        Parsed list of dictionaries from the JSON file.

    Raises:
        FileNotFoundError: If the dataset file cannot be located.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if data_dir is not None:
        path = Path(data_dir) / filename
    elif config is not None:
        path = config.get_dataset_dir() / filename
    else:
        path = Path(__file__).parent / "data" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Ensure the file exists or specify a custom dataset_dir in BenchmarkConfig."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )

    return data
