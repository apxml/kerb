"""Configuration file I/O utilities.

This module provides functions for loading and saving configuration files.
"""

import json
from pathlib import Path
from typing import Union

from .types import AppConfig


def load_config(file_path: str) -> AppConfig:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to configuration file

    Returns:
        AppConfig instance
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, "r") as f:
        data = json.load(f)

    return AppConfig.from_dict(data)


def save_config(
    config: AppConfig,
    file_path: str,
    include_secrets: bool = False,
) -> None:
    """Save configuration to a JSON file.

    Args:
        config: Configuration to save
        file_path: Path to save file
        include_secrets: Include sensitive data (not recommended)
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    if not include_secrets:
        for provider_data in data.get("providers", {}).values():
            provider_data["api_key"] = None

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
