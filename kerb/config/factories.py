"""Factory functions for creating configuration objects.

This module provides convenient factory functions for creating configuration instances.
"""

from typing import Optional, Union

from .enums import ProviderType
from .manager import ConfigManager
from .types import ModelConfig, ProviderConfig


def create_config_manager(
    app_name: str = "llm_app",
    config_file: Optional[str] = None,
    encryption_key: Optional[str] = None,
) -> ConfigManager:
    """Create a new configuration manager.

    Args:
        app_name: Application name
        config_file: Optional configuration file to load
        encryption_key: Optional encryption key for secrets

    Returns:
        Configured ConfigManager instance
    """
    return ConfigManager(
        app_name=app_name,
        config_file=config_file,
        encryption_key=encryption_key,
    )


def create_model_config(
    name: str,
    provider: Union[ProviderType, str],
    **kwargs,
) -> ModelConfig:
    """Create a model configuration.

    Args:
        name: Model name
        provider: Provider type or string
        **kwargs: Additional model configuration parameters

    Returns:
        ModelConfig instance
    """
    if isinstance(provider, str):
        provider = ProviderType(provider.lower())

    return ModelConfig(name=name, provider=provider, **kwargs)


def create_provider_config(
    provider: Union[ProviderType, str],
    **kwargs,
) -> ProviderConfig:
    """Create a provider configuration.

    Args:
        provider: Provider type or string
        **kwargs: Additional provider configuration parameters

    Returns:
        ProviderConfig instance
    """
    if isinstance(provider, str):
        provider = ProviderType(provider.lower())

    return ProviderConfig(provider=provider, **kwargs)
