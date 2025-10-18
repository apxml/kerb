"""Provider configuration utilities.

This module provides utilities for working with LLM provider configurations,
including default configurations and credential validation.
"""

from typing import Optional

from .enums import ProviderType
from .types import ProviderConfig


def get_openai_config(api_key_env_var: str = "OPENAI_API_KEY") -> ProviderConfig:
    """Get default OpenAI provider configuration.
    
    Args:
        api_key_env_var: Environment variable name for API key
    
    Returns:
        ProviderConfig for OpenAI
    """
    return ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var=api_key_env_var,
        base_url="https://api.openai.com/v1",
        timeout=60.0,
        max_retries=3,
        models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
    )


def get_anthropic_config(api_key_env_var: str = "ANTHROPIC_API_KEY") -> ProviderConfig:
    """Get default Anthropic provider configuration.
    
    Args:
        api_key_env_var: Environment variable name for API key
    
    Returns:
        ProviderConfig for Anthropic
    """
    return ProviderConfig(
        provider=ProviderType.ANTHROPIC,
        api_key_env_var=api_key_env_var,
        base_url="https://api.anthropic.com",
        timeout=60.0,
        max_retries=3,
        models=["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    )


def get_google_config(api_key_env_var: str = "GOOGLE_API_KEY") -> ProviderConfig:
    """Get default Google (Gemini) provider configuration.
    
    Args:
        api_key_env_var: Environment variable name for API key
    
    Returns:
        ProviderConfig for Google
    """
    return ProviderConfig(
        provider=ProviderType.GOOGLE,
        api_key_env_var=api_key_env_var,
        timeout=60.0,
        max_retries=3,
        models=["gemini-pro", "gemini-pro-vision"],
    )


def validate_credentials(
    provider: ProviderType,
    api_key: str,
) -> bool:
    """Validate provider credentials (basic check).
    
    Note: This is a basic validation. For production, implement
    actual API calls to verify credentials.
    
    Args:
        provider: Provider type
        api_key: API key to validate
    
    Returns:
        True if credentials appear valid (basic check)
    """
    if not api_key or len(api_key) < 10:
        return False
    
    # Basic prefix checks
    prefix_checks = {
        ProviderType.OPENAI: api_key.startswith('sk-'),
        ProviderType.ANTHROPIC: api_key.startswith('sk-ant-'),
        ProviderType.COHERE: len(api_key) > 20,
        ProviderType.GOOGLE: len(api_key) > 20,
    }
    
    return prefix_checks.get(provider, len(api_key) > 10)
