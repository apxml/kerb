"""Configuration enumerations.

This module defines enumeration types used throughout the config system.
"""

from enum import Enum


class ConfigSource(Enum):
    """Configuration source types."""

    ENVIRONMENT = "environment"
    FILE = "file"
    CODE = "code"
    DEFAULT = "default"


class ProviderType(Enum):
    """LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
