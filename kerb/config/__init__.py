"""Configuration management utilities for LLM applications.

This module provides comprehensive tools for managing LLM-specific configuration
across different providers and models.

Usage:
    # Common imports - top-level
    from kerb.config import (
        Config,              # Alias for ConfigManager
        ConfigManager,       # Main configuration manager
        load_config,         # Load configuration from file
        save_config,         # Save configuration to file
    )
    
    # Providers - specialized
    from kerb.config.providers import (
        get_openai_config,
        get_anthropic_config,
        get_google_config,
        validate_credentials,
    )

Features:
- Centralized LLM configuration management
- Multi-provider support (OpenAI, Anthropic, Google, Cohere, Azure, HuggingFace)
- Secure API key management with environment variables
- Encrypted secrets storage for prototyping/development
- Model configuration with full parameter control (temperature, max_tokens, etc.)
- Provider switching utilities
- Configuration validation
- File-based and environment variable loading
- Change listeners for reactive updates
- Configuration history and rollback
"""

# Core classes and common functions (top-level imports)
from .manager import ConfigManager
from .io import load_config, save_config

# Submodules for specialized functionality
from . import providers, factories

# Export core types for convenience
from .enums import ConfigSource, ProviderType
from .types import AppConfig, ModelConfig, ProviderConfig

# Export factory functions at top level for convenience
from .factories import (
    create_config_manager,
    create_model_config,
    create_provider_config,
)

# Export provider utilities at top level for convenience
from .providers import (
    get_openai_config,
    get_anthropic_config,
    get_google_config,
    validate_credentials,
)

# Backward compatibility aliases (old API)
load_config_from_file = load_config
save_config_to_file = save_config
get_default_openai_config = get_openai_config
get_default_anthropic_config = get_anthropic_config
get_default_google_config = get_google_config
validate_provider_credentials = validate_credentials

# Alias for convenience
Config = ConfigManager

__all__ = [
    # Core
    "Config",
    "ConfigManager",
    "load_config",
    "save_config",
    
    # Submodules
    "providers",
    "factories",
    
    # Types (for type hints and direct access)
    "ConfigSource",
    "ProviderType",
    "AppConfig",
    "ModelConfig",
    "ProviderConfig",
    
    # Factory functions
    "create_config_manager",
    "create_model_config",
    "create_provider_config",
    
    # Provider utilities
    "get_openai_config",
    "get_anthropic_config",
    "get_google_config",
    "validate_credentials",
    
    # Backward compatibility (old API)
    "load_config_from_file",
    "save_config_to_file",
    "get_default_openai_config",
    "get_default_anthropic_config",
    "get_default_google_config",
    "validate_provider_credentials",
]
