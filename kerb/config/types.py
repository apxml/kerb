"""Configuration data models.

This module defines the data classes used for configuration management.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .enums import ProviderType


@dataclass
class ModelConfig:
    """Configuration for a specific model.
    
    Attributes:
        name: Model identifier (e.g., "gpt-4", "claude-3-opus")
        provider: LLM provider
        api_key_env_var: Environment variable name for API key
        max_tokens: Maximum tokens for this model
        temperature: Default temperature (0.0 to 2.0)
        top_p: Default top_p sampling parameter
        frequency_penalty: Frequency penalty for repetition
        presence_penalty: Presence penalty for topic diversity
        endpoint: Custom API endpoint (optional)
        api_version: API version for provider (e.g., Azure)
        deployment_name: Deployment name (for Azure)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        metadata: Additional metadata
    """
    name: str
    provider: ProviderType
    api_key_env_var: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enum serialization."""
        data = asdict(self)
        data['provider'] = self.provider.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary, handling enum deserialization."""
        if isinstance(data.get('provider'), str):
            data['provider'] = ProviderType(data['provider'])
        return cls(**data)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.
    
    Attributes:
        provider: Provider type
        api_key: API key (prefer api_key_env_var for security)
        api_key_env_var: Environment variable containing API key
        base_url: Base URL for API endpoint
        organization: Organization ID (for OpenAI)
        timeout: Default timeout in seconds
        max_retries: Default max retry attempts
        rate_limit: Rate limit (requests per minute)
        models: Available models for this provider
        metadata: Additional provider metadata
    """
    provider: ProviderType
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    rate_limit: Optional[int] = None
    models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from env var or direct value.
        
        Priority:
        1. Environment variable (if set and exists)
        2. Direct API key value
        """
        if self.api_key_env_var:
            env_key = os.getenv(self.api_key_env_var)
            if env_key:
                return env_key
        return self.api_key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['provider'] = self.provider.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderConfig':
        """Create from dictionary."""
        if isinstance(data.get('provider'), str):
            data['provider'] = ProviderType(data['provider'])
        return cls(**data)


@dataclass
class AppConfig:
    """Complete application configuration.
    
    Attributes:
        app_name: Application name
        default_model: Default model to use
        providers: Provider configurations
        models: Model configurations
        metadata: Additional application metadata
    """
    app_name: str
    default_model: Optional[str] = None
    providers: Dict[ProviderType, ProviderConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'app_name': self.app_name,
            'default_model': self.default_model,
            'providers': {p.value: pc.to_dict() for p, pc in self.providers.items()},
            'models': {k: mc.to_dict() for k, mc in self.models.items()},
            'metadata': self.metadata,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create from dictionary."""
        if 'providers' in data:
            data['providers'] = {
                ProviderType(k) if isinstance(k, str) else k: ProviderConfig.from_dict(v)
                for k, v in data['providers'].items()
            }
        
        if 'models' in data:
            data['models'] = {
                k: ModelConfig.from_dict(v)
                for k, v in data['models'].items()
            }
        
        return cls(**data)
