"""Core configuration manager.

This module contains the main ConfigManager class for centralized configuration management.
"""

import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .enums import ProviderType
from .types import AppConfig, ModelConfig, ProviderConfig


class ConfigManager:
    """Centralized configuration manager for LLM applications.

    Manages model configs, provider settings, and API keys in a unified way.
    Focuses on LLM-specific configuration without general application settings.
    """

    def __init__(
        self,
        app_name: str = "llm_app",
        config_file: Optional[str] = None,
        auto_load_env: bool = True,
        encryption_key: Optional[str] = None,
        encryption_salt: Optional[bytes] = None,
    ):
        """Initialize configuration manager.

        Args:
            app_name: Application name
            config_file: Path to configuration file
            auto_load_env: Automatically load from environment variables
            encryption_key: Optional encryption key for secrets (auto-generated if None)
            encryption_salt: Optional salt for key derivation (auto-generated if None)
        """
        self.app_name = app_name
        self._config: AppConfig = AppConfig(app_name=app_name)
        self._config_history: List[AppConfig] = []
        self._change_listeners: List[Callable[[AppConfig], None]] = []

        # Initialize secure secrets storage
        self._secrets: Dict[str, bytes] = {}
        self._init_encryption(encryption_key, encryption_salt)

        if config_file:
            self.load_from_file(config_file)

        if auto_load_env:
            self.load_from_environment()

    def _init_encryption(
        self, key: Optional[str] = None, salt: Optional[bytes] = None
    ) -> None:
        """Initialize encryption for secrets storage.

        Args:
            key: Optional encryption key (auto-generated if None)
            salt: Optional salt for key derivation (auto-generated if None)
        """
        try:
            import base64
            import secrets

            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            if key:
                # Generate or use provided salt
                if salt is None:
                    # Generate a cryptographically secure random salt
                    salt = secrets.token_bytes(16)

                # Store salt for potential serialization/persistence needs
                self._encryption_salt = salt

                # Derive a proper key from the provided key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
                self._fernet = Fernet(derived_key)
            else:
                # Generate a new key for this session
                self._fernet = Fernet(Fernet.generate_key())
                self._encryption_salt = None

            self._encryption_available = True
        except ImportError:
            # Fallback to no encryption if cryptography not available
            self._fernet = None
            self._encryption_available = False
            self._encryption_salt = None

    def get_config(self) -> AppConfig:
        """Get current application configuration."""
        return deepcopy(self._config)

    def set_config(self, config: AppConfig) -> None:
        """Set application configuration and notify listeners."""
        self._config_history.append(deepcopy(self._config))
        self._config = config
        self._notify_listeners()

    def add_change_listener(self, listener: Callable[[AppConfig], None]) -> None:
        """Add a listener for configuration changes."""
        self._change_listeners.append(listener)

    def _notify_listeners(self) -> None:
        """Notify all listeners of configuration changes."""
        for listener in self._change_listeners:
            try:
                listener(self.get_config())
            except Exception:
                pass  # Don't let listener errors break the manager

    def _save_history(self) -> None:
        """Save current configuration to history before making changes."""
        self._config_history.append(deepcopy(self._config))

    # ========================================================================
    # Model Configuration
    # ========================================================================

    def add_model(self, config: ModelConfig) -> None:
        """Add or update a model configuration."""
        self._save_history()
        self._config.models[config.name] = config
        self._notify_listeners()

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._config.models.get(name)

    def remove_model(self, name: str) -> bool:
        """Remove a model configuration."""
        if name in self._config.models:
            del self._config.models[name]
            self._notify_listeners()
            return True
        return False

    def list_models(self, provider: Optional[ProviderType] = None) -> List[str]:
        """List all configured models, optionally filtered by provider."""
        if provider:
            return [
                name
                for name, config in self._config.models.items()
                if config.provider == provider
            ]
        return list(self._config.models.keys())

    def set_default_model(self, name: str) -> None:
        """Set the default model."""
        if name not in self._config.models:
            raise ValueError(f"Model '{name}' not found in configuration")
        self._config.default_model = name
        self._notify_listeners()

    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        if self._config.default_model:
            return self._config.models.get(self._config.default_model)
        return None

    # ========================================================================
    # Provider Configuration
    # ========================================================================

    def add_provider(self, config: ProviderConfig) -> None:
        """Add or update a provider configuration."""
        self._config.providers[config.provider] = config
        self._notify_listeners()

    def get_provider(self, provider: ProviderType) -> Optional[ProviderConfig]:
        """Get provider configuration."""
        return self._config.providers.get(provider)

    def remove_provider(self, provider: ProviderType) -> bool:
        """Remove a provider configuration."""
        if provider in self._config.providers:
            del self._config.providers[provider]
            self._notify_listeners()
            return True
        return False

    def list_providers(self) -> List[ProviderType]:
        """List all configured providers."""
        return list(self._config.providers.keys())

    def switch_provider(
        self,
        from_provider: ProviderType,
        to_provider: ProviderType,
        model_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Switch from one provider to another.

        Args:
            from_provider: Current provider
            to_provider: Target provider
            model_mapping: Optional mapping of old model names to new ones
        """
        if to_provider not in self._config.providers:
            raise ValueError(f"Target provider '{to_provider}' not configured")

        # Update models that use the old provider
        models_to_update = []
        for name, config in list(self._config.models.items()):
            if config.provider == from_provider:
                models_to_update.append((name, config))

        for old_name, config in models_to_update:
            config.provider = to_provider

            # Handle renaming if mapping provided
            if model_mapping and old_name in model_mapping:
                new_name = model_mapping[old_name]
                config.name = new_name

                # Remove old entry and add new one
                del self._config.models[old_name]
                self._config.models[new_name] = config

                # Update default model if needed
                if self._config.default_model == old_name:
                    self._config.default_model = new_name

        self._notify_listeners()

    # ========================================================================
    # API Key Management
    # ========================================================================

    def set_api_key(
        self,
        provider: ProviderType,
        api_key: Optional[str] = None,
        env_var: Optional[str] = None,
    ) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider type
            api_key: Direct API key (not recommended for production)
            env_var: Environment variable name containing API key (recommended)
        """
        if provider not in self._config.providers:
            raise ValueError(f"Provider '{provider}' not configured")

        provider_config = self._config.providers[provider]

        if env_var:
            provider_config.api_key_env_var = env_var
            provider_config.api_key = None
        elif api_key:
            provider_config.api_key = api_key
            provider_config.api_key_env_var = None
        else:
            raise ValueError("Either api_key or env_var must be provided")

        self._notify_listeners()

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """Get API key for a provider (resolves env vars)."""
        provider_config = self._config.providers.get(provider)
        if not provider_config:
            return None
        return provider_config.get_api_key()

    def validate_api_keys(self) -> Dict[ProviderType, bool]:
        """Validate that all configured providers have API keys.

        Returns:
            Dictionary mapping providers to validation status
        """
        results = {}
        for provider, config in self._config.providers.items():
            api_key = config.get_api_key()
            results[provider] = api_key is not None and len(api_key) > 0
        return results

    # ========================================================================
    # File I/O
    # ========================================================================

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from JSON file.

        Args:
            file_path: Path to configuration file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(path, "r") as f:
            data = json.load(f)

        self._config = AppConfig.from_dict(data)
        self._notify_listeners()

    def save_to_file(self, file_path: str, include_secrets: bool = False) -> None:
        """Save configuration to JSON file.

        Args:
            file_path: Path to save configuration
            include_secrets: Include API keys in export (not recommended)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config.to_dict()

        if not include_secrets:
            # Remove sensitive data
            for provider_data in data.get("providers", {}).values():
                provider_data["api_key"] = None

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_environment(self) -> None:
        """Load configuration from environment variables.

        Looks for variables in the format:
        - {APP_NAME}_MODEL_CONFIG__{MODEL_NAME}__{PARAM}
        - {APP_NAME}_PROVIDER__{PROVIDER}__{PARAM}
        - {APP_NAME}__{PARAM}
        """
        prefix = self.app_name.upper().replace("-", "_")

        # Load general settings
        for key, value in os.environ.items():
            if key.startswith(f"{prefix}__"):
                param = key[len(f"{prefix}__") :].lower()

                if param == "default_model":
                    self._config.default_model = value

        self._notify_listeners()

    def export_environment_vars(self) -> Dict[str, str]:
        """Export configuration as environment variables.

        Returns:
            Dictionary of environment variable names to values
        """
        prefix = self.app_name.upper().replace("-", "_")
        env_vars = {}

        # Export default model
        if self._config.default_model:
            env_vars[f"{prefix}__DEFAULT_MODEL"] = self._config.default_model

        # Export provider API key env vars
        for provider, config in self._config.providers.items():
            if config.api_key_env_var:
                provider_prefix = f"{prefix}_PROVIDER__{provider.value.upper()}"
                env_vars[f"{provider_prefix}__API_KEY_ENV"] = config.api_key_env_var

        return env_vars

    # ========================================================================
    # Validation
    # ========================================================================

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check default model exists
        if self._config.default_model:
            if self._config.default_model not in self._config.models:
                issues.append(
                    f"Default model '{self._config.default_model}' not found in models"
                )

        # Check models reference valid providers
        for name, model in self._config.models.items():
            if model.provider not in self._config.providers:
                issues.append(
                    f"Model '{name}' references unconfigured provider '{model.provider.value}'"
                )

        # Check API keys
        api_key_status = self.validate_api_keys()
        for provider, has_key in api_key_status.items():
            if not has_key:
                issues.append(f"Provider '{provider.value}' missing API key")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    # ========================================================================
    # Secrets Management
    # ========================================================================

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret value with encryption.

        Secrets are encrypted in memory using Fernet symmetric encryption.
        While not as secure as dedicated secrets management services,
        this provides reasonable protection for prototyping and development.

        Args:
            key: Secret identifier
            value: Secret value to store
        """
        if self._encryption_available and self._fernet:
            # Encrypt the secret before storing
            encrypted_value = self._fernet.encrypt(value.encode())
            self._secrets[key] = encrypted_value
        else:
            # Fallback: store without encryption (with warning)
            warnings.warn(
                "Cryptography library not available. Secrets stored without encryption. "
                "Install with: pip install cryptography",
                RuntimeWarning,
            )
            self._secrets[key] = value.encode()

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt a secret value.

        Args:
            key: Secret identifier

        Returns:
            Decrypted secret value or None if not found
        """
        if key not in self._secrets:
            return None

        encrypted_value = self._secrets[key]

        if self._encryption_available and self._fernet:
            try:
                # Decrypt the secret
                decrypted_value = self._fernet.decrypt(encrypted_value)
                return decrypted_value.decode()
            except Exception:
                return None
        else:
            # Fallback: return unencrypted value
            return encrypted_value.decode()

    def remove_secret(self, key: str) -> bool:
        """Remove a secret and securely clear it from memory.

        Args:
            key: Secret identifier

        Returns:
            True if secret was removed, False if not found
        """
        if key in self._secrets:
            # Securely overwrite before deletion
            self._secrets[key] = b"\x00" * len(self._secrets[key])
            del self._secrets[key]
            return True
        return False

    def clear_secrets(self) -> None:
        """Clear all secrets from memory securely."""
        for key in list(self._secrets.keys()):
            self.remove_secret(key)

    def list_secret_keys(self) -> List[str]:
        """List all secret keys (not values).

        Returns:
            List of secret identifiers
        """
        return list(self._secrets.keys())

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists.

        Args:
            key: Secret identifier

        Returns:
            True if secret exists
        """
        return key in self._secrets

    # ========================================================================
    # Utilities
    # ========================================================================

    def reset(self) -> None:
        """Reset configuration to initial state and clear secrets."""
        self._config = AppConfig(app_name=self.app_name)
        self.clear_secrets()
        self._notify_listeners()

    def rollback(self) -> bool:
        """Rollback to previous configuration.

        Returns:
            True if rollback successful, False if no history
        """
        if self._config_history:
            self._config = self._config_history.pop()
            self._notify_listeners()
            return True
        return False

    def merge_config(self, other: AppConfig, override: bool = True) -> None:
        """Merge another configuration into current.

        Args:
            other: Configuration to merge
            override: Whether to override existing values
        """
        if override or not self._config.default_model:
            self._config.default_model = other.default_model

        # Merge providers
        for provider, config in other.providers.items():
            if override or provider not in self._config.providers:
                self._config.providers[provider] = config

        # Merge models
        for name, config in other.models.items():
            if override or name not in self._config.models:
                self._config.models[name] = config

        # Merge metadata
        if override:
            self._config.metadata.update(other.metadata)
        else:
            for k, v in other.metadata.items():
                if k not in self._config.metadata:
                    self._config.metadata[k] = v

        self._notify_listeners()

    def get_model_for_task(
        self,
        task: str,
        fallback: Optional[str] = None,
    ) -> Optional[ModelConfig]:
        """Get recommended model for a specific task.

        Args:
            task: Task type (e.g., "completion", "embedding", "chat")
            fallback: Fallback model name if no match found

        Returns:
            Model configuration or None
        """
        # Check metadata for task-specific models
        for name, config in self._config.models.items():
            if config.metadata.get("recommended_for") == task:
                return config

        # Use fallback
        if fallback:
            return self._config.models.get(fallback)

        # Use default
        return self.get_default_model()

    def clone(self) -> "ConfigManager":
        """Create a deep copy of the configuration manager.

        Note: Secrets are not cloned for security reasons.
        """
        new_manager = ConfigManager(
            app_name=self.app_name,
            auto_load_env=False,
        )
        new_manager._config = deepcopy(self._config)
        return new_manager
