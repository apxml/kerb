"""Tests for config module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from kerb.config import (  # Enums; Data classes; Main class; Factory functions; File I/O; Default configurations; Utilities
    AppConfig, ConfigManager, ConfigSource, ModelConfig, ProviderConfig,
    ProviderType, create_config_manager, create_model_config,
    create_provider_config, get_default_anthropic_config,
    get_default_google_config, get_default_openai_config,
    load_config_from_file, save_config_to_file, validate_provider_credentials)

# ============================================================================
# Enum Tests
# ============================================================================


# REMOVED: Environment enum removed - not needed for LLM toolkit
# def test_environment_enum():
#     """Test Environment enum."""
#     assert Environment.DEVELOPMENT.value == "development"
#     assert Environment.STAGING.value == "staging"
#     assert Environment.PRODUCTION.value == "production"
#     assert Environment.TESTING.value == "testing"


def test_provider_type_enum():
    """Test ProviderType enum."""
    assert ProviderType.OPENAI.value == "openai"
    assert ProviderType.ANTHROPIC.value == "anthropic"
    assert ProviderType.GOOGLE.value == "google"
    assert ProviderType.COHERE.value == "cohere"
    assert ProviderType.AZURE_OPENAI.value == "azure_openai"


def test_config_source_enum():
    """Test ConfigSource enum."""
    assert ConfigSource.ENVIRONMENT.value == "environment"
    assert ConfigSource.FILE.value == "file"
    assert ConfigSource.CODE.value == "code"
    assert ConfigSource.DEFAULT.value == "default"


# ============================================================================
# Data Class Tests
# ============================================================================


def test_model_config_creation():
    """Test ModelConfig creation."""
    config = ModelConfig(
        name="gpt-4",
        provider=ProviderType.OPENAI,
        max_tokens=8000,
        temperature=0.7,
    )

    assert config.name == "gpt-4"
    assert config.provider == ProviderType.OPENAI
    assert config.max_tokens == 8000
    assert config.temperature == 0.7


def test_model_config_to_dict():
    """Test ModelConfig serialization."""
    config = ModelConfig(
        name="gpt-4", provider=ProviderType.OPENAI, metadata={"cost_per_1k": 0.03}
    )

    data = config.to_dict()
    assert data["name"] == "gpt-4"
    assert data["provider"] == "openai"
    assert data["metadata"]["cost_per_1k"] == 0.03


def test_model_config_from_dict():
    """Test ModelConfig deserialization."""
    data = {"name": "gpt-4", "provider": "openai", "max_tokens": 8000, "metadata": {}}

    config = ModelConfig.from_dict(data)
    assert config.name == "gpt-4"
    assert config.provider == ProviderType.OPENAI
    assert config.max_tokens == 8000


def test_provider_config_creation():
    """Test ProviderConfig creation."""
    config = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
        timeout=30.0,
    )

    assert config.provider == ProviderType.OPENAI
    assert config.api_key_env_var == "OPENAI_API_KEY"
    assert config.timeout == 30.0


def test_provider_config_get_api_key():
    """Test ProviderConfig API key retrieval."""
    # Test with direct key
    config = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key="test-key-123",
    )
    assert config.get_api_key() == "test-key-123"

    # Test with env var
    os.environ["TEST_API_KEY"] = "env-key-456"
    config = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var="TEST_API_KEY",
    )
    assert config.get_api_key() == "env-key-456"
    del os.environ["TEST_API_KEY"]


# REMOVED: test_environment_config_creation - environment functionality removed
# def test_environment_config_creation():
#     """Test EnvironmentConfig creation."""
#     config = EnvironmentConfig(
#         environment=Environment.PRODUCTION,
#         debug=False,
#         cache_enabled=True,
#     )

#     assert config.environment == Environment.PRODUCTION
#     assert config.debug is False
#     assert config.cache_enabled is True


def test_app_config_creation():
    """Test AppConfig creation."""
    config = AppConfig(
        app_name="test_app",
        default_model="gpt-4",
    )

    assert config.app_name == "test_app"
    assert config.default_model == "gpt-4"


# ============================================================================
# ConfigManager Tests
# ============================================================================


def test_config_manager_creation():
    """Test ConfigManager creation."""
    manager = ConfigManager(app_name="test_app", auto_load_env=False)
    assert manager.app_name == "test_app"
    assert manager._config is not None


def test_config_manager_add_model():
    """Test adding model to ConfigManager."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model = ModelConfig(
        name="gpt-4",
        provider=ProviderType.OPENAI,
    )

    manager.add_model(model)
    assert "gpt-4" in manager.list_models()

    retrieved = manager.get_model("gpt-4")
    assert retrieved is not None
    assert retrieved.name == "gpt-4"


def test_config_manager_remove_model():
    """Test removing model from ConfigManager."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)

    assert manager.remove_model("gpt-4")
    assert "gpt-4" not in manager.list_models()
    assert not manager.remove_model("nonexistent")


def test_config_manager_default_model():
    """Test default model management."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)

    manager.set_default_model("gpt-4")
    default = manager.get_default_model()

    assert default is not None
    assert default.name == "gpt-4"


def test_config_manager_default_model_not_found():
    """Test setting default model that doesn't exist."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    with pytest.raises(ValueError, match="not found"):
        manager.set_default_model("nonexistent")


def test_config_manager_add_provider():
    """Test adding provider to ConfigManager."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    provider = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
    )

    manager.add_provider(provider)
    assert ProviderType.OPENAI in manager.list_providers()

    retrieved = manager.get_provider(ProviderType.OPENAI)
    assert retrieved is not None
    assert retrieved.provider == ProviderType.OPENAI


def test_config_manager_remove_provider():
    """Test removing provider from ConfigManager."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    provider = ProviderConfig(provider=ProviderType.OPENAI)
    manager.add_provider(provider)

    assert manager.remove_provider(ProviderType.OPENAI)
    assert ProviderType.OPENAI not in manager.list_providers()


def test_config_manager_api_key_management():
    """Test API key management."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    provider = ProviderConfig(provider=ProviderType.OPENAI)
    manager.add_provider(provider)

    # Set with direct key
    manager.set_api_key(ProviderType.OPENAI, api_key="test-key")
    assert manager.get_api_key(ProviderType.OPENAI) == "test-key"

    # Set with env var
    os.environ["TEST_KEY"] = "env-key"
    manager.set_api_key(ProviderType.OPENAI, env_var="TEST_KEY")
    assert manager.get_api_key(ProviderType.OPENAI) == "env-key"
    del os.environ["TEST_KEY"]


def test_config_manager_validate_api_keys():
    """Test API key validation."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    provider1 = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key="test-key",
    )
    provider2 = ProviderConfig(
        provider=ProviderType.ANTHROPIC,
        api_key=None,
    )

    manager.add_provider(provider1)
    manager.add_provider(provider2)

    results = manager.validate_api_keys()
    assert results[ProviderType.OPENAI] is True
    assert results[ProviderType.ANTHROPIC] is False


def test_config_manager_switch_provider():
    """Test provider switching."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    # Add providers
    manager.add_provider(ProviderConfig(provider=ProviderType.OPENAI))
    manager.add_provider(ProviderConfig(provider=ProviderType.ANTHROPIC))

    # Add model with OpenAI
    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)
    manager.set_default_model("gpt-4")

    # Switch to Anthropic
    manager.switch_provider(
        ProviderType.OPENAI,
        ProviderType.ANTHROPIC,
        model_mapping={"gpt-4": "claude-3-opus"},
    )

    # Check model was updated
    updated = manager.get_model("claude-3-opus")
    assert updated is not None
    assert updated.provider == ProviderType.ANTHROPIC


def test_config_manager_validation():
    """Test configuration validation."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    # Add model without provider
    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)
    manager.set_default_model("gpt-4")

    issues = manager.validate()
    assert len(issues) > 0
    assert not manager.is_valid()

    # Add provider
    provider = ProviderConfig(provider=ProviderType.OPENAI, api_key="test-key")
    manager.add_provider(provider)

    issues = manager.validate()
    assert len(issues) == 0
    assert manager.is_valid()


# REMOVED: test_config_manager_environment_config - environment functionality removed
# def test_config_manager_environment_config():
#     """Test environment configuration management."""
#     manager = ConfigManager(app_name="test", auto_load_env=False)

#     env_config = EnvironmentConfig(
#         environment=Environment.PRODUCTION,
#         debug=False,
#     )

#     manager.set_environment_config(env_config)
#     retrieved = manager.get_environment_config()

#     assert retrieved is not None
#     assert retrieved.environment == Environment.PRODUCTION
#     assert retrieved.debug is False


# REMOVED: test_config_manager_is_production - environment functionality removed
# def test_config_manager_is_production():
#     """Test production environment check."""
#     manager = ConfigManager(
#         app_name="test",
#         environment=Environment.PRODUCTION,
#         auto_load_env=False,
#     )
#     assert manager.is_production()

#     manager = ConfigManager(
#         app_name="test",
#         environment=Environment.DEVELOPMENT,
#         auto_load_env=False,
#     )
#     assert not manager.is_production()


def test_config_manager_secrets():
    """Test secrets management."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    manager.set_secret("api_key", "secret-value")
    assert manager.get_secret("api_key") == "secret-value"

    assert manager.remove_secret("api_key")
    assert manager.get_secret("api_key") is None


def test_config_manager_reset():
    """Test configuration reset."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)
    assert len(manager.list_models()) > 0

    manager.reset()
    assert len(manager.list_models()) == 0


def test_config_manager_rollback():
    """Test configuration rollback."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    # Initial state
    model1 = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model1)

    # Change state
    model2 = ModelConfig(name="claude-3", provider=ProviderType.ANTHROPIC)
    manager.add_model(model2)

    # Rollback
    assert manager.rollback()
    assert "claude-3" not in manager.list_models()
    assert "gpt-4" in manager.list_models()


def test_config_manager_merge():
    """Test configuration merging."""
    manager1 = ConfigManager(app_name="test1", auto_load_env=False)
    manager2 = ConfigManager(app_name="test2", auto_load_env=False)

    # Add model to manager1
    model1 = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager1.add_model(model1)

    # Add model to manager2
    model2 = ModelConfig(name="claude-3", provider=ProviderType.ANTHROPIC)
    manager2.add_model(model2)

    # Merge
    config2 = manager2.get_config()
    manager1.merge_config(config2)

    # Check both models exist
    assert "gpt-4" in manager1.list_models()
    assert "claude-3" in manager1.list_models()


def test_config_manager_get_model_for_task():
    """Test task-based model retrieval."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model1 = ModelConfig(
        name="gpt-4", provider=ProviderType.OPENAI, metadata={"recommended_for": "chat"}
    )
    model2 = ModelConfig(
        name="text-embedding-ada-002",
        provider=ProviderType.OPENAI,
        metadata={"recommended_for": "embedding"},
    )

    manager.add_model(model1)
    manager.add_model(model2)
    manager.set_default_model("gpt-4")

    # Get model for specific task
    chat_model = manager.get_model_for_task("chat")
    assert chat_model is not None
    assert chat_model.name == "gpt-4"

    embedding_model = manager.get_model_for_task("embedding")
    assert embedding_model is not None
    assert embedding_model.name == "text-embedding-ada-002"

    # Get default for unknown task
    default = manager.get_model_for_task("unknown")
    assert default is not None
    assert default.name == "gpt-4"


def test_config_manager_clone():
    """Test configuration manager cloning."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)

    cloned = manager.clone()
    assert cloned.app_name == manager.app_name
    assert "gpt-4" in cloned.list_models()

    # Verify independence
    cloned.remove_model("gpt-4")
    assert "gpt-4" not in cloned.list_models()
    assert "gpt-4" in manager.list_models()


def test_config_manager_change_listeners():
    """Test configuration change listeners."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    changes = []

    def listener(config):
        changes.append(config)

    manager.add_change_listener(listener)

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)

    assert len(changes) > 0


# ============================================================================
# File I/O Tests
# ============================================================================


def test_save_and_load_config():
    """Test saving and loading configuration."""
    manager = ConfigManager(app_name="test", auto_load_env=False)

    # Add some configuration
    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    manager.add_model(model)

    provider = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
    )
    manager.add_provider(provider)

    # Save to file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        manager.save_to_file(temp_path)

        # Load from file
        manager2 = ConfigManager(
            app_name="test2",
            config_file=temp_path,
            auto_load_env=False,
        )

        assert "gpt-4" in manager2.list_models()
        assert ProviderType.OPENAI in manager2.list_providers()
    finally:
        Path(temp_path).unlink()


def test_load_config_from_file():
    """Test loading configuration from file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        config_data = {
            "app_name": "test",
            "default_model": "gpt-4",
            "providers": {},
            "models": {
                "gpt-4": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "max_tokens": 8000,
                    "metadata": {},
                }
            },
            "metadata": {},
        }
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config_from_file(temp_path)
        assert config.app_name == "test"
        assert "gpt-4" in config.models
    finally:
        Path(temp_path).unlink()


def test_save_config_to_file():
    """Test saving configuration to file."""
    config = AppConfig(
        app_name="test",
    )

    model = ModelConfig(name="gpt-4", provider=ProviderType.OPENAI)
    config.models["gpt-4"] = model

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        save_config_to_file(config, temp_path)

        # Verify file was created
        assert Path(temp_path).exists()

        # Load and verify
        with open(temp_path, "r") as f:
            data = json.load(f)

        assert data["app_name"] == "test"
        assert "gpt-4" in data["models"]
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Factory Function Tests
# ============================================================================


def test_create_config_manager():
    """Test config manager factory function."""
    manager = create_config_manager(
        app_name="test",
    )

    assert manager.app_name == "test"
    assert manager._config is not None


def test_create_model_config():
    """Test model config factory function."""
    config = create_model_config(
        name="gpt-4",
        provider="openai",
        max_tokens=8000,
    )

    assert config.name == "gpt-4"
    assert config.provider == ProviderType.OPENAI
    assert config.max_tokens == 8000


def test_create_provider_config():
    """Test provider config factory function."""
    config = create_provider_config(
        provider="openai",
        api_key_env_var="OPENAI_API_KEY",
    )

    assert config.provider == ProviderType.OPENAI
    assert config.api_key_env_var == "OPENAI_API_KEY"


# REMOVED: test_create_environment_config - environment functionality removed
# def test_create_environment_config():
#     """Test environment config factory function."""
#     config = create_environment_config(
#         environment="production",
#         debug=False,
#     )

#     assert config.environment == Environment.PRODUCTION
#     assert config.debug is False


# ============================================================================
# Default Configuration Tests
# ============================================================================


def test_get_default_openai_config():
    """Test default OpenAI configuration."""
    config = get_default_openai_config()

    assert config.provider == ProviderType.OPENAI
    assert config.api_key_env_var == "OPENAI_API_KEY"
    assert "gpt-4" in config.models


def test_get_default_anthropic_config():
    """Test default Anthropic configuration."""
    config = get_default_anthropic_config()

    assert config.provider == ProviderType.ANTHROPIC
    assert config.api_key_env_var == "ANTHROPIC_API_KEY"
    assert len(config.models) > 0


def test_get_default_google_config():
    """Test default Google configuration."""
    config = get_default_google_config()

    assert config.provider == ProviderType.GOOGLE
    assert config.api_key_env_var == "GOOGLE_API_KEY"
    assert len(config.models) > 0


# ============================================================================
# Utility Function Tests
# ============================================================================


# REMOVED: test_detect_environment - environment functionality removed
# def test_detect_environment():
#     """Test environment detection."""
# Test default
#     env = detect_environment()
#     assert isinstance(env, Environment)

# Test with env var
#     os.environ['ENVIRONMENT'] = 'production'
#     env = detect_environment()
#     assert env == Environment.PRODUCTION
#     del os.environ['ENVIRONMENT']


def test_validate_provider_credentials():
    """Test credential validation."""
    # Valid OpenAI key
    assert validate_provider_credentials(ProviderType.OPENAI, "sk-1234567890abcdef")

    # Valid Anthropic key
    assert validate_provider_credentials(
        ProviderType.ANTHROPIC, "sk-ant-1234567890abcdef"
    )

    # Invalid key (too short)
    assert not validate_provider_credentials(ProviderType.OPENAI, "sk-123")

    # Invalid key (empty)
    assert not validate_provider_credentials(ProviderType.OPENAI, "")


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_configuration_workflow():
    """Test complete configuration workflow."""
    # Create manager
    manager = create_config_manager(app_name="my_app")

    # Add providers with API keys
    openai_provider = get_default_openai_config()
    openai_provider.api_key = "sk-test-key-123"  # Add API key for validation
    manager.add_provider(openai_provider)

    anthropic_provider = get_default_anthropic_config()
    anthropic_provider.api_key = "sk-ant-test-key-456"  # Add API key for validation
    manager.add_provider(anthropic_provider)

    # Add models
    gpt4 = create_model_config(
        name="gpt-4",
        provider="openai",
        max_tokens=8000,
        temperature=0.7,
    )
    manager.add_model(gpt4)

    claude = create_model_config(
        name="claude-3-opus",
        provider="anthropic",
        max_tokens=4000,
        temperature=0.8,
    )
    manager.add_model(claude)

    # Set default
    manager.set_default_model("gpt-4")

    # Validate - check issues if validation fails
    issues = manager.validate()
    if issues:
        print(f"Validation issues: {issues}")
    assert manager.is_valid(), f"Validation failed with issues: {issues}"

    # Test retrieval
    default = manager.get_default_model()
    assert default is not None
    assert default.name == "gpt-4"

    # Test listing
    assert len(manager.list_models()) == 2
    assert len(manager.list_providers()) == 2


# REMOVED: test_multi_environment_config - environment functionality removed
# def test_multi_environment_config():
#     """Test configuration across multiple environments."""
# Development config
#     dev_manager = create_config_manager(
#         app_name="app",
#         environment="development",
#     )

#     dev_env_config = create_environment_config(
#         environment="development",
#         debug=True,
#         cache_enabled=True,
#     )
#     dev_manager.set_environment_config(dev_env_config)

# Production config
#     prod_manager = create_config_manager(
#         app_name="app",
#         environment="production",
#     )

#     prod_env_config = create_environment_config(
#         environment="production",
#         debug=False,
#         cache_enabled=True,
#         monitoring_enabled=True,
#     )
#     prod_manager.set_environment_config(prod_env_config)

# Verify differences
#     assert dev_manager.is_debug()
#     assert not prod_manager.is_debug()
#     assert prod_manager.is_production()
#     assert not dev_manager.is_production()
