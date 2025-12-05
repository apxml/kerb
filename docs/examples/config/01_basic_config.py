"""
Basic Configuration Example
===========================

===========================

This example demonstrates fundamental configuration management for LLM applications.

Main concepts:
- Creating a ConfigManager instance
- Setting up model configurations
- Managing provider settings
- Loading configuration from environment variables
"""

from kerb.config import ConfigManager, ModelConfig, ProviderConfig
from kerb.config.enums import ProviderType


def main():
    """Run basic configuration example."""
    
    print("="*80)
    print("BASIC CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Create a configuration manager
    config = ConfigManager(
        app_name="my_llm_app",
        auto_load_env=True
    )
    
    print("\nStep 1: Create Configuration Manager")
    print("-"*80)
    print(f"App name: {config.app_name}")
    
    # Add a provider configuration
    print("\nStep 2: Add Provider Configuration")
    print("-"*80)
    
    openai_provider = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        timeout=60.0,
        max_retries=3
    )
    
    config.add_provider(openai_provider)
    print(f"Added provider: {openai_provider.provider.value}")
    print(f"API key from env: {openai_provider.api_key_env_var}")
    
    # Add a model configuration
    print("\nStep 3: Add Model Configuration")
    print("-"*80)
    
    gpt4_config = ModelConfig(
        name="gpt-4",
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        timeout=60.0
    )
    
    config.add_model(gpt4_config)
    print(f"Added model: {gpt4_config.name}")
    print(f"Provider: {gpt4_config.provider.value}")
    print(f"Max tokens: {gpt4_config.max_tokens}")
    print(f"Temperature: {gpt4_config.temperature}")
    
    # Set default model
    print("\nStep 4: Set Default Model")
    print("-"*80)
    
    config.set_default_model("gpt-4")
    app_config = config.get_config()
    print(f"Default model: {app_config.default_model}")
    
    # Retrieve model configuration
    print("\nStep 5: Retrieve Model Configuration")
    print("-"*80)
    
    retrieved_model = config.get_model("gpt-4")
    if retrieved_model:
        print(f"Model name: {retrieved_model.name}")
        print(f"Provider: {retrieved_model.provider.value}")
        print(f"Temperature: {retrieved_model.temperature}")
        print(f"Max tokens: {retrieved_model.max_tokens}")
    
    # List all models
    print("\nStep 6: List All Models")
    print("-"*80)
    
    all_models = config.list_models()
    print(f"Total models configured: {len(all_models)}")
    for model_name in all_models:
        model = config.get_model(model_name)
        print(f"  - {model_name} ({model.provider.value})")
    
    # Update model configuration
    print("\nStep 7: Update Model Configuration")
    print("-"*80)
    
    # Update by replacing the model
    updated_config = config.get_model("gpt-4")
    updated_config.temperature = 0.3
    updated_config.max_tokens = 2048
    config.add_model(updated_config)  # Re-add to update
    
    updated_model = config.get_model("gpt-4")
    print(f"Updated temperature: {updated_model.temperature}")
    print(f"Updated max_tokens: {updated_model.max_tokens}")
    
    print("\n" + "="*80)
    print("Configuration management completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
