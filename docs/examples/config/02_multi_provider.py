"""Multi-Provider Configuration Example

This example demonstrates managing multiple LLM providers in a single application.

Main concepts:
- Configuring multiple providers (OpenAI, Anthropic, Google)
- Provider-specific settings
- Provider factory functions
- Switching between providers
- Validating provider credentials
"""

from kerb.config import (
    ConfigManager,
    get_openai_config,
    get_anthropic_config,
    get_google_config,
    create_model_config
)
from kerb.config.enums import ProviderType


def main():
    """Run multi-provider configuration example."""
    
    print("="*80)
    print("MULTI-PROVIDER CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Create configuration manager
    config = ConfigManager(app_name="multi_provider_app")
    
    # Add multiple providers using convenience functions
    print("\nStep 1: Add Multiple Providers")
    print("-"*80)
    
    # OpenAI
    openai_config = get_openai_config()
    config.add_provider(openai_config)
    print(f"Added: {openai_config.provider.value}")
    print(f"  Models: {', '.join(openai_config.models)}")
    
    # Anthropic
    anthropic_config = get_anthropic_config()
    config.add_provider(anthropic_config)
    print(f"Added: {anthropic_config.provider.value}")
    print(f"  Models: {', '.join(anthropic_config.models)}")
    
    # Google
    google_config = get_google_config()
    config.add_provider(google_config)
    print(f"Added: {google_config.provider.value}")
    print(f"  Models: {', '.join(google_config.models)}")
    
    # Add model configurations for different providers
    print("\nStep 2: Configure Models for Each Provider")
    print("-"*80)
    
    models = [
        create_model_config(
            name="gpt-4",
            provider=ProviderType.OPENAI,
            max_tokens=4096,
            temperature=0.7,
            api_key_env_var="OPENAI_API_KEY"
        ),
        create_model_config(
            name="gpt-3.5-turbo",
            provider=ProviderType.OPENAI,
            max_tokens=4096,
            temperature=0.5,
            api_key_env_var="OPENAI_API_KEY"
        ),
        create_model_config(
            name="claude-3-opus",
            provider=ProviderType.ANTHROPIC,
            max_tokens=4096,
            temperature=0.7,
            api_key_env_var="ANTHROPIC_API_KEY"
        ),
        create_model_config(
            name="claude-3-sonnet",
            provider=ProviderType.ANTHROPIC,
            max_tokens=4096,
            temperature=0.5,
            api_key_env_var="ANTHROPIC_API_KEY"
        ),
        create_model_config(
            name="gemini-pro",
            provider=ProviderType.GOOGLE,
            max_tokens=2048,
            temperature=0.7,
            api_key_env_var="GOOGLE_API_KEY"
        )
    ]
    
    for model in models:
        config.add_model(model)
        print(f"Added: {model.name} ({model.provider.value})")
    
    # List models by provider
    print("\nStep 3: List Models by Provider")
    print("-"*80)
    
    providers = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]
    
    for provider in providers:
        provider_models = config.list_models(provider=provider)
        print(f"\n{provider.value.upper()} models:")
        for model_name in provider_models:
            model = config.get_model(model_name)
            print(f"  - {model_name}: temp={model.temperature}, max_tokens={model.max_tokens}")
    
    # Switch between providers
    print("\nStep 4: Provider Switching")
    print("-"*80)
    
    # Simulate switching from OpenAI to Anthropic for a task
    print("\nScenario: Need to switch from GPT-4 to Claude for a specific task")
    
    current_model = config.get_model("gpt-4")
    print(f"Current model: {current_model.name} ({current_model.provider.value})")
    
    # Find equivalent Claude model
    claude_model = config.get_model("claude-3-opus")
    print(f"Switching to: {claude_model.name} ({claude_model.provider.value})")
    print("Reason: Better performance on complex reasoning tasks")
    
    # Batch switch from one provider to another
    print("\nStep 5: Batch Provider Migration")
    print("-"*80)
    
    print("Migrating OpenAI models to Anthropic equivalents...")
    
    model_mapping = {
        "gpt-4": "claude-3-opus-migration",
        "gpt-3.5-turbo": "claude-3-sonnet-migration"
    }
    
    config.switch_provider(
        from_provider=ProviderType.OPENAI,
        to_provider=ProviderType.ANTHROPIC,
        model_mapping=model_mapping
    )
    
    print("Migration mapping:")
    for old, new in model_mapping.items():
        migrated_model = config.get_model(new)
        if migrated_model:
            print(f"  {old} -> {new} ({migrated_model.provider.value})")
    
    # Validate API keys
    print("\nStep 6: Validate Provider Credentials")
    print("-"*80)
    
    validation_results = config.validate_api_keys()
    print("Credential validation results:")
    for provider, is_valid in validation_results.items():
        status = "CONFIGURED" if is_valid else "MISSING"
        print(f"  {provider.value}: {status}")
    
    # Provider-specific configuration
    print("\nStep 7: Provider-Specific Settings")
    print("-"*80)
    
    # Update OpenAI provider with organization ID
    openai_provider = config._config.providers[ProviderType.OPENAI]
    openai_provider.organization = "org-example-123"
    print(f"OpenAI organization: {openai_provider.organization}")
    
    # Update rate limits
    anthropic_provider = config._config.providers[ProviderType.ANTHROPIC]
    anthropic_provider.rate_limit = 100  # requests per minute
    print(f"Anthropic rate limit: {anthropic_provider.rate_limit} req/min")
    
    print("\n" + "="*80)
    print("Multi-provider configuration completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
