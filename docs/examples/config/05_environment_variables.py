"""
Environment Variable Management Example
=======================================

This example demonstrates secure API key management using environment variables.

Main concepts:
- Loading API keys from environment variables
- Multiple environment configurations
- Secure credential management
- Environment variable validation
- Development vs production key management
"""

import os
from kerb.config import ConfigManager, create_model_config, validate_credentials
from kerb.config.enums import ProviderType


def setup_mock_env_vars():
    """Set up mock environment variables for demonstration."""
    # Mock API keys for demonstration (not real keys)
    os.environ["OPENAI_API_KEY"] = "sk-mock-openai-key-123456789"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-mock-anthropic-key-123456789"
    os.environ["GOOGLE_API_KEY"] = "mock-google-key-123456789"
    os.environ["OPENAI_API_KEY_DEV"] = "sk-mock-openai-dev-key-123456789"
    os.environ["OPENAI_API_KEY_STAGING"] = "sk-mock-openai-staging-key-123456789"
    os.environ["OPENAI_API_KEY_PROD"] = "sk-mock-openai-prod-key-123456789"


def main():
    """Run environment variable management example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("ENVIRONMENT VARIABLE MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Set up demonstration environment variables
    setup_mock_env_vars()
    
    # Step 1: Auto-load from environment
    print("\nStep 1: Auto-load Configuration from Environment")
    print("-"*80)
    
    config = ConfigManager(
        app_name="env_config_demo",
        auto_load_env=True
    )
    
    print("ConfigManager created with auto_load_env=True")
    print("Will automatically discover API keys from environment variables")
    
    # Step 2: Configure models with environment variables
    print("\nStep 2: Configure Models with Environment Variables")
    print("-"*80)
    
    models = [
        {
            "name": "gpt-4-prod",
            "provider": ProviderType.OPENAI,
            "env_var": "OPENAI_API_KEY",
            "description": "Production OpenAI model"
        },
        {
            "name": "claude-prod",
            "provider": ProviderType.ANTHROPIC,
            "env_var": "ANTHROPIC_API_KEY",
            "description": "Production Anthropic model"
        },
        {
            "name": "gemini-prod",
            "provider": ProviderType.GOOGLE,
            "env_var": "GOOGLE_API_KEY",
            "description": "Production Google model"
        }
    ]
    
    for model_info in models:
        model = create_model_config(
            name=model_info["name"],
            provider=model_info["provider"],
            api_key_env_var=model_info["env_var"],
            max_tokens=4096,
            temperature=0.7
        )
        config.add_model(model)
        
        # Check if env var is set
        env_set = os.getenv(model_info["env_var"]) is not None
        status = "FOUND" if env_set else "MISSING"
        
        print(f"{model_info['description']:30} - {model_info['env_var']:25} [{status}]")
    
    # Step 3: Environment-specific configurations
    print("\nStep 3: Environment-Specific Key Management")
    print("-"*80)
    
    environments = {
        "development": "OPENAI_API_KEY_DEV",
        "staging": "OPENAI_API_KEY_STAGING",
        "production": "OPENAI_API_KEY_PROD"
    }
    
    print("Multi-environment API key setup:")
    for env_name, env_var in environments.items():
        model = create_model_config(
            name=f"gpt-4-{env_name}",
            provider=ProviderType.OPENAI,
            api_key_env_var=env_var,
            max_tokens=4096
        )
        config.add_model(model)
        
        key_exists = os.getenv(env_var) is not None
        print(f"  {env_name:12} -> {env_var:30} [{'SET' if key_exists else 'NOT SET'}]")
    
    # Step 4: Validate environment variables
    print("\nStep 4: Environment Variable Validation")
    print("-"*80)
    
    print("Checking all configured environment variables:")
    
    all_env_vars = set()
    for model_name in config.list_models():
        model = config.get_model(model_name)
        if model.api_key_env_var:
            all_env_vars.add(model.api_key_env_var)
    
    for env_var in sorted(all_env_vars):
        value = os.getenv(env_var)
        if value:
            # Show masked value for security
            masked = value[:5] + "*" * (len(value) - 8) + value[-3:] if len(value) > 8 else "***"
            print(f"  {env_var:30} = {masked}")
        else:
            print(f"  {env_var:30} = NOT SET")
    
    # Step 5: API key retrieval and validation
    print("\nStep 5: API Key Retrieval and Validation")
    print("-"*80)
    
    test_models = ["gpt-4-prod", "claude-prod", "gemini-prod"]
    
    for model_name in test_models:
        model = config.get_model(model_name)
        if model:
            # Resolve env var to actual key
            api_key = os.getenv(model.api_key_env_var) if model.api_key_env_var else model.api_key
            
            print(f"\n{model_name}:")
            print(f"  Provider: {model.provider.value}")
            print(f"  Env var: {model.api_key_env_var}")
            
            if api_key:
                # Validate key format
                is_valid = validate_credentials(model.provider, api_key)
                print(f"  Key status: {'VALID' if is_valid else 'INVALID FORMAT'}")
                print(f"  Key preview: {api_key[:8]}...")
            else:
                print(f"  Key status: NOT FOUND")
    
    # Step 6: Fallback configuration
    print("\nStep 6: Fallback Configuration Strategy")
    print("-"*80)
    
    print("\nImplementing fallback chain: prod -> staging -> dev")
    
    # Try production first, fall back if not available
    fallback_chain = [
        ("OPENAI_API_KEY_PROD", "Production"),
        ("OPENAI_API_KEY_STAGING", "Staging"),
        ("OPENAI_API_KEY_DEV", "Development"),
        ("OPENAI_API_KEY", "Default")
    ]
    
    active_key = None
    active_env = None
    
    for env_var, env_name in fallback_chain:
        if os.getenv(env_var):
            active_key = env_var
            active_env = env_name
            print(f"Found API key in: {env_name} ({env_var})")
            break
    
    if active_key:
        fallback_model = create_model_config(
            name="gpt-4-fallback",
            provider=ProviderType.OPENAI,
            api_key_env_var=active_key,
            metadata={"environment": active_env, "fallback": True}
        )
        config.add_model(fallback_model)
        print(f"Using: {active_key}")
    else:
        print("No API key found in fallback chain")
    
    # Step 7: Secure practices demonstration
    print("\nStep 7: Secure Configuration Practices")
    print("-"*80)
    
    print("\nBest Practices:")
    print("1. Store API keys in environment variables (NEVER in code)")
    print("2. Use different keys for different environments")
    print("3. Rotate keys regularly")
    print("4. Never commit .env files to version control")
    print("5. Use secret management services in production")
    
    print("\nEnvironment variable naming conventions:")
    conventions = [
        ("PROVIDER_API_KEY", "Basic format"),
        ("PROVIDER_API_KEY_ENV", "Environment-specific"),
        ("APP_PROVIDER_API_KEY", "Application-scoped"),
        ("PROVIDER_API_KEY_SERVICE", "Service-specific")
    ]
    
    for pattern, description in conventions:
        print(f"  {pattern:30} - {description}")
    
    # Step 8: Configuration validation
    print("\nStep 8: Complete Configuration Validation")
    print("-"*80)
    
    validation_results = config.validate_api_keys()
    
    print("\nProvider validation results:")
    for provider, is_valid in validation_results.items():
        status = "CONFIGURED" if is_valid else "MISSING"
        symbol = "[+]" if is_valid else "[-]"
        print(f"  {symbol} {provider.value:15} : {status}")
    
    # Count configured vs missing
    configured = sum(1 for v in validation_results.values() if v)
    total = len(validation_results)
    
    print(f"\nSummary: {configured}/{total} providers configured")
    
    # Step 9: Loading from .env file (simulation)
    print("\nStep 9: .env File Integration Pattern")
    print("-"*80)
    
    print("\nExample .env file structure:")
    print("""
# Production API Keys
OPENAI_API_KEY=sk-prod-...
ANTHROPIC_API_KEY=sk-ant-prod-...
GOOGLE_API_KEY=...

# Development API Keys
OPENAI_API_KEY_DEV=sk-dev-...
ANTHROPIC_API_KEY_DEV=sk-ant-dev-...

# Optional: Service-specific
CHATBOT_OPENAI_KEY=sk-...
ANALYTICS_OPENAI_KEY=sk-...
    """)
    
    print("\nTo use .env files, install python-dotenv:")
    print("  pip install python-dotenv")
    print("\nThen load in your application:")
    print("  from dotenv import load_dotenv")
    print("  load_dotenv()  # Load .env file")
    print("  config = ConfigManager(auto_load_env=True)")
    
    print("\n" + "="*80)
    print("Environment variable management completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
