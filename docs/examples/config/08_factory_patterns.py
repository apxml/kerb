"""Configuration Factory Patterns Example

This example demonstrates using factory functions for rapid configuration setup.

Main concepts:
- Configuration factory functions
- Quick setup patterns
- Template-based configurations
- Preset configurations for common use cases
- Configuration composition
"""

from kerb.config import (
    create_config_manager,
    create_model_config,
    create_provider_config,
    get_openai_config,
    get_anthropic_config,
    get_google_config
)
from kerb.config.enums import ProviderType


def create_chatbot_config():
    """Factory function for chatbot configuration."""
    config = create_config_manager(app_name="chatbot")
    
    # Add chatbot-optimized models
    chatbot_models = [
        create_model_config(
            name="chat-gpt-4",
            provider=ProviderType.OPENAI,
            temperature=0.7,
            max_tokens=2048,
            presence_penalty=0.3,
            frequency_penalty=0.3,
            metadata={"use_case": "conversational"}
        ),
        create_model_config(
            name="chat-claude",
            provider=ProviderType.ANTHROPIC,
            temperature=0.6,
            max_tokens=2048,
            metadata={"use_case": "conversational"}
        )
    ]
    
    for model in chatbot_models:
        config.add_model(model)
    
    config.set_default_model("chat-gpt-4")
    return config


def create_code_assistant_config():
    """Factory function for code assistant configuration."""
    config = create_config_manager(app_name="code_assistant")
    
    # Code-optimized models
    code_models = [
        create_model_config(
            name="code-gpt-4",
            provider=ProviderType.OPENAI,
            temperature=0.2,
            max_tokens=4096,
            metadata={"use_case": "code_generation", "language": "multi"}
        ),
        create_model_config(
            name="code-claude",
            provider=ProviderType.ANTHROPIC,
            temperature=0.3,
            max_tokens=4096,
            metadata={"use_case": "code_generation"}
        )
    ]
    
    for model in code_models:
        config.add_model(model)
    
    config.set_default_model("code-gpt-4")
    return config


def create_analytics_config():
    """Factory function for data analytics configuration."""
    config = create_config_manager(app_name="analytics")
    
    # Analytics-optimized models
    analytics_models = [
        create_model_config(
            name="analytics-gpt-4",
            provider=ProviderType.OPENAI,
            temperature=0.1,
            max_tokens=4096,
            metadata={"use_case": "data_analysis", "deterministic": True}
        ),
        create_model_config(
            name="analytics-gemini",
            provider=ProviderType.GOOGLE,
            temperature=0.2,
            max_tokens=4096,
            metadata={"use_case": "data_analysis"}
        )
    ]
    
    for model in analytics_models:
        config.add_model(model)
    
    config.set_default_model("analytics-gpt-4")
    return config


def create_content_generation_config():
    """Factory function for content generation configuration."""
    config = create_config_manager(app_name="content_generator")
    
    # Content-optimized models
    content_models = [
        create_model_config(
            name="creative-gpt-4",
            provider=ProviderType.OPENAI,
            temperature=0.9,
            max_tokens=4096,
            presence_penalty=0.6,
            frequency_penalty=0.3,
            metadata={"use_case": "creative_writing"}
        ),
        create_model_config(
            name="creative-claude",
            provider=ProviderType.ANTHROPIC,
            temperature=0.8,
            max_tokens=4096,
            presence_penalty=0.5,
            metadata={"use_case": "creative_writing"}
        )
    ]
    
    for model in content_models:
        config.add_model(model)
    
    config.set_default_model("creative-gpt-4")
    return config


def main():
    """Run configuration factory patterns example."""
    
    print("="*80)
    print("CONFIGURATION FACTORY PATTERNS EXAMPLE")
    print("="*80)
    
    # Step 1: Quick provider setup with factory functions
    print("\nStep 1: Quick Provider Setup")
    print("-"*80)
    
    config = create_config_manager(app_name="factory_demo")
    
    # Use convenience functions for providers
    openai = get_openai_config()
    anthropic = get_anthropic_config()
    google = get_google_config()
    
    config.add_provider(openai)
    config.add_provider(anthropic)
    config.add_provider(google)
    
    print("Providers added using factory functions:")
    print(f"  - OpenAI: {len(openai.models)} default models")
    print(f"  - Anthropic: {len(anthropic.models)} default models")
    print(f"  - Google: {len(google.models)} default models")
    
    # Step 2: Model factory patterns
    print("\nStep 2: Model Factory Patterns")
    print("-"*80)
    
    # Quick model creation
    models = [
        create_model_config("gpt-4", ProviderType.OPENAI, temperature=0.7),
        create_model_config("claude-3-opus", ProviderType.ANTHROPIC, temperature=0.6),
        create_model_config("gemini-pro", ProviderType.GOOGLE, temperature=0.5)
    ]
    
    for model in models:
        config.add_model(model)
        print(f"Created: {model.name} ({model.provider.value}) - temp: {model.temperature}")
    
    # Step 3: Use case-specific factories
    print("\nStep 3: Use Case-Specific Configuration Factories")
    print("-"*80)
    
    use_cases = {
        "Chatbot": create_chatbot_config(),
        "Code Assistant": create_code_assistant_config(),
        "Analytics": create_analytics_config(),
        "Content Generator": create_content_generation_config()
    }
    
    print("\nPre-configured use cases:")
    for use_case, use_config in use_cases.items():
        app_config = use_config.get_config()
        print(f"\n{use_case}:")
        print(f"  App name: {app_config.app_name}")
        print(f"  Models: {len(app_config.models)}")
        print(f"  Default: {app_config.default_model}")
        
        # Show first model details
        if app_config.models:
            first_model = list(app_config.models.values())[0]
            print(f"  Temperature: {first_model.temperature}")
            print(f"  Use case: {first_model.metadata.get('use_case', 'N/A')}")
    
    # Step 4: Configuration templates
    print("\nStep 4: Configuration Templates")
    print("-"*80)
    
    def create_from_template(template_name: str, app_name: str):
        """Create configuration from template."""
        templates = {
            "minimal": {
                "models": [
                    {"name": "gpt-3.5-turbo", "provider": ProviderType.OPENAI, "temperature": 0.7}
                ]
            },
            "balanced": {
                "models": [
                    {"name": "gpt-4", "provider": ProviderType.OPENAI, "temperature": 0.5},
                    {"name": "claude-3-sonnet", "provider": ProviderType.ANTHROPIC, "temperature": 0.5}
                ]
            },
            "comprehensive": {
                "models": [
                    {"name": "gpt-4", "provider": ProviderType.OPENAI, "temperature": 0.7},
                    {"name": "claude-3-opus", "provider": ProviderType.ANTHROPIC, "temperature": 0.6},
                    {"name": "gemini-pro", "provider": ProviderType.GOOGLE, "temperature": 0.5}
                ]
            }
        }
        
        template = templates.get(template_name, templates["minimal"])
        config = create_config_manager(app_name=app_name)
        
        for model_def in template["models"]:
            model = create_model_config(**model_def)
            config.add_model(model)
        
        if template["models"]:
            config.set_default_model(template["models"][0]["name"])
        
        return config
    
    templates = ["minimal", "balanced", "comprehensive"]
    print("Available templates:")
    
    for template in templates:
        template_config = create_from_template(template, f"{template}_app")
        models = template_config.list_models()
        print(f"\n  {template.capitalize()} template:")
        print(f"    Models: {', '.join(models)}")
    
    # Step 5: Composition patterns
    print("\nStep 5: Configuration Composition")
    print("-"*80)
    
    def compose_configs(*configs):
        """Compose multiple configurations into one."""
        composed = create_config_manager(app_name="composed_config")
        
        for cfg in configs:
            app_cfg = cfg.get_config()
            for model_name, model in app_cfg.models.items():
                # Rename to avoid conflicts
                new_name = f"{app_cfg.app_name}_{model_name}"
                model.name = new_name
                composed.add_model(model)
        
        return composed
    
    # Compose chatbot and code assistant configs
    chatbot = create_chatbot_config()
    code_assistant = create_code_assistant_config()
    
    combined = compose_configs(chatbot, code_assistant)
    
    print("Composed configuration from multiple sources:")
    print(f"  Total models: {len(combined.list_models())}")
    print(f"  Models: {', '.join(combined.list_models()[:3])}...")
    
    # Step 6: Builder pattern
    print("\nStep 6: Configuration Builder Pattern")
    print("-"*80)
    
    class ConfigBuilder:
        """Builder for fluent configuration creation."""
        
        def __init__(self, app_name: str):
            self.config = create_config_manager(app_name=app_name)
        
        def add_openai_model(self, name: str, **kwargs):
            model = create_model_config(name, ProviderType.OPENAI, **kwargs)
            self.config.add_model(model)
            return self
        
        def add_anthropic_model(self, name: str, **kwargs):
            model = create_model_config(name, ProviderType.ANTHROPIC, **kwargs)
            self.config.add_model(model)
            return self
        
        def with_default(self, model_name: str):
            self.config.set_default_model(model_name)
            return self
        
        def build(self):
            return self.config
    
    # Use builder pattern
    builder_config = (ConfigBuilder("builder_demo")
        .add_openai_model("gpt-4", temperature=0.7, max_tokens=4096)
        .add_anthropic_model("claude-3-opus", temperature=0.6, max_tokens=4096)
        .with_default("gpt-4")
        .build()
    )
    
    print("Configuration built with builder pattern:")
    print(f"  App: {builder_config.app_name}")
    print(f"  Models: {', '.join(builder_config.list_models())}")
    print(f"  Default: {builder_config.get_config().default_model}")
    
    # Step 7: Quick start configurations
    print("\nStep 7: Quick Start Configurations")
    print("-"*80)
    
    quick_configs = {
        "Development": create_config_manager("dev_app"),
        "Staging": create_config_manager("staging_app"),
        "Production": create_config_manager("prod_app")
    }
    
    for env, quick_config in quick_configs.items():
        # Add environment-appropriate model
        temp = 0.8 if env == "Development" else 0.5 if env == "Staging" else 0.2
        
        model = create_model_config(
            name=f"gpt-4-{env.lower()}",
            provider=ProviderType.OPENAI,
            temperature=temp,
            max_tokens=4096,
            metadata={"environment": env}
        )
        quick_config.add_model(model)
        
        print(f"{env:12} - Temperature: {temp} - Model: {model.name}")
    
    print("\n" + "="*80)
    print("Configuration factory patterns completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
