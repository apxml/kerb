"""Configuration Change Listeners Example

This example demonstrates reactive configuration updates using change listeners.

Main concepts:
- Registering change listeners
- Reactive configuration updates
- Configuration history tracking
- Rollback mechanisms
- Event-driven configuration management
"""

from kerb.config import ConfigManager, create_model_config
from kerb.config.enums import ProviderType
from kerb.config.types import AppConfig


# Global tracking for demonstration
change_log = []


def log_config_change(config: AppConfig):
    """Log configuration changes for auditing."""
    change_info = {
        "app_name": config.app_name,
        "models_count": len(config.models),
        "default_model": config.default_model,
        "providers_count": len(config.providers)
    }
    change_log.append(change_info)
    print(f"[CHANGE LOGGED] Models: {change_info['models_count']}, Default: {change_info['default_model']}")


def notify_monitoring_system(config: AppConfig):
    """Simulate notifying a monitoring system of config changes."""
    print(f"[MONITORING] Configuration updated for: {config.app_name}")


def validate_config_on_change(config: AppConfig):
    """Validate configuration on every change."""
    if not config.models:
        print("[WARNING] No models configured!")
    elif not config.default_model:
        print("[WARNING] No default model set!")
    else:
        print(f"[VALIDATION] Config OK - {len(config.models)} models, default: {config.default_model}")


def main():
    """Run configuration change listeners example."""
    
    print("="*80)
    print("CONFIGURATION CHANGE LISTENERS EXAMPLE")
    print("="*80)
    
    # Step 1: Create config manager with listeners
    print("\nStep 1: Create ConfigManager with Change Listeners")
    print("-"*80)
    
    config = ConfigManager(app_name="reactive_config_demo")
    
    # Register multiple listeners
    config.add_change_listener(log_config_change)
    config.add_change_listener(notify_monitoring_system)
    config.add_change_listener(validate_config_on_change)
    
    print("Registered 3 change listeners:")
    print("  1. log_config_change - Audit logging")
    print("  2. notify_monitoring_system - System notifications")
    print("  3. validate_config_on_change - Configuration validation")
    
    # Step 2: Trigger changes and observe listeners
    print("\nStep 2: Trigger Configuration Changes")
    print("-"*80)
    
    print("\nAdding first model...")
    model1 = create_model_config(
        name="gpt-4",
        provider=ProviderType.OPENAI,
        temperature=0.7
    )
    config.add_model(model1)
    
    print("\nSetting default model...")
    config.set_default_model("gpt-4")
    
    print("\nAdding second model...")
    model2 = create_model_config(
        name="claude-3-opus",
        provider=ProviderType.ANTHROPIC,
        temperature=0.5
    )
    config.add_model(model2)
    
    print("\nUpdating model parameters...")
    gpt4_model = config.get_model("gpt-4")
    gpt4_model.temperature = 0.3
    gpt4_model.max_tokens = 2048
    config.add_model(gpt4_model)
    
    # Step 3: Configuration history tracking
    print("\nStep 3: Configuration History")
    print("-"*80)
    
    print(f"\nTotal changes logged: {len(change_log)}")
    print("\nChange history:")
    for i, change in enumerate(change_log, 1):
        print(f"  {i}. Models: {change['models_count']}, Default: {change['default_model']}")
    
    # Step 4: Rollback demonstration
    print("\nStep 4: Configuration Rollback")
    print("-"*80)
    
    print("\nCurrent configuration:")
    current = config.get_config()
    print(f"  Models: {len(current.models)}")
    print(f"  Default: {current.default_model}")
    
    # Access history
    if config._config_history:
        print(f"\nHistory size: {len(config._config_history)} previous states")
        
        # Simulate rollback
        print("\nSimulating rollback to previous state...")
        if len(config._config_history) >= 2:
            previous_state = config._config_history[-2]
            print(f"Previous state had {len(previous_state.models)} models")
    
    # Step 5: Custom validation listener
    print("\nStep 5: Custom Validation Listener")
    print("-"*80)
    
    def cost_optimization_validator(config: AppConfig):
        """Custom listener to validate cost optimization."""
        total_max_tokens = 0
        high_temp_models = []
        
        for model_name, model in config.models.items():
            total_max_tokens += model.max_tokens
            if model.temperature > 0.8:
                high_temp_models.append(model_name)
        
        print(f"[COST CHECK] Total max tokens across models: {total_max_tokens}")
        if high_temp_models:
            print(f"[COST CHECK] High temperature models: {', '.join(high_temp_models)}")
    
    config.add_change_listener(cost_optimization_validator)
    print("Added cost optimization validator")
    
    print("\nAdding high-temperature model to trigger validator...")
    creative_model = create_model_config(
        name="gpt-4-creative",
        provider=ProviderType.OPENAI,
        temperature=0.9,
        max_tokens=4096
    )
    config.add_model(creative_model)
    
    # Step 6: Conditional actions based on changes
    print("\nStep 6: Conditional Actions on Changes")
    print("-"*80)
    
    def production_readiness_check(config: AppConfig):
        """Check if configuration is production-ready."""
        issues = []
        
        if not config.default_model:
            issues.append("No default model set")
        
        if len(config.models) < 2:
            issues.append("Less than 2 models configured (no fallback)")
        
        # Check for API key configuration
        for model_name, model in config.models.items():
            if not model.api_key_env_var and not model.api_key:
                issues.append(f"Model {model_name} has no API key configured")
        
        if issues:
            print(f"[PRODUCTION CHECK] Issues found: {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("[PRODUCTION CHECK] Configuration is production-ready!")
    
    config.add_change_listener(production_readiness_check)
    print("Added production readiness checker")
    
    print("\nUpdating configuration...")
    gpt4_model = config.get_model("gpt-4")
    gpt4_model.api_key_env_var = "OPENAI_API_KEY"
    config.add_model(gpt4_model)
    
    claude_model = config.get_model("claude-3-opus")
    claude_model.api_key_env_var = "ANTHROPIC_API_KEY"
    config.add_model(claude_model)
    
    # Step 7: Listener for specific events
    print("\nStep 7: Event-Specific Listeners")
    print("-"*80)
    
    def model_threshold_alert(config: AppConfig):
        """Alert when model count exceeds threshold."""
        threshold = 5
        count = len(config.models)
        
        if count > threshold:
            print(f"[ALERT] Model count ({count}) exceeds threshold ({threshold})")
        else:
            print(f"[INFO] Model count: {count}/{threshold}")
    
    config.add_change_listener(model_threshold_alert)
    
    print("Adding models to test threshold...")
    for i in range(3):
        model = create_model_config(
            name=f"test-model-{i}",
            provider=ProviderType.OPENAI,
            temperature=0.5
        )
        config.add_model(model)
    
    # Step 8: Aggregated change notifications
    print("\nStep 8: Change Summary")
    print("-"*80)
    
    print(f"\nTotal configuration changes: {len(change_log)}")
    print(f"Current models: {len(config.list_models())}")
    print(f"Default model: {config.get_config().default_model}")
    
    # Analyze change patterns
    print("\nChange pattern analysis:")
    model_counts = [change['models_count'] for change in change_log]
    print(f"  Min models: {min(model_counts)}")
    print(f"  Max models: {max(model_counts)}")
    print(f"  Final count: {model_counts[-1]}")
    
    # Step 9: Cleanup and listener management
    print("\nStep 9: Listener Management")
    print("-"*80)
    
    print(f"\nTotal listeners registered: {len(config._change_listeners)}")
    print("Listeners can be used for:")
    print("  - Audit logging")
    print("  - Monitoring and alerting")
    print("  - Validation and compliance checks")
    print("  - Cost tracking")
    print("  - Production readiness verification")
    print("  - Integration with external systems")
    
    # Demonstrate listener impact
    print("\nFinal configuration update (all listeners active)...")
    gpt4_final = config.get_model("gpt-4")
    gpt4_final.temperature = 0.2
    config.add_model(gpt4_final)
    
    print("\n" + "="*80)
    print("Configuration change listeners completed successfully!")
    print(f"Total changes tracked: {len(change_log)}")
    print("="*80)


if __name__ == "__main__":
    main()
