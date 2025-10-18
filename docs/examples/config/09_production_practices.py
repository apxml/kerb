"""Production Configuration Best Practices Example

This example demonstrates production-ready configuration management patterns.

Main concepts:
- Production security practices
- Configuration validation
- Error handling and fallbacks
- Monitoring and logging integration
- Configuration versioning
- Disaster recovery patterns
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

from kerb.config import ConfigManager, create_model_config, validate_credentials
from kerb.config.enums import ProviderType


def validate_production_config(config: ConfigManager) -> dict:
    """Validate configuration for production readiness."""
    issues = []
    warnings = []
    
    app_config = config.get_config()
    
    # Check 1: Default model set
    if not app_config.default_model:
        issues.append("No default model configured")
    
    # Check 2: Multiple models for fallback
    if len(app_config.models) < 2:
        warnings.append("Less than 2 models - consider adding fallback")
    
    # Check 3: API key configuration
    for model_name, model in app_config.models.items():
        if not model.api_key_env_var and not model.api_key:
            issues.append(f"Model '{model_name}' has no API key configured")
    
    # Check 4: Timeout configuration
    for model_name, model in app_config.models.items():
        if model.timeout < 30:
            warnings.append(f"Model '{model_name}' has low timeout: {model.timeout}s")
    
    # Check 5: Temperature settings
    for model_name, model in app_config.models.items():
        if model.temperature > 0.9:
            warnings.append(f"Model '{model_name}' has high temperature: {model.temperature}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }


def setup_production_config():
    """Create a production-ready configuration."""
    config = ConfigManager(app_name="production_app")
    
    # Primary production model
    primary = create_model_config(
        name="gpt-4-primary",
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY_PROD",
        temperature=0.3,
        max_tokens=4096,
        timeout=90.0,
        max_retries=5,
        metadata={
            "tier": "primary",
            "priority": 1,
            "cost_center": "PROD-001"
        }
    )
    
    # Fallback model
    fallback = create_model_config(
        name="gpt-4-fallback",
        provider=ProviderType.OPENAI,
        api_key_env_var="OPENAI_API_KEY_FALLBACK",
        temperature=0.3,
        max_tokens=4096,
        timeout=90.0,
        max_retries=3,
        metadata={
            "tier": "fallback",
            "priority": 2,
            "cost_center": "PROD-001"
        }
    )
    
    # Secondary provider fallback
    secondary = create_model_config(
        name="claude-secondary",
        provider=ProviderType.ANTHROPIC,
        api_key_env_var="ANTHROPIC_API_KEY_PROD",
        temperature=0.3,
        max_tokens=4096,
        timeout=90.0,
        max_retries=3,
        metadata={
            "tier": "secondary",
            "priority": 3,
            "cost_center": "PROD-002"
        }
    )
    
    config.add_model(primary)
    config.add_model(fallback)
    config.add_model(secondary)
    config.set_default_model("gpt-4-primary")
    
    return config


def main():
    """Run production configuration best practices example."""
    
    print("="*80)
    print("PRODUCTION CONFIGURATION BEST PRACTICES")
    print("="*80)
    
    # Step 1: Production setup
    print("\nStep 1: Production Configuration Setup")
    print("-"*80)
    
    prod_config = setup_production_config()
    
    print("Production configuration created:")
    print(f"  App: {prod_config.app_name}")
    print(f"  Models: {len(prod_config.list_models())}")
    print(f"  Default: {prod_config.get_config().default_model}")
    
    # Step 2: Configuration validation
    print("\nStep 2: Production Validation")
    print("-"*80)
    
    validation = validate_production_config(prod_config)
    
    print(f"Validation status: {'PASSED' if validation['valid'] else 'FAILED'}")
    
    if validation['issues']:
        print(f"\nCritical issues ({len(validation['issues'])}):")
        for issue in validation['issues']:
            print(f"  [ERROR] {issue}")
    
    if validation['warnings']:
        print(f"\nWarnings ({len(validation['warnings'])}):")
        for warning in validation['warnings']:
            print(f"  [WARN] {warning}")
    
    if validation['valid'] and not validation['warnings']:
        print("\nConfiguration is production-ready!")
    
    # Step 3: Secure configuration export
    print("\nStep 3: Secure Configuration Export")
    print("-"*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export without secrets (safe for version control)
        safe_path = os.path.join(temp_dir, "config_prod.json")
        prod_config.save_to_file(safe_path, include_secrets=False)
        
        print(f"Safe export (no secrets): {safe_path}")
        
        # Show that API keys are not included
        with open(safe_path, 'r') as f:
            safe_data = json.load(f)
            model_keys = [m.get('api_key') for m in safe_data.get('models', {}).values()]
            print(f"  API keys in export: {all(k is None for k in model_keys)}")
    
    # Step 4: Configuration versioning
    print("\nStep 4: Configuration Versioning")
    print("-"*80)
    
    version_info = {
        "version": "1.2.0",
        "timestamp": datetime.now().isoformat(),
        "environment": "production",
        "deployed_by": "deployment_system",
        "change_notes": "Updated primary model timeout to 90s"
    }
    
    # Add version metadata to config
    app_config = prod_config.get_config()
    app_config.metadata.update(version_info)
    
    print("Configuration version:")
    for key, value in version_info.items():
        print(f"  {key}: {value}")
    
    # Step 5: Error handling and fallbacks
    print("\nStep 5: Error Handling and Fallback Strategy")
    print("-"*80)
    
    def get_model_with_fallback(config: ConfigManager, preferred: str, fallbacks: list):
        """Get model with fallback chain."""
        models_to_try = [preferred] + fallbacks
        
        for model_name in models_to_try:
            model = config.get_model(model_name)
            if model:
                # Check if API key is available
                api_key = os.getenv(model.api_key_env_var) if model.api_key_env_var else model.api_key
                if api_key:
                    return model, model_name
        
        return None, None
    
    fallback_chain = ["gpt-4-primary", "gpt-4-fallback", "claude-secondary"]
    
    print("Fallback chain:")
    for i, model_name in enumerate(fallback_chain, 1):
        model = prod_config.get_model(model_name)
        if model:
            priority = model.metadata.get('priority', 'N/A')
            tier = model.metadata.get('tier', 'N/A')
            print(f"  {i}. {model_name} (tier: {tier}, priority: {priority})")
    
    # Step 6: Monitoring and logging
    print("\nStep 6: Monitoring and Logging Integration")
    print("-"*80)
    
    def log_config_access(model_name: str, success: bool):
        """Log configuration access for monitoring."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "success": success,
            "environment": "production"
        }
        print(f"[LOG] Model access: {json.dumps(log_entry)}")
    
    # Add logging listener
    def monitoring_listener(config):
        """Monitor configuration changes."""
        print(f"[MONITOR] Config changed - models: {len(config.models)}")
    
    prod_config.add_change_listener(monitoring_listener)
    
    print("Monitoring enabled:")
    print("  - Configuration change tracking")
    print("  - Model access logging")
    print("  - Performance metrics")
    
    # Step 7: Disaster recovery
    print("\nStep 7: Disaster Recovery Configuration")
    print("-"*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Backup configuration
        backup_path = os.path.join(temp_dir, f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        prod_config.save_to_file(backup_path, include_secrets=False)
        
        print(f"Configuration backup created: {Path(backup_path).name}")
        
        # Simulate disaster recovery
        print("\nDisaster recovery procedure:")
        print("  1. Load backup configuration")
        print("  2. Validate configuration")
        print("  3. Restore API keys from secure storage")
        print("  4. Verify all models are accessible")
        print("  5. Update monitoring systems")
    
    # Step 8: Health check
    print("\nStep 8: Configuration Health Check")
    print("-"*80)
    
    def health_check(config: ConfigManager):
        """Perform health check on configuration."""
        health = {
            "status": "healthy",
            "checks": []
        }
        
        app_config = config.get_config()
        
        # Check 1: Models configured
        if len(app_config.models) > 0:
            health["checks"].append({"name": "models_configured", "status": "pass"})
        else:
            health["status"] = "unhealthy"
            health["checks"].append({"name": "models_configured", "status": "fail"})
        
        # Check 2: Default model set
        if app_config.default_model:
            health["checks"].append({"name": "default_model", "status": "pass"})
        else:
            health["status"] = "degraded"
            health["checks"].append({"name": "default_model", "status": "warn"})
        
        # Check 3: API keys configured
        api_key_check = config.validate_api_keys()
        if any(api_key_check.values()):
            health["checks"].append({"name": "api_keys", "status": "pass"})
        else:
            health["status"] = "unhealthy"
            health["checks"].append({"name": "api_keys", "status": "fail"})
        
        return health
    
    health = health_check(prod_config)
    
    print(f"Health status: {health['status'].upper()}")
    print("\nHealth checks:")
    for check in health["checks"]:
        status_symbol = {"pass": "[+]", "warn": "[!]", "fail": "[-]"}
        symbol = status_symbol.get(check["status"], "[?]")
        print(f"  {symbol} {check['name']}: {check['status']}")
    
    # Step 9: Best practices summary
    print("\nStep 9: Production Best Practices Summary")
    print("-"*80)
    
    best_practices = [
        "Use environment variables for API keys",
        "Configure multiple models for fallback",
        "Set appropriate timeouts and retries",
        "Implement configuration validation",
        "Version your configurations",
        "Regular backup configurations",
        "Monitor configuration changes",
        "Implement health checks",
        "Use different configs per environment",
        "Never commit secrets to version control",
        "Use encryption_salt parameter for secrets (unique per app)",
        "Store encryption salt separately from encryption key",
        "Implement disaster recovery procedures",
        "Log configuration access and changes",
        "Set up alerting for config issues",
        "Regular security audits",
        "Document configuration decisions"
    ]
    
    print("\nProduction Configuration Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i:2}. {practice}")
    
    print("\n" + "="*80)
    print("Production configuration best practices completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
