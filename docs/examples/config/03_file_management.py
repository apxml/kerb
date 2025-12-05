"""
Configuration File Management Example
=====================================

=====================================

This example demonstrates saving and loading configurations from files.

Main concepts:
- Saving configuration to JSON files
- Loading configuration from files
- Exporting configuration (with/without secrets)
- Configuration versioning
- Environment-specific configs
"""

import os
import tempfile
from pathlib import Path

from kerb.config import (
    ConfigManager,
    load_config,
    save_config,
    create_model_config
)
from kerb.config.enums import ProviderType


def main():
    """Run configuration file management example."""
    
    print("="*80)
    print("CONFIGURATION FILE MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Create a temporary directory for examples
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Step 1: Create and configure
        print("\nStep 1: Create Configuration")
        print("-"*80)
        
        config = ConfigManager(app_name="file_config_demo")
        
        # Add some models
        models = [
            create_model_config(
                name="gpt-4-production",
                provider=ProviderType.OPENAI,
                max_tokens=4096,
                temperature=0.2,  # Lower temp for production
                api_key_env_var="OPENAI_API_KEY"
            ),
            create_model_config(
                name="gpt-4-development",
                provider=ProviderType.OPENAI,
                max_tokens=2048,
                temperature=0.7,  # Higher temp for experimentation
                api_key_env_var="OPENAI_API_KEY_DEV"
            )
        ]
        
        for model in models:
            config.add_model(model)
            print(f"Added model: {model.name}")
        
        config.set_default_model("gpt-4-production")
        
        # Step 2: Save configuration (without secrets)
        print("\nStep 2: Save Configuration (Safe Export)")
        print("-"*80)
        
        safe_config_path = os.path.join(temp_dir, "config_safe.json")
        config.save_to_file(safe_config_path, include_secrets=False)
        print(f"Saved configuration to: {safe_config_path}")
        print("API keys excluded for security")
        
        # Read and display
        with open(safe_config_path, 'r') as f:
            content = f.read()
            print(f"\nFile content preview (first 300 chars):\n{content[:300]}...")
        
        # Step 3: Save with secrets (development only)
        print("\nStep 3: Save Configuration (Development Export)")
        print("-"*80)
        
        dev_config_path = os.path.join(temp_dir, "config_dev.json")
        config.save_to_file(dev_config_path, include_secrets=True)
        print(f"Saved development config to: {dev_config_path}")
        print("WARNING: Contains API keys - use only in secure environments")
        
        # Step 4: Load configuration from file
        print("\nStep 4: Load Configuration from File")
        print("-"*80)
        
        loaded_config = load_config(safe_config_path)
        print(f"Loaded app: {loaded_config.app_name}")
        print(f"Default model: {loaded_config.default_model}")
        print(f"Models count: {len(loaded_config.models)}")
        
        # Step 5: Create environment-specific configs
        print("\nStep 5: Environment-Specific Configurations")
        print("-"*80)
        
        environments = ["development", "staging", "production"]
        
        for env in environments:
            env_config = ConfigManager(app_name=f"llm_app_{env}")
            
            # Different settings per environment
            if env == "development":
                temp = 0.8
                max_tokens = 2048
                timeout = 30.0
            elif env == "staging":
                temp = 0.5
                max_tokens = 4096
                timeout = 45.0
            else:  # production
                temp = 0.2
                max_tokens = 4096
                timeout = 60.0
            
            model = create_model_config(
                name=f"gpt-4-{env}",
                provider=ProviderType.OPENAI,
                temperature=temp,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key_env_var=f"OPENAI_API_KEY_{env.upper()}"
            )
            
            env_config.add_model(model)
            env_config.set_default_model(f"gpt-4-{env}")
            
            env_file = os.path.join(temp_dir, f"config_{env}.json")
            env_config.save_to_file(env_file, include_secrets=False)
            
            print(f"{env.capitalize():12} - temp={temp}, tokens={max_tokens}, timeout={timeout}s")
        
        # Step 6: Load specific environment
        print("\nStep 6: Load Environment-Specific Config")
        print("-"*80)
        
        prod_config_path = os.path.join(temp_dir, "config_production.json")
        production_config = ConfigManager(
            app_name="prod_app",
            config_file=prod_config_path
        )
        
        prod_model = production_config.get_model("gpt-4-production")
        if prod_model:
            print(f"Production model loaded: {prod_model.name}")
            print(f"  Temperature: {prod_model.temperature}")
            print(f"  Max tokens: {prod_model.max_tokens}")
            print(f"  Timeout: {prod_model.timeout}s")
        
        # Step 7: Configuration with metadata
        print("\nStep 7: Configuration with Metadata")
        print("-"*80)
        
        metadata_config = ConfigManager(app_name="metadata_demo")
        
        # Add model with metadata
        model_with_metadata = create_model_config(
            name="gpt-4-analytics",
            provider=ProviderType.OPENAI,
            max_tokens=4096,
            temperature=0.3,
            metadata={
                "use_case": "data_analysis",
                "department": "analytics",
                "cost_center": "CC-1234",
                "version": "1.0.0",
                "created_by": "team_lead",
                "tags": ["analytics", "production", "high-priority"]
            }
        )
        
        metadata_config.add_model(model_with_metadata)
        
        metadata_path = os.path.join(temp_dir, "config_metadata.json")
        metadata_config.save_to_file(metadata_path, include_secrets=False)
        
        retrieved = metadata_config.get_model("gpt-4-analytics")
        print(f"Model: {retrieved.name}")
        print(f"Metadata:")
        for key, value in retrieved.metadata.items():
            print(f"  {key}: {value}")
        
        # Step 8: List all created config files
        print("\nStep 8: Summary of Created Config Files")
        print("-"*80)
        
        config_files = list(Path(temp_dir).glob("*.json"))
        print(f"Total configuration files: {len(config_files)}")
        for file in sorted(config_files):
            size = file.stat().st_size
            print(f"  - {file.name} ({size} bytes)")
    
    print("\n" + "="*80)
    print("Configuration file management completed successfully!")
    print("Note: All files were created in temporary directory and cleaned up")
    print("="*80)


if __name__ == "__main__":
    main()
