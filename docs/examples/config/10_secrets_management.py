"""
Secrets Management Example
==========================

This example demonstrates secure secrets management in ConfigManager.

Main concepts:
- Encrypted secrets storage
- Custom encryption keys and salts
- Secrets persistence and retrieval
- Security best practices
"""

import os
import tempfile
from pathlib import Path

from kerb.config import ConfigManager


def basic_secrets_usage():
    """Basic secrets management without custom encryption."""
    print("\n" + "="*60)
    print("Basic Secrets Management")
    print("="*60)
    
    # Create config manager with auto-generated encryption key
    config = ConfigManager(app_name="secrets_demo")
    
    # Store secrets
    config.set_secret("api_key", "sk-test-key-12345")
    config.set_secret("database_password", "super_secure_password")
    config.set_secret("webhook_secret", "whsec_abc123")
    
    # Retrieve secrets
    api_key = config.get_secret("api_key")
    db_pass = config.get_secret("database_password")
    
    print(f"API Key retrieved: {api_key[:10]}...")
    print(f"DB Password retrieved: {db_pass[:10]}...")
    
    # List available secrets (keys only, not values)
    print(f"\nStored secrets: {config.list_secret_keys()}")
    
    # Check if secret exists
    has_api_key = config.has_secret("api_key")
    has_missing = config.has_secret("missing_key")
    print(f"\nHas 'api_key': {has_api_key}")
    print(f"Has 'missing_key': {has_missing}")
    
    # Remove a secret
    removed = config.remove_secret("webhook_secret")
    print(f"\nRemoved 'webhook_secret': {removed}")
    print(f"Remaining secrets: {config.list_secret_keys()}")


def custom_encryption_key():
    """Use a custom encryption key for secrets."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "="*60)
    print("Custom Encryption Key")
    print("="*60)
    
    # Provide your own encryption key (e.g., from environment or key vault)
    # In production, load this from a secure source
    custom_key = os.environ.get("ENCRYPTION_KEY", "my-secure-master-key-2024")
    
    config = ConfigManager(
        app_name="secure_app",
        encryption_key=custom_key
    )
    
    # Store secrets with custom encryption
    config.set_secret("prod_api_key", "sk-prod-key-67890")
    config.set_secret("oauth_secret", "oauth_abc_xyz_123")
    
    print(f"Stored {len(config.list_secret_keys())} secrets with custom encryption")
    
    # Retrieve with same key
    api_key = config.get_secret("prod_api_key")
    print(f"Retrieved secret: {api_key[:15]}...")
    
    # Important: The same encryption key is needed to decrypt
    print("\n⚠️  Remember: You need the same encryption_key to decrypt later!")



# %%
# Custom Encryption Salt
# ----------------------

def custom_encryption_salt():
    """Use a custom salt for key derivation"""
    print("\n" + "="*60)
    print("Custom Encryption Salt")
    print("="*60)
    
    # Generate or load your application-specific salt
    # This should be unique per application but consistent across sessions
    # In production, generate once and store securely
    
    import secrets
    
    # Option 1: Generate a new salt (store this securely!)
    app_salt = secrets.token_bytes(16)
    print(f"Generated salt: {app_salt.hex()}")
    
    # Option 2: Use a consistent salt from environment or config
    # app_salt = bytes.fromhex(os.environ.get("APP_SALT"))
    
    encryption_key = "my-application-password"
    
    config = ConfigManager(
        app_name="salted_app",
        encryption_key=encryption_key,
        encryption_salt=app_salt  # Custom salt for this app
    )
    
    # Store secrets
    config.set_secret("service_token", "token_abc123xyz")
    
    # The salt is stored internally for this session
    print(f"Encryption salt stored: {config._encryption_salt.hex()}")
    
    # Retrieve secret
    token = config.get_secret("service_token")
    print(f"Retrieved token: {token}")
    
    print("\n✅ Best Practice: Each application should use its own salt")
    print("   This ensures secrets can't be decrypted across different apps")
    print("   even if they share the same encryption key.")


def production_secret_workflow():
    """Production-ready secret management workflow."""
    print("\n" + "="*60)
    print("Production Secret Management Workflow")
    print("="*60)
    
    # Step 1: Load encryption credentials from secure source
    # In production: AWS Secrets Manager, HashiCorp Vault, Azure Key Vault, etc.
    encryption_key = os.environ.get("MASTER_ENCRYPTION_KEY", "prod-key-2024")
    
    # Load or generate persistent salt (should be stored securely)
    salt_hex = os.environ.get("APP_ENCRYPTION_SALT")
    if salt_hex:
        app_salt = bytes.fromhex(salt_hex)
        print("✅ Loaded existing salt from environment")
    else:
        import secrets
        app_salt = secrets.token_bytes(16)
        print(f"⚠️  Generated new salt: {app_salt.hex()}")
        print("   Save this to environment: APP_ENCRYPTION_SALT")
    
    # Step 2: Initialize config with encryption
    config = ConfigManager(
        app_name="production_app",
        encryption_key=encryption_key,
        encryption_salt=app_salt,
        auto_load_env=True
    )
    
    # Step 3: Store runtime secrets (never hardcode these!)
    # Load from secure sources
    openai_key = os.environ.get("OPENAI_API_KEY", "sk-test-key")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-test")
    
    config.set_secret("openai_api_key", openai_key)
    config.set_secret("anthropic_api_key", anthropic_key)
    
    print(f"\n✅ Stored {len(config.list_secret_keys())} secrets securely")
    
    # Step 4: Use secrets at runtime

# %%
# Make Api Call
# -------------

    def make_api_call():
        api_key = config.get_secret("openai_api_key")
        # Use api_key for API calls
        print(f"   Using API key: {api_key[:10]}...")
    
    make_api_call()
    
    # Step 5: Clear secrets when done (security best practice)
    print("\n🔒 Clearing all secrets from memory...")
    config.clear_secrets()
    print(f"   Remaining secrets: {config.list_secret_keys()}")



# %%
# Persistence Pattern
# -------------------

def persistence_pattern():
    """Pattern for persisting salt across application restarts."""
    print("\n" + "="*60)
    print("Salt Persistence Pattern")
    print("="*60)
    
    import secrets
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        salt_file = Path(tmpdir) / "app_salt.json"
        
        # First run: Generate and save salt
        if not salt_file.exists():
            print("First run: Generating new salt...")
            app_salt = secrets.token_bytes(16)
            
            # Save salt to secure location (in production: use key vault)
            salt_data = {
                "salt": app_salt.hex(),
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            salt_file.write_text(json.dumps(salt_data, indent=2))
            print(f"✅ Salt saved to: {salt_file}")
        else:
            # Subsequent runs: Load existing salt
            print("Loading existing salt...")
            salt_data = json.loads(salt_file.read_text())
            app_salt = bytes.fromhex(salt_data["salt"])
            print("✅ Salt loaded from storage")
        
        # Use the consistent salt
        config = ConfigManager(
            app_name="persistent_app",
            encryption_key="my-key",
            encryption_salt=app_salt
        )
        
        config.set_secret("test_secret", "test_value")
        
        print(f"\nSalt (hex): {app_salt.hex()}")
        print(f"Secret stored: {config.get_secret('test_secret')}")
        
        print("\n✅ Same salt used across restarts ensures consistent encryption")


def security_best_practices():
    """Display security best practices for secrets management."""
    print("\n" + "="*60)
    print("Security Best Practices")
    print("="*60)
    
    practices = [
        "1. Never hardcode encryption keys or salts in source code",
        "2. Use environment variables or key management services (KMS)",
        "3. Generate unique salt per application (use encryption_salt parameter)",
        "4. Store salt securely but separately from encryption key",
        "5. Rotate encryption keys periodically",
        "6. Clear secrets from memory when no longer needed (clear_secrets())",
        "7. Use cryptography library for proper encryption (auto-installed)",
        "8. Never commit secrets to version control",
        "9. Use separate keys for dev/staging/production",
        "10. Monitor and audit secret access",
    ]
    
    for practice in practices:
        print(f"  {practice}")
    
    print("\n" + "="*60)
    print("Key Management Service Examples:")
    print("="*60)
    print("  • AWS Secrets Manager: boto3.client('secretsmanager')")
    print("  • Azure Key Vault: azure.keyvault.secrets")
    print("  • GCP Secret Manager: google.cloud.secretmanager")
    print("  • HashiCorp Vault: hvac.Client()")
    print("  • Environment Variables: os.environ.get()")



# %%
# Main
# ----

def main():
    """Run all examples."""
    print("="*60)
    print("Secrets Management Examples")
    print("="*60)
    
    # Basic usage
    basic_secrets_usage()
    
    # Custom encryption key
    custom_encryption_key()
    
    # Custom salt (IMPORTANT for libraries!)
    custom_encryption_salt()
    
    # Production workflow
    production_secret_workflow()
    
    # Persistence pattern
    persistence_pattern()
    
    # Best practices
    security_best_practices()
    
    print("\n" + "="*60)
    print("✅ All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
