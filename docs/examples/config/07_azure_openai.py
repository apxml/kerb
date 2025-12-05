"""
Azure OpenAI Configuration Example
==================================

==================================

This example demonstrates Azure-specific LLM configuration.

Main concepts:
- Azure OpenAI endpoint configuration
- Deployment-specific settings
- API version management
- Azure-specific authentication
- Region-based deployments
"""

from kerb.config import ConfigManager, create_model_config, ProviderConfig
from kerb.config.enums import ProviderType


def main():
    """Run Azure OpenAI configuration example."""
    
    print("="*80)
    print("AZURE OPENAI CONFIGURATION EXAMPLE")
    print("="*80)
    
    config = ConfigManager(app_name="azure_openai_demo")
    
    # Step 1: Configure Azure OpenAI provider
    print("\nStep 1: Azure OpenAI Provider Configuration")
    print("-"*80)
    
    azure_provider = ProviderConfig(
        provider=ProviderType.AZURE_OPENAI,
        api_key_env_var="AZURE_OPENAI_API_KEY",
        base_url="https://your-resource-name.openai.azure.com/",
        timeout=90.0,
        max_retries=3,
        metadata={
            "region": "eastus",
            "subscription_id": "your-subscription-id",
            "resource_group": "your-resource-group"
        }
    )
    
    config.add_provider(azure_provider)
    print(f"Provider: {azure_provider.provider.value}")
    print(f"Base URL: {azure_provider.base_url}")
    print(f"Region: {azure_provider.metadata['region']}")
    
    # Step 2: Azure deployment configurations
    print("\nStep 2: Azure Deployment Configurations")
    print("-"*80)
    
    # GPT-4 deployment
    gpt4_deployment = create_model_config(
        name="gpt-4-azure",
        provider=ProviderType.AZURE_OPENAI,
        deployment_name="gpt-4-deployment",  # Azure deployment name
        api_version="2024-02-15-preview",
        endpoint="https://your-resource-name.openai.azure.com/",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.7,
        metadata={
            "deployment_id": "deploy-001",
            "model_version": "0613",
            "capacity": "high"
        }
    )
    config.add_model(gpt4_deployment)
    
    print(f"Deployment: {gpt4_deployment.name}")
    print(f"  Deployment name: {gpt4_deployment.deployment_name}")
    print(f"  API version: {gpt4_deployment.api_version}")
    print(f"  Model version: {gpt4_deployment.metadata['model_version']}")
    
    # GPT-3.5 Turbo deployment
    gpt35_deployment = create_model_config(
        name="gpt-35-turbo-azure",
        provider=ProviderType.AZURE_OPENAI,
        deployment_name="gpt-35-turbo-deployment",
        api_version="2024-02-15-preview",
        endpoint="https://your-resource-name.openai.azure.com/",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        max_tokens=4096,
        temperature=0.5,
        metadata={
            "deployment_id": "deploy-002",
            "model_version": "0613",
            "capacity": "standard"
        }
    )
    config.add_model(gpt35_deployment)
    
    print(f"\nDeployment: {gpt35_deployment.name}")
    print(f"  Deployment name: {gpt35_deployment.deployment_name}")
    print(f"  Capacity: {gpt35_deployment.metadata['capacity']}")
    
    # Step 3: Multi-region Azure deployments
    print("\nStep 3: Multi-Region Azure Deployments")
    print("-"*80)
    
    regions = [
        {
            "region": "eastus",
            "endpoint": "https://eastus-resource.openai.azure.com/",
            "deployment": "gpt-4-eastus"
        },
        {
            "region": "westeurope",
            "endpoint": "https://westeu-resource.openai.azure.com/",
            "deployment": "gpt-4-westeu"
        },
        {
            "region": "japaneast",
            "endpoint": "https://japaneast-resource.openai.azure.com/",
            "deployment": "gpt-4-japaneast"
        }
    ]
    
    print("Configuring regional deployments:")
    for region_info in regions:
        model = create_model_config(
            name=f"gpt-4-{region_info['region']}",
            provider=ProviderType.AZURE_OPENAI,
            deployment_name=region_info['deployment'],
            endpoint=region_info['endpoint'],
            api_version="2024-02-15-preview",
            api_key_env_var=f"AZURE_OPENAI_API_KEY_{region_info['region'].upper()}",
            max_tokens=4096,
            temperature=0.7,
            metadata={
                "region": region_info['region'],
                "purpose": "regional_failover"
            }
        )
        config.add_model(model)
        print(f"  {region_info['region']:15} - {region_info['deployment']}")
    
    # Step 4: Azure API version management
    print("\nStep 4: Azure API Version Management")
    print("-"*80)
    
    api_versions = [
        "2023-05-15",
        "2023-12-01-preview",
        "2024-02-15-preview"
    ]
    
    print("Available Azure OpenAI API versions:")
    for i, version in enumerate(api_versions, 1):
        print(f"  {i}. {version}")
    
    # Use latest version
    latest_version = api_versions[-1]
    print(f"\nUsing latest API version: {latest_version}")
    
    azure_model = config.get_model("gpt-4-azure")
    azure_model.api_version = latest_version
    config.add_model(azure_model)
    
    # Step 5: Azure-specific parameters
    print("\nStep 5: Azure-Specific Configuration")
    print("-"*80)
    
    azure_specific_model = create_model_config(
        name="gpt-4-azure-advanced",
        provider=ProviderType.AZURE_OPENAI,
        deployment_name="gpt-4-advanced-deployment",
        api_version="2024-02-15-preview",
        endpoint="https://your-resource-name.openai.azure.com/",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        max_tokens=8192,
        temperature=0.7,
        timeout=120.0,
        max_retries=5,
        metadata={
            "deployment_type": "provisioned_throughput",
            "tokens_per_minute": 100000,
            "priority": "high",
            "cost_tracking": True,
            "quota_limit": 1000000
        }
    )
    config.add_model(azure_specific_model)
    
    print("Advanced Azure configuration:")
    print(f"  Deployment type: {azure_specific_model.metadata['deployment_type']}")
    print(f"  Throughput: {azure_specific_model.metadata['tokens_per_minute']} TPM")
    print(f"  Priority: {azure_specific_model.metadata['priority']}")
    print(f"  Quota limit: {azure_specific_model.metadata['quota_limit']}")
    
    # Step 6: Environment-based Azure configuration
    print("\nStep 6: Environment-Based Azure Configuration")
    print("-"*80)
    
    environments = {
        "development": {
            "endpoint": "https://dev-resource.openai.azure.com/",
            "deployment": "gpt-4-dev",
            "capacity": "low"
        },
        "staging": {
            "endpoint": "https://staging-resource.openai.azure.com/",
            "deployment": "gpt-4-staging",
            "capacity": "medium"
        },
        "production": {
            "endpoint": "https://prod-resource.openai.azure.com/",
            "deployment": "gpt-4-prod",
            "capacity": "high"
        }
    }
    
    print("Environment configurations:")
    for env, settings in environments.items():
        model = create_model_config(
            name=f"gpt-4-{env}",
            provider=ProviderType.AZURE_OPENAI,
            deployment_name=settings['deployment'],
            endpoint=settings['endpoint'],
            api_version="2024-02-15-preview",
            api_key_env_var=f"AZURE_OPENAI_API_KEY_{env.upper()}",
            max_tokens=4096,
            metadata={
                "environment": env,
                "capacity": settings['capacity']
            }
        )
        config.add_model(model)
        print(f"  {env:12} - Capacity: {settings['capacity']:6} - {settings['deployment']}")
    
    # Step 7: Azure managed identity (simulation)
    print("\nStep 7: Azure Managed Identity Configuration")
    print("-"*80)
    
    managed_identity_model = create_model_config(
        name="gpt-4-managed-identity",
        provider=ProviderType.AZURE_OPENAI,
        deployment_name="gpt-4-mi-deployment",
        api_version="2024-02-15-preview",
        endpoint="https://your-resource-name.openai.azure.com/",
        metadata={
            "auth_type": "managed_identity",
            "client_id": "your-client-id",
            "tenant_id": "your-tenant-id",
            "resource_id": "/subscriptions/.../resourceGroups/.../providers/Microsoft.CognitiveServices/accounts/..."
        }
    )
    config.add_model(managed_identity_model)
    
    print("Managed Identity Configuration:")
    print(f"  Auth type: {managed_identity_model.metadata['auth_type']}")
    print(f"  Client ID: {managed_identity_model.metadata['client_id']}")
    print("  Note: In production, use Azure SDK for managed identity authentication")
    
    # Step 8: Cost and quota tracking
    print("\nStep 8: Cost and Quota Tracking Setup")
    print("-"*80)
    
    print("\nAzure cost optimization configurations:")
    cost_models = [
        {
            "name": "gpt-4-premium",
            "quota": 1000000,
            "priority": "high",
            "cost_per_1k": 0.03
        },
        {
            "name": "gpt-35-standard",
            "quota": 5000000,
            "priority": "medium",
            "cost_per_1k": 0.002
        }
    ]
    
    for cost_info in cost_models:
        print(f"\n{cost_info['name']}:")
        print(f"  Monthly quota: {cost_info['quota']:,} tokens")
        print(f"  Priority: {cost_info['priority']}")
        print(f"  Cost per 1K tokens: ${cost_info['cost_per_1k']}")
    
    # Step 9: Summary
    print("\nStep 9: Azure Configuration Summary")
    print("-"*80)
    
    azure_models = config.list_models(provider=ProviderType.AZURE_OPENAI)
    print(f"\nTotal Azure deployments configured: {len(azure_models)}")
    
    print("\nConfigured deployments:")
    for model_name in azure_models:
        model = config.get_model(model_name)
        print(f"\n  {model_name}:")
        print(f"    Deployment: {model.deployment_name}")
        print(f"    API Version: {model.api_version}")
        if model.metadata:
            if 'region' in model.metadata:
                print(f"    Region: {model.metadata['region']}")
            if 'environment' in model.metadata:
                print(f"    Environment: {model.metadata['environment']}")
    
    print("\n" + "="*80)
    print("Azure OpenAI configuration completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
