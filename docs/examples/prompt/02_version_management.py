"""Prompt versioning and A/B testing for LLM applications.

This example demonstrates:
- Creating and managing prompt versions
- A/B testing different prompt variations
- Comparing prompt performance
- Version selection strategies
"""

from kerb.prompt import (
    create_version,
    register_prompt,
    get_prompt,
    list_versions,
    compare_versions,
    select_version
)


def create_prompt_versions():
    """Create multiple versions of a prompt for testing."""
    print("=" * 80)
    print("CREATING PROMPT VERSIONS")
    print("=" * 80)
    
    # Version 1: Simple, direct approach
    v1 = create_version(
        name="code_reviewer",
        version="1.0",
        template="""Review this code and provide feedback:

{{code}}

Focus on bugs and improvements.""",
        description="Direct and concise code review prompt",
        metadata={"tokens_avg": 50, "response_quality": 0.75}
    )
    
    # Version 2: More structured with specific guidelines
    v2 = create_version(
        name="code_reviewer",
        version="2.0",
        template="""You are an expert code reviewer. Analyze the following code:

{{code}}

Provide feedback on:
1. Code correctness and bugs
2. Performance optimizations
3. Best practices and style
4. Security considerations

Format your response with clear sections.""",
        description="Structured code review with explicit guidelines",
        metadata={"tokens_avg": 120, "response_quality": 0.85}
    )
    
    # Version 3: Concise version for faster responses
    v3 = create_version(
        name="code_reviewer",
        version="3.0-fast",
        template="""Code review: {{code}}

Quick assessment: correctness, performance, style.""",
        description="Ultra-concise for fast responses",
        metadata={"tokens_avg": 30, "response_quality": 0.70}
    )
    
    # Register all versions
    register_prompt(v1)
    register_prompt(v2)
    register_prompt(v3)
    
    print("\nCreated 3 versions of 'code_reviewer' prompt")
    print(f"- v1.0: {v1.description}")
    print(f"- v2.0: {v2.description}")
    print(f"- v3.0-fast: {v3.description}")


def retrieve_and_render_versions():
    """Retrieve and use prompt versions."""
    print("\n" + "=" * 80)
    print("RETRIEVING AND RENDERING VERSIONS")
    print("=" * 80)
    
    # Get a specific version
    v2 = get_prompt("code_reviewer", "2.0")
    
    # Render it with actual code
    code_sample = """def calculate(x, y):
    return x + y * 2"""
    
    rendered = v2.render({"code": code_sample})
    
    print("\nUsing version 2.0:")
    print(rendered)


def list_available_versions():
    """List all versions of a prompt."""
    print("\n" + "=" * 80)
    print("LISTING AVAILABLE VERSIONS")
    print("=" * 80)
    
    versions = list_versions("code_reviewer")
    
    print(f"\nAvailable versions of 'code_reviewer': {versions}")
    print("\nThis is useful for:")
    print("- Dashboard UI showing available prompt variants")
    print("- Automated testing across all versions")
    print("- Version migration planning")


def compare_prompt_versions():
    """Compare different versions for A/B testing analysis."""
    print("\n" + "=" * 80)
    print("COMPARING PROMPT VERSIONS")
    print("=" * 80)
    
    comparison = compare_versions("code_reviewer")
    
    print("\nComparison of 'code_reviewer' versions:")
    versions_dict = comparison.get("versions", {})
    print(f"Total versions: {len(versions_dict)}")
    
    for version_id, version_info in versions_dict.items():
        print(f"\nVersion: {version_id}")
        print(f"  Description: {version_info['description']}")
        print(f"  Template length: {version_info['length']} chars")
        if version_info.get('metadata'):
            print(f"  Metadata: {version_info['metadata']}")


def ab_testing_selection():
    """Select versions for A/B testing."""
    print("\n" + "=" * 80)
    print("A/B TESTING VERSION SELECTION")
    print("=" * 80)
    
    print("\nRandom selection (for A/B testing):")
    for i in range(3):
        selected = select_version("code_reviewer", strategy="random")
        print(f"  Request {i+1}: {selected.version}")
    
    print("\nBest performing version (based on metadata):")
    selected = select_version("code_reviewer", strategy="best_performing")
    if selected:
        print(f"  Selected: {selected.version} (quality: {selected.metadata.get('response_quality')})")
    
    print("\nLatest version:")
    selected = select_version("code_reviewer", strategy="latest")
    print(f"  Selected: {selected.version}")


def production_scenario():
    """Demonstrate production A/B testing workflow."""
    print("\n" + "=" * 80)
    print("PRODUCTION A/B TESTING SCENARIO")
    print("=" * 80)
    
    # Create sentiment analysis prompt versions
    baseline = create_version(
        name="sentiment",
        version="baseline",
        template="Analyze sentiment: {{text}}",
        description="Baseline sentiment analysis",
        metadata={"accuracy": 0.82, "latency_ms": 150}
    )
    
    experimental = create_version(
        name="sentiment",
        version="experimental",
        template="""Analyze the sentiment of this text with reasoning:

{{text}}

Provide: sentiment (positive/negative/neutral) and confidence score.""",
        description="Experimental with reasoning",
        metadata={"accuracy": 0.87, "latency_ms": 220}
    )
    
    register_prompt(baseline)
    register_prompt(experimental)
    
    print("\nA/B Testing Setup:")
    print("- Baseline: Fast but lower accuracy")
    print("- Experimental: More accurate but slower")
    
    # Simulate traffic split
    print("\nSimulating 50/50 traffic split:")
    import random
    random.seed(42)  # For reproducibility
    
    for i in range(6):
        strategy = "random"  # Random selection for A/B test
        selected = select_version("sentiment", strategy=strategy)
        print(f"  Request {i+1}: {selected.version} " + 
              f"(accuracy: {selected.metadata['accuracy']}, " +
              f"latency: {selected.metadata['latency_ms']}ms)")


def version_rollout_strategy():
    """Demonstrate gradual version rollout."""
    print("\n" + "=" * 80)
    print("GRADUAL VERSION ROLLOUT")
    print("=" * 80)
    
    # Create versions with rollout weights
    stable = create_version(
        name="summarizer",
        version="stable",
        template="Summarize: {{text}}",
        metadata={"rollout_weight": 0.8}  # 80% traffic
    )
    
    canary = create_version(
        name="summarizer",
        version="canary",
        template="Create a concise summary with key points: {{text}}",
        metadata={"rollout_weight": 0.2}  # 20% traffic
    )
    
    register_prompt(stable)
    register_prompt(canary)
    
    print("\nRollout configuration:")
    print("- Stable version: 80% of traffic")
    print("- Canary version: 20% of traffic")
    print("\nThis allows safe testing of new prompts in production")


def main():
    """Run all versioning examples."""
    print("\n" + "=" * 80)
    print("PROMPT VERSIONING EXAMPLES FOR LLM DEVELOPERS")
    print("=" * 80)
    
    create_prompt_versions()
    retrieve_and_render_versions()
    list_available_versions()
    compare_prompt_versions()
    ab_testing_selection()
    production_scenario()
    version_rollout_strategy()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
