"""
Batch Generation Example
========================

This example demonstrates processing multiple prompts concurrently.

Main concepts:
- Using generate_batch() for concurrent processing
- Managing parallelism with max_concurrent
- Progress tracking for batch operations
- Comparing batch vs sequential processing
- Handling errors in batch operations
"""

import time
from typing import List
from kerb.generation import generate, generate_batch, ModelName
from kerb.generation.config import GenerationResponse


def example_basic_batch():
    """Basic batch generation with multiple prompts."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Batch Generation")
    print("="*80)
    
    prompts = [
        "What is a Python generator?",
        "Explain Python decorators.",
        "What are context managers?",
        "Define duck typing.",
        "What is the GIL?"
    ]
    
    print(f"\nProcessing {len(prompts)} prompts in batch...")
    
    start_time = time.time()
    responses = generate_batch(
        prompts,
        model=ModelName.GPT_4O_MINI,
        max_concurrent=3  # Process 3 at a time
    )
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.2f}s\n")
    
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"{i}. {prompt}")
        print(f"   Response: {response.content[:80]}...")
        print(f"   Tokens: {response.usage.total_tokens}, Cost: ${response.cost:.6f}\n")


def example_batch_with_progress():
    """Batch generation with progress tracking."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch with Progress Tracking")
    print("="*80)
    
    # Generate prompts for code review tasks
    prompts = [
        f"Review this code pattern: {pattern}"
        for pattern in [
            "using global variables",
            "deeply nested if statements",
            "long parameter lists",
            "magic numbers in code"
        ]
    ]
    
    print(f"\nProcessing {len(prompts)} code reviews...")
    
    responses = generate_batch(
        prompts,
        model=ModelName.GPT_4O_MINI,
        max_concurrent=2,
        show_progress=True,  # Enable progress display
        temperature=0.5,
        max_tokens=100
    )
    
    print(f"\nReceived {len(responses)} responses")
    
    # Calculate aggregate statistics
    total_tokens = sum(r.usage.total_tokens for r in responses)
    total_cost = sum(r.cost for r in responses)
    
    print(f"Total tokens used: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f}")



# %%
# Example Batch Vs Sequential
# ---------------------------

def example_batch_vs_sequential():
    """Compare batch processing to sequential processing."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch vs Sequential Performance")
    print("="*80)
    
    prompts = [
        "Name a Python web framework.",
        "Name a Python testing framework.",
        "Name a Python data science library.",
        "Name a Python async library.",
    ]
    
    # Sequential processing
    print("\n1. Sequential Processing:")
    start_seq = time.time()
    
    sequential_responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"   Processing {i}/{len(prompts)}...", end="\r")
        response = generate(prompt, model=ModelName.GPT_4O_MINI, max_tokens=20)
        sequential_responses.append(response)
    
    elapsed_seq = time.time() - start_seq
    print(f"   Completed in {elapsed_seq:.2f}s" + " " * 20)
    
    # Batch processing
    print("\n2. Batch Processing:")
    start_batch = time.time()
    
    batch_responses = generate_batch(
        prompts,
        model=ModelName.GPT_4O_MINI,
        max_concurrent=4,
        max_tokens=20
    )
    
    elapsed_batch = time.time() - start_batch
    print(f"   Completed in {elapsed_batch:.2f}s")
    
    # Comparison
    speedup = elapsed_seq / elapsed_batch
    print(f"\nPerformance Improvement:")
    print(f"   Sequential: {elapsed_seq:.2f}s")
    print(f"   Batch: {elapsed_batch:.2f}s")
    print(f"   Speedup: {speedup:.2f}x faster")


def example_batch_with_config():
    """Batch generation with custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch with Custom Configuration")
    print("="*80)
    
    from kerb.generation import GenerationConfig
    
    # Create a specific configuration
    config = GenerationConfig(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=50,
        top_p=0.9
    )
    
    # Generate summaries for different topics
    topics = [
        "machine learning",
        "cloud computing",
        "microservices",
    ]
    
    prompts = [f"Define {topic} in one sentence." for topic in topics]
    
    print(f"\nGenerating definitions with temp={config.temperature}...\n")
    
    responses = generate_batch(
        prompts,
        config=config,
        max_concurrent=3
    )
    
    for topic, response in zip(topics, responses):
        print(f"{topic.title()}:")
        print(f"  {response.content}")
        print(f"  ({response.usage.total_tokens} tokens)\n")



# %%
# Example Batch Error Handling
# ----------------------------

def example_batch_error_handling():
    """Handle errors gracefully in batch processing."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Error Handling")
    print("="*80)
    
    # Mix of valid prompts and potentially problematic ones
    prompts = [
        "What is Python?",
        "Explain TypeScript.",
        "Define Rust.",
        "What is Go?",
    ]
    
    print(f"\nProcessing {len(prompts)} prompts with error handling...\n")
    
    successful = 0
    failed = 0
    
    try:
        responses = generate_batch(
            prompts,
            model=ModelName.GPT_4O_MINI,
            max_concurrent=2,
            max_tokens=50
        )
        
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            if response and response.content:
                print(f"[OK] Prompt {i}: {prompt}")
                print(f"     {response.content[:60]}...\n")
                successful += 1
            else:
                print(f"[FAIL] Prompt {i}: {prompt}\n")
                failed += 1
    
    except Exception as e:
        print(f"Batch processing error: {e}")
        failed = len(prompts)
    
    print(f"Results: {successful} successful, {failed} failed")


def example_batch_different_lengths():
    """Process prompts of varying complexity."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Varying Prompt Complexity")
    print("="*80)
    
    prompts = [
        "Define API.",  # Short, simple
        "Explain REST API design principles.",  # Medium
        "Compare GraphQL and REST APIs, including pros and cons.",  # Long
        "What is HTTP?",  # Short
    ]
    
    print("\nProcessing prompts of varying complexity...\n")
    
    start = time.time()
    responses = generate_batch(
        prompts,
        model=ModelName.GPT_4O_MINI,
        max_concurrent=2,
        max_tokens=150
    )
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f}s\n")
    
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"{i}. Prompt ({len(prompt)} chars): {prompt}")
        print(f"   Response ({len(response.content)} chars, {response.usage.total_tokens} tokens)")
        print(f"   Latency: {response.latency:.3f}s\n")



# %%
# Main
# ----

def main():
    """Run all batch generation examples."""
    print("\n" + "#"*80)
    print("# BATCH GENERATION EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_batch()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_batch_with_progress()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_batch_vs_sequential()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_batch_with_config()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_batch_error_handling()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_batch_different_lengths()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
