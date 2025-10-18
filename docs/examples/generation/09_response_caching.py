"""Response Caching Example

This example demonstrates intelligent caching for LLM responses.

Main concepts:
- Using ResponseCache to cache LLM responses
- Reducing API calls and costs
- Cache hit/miss tracking
- TTL (time-to-live) management
- Cache invalidation strategies
"""

import time
from kerb.generation import generate, ModelName, GenerationConfig
from kerb.generation.utils import ResponseCache
from kerb.core import Message
from kerb.core.types import MessageRole


def example_basic_caching():
    """Demonstrate basic response caching."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Response Caching")
    print("="*80)
    
    cache = ResponseCache(max_size=100, ttl=3600)
    
    prompt = "What is Python?"
    config = GenerationConfig(model="gpt-4o-mini")
    messages = [Message(role=MessageRole.USER, content=prompt)]
    
    print(f"\nPrompt: {prompt}\n")
    
    # First request (cache miss)
    print("First request (cache miss):")
    start = time.time()
    response1 = generate(prompt, model=ModelName.GPT_4O_MINI)
    elapsed1 = time.time() - start
    
    # Store in cache
    cache.set(messages, config, response1)
    
    print(f"  Time: {elapsed1:.3f}s")
    print(f"  Response: {response1.content[:60]}...")
    print(f"  Cached: {response1.cached}")
    
    # Second request (cache hit)
    print("\nSecond request (cache hit):")
    start = time.time()
    cached_response = cache.get(messages, config)
    elapsed2 = time.time() - start
    
    if cached_response:
        print(f"  Time: {elapsed2:.6f}s (from cache)")
        print(f"  Response: {cached_response.content[:60]}...")
        print(f"  Cached: {cached_response.cached}")
        print(f"\n  Speedup: {elapsed1/elapsed2:.0f}x faster")
    else:
        print("  Cache miss!")


def example_cache_with_generation():
    """Use caching with generate() function."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Automatic Caching with generate()")
    print("="*80)
    
    prompts = [
        "Define API",
        "What is REST?",
        "Define API",  # Duplicate - should use cache
        "What is REST?",  # Duplicate - should use cache
    ]
    
    print("\nProcessing prompts with caching enabled...\n")
    
    for i, prompt in enumerate(prompts, 1):
        start = time.time()
        response = generate(
            prompt,
            model=ModelName.GPT_4O_MINI,
            use_cache=True,
            max_tokens=50
        )
        elapsed = time.time() - start
        
        cache_status = "HIT" if response.cached else "MISS"
        print(f"[{i}] {prompt}")
        print(f"    Cache: {cache_status}, Time: {elapsed:.3f}s")
        print(f"    Response: {response.content[:50]}...\n")


def example_cache_ttl():
    """Demonstrate cache expiration (TTL)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Cache TTL (Time-To-Live)")
    print("="*80)
    
    # Short TTL for demonstration
    cache = ResponseCache(max_size=100, ttl=2)  # 2 second TTL
    
    prompt = "What is caching?"
    config = GenerationConfig(model="gpt-4o-mini")
    messages = [Message(role=MessageRole.USER, content=prompt)]
    
    print(f"\nCache TTL: 2 seconds")
    print(f"Prompt: {prompt}\n")
    
    # First request
    print("1. Initial request:")
    response1 = generate(prompt, model=ModelName.GPT_4O_MINI, max_tokens=30)
    cache.set(messages, config, response1)
    print(f"   Stored in cache")
    
    # Immediate check (should hit)
    print("\n2. Immediate second request:")
    cached = cache.get(messages, config)
    print(f"   Cache: {'HIT' if cached else 'MISS'}")
    
    # Wait for TTL to expire
    print("\n3. Waiting for cache to expire (2s)...", end="", flush=True)
    time.sleep(2.1)
    print(" done")
    
    # Check after expiration (should miss)
    print("\n4. Request after TTL expiration:")
    cached = cache.get(messages, config)
    print(f"   Cache: {'HIT' if cached else 'MISS'}")


def example_cache_size_limit():
    """Demonstrate cache size management."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Cache Size Limit")
    print("="*80)
    
    # Small cache for demonstration
    cache = ResponseCache(max_size=3, ttl=3600)
    
    print(f"\nCache max size: 3 entries\n")
    
    # Add more items than cache can hold
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Go?",
        "What is Rust?",  # This will evict oldest
    ]
    
    for i, prompt in enumerate(prompts, 1):
        config = GenerationConfig(model="gpt-4o-mini")
        messages = [Message(role=MessageRole.USER, content=prompt)]
        
        response = generate(prompt, model=ModelName.GPT_4O_MINI, max_tokens=20)
        cache.set(messages, config, response)
        
        print(f"{i}. Added: '{prompt}'")
        print(f"   Cache size: {len(cache.cache)}/{cache.max_size}")
    
    print("\nChecking which items are still cached:")
    
    for prompt in prompts:
        config = GenerationConfig(model="gpt-4o-mini")
        messages = [Message(role=MessageRole.USER, content=prompt)]
        cached = cache.get(messages, config)
        status = "YES" if cached else "NO"
        print(f"  '{prompt}': {status}")


def example_cost_savings():
    """Calculate cost savings from caching."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Cost Savings from Caching")
    print("="*80)
    
    from kerb.generation.utils import CostTracker
    
    tracker = CostTracker()
    
    # Simulate repeated queries
    queries = [
        "What is a database?",
        "What is SQL?",
        "What is a database?",  # Duplicate
        "What is NoSQL?",
        "What is SQL?",  # Duplicate
        "What is a database?",  # Duplicate
    ]
    
    print(f"\nProcessing {len(queries)} queries (with duplicates)...\n")
    
    cache_hits = 0
    cache_misses = 0
    
    for i, query in enumerate(queries, 1):
        response = generate(
            query,
            model=ModelName.GPT_4O_MINI,
            cost_tracker=tracker,
            use_cache=True,
            max_tokens=30
        )
        
        if response.cached:
            cache_hits += 1
            print(f"[{i}] CACHED: {query}")
        else:
            cache_misses += 1
            print(f"[{i}] NEW: {query}")
    
    # Calculate savings
    unique_queries = len(set(queries))
    total_cost = tracker.total_cost
    potential_cost = total_cost * (len(queries) / unique_queries)
    savings = potential_cost - total_cost
    savings_pct = (savings / potential_cost * 100) if potential_cost > 0 else 0
    
    print("\n" + "-"*80)
    print("COST ANALYSIS")
    print("-"*80)
    print(f"Total queries: {len(queries)}")
    print(f"Unique queries: {unique_queries}")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache misses: {cache_misses}")
    print(f"\nActual cost: ${total_cost:.6f}")
    print(f"Cost without cache: ${potential_cost:.6f}")
    print(f"Savings: ${savings:.6f} ({savings_pct:.1f}%)")


def example_conditional_caching():
    """Implement conditional caching strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Conditional Caching")
    print("="*80)
    
    def should_cache(prompt: str, config: GenerationConfig) -> bool:
        """Determine if response should be cached."""
        # Don't cache if temperature is high (non-deterministic)
        if config.temperature > 0.7:
            return False
        
        # Don't cache very short prompts
        if len(prompt) < 10:
            return False
        
        # Cache everything else
        return True
    
    test_cases = [
        {
            "prompt": "What is Python?",
            "temp": 0.5,
            "should_cache": True
        },
        {
            "prompt": "Create a random story about aliens.",
            "temp": 0.9,
            "should_cache": False  # High temp
        },
        {
            "prompt": "Hi",
            "temp": 0.5,
            "should_cache": False  # Too short
        },
    ]
    
    print("\nTesting caching conditions:\n")
    
    for i, test in enumerate(test_cases, 1):
        config = GenerationConfig(
            model="gpt-4o-mini",
            temperature=test["temp"],
            max_tokens=30
        )
        
        cache_decision = should_cache(test["prompt"], config)
        
        print(f"{i}. Prompt: '{test['prompt']}'")
        print(f"   Temperature: {test['temp']}")
        print(f"   Should cache: {cache_decision}")
        print(f"   Expected: {test['should_cache']}")
        print(f"   Result: {'MATCH' if cache_decision == test['should_cache'] else 'MISMATCH'}\n")


def main():
    """Run all response caching examples."""
    print("\n" + "#"*80)
    print("# RESPONSE CACHING EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_caching()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_cache_with_generation()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_cache_ttl()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_cache_size_limit()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_cost_savings()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_conditional_caching()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
