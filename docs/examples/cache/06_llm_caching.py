"""LLM Prompt Caching Example

This example demonstrates caching LLM prompts and responses.

Main concepts:
- Generating cache keys for prompts
- Avoiding redundant LLM API calls
- Cost tracking and savings
- Handling different models and parameters
- Cache invalidation strategies
"""

from kerb.cache import (
    create_memory_cache,
    generate_prompt_key,
)


def mock_llm_api_call(prompt, model="gpt-4", temperature=0.7, max_tokens=100):
    """Simulate an expensive LLM API call.
    
    In production, this would be a real API call to OpenAI, Anthropic, etc.
    """
    # Simulate cost (in USD)
    cost = 0.01 if model == "gpt-4" else 0.001
    
    # Mock response based on prompt
    if "weather" in prompt.lower():
        response = "The weather is sunny with a temperature of 72°F and clear skies."
    elif "python" in prompt.lower():
        response = "Python is a high-level, interpreted programming language known for its simplicity."
    elif "capital" in prompt.lower():
        response = "I'd be happy to help with capital cities. Which country are you asking about?"
    else:
        response = f"This is a response to: {prompt[:50]}..."
    
    return {
        "response": response,
        "model": model,
        "prompt_tokens": len(prompt.split()) * 4,
        "completion_tokens": len(response.split()) * 4,
        "cost": cost
    }


def main():
    """Run LLM prompt caching example."""
    
    print("="*80)
    print("LLM PROMPT CACHING EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Basic prompt caching
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC PROMPT CACHING")
    print("-"*80)
    
    cache = create_memory_cache()
    
    prompt1 = "What is the weather like today?"
    
    # First call - cache miss
    print(f"\nPrompt: '{prompt1}'")
    print("First call (cache miss)...")
    
    key1 = generate_prompt_key(prompt1, model="gpt-4", temperature=0.7)
    cached = cache.get(key1)
    
    if cached is None:
        print("  ✗ Cache miss - calling API")
        response1 = mock_llm_api_call(prompt1, model="gpt-4", temperature=0.7)
        cache.set(key1, response1, metadata={"cost": response1["cost"]})
        print(f"  Response: {response1['response']}")
        print(f"  Cost: ${response1['cost']:.4f}")
    
    # Second call - cache hit!
    print("\nSecond call (should hit cache)...")
    cached = cache.get(key1)
    
    if cached:
        print("  ✓ Cache hit - no API call needed!")
        print(f"  Response: {cached['response']}")
        print(f"  Saved: ${cached['cost']:.4f}")
    
    # ========================================================================
    # 2. Different parameters = different cache keys
    # ========================================================================
    print("\n" + "-"*80)
    print("2. PARAMETER SENSITIVITY")
    print("-"*80)
    
    same_prompt = "Explain Python programming"
    
    # Same prompt, different parameters
    key_gpt4 = generate_prompt_key(same_prompt, model="gpt-4", temperature=0.7)
    key_gpt35 = generate_prompt_key(same_prompt, model="gpt-3.5-turbo", temperature=0.7)
    key_temp1 = generate_prompt_key(same_prompt, model="gpt-4", temperature=1.0)
    
    print(f"\nPrompt: '{same_prompt}'")
    print("\nCache keys for different parameters:")
    print(f"  model=gpt-4, temp=0.7:         {key_gpt4[:16]}...")
    print(f"  model=gpt-3.5-turbo, temp=0.7: {key_gpt35[:16]}...")
    print(f"  model=gpt-4, temp=1.0:         {key_temp1[:16]}...")
    print("\nEach combination gets its own cache entry!")
    
    # ========================================================================
    # 3. Cost tracking
    # ========================================================================
    print("\n" + "-"*80)
    print("3. COST TRACKING")
    print("-"*80)
    
    cost_cache = create_memory_cache()
    total_cost = 0.0
    total_saved = 0.0
    
    prompts = [
        "What is the weather like today?",
        "Explain Python programming",
        "What is the weather like today?",  # Duplicate
        "What is the capital of France?",
        "Explain Python programming",        # Duplicate
        "What is the weather like today?",  # Duplicate again
    ]
    
    print(f"\nProcessing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts, 1):
        key = generate_prompt_key(prompt, model="gpt-4")
        cached = cost_cache.get(key)
        
        if cached is None:
            # API call needed
            response = mock_llm_api_call(prompt, model="gpt-4")
            cost_cache.set(key, response, metadata={"cost": response["cost"]})
            total_cost += response["cost"]
            print(f"  {i}. ✗ API call: '{prompt[:40]}...' (${response['cost']:.4f})")
        else:
            # Cache hit - saved money!
            total_saved += cached["cost"]
            print(f"  {i}. ✓ Cached:   '{prompt[:40]}...' (saved ${cached['cost']:.4f})")
    
    print(f"\nCost Summary:")
    print(f"  Total API cost: ${total_cost:.4f}")
    print(f"  Total saved:    ${total_saved:.4f}")
    print(f"  Savings rate:   {(total_saved / (total_cost + total_saved) * 100):.1f}%")
    
    # ========================================================================
    # 4. Practical: LLM client with caching
    # ========================================================================
    print("\n" + "-"*80)
    print("4. PRACTICAL: LLM CLIENT WITH CACHING")
    print("-"*80)
    
    class CachedLLMClient:
        """LLM client with automatic caching."""
        
        def __init__(self):
            self.cache = create_memory_cache()
            self.api_calls = 0
            self.cache_hits = 0
        
        def generate(self, prompt, model="gpt-4", temperature=0.7, max_tokens=100):
            """Generate response with automatic caching."""
            # Generate cache key
            key = generate_prompt_key(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Check cache
            cached = self.cache.get(key)
            if cached:
                self.cache_hits += 1
                return cached["response"]
            
            # Call API
            self.api_calls += 1
            response = mock_llm_api_call(prompt, model, temperature, max_tokens)
            
            # Cache response
            self.cache.set(key, response, metadata={"cost": response["cost"]})
            
            return response["response"]
        
        def stats(self):
            """Get usage statistics."""
            total = self.api_calls + self.cache_hits
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
            return {
                "total_requests": total,
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
                "hit_rate": f"{hit_rate:.1f}%"
            }
    
    # Use the cached client
    client = CachedLLMClient()
    
    print("\nUsing CachedLLMClient:")
    
    # First request
    response1 = client.generate("What is machine learning?")
    print(f"  Request 1: {response1[:60]}...")
    
    # Duplicate request
    response2 = client.generate("What is machine learning?")
    print(f"  Request 2 (duplicate): {response2[:60]}...")
    
    # Different request
    response3 = client.generate("Explain neural networks")
    print(f"  Request 3: {response3[:60]}...")
    
    # Another duplicate
    response4 = client.generate("What is machine learning?")
    print(f"  Request 4 (duplicate): {response4[:60]}...")
    
    # Show statistics
    stats = client.stats()
    print(f"\nClient Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  API calls:      {stats['api_calls']}")
    print(f"  Cache hits:     {stats['cache_hits']}")
    print(f"  Hit rate:       {stats['hit_rate']}")
    
    # ========================================================================
    # 5. Cache invalidation strategies
    # ========================================================================
    print("\n" + "-"*80)
    print("5. CACHE INVALIDATION STRATEGIES")
    print("-"*80)
    
    # Strategy 1: TTL-based (time-based expiration)
    ttl_cache = create_memory_cache()
    prompt = "What's the current stock price?"
    key = generate_prompt_key(prompt, model="gpt-4")
    
    response = mock_llm_api_call(prompt)
    ttl_cache.set(key, response, ttl=300)  # 5 minutes
    
    print("\nStrategy 1: TTL-based invalidation")
    print("  • Use case: Time-sensitive data (stock prices, weather)")
    print("  • Set TTL=300 (5 minutes)")
    print("  • Cache automatically expires after TTL")
    
    # Strategy 2: Manual invalidation
    manual_cache = create_memory_cache()
    prompt2 = "Describe our product"
    key2 = generate_prompt_key(prompt2)
    
    response2 = mock_llm_api_call(prompt2)
    manual_cache.set(key2, response2)
    
    print("\nStrategy 2: Manual invalidation")
    print("  • Use case: Content updates (product changes)")
    print("  • Explicitly delete when content changes")
    
    # Simulate content update
    print("  • Product updated - invalidating cache...")
    manual_cache.delete(key2)
    print("  • Cache cleared for product description")
    
    # Strategy 3: Version-based keys
    print("\nStrategy 3: Version-based keys")
    print("  • Use case: Model or system updates")
    print("  • Include version in cache key")
    
    version = "v2"
    versioned_key = generate_prompt_key(
        "Analyze sentiment",
        model="gpt-4",
        version=version  # Additional parameter
    )
    print(f"  • Key includes version: ...{versioned_key[-16:]}")
    print("  • New version = new cache, old cache ignored")
    
    # ========================================================================
    # 6. Batch caching
    # ========================================================================
    print("\n" + "-"*80)
    print("6. BATCH CACHING")
    print("-"*80)
    
    batch_cache = create_memory_cache()
    
    questions = [
        "What is AI?",
        "What is ML?", 
        "What is AI?",  # Duplicate
        "What is DL?",
        "What is ML?",  # Duplicate
    ]
    
    print(f"\nProcessing batch of {len(questions)} questions:")
    
    results = []
    for q in questions:
        key = generate_prompt_key(q, model="gpt-4")
        cached = batch_cache.get(key)
        
        if cached is None:
            response = mock_llm_api_call(q, model="gpt-4")
            batch_cache.set(key, response)
            results.append(response["response"])
            print(f"  ✗ API: {q}")
        else:
            results.append(cached["response"])
            print(f"  ✓ Cache: {q}")
    
    print(f"\nProcessed {len(questions)} questions with only {batch_cache.size()} API calls")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • Cache LLM responses to avoid redundant API calls")
    print("  • Use generate_prompt_key() for consistent cache keys")
    print("  • Different parameters = different cache entries")
    print("  • Track costs to measure savings")
    print("  • Use TTL for time-sensitive data")
    print("  • Implement cache invalidation strategy")
    print("  • Typical savings: 50-90% reduction in API costs")


if __name__ == "__main__":
    main()
