"""
Cache Decorator Example
=======================

This example demonstrates using the @cached decorator for automatic function caching.

Main concepts:
- Using @cached decorator for transparent caching
- Automatic cache key generation from function arguments
- Custom key generation functions
- Decorator with different cache backends
- Method caching in classes
"""

import time
from kerb.cache import (
    cached,
    create_memory_cache,
    create_disk_cache,
)


def main():
    """Run cache decorator example."""
    
    print("="*80)
    print("CACHE DECORATOR EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Basic decorator usage
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC DECORATOR USAGE")
    print("-"*80)
    
    # Decorator creates a default cache automatically
    @cached()
    def expensive_function(x):
        """Simulate expensive computation."""

# %%
# Setup and Imports
# -----------------
        print(f"    Computing for x={x}...")
        time.sleep(0.1)  # Simulate work
        return x * x
    
    print("\nCalling expensive_function(5)...")
    result1 = expensive_function(5)
    print(f"Result: {result1}")
    
    print("\nCalling expensive_function(5) again...")
    result2 = expensive_function(5)
    print(f"Result: {result2} (from cache - no 'Computing' message)")
    
    print("\nCalling expensive_function(10)...")
    result3 = expensive_function(10)
    print(f"Result: {result3}")
    
    # ========================================================================
    # 2. Decorator with custom cache
    # ========================================================================
    print("\n" + "-"*80)
    print("2. DECORATOR WITH CUSTOM CACHE")
    print("-"*80)
    
    # Create a custom cache with specific settings
    my_cache = create_memory_cache(max_size=100, default_ttl=3600)
    
    @cached(cache=my_cache)

# %%
# Fetch User Data
# ---------------

    def fetch_user_data(user_id):
        """Simulate API call to fetch user data."""
        print(f"    Fetching data for user {user_id} from API...")
        return {"id": user_id, "name": f"User_{user_id}", "active": True}
    
    print("\nFetching user 123...")
    user1 = fetch_user_data(123)
    print(f"Result: {user1}")
    
    print("\nFetching user 123 again...")
    user2 = fetch_user_data(123)
    print(f"Result: {user2} (cached)")
    
    # ========================================================================
    # 3. Decorator with TTL
    # ========================================================================
    print("\n" + "-"*80)
    print("3. DECORATOR WITH TTL")
    print("-"*80)
    
    @cached(ttl=2.0)  # Cache for 2 seconds
    def get_current_time():
        """Get current timestamp."""
        print("    Fetching current time...")
        return time.time()
    
    print("\nFirst call:")
    t1 = get_current_time()
    print(f"Time: {t1:.2f}")
    
    print("\nImmediate second call (cached):")
    t2 = get_current_time()
    print(f"Time: {t2:.2f}")
    
    print("\nWaiting 2.5 seconds...")
    time.sleep(2.5)
    
    print("\nThird call (cache expired):")
    t3 = get_current_time()
    print(f"Time: {t3:.2f}")
    
    # ========================================================================
    # 4. Custom key generation
    # ========================================================================
    print("\n" + "-"*80)
    print("4. CUSTOM KEY GENERATION")
    print("-"*80)
    

# %%
# Custom Key Fn
# -------------

    def custom_key_fn(*args, **kwargs):
        """Custom function to generate cache keys."""
        # Only use first argument, ignore case
        if args:
            return f"custom:{str(args[0]).lower()}"
        return "custom:default"
    
    @cached(key_fn=custom_key_fn)
    def search_database(query, limit=10):
        """Search database with query."""
        print(f"    Searching for '{query}' with limit={limit}...")
        return [f"Result for {query}"]
    
    print("\nSearching for 'Python':")
    r1 = search_database("Python", limit=10)
    print(f"Results: {r1}")
    
    print("\nSearching for 'python' (lowercase):")
    r2 = search_database("python", limit=10)
    print(f"Results: {r2} (cached due to custom key)")
    
    print("\nSearching for 'Python' with different limit:")
    r3 = search_database("Python", limit=20)
    print(f"Results: {r3} (still cached - limit ignored by key_fn)")
    
    # ========================================================================
    # 5. Method caching in classes
    # ========================================================================
    print("\n" + "-"*80)
    print("5. METHOD CACHING IN CLASSES")
    print("-"*80)
    
    class DataProcessor:
        """Example class with cached methods."""
        
        def __init__(self):
            self.cache = create_memory_cache(max_size=100)
            self.api_calls = 0
        
        @cached()
        def process_data(self, data_id):
            """Process data with caching."""
            self.api_calls += 1
            print(f"    Processing data {data_id}...")
            return {"id": data_id, "processed": True}
        
        @cached(ttl=5.0)

# %%
# Get Stats
# ---------

        def get_stats(self):
            """Get statistics with 5-second cache."""
            print("    Computing statistics...")
            return {"total": 100, "processed": 75}
    
    processor = DataProcessor()
    
    print("\nProcessing data 1:")
    result1 = processor.process_data(1)
    print(f"Result: {result1}, API calls: {processor.api_calls}")
    
    print("\nProcessing data 1 again:")
    result2 = processor.process_data(1)
    print(f"Result: {result2}, API calls: {processor.api_calls}")
    
    print("\nProcessing data 2:")
    result3 = processor.process_data(2)
    print(f"Result: {result3}, API calls: {processor.api_calls}")
    
    # ========================================================================
    # 6. Practical: API client with caching
    # ========================================================================
    print("\n" + "-"*80)
    print("6. PRACTICAL: API CLIENT WITH CACHING")
    print("-"*80)
    
    class WeatherAPIClient:
        """Weather API client with automatic caching."""
        

# %%
#   Init  
# --------

        def __init__(self):
            self.cache = create_memory_cache(default_ttl=300)  # 5 min cache
            self.request_count = 0
        
        @cached(ttl=300)  # Cache weather for 5 minutes

# %%
# Get Weather
# -----------

        def get_weather(self, city):
            """Get weather for a city."""
            self.request_count += 1
            print(f"    Making API request for {city}...")
            # Simulate API call
            return {
                "city": city,
                "temp": 72,
                "condition": "sunny",
                "timestamp": time.time()
            }
        
        @cached(ttl=3600)  # Cache forecast for 1 hour
        def get_forecast(self, city, days=7):
            """Get weather forecast."""
            self.request_count += 1
            print(f"    Making API request for {city} {days}-day forecast...")
            return {
                "city": city,
                "days": days,
                "forecast": ["sunny"] * days
            }
    
    api = WeatherAPIClient()
    
    print("\nGetting weather for New York:")
    weather1 = api.get_weather("New York")
    print(f"Weather: {weather1['temp']}°F, {weather1['condition']}")
    
    print("\nGetting weather for New York again:")
    weather2 = api.get_weather("New York")
    print(f"Weather: {weather2['temp']}°F (cached)")
    
    print("\nGetting forecast for New York:")
    forecast1 = api.get_forecast("New York", days=7)
    print(f"Forecast: {len(forecast1['forecast'])} days")
    
    print(f"\nTotal API requests made: {api.request_count}")
    print("(Without caching, would have made 3 requests)")
    
    # ========================================================================
    # 7. Combining decorators
    # ========================================================================
    print("\n" + "-"*80)
    print("7. DECORATOR WITH DISK CACHE")
    print("-"*80)
    
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    disk_cache = create_disk_cache(
        cache_dir=os.path.join(temp_dir, "func_cache")
    )
    
    @cached(cache=disk_cache)

# %%
# Load Model Weights
# ------------------

    def load_model_weights(model_name):
        """Load model weights (expensive operation)."""
        print(f"    Loading weights for {model_name}...")
        return {"model": model_name, "weights": [1.0, 2.0, 3.0]}
    
    print("\nLoading model weights:")
    weights1 = load_model_weights("bert-base")
    print(f"Loaded: {weights1['model']}")
    
    print("\nLoading same model again:")
    weights2 = load_model_weights("bert-base")
    print(f"Loaded: {weights2['model']} (from disk cache)")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ========================================================================
    # 8. Advanced: Cache key with multiple arguments
    # ========================================================================
    print("\n" + "-"*80)
    print("8. MULTIPLE ARGUMENTS CACHING")
    print("-"*80)
    
    @cached()
    def compute_similarity(text1, text2, method="cosine"):
        """Compute text similarity."""
        print(f"    Computing {method} similarity...")
        # Mock computation
        return 0.85
    
    print("\nComputing similarity('hello', 'hi', 'cosine'):")
    sim1 = compute_similarity("hello", "hi", method="cosine")
    print(f"Similarity: {sim1}")
    
    print("\nComputing similarity('hello', 'hi', 'cosine') again:")
    sim2 = compute_similarity("hello", "hi", method="cosine")
    print(f"Similarity: {sim2} (cached)")
    
    print("\nComputing similarity('hello', 'hi', 'euclidean'):")
    sim3 = compute_similarity("hello", "hi", method="euclidean")
    print(f"Similarity: {sim3} (different method, not cached)")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • @cached decorator enables transparent function caching")
    print("  • Works with any function or method")
    print("  • Automatically generates cache keys from arguments")
    print("  • Support custom key generation with key_fn")
    print("  • Can specify TTL per decorator")
    print("  • Can use custom cache backends (memory, disk, tiered)")
    print("  • Perfect for: API calls, expensive computations, I/O operations")


if __name__ == "__main__":
    main()
