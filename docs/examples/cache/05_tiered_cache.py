"""Tiered Cache Example

This example demonstrates two-tier caching (memory + disk).

Main concepts:
- Fast memory cache for hot data
- Persistent disk cache for cold data
- Automatic data promotion/demotion between tiers
- Best of both worlds: speed + persistence
- Production-ready caching strategy
"""

import os
import tempfile
from kerb.cache import create_tiered_cache


def main():
    """Run tiered cache example."""
    
    print("="*80)
    print("TIERED CACHE EXAMPLE")
    print("="*80)
    
    # Use temp directory for demo
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, "tiered_cache")
    
    # ========================================================================
    # 1. Creating a tiered cache
    # ========================================================================
    print("\n" + "-"*80)
    print("1. CREATING A TIERED CACHE")
    print("-"*80)
    
    # Tiered cache: small fast memory cache + large persistent disk cache
    cache = create_tiered_cache(
        memory_max_size=5,      # Only 5 items in fast memory
        disk_cache_dir=cache_dir,
        disk_max_size=100       # 100 items on disk
    )
    
    print("Created tiered cache:")
    print("  • Memory tier: max_size=5 (fast, volatile)")
    print("  • Disk tier: max_size=100 (slower, persistent)")
    
    # ========================================================================
    # 2. Automatic tiering
    # ========================================================================
    print("\n" + "-"*80)
    print("2. AUTOMATIC TIERING")
    print("-"*80)
    
    # Add items - first 5 go to memory
    print("\nAdding 8 items...")
    for i in range(8):
        cache.set(f"item:{i}", {"id": i, "data": f"value_{i}"})
        print(f"  Added item:{i}")
    
    print(f"\nTotal cache size: {cache.size()} items")
    
    # Access an item - it's automatically retrieved from the right tier
    print("\nAccessing items:")
    item_0 = cache.get("item:0")
    print(f"  item:0 (likely in disk tier): {item_0}")
    
    item_7 = cache.get("item:7")
    print(f"  item:7 (likely in memory tier): {item_7}")
    
    print("\nTiered cache automatically:")
    print("  • Stores hot data in memory (fast access)")
    print("  • Moves cold data to disk (saves memory)")
    print("  • Promotes frequently accessed data to memory")
    
    # ========================================================================
    # 3. Access patterns and promotion
    # ========================================================================
    print("\n" + "-"*80)
    print("3. ACCESS PATTERNS AND PROMOTION")
    print("-"*80)
    
    cache2 = create_tiered_cache(
        memory_max_size=3,
        disk_cache_dir=os.path.join(temp_dir, "tier_demo")
    )
    
    # Add more items than memory can hold
    print("\nAdding 10 items to cache (memory holds only 3):")
    for i in range(10):
        cache2.set(f"data:{i}", f"content_{i}")
    
    print(f"Total items: {cache2.size()}")
    print("Items 0-6 moved to disk, items 7-9 in memory")
    
    # Frequently access an old item
    print("\nFrequently accessing 'data:2' (from disk)...")
    for _ in range(3):
        _ = cache2.get("data:2")
    
    print("data:2 gets promoted to memory tier due to frequent access")
    
    # ========================================================================
    # 4. Persistence across restarts
    # ========================================================================
    print("\n" + "-"*80)
    print("4. PERSISTENCE ACROSS RESTARTS")
    print("-"*80)
    
    persist_dir = os.path.join(temp_dir, "persistent")
    
    # First session
    print("Session 1: Creating cache and storing data...")
    cache_session1 = create_tiered_cache(
        memory_max_size=5,
        disk_cache_dir=persist_dir
    )
    
    cache_session1.set("important:config", {"setting": "value"})
    cache_session1.set("user:data", {"name": "Alice"})
    
    print(f"  Stored 2 items")
    print(f"  Keys: {cache_session1.keys()}")
    
    # Simulate program restart
    print("\nSimulating program restart...")
    del cache_session1
    
    # Second session
    print("\nSession 2: Recreating cache from disk...")
    cache_session2 = create_tiered_cache(
        memory_max_size=5,
        disk_cache_dir=persist_dir
    )
    
    print(f"  Restored keys: {cache_session2.keys()}")
    config = cache_session2.get("important:config")
    print(f"  Retrieved config: {config}")
    print("\n  ✓ Data survived restart (from disk tier)!")
    
    # ========================================================================
    # 5. Performance characteristics
    # ========================================================================
    print("\n" + "-"*80)
    print("5. PERFORMANCE CHARACTERISTICS")
    print("-"*80)
    
    perf_cache = create_tiered_cache(
        memory_max_size=100,
        disk_cache_dir=os.path.join(temp_dir, "perf")
    )
    
    import time
    
    # Measure memory tier access
    perf_cache.set("memory:item", "value")
    start = time.time()
    for _ in range(1000):
        _ = perf_cache.get("memory:item")
    memory_time = time.time() - start
    
    print(f"\nMemory tier: 1000 accesses in {memory_time*1000:.2f}ms")
    print("  → Very fast (in-memory)")
    
    # Add many items to push first item to disk
    for i in range(150):
        perf_cache.set(f"item:{i}", f"value_{i}")
    
    # Measure disk tier access (for older items)
    start = time.time()
    for _ in range(100):
        _ = perf_cache.get("item:10")
    disk_time = time.time() - start
    
    print(f"\nDisk tier: 100 accesses in {disk_time*1000:.2f}ms")
    print("  → Slower but still cached (disk I/O)")
    
    # ========================================================================
    # 6. Production use case: API caching
    # ========================================================================
    print("\n" + "-"*80)
    print("6. PRODUCTION USE CASE: API CACHING")
    print("-"*80)
    
    # Realistic production setup
    api_cache = create_tiered_cache(
        memory_max_size=1000,    # 1000 hot responses in memory
        disk_cache_dir=os.path.join(temp_dir, "api_cache"),
        disk_max_size=10000,     # 10,000 total responses
        default_ttl=3600         # 1 hour TTL
    )
    
    print("API Cache Configuration:")
    print("  • Memory: 1,000 most recent/frequent requests")
    print("  • Disk: 10,000 total cached responses")
    print("  • TTL: 1 hour")
    
    # Simulate API requests
    def mock_api_call(endpoint):
        """Simulate API call."""
        return {"endpoint": endpoint, "data": "response", "timestamp": time.time()}
    
    def get_with_cache(endpoint):
        """Get API response with caching."""
        cached = api_cache.get(f"api:{endpoint}")
        if cached:
            print(f"  ✓ Cache hit: {endpoint}")
            return cached
        
        print(f"  ✗ Cache miss: {endpoint} (calling API)")
        response = mock_api_call(endpoint)
        api_cache.set(f"api:{endpoint}", response)
        return response
    
    print("\nSimulating API requests:")
    
    # First request - cache miss
    get_with_cache("users/123")
    
    # Second request - cache hit (from memory)
    get_with_cache("users/123")
    
    # Many other requests
    for i in range(20):
        get_with_cache(f"data/{i}")
    
    # Original request - still cached (might be in disk now)
    get_with_cache("users/123")
    
    print(f"\nTotal cached responses: {api_cache.size()}")
    
    # ========================================================================
    # 7. Memory optimization
    # ========================================================================
    print("\n" + "-"*80)
    print("7. MEMORY OPTIMIZATION")
    print("-"*80)
    
    # Small memory footprint, large disk storage
    optimized_cache = create_tiered_cache(
        memory_max_size=10,      # Only 10 items in RAM
        disk_cache_dir=os.path.join(temp_dir, "optimized"),
        disk_max_size=10000      # But 10,000 on disk
    )
    
    print("Memory-optimized configuration:")
    print("  • RAM: Only 10 items (~KB)")
    print("  • Disk: 10,000 items (~MB/GB)")
    print("\nBenefits:")
    print("  • Low memory usage")
    print("  • Large cache capacity")
    print("  • Hot data still fast")
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Cleaned up temporary directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • Tiered cache = memory speed + disk persistence")
    print("  • Hot data automatically stays in memory")
    print("  • Cold data moved to disk to save RAM")
    print("  • Transparent access - cache handles tier management")
    print("  • Perfect for production: fast, persistent, memory-efficient")
    print("  • Use for: API caching, LLM responses, large datasets")


if __name__ == "__main__":
    main()
