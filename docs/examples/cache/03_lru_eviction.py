"""LRU Eviction Example

This example demonstrates Least Recently Used (LRU) cache eviction.

Main concepts:
- Understanding max_size limits
- LRU eviction policy (oldest accessed entries removed first)
- How access patterns affect eviction
- Preventing eviction by accessing entries
- Practical use cases for LRU caching
"""

from kerb.cache import create_memory_cache


def main():
    """Run LRU eviction example."""
    
    print("="*80)
    print("LRU EVICTION EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Basic LRU eviction
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC LRU EVICTION")
    print("-"*80)
    
    # Create small cache that can hold only 3 items
    cache = create_memory_cache(max_size=3)
    
    print("Created cache with max_size=3")
    print("\nAdding 3 items:")
    
    cache.set("first", "value1")
    print(f"  Added 'first', cache keys: {cache.keys()}")
    
    cache.set("second", "value2")
    print(f"  Added 'second', cache keys: {cache.keys()}")
    
    cache.set("third", "value3")
    print(f"  Added 'third', cache keys: {cache.keys()}")
    
    print(f"\nCache is now full (3/3 items)")
    
    # Add 4th item - should evict 'first' (least recently used)
    print("\nAdding 4th item (will trigger eviction):")
    cache.set("fourth", "value4")
    print(f"  Added 'fourth', cache keys: {cache.keys()}")
    print(f"  'first' was evicted (least recently used)")
    
    # Verify 'first' is gone
    print(f"\nTrying to get 'first': {cache.get('first')}")
    
    # ========================================================================
    # 2. Accessing prevents eviction
    # ========================================================================
    print("\n" + "-"*80)
    print("2. ACCESSING PREVENTS EVICTION")
    print("-"*80)
    
    cache2 = create_memory_cache(max_size=3)
    
    # Add 3 items
    cache2.set("a", "value_a")
    cache2.set("b", "value_b")
    cache2.set("c", "value_c")
    
    print("Added 3 items: a, b, c")
    print(f"Cache keys: {cache2.keys()}")
    
    # Access 'a' to make it recently used
    print("\nAccessing 'a' to mark it as recently used...")
    _ = cache2.get("a")
    
    # Add 4th item - should evict 'b' now (not 'a')
    print("\nAdding 4th item:")
    cache2.set("d", "value_d")
    print(f"Cache keys: {cache2.keys()}")
    print("'b' was evicted (least recently used)")
    print("'a' survived because we accessed it")
    
    # ========================================================================
    # 3. Update doesn't trigger eviction
    # ========================================================================
    print("\n" + "-"*80)
    print("3. UPDATE DOESN'T TRIGGER EVICTION")
    print("-"*80)
    
    cache3 = create_memory_cache(max_size=3)
    
    cache3.set("x", "value1")
    cache3.set("y", "value2")
    cache3.set("z", "value3")
    
    print("Cache full with keys: x, y, z")
    
    # Update existing key
    cache3.set("x", "updated_value")
    print("\nUpdated 'x' to new value")
    print(f"Cache keys (no eviction): {cache3.keys()}")
    print(f"Value of 'x': {cache3.get('x')}")
    
    # ========================================================================
    # 4. LRU with access patterns
    # ========================================================================
    print("\n" + "-"*80)
    print("4. LRU WITH ACCESS PATTERNS")
    print("-"*80)
    
    cache4 = create_memory_cache(max_size=5)
    
    # Simulate a workload
    print("Simulating workload with max_size=5:")
    print("\nAdding 5 items:")
    for i in range(1, 6):
        cache4.set(f"item{i}", f"value{i}")
        print(f"  Added item{i}, cache keys: {cache4.keys()}")
    
    print("\nAccessing item1, item2, item3 (making them recently used):")
    cache4.get("item1")
    cache4.get("item2")
    cache4.get("item3")
    
    print("\nAdding 2 more items (will evict item4 and item5):")
    cache4.set("item6", "value6")
    print(f"  Added item6, cache keys: {cache4.keys()}")
    
    cache4.set("item7", "value7")
    print(f"  Added item7, cache keys: {cache4.keys()}")
    
    print("\nitem4 and item5 were evicted (least recently used)")
    print("item1, item2, item3 survived because we accessed them")
    
    # ========================================================================
    # 5. Practical: Hot data caching
    # ========================================================================
    print("\n" + "-"*80)
    print("5. PRACTICAL: HOT DATA CACHING")
    print("-"*80)
    
    # Simulate an API cache with limited size
    api_cache = create_memory_cache(max_size=100)
    
    print("Simulating API response cache (max_size=100)")
    
    # Add popular endpoints (frequently accessed)
    popular_endpoints = ["users", "posts", "comments"]
    for endpoint in popular_endpoints:
        api_cache.set(f"api:{endpoint}", {"data": f"{endpoint} response"})
        print(f"  Cached popular endpoint: api:{endpoint}")
    
    # Add many other endpoints
    print("\nAdding 100 other endpoints...")
    for i in range(100):
        api_cache.set(f"api:misc:{i}", {"data": f"response {i}"})
    
    # Access popular endpoints again
    print("\nAccessing popular endpoints frequently...")
    for endpoint in popular_endpoints:
        _ = api_cache.get(f"api:{endpoint}")
    
    # Add more endpoints
    print("\nAdding 20 more endpoints (will trigger evictions)...")
    for i in range(100, 120):
        api_cache.set(f"api:misc:{i}", {"data": f"response {i}"})
    
    # Check if popular endpoints survived
    print("\nChecking if popular endpoints are still cached:")
    for endpoint in popular_endpoints:
        exists = api_cache.exists(f"api:{endpoint}")
        status = "✓ Cached" if exists else "✗ Evicted"
        print(f"  api:{endpoint}: {status}")
    
    print("\nPopular endpoints survived due to frequent access!")
    
    # ========================================================================
    # 6. Demonstrating LRU order
    # ========================================================================
    print("\n" + "-"*80)
    print("6. DEMONSTRATING LRU ORDER")
    print("-"*80)
    
    lru_cache = create_memory_cache(max_size=4)
    
    print("Adding items in order: A, B, C, D")
    lru_cache.set("A", 1)
    lru_cache.set("B", 2)
    lru_cache.set("C", 3)
    lru_cache.set("D", 4)
    print(f"Cache keys: {lru_cache.keys()}")
    
    print("\nAccessing in order: B, A (making them recently used)")
    lru_cache.get("B")
    lru_cache.get("A")
    
    print("\nAdding E (will evict C, the least recently used)")
    lru_cache.set("E", 5)
    print(f"Cache keys: {lru_cache.keys()}")
    
    print("\nAdding F (will evict D, now the least recently used)")
    lru_cache.set("F", 6)
    print(f"Cache keys: {lru_cache.keys()}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • LRU evicts least recently used items when max_size is reached")
    print("  • Accessing (get) or updating (set) an entry marks it as recently used")
    print("  • Frequently accessed items stay in cache longer")
    print("  • Perfect for caching hot data with limited memory")


if __name__ == "__main__":
    main()
