"""
Disk Cache Example
==================

This example demonstrates persistent disk-based caching.

Main concepts:
- Creating a disk cache with persistent storage
- Data survives program restarts
- Different serialization formats (pickle, json)
- Managing cache files on disk
- Use cases for persistent caching
"""

import os
import tempfile
from pathlib import Path
from kerb.cache import create_disk_cache


def main():
    """Run disk cache example."""
    
    print("="*80)
    print("DISK CACHE EXAMPLE")
    print("="*80)
    
    # Use a temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, "cache_demo")
    
    print(f"Using cache directory: {cache_dir}")
    
    # ========================================================================
    # 1. Basic disk cache
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC DISK CACHE")
    print("-"*80)
    
    # Create disk cache (uses pickle by default)
    cache = create_disk_cache(
        cache_dir=cache_dir,
        max_size=100
    )
    
    print("Created disk cache with pickle serialization")
    
    # Store data
    cache.set("user:alice", {"name": "Alice", "age": 30})
    cache.set("user:bob", {"name": "Bob", "age": 25})
    cache.set("config", {"theme": "dark", "language": "en"})
    
    print("\nStored 3 entries:")
    print(f"  Keys: {cache.keys()}")
    
    # Retrieve data
    alice = cache.get("user:alice")
    print(f"\nRetrieved user:alice: {alice}")
    
    # Check disk storage
    cache_files = list(Path(cache_dir).glob("*"))
    print(f"\nCache files on disk: {len(cache_files)}")
    
    # ========================================================================
    # 2. Persistence across sessions
    # ========================================================================
    print("\n" + "-"*80)
    print("2. PERSISTENCE ACROSS SESSIONS")
    print("-"*80)
    
    print("Simulating program restart...")
    
    # "Close" the first cache (in practice, this is program exit)
    del cache
    
    # Create new cache instance pointing to same directory
    cache_restored = create_disk_cache(cache_dir=cache_dir)
    
    print("\nCreated new cache instance")
    print(f"Keys from previous session: {cache_restored.keys()}")
    
    # Data is still there!
    alice_restored = cache_restored.get("user:alice")
    print(f"Retrieved persisted data: {alice_restored}")
    
    # ========================================================================
    # 3. JSON serialization
    # ========================================================================
    print("\n" + "-"*80)
    print("3. JSON SERIALIZATION")
    print("-"*80)
    
    json_cache_dir = os.path.join(temp_dir, "json_cache")
    
    # Use JSON serializer (human-readable, but limited to JSON-compatible types)
    json_cache = create_disk_cache(
        cache_dir=json_cache_dir,
        serializer="json"
    )
    
    print("Created disk cache with JSON serialization")
    
    # Store JSON-compatible data
    json_cache.set("api:response", {
        "status": 200,
        "data": {"items": [1, 2, 3]},
        "timestamp": "2024-01-15T10:30:00"
    })
    
    print("\nStored API response")
    
    # JSON files are human-readable
    json_files = list(Path(json_cache_dir).glob("*.json"))
    if json_files:
        print(f"JSON cache file created: {json_files[0].name}")
    
    # ========================================================================
    # 4. Complex Python objects (pickle only)
    # ========================================================================
    print("\n" + "-"*80)
    print("4. COMPLEX PYTHON OBJECTS (PICKLE)")
    print("-"*80)
    
    pickle_cache = create_disk_cache(
        cache_dir=os.path.join(temp_dir, "pickle_cache"),
        serializer="pickle"
    )
    
    print("Created cache with pickle serialization")
    
    # Pickle can handle complex Python objects like lists, dicts, tuples
    complex_data = {
        "name": "example",
        "values": [1, 2, 3],
        "nested": {
            "items": [(1, "a"), (2, "b")],
            "metadata": {"created": "2024-01-15"}
        }
    }
    
    # Store complex object
    pickle_cache.set("complex:data", complex_data)
    
    print(f"\nStored complex data structure")
    
    # Retrieve it
    retrieved_data = pickle_cache.get("complex:data")
    print(f"Retrieved data: {retrieved_data}")
    print(f"Type: {type(retrieved_data).__name__}")
    
    # ========================================================================
    # 5. Practical: Caching expensive computations
    # ========================================================================
    print("\n" + "-"*80)
    print("5. PRACTICAL: CACHING EXPENSIVE COMPUTATIONS")
    print("-"*80)
    
    computation_cache = create_disk_cache(
        cache_dir=os.path.join(temp_dir, "computation_cache")
    )
    
    def expensive_computation(n):
        """Simulate expensive computation."""

# %%
# Setup and Imports
# -----------------
        # In real scenario, this might be model inference, data processing, etc.
        result = sum(range(n))
        return {"input": n, "result": result, "computed": True}
    
    print("Checking cache for computation result...")
    
    # Check if result is cached
    cache_key = "compute:sum:1000000"
    result = computation_cache.get(cache_key)
    
    if result is None:
        print("  Cache miss - computing...")
        result = expensive_computation(1000000)
        computation_cache.set(cache_key, result)
        print(f"  Computed and cached: {result}")
    else:
        print(f"  Cache hit - using cached result: {result}")
    
    # Second call uses cache
    print("\nSecond call (should hit cache)...")
    result2 = computation_cache.get(cache_key)
    if result2:
        print(f"  Cache hit! Result: {result2}")
    
    # ========================================================================
    # 6. Cache management
    # ========================================================================
    print("\n" + "-"*80)
    print("6. CACHE MANAGEMENT")
    print("-"*80)
    
    mgmt_cache = create_disk_cache(
        cache_dir=os.path.join(temp_dir, "mgmt_cache")
    )
    
    # Add some data
    for i in range(5):
        mgmt_cache.set(f"item:{i}", f"value_{i}")
    
    print(f"Added 5 items, cache size: {mgmt_cache.size()}")
    
    # Delete specific item
    mgmt_cache.delete("item:2")
    print(f"Deleted item:2, cache size: {mgmt_cache.size()}")
    
    # Clear all
    print("\nClearing cache...")
    mgmt_cache.clear()
    print(f"After clear, cache size: {mgmt_cache.size()}")
    
    # Check disk
    disk_files = list(Path(os.path.join(temp_dir, "mgmt_cache")).glob("*"))
    print(f"Files on disk: {len(disk_files)}")
    
    # ========================================================================
    # 7. Disk cache with TTL
    # ========================================================================
    print("\n" + "-"*80)
    print("7. DISK CACHE WITH TTL")
    print("-"*80)
    
    ttl_cache = create_disk_cache(
        cache_dir=os.path.join(temp_dir, "ttl_cache"),
        default_ttl=3600  # 1 hour
    )
    
    print("Created disk cache with 1-hour TTL")
    
    # Even persistent cache can have expiration
    ttl_cache.set("session:temp", {"data": "temporary"}, ttl=300)  # 5 minutes
    ttl_cache.set("config:permanent", {"data": "permanent"}, ttl=None)  # No expiry
    
    print("\nStored entries with different TTLs:")
    print("  • session:temp - 5 minutes")
    print("  • config:permanent - no expiration")
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Cleaned up temporary directory: {temp_dir}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • Disk cache persists data across program restarts")
    print("  • Pickle serializer handles any Python object")
    print("  • JSON serializer creates human-readable files")
    print("  • Perfect for caching expensive computations")
    print("  • Supports TTL for automatic expiration")
    print("  • Use for: model weights, preprocessed data, API responses")


if __name__ == "__main__":
    main()
