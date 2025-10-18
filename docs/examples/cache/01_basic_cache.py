"""Basic Cache Example

This example demonstrates fundamental cache operations.

Main concepts:
- Creating a memory cache
- Setting and getting values
- Checking key existence
- Deleting entries
- Listing cache keys
- Clearing the cache
"""

from kerb.cache import create_memory_cache


def main():
    """Run basic cache example."""
    
    print("="*80)
    print("BASIC CACHE EXAMPLE")
    print("="*80)
    
    # Create a simple memory cache
    cache = create_memory_cache(max_size=100)
    
    print(f"\nCreated memory cache with max_size=100")
    
    # ========================================================================
    # 1. Setting values
    # ========================================================================
    print("\n" + "-"*80)
    print("1. SETTING VALUES")
    print("-"*80)
    
    # Store simple values
    cache.set("user:alice", {"name": "Alice", "role": "admin"})
    cache.set("user:bob", {"name": "Bob", "role": "user"})
    cache.set("config:theme", "dark")
    cache.set("config:language", "en")
    
    print("Stored 4 values in cache:")
    print("  - user:alice -> {'name': 'Alice', 'role': 'admin'}")
    print("  - user:bob -> {'name': 'Bob', 'role': 'user'}")
    print("  - config:theme -> 'dark'")
    print("  - config:language -> 'en'")
    
    # ========================================================================
    # 2. Getting values
    # ========================================================================
    print("\n" + "-"*80)
    print("2. GETTING VALUES")
    print("-"*80)
    
    alice = cache.get("user:alice")
    theme = cache.get("config:theme")
    missing = cache.get("nonexistent")
    
    print(f"cache.get('user:alice') -> {alice}")
    print(f"cache.get('config:theme') -> {theme}")
    print(f"cache.get('nonexistent') -> {missing}")
    
    # ========================================================================
    # 3. Checking existence
    # ========================================================================
    print("\n" + "-"*80)
    print("3. CHECKING EXISTENCE")
    print("-"*80)
    
    print(f"cache.exists('user:alice') -> {cache.exists('user:alice')}")
    print(f"cache.exists('user:charlie') -> {cache.exists('user:charlie')}")
    
    # ========================================================================
    # 4. Listing keys
    # ========================================================================
    print("\n" + "-"*80)
    print("4. LISTING KEYS")
    print("-"*80)
    
    all_keys = cache.keys()
    print(f"cache.keys() -> {all_keys}")
    print(f"Total entries: {cache.size()}")
    
    # ========================================================================
    # 5. Deleting entries
    # ========================================================================
    print("\n" + "-"*80)
    print("5. DELETING ENTRIES")
    print("-"*80)
    
    deleted = cache.delete("config:theme")
    print(f"cache.delete('config:theme') -> {deleted}")
    print(f"Remaining keys: {cache.keys()}")
    
    # Try to delete non-existent key
    deleted = cache.delete("nonexistent")
    print(f"cache.delete('nonexistent') -> {deleted}")
    
    # ========================================================================
    # 6. Cache size
    # ========================================================================
    print("\n" + "-"*80)
    print("6. CACHE SIZE")
    print("-"*80)
    
    print(f"Current cache size: {cache.size()} entries")
    
    # ========================================================================
    # 7. Clearing cache
    # ========================================================================
    print("\n" + "-"*80)
    print("7. CLEARING CACHE")
    print("-"*80)
    
    print(f"Before clear: {cache.size()} entries")
    cache.clear()
    print(f"After clear: {cache.size()} entries")
    print(f"Keys after clear: {cache.keys()}")
    
    # ========================================================================
    # 8. Working with complex data
    # ========================================================================
    print("\n" + "-"*80)
    print("8. COMPLEX DATA TYPES")
    print("-"*80)
    
    # Store various Python objects
    cache.set("list", [1, 2, 3, 4, 5])
    cache.set("dict", {"a": 1, "b": 2, "nested": {"c": 3}})
    cache.set("tuple", (10, 20, 30))
    
    print(f"Stored list: {cache.get('list')}")
    print(f"Stored dict: {cache.get('dict')}")
    print(f"Stored tuple: {cache.get('tuple')}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
