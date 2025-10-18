"""TTL and Expiration Example

This example demonstrates time-to-live (TTL) and automatic cache expiration.

Main concepts:
- Setting default TTL for a cache
- Setting custom TTL per entry
- Understanding automatic expiration
- Retrieving expired entries (returns None)
- Using TTL for different data types (sessions, temporary data, etc.)
"""

import time
from kerb.cache import create_memory_cache


def main():
    """Run TTL and expiration example."""
    
    print("="*80)
    print("TTL AND EXPIRATION EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Cache with default TTL
    # ========================================================================
    print("\n" + "-"*80)
    print("1. CACHE WITH DEFAULT TTL")
    print("-"*80)
    
    # Create cache with 3-second default TTL
    cache = create_memory_cache(max_size=100, default_ttl=3.0)
    
    print("Created cache with default_ttl=3.0 seconds")
    
    cache.set("session:user123", {"user_id": 123, "logged_in": True})
    print("\nStored session:user123")
    
    # Immediately retrieve
    session = cache.get("session:user123")
    print(f"Immediately after: {session}")
    
    # Wait 1 second
    print("\nWaiting 1 second...")
    time.sleep(1)
    session = cache.get("session:user123")
    print(f"After 1 second: {session}")
    
    # Wait 3 more seconds (total 4 seconds, past TTL)
    print("\nWaiting 3 more seconds (past TTL)...")
    time.sleep(3)
    session = cache.get("session:user123")
    print(f"After 4 seconds (expired): {session}")
    
    # ========================================================================
    # 2. Custom TTL per entry
    # ========================================================================
    print("\n" + "-"*80)
    print("2. CUSTOM TTL PER ENTRY")
    print("-"*80)
    
    cache_no_default = create_memory_cache(max_size=100)
    
    # Set entries with different TTLs
    cache_no_default.set("quick", "expires in 1s", ttl=1.0)
    cache_no_default.set("medium", "expires in 2s", ttl=2.0)
    cache_no_default.set("long", "expires in 5s", ttl=5.0)
    cache_no_default.set("permanent", "never expires", ttl=None)
    
    print("Set 4 entries with different TTLs:")
    print("  - quick: TTL=1s")
    print("  - medium: TTL=2s")
    print("  - long: TTL=5s")
    print("  - permanent: TTL=None (never expires)")
    
    print("\nImmediately after setting:")
    print(f"  quick: {cache_no_default.get('quick')}")
    print(f"  medium: {cache_no_default.get('medium')}")
    print(f"  long: {cache_no_default.get('long')}")
    print(f"  permanent: {cache_no_default.get('permanent')}")
    
    print("\nWaiting 1.5 seconds...")
    time.sleep(1.5)
    print(f"  quick (expired): {cache_no_default.get('quick')}")
    print(f"  medium (alive): {cache_no_default.get('medium')}")
    print(f"  long (alive): {cache_no_default.get('long')}")
    print(f"  permanent (alive): {cache_no_default.get('permanent')}")
    
    print("\nWaiting another 1 second (total 2.5s)...")
    time.sleep(1)
    print(f"  quick (expired): {cache_no_default.get('quick')}")
    print(f"  medium (expired): {cache_no_default.get('medium')}")
    print(f"  long (alive): {cache_no_default.get('long')}")
    print(f"  permanent (alive): {cache_no_default.get('permanent')}")
    
    # ========================================================================
    # 3. Practical TTL use cases
    # ========================================================================
    print("\n" + "-"*80)
    print("3. PRACTICAL TTL USE CASES")
    print("-"*80)
    
    app_cache = create_memory_cache(max_size=1000)
    
    # Short-lived: Session tokens (5 minutes)
    app_cache.set("session:abc123", 
                  {"token": "xyz", "user": "alice"},
                  ttl=300)  # 5 minutes
    print("✓ Session token cached (TTL: 5 minutes)")
    
    # Medium-lived: API responses (1 hour)
    app_cache.set("api:weather:nyc",
                  {"temp": 72, "condition": "sunny"},
                  ttl=3600)  # 1 hour
    print("✓ Weather API response cached (TTL: 1 hour)")
    
    # Long-lived: Daily reports (24 hours)
    app_cache.set("report:daily:2024-01-15",
                  {"views": 1000, "clicks": 50},
                  ttl=86400)  # 24 hours
    print("✓ Daily report cached (TTL: 24 hours)")
    
    # Permanent: Configuration
    app_cache.set("config:app",
                  {"theme": "dark", "lang": "en"},
                  ttl=None)  # Never expires
    print("✓ App configuration cached (no expiration)")
    
    # ========================================================================
    # 4. Checking expiration status
    # ========================================================================
    print("\n" + "-"*80)
    print("4. EXPIRATION STATUS")
    print("-"*80)
    
    test_cache = create_memory_cache()
    
    # Set a value with 2 second TTL
    test_cache.set("test", "value", ttl=2.0)
    
    print("Set test entry with 2-second TTL")
    print(f"Exists immediately: {test_cache.exists('test')}")
    
    time.sleep(1)
    print(f"Exists after 1s: {test_cache.exists('test')}")
    
    time.sleep(1.5)
    print(f"Exists after 2.5s (expired): {test_cache.exists('test')}")
    
    # ========================================================================
    # 5. Overriding default TTL
    # ========================================================================
    print("\n" + "-"*80)
    print("5. OVERRIDING DEFAULT TTL")
    print("-"*80)
    
    cache_with_default = create_memory_cache(default_ttl=2.0)
    
    # Uses default TTL
    cache_with_default.set("uses_default", "value1")
    
    # Override with longer TTL
    cache_with_default.set("custom_longer", "value2", ttl=5.0)
    
    # Override with no expiration
    cache_with_default.set("no_expiry", "value3", ttl=None)
    
    print("Set 3 entries:")
    print("  - uses_default: uses default (2s)")
    print("  - custom_longer: custom (5s)")
    print("  - no_expiry: no expiration (None)")
    
    time.sleep(2.5)
    print("\nAfter 2.5 seconds:")
    print(f"  uses_default (expired): {cache_with_default.get('uses_default')}")
    print(f"  custom_longer (alive): {cache_with_default.get('custom_longer')}")
    print(f"  no_expiry (alive): {cache_with_default.get('no_expiry')}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • TTL controls automatic expiration")
    print("  • Set default_ttl on cache creation")
    print("  • Override TTL per entry with ttl parameter")
    print("  • TTL=None means no expiration")
    print("  • Expired entries return None and are auto-deleted")


if __name__ == "__main__":
    main()
