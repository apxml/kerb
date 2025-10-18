"""Cache Monitoring and Statistics Example

This example demonstrates monitoring cache performance and statistics.

Ke    print(f"\nDetailed Statistics:")
    print(f"  Requests:     {stats.total_requests}")
    print(f"  Hits:         {stats.hits}")
    print(f"  Misses:       {stats.misses}")
    print(f"  Hit rate:     {stats.hit_rate:.2%}")
    print(f"  Cache size:   {stats.size}")
    print(f"  Evictions:    {stats.evictions}")pts:
- Getting cache statistics (hits, misses, hit rate)
- Monitoring cache performance
- Analyzing cache efficiency
- Exporting cache statistics
- Using stats for optimization decisions
"""

import time
from kerb.cache import (
    create_memory_cache,
    create_tiered_cache,
    export_cache_stats,
)


def main():
    """Run cache monitoring and statistics example."""
    
    print("="*80)
    print("CACHE MONITORING AND STATISTICS EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Basic cache statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC CACHE STATISTICS")
    print("-"*80)
    
    cache = create_memory_cache(max_size=100)
    
    # Simulate some cache operations
    print("\nSimulating cache operations...")
    
    # Some hits
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    _ = cache.get("key1")  # Hit
    _ = cache.get("key1")  # Hit
    _ = cache.get("key2")  # Hit
    _ = cache.get("key99") # Miss
    _ = cache.get("key99") # Miss
    
    # Get statistics
    stats = cache.get_stats()
    
    print("\nCache Statistics:")
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Hits:           {stats.hits}")
    print(f"  Misses:         {stats.misses}")
    print(f"  Hit rate:       {stats.hit_rate:.2%}")
    print(f"  Cache size:     {stats.size} entries")
    
    # ========================================================================
    # 2. Monitoring cache performance over time
    # ========================================================================
    print("\n" + "-"*80)
    print("2. MONITORING PERFORMANCE OVER TIME")
    print("-"*80)
    
    monitor_cache = create_memory_cache(max_size=50)
    
    print("\nSimulating workload...")
    
    # Simulate a workload pattern
    requests = [
        "popular1", "popular1", "popular1",  # Frequently accessed
        "popular2", "popular2",
        "rare1",                              # Rarely accessed
        "popular1", "popular1",
        "rare2",
        "popular2",
        "rare3", "rare4", "rare5",
        "popular1", "popular2",
    ]
    
    for req in requests:
        monitor_cache.set(req, f"data_{req}")
        _ = monitor_cache.get(req)
    
    stats = monitor_cache.get_stats()
    
    print(f"\nWorkload Analysis:")
    print(f"  Total requests:   {stats.total_requests}")
    print(f"  Hit rate:         {stats.hit_rate:.2%}")
    print(f"  Cache efficiency: {'Good' if stats.hit_rate > 0.5 else 'Needs improvement'}")
    
    # ========================================================================
    # 3. Detailed statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("3. DETAILED STATISTICS")
    print("-"*80)
    
    detail_cache = create_memory_cache(max_size=100)
    
    # Add entries with metadata
    detail_cache.set("api:users", {"data": "users"}, metadata={"cost": 0.01})
    detail_cache.set("api:posts", {"data": "posts"}, metadata={"cost": 0.005})
    detail_cache.set("api:comments", {"data": "comments"}, metadata={"cost": 0.002})
    
    # Simulate access pattern
    for _ in range(5):
        _ = detail_cache.get("api:users")    # Popular
    for _ in range(2):
        _ = detail_cache.get("api:posts")    # Less popular
    _ = detail_cache.get("api:comments")     # Least popular
    _ = detail_cache.get("api:unknown")      # Miss
    
    stats = detail_cache.get_stats()
    
    print("\nDetailed Statistics:")
    print(f"  Requests:     {stats.total_requests}")
    print(f"  Hits:         {stats.hits}")
    print(f"  Misses:       {stats.misses}")
    print(f"  Hit rate:     {stats.hit_rate:.2%}")
    print(f"  Cache size:   {stats.size}")
    print(f"  Evictions:    {stats.evictions}")
    
    # Calculate savings (if cost metadata is used)
    print(f"\nCost Analysis:")
    print(f"  API calls saved: {stats.hits}")
    print(f"  Estimated savings: ${stats.hits * 0.005:.4f}")
    
    # ========================================================================
    # 4. Comparing cache configurations
    # ========================================================================
    print("\n" + "-"*80)
    print("4. COMPARING CACHE CONFIGURATIONS")
    print("-"*80)
    
    # Test different cache sizes
    test_data = [f"item_{i}" for i in range(20)]
    
    configs = [
        ("Small cache (5 entries)", create_memory_cache(max_size=5)),
        ("Medium cache (10 entries)", create_memory_cache(max_size=10)),
        ("Large cache (20 entries)", create_memory_cache(max_size=20)),
    ]
    
    print("\nTesting workload with different cache sizes:")
    
    for name, test_cache in configs:
        # Reset and run workload
        test_cache.clear()
        test_cache.reset_stats()
        
        # Add all items
        for item in test_data:
            test_cache.set(item, f"value_{item}")
        
        # Access first 10 items repeatedly
        for item in test_data[:10]:
            for _ in range(3):
                _ = test_cache.get(item)
        
        stats = test_cache.get_stats()
        print(f"\n  {name}:")
        print(f"    Hit rate:   {stats.hit_rate:.2%}")
        print(f"    Hits:       {stats.hits}")
        print(f"    Misses:     {stats.misses}")
        print(f"    Evictions:  {stats.evictions}")
    
    # ========================================================================
    # 5. Real-time monitoring
    # ========================================================================
    print("\n" + "-"*80)
    print("5. REAL-TIME MONITORING")
    print("-"*80)
    
    realtime_cache = create_memory_cache(max_size=100)
    
    def show_stats(label):
        """Display current cache statistics."""
        stats = realtime_cache.get_stats()
        print(f"\n{label}:")
        print(f"  Hit rate: {stats.hit_rate:.2%} | "
              f"Hits: {stats.hits} | "
              f"Misses: {stats.misses} | "
              f"Size: {stats.size}")
    
    # Phase 1: Adding data
    print("\nPhase 1: Adding initial data")
    for i in range(10):
        realtime_cache.set(f"key{i}", f"value{i}")
    show_stats("After adding 10 items")
    
    # Phase 2: High hit rate
    print("\nPhase 2: Accessing existing items")
    for i in range(5):
        _ = realtime_cache.get(f"key{i}")
    show_stats("After accessing existing items")
    
    # Phase 3: Mixed access
    print("\nPhase 3: Mixed access pattern")
    for i in range(10):
        _ = realtime_cache.get(f"key{i}")      # Exists
        _ = realtime_cache.get(f"missing{i}")  # Doesn't exist
    show_stats("After mixed access")
    
    # ========================================================================
    # 6. Performance benchmarking
    # ========================================================================
    print("\n" + "-"*80)
    print("6. PERFORMANCE BENCHMARKING")
    print("-"*80)
    
    perf_cache = create_memory_cache(max_size=1000)
    
    # Benchmark writes
    start = time.time()
    for i in range(1000):
        perf_cache.set(f"perf_key_{i}", f"value_{i}")
    write_time = time.time() - start
    
    # Benchmark reads (hits)
    start = time.time()
    for i in range(1000):
        _ = perf_cache.get(f"perf_key_{i}")
    read_time = time.time() - start
    
    # Benchmark misses
    start = time.time()
    for i in range(1000):
        _ = perf_cache.get(f"missing_{i}")
    miss_time = time.time() - start
    
    print("\nPerformance Metrics:")
    print(f"  1000 writes: {write_time*1000:.2f}ms ({1000/write_time:.0f} ops/sec)")
    print(f"  1000 reads:  {read_time*1000:.2f}ms ({1000/read_time:.0f} ops/sec)")
    print(f"  1000 misses: {miss_time*1000:.2f}ms ({1000/miss_time:.0f} ops/sec)")
    
    stats = perf_cache.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Hit rate: {stats.hit_rate:.2%}")
    print(f"  Total ops: {stats.total_requests}")
    
    # ========================================================================
    # 7. Exporting statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("7. EXPORTING STATISTICS")
    print("-"*80)
    
    export_cache = create_memory_cache(max_size=100)
    
    # Simulate some activity
    export_cache.set("a", 1)
    export_cache.set("b", 2)
    _ = export_cache.get("a")
    _ = export_cache.get("b")
    _ = export_cache.get("c")  # Miss
    
    # Export as dict
    stats_dict = export_cache.get_stats()
    print("\nStatistics as dictionary:")
    print(f"  {vars(stats_dict)}")
    
    # Export formatted
    stats_json = export_cache_stats(export_cache)
    print(f"\nStatistics as JSON:")
    print(f"  {stats_json}")
    
    # ========================================================================
    # 8. Optimization recommendations
    # ========================================================================
    print("\n" + "-"*80)
    print("8. OPTIMIZATION RECOMMENDATIONS")
    print("-"*80)
    
    def analyze_cache_performance(cache):
        """Analyze cache and provide recommendations."""
        stats = cache.get_stats()
        recommendations = []
        
        if stats.hit_rate < 0.3:
            recommendations.append("⚠️  Low hit rate (<30%) - Consider increasing cache size")
        elif stats.hit_rate < 0.5:
            recommendations.append("📊 Moderate hit rate (30-50%) - Cache is working but could be better")
        else:
            recommendations.append("✅ Good hit rate (>50%) - Cache is effective")
        
        if stats.evictions > stats.size * 2:
            recommendations.append("⚠️  High eviction rate - Increase max_size")
        
        if stats.total_requests > 0:
            miss_rate = stats.misses / stats.total_requests
            if miss_rate > 0.7:
                recommendations.append("⚠️  High miss rate - Review cache key strategy")
        
        return recommendations
    
    # Test with different scenarios
    scenarios = [
        ("Good cache", create_memory_cache(max_size=100)),
        ("Too small cache", create_memory_cache(max_size=3)),
    ]
    
    for name, test_cache in scenarios:
        print(f"\n{name}:")
        
        # Simulate workload
        for i in range(20):
            test_cache.set(f"item{i}", i)
        
        for i in range(10):
            _ = test_cache.get(f"item{i}")
            _ = test_cache.get(f"item{i}")  # Access again
        
        stats = test_cache.get_stats()
        print(f"  Hit rate: {stats.hit_rate:.2%}")
        
        recommendations = analyze_cache_performance(test_cache)
        for rec in recommendations:
            print(f"  {rec}")
    
    # ========================================================================
    # 9. Reset statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("9. RESETTING STATISTICS")
    print("-"*80)
    
    reset_cache = create_memory_cache(max_size=100)
    
    # Generate some stats
    reset_cache.set("x", 1)
    _ = reset_cache.get("x")
    _ = reset_cache.get("y")
    
    stats_before = reset_cache.get_stats()
    print(f"\nBefore reset:")
    print(f"  Requests: {stats_before.total_requests}")
    print(f"  Hits: {stats_before.hits}")
    print(f"  Misses: {stats_before.misses}")
    
    # Reset statistics (keeps cache data)
    reset_cache.reset_stats()
    
    stats_after = reset_cache.get_stats()
    print(f"\nAfter reset:")
    print(f"  Requests: {stats_after.total_requests}")
    print(f"  Hits: {stats_after.hits}")
    print(f"  Misses: {stats_after.misses}")
    print(f"  Cache size: {stats_after.size} (data preserved)")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • Monitor hit rate to measure cache effectiveness")
    print("  • Target >50% hit rate for good performance")
    print("  • Track evictions to optimize cache size")
    print("  • Use statistics to make informed decisions")
    print("  • Export stats for dashboards and monitoring")
    print("  • Reset stats to measure specific workloads")
    print("  • Benchmark to understand performance characteristics")


if __name__ == "__main__":
    main()
