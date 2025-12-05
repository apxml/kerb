"""
Examples demonstrating high-performance JSONL utilities.
========================================================

This example showcases the enhanced JSONL utilities for handling large
training datasets efficiently with compression, parallel processing,
streaming, and advanced filtering.
"""

from kerb.fine_tuning import jsonl
from pathlib import Path
import time


# Module-level transformation function for parallel processing (must be picklable)
def normalize_and_add_features(item):
    """Transform function that normalizes text and adds features."""
    return {
        "text": item["text"].lower(),
        "value": item["value"],
        "value_squared": item["value"] ** 2,
        "is_even": item["value"] % 2 == 0
    }


def example_compression():
    """Example: Using compression to save disk space."""

# %%
# Setup and Imports
# -----------------
    print("=" * 60)
    print("Example 1: Compression Support")
    print("=" * 60)
    
    # Create sample data
    data = [
        {"messages": [{"role": "user", "content": f"Question {i}"}, 
                      {"role": "assistant", "content": f"Answer {i}"}]}
        for i in range(10000)
    ]
    
    # Write uncompressed
    start = time.time()
    jsonl.write_jsonl(data, "temp_uncompressed.jsonl")
    uncompressed_time = time.time() - start
    uncompressed_size = Path("temp_uncompressed.jsonl").stat().st_size
    
    # Write with gzip compression
    start = time.time()
    jsonl.write_jsonl(data, "temp_compressed.jsonl", compress=True, compression_type='gz')
    compressed_time = time.time() - start
    compressed_size = Path("temp_compressed.jsonl.gz").stat().st_size
    
    print(f"Uncompressed: {uncompressed_size:,} bytes in {uncompressed_time:.3f}s")
    print(f"Compressed:   {compressed_size:,} bytes in {compressed_time:.3f}s")
    print(f"Compression ratio: {uncompressed_size / compressed_size:.1f}x")
    print(f"Space saved: {(1 - compressed_size/uncompressed_size)*100:.1f}%")
    
    # Cleanup
    Path("temp_uncompressed.jsonl").unlink()
    Path("temp_compressed.jsonl.gz").unlink()
    print()



# %%
# Example Parallel Reading
# ------------------------

def example_parallel_reading():
    """Example: Parallel reading for faster processing."""
    print("=" * 60)
    print("Example 2: Parallel Reading")
    print("=" * 60)
    
    # Create a large file
    large_data = [
        {"id": i, "data": {"value": i * 2, "label": f"item_{i}"}}
        for i in range(50000)
    ]
    jsonl.write_jsonl(large_data, "temp_large.jsonl")
    
    # Sequential read
    start = time.time()
    data_sequential = jsonl.read_jsonl("temp_large.jsonl")
    sequential_time = time.time() - start
    
    # Parallel read
    start = time.time()
    data_parallel = jsonl.parallel_read_jsonl("temp_large.jsonl", num_workers=4)
    parallel_time = time.time() - start
    
    print(f"Sequential read: {sequential_time:.3f}s")
    print(f"Parallel read:   {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"Items read: {len(data_parallel):,}")
    
    # Cleanup
    Path("temp_large.jsonl").unlink()
    print()


def example_streaming_and_filtering():
    """Example: Memory-efficient streaming with filtering."""
    print("=" * 60)
    print("Example 3: Streaming and Filtering")
    print("=" * 60)
    
    # Create dataset
    data = [
        {"id": i, "score": i % 100, "category": "A" if i % 2 == 0 else "B"}
        for i in range(100000)
    ]
    jsonl.write_jsonl(data, "temp_data.jsonl")
    
    # Stream with filter: only high scores from category A
    filtered_count = 0
    for batch in jsonl.stream_jsonl(
        "temp_data.jsonl",
        batch_size=5000,
        filter_fn=lambda x: x["category"] == "A" and x["score"] > 50
    ):
        filtered_count += len(batch)
    
    print(f"Original dataset: {len(data):,} items")
    print(f"After filtering:  {filtered_count:,} items")
    print(f"Filtered out:     {len(data) - filtered_count:,} items")
    
    # Save filtered data
    jsonl.filter_jsonl(
        "temp_data.jsonl",
        "temp_filtered.jsonl",
        filter_fn=lambda x: x["category"] == "A" and x["score"] > 50
    )
    
    actual_count = jsonl.count_jsonl_lines("temp_filtered.jsonl")
    print(f"Saved to file:    {actual_count:,} items")
    
    # Cleanup
    Path("temp_data.jsonl").unlink()
    Path("temp_filtered.jsonl").unlink()
    print()



# %%
# Example Transformation
# ----------------------

def example_transformation():
    """Example: Transforming data in parallel."""
    print("=" * 60)
    print("Example 4: Parallel Transformation")
    print("=" * 60)
    
    # Create dataset
    data = [
        {"text": f"Sample text {i}", "value": i}
        for i in range(20000)
    ]
    jsonl.write_jsonl(data, "temp_input.jsonl")
    
    # Transform sequentially
    start = time.time()
    jsonl.transform_jsonl("temp_input.jsonl", "temp_seq.jsonl", 
                          normalize_and_add_features, parallel=False)
    sequential_time = time.time() - start
    
    # Transform in parallel
    start = time.time()
    jsonl.transform_jsonl("temp_input.jsonl", "temp_par.jsonl", 
                          normalize_and_add_features, parallel=True, num_workers=4)
    parallel_time = time.time() - start
    
    print(f"Sequential transformation: {sequential_time:.3f}s")
    print(f"Parallel transformation:   {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Verify output
    sample = jsonl.read_jsonl("temp_par.jsonl", max_lines=1)[0]
    print(f"Sample transformed item: {sample}")
    
    # Cleanup
    for f in ["temp_input.jsonl", "temp_seq.jsonl", "temp_par.jsonl"]:
        Path(f).unlink()
    print()


def example_splitting():
    """Example: Split large file into chunks."""
    print("=" * 60)
    print("Example 5: Splitting Large Files")
    print("=" * 60)
    
    # Create a large dataset
    data = [{"id": i, "content": f"Content for item {i}"} for i in range(25000)]
    jsonl.write_jsonl(data, "temp_large_dataset.jsonl")
    
    # Split into chunks of 10k lines each
    output_files = jsonl.split_jsonl(
        "temp_large_dataset.jsonl",
        "temp_chunk",
        split_size=10000,
        by_line=True
    )
    
    print(f"Original file: {len(data):,} lines")
    print(f"Split into {len(output_files)} files:")
    for i, file in enumerate(output_files):
        count = jsonl.count_jsonl_lines(file)
        print(f"  {file}: {count:,} lines")
    
    # Cleanup
    Path("temp_large_dataset.jsonl").unlink()
    for file in output_files:
        Path(file).unlink()
    print()



# %%
# Example Deduplication
# ---------------------

def example_deduplication():
    """Example: Remove duplicate entries."""
    print("=" * 60)
    print("Example 6: Deduplication")
    print("=" * 60)
    
    # Create dataset with duplicates
    data = [
        {"prompt": "What is AI?", "completion": "AI is artificial intelligence"},
        {"prompt": "What is ML?", "completion": "ML is machine learning"},
        {"prompt": "What is AI?", "completion": "AI is artificial intelligence"},  # duplicate
        {"prompt": "What is DL?", "completion": "DL is deep learning"},
        {"prompt": "What is ML?", "completion": "ML is machine learning"},  # duplicate
    ]
    jsonl.write_jsonl(data, "temp_with_dupes.jsonl")
    
    # Deduplicate based on prompt field
    removed = jsonl.deduplicate_jsonl(
        "temp_with_dupes.jsonl",
        "temp_deduped.jsonl",
        key_fn=lambda x: x.get("prompt", ""),
        keep='first'
    )
    
    print(f"Original: {len(data)} items")
    print(f"Duplicates removed: {removed}")
    print(f"After deduplication: {len(data) - removed} items")
    
    # Show deduplicated data
    deduped = jsonl.read_jsonl("temp_deduped.jsonl")
    print("\nDeduplicated data:")
    for item in deduped:
        print(f"  - {item['prompt']}")
    
    # Cleanup
    Path("temp_with_dupes.jsonl").unlink()
    Path("temp_deduped.jsonl").unlink()
    print()


def example_sampling():
    """Example: Sample random subset."""
    print("=" * 60)
    print("Example 7: Random Sampling")
    print("=" * 60)
    
    # Create dataset
    data = [{"id": i, "value": i} for i in range(10000)]
    jsonl.write_jsonl(data, "temp_full.jsonl")
    
    # Sample 10% of data
    jsonl.sample_jsonl(
        "temp_full.jsonl",
        "temp_sample.jsonl",
        fraction=0.1,
        random_state=42
    )
    
    full_count = jsonl.count_jsonl_lines("temp_full.jsonl")
    sample_count = jsonl.count_jsonl_lines("temp_sample.jsonl")
    
    print(f"Original dataset: {full_count:,} items")
    print(f"Sampled dataset:  {sample_count:,} items ({sample_count/full_count*100:.1f}%)")
    
    # Cleanup
    Path("temp_full.jsonl").unlink()
    Path("temp_sample.jsonl").unlink()
    print()



# %%
# Example File Stats
# ------------------

def example_file_stats():
    """Example: Get file statistics."""
    print("=" * 60)
    print("Example 8: File Statistics")
    print("=" * 60)
    
    # Create diverse dataset
    data = [
        {"messages": [{"role": "user", "content": f"Q{i}"}, 
                      {"role": "assistant", "content": f"A{i}"}],
         "metadata": {"source": "synthetic", "quality": i % 5}}
        for i in range(5000)
    ]
    jsonl.write_jsonl(data, "temp_stats.jsonl")
    
    # Get statistics
    stats = jsonl.get_jsonl_stats("temp_stats.jsonl", sample_size=1000)
    
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Total bytes: {stats['total_bytes']:,}")
    print(f"Avg bytes per line: {stats['avg_bytes_per_line']:.1f}")
    print(f"Compressed: {stats['compressed']}")
    print(f"Keys found: {', '.join(stats['keys'])}")
    print(f"Sample items analyzed: {len(stats['sample_items'])}")
    
    # Cleanup
    Path("temp_stats.jsonl").unlink()
    print()


def example_merge_files():
    """Example: Merge multiple JSONL files."""
    print("=" * 60)
    print("Example 9: Merging Files")
    print("=" * 60)
    
    # Create multiple files
    file1_data = [{"source": "file1", "id": i} for i in range(1000)]
    file2_data = [{"source": "file2", "id": i} for i in range(1000)]
    file3_data = [{"source": "file3", "id": i} for i in range(1000)]
    
    jsonl.write_jsonl(file1_data, "temp_file1.jsonl")
    jsonl.write_jsonl(file2_data, "temp_file2.jsonl")
    jsonl.write_jsonl(file3_data, "temp_file3.jsonl")
    
    # Merge files
    start = time.time()
    jsonl.merge_jsonl(
        ["temp_file1.jsonl", "temp_file2.jsonl", "temp_file3.jsonl"],
        "temp_merged.jsonl",
        parallel=True
    )
    merge_time = time.time() - start
    
    merged_count = jsonl.count_jsonl_lines("temp_merged.jsonl")
    
    print(f"Merged 3 files in {merge_time:.3f}s")
    print(f"Total items: {merged_count:,}")
    print(f"Expected: {len(file1_data) + len(file2_data) + len(file3_data):,}")
    
    # Cleanup
    for f in ["temp_file1.jsonl", "temp_file2.jsonl", "temp_file3.jsonl", "temp_merged.jsonl"]:
        Path(f).unlink()
    print()



# %%
# Example Compressed Workflow
# ---------------------------

def example_compressed_workflow():
    """Example: Complete workflow with compression."""
    print("=" * 60)
    print("Example 10: Compressed Workflow")
    print("=" * 60)
    
    # Create data
    data = [{"id": i, "text": f"Text {i}"} for i in range(10000)]
    
    # Write compressed
    jsonl.write_jsonl(data, "temp_data.jsonl.gz", compress=True)
    
    # Read compressed (automatic detection)
    read_data = jsonl.read_jsonl("temp_data.jsonl.gz", max_lines=5)
    print(f"Read {len(read_data)} samples from compressed file")
    
    # Stream compressed file
    count = 0
    for batch in jsonl.stream_jsonl("temp_data.jsonl.gz", batch_size=1000):
        count += len(batch)
    print(f"Streamed {count:,} items from compressed file")
    
    # Filter and save compressed
    jsonl.filter_jsonl(
        "temp_data.jsonl.gz",
        "temp_filtered.jsonl.gz",
        filter_fn=lambda x: x["id"] % 2 == 0,
        compress_output=True
    )
    filtered_count = jsonl.count_jsonl_lines("temp_filtered.jsonl.gz")
    print(f"Filtered to {filtered_count:,} items (compressed)")
    
    # Cleanup
    Path("temp_data.jsonl.gz").unlink()
    Path("temp_filtered.jsonl.gz").unlink()
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "High-Performance JSONL Utilities" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Run all examples
    example_compression()
    example_parallel_reading()
    example_streaming_and_filtering()
    example_transformation()
    example_splitting()
    example_deduplication()
    example_sampling()
    example_file_stats()
    example_merge_files()
    example_compressed_workflow()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
