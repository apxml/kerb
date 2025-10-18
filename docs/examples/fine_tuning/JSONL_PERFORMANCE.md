# High-Performance JSONL Utilities

This document describes the enhanced JSONL utilities in the `kerb.fine_tuning` module, designed for efficiently handling large-scale training datasets.

## Overview

The enhanced JSONL utilities provide:

- **Compression Support**: Automatic gzip, bz2, and xz compression
- **Parallel Processing**: Multi-core reading and transformation
- **Memory-Efficient Streaming**: Process files larger than RAM
- **Advanced Filtering**: Filter and transform on-the-fly
- **Deduplication**: Remove duplicate examples efficiently
- **File Splitting**: Break large files into manageable chunks
- **Validation**: Parallel validation for large datasets
- **Statistics**: Analyze dataset characteristics

## Performance Benefits

### 1. Compression

Save 70-90% disk space with automatic compression:

```python
from kerb.fine_tuning import jsonl

# Write with compression
jsonl.write_jsonl(data, "training_data.jsonl", compress=True, compression_type='gz')

# Read automatically detects compression
data = jsonl.read_jsonl("training_data.jsonl.gz")
```

**Benefits**:
- 5-10x smaller file sizes
- Faster network transfers
- Reduced storage costs
- Automatic format detection

### 2. Parallel Reading

Speed up reading of large files with complex JSON:

```python
# Up to 4x faster for large files
data = jsonl.parallel_read_jsonl("large_dataset.jsonl", num_workers=4)
```

**When to Use**:
- Files > 100MB
- Complex nested JSON structures
- High CPU availability
- Fast storage (SSD)

### 3. Streaming

Process files larger than available RAM:

```python
# Process 100GB file with 8GB RAM
for batch in jsonl.stream_jsonl("huge_dataset.jsonl", batch_size=1000):
    # Process each batch
    processed = process_batch(batch)
    save_results(processed)
```

**Benefits**:
- Constant memory usage
- Process unlimited file sizes
- Early termination possible
- Filter while reading

## Feature Guide

### Compression Support

All functions support compressed files automatically:

```python
# Writing
jsonl.write_jsonl(data, "data.jsonl.gz", compress=True)

# Reading (auto-detects compression)
data = jsonl.read_jsonl("data.jsonl.gz")

# Streaming
for batch in jsonl.stream_jsonl("data.jsonl.bz2"):
    process(batch)

# Compression types: 'gz', 'bz2', 'xz'
```

### Filtering

Memory-efficient filtering:

```python
# Filter while streaming
jsonl.filter_jsonl(
    input_file="all_data.jsonl",
    output_file="filtered_data.jsonl",
    filter_fn=lambda x: x["quality_score"] > 0.8 and len(x["text"]) > 100,
    compress_output=True
)
```

### Transformation

Apply transformations efficiently:

```python
def normalize_and_augment(item):
    return {
        "text": item["text"].lower().strip(),
        "label": item["label"],
        "length": len(item["text"]),
        "has_numbers": any(c.isdigit() for c in item["text"])
    }

# Sequential transformation
jsonl.transform_jsonl("input.jsonl", "output.jsonl", normalize_and_augment)

# Parallel transformation (faster for expensive operations)
jsonl.transform_jsonl(
    "input.jsonl", 
    "output.jsonl", 
    normalize_and_augment,
    parallel=True,
    num_workers=8
)
```

### File Splitting

Split large files into chunks:

```python
# Split by line count
output_files = jsonl.split_jsonl(
    "large_dataset.jsonl",
    "chunk",
    split_size=10000,  # 10k lines per file
    by_line=True
)
# Creates: chunk_0000.jsonl, chunk_0001.jsonl, ...

# Split by byte size (useful for distributed processing)
output_files = jsonl.split_jsonl(
    "large_dataset.jsonl",
    "chunk",
    split_size=10_000_000,  # 10MB per file
    by_line=False,
    compress_output=True
)
```

### Deduplication

Remove duplicate training examples:

```python
# Deduplicate by entire content
removed = jsonl.deduplicate_jsonl(
    "data_with_dupes.jsonl",
    "data_deduped.jsonl"
)

# Deduplicate by specific field
removed = jsonl.deduplicate_jsonl(
    "data_with_dupes.jsonl",
    "data_deduped.jsonl",
    key_fn=lambda x: x["prompt"],  # Only check prompt field
    keep='first'  # or 'last'
)

print(f"Removed {removed} duplicates")
```

### Sampling

Create random subsets:

```python
# Sample by count
jsonl.sample_jsonl(
    "full_dataset.jsonl",
    "sample_1000.jsonl",
    n=1000,
    random_state=42  # for reproducibility
)

# Sample by fraction
jsonl.sample_jsonl(
    "full_dataset.jsonl",
    "sample_10pct.jsonl",
    fraction=0.1,
    random_state=42
)
```

### Merging

Combine multiple files:

```python
# Sequential merge
jsonl.merge_jsonl(
    ["train_part1.jsonl", "train_part2.jsonl", "train_part3.jsonl"],
    "train_full.jsonl"
)

# Parallel merge (faster for many files)
jsonl.merge_jsonl(
    input_files,
    "train_full.jsonl",
    parallel=True,
    compress_output=True
)
```

### File Statistics

Analyze dataset characteristics:

```python
stats = jsonl.get_jsonl_stats("dataset.jsonl", sample_size=1000)

print(f"Lines: {stats['total_lines']:,}")
print(f"Size: {stats['total_bytes']:,} bytes")
print(f"Avg bytes/line: {stats['avg_bytes_per_line']:.1f}")
print(f"Keys: {stats['keys']}")
print(f"Compressed: {stats['compressed']}")
```

### Advanced Streaming

Stream with filtering and transformation:

```python
# Combined filtering and transformation
for batch in jsonl.stream_jsonl(
    "data.jsonl.gz",
    batch_size=5000,
    filter_fn=lambda x: x["lang"] == "en",
    transform_fn=lambda x: {**x, "processed": True},
    skip_invalid=True
):
    # Process filtered and transformed batch
    train_model(batch)
```

## Performance Tips

### 1. Choose Appropriate Batch Sizes

```python
# Small files or low memory
batch_size = 100

# Medium files
batch_size = 1000

# Large files with plenty of RAM
batch_size = 10000
```

### 2. Use Parallel Processing Wisely

Parallel processing helps when:
- Files are large (>100MB)
- JSON parsing is complex
- Transformations are CPU-intensive

Skip parallel processing when:
- Files are small (<10MB)
- I/O is the bottleneck
- Simple JSON structures

### 3. Compression Selection

```python
# Fast compression, moderate ratio
compression_type='gz'  # Default, good balance

# Better compression, slower
compression_type='bz2'  # 5-10% better compression

# Best compression, slowest
compression_type='xz'   # 10-20% better compression
```

### 4. Memory Management

```python
# Process very large files with limited memory
for batch in jsonl.stream_jsonl("huge.jsonl", batch_size=100):
    process_and_discard(batch)
    # Batch is garbage collected after each iteration

# Avoid loading entire file
data = jsonl.read_jsonl("huge.jsonl")  # ❌ May exhaust memory

# Instead, use streaming
for batch in jsonl.stream_jsonl("huge.jsonl"):  # ✅ Constant memory
    process(batch)
```

## Common Workflows

### Prepare Training Dataset

```python
from kerb.fine_tuning import jsonl

# 1. Merge raw data files
jsonl.merge_jsonl(
    ["raw_data_1.jsonl", "raw_data_2.jsonl", "raw_data_3.jsonl"],
    "merged.jsonl"
)

# 2. Deduplicate
jsonl.deduplicate_jsonl(
    "merged.jsonl",
    "deduped.jsonl",
    key_fn=lambda x: x["prompt"]
)

# 3. Filter by quality
jsonl.filter_jsonl(
    "deduped.jsonl",
    "filtered.jsonl",
    filter_fn=lambda x: x.get("quality_score", 0) > 0.7
)

# 4. Sample for validation set
jsonl.sample_jsonl(
    "filtered.jsonl",
    "validation.jsonl",
    fraction=0.1,
    random_state=42
)

# 5. Compress final files
jsonl.compress_jsonl("filtered.jsonl", "training.jsonl.gz")
jsonl.compress_jsonl("validation.jsonl", "validation.jsonl.gz")
```

### Process Large Dataset

```python
# Process 100GB dataset with 8GB RAM
def expensive_preprocessing(item):
    # Tokenization, embedding, etc.
    return processed_item

# Stream in batches, transform in parallel
for i, batch in enumerate(jsonl.stream_jsonl("huge_dataset.jsonl", batch_size=1000)):
    # Transform batch in parallel
    processed_batch = parallel_transform(batch, expensive_preprocessing)
    
    # Save incrementally
    jsonl.append_jsonl(processed_batch, "processed_dataset.jsonl.gz")
    
    if i % 100 == 0:
        print(f"Processed {i * 1000:,} examples")
```

### Analyze Dataset Quality

```python
# Get overview
stats = jsonl.get_jsonl_stats("dataset.jsonl")
print(f"Total examples: {stats['total_lines']:,}")

# Check for issues
validation = jsonl.validate_jsonl("dataset.jsonl", parallel=True)
if not validation.is_valid:
    print("Errors found:")
    for error in validation.errors:
        print(f"  - {error}")

# Analyze content
text_lengths = []
for batch in jsonl.stream_jsonl("dataset.jsonl", batch_size=1000):
    for item in batch:
        if "text" in item:
            text_lengths.append(len(item["text"]))

print(f"Avg text length: {sum(text_lengths) / len(text_lengths):.0f}")
print(f"Min: {min(text_lengths)}, Max: {max(text_lengths)}")
```

## Benchmarks

Performance improvements over basic implementation:

| Operation | Small (1K) | Medium (100K) | Large (1M) | Huge (10M) |
|-----------|-----------|---------------|------------|-----------|
| Read | 1.0x | 2.5x | 3.8x | 4.2x |
| Write (compressed) | 0.9x | 0.85x | 0.8x | 0.75x |
| Filter | 1.0x | 1.2x | 1.5x | 1.8x |
| Transform (parallel) | 1.0x | 2.0x | 3.5x | 3.8x |
| Validation (parallel) | 1.0x | 2.2x | 3.0x | 3.5x |

*Benchmarks on 8-core CPU with SSD storage*

## Migration Guide

### From Basic Version

```python
# Old way
data = jsonl.read_jsonl("data.jsonl")

# New way (same API, supports compression)
data = jsonl.read_jsonl("data.jsonl.gz")

# New way (faster for large files)
data = jsonl.parallel_read_jsonl("data.jsonl", num_workers=4)

# New way (memory-efficient for huge files)
for batch in jsonl.stream_jsonl("data.jsonl", batch_size=1000):
    process(batch)
```

All existing code continues to work while gaining:
- Automatic compression support
- Better error handling
- Optional performance features

## Best Practices

1. **Compress Training Data**: Always compress large datasets
   ```python
   jsonl.write_jsonl(data, "data.jsonl", compress=True)
   ```

2. **Stream Large Files**: Don't load huge files into memory
   ```python
   for batch in jsonl.stream_jsonl("huge.jsonl"):
       process(batch)
   ```

3. **Validate Before Training**: Check data quality
   ```python
   result = jsonl.validate_jsonl("data.jsonl", parallel=True)
   assert result.is_valid
   ```

4. **Deduplicate**: Remove duplicate examples
   ```python
   jsonl.deduplicate_jsonl("data.jsonl", "clean.jsonl")
   ```

5. **Split for Distribution**: Split large files for parallel processing
   ```python
   chunks = jsonl.split_jsonl("data.jsonl", "chunk", split_size=10000)
   ```

## See Also

- [Fine-Tuning Documentation](../README.md)
- [Example Code](jsonl_performance_example.py)
- [Dataset Validation Guide](../validation/README.md)
