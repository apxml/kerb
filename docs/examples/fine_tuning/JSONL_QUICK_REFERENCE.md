# JSONL Quick Reference

## Common Operations

### Basic I/O
```python
from kerb.fine_tuning import jsonl

# Write
jsonl.write_jsonl(data, "output.jsonl")

# Read
data = jsonl.read_jsonl("input.jsonl")

# Append
jsonl.append_jsonl(more_data, "output.jsonl")
```

### Compression (Recommended for Large Files)
```python
# Write compressed (70-90% smaller)
jsonl.write_jsonl(data, "data.jsonl", compress=True)

# Read compressed (auto-detects format)
data = jsonl.read_jsonl("data.jsonl.gz")

# Convert existing file
jsonl.compress_jsonl("data.jsonl")  # → data.jsonl.gz
```

### Performance (Large Files)
```python
# Parallel reading (2-4x faster)
data = jsonl.parallel_read_jsonl("large.jsonl", num_workers=4)

# Streaming (unlimited file size)
for batch in jsonl.stream_jsonl("huge.jsonl", batch_size=1000):
    process(batch)
```

### Filtering
```python
# Filter by condition
jsonl.filter_jsonl(
    "input.jsonl",
    "output.jsonl",
    filter_fn=lambda x: x["score"] > 0.8
)

# Stream with filter
for batch in jsonl.stream_jsonl(
    "data.jsonl",
    filter_fn=lambda x: x["lang"] == "en"
):
    process(batch)
```

### Transformation
```python
# Transform data
def normalize(item):
    return {
        "text": item["text"].lower(),
        "label": item["label"]
    }

jsonl.transform_jsonl("in.jsonl", "out.jsonl", normalize)

# Parallel transformation (faster)
jsonl.transform_jsonl(
    "in.jsonl", "out.jsonl", 
    normalize, 
    parallel=True, 
    num_workers=8
)
```

### Data Cleaning
```python
# Remove duplicates
removed = jsonl.deduplicate_jsonl(
    "data.jsonl",
    "clean.jsonl",
    key_fn=lambda x: x["prompt"]  # dedupe by field
)

# Random sampling
jsonl.sample_jsonl(
    "full.jsonl",
    "sample.jsonl",
    fraction=0.1,  # 10% of data
    random_state=42
)
```

### File Operations
```python
# Merge files
jsonl.merge_jsonl(
    ["file1.jsonl", "file2.jsonl", "file3.jsonl"],
    "merged.jsonl"
)

# Split large file
chunks = jsonl.split_jsonl(
    "large.jsonl",
    "chunk",
    split_size=10000  # lines per file
)

# Count lines
count = jsonl.count_jsonl_lines("data.jsonl")

# Get statistics
stats = jsonl.get_jsonl_stats("data.jsonl")
print(f"Lines: {stats['total_lines']}, Size: {stats['total_bytes']}")
```

### Validation
```python
# Validate file
result = jsonl.validate_jsonl("data.jsonl", parallel=True)
if result.is_valid:
    print(f"✓ Valid: {result.valid_examples} examples")
else:
    print(f"✗ Errors: {result.errors}")
```

## When to Use What

| Scenario | Use | Benefit |
|----------|-----|---------|
| Small file (<10MB) | `read_jsonl()` | Simple, fast |
| Large file (>100MB) | `parallel_read_jsonl()` | 2-4x faster |
| Huge file (>1GB) | `stream_jsonl()` | Constant memory |
| Storage cost matters | `compress=True` | 70-90% smaller |
| Need to filter | `filter_jsonl()` | Memory efficient |
| Complex transform | `transform_jsonl(parallel=True)` | Multi-core speed |
| Check quality | `validate_jsonl()` | Find issues |
| Remove dupes | `deduplicate_jsonl()` | Clean data |

## Performance Tips

### ✅ Do
- Compress large files: `compress=True`
- Stream huge files: `stream_jsonl()`
- Use parallel for >100MB files
- Filter while streaming
- Deduplicate before training

### ❌ Don't
- Load 1GB+ files into memory
- Parse JSON repeatedly
- Skip validation on new data
- Store uncompressed training data
- Process entire file if you need subset

## Example Workflow

```python
from kerb.fine_tuning import jsonl

# 1. Merge source files
jsonl.merge_jsonl(
    ["raw1.jsonl", "raw2.jsonl"],
    "merged.jsonl"
)

# 2. Validate
result = jsonl.validate_jsonl("merged.jsonl")
assert result.is_valid

# 3. Filter quality
jsonl.filter_jsonl(
    "merged.jsonl",
    "filtered.jsonl",
    filter_fn=lambda x: len(x.get("text", "")) > 50
)

# 4. Deduplicate
jsonl.deduplicate_jsonl(
    "filtered.jsonl",
    "clean.jsonl",
    key_fn=lambda x: x["prompt"]
)

# 5. Split train/val
jsonl.sample_jsonl("clean.jsonl", "val.jsonl", fraction=0.1)

# 6. Compress for storage
jsonl.compress_jsonl("clean.jsonl", "train.jsonl.gz")
jsonl.compress_jsonl("val.jsonl", "val.jsonl.gz")
```

## Compression Formats

| Format | Extension | Speed | Ratio | Use Case |
|--------|-----------|-------|-------|----------|
| gzip | `.gz` | Fast | 8:1 | Default choice |
| bzip2 | `.bz2` | Medium | 10:1 | Better compression |
| xz | `.xz` | Slow | 12:1 | Best compression |

## All Functions

```python
# Core I/O
write_jsonl(data, filepath, compress=False)
read_jsonl(filepath, max_lines=None, skip_invalid=False)
append_jsonl(data, filepath)

# Streaming
stream_jsonl(filepath, batch_size=1000, filter_fn=None, transform_fn=None)
parallel_read_jsonl(filepath, num_workers=None, chunk_size=10000)

# Manipulation
filter_jsonl(input_file, output_file, filter_fn)
transform_jsonl(input_file, output_file, transform_fn, parallel=False)
deduplicate_jsonl(input_file, output_file, key_fn=None, keep='first')
sample_jsonl(input_file, output_file, n=None, fraction=None)

# File ops
merge_jsonl(input_files, output_file, parallel=False)
split_jsonl(input_file, output_prefix, split_size=10000, by_line=True)
compress_jsonl(input_file, output_file=None, compression_type='gz')
decompress_jsonl(input_file, output_file=None)

# Analysis
validate_jsonl(filepath, schema=None, parallel=False)
count_jsonl_lines(filepath, fast=True)
get_jsonl_stats(filepath, sample_size=1000)
```
