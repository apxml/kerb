"""JSONL file utilities for reading and writing training data.

This module provides high-performance utilities for working with JSONL files,
including parallel processing, memory-efficient streaming, compression support,
and advanced filtering capabilities for large-scale datasets.
"""

import bz2
import gzip
import json
import lzma
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .types import TrainingDataset, ValidationResult


def _get_file_opener(filepath: str):
    """Get appropriate file opener based on file extension.

    Args:
        filepath: File path

    Returns:
        Tuple of (open_func, mode_read, mode_write)
    """
    if filepath.endswith(".gz"):
        return gzip.open, "rt", "wt"
    elif filepath.endswith(".bz2"):
        return bz2.open, "rt", "wt"
    elif filepath.endswith(".xz") or filepath.endswith(".lzma"):
        return lzma.open, "rt", "wt"
    else:
        return open, "r", "w"


def _parse_json_chunk(chunk: List[str]) -> List[Dict[str, Any]]:
    """Parse a chunk of JSON lines (helper for parallel processing).

    Args:
        chunk: List of JSON string lines

    Returns:
        List of parsed dictionaries
    """
    return [json.loads(line) for line in chunk]


def write_jsonl(
    data: Union[List[Dict[str, Any]], TrainingDataset],
    filepath: str,
    compress: bool = False,
    compression_type: str = "gz",
    buffer_size: int = 8192,
):
    """Write data to JSONL file with optional compression.

    Args:
        data: Data to write (list of dicts or TrainingDataset)
        filepath: Output file path
        compress: Whether to compress the output
        compression_type: Type of compression ('gz', 'bz2', 'xz')
        buffer_size: Buffer size for writing
    """
    if isinstance(data, TrainingDataset):
        data = data.to_list()

    # Add compression extension if needed
    if compress and not any(
        filepath.endswith(ext) for ext in [".gz", ".bz2", ".xz", ".lzma"]
    ):
        filepath = f"{filepath}.{compression_type}"

    open_func, _, mode_write = _get_file_opener(filepath)

    with open_func(filepath, mode_write, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(
    filepath: str, max_lines: Optional[int] = None, skip_invalid: bool = False
) -> List[Dict[str, Any]]:
    """Read data from JSONL file with automatic compression detection.

    Args:
        filepath: Input file path (supports .gz, .bz2, .xz compression)
        max_lines: Maximum number of lines to read (None for all)
        skip_invalid: Whether to skip invalid JSON lines

    Returns:
        List of dictionaries
    """
    data = []
    open_func, mode_read, _ = _get_file_opener(filepath)

    with open_func(filepath, mode_read, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break

            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    if not skip_invalid:
                        raise
    return data


def append_jsonl(
    data: Union[List[Dict[str, Any]], TrainingDataset],
    filepath: str,
    create_if_missing: bool = True,
):
    """Append data to JSONL file with compression support.

    Args:
        data: Data to append
        filepath: Target file path
        create_if_missing: Create file if it doesn't exist
    """
    if isinstance(data, TrainingDataset):
        data = data.to_list()

    if not Path(filepath).exists() and not create_if_missing:
        raise FileNotFoundError(f"File not found: {filepath}")

    open_func, _, mode_write = _get_file_opener(filepath)
    mode = "at" if mode_write == "wt" else "a"

    with open_func(filepath, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_jsonl(
    input_files: List[str],
    output_file: str,
    parallel: bool = False,
    compress_output: bool = False,
):
    """Merge multiple JSONL files into one with optional parallel processing.

    Args:
        input_files: List of input file paths
        output_file: Output file path
        parallel: Use parallel processing for large files
        compress_output: Compress the output file
    """
    if compress_output and not any(
        output_file.endswith(ext) for ext in [".gz", ".bz2", ".xz"]
    ):
        output_file = f"{output_file}.gz"

    open_func_out, _, mode_write = _get_file_opener(output_file)

    with open_func_out(output_file, mode_write, encoding="utf-8") as outf:
        if parallel and len(input_files) > 1:
            # Use parallel reading for multiple files
            with ThreadPoolExecutor(max_workers=min(4, len(input_files))) as executor:

                def read_file(input_file):
                    open_func, mode_read, _ = _get_file_opener(input_file)
                    with open_func(input_file, mode_read, encoding="utf-8") as inf:
                        return inf.read()

                futures = [executor.submit(read_file, f) for f in input_files]
                for future in futures:
                    outf.write(future.result())
        else:
            # Sequential processing
            for input_file in input_files:
                open_func, mode_read, _ = _get_file_opener(input_file)
                with open_func(input_file, mode_read, encoding="utf-8") as inf:
                    for line in inf:
                        outf.write(line)


def validate_jsonl(
    filepath: str, schema: Optional[Dict[str, Any]] = None, parallel: bool = False
) -> ValidationResult:
    """Validate JSONL file format with optional schema validation.

    Args:
        filepath: File path to validate
        schema: Optional JSON schema to validate against
        parallel: Use parallel validation for large files

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    try:
        open_func, mode_read, _ = _get_file_opener(filepath)

        with open_func(filepath, mode_read, encoding="utf-8") as f:
            lines = [(i, line) for i, line in enumerate(f, 1)]

        if parallel and len(lines) > 1000:
            # Parallel validation for large files
            def validate_line(item):
                i, line = item
                line = line.strip()
                if not line:
                    return None

                try:
                    data = json.loads(line)
                    # TODO: Add schema validation if schema is provided
                    return ("valid", i, None)
                except json.JSONDecodeError as e:
                    return ("invalid", i, str(e))

            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                results = executor.map(validate_line, lines)

                for res in results:
                    if res is None:
                        continue
                    status, line_num, error = res
                    if status == "valid":
                        result.valid_examples += 1
                    else:
                        result.add_error(f"Line {line_num}: Invalid JSON - {error}")
                        result.invalid_examples += 1
        else:
            # Sequential validation
            for i, line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # TODO: Add schema validation if schema is provided
                    result.valid_examples += 1
                except json.JSONDecodeError as e:
                    result.add_error(f"Line {i}: Invalid JSON - {e}")
                    result.invalid_examples += 1

        result.total_examples = result.valid_examples + result.invalid_examples

    except FileNotFoundError:
        result.add_error(f"File not found: {filepath}")
    except Exception as e:
        result.add_error(f"Error reading file: {e}")

    return result


def count_jsonl_lines(filepath: str, fast: bool = True) -> int:
    """Count lines in JSONL file with optimized performance.

    Args:
        filepath: File path
        fast: Use fast byte-counting method (doesn't validate JSON)

    Returns:
        Number of lines
    """
    open_func, mode_read, _ = _get_file_opener(filepath)

    if fast:
        # Fast method: count newlines without parsing
        count = 0
        with open_func(filepath, mode_read, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    else:
        # Thorough method: parse and validate
        count = 0
        with open_func(filepath, mode_read, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        pass
        return count


def stream_jsonl(
    filepath: str,
    batch_size: int = 1000,
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    skip_invalid: bool = False,
) -> Iterator[List[Dict[str, Any]]]:
    """Stream large JSONL files in batches with optional filtering and transformation.

    Args:
        filepath: File path (supports compressed files)
        batch_size: Number of examples per batch
        filter_fn: Optional function to filter examples
        transform_fn: Optional function to transform examples
        skip_invalid: Whether to skip invalid JSON lines

    Yields:
        Batches of examples
    """
    batch = []
    open_func, mode_read, _ = _get_file_opener(filepath)

    with open_func(filepath, mode_read, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                # Apply filter if provided
                if filter_fn and not filter_fn(item):
                    continue

                # Apply transformation if provided
                if transform_fn:
                    item = transform_fn(item)

                batch.append(item)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except json.JSONDecodeError:
                if not skip_invalid:
                    raise

        if batch:
            yield batch


def parallel_read_jsonl(
    filepath: str, num_workers: Optional[int] = None, chunk_size: int = 10000
) -> List[Dict[str, Any]]:
    """Read JSONL file in parallel for maximum performance.

    Uses multiprocessing to parse JSON lines in parallel, significantly
    faster for large files with complex JSON structures.

    Args:
        filepath: File path to read
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Lines to process per worker batch

    Returns:
        List of dictionaries
    """
    open_func, mode_read, _ = _get_file_opener(filepath)

    # Read all lines first
    with open_func(filepath, mode_read, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return []

    # Use parallel processing for large files
    if len(lines) > chunk_size:
        num_workers = num_workers or mp.cpu_count()
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_parse_json_chunk, chunks)

        # Flatten results
        data = []
        for chunk_result in results:
            data.extend(chunk_result)
        return data
    else:
        # Sequential for small files
        return [json.loads(line) for line in lines]


def filter_jsonl(
    input_file: str,
    output_file: str,
    filter_fn: Callable[[Dict[str, Any]], bool],
    batch_size: int = 1000,
    compress_output: bool = False,
):
    """Filter JSONL file based on a predicate function.

    Memory-efficient streaming filter for large files.

    Args:
        input_file: Input file path
        output_file: Output file path
        filter_fn: Function that returns True for items to keep
        batch_size: Batch size for streaming
        compress_output: Whether to compress output
    """
    if compress_output and not any(
        output_file.endswith(ext) for ext in [".gz", ".bz2", ".xz"]
    ):
        output_file = f"{output_file}.gz"

    open_func_out, _, mode_write = _get_file_opener(output_file)

    with open_func_out(output_file, mode_write, encoding="utf-8") as outf:
        for batch in stream_jsonl(
            input_file, batch_size=batch_size, filter_fn=filter_fn
        ):
            for item in batch:
                outf.write(json.dumps(item, ensure_ascii=False) + "\n")


def transform_jsonl(
    input_file: str,
    output_file: str,
    transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    batch_size: int = 1000,
    parallel: bool = False,
    num_workers: Optional[int] = None,
    compress_output: bool = False,
):
    """Transform JSONL file by applying a function to each item.

    Supports parallel processing for expensive transformations.

    Args:
        input_file: Input file path
        output_file: Output file path
        transform_fn: Function to transform each item
        batch_size: Batch size for streaming
        parallel: Use parallel processing
        num_workers: Number of parallel workers
        compress_output: Whether to compress output
    """
    if compress_output and not any(
        output_file.endswith(ext) for ext in [".gz", ".bz2", ".xz"]
    ):
        output_file = f"{output_file}.gz"

    open_func_out, _, mode_write = _get_file_opener(output_file)

    with open_func_out(output_file, mode_write, encoding="utf-8") as outf:
        if parallel:
            # Parallel transformation
            num_workers = num_workers or mp.cpu_count()

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for batch in stream_jsonl(input_file, batch_size=batch_size):
                    transformed_batch = list(executor.map(transform_fn, batch))
                    for item in transformed_batch:
                        outf.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            # Sequential transformation
            for batch in stream_jsonl(
                input_file, batch_size=batch_size, transform_fn=transform_fn
            ):
                for item in batch:
                    outf.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_jsonl(
    input_file: str,
    output_prefix: str,
    split_size: int = 10000,
    by_line: bool = True,
    compress_output: bool = False,
) -> List[str]:
    """Split large JSONL file into smaller chunks.

    Useful for distributing processing or managing large datasets.

    Args:
        input_file: Input file path
        output_prefix: Prefix for output files
        split_size: Lines per split (if by_line) or bytes per split
        by_line: Split by line count vs byte size
        compress_output: Whether to compress output files

    Returns:
        List of output file paths
    """
    output_files = []
    current_file_idx = 0
    current_count = 0

    def get_output_path(idx):
        ext = ".gz" if compress_output else ""
        return f"{output_prefix}_{idx:04d}.jsonl{ext}"

    open_func_in, mode_read, _ = _get_file_opener(input_file)

    current_output = get_output_path(current_file_idx)
    output_files.append(current_output)

    open_func_out, _, mode_write = _get_file_opener(current_output)
    outf = open_func_out(current_output, mode_write, encoding="utf-8")

    try:
        with open_func_in(input_file, mode_read, encoding="utf-8") as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue

                if by_line:
                    if current_count >= split_size:
                        outf.close()
                        current_file_idx += 1
                        current_output = get_output_path(current_file_idx)
                        output_files.append(current_output)
                        outf = open_func_out(
                            current_output, mode_write, encoding="utf-8"
                        )
                        current_count = 0

                    outf.write(line + "\n")
                    current_count += 1
                else:
                    # Split by byte size
                    line_bytes = len(line.encode("utf-8"))
                    if current_count + line_bytes > split_size and current_count > 0:
                        outf.close()
                        current_file_idx += 1
                        current_output = get_output_path(current_file_idx)
                        output_files.append(current_output)
                        outf = open_func_out(
                            current_output, mode_write, encoding="utf-8"
                        )
                        current_count = 0

                    outf.write(line + "\n")
                    current_count += line_bytes
    finally:
        outf.close()

    return output_files


def deduplicate_jsonl(
    input_file: str,
    output_file: str,
    key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    keep: str = "first",
    compress_output: bool = False,
) -> int:
    """Remove duplicate entries from JSONL file.

    Memory-efficient deduplication using streaming and hashing.

    Args:
        input_file: Input file path
        output_file: Output file path
        key_fn: Function to extract deduplication key (default: entire dict)
        keep: Which duplicate to keep ('first' or 'last')
        compress_output: Whether to compress output

    Returns:
        Number of duplicates removed
    """
    if compress_output and not any(
        output_file.endswith(ext) for ext in [".gz", ".bz2", ".xz"]
    ):
        output_file = f"{output_file}.gz"

    seen = set()
    duplicates_removed = 0

    if keep == "first":
        open_func_out, _, mode_write = _get_file_opener(output_file)

        with open_func_out(output_file, mode_write, encoding="utf-8") as outf:
            for batch in stream_jsonl(input_file, batch_size=1000):
                for item in batch:
                    # Generate key for deduplication
                    if key_fn:
                        key = key_fn(item)
                    else:
                        key = json.dumps(item, sort_keys=True)

                    if key not in seen:
                        seen.add(key)
                        outf.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:
                        duplicates_removed += 1
    else:
        # Keep last: need to read all data first
        items = []
        for batch in stream_jsonl(input_file, batch_size=1000):
            items.extend(batch)

        # Process in reverse and keep track of seen
        unique_items = []
        for item in reversed(items):
            if key_fn:
                key = key_fn(item)
            else:
                key = json.dumps(item, sort_keys=True)

            if key not in seen:
                seen.add(key)
                unique_items.append(item)
            else:
                duplicates_removed += 1

        # Reverse back to original order
        unique_items.reverse()

        # Write output
        write_jsonl(unique_items, output_file, compress=compress_output)

    return duplicates_removed


def sample_jsonl(
    input_file: str,
    output_file: str,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: Optional[int] = None,
    compress_output: bool = False,
):
    """Sample random subset from JSONL file.

    Memory-efficient reservoir sampling for large files.

    Args:
        input_file: Input file path
        output_file: Output file path
        n: Number of samples to take
        fraction: Fraction of data to sample (alternative to n)
        random_state: Random seed for reproducibility
        compress_output: Whether to compress output
    """
    import random

    if random_state is not None:
        random.seed(random_state)

    if n is None and fraction is None:
        raise ValueError("Must specify either n or fraction")

    # Count total lines if fraction is specified
    if fraction is not None:
        total = count_jsonl_lines(input_file, fast=True)
        n = int(total * fraction)

    # Reservoir sampling
    reservoir = []
    open_func_in, mode_read, _ = _get_file_opener(input_file)

    with open_func_in(input_file, mode_read, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if i < n:
                reservoir.append(item)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = item

    # Write sampled data
    write_jsonl(reservoir, output_file, compress=compress_output)


def get_jsonl_stats(filepath: str, sample_size: int = 1000) -> Dict[str, Any]:
    """Get statistics about JSONL file.

    Provides useful metadata about file size, structure, and content.

    Args:
        filepath: File path
        sample_size: Number of items to sample for detailed stats

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_lines": 0,
        "total_bytes": 0,
        "compressed": any(
            filepath.endswith(ext) for ext in [".gz", ".bz2", ".xz", ".lzma"]
        ),
        "keys": set(),
        "sample_items": [],
    }

    # Get file size
    stats["total_bytes"] = Path(filepath).stat().st_size

    # Stream through file
    open_func, mode_read, _ = _get_file_opener(filepath)

    with open_func(filepath, mode_read, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            stats["total_lines"] += 1

            # Sample first N items for analysis
            if i < sample_size:
                try:
                    item = json.loads(line)
                    stats["sample_items"].append(item)
                    if isinstance(item, dict):
                        stats["keys"].update(item.keys())
                except json.JSONDecodeError:
                    pass

    # Convert set to list for JSON serialization
    stats["keys"] = sorted(list(stats["keys"]))
    stats["avg_bytes_per_line"] = (
        stats["total_bytes"] / stats["total_lines"] if stats["total_lines"] > 0 else 0
    )

    return stats


def compress_jsonl(
    input_file: str, output_file: Optional[str] = None, compression_type: str = "gz"
) -> str:
    """Compress an existing JSONL file.

    Args:
        input_file: Input file path
        output_file: Output file path (default: input + compression extension)
        compression_type: Compression type ('gz', 'bz2', 'xz')

    Returns:
        Path to compressed file
    """
    if output_file is None:
        output_file = f"{input_file}.{compression_type}"

    data = read_jsonl(input_file)
    write_jsonl(data, output_file, compress=True, compression_type=compression_type)

    return output_file


def decompress_jsonl(input_file: str, output_file: Optional[str] = None) -> str:
    """Decompress a compressed JSONL file.

    Args:
        input_file: Compressed input file path
        output_file: Output file path (default: remove compression extension)

    Returns:
        Path to decompressed file
    """
    if output_file is None:
        # Remove compression extension
        for ext in [".gz", ".bz2", ".xz", ".lzma"]:
            if input_file.endswith(ext):
                output_file = input_file[: -len(ext)]
                break
        else:
            output_file = f"{input_file}.decompressed.jsonl"

    data = read_jsonl(input_file)
    write_jsonl(data, output_file, compress=False)

    return output_file
