"""Utility functions for chunk manipulation and optimization."""

from typing import Callable, List, Optional


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 0) -> List[str]:
    """Simple utility function to split text into chunks of specified size.

    This is a convenience function for basic chunking needs without creating
    a chunker instance.

    Args:
        text (str): The text to chunk
        chunk_size (int): Maximum size of each chunk. Defaults to 1000.
        overlap (int): Number of characters to overlap between chunks. Defaults to 0.

    Returns:
        List[str]: List of text chunks

    Examples:
        >>> text = "Your long document here..."
        >>> chunks = chunk_text(text, chunk_size=500, overlap=50)
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def merge_chunks(
    chunks: List[str], max_size: int = 2000, separator: str = "\n\n"
) -> List[str]:
    """Merge smaller chunks together up to a maximum size.

    Useful for optimizing chunk sizes after initial splitting or when dealing
    with many small chunks that could be combined for better efficiency.

    Args:
        chunks (List[str]): List of text chunks to merge
        max_size (int): Maximum size of merged chunks. Defaults to 2000.
        separator (str): Separator to use when joining chunks. Defaults to "\\n\\n".

    Returns:
        List[str]: List of merged chunks

    Examples:
        >>> small_chunks = ["chunk1", "chunk2", "chunk3"]
        >>> merged = merge_chunks(small_chunks, max_size=100)
    """
    if not chunks:
        return []

    merged = []
    current_chunk = []
    current_size = 0

    for chunk in chunks:
        chunk_size = len(chunk)
        sep_size = len(separator) if current_chunk else 0

        # Check if adding this chunk would exceed max size
        if current_size + chunk_size + sep_size > max_size and current_chunk:
            merged.append(separator.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(chunk)
        current_size += chunk_size + sep_size

    # Add remaining chunk
    if current_chunk:
        merged.append(separator.join(current_chunk))

    return merged


def optimize_chunk_size(
    text: str, target_size: int = 1000, tolerance: float = 0.2
) -> int:
    """Calculate an optimized chunk size based on text length and target.

    Adjusts the chunk size to minimize uneven chunks and ensure better
    distribution of content across chunks.

    Args:
        text (str): The text to analyze
        target_size (int): Target chunk size. Defaults to 1000.
        tolerance (float): Acceptable variance from target (0.0-1.0). Defaults to 0.2.

    Returns:
        int: Optimized chunk size

    Examples:
        >>> text = "Your long document..."
        >>> optimal_size = optimize_chunk_size(text, target_size=500, tolerance=0.15)
    """
    if not text:
        return target_size

    text_length = len(text)

    # If text is smaller than target, return text length
    if text_length <= target_size:
        return text_length

    # Calculate number of chunks with target size
    num_chunks = text_length / target_size

    # Round to nearest integer
    num_chunks_rounded = round(num_chunks)

    # Calculate optimized size
    optimized_size = text_length // num_chunks_rounded

    # Ensure it's within tolerance
    min_size = int(target_size * (1 - tolerance))
    max_size = int(target_size * (1 + tolerance))

    # Clamp to tolerance range
    if optimized_size < min_size:
        optimized_size = min_size
    elif optimized_size > max_size:
        optimized_size = max_size

    return optimized_size


def custom_chunker(
    text: str,
    chunk_size: int = 1000,
    split_fn: Optional[Callable[[str], List[str]]] = None,
) -> List[str]:
    """Split text using a custom splitting function.

    Provides flexibility for domain-specific chunking strategies.

    Args:
        text (str): The text to chunk
        chunk_size (int): Target chunk size. Defaults to 1000.
        split_fn (Callable, optional): Custom function that takes text and returns list of segments.
            If None, uses simple character-based splitting.

    Returns:
        List[str]: List of custom-split chunks

    Examples:
        >>> def my_splitter(text):
        ...     return text.split('|')  # Split on custom delimiter
        >>> chunks = custom_chunker(text, split_fn=my_splitter)
    """
    if not text:
        return []

    if split_fn is None:
        return chunk_text(text, chunk_size)

    # Use custom split function
    segments = split_fn(text)

    # Combine segments into chunks of appropriate size
    chunks = []
    current_chunk = []
    current_size = 0

    for segment in segments:
        segment_size = len(segment)

        if current_size + segment_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(segment)
        current_size += segment_size + 1  # +1 for space

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
