"""Context compression utilities.

This module provides functions for compressing context to fit within token limits
using various compression strategies.
"""

import re

from kerb.tokenizer import Tokenizer, count_tokens

from .types import (CompressionMethod, CompressionResult, ContextItem,
                    ContextWindow)


def compress_context(
    content: str,
    target_tokens: int,
    method: CompressionMethod = CompressionMethod.SUMMARIZE,
    model: str = "gpt-3.5-turbo",
) -> CompressionResult:
    """Compress context to target token count.

    Args:
        content: Content to compress
        target_tokens: Target token count
        method: Compression method to use
        model: Model for token estimation (not used with tokenizer module, kept for backward compatibility)

    Returns:
        CompressionResult: Compression result with metrics

    Example:
        >>> result = compress_context(long_text, target_tokens=500)
        >>> print(f"Compressed to {result.compression_ratio:.1%}")
    """
    original_tokens = count_tokens(content, Tokenizer.CL100K_BASE)

    if original_tokens <= target_tokens:
        return CompressionResult(
            compressed_content=content,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            method=method,
        )

    if method == CompressionMethod.SUMMARIZE:
        compressed = _compress_by_summarize(content, target_tokens)
    elif method == CompressionMethod.EXTRACT_KEY_INFO:
        compressed = _compress_by_extraction(content, target_tokens)
    elif method == CompressionMethod.REMOVE_REDUNDANCY:
        compressed = _compress_by_deduplication(content, target_tokens)
    elif method == CompressionMethod.ABBREVIATE:
        compressed = _compress_by_abbreviation(content, target_tokens)
    else:  # HYBRID
        compressed = _compress_hybrid(content, target_tokens)

    compressed_tokens = count_tokens(compressed, Tokenizer.CL100K_BASE)

    return CompressionResult(
        compressed_content=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=(
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        ),
        method=method,
    )


def auto_compress_window(
    window: ContextWindow,
    target_ratio: float = 0.7,
    method: CompressionMethod = CompressionMethod.SUMMARIZE,
) -> ContextWindow:
    """Automatically compress context window items.

    Args:
        window: Context window to compress
        target_ratio: Target compression ratio (0-1)
        method: Compression method to use

    Returns:
        ContextWindow: Window with compressed items

    Example:
        >>> compressed_window = auto_compress_window(window, target_ratio=0.7)
    """
    target_tokens = int(window.current_tokens * target_ratio)

    compressed_items = []
    for item in window.items:
        if item.token_count:
            item_target = int(item.token_count * target_ratio)
            result = compress_context(item.content, item_target, method)

            compressed_item = ContextItem(
                content=result.compressed_content,
                priority=item.priority,
                token_count=result.compressed_tokens,
                metadata={
                    **item.metadata,
                    "compressed": True,
                    "compression_ratio": result.compression_ratio,
                },
                timestamp=item.timestamp,
                item_type=item.item_type,
            )
            compressed_items.append(compressed_item)
        else:
            compressed_items.append(item)

    total_tokens = sum(item.token_count or 0 for item in compressed_items)

    return ContextWindow(
        items=compressed_items,
        max_tokens=window.max_tokens,
        current_tokens=total_tokens,
        strategy=window.strategy,
        metadata={**window.metadata, "auto_compressed": True},
    )


# ============================================================================
# Internal Compression Helpers
# ============================================================================


def _compress_by_summarize(content: str, target_tokens: int) -> str:
    """Compress by creating summary (placeholder for LLM-based summarization)."""
    # In production, this would call an LLM to summarize
    # For now, use simple truncation with ellipsis
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    result = []
    current_tokens = 0
    target_chars = target_tokens * 4  # Approximate

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, Tokenizer.CL100K_BASE)
        if current_tokens + sentence_tokens <= target_tokens:
            result.append(sentence)
            current_tokens += sentence_tokens
        else:
            break

    return ". ".join(result) + "."


def _compress_by_extraction(content: str, target_tokens: int) -> str:
    """Extract key information (entities, facts, etc.)."""
    # Simple extraction: get important-looking sentences
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Score sentences by importance indicators
    scored_sentences = []
    for sentence in sentences:
        score = 0
        # Boost sentences with numbers, proper nouns (capitalized words)
        score += len(re.findall(r"\d+", sentence)) * 2
        score += len(re.findall(r"\b[A-Z][a-z]+", sentence))
        # Boost sentences with key words
        keywords = ["important", "key", "critical", "main", "primary", "essential"]
        for keyword in keywords:
            if keyword in sentence.lower():
                score += 3

        scored_sentences.append((score, sentence))

    # Sort by score and select top sentences
    scored_sentences.sort(reverse=True)

    result = []
    current_tokens = 0

    for score, sentence in scored_sentences:
        sentence_tokens = count_tokens(sentence, Tokenizer.CL100K_BASE)
        if current_tokens + sentence_tokens <= target_tokens:
            result.append(sentence)
            current_tokens += sentence_tokens

    return ". ".join(result) + "."


def _compress_by_deduplication(content: str, target_tokens: int) -> str:
    """Remove redundant information."""
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Track seen content using simple word sets
    seen_sentences = set()
    unique_sentences = []

    for sentence in sentences:
        # Create a normalized version for comparison
        words = set(sentence.lower().split())
        sentence_sig = frozenset(words)

        # Check similarity to seen sentences
        is_duplicate = False
        for seen_sig in seen_sentences:
            # If more than 70% overlap, consider duplicate
            overlap = len(sentence_sig & seen_sig)
            union = len(sentence_sig | seen_sig)
            if union > 0 and overlap / union > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence_sig)

    # Now truncate to target tokens
    result = []
    current_tokens = 0

    for sentence in unique_sentences:
        sentence_tokens = count_tokens(sentence, Tokenizer.CL100K_BASE)
        if current_tokens + sentence_tokens <= target_tokens:
            result.append(sentence)
            current_tokens += sentence_tokens
        else:
            break

    return ". ".join(result) + "."


def _compress_by_abbreviation(content: str, target_tokens: int) -> str:
    """Compress using abbreviations and shorter forms."""
    # Common abbreviations
    abbreviations = {
        "and": "&",
        "approximately": "~",
        "for example": "e.g.",
        "that is": "i.e.",
        "versus": "vs",
        "with respect to": "w.r.t.",
        "without": "w/o",
    }

    compressed = content
    for full, abbr in abbreviations.items():
        compressed = re.sub(r"\b" + full + r"\b", abbr, compressed, flags=re.IGNORECASE)

    # Remove extra whitespace
    compressed = re.sub(r"\s+", " ", compressed)

    # Truncate if still too long
    if count_tokens(compressed, Tokenizer.CL100K_BASE) > target_tokens:
        target_chars = target_tokens * 4
        compressed = compressed[:target_chars]

    return compressed


def _compress_hybrid(content: str, target_tokens: int) -> str:
    """Use multiple compression methods."""
    # First remove redundancy
    step1 = _compress_by_deduplication(content, int(target_tokens * 1.5))
    # Then extract key info
    step2 = _compress_by_extraction(step1, int(target_tokens * 1.2))
    # Finally abbreviate
    step3 = _compress_by_abbreviation(step2, target_tokens)

    return step3
