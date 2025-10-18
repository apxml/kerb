"""Text deduplication operations."""

import hashlib
import re
from typing import Callable, Dict, List, Optional

from .enums import DeduplicationMode
from .text import normalize_whitespace


def deduplicate_exact(texts: List[str], keep_order: bool = True) -> List[str]:
    """Remove exact duplicates.

    Args:
        texts: List of texts
        keep_order: Preserve original order

    Returns:
        List with duplicates removed

    Examples:
        >>> deduplicate_exact(["a", "b", "a", "c"])
        ['a', 'b', 'c']
    """
    if keep_order:
        seen = set()
        result = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                result.append(text)
        return result
    else:
        return list(set(texts))


def deduplicate_fuzzy(
    texts: List[str], similarity_threshold: float = 0.9, keep_order: bool = True
) -> List[str]:
    """Remove fuzzy/near duplicates.

    Args:
        texts: List of texts
        similarity_threshold: Similarity threshold (0-1)
        keep_order: Preserve original order

    Returns:
        List with fuzzy duplicates removed

    Examples:
        >>> deduplicate_fuzzy(["hello world", "hello  world", "goodbye"])
        ['hello world', 'goodbye']
    """
    if not texts:
        return []

    # Normalize texts for comparison
    normalized = [normalize_whitespace(t.lower()) for t in texts]

    result = []
    seen_normalized = []

    for i, text in enumerate(texts):
        norm = normalized[i]

        # Check similarity with already seen texts
        is_duplicate = False
        for seen_norm in seen_normalized:
            similarity = _simple_similarity(norm, seen_norm)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(text)
            seen_normalized.append(norm)

    return result


def deduplicate_semantic(
    texts: List[str],
    similarity_threshold: float = 0.85,
    embed_fn: Optional[Callable] = None,
) -> List[str]:
    """Remove semantically similar texts.

    Args:
        texts: List of texts
        similarity_threshold: Semantic similarity threshold (0-1)
        embed_fn: Optional embedding function (uses simple fallback if None)

    Returns:
        List with semantic duplicates removed

    Examples:
        >>> deduplicate_semantic(["hello", "hi", "goodbye"])
        ['hello', 'goodbye']
    """
    if not texts or len(texts) <= 1:
        return texts

    # If no embedding function provided, fall back to fuzzy
    if embed_fn is None:
        return deduplicate_fuzzy(texts, similarity_threshold)

    # Use provided embedding function
    embeddings = [embed_fn(text) for text in texts]

    result = []
    kept_embeddings = []

    for i, text in enumerate(texts):
        emb = embeddings[i]

        # Check similarity with kept texts
        is_duplicate = False
        for kept_emb in kept_embeddings:
            similarity = _cosine_similarity(emb, kept_emb)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(text)
            kept_embeddings.append(emb)

    return result


def deduplicate_lines(text: str, keep_order: bool = True) -> str:
    """Remove duplicate lines.

    Args:
        text: Input text
        keep_order: Preserve line order

    Returns:
        Text with duplicate lines removed

    Examples:
        >>> deduplicate_lines("line1\\nline2\\nline1\\nline3")
        'line1\\nline2\\nline3'
    """
    if not text:
        return text

    lines = text.split("\n")
    unique_lines = deduplicate_exact(lines, keep_order)
    return "\n".join(unique_lines)


def deduplicate_sentences(text: str, keep_order: bool = True) -> str:
    """Remove duplicate sentences.

    Args:
        text: Input text
        keep_order: Preserve sentence order

    Returns:
        Text with duplicate sentences removed

    Examples:
        >>> deduplicate_sentences("Hello. World. Hello.")
        'Hello. World.'
    """
    if not text:
        return text

    # Simple sentence segmentation
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    unique_sentences = deduplicate_exact(sentences, keep_order)
    return " ".join(unique_sentences)


def find_duplicates(
    texts: List[str], mode: DeduplicationMode = DeduplicationMode.EXACT
) -> List[List[int]]:
    """Find duplicate texts without removing.

    Args:
        texts: List of texts
        mode: Deduplication mode

    Returns:
        List of index groups representing duplicates

    Examples:
        >>> find_duplicates(["a", "b", "a", "c", "b"])
        [[0, 2], [1, 4]]
    """
    if not texts:
        return []

    if mode == DeduplicationMode.EXACT:
        # Group by exact match
        groups: Dict[str, List[int]] = {}
        for i, text in enumerate(texts):
            if text not in groups:
                groups[text] = []
            groups[text].append(i)

        # Return only groups with duplicates
        return [indices for indices in groups.values() if len(indices) > 1]

    else:
        # For fuzzy/semantic, use pairwise comparison
        duplicate_groups = []
        assigned = set()

        for i in range(len(texts)):
            if i in assigned:
                continue

            group = [i]
            for j in range(i + 1, len(texts)):
                if j in assigned:
                    continue

                if mode == DeduplicationMode.FUZZY:
                    similarity = _simple_similarity(texts[i].lower(), texts[j].lower())
                    if similarity >= 0.9:
                        group.append(j)
                        assigned.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                assigned.update(group)

        return duplicate_groups


def compute_text_hash(text: str, algorithm: str = "md5") -> str:
    """Compute stable text hash for deduplication.

    Args:
        text: Input text
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex hash string

    Examples:
        >>> hash1 = compute_text_hash("hello")
        >>> hash2 = compute_text_hash("hello")
        >>> hash1 == hash2
        True
    """
    if not text:
        return ""

    # Normalize text for consistent hashing
    normalized = normalize_whitespace(text.strip().lower())

    if algorithm == "md5":
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


# ============================================================================
# Helper Functions
# ============================================================================


def _simple_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    # Character-based Jaccard similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
