"""Text-based chunking implementations."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class Chunker(ABC):
    """Abstract base class for all chunker implementations.

    All chunker classes should inherit from this base class and implement
    the chunk method.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text (str): The text to chunk

        Returns:
            List[str]: List of text chunks
        """
        pass


class RecursiveChunker(Chunker):
    """Recursively split text using a hierarchy of separators.

    Tries to split on larger semantic boundaries first (paragraphs, sentences)
    before falling back to character-level splitting. Similar to LangChain's
    RecursiveCharacterTextSplitter.

    Args:
        chunk_size (int): Target size for each chunk. Defaults to 1000.
        separators (List[str], optional): List of separators in priority order.
            Defaults to ['\\n\\n', '\\n', '. ', ' ', ''].

    Examples:
        >>> chunker = RecursiveChunker(chunk_size=500)
        >>> chunks = chunker.chunk("Your long text here...")
    """

    def __init__(self, chunk_size: int = 1000, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.separators = (
            separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks recursively.

        Args:
            text (str): The text to chunk

        Returns:
            List[str]: List of recursively split chunks
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Internal recursive splitting logic."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []

        # Try each separator in order
        for separator in separators:
            if separator == "":
                # Last resort: split by character
                from .utils import chunk_text as _chunk_text

                return _chunk_text(text, chunk_size=self.chunk_size)

            if separator in text:
                splits = text.split(separator)
                current_chunk = []
                current_size = 0

                for split in splits:
                    split_with_sep = split + separator if split != splits[-1] else split
                    split_size = len(split_with_sep)

                    # If single split is too large, recurse with next separator
                    if split_size > self.chunk_size:
                        # Save current chunk if exists
                        if current_chunk:
                            chunks.append("".join(current_chunk).rstrip(separator))
                            current_chunk = []
                            current_size = 0

                        # Recurse with remaining separators
                        remaining_seps = separators[separators.index(separator) + 1 :]
                        chunks.extend(self._recursive_split(split, remaining_seps))
                        continue

                    # Check if adding this split would exceed chunk size
                    if current_size + split_size > self.chunk_size and current_chunk:
                        chunks.append("".join(current_chunk).rstrip(separator))
                        current_chunk = []
                        current_size = 0

                    current_chunk.append(split_with_sep)
                    current_size += split_size

                # Add remaining chunk
                if current_chunk:
                    chunks.append("".join(current_chunk).rstrip(separator))

                return chunks

        return [text]


class SentenceChunker(Chunker):
    """Split text into chunks based on sentence boundaries with optional overlap.

    Args:
        window_sentences (int): Number of sentences per chunk. Defaults to 5.
        overlap_sentences (int): Number of sentences to overlap. Defaults to 1.

    Examples:
        >>> chunker = SentenceChunker(window_sentences=3, overlap_sentences=1)
        >>> chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")
    """

    def __init__(self, window_sentences: int = 5, overlap_sentences: int = 1):
        self.window_sentences = window_sentences
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str) -> List[str]:
        """Split text into sentence-based chunks with overlap.

        Args:
            text (str): The text to chunk

        Returns:
            List[str]: List of sentence-windowed chunks
        """
        if not text:
            return []

        # Split into sentences (simple approach)
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

        if not sentences:
            return []

        chunks = []
        i = 0

        while i < len(sentences):
            chunk_sentences = sentences[i : i + self.window_sentences]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)

            # Move forward by (window_sentences - overlap_sentences)
            stride = max(1, self.window_sentences - self.overlap_sentences)
            i += stride

            if i >= len(sentences):
                break

        return chunks


def simple_chunker(text: str, chunk_size: int = 1000, overlap: int = 0) -> List[str]:
    """Split text into chunks of specified size.

    Args:
        text (str): The text to chunk
        chunk_size (int): Maximum size of each chunk. Defaults to 1000.
        overlap (int): Number of characters to overlap between chunks. Defaults to 0.

    Returns:
        List[str]: List of text chunks
    """
    from .utils import chunk_text as _chunk_text

    return _chunk_text(text, chunk_size, overlap)


def overlap_chunker(
    text: str, chunk_size: int = 1000, overlap_ratio: float = 0.1
) -> List[str]:
    """Split text with proportional overlap between chunks.

    Args:
        text (str): The text to chunk
        chunk_size (int): Maximum size of each chunk. Defaults to 1000.
        overlap_ratio (float): Proportion of chunk to overlap (0.0-1.0). Defaults to 0.1.

    Returns:
        List[str]: List of overlapping text chunks
    """
    from .utils import chunk_text as _chunk_text

    overlap = int(chunk_size * overlap_ratio)
    return _chunk_text(text, chunk_size, overlap)


def paragraph_chunker(text: str, max_paragraphs: int = 3) -> List[str]:
    """Split text into chunks based on paragraph boundaries.

    Args:
        text (str): The text to chunk
        max_paragraphs (int): Maximum number of paragraphs per chunk. Defaults to 3.

    Returns:
        List[str]: List of paragraph-based chunks
    """
    if not text:
        return []

    # Split by double newlines (common paragraph separator)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk_paragraphs = paragraphs[i : i + max_paragraphs]
        chunk = "\n\n".join(chunk_paragraphs)
        chunks.append(chunk)

    return chunks


def sliding_window_chunker(
    text: str, window_size: int = 1000, stride: int = 500
) -> List[str]:
    """Create chunks using a sliding window approach.

    Similar to simple_chunker with overlap, but stride-based for more control.
    Common in NLP tasks and document processing pipelines.

    Args:
        text (str): The text to chunk
        window_size (int): Size of each window/chunk. Defaults to 1000.
        stride (int): Number of characters to move forward for next window. Defaults to 500.

    Returns:
        List[str]: List of sliding window chunks
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += stride

    return chunks


def token_based_chunker(text: str, max_tokens: int = 512, tokenizer=None) -> List[str]:
    """Split text based on token count.

    Uses the specified tokenizer to estimate chunk sizes. For accurate token-based
    chunking with OpenAI models, ensure tiktoken is installed.

    Args:
        text (str): The text to chunk
        max_tokens (int): Maximum tokens per chunk. Defaults to 512.
        tokenizer: Tokenizer to use for estimation. If None, uses character approximation.

    Returns:
        List[str]: List of token-based chunks

    Examples:
        >>> from kerb.tokenizer import Tokenizer
        >>> chunks = token_based_chunker(text, max_tokens=512, tokenizer=Tokenizer.CL100K_BASE)
    """
    if not text:
        return []

    from .utils import chunk_text as _chunk_text

    # Convert tokens to approximate character count
    if tokenizer is not None:
        from ..tokenizer import tokens_to_chars

        chunk_size = tokens_to_chars(max_tokens, tokenizer)
    else:
        # Rough approximation: 1 token ≈ 4 characters
        chunk_size = max_tokens * 4

    return _chunk_text(text, chunk_size=chunk_size, overlap=0)


def recursive_chunker(
    text: str, chunk_size: int = 1000, separators: Optional[List[str]] = None
) -> List[str]:
    """Recursively split text using a hierarchy of separators.

    Functional interface for RecursiveChunker.

    Args:
        text (str): The text to chunk
        chunk_size (int): Target size for each chunk. Defaults to 1000.
        separators (List[str], optional): List of separators in priority order.
            Defaults to ['\\n\\n', '\\n', '. ', ' ', ''].

    Returns:
        List[str]: List of recursively split chunks
    """
    chunker = RecursiveChunker(chunk_size=chunk_size, separators=separators)
    return chunker.chunk(text)


def sentence_window_chunker(
    text: str, window_sentences: int = 5, overlap_sentences: int = 1
) -> List[str]:
    """Create overlapping chunks based on sentence boundaries.

    Functional interface for SentenceChunker.

    Args:
        text (str): The text to chunk
        window_sentences (int): Number of sentences per chunk. Defaults to 5.
        overlap_sentences (int): Number of sentences to overlap. Defaults to 1.

    Returns:
        List[str]: List of sentence-windowed chunks
    """
    chunker = SentenceChunker(
        window_sentences=window_sentences, overlap_sentences=overlap_sentences
    )
    return chunker.chunk(text)
