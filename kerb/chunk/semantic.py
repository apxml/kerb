"""Semantic-based chunking implementations."""

import re
from typing import List

from .text import Chunker


class SemanticChunker(Chunker):
    """Split text into semantic chunks based on sentences.

    This chunker groups sentences together into chunks, attempting to maintain
    semantic coherence by keeping related sentences together.

    Args:
        sentences_per_chunk (int): Number of sentences per chunk. Defaults to 3.

    Examples:
        >>> chunker = SemanticChunker(sentences_per_chunk=5)
        >>> chunks = chunker.chunk("Your text here...")
    """

    def __init__(self, sentences_per_chunk: int = 3):
        self.sentences_per_chunk = sentences_per_chunk

    def chunk(self, text: str) -> List[str]:
        """Split text into semantic chunks.

        Args:
            text (str): The text to chunk

        Returns:
            List[str]: List of semantic text chunks
        """
        if not text:
            return []

        # Simple sentence splitting (can be enhanced with more sophisticated NLP)
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.sentences_per_chunk]
            chunk = ". ".join(chunk_sentences) + "."
            chunks.append(chunk)

        return chunks
