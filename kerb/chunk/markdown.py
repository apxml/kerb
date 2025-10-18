"""Markdown-aware chunking implementations."""

import re
from typing import List

from .text import Chunker, paragraph_chunker


class MarkdownChunker(Chunker):
    """Split markdown text based on heading hierarchy.

    Respects markdown structure by splitting on headers while trying
    to keep related content together.

    Args:
        max_chunk_size (int): Maximum size per chunk. Defaults to 1000.

    Examples:
        >>> chunker = MarkdownChunker(max_chunk_size=500)
        >>> chunks = chunker.chunk(markdown_text)
    """

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """Split markdown text into chunks.

        Args:
            text (str): Markdown text to chunk

        Returns:
            List[str]: List of markdown-aware chunks
        """
        if not text:
            return []

        # Split on markdown headers (# ## ### etc)
        header_pattern = r"\n(?=#{1,6}\s)"
        sections = re.split(header_pattern, text)

        chunks = []
        current_chunk = []
        current_size = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_size = len(section)

            # If single section is too large, split it further
            if section_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large section by paragraphs
                para_chunks = paragraph_chunker(section, max_paragraphs=2)
                chunks.extend(para_chunks)
                continue

            # Check if adding section would exceed max size
            if current_size + section_size > self.max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(section)
            current_size += section_size + 2  # +2 for newlines

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
