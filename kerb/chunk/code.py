"""Code-aware chunking implementations."""

import re
from typing import List

from .text import Chunker, paragraph_chunker


class CodeChunker(Chunker):
    """Split code into chunks while respecting code structure.

    Attempts to split on function/class boundaries to maintain semantic coherence.

    Args:
        max_chunk_size (int): Maximum size per chunk. Defaults to 1000.
        language (str): Programming language (for language-specific handling). Defaults to "python".

    Examples:
        >>> chunker = CodeChunker(max_chunk_size=500, language="python")
        >>> chunks = chunker.chunk(code_text)
    """

    def __init__(self, max_chunk_size: int = 1000, language: str = "python"):
        self.max_chunk_size = max_chunk_size
        self.language = language

    def chunk(self, text: str) -> List[str]:
        """Split code into chunks.

        Args:
            text (str): Code text to chunk

        Returns:
            List[str]: List of code chunks
        """
        if not text:
            return []

        # Python-specific patterns (can be extended for other languages)
        if self.language.lower() == "python":
            # Split on function and class definitions
            pattern = r"\n(?=(?:def |class |async def ))"
            sections = re.split(pattern, text)
        else:
            # Generic split on double newlines for other languages
            sections = re.split(r"\n\n", text)

        chunks = []
        current_chunk = []
        current_size = 0

        for section in sections:
            if not section.strip():
                continue

            section_size = len(section)

            # If single section (e.g., large function) is too large
            if section_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large section by lines
                chunks.append(section)
                continue

            # Check if adding section would exceed max size
            if current_size + section_size > self.max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(section)
            current_size += section_size + 1  # +1 for newline

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
