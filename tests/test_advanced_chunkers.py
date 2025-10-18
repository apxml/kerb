import pytest

from kerb.chunk import (CodeChunker, MarkdownChunker, custom_chunker,
                        paragraph_chunker, recursive_chunker,
                        sentence_window_chunker, sliding_window_chunker,
                        token_based_chunker)


def test_paragraph_chunker():
    """Test paragraph-based chunking."""
    text = """First paragraph with some content.

Second paragraph with more information.

Third paragraph here.

Fourth paragraph content."""

    chunks = paragraph_chunker(text, max_paragraphs=2)

    assert len(chunks) == 2
    assert "First paragraph" in chunks[0]
    assert "Second paragraph" in chunks[0]
    assert "Third paragraph" in chunks[1]
    assert "Fourth paragraph" in chunks[1]


def test_token_based_chunker():
    """Test approximate token-based chunking."""
    text = "This is a test " * 200  # ~600 words, ~800 tokens approx

    chunks = token_based_chunker(text, max_tokens=256)

    assert len(chunks) >= 2
    # Each chunk should be roughly 256 * 4 = 1024 chars
    for chunk in chunks[:-1]:
        assert len(chunk) <= 1100  # Allow some buffer


def test_recursive_chunker():
    """Test recursive chunking with hierarchy of separators."""
    text = """# Section 1

Paragraph 1 content here.

Paragraph 2 content here.

## Subsection

More content in subsection."""

    chunks = recursive_chunker(text, chunk_size=50)

    # Should split on paragraph boundaries first
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_sliding_window_chunker():
    """Test sliding window approach for overlapping context."""
    text = "A" * 1000

    chunks = sliding_window_chunker(text, window_size=300, stride=200)

    # Should create overlapping windows
    assert len(chunks) >= 4
    # Verify overlap exists
    if len(chunks) > 1:
        assert chunks[0][200:300] == chunks[1][:100]


def test_markdown_chunker():
    """Test markdown-aware chunking."""
    text = """# Main Title

Introduction paragraph.

## Section 1

Content for section 1.

## Section 2

Content for section 2.

### Subsection 2.1

Subsection content."""

    chunker = MarkdownChunker(max_chunk_size=100)
    chunks = chunker.chunk(text)

    # Should respect markdown structure
    assert len(chunks) >= 2
    # First chunk should contain main title
    assert "# Main Title" in chunks[0]
    # Sections should be properly split
    assert any("Section 2" in chunk for chunk in chunks)


def test_code_chunker_python():
    """Test code chunking for Python code."""
    text = """import os

def function_one():
    '''First function.'''
    return 1

def function_two():
    '''Second function.'''
    return 2

class MyClass:
    '''A sample class.'''
    def method(self):
        return 3"""

    chunker = CodeChunker(max_chunk_size=100, language="python")
    chunks = chunker.chunk(text)

    # Should split on function/class boundaries
    assert len(chunks) >= 2
    # Make sure we captured some code
    assert any("def function" in chunk for chunk in chunks)


def test_code_chunker_python():
    """Test code-aware chunking for Python."""
    code = """def function1():
    '''First function.'''
    pass

class MyClass:
    '''A test class.'''
    
    def method1(self):
        pass

def function2():
    '''Second function.'''
    return 42"""

    chunker = CodeChunker(max_chunk_size=80, language="python")
    chunks = chunker.chunk(code)

    assert len(chunks) > 0
    # Should have split on function/class boundaries
    assert any("function1" in chunk for chunk in chunks)
    assert any("MyClass" in chunk for chunk in chunks)


def test_custom_chunker():
    """Test custom chunker with user-defined split function."""
    text = "word1,word2,word3,word4,word5,word6,word7,word8"

    # Custom split on commas
    def comma_splitter(t):
        return t.split(",")

    chunks = custom_chunker(text, chunk_size=20, split_fn=comma_splitter)

    # Should combine words until chunk size reached
    assert len(chunks) >= 2
    assert all("," not in chunk or chunk.count(",") < 4 for chunk in chunks)


def test_sentence_window_chunker():
    """Test sentence window chunker with overlap."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."

    chunks = sentence_window_chunker(text, window_sentences=3, overlap_sentences=1)

    # Should create overlapping sentence windows
    assert len(chunks) >= 2
    # Each chunk should end with period
    assert all(chunk.strip().endswith(".") for chunk in chunks)
    # Check for overlap
    if len(chunks) > 1:
        # Last sentence of first window should appear in second window
        assert "Third sentence" in chunks[0]
        assert "Third sentence" in chunks[1]


def test_recursive_chunker_large_text():
    """Test recursive chunker on realistic document."""
    text = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.

## Supervised Learning

In supervised learning, we train models on labeled data. The model learns to map inputs to outputs.

### Classification

Classification is used for categorical predictions. Examples include spam detection and image recognition.

### Regression

Regression predicts continuous values. Common applications include price prediction and forecasting."""

    chunks = recursive_chunker(text, chunk_size=150)

    # Should create multiple chunks
    assert len(chunks) >= 3
    # Verify content preservation
    combined = "".join(chunks)
    # Check that key content is preserved (allowing for separator differences)
    assert "Machine learning" in combined
    assert "Supervised Learning" in combined
    assert "Classification" in combined
    assert "Regression" in combined


def test_paragraph_chunker_single_paragraph():
    """Test paragraph chunker with single paragraph."""
    text = "This is just one paragraph without any line breaks."

    chunks = paragraph_chunker(text, max_paragraphs=2)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_sliding_window_edge_case():
    """Test sliding window with text smaller than window."""
    text = "Short text"

    chunks = sliding_window_chunker(text, window_size=100, stride=50)

    assert len(chunks) == 1
    assert chunks[0] == text
