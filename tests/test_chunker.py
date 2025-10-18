import pytest

from kerb.chunk import SemanticChunker, overlap_chunker, simple_chunker


def test_simple_chunker_documentation():
    """Test chunking technical documentation for RAG systems."""
    text = """Machine learning models require careful data preparation. The quality of your training data directly impacts model performance. Common preprocessing steps include normalization, handling missing values, and feature engineering. When working with text data, tokenization is essential. Modern transformers use subword tokenization methods like BPE or WordPiece."""

    chunks = simple_chunker(text, chunk_size=150, overlap=0)

    # Should split into 2-3 chunks depending on exact length
    assert len(chunks) >= 2
    assert all(
        len(chunk) <= 150 for chunk in chunks[:-1]
    )  # All but last should be max size
    assert "Machine learning models" in chunks[0]
    assert "".join(chunks) == text  # Verify no data loss


def test_simple_chunker_overlap_for_context():
    """Test overlapping chunks to maintain context between segments."""
    text = """The transformer architecture revolutionized NLP. It uses self-attention mechanisms to process sequences in parallel. This allows for much faster training compared to RNNs. The key innovation is the attention mechanism."""

    chunks = simple_chunker(text, chunk_size=100, overlap=30)

    # Verify chunks have overlap to maintain context
    assert len(chunks) >= 2
    # Check that consecutive chunks share content
    if len(chunks) > 1:
        # Last 30 chars of first chunk should appear in second chunk
        overlap_text = chunks[0][-30:]
        assert overlap_text in chunks[1]


def test_semantic_chunker_paragraph_structure():
    """Test semantic chunking for maintaining logical document structure."""
    text = """Large language models are trained on massive datasets. They learn to predict the next token in a sequence. This training approach is called autoregressive modeling. The models develop emergent capabilities at scale. Few-shot learning is one such capability. Chain-of-thought prompting improves reasoning. It guides the model through intermediate steps."""

    semantic_chunker = SemanticChunker(sentences_per_chunk=2)
    chunks = semantic_chunker.chunk(text)

    # Should create 4 chunks (7 sentences / 2 per chunk = 3.5 -> 4 chunks)
    assert len(chunks) == 4
    assert (
        "Large language models are trained on massive datasets. They learn to predict the next token in a sequence."
        == chunks[0]
    )
    assert (
        "This training approach is called autoregressive modeling. The models develop emergent capabilities at scale."
        == chunks[1]
    )
    # Verify each chunk maintains complete thoughts
    assert all(chunk.endswith(".") for chunk in chunks)


def test_overlap_chunker_code_documentation():
    """Test chunking code documentation with proportional overlap."""
    text = """def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    '''Generate embeddings for input text using OpenAI API.
    
    Args:
        text: Input text to embed
        model: Embedding model name
        
    Returns:
        List of float values representing the embedding vector
    '''
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding"""

    chunks = overlap_chunker(text, chunk_size=120, overlap_ratio=0.25)

    # Verify chunking with 25% overlap
    assert len(chunks) >= 2
    # Check function signature appears in first chunk
    assert "def embed_text" in chunks[0]
    # Verify overlap maintains context
    if len(chunks) > 1:
        overlap_size = int(120 * 0.25)
        assert chunks[0][-overlap_size:] in chunks[1]


def test_semantic_chunker_qa_pairs():
    """Test chunking Q&A content for training or RAG systems."""
    text = """What is prompt engineering. Prompt engineering is the practice of designing effective inputs for LLMs. How do you improve prompt quality. Use clear instructions, provide examples, and specify output format. What are common prompting techniques. Few-shot learning, chain-of-thought, and retrieval-augmented generation are widely used."""

    semantic_chunker = SemanticChunker(sentences_per_chunk=2)
    chunks = semantic_chunker.chunk(text)

    # Should maintain Q&A pairs together
    assert len(chunks) == 3
    # Each chunk should contain a question and answer pair
    for chunk in chunks:
        # Count sentences (should be 2 per chunk for this input)
        assert chunk.count(".") == 2


def test_simple_chunker_api_response():
    """Test chunking large API responses or logs."""
    # Simulate a JSON-like API response log
    text = """{"model": "gpt-4", "usage": {"prompt_tokens": 150, "completion_tokens": 200, "total_tokens": 350}, "choices": [{"message": {"role": "assistant", "content": "Here is a detailed explanation of the concept..."}}]}"""

    chunks = simple_chunker(text, chunk_size=80, overlap=10)

    # Should create multiple chunks due to small chunk size
    assert len(chunks) >= 2
    # First chunk should start with opening brace
    assert chunks[0].startswith('{"model"')
    # Last chunk should end with closing brace
    assert chunks[-1].endswith("]}")
    # Verify overlap exists between chunks
    for i in range(len(chunks) - 1):
        overlap = chunks[i][-10:]
        assert overlap in chunks[i + 1]
