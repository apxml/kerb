"""Test that all kerb modules can be imported without circular import errors.

This test ensures the package structure doesn't introduce circular dependencies
that would break for end users installing via pip.
"""

import pytest


def test_import_chunk():
    """Test chunk module can be imported."""
    from kerb.chunk import chunk_text
    assert chunk_text is not None


def test_import_agent():
    """Test agent module can be imported."""
    from kerb.agent import Agent, run_agent
    assert Agent is not None
    assert run_agent is not None


def test_import_core():
    """Test core module can be imported."""
    from kerb.core import Document, Message
    assert Document is not None
    assert Message is not None


def test_import_embedding():
    """Test embedding module can be imported."""
    from kerb.embedding import embed
    assert embed is not None


def test_import_generation():
    """Test generation module can be imported."""
    from kerb.generation import generate
    assert generate is not None


def test_import_memory():
    """Test memory module can be imported."""
    from kerb.memory import ConversationBuffer
    assert ConversationBuffer is not None


def test_import_retrieval():
    """Test retrieval module can be imported."""
    from kerb.retrieval import rerank_results
    assert rerank_results is not None


def test_import_prompt():
    """Test prompt module can be imported."""
    from kerb.prompt import PromptVersion
    assert PromptVersion is not None


def test_import_tokenizer():
    """Test tokenizer module can be imported."""
    from kerb.tokenizer import count_tokens
    assert count_tokens is not None


def test_import_evaluation():
    """Test evaluation module can be imported."""
    from kerb.evaluation import calculate_bleu
    assert calculate_bleu is not None


def test_import_parsing():
    """Test parsing module can be imported."""
    from kerb.parsing import parse_json
    assert parse_json is not None


def test_import_document():
    """Test document module can be imported."""
    from kerb.document import load_document
    assert load_document is not None


def test_import_preprocessing():
    """Test preprocessing module can be imported."""
    from kerb.preprocessing import normalize_text
    assert normalize_text is not None


def test_import_cache():
    """Test cache module can be imported."""
    from kerb.cache import MemoryCache
    assert MemoryCache is not None


def test_import_config():
    """Test config module can be imported."""
    from kerb.config import ModelConfig
    assert ModelConfig is not None


def test_import_safety():
    """Test safety module can be imported."""
    from kerb.safety import check_profanity
    assert check_profanity is not None


def test_import_all_modules_together():
    """Test that all modules can be imported together without conflicts."""
    from kerb.chunk import chunk_text
    from kerb.agent import Agent
    from kerb.core import Document
    from kerb.embedding import embed
    from kerb.generation import generate
    from kerb.memory import ConversationBuffer
    
    # If we get here without ImportError, the test passes
    assert True


def test_lazy_import_from_kerb():
    """Test that lazy importing works from top-level kerb package."""
    import kerb
    
    # Access submodules through lazy import
    assert hasattr(kerb, 'chunk')
    assert hasattr(kerb, 'agent')
    assert hasattr(kerb, 'core')
    
    # Verify they're actually modules
    assert kerb.chunk is not None
    assert kerb.agent is not None
    assert kerb.core is not None


def test_version_accessible():
    """Test that __version__ is accessible."""
    from kerb import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0
