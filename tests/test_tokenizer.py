"""Tests for tokenizer module."""

import pytest

from kerb.tokenizer import (Tokenizer, batch_count_tokens, chars_to_tokens,
                            count_tokens, count_tokens_for_messages,
                            tokens_to_chars, truncate_to_token_limit)


def test_count_tokens_approximate_method():
    """Test approximate token counting without external dependencies."""
    text = "Hello world! This is a test."
    # Using CHAR_4 approximation
    count = count_tokens(text, tokenizer=Tokenizer.CHAR_4)
    # 29 chars / 4 ≈ 7 tokens
    assert count == 7


def test_count_tokens_empty_string():
    """Test token counting with empty string."""
    assert count_tokens("", tokenizer=Tokenizer.CL100K_BASE) == 0
    assert count_tokens("", tokenizer=Tokenizer.CHAR_4) == 0


def test_count_tokens_different_tokenizers():
    """Test that different tokenizers use appropriate approximations."""
    text = "The quick brown fox jumps over the lazy dog"

    # CHAR_4 tokenizer uses 4 chars per token
    char4_count = count_tokens(text, tokenizer=Tokenizer.CHAR_4)
    assert char4_count == len(text) // 4

    # CHAR_5 uses 5 chars per token
    char5_count = count_tokens(text, tokenizer=Tokenizer.CHAR_5)
    assert char5_count == len(text) // 5

    # WORD tokenizer
    word_count = count_tokens(text, tokenizer=Tokenizer.WORD)
    words = len(text.split())
    assert word_count == int(words * 1.3)


def test_batch_count_tokens():
    """Test batch token counting."""
    texts = [
        "Hello world!",
        "How are you today?",
        "This is a longer sentence with more words.",
    ]

    counts = batch_count_tokens(texts, tokenizer=Tokenizer.CHAR_4)

    assert len(counts) == 3
    assert all(isinstance(c, int) for c in counts)
    assert counts[0] == len(texts[0]) // 4
    assert counts[1] == len(texts[1]) // 4
    assert counts[2] == len(texts[2]) // 4


def test_batch_count_tokens_empty_list():
    """Test batch counting with empty list."""
    counts = batch_count_tokens([], tokenizer=Tokenizer.CL100K_BASE)
    assert counts == []


def test_count_tokens_for_messages_basic():
    """Test token counting for chat messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # This will use approximation method since tiktoken is optional
    count = count_tokens_for_messages(messages, tokenizer=Tokenizer.CL100K_BASE)

    # Should include content tokens + overhead
    assert count > 0
    # Each message has ~4 token overhead + reply priming (3 tokens)
    min_expected = (
        (len(messages[0]["content"]) // 4) + (len(messages[1]["content"]) // 4) + 11
    )
    assert count >= min_expected


def test_count_tokens_for_messages_empty():
    """Test message counting with empty list."""
    assert count_tokens_for_messages([], tokenizer=Tokenizer.CL100K_BASE) == 0


def test_tokens_to_chars_conversion():
    """Test converting tokens to estimated character count."""
    # CL100K_BASE: 4 chars per token
    assert tokens_to_chars(100, tokenizer=Tokenizer.CL100K_BASE) == 400
    assert tokens_to_chars(50, tokenizer=Tokenizer.P50K_BASE) == 200
    assert tokens_to_chars(0, tokenizer=Tokenizer.CL100K_BASE) == 0


def test_chars_to_tokens_conversion():
    """Test converting characters to estimated token count."""
    # CL100K_BASE: 4 chars per token
    assert chars_to_tokens(400, tokenizer=Tokenizer.CL100K_BASE) == 100
    assert chars_to_tokens(200, tokenizer=Tokenizer.P50K_BASE) == 50
    assert chars_to_tokens(0, tokenizer=Tokenizer.CL100K_BASE) == 0


def test_truncate_to_token_limit_no_truncation():
    """Test truncation when text is within limit."""
    text = "Hello world!"
    # Text is short, should not be truncated
    result = truncate_to_token_limit(
        text, max_tokens=100, tokenizer=Tokenizer.CL100K_BASE
    )
    assert result == text


def test_truncate_to_token_limit_from_beginning():
    """Test truncation keeping beginning of text."""
    text = "A" * 200  # 200 chars = ~50 tokens
    result = truncate_to_token_limit(
        text, max_tokens=10, tokenizer=Tokenizer.CL100K_BASE
    )

    # Should be truncated and end with ellipsis
    assert len(result) < len(text)
    assert result.endswith("...")
    assert result.startswith("A")


def test_truncate_to_token_limit_preserve_end():
    """Test truncation keeping end of text."""
    text = "A" * 200  # 200 chars = ~50 tokens
    result = truncate_to_token_limit(
        text, max_tokens=10, tokenizer=Tokenizer.CL100K_BASE, preserve_end=True
    )

    # Should be truncated and start with ellipsis
    assert len(result) < len(text)
    assert result.startswith("...")
    assert result.endswith("A")


def test_truncate_to_token_limit_custom_ellipsis():
    """Test truncation with custom ellipsis."""
    text = "A" * 200
    result = truncate_to_token_limit(
        text, max_tokens=10, tokenizer=Tokenizer.CL100K_BASE, ellipsis=" [truncated]"
    )

    assert " [truncated]" in result


def test_truncate_to_token_limit_empty_string():
    """Test truncation with empty string."""
    result = truncate_to_token_limit("", max_tokens=100)
    assert result == ""


def test_different_tokenizers():
    """Test different tokenizer enums."""
    text = "Hello world!"

    # Test CL100K_BASE (GPT-4, GPT-3.5)
    count_cl100k = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
    assert count_cl100k > 0

    # Test P50K_BASE (Codex)
    count_p50k = count_tokens(text, tokenizer=Tokenizer.P50K_BASE)
    assert count_p50k > 0


def test_huggingface_model_string():
    """Test using HuggingFace model names as strings."""
    text = "Hello world!"

    # Should handle HF models passed as strings
    count = count_tokens(text, tokenizer="bert-base-uncased")
    assert count > 0

    # Model with slash (HF format)
    count = count_tokens(text, tokenizer="meta-llama/Llama-2-7b-hf")
    assert count > 0


def test_batch_count_with_mixed_lengths():
    """Test batch counting with various text lengths."""
    texts = [
        "",
        "Short",
        "A" * 100,
        "This is a medium length text with several words.",
    ]

    counts = batch_count_tokens(texts, tokenizer=Tokenizer.CHAR_4)

    assert len(counts) == 4
    assert counts[0] == 0  # Empty string
    assert counts[1] > 0  # Short text
    assert counts[2] == 25  # 100 chars / 4
    assert counts[3] > 0  # Medium text


def test_count_tokens_with_special_characters():
    """Test token counting with special characters and unicode."""
    text = "Hello 👋 world! 🌍 How are you? 😊"
    count = count_tokens(text, tokenizer=Tokenizer.CHAR_4)

    # Should handle unicode characters
    assert count > 0


def test_count_tokens_multiline_text():
    """Test token counting with multiline text."""
    text = """This is a multiline text.
It has several lines.
Each line has some content.
Token counting should work correctly."""

    count = count_tokens(text, tokenizer=Tokenizer.CHAR_4)
    assert count == len(text) // 4


def test_truncate_with_very_small_limit():
    """Test truncation with very small token limit."""
    text = "Hello world! This is a long text."
    result = truncate_to_token_limit(
        text, max_tokens=1, tokenizer=Tokenizer.CL100K_BASE
    )

    # Should handle very small limits gracefully
    assert "..." in result


def test_message_counting_with_name_field():
    """Test message token counting with name field."""
    messages = [
        {"role": "system", "content": "You are helpful.", "name": "system_prompt"},
        {"role": "user", "content": "Hello!"},
    ]

    count = count_tokens_for_messages(messages, tokenizer=Tokenizer.CL100K_BASE)
    assert count > 0


def test_word_based_approximation():
    """Test word-based approximation."""
    text = "one two three four five"
    count = count_tokens(text, tokenizer=Tokenizer.WORD)

    # Should use word-based approximation: 5 words * 1.3 ≈ 6-7 tokens
    assert 6 <= count <= 7


def test_invalid_tokenizer():
    """Test that invalid tokenizer string is handled gracefully."""
    # Invalid tokenizer strings are treated as HuggingFace model names
    # and will fall back to approximation with a warning
    text = "Hello world!"
    # Will warn either about transformers not installed OR invalid model identifier
    with pytest.warns(UserWarning):
        count = count_tokens(text, tokenizer="invalid_tokenizer_enum")
    # Should still return a count (using approximation fallback)
    assert count > 0


# Optional tests that run only if dependencies are installed
try:
    import tiktoken

    def test_count_tokens_tiktoken_cl100k():
        """Test accurate token counting with tiktoken for CL100K_BASE."""
        text = "Hello world!"
        count = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
        # Actual token count with tiktoken
        assert count > 0
        assert isinstance(count, int)

    def test_truncate_with_tiktoken():
        """Test truncation using tiktoken for accurate token-level truncation."""
        text = "This is a longer text that needs to be truncated to fit within limits."
        result = truncate_to_token_limit(
            text, max_tokens=10, tokenizer=Tokenizer.CL100K_BASE
        )

        # Should be properly truncated at token boundary
        assert len(result) < len(text)
        assert result.endswith("...")

        # Verify the result is within token limit
        result_tokens = count_tokens(result, tokenizer=Tokenizer.CL100K_BASE)
        assert result_tokens <= 10

except ImportError:
    pass  # Skip tiktoken tests if not installed


try:
    from transformers import AutoTokenizer

    def test_count_tokens_transformers_bert():
        """Test token counting with transformers for BERT."""
        text = "Hello world!"
        count = count_tokens(text, tokenizer="bert-base-uncased")
        assert count > 0
        assert isinstance(count, int)

except ImportError:
    pass  # Skip transformers tests if not installed
