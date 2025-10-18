"""Comprehensive tests for the generation module."""

import pytest

from kerb.generation import (MODEL_PRICING, CostTracker, GenerationConfig,
                             GenerationResponse, LLMProvider, Message,
                             MessageRole, RateLimiter, ResponseCache,
                             StreamChunk, Usage, calculate_cost,
                             format_messages, generate, generate_batch,
                             generate_stream, get_cost_summary,
                             parse_json_response, reset_cost_tracking,
                             validate_response)
# Use tokenizer module for token counting instead of deprecated count_tokens_estimate
from kerb.tokenizer import Tokenizer, count_tokens

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_message():
    """Create a sample message."""
    return Message(role=MessageRole.USER, content="Hello, how are you?")


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is Python?"),
    ]


@pytest.fixture
def sample_config():
    """Create a sample generation config."""
    return GenerationConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=100)


@pytest.fixture
def sample_response():
    """Create a sample generation response."""
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return GenerationResponse(
        content="Python is a high-level programming language.",
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        usage=usage,
        finish_reason="stop",
        latency=0.5,
        cost=0.001,
    )


@pytest.fixture(autouse=True)
def reset_costs():
    """Reset cost tracking before each test."""
    reset_cost_tracking()
    yield
    reset_cost_tracking()


# ============================================================================
# Data Class Tests
# ============================================================================


def test_message_creation():
    """Test Message data class creation."""
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.name is None
    assert msg.function_call is None


def test_message_to_dict():
    """Test Message to_dict conversion."""
    msg = Message(role=MessageRole.USER, content="Hello", name="Alice")
    msg_dict = msg.to_dict()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Hello"
    assert msg_dict["name"] == "Alice"


def test_generation_config_defaults():
    """Test GenerationConfig default values."""
    config = GenerationConfig(model="gpt-4o-mini")
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.top_p == 1.0
    assert config.stream is False


def test_usage_total_tokens():
    """Test Usage total tokens calculation."""
    usage = Usage(prompt_tokens=10, completion_tokens=20)
    assert usage.total_tokens == 0  # Not automatically calculated in dataclass

    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.total_tokens == 30


def test_generation_response_to_dict(sample_response):
    """Test GenerationResponse to_dict conversion."""
    response_dict = sample_response.to_dict()
    assert response_dict["content"] == sample_response.content
    assert response_dict["model"] == sample_response.model
    assert response_dict["provider"] == "openai"
    assert "usage" in response_dict


def test_stream_chunk_creation():
    """Test StreamChunk creation."""
    chunk = StreamChunk(content="Hello", finish_reason=None, model="gpt-4o-mini")
    assert chunk.content == "Hello"
    assert chunk.finish_reason is None
    assert chunk.model == "gpt-4o-mini"


# ============================================================================
# Provider Detection Tests
# ============================================================================


def test_provider_validation():
    """Test that provider parameter is validated."""
    import pytest

    # Should raise ValueError when provider not specified
    with pytest.raises(ValueError, match="Provider must be specified"):
        generate("Test", model="gpt-4o-mini")

    # Should work when provider is spxecified
    response = generate("Test", model="custom-model", provider=LLMProvider.LOCAL)
    assert response.provider == LLMProvider.LOCAL


# ============================================================================
# Cost Calculation Tests
# ============================================================================


def test_calculate_cost_openai():
    """Test cost calculation for OpenAI models."""
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = calculate_cost("gpt-4o-mini", usage)
    expected = (1000 / 1_000_000 * 0.150) + (500 / 1_000_000 * 0.600)
    assert abs(cost - expected) < 0.0001


def test_calculate_cost_anthropic():
    """Test cost calculation for Anthropic models."""
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = calculate_cost("claude-3-5-sonnet-20241022", usage)
    expected = (1000 / 1_000_000 * 3.00) + (500 / 1_000_000 * 15.00)
    assert abs(cost - expected) < 0.0001


def test_calculate_cost_unknown_model():
    """Test cost calculation for unknown models."""
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = calculate_cost("unknown-model", usage)
    assert cost == 0.0


# ============================================================================
# Cost Tracking Tests
# ============================================================================


def test_cost_tracker_initialization():
    """Test CostTracker initialization."""
    tracker = CostTracker()
    assert tracker.total_cost == 0.0
    assert tracker.total_tokens == 0


def test_cost_tracker_add_request():
    """Test adding a request to cost tracker."""
    tracker = CostTracker()
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    tracker.add_request("gpt-4o-mini", usage, 0.001)

    assert tracker.total_cost == 0.001
    assert tracker.total_tokens == 150
    assert tracker.requests_by_model["gpt-4o-mini"] == 1


def test_cost_tracker_summary():
    """Test cost tracker summary."""
    tracker = CostTracker()
    usage1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    usage2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)

    tracker.add_request("gpt-4o-mini", usage1, 0.001)
    tracker.add_request("gpt-4o-mini", usage2, 0.002)

    summary = tracker.get_summary()
    assert summary["total_cost"] == 0.003
    assert summary["total_tokens"] == 450
    assert summary["total_requests"] == 2
    assert "gpt-4o-mini" in summary["by_model"]


def test_cost_tracker_reset():
    """Test resetting cost tracker."""
    tracker = CostTracker()
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    tracker.add_request("gpt-4o-mini", usage, 0.001)

    tracker.reset()
    assert tracker.total_cost == 0.0
    assert tracker.total_tokens == 0


def test_global_cost_tracking():
    """Test global cost tracking functions."""
    reset_cost_tracking()
    summary = get_cost_summary()
    assert summary["total_cost"] == 0.0


# ============================================================================
# Rate Limiter Tests
# ============================================================================


def test_rate_limiter_initialization():
    """Test RateLimiter initialization."""
    limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100000)
    assert limiter.requests_per_minute == 60
    assert limiter.tokens_per_minute == 100000


def test_rate_limiter_wait_under_limit():
    """Test rate limiter when under limit."""
    limiter = RateLimiter(requests_per_minute=60)
    # Should not block when under limit
    limiter.wait_if_needed(1000)
    assert len(limiter.request_times) == 1


def test_rate_limiter_request_tracking():
    """Test rate limiter tracks requests."""
    limiter = RateLimiter(requests_per_minute=60)
    for _ in range(3):
        limiter.wait_if_needed(100)
    assert len(limiter.request_times) == 3


# ============================================================================
# Response Cache Tests
# ============================================================================


def test_response_cache_initialization():
    """Test ResponseCache initialization."""
    cache = ResponseCache(max_size=100, ttl=3600)
    assert cache.max_size == 100
    assert cache.ttl == 3600


def test_response_cache_set_and_get(sample_messages, sample_config, sample_response):
    """Test caching and retrieving responses."""
    cache = ResponseCache()
    cache.set(sample_messages, sample_config, sample_response)

    cached = cache.get(sample_messages, sample_config)
    assert cached is not None
    assert cached.content == sample_response.content
    assert cached.cached is True


def test_response_cache_miss(sample_messages, sample_config):
    """Test cache miss."""
    cache = ResponseCache()
    cached = cache.get(sample_messages, sample_config)
    assert cached is None


def test_response_cache_different_messages(
    sample_messages, sample_config, sample_response
):
    """Test cache differentiation by messages."""
    cache = ResponseCache()
    cache.set(sample_messages, sample_config, sample_response)

    different_messages = [Message(role=MessageRole.USER, content="Different")]
    cached = cache.get(different_messages, sample_config)
    assert cached is None


# ============================================================================
# Utility Function Tests
# ============================================================================


def test_format_messages_simple():
    """Test format_messages with simple inputs."""
    messages = format_messages(user="Hello")
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hello"


def test_format_messages_with_system():
    """Test format_messages with system message."""
    messages = format_messages(system="You are helpful", user="Hello")
    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER


def test_format_messages_with_history():
    """Test format_messages with conversation history."""
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    messages = format_messages(history=history, user="How are you?")
    assert len(messages) == 3
    assert messages[2].content == "How are you?"


def test_count_tokens():
    """Test token counting using tokenizer module."""
    text = "This is a test sentence with multiple words"
    tokens = count_tokens(text, Tokenizer.CL100K_BASE)
    assert tokens > 0
    # Token count should be reasonable (typically less than word count)
    word_count = len(text.split())
    assert tokens <= word_count * 2  # Sanity check


# ============================================================================
# Response Parsing Tests
# ============================================================================


def test_parse_json_response_plain():
    """Test parsing plain JSON response."""
    json_str = '{"name": "Alice", "age": 30}'
    result = parse_json_response(json_str)
    assert result["name"] == "Alice"
    assert result["age"] == 30


def test_parse_json_response_with_markdown():
    """Test parsing JSON with markdown code blocks."""
    json_str = '```json\n{"name": "Bob", "age": 25}\n```'
    result = parse_json_response(json_str)
    assert result["name"] == "Bob"
    assert result["age"] == 25


def test_parse_json_response_from_generation_response():
    """Test parsing JSON from GenerationResponse."""
    response = GenerationResponse(
        content='{"status": "success"}',
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    result = parse_json_response(response)
    assert result["status"] == "success"


def test_parse_json_response_invalid():
    """Test parsing invalid JSON raises error."""
    with pytest.raises(ValueError):
        parse_json_response("not valid json")


def test_validate_response_min_length(sample_response):
    """Test response validation with minimum length."""
    assert validate_response(sample_response, min_length=10)
    assert not validate_response(sample_response, min_length=1000)


def test_validate_response_max_length(sample_response):
    """Test response validation with maximum length."""
    assert validate_response(sample_response, max_length=100)
    assert not validate_response(sample_response, max_length=5)


def test_validate_response_must_contain(sample_response):
    """Test response validation with must_contain."""
    assert validate_response(sample_response, must_contain=["Python"])
    assert not validate_response(sample_response, must_contain=["JavaScript"])


def test_validate_response_must_not_contain(sample_response):
    """Test response validation with must_not_contain."""
    assert validate_response(sample_response, must_not_contain=["JavaScript"])
    assert not validate_response(sample_response, must_not_contain=["Python"])


def test_validate_response_pattern(sample_response):
    """Test response validation with regex pattern."""
    assert validate_response(sample_response, pattern=r"\w+ is a")
    assert not validate_response(sample_response, pattern=r"^\d+$")


# ============================================================================
# Mock Generation Tests (no API calls)
# ============================================================================


def test_generate_mock_simple():
    """Test mock generation with simple string input."""
    # Use LOCAL provider for mock generation
    response = generate("Hello", model="custom-local-model", provider=LLMProvider.LOCAL)
    assert response.content is not None
    assert len(response.content) > 0
    assert response.model == "custom-local-model"


def test_generate_mock_with_messages():
    """Test mock generation with message list."""
    messages = [Message(role=MessageRole.USER, content="What is AI?")]
    response = generate(messages, model="custom-model", provider=LLMProvider.LOCAL)
    assert response.content is not None


def test_generate_mock_with_config():
    """Test mock generation with custom config."""
    config = GenerationConfig(
        model="local-custom-model", temperature=0.5, max_tokens=50
    )
    response = generate(
        "Test", config=config, model="local-custom-model", provider=LLMProvider.LOCAL
    )
    assert response.model == "local-custom-model"


# ============================================================================
# Message Role Enum Tests
# ============================================================================


def test_message_role_enum_values():
    """Test MessageRole enum values."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"


# ============================================================================
# LLM Provider Enum Tests
# ============================================================================


def test_llm_provider_enum_values():
    """Test LLMProvider enum values."""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.ANTHROPIC.value == "anthropic"
    assert LLMProvider.GOOGLE.value == "google"


# ============================================================================
# Model Pricing Tests
# ============================================================================


def test_model_pricing_exists():
    """Test that MODEL_PRICING contains expected models."""
    assert "gpt-4o-mini" in MODEL_PRICING
    assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
    assert "gemini-1.5-pro" in MODEL_PRICING


def test_model_pricing_format():
    """Test that pricing is in correct format (input, output)."""
    for model, pricing in MODEL_PRICING.items():
        assert isinstance(pricing, tuple)
        assert len(pricing) == 2
        assert isinstance(pricing[0], (int, float))
        assert isinstance(pricing[1], (int, float))


# ============================================================================
# Integration Tests (Mock-based)
# ============================================================================


def test_generate_with_cost_tracking():
    """Test generation with automatic cost tracking."""
    reset_cost_tracking()

    response = generate(
        "Test prompt", model="custom-model", provider=LLMProvider.LOCAL, track_cost=True
    )

    summary = get_cost_summary()
    assert summary["total_requests"] >= 1


def test_generate_batch_mock():
    """Test batch generation with mock provider."""
    prompts = ["Hello", "How are you?", "Goodbye"]
    responses = generate_batch(
        prompts, model="custom-model", provider=LLMProvider.LOCAL, max_concurrent=2
    )

    assert len(responses) == len(prompts)
    for response in responses:
        assert response.content is not None


def test_message_conversion_from_dict():
    """Test automatic message conversion from dict."""
    messages_dict = [{"role": "user", "content": "Hello"}]
    response = generate(messages_dict, model="custom-model", provider=LLMProvider.LOCAL)
    assert response.content is not None


# ============================================================================
# Universal Generator Tests
# ============================================================================


def test_universal_generator_with_model_enum():
    """Test universal generator with ModelName enum."""
    from kerb.generation import Generator, ModelName

    gen = Generator(model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI)
    assert gen.model == ModelName.GPT_4O_MINI
    assert gen.provider == LLMProvider.OPENAI


def test_universal_generator_with_provider():
    """Test universal generator with provider parameter."""
    from kerb.generation import Generator, LLMProvider

    # Custom model name with provider
    gen = Generator(model="my-custom-gpt-model", provider=LLMProvider.OPENAI)
    assert gen.model == "my-custom-gpt-model"
    assert gen.provider == LLMProvider.OPENAI


def test_universal_generator_with_string_model():
    """Test universal generator with string model."""
    from kerb.generation import Generator

    gen = Generator(model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    assert gen.model == "gpt-4o-mini"
    assert gen.provider == LLMProvider.OPENAI


def test_generate_with_provider():
    """Test generate function with provider parameter."""
    response = generate(
        "Hello, test", model="custom-model-name", provider=LLMProvider.LOCAL
    )
    assert response is not None
    assert response.provider == LLMProvider.LOCAL


def test_generate_provider_routing():
    """Test that provider parameter controls routing."""
    response = generate(
        "Test", model="gpt-like-custom-model", provider=LLMProvider.LOCAL
    )
    # Should route to specified provider
    assert response.provider == LLMProvider.LOCAL


def test_generator_class_easy_model_switching():
    """Test that Generator class makes model switching easy."""
    from kerb.generation import Generator, ModelName

    # Create generators for different models with same config
    config = {"temperature": 0.7, "max_tokens": 100}

    gen_gpt = Generator(
        model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI, **config
    )
    gen_claude = Generator(
        model=ModelName.CLAUDE_35_HAIKU, provider=LLMProvider.ANTHROPIC, **config
    )

    # Both should have same config but different models
    assert gen_gpt.default_config["temperature"] == 0.7
    assert gen_claude.default_config["temperature"] == 0.7
    assert gen_gpt.model != gen_claude.model


def test_generator_with_custom_model_and_provider():
    """Test Generator with custom model name and provider."""
    from kerb.generation import Generator, LLMProvider

    gen = Generator(
        model="my-company-internal-gpt", provider=LLMProvider.OPENAI, temperature=0.5
    )

    assert gen.model == "my-company-internal-gpt"
    assert gen.provider == LLMProvider.OPENAI
    assert gen.default_config["temperature"] == 0.5


def test_provider_routing():
    """Test that provider routing uses provider parameter."""
    response = generate("Test", model="xyzabc-model-123", provider=LLMProvider.LOCAL)

    # Should route to specified provider
    assert response.provider == LLMProvider.LOCAL


def test_provider_required():
    """Test that provider parameter is required."""
    import pytest

    # Should raise ValueError when provider not specified
    with pytest.raises(ValueError, match="Provider must be specified"):
        generate("Test", model="gpt-4o-mini")
