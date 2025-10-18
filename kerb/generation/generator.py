"""Main generation functions for LLM interaction.

This module provides the core generation functions that orchestrate calls
to different LLM providers.
"""

import asyncio
import os
import time
from typing import Callable, Dict, Iterator, List, Optional, Union

from kerb.core.types import Message, MessageRole

# Import from our reorganized modules
from .config import GenerationConfig, GenerationResponse, StreamChunk, Usage
from .enums import LLMProvider, ModelName
from .providers.anthropic import (_generate_anthropic,
                                  _generate_stream_anthropic)
from .providers.cohere import _generate_cohere, _generate_stream_cohere
from .providers.google import _generate_google, _generate_stream_google
from .providers.mistral import _generate_mistral, _generate_stream_mistral
# Import provider-specific functions
from .providers.openai import _generate_openai, _generate_stream_openai
from .utils import (CostTracker, RateLimiter, ResponseCache,
                    _global_cost_tracker, calculate_cost,
                    retry_with_exponential_backoff)


def generate(
    messages: Union[List[Message], List[Dict[str, str]], str],
    model: Optional[Union[str, ModelName]] = None,
    config: Optional[GenerationConfig] = None,
    api_key: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    use_cache: bool = True,
    cost_tracker: Optional[CostTracker] = None,
    track_cost: bool = False,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 3,
    **kwargs,
) -> GenerationResponse:
    """Universal generator function - generate responses from any LLM provider.

    This is the main generation function that routes to the appropriate provider
    based on the model and provider parameters.

    Args:
        messages: Input messages (can be string, list of dicts, or list of Message objects)
        model: Model to use (ModelName enum or string for custom models).
               If not provided, must be specified in config.
        config: Generation configuration
        api_key: API key (if not provided, uses environment variable)
        provider: LLMProvider enum specifying which API to use
        use_cache: Whether to use response caching
        cost_tracker: Optional cost tracker instance
        track_cost: Whether to track costs in global tracker
        rate_limiter: Optional rate limiter instance
        max_retries: Maximum retry attempts for failed requests
        **kwargs: Additional config parameters

    Returns:
        GenerationResponse: The generated response

    Examples:
        >>> # Using ModelName enum
        >>> response = generate("Hello", model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI)

        >>> # Using custom model name
        >>> response = generate("Hello", model="my-custom-gpt", provider=LLMProvider.OPENAI)

        >>> # Different providers
        >>> response = generate("Hello", model=ModelName.CLAUDE_35_HAIKU, provider=LLMProvider.ANTHROPIC)
    """
    if not messages:
        raise ValueError("Messages cannot be empty")

    # Determine the model to use
    if config is not None and model is None:
        # Use model from config
        model_str = config.model
        model_for_detection = config.model
    elif model is not None:
        # Convert ModelName enum to string for internal use
        model_str = model.value if isinstance(model, ModelName) else model
        model_for_detection = model
    else:
        raise ValueError(
            "Either 'model' parameter or 'config' with a model must be provided"
        )

    # Convert string to messages
    if isinstance(messages, str):
        messages = [Message(role=MessageRole.USER, content=messages)]
    elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
        messages = [
            Message(role=m.get("role", "user"), content=m["content"]) for m in messages
        ]

    # Create or update config
    if config is None:
        config = GenerationConfig(model=model_str)
    elif model is not None:
        # If both config and model are provided, model parameter takes precedence
        config.model = model_str

    # Apply kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Validate provider
    if provider is None:
        raise ValueError(
            "Provider must be specified. Pass the provider parameter.\\n"
            "Example: generate('Hello', model='gpt-4o-mini', provider=LLMProvider.OPENAI)"
        )

    # Validate API key
    if api_key is None:
        if provider == LLMProvider.OPENAI and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found")
        elif provider == LLMProvider.ANTHROPIC and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found")
        elif provider == LLMProvider.GOOGLE and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Google API key not found")
        elif provider == LLMProvider.COHERE and not os.getenv("COHERE_API_KEY"):
            raise ValueError("Cohere API key not found")
        elif provider == LLMProvider.MISTRAL and not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("Mistral API key not found")

    # Check cache
    if use_cache:
        cache = ResponseCache()
        cached_response = cache.get(messages, config)
        if cached_response:
            return cached_response

    # Rate limiting
    if rate_limiter:
        estimated_tokens = sum(len(m.content.split()) * 1.3 for m in messages)
        rate_limiter.wait_if_needed(int(estimated_tokens))

    # Generate
    def _generate():
        start_time = time.time()
        if provider == LLMProvider.OPENAI:
            response = _generate_openai(messages, config, api_key)
        elif provider == LLMProvider.ANTHROPIC:
            response = _generate_anthropic(messages, config, api_key)
        elif provider == LLMProvider.GOOGLE:
            response = _generate_google(messages, config, api_key)
        elif provider == LLMProvider.COHERE:
            response = _generate_cohere(messages, config, api_key)
        elif provider == LLMProvider.MISTRAL:
            response = _generate_mistral(messages, config, api_key)
        else:
            response = _generate_mock(messages, config, provider)
        response.latency = time.time() - start_time
        return response

    response = retry_with_exponential_backoff(_generate, max_retries=max_retries)
    response.cost = calculate_cost(model_for_detection, response.usage)

    # Track cost
    if track_cost or cost_tracker:
        tracker = cost_tracker if cost_tracker else _global_cost_tracker
        tracker.add_request(model_str, response.usage, response.cost)

    # Cache
    if use_cache:
        cache.set(messages, config, response)

    return response


def generate_stream(
    messages: Union[List[Message], List[Dict[str, str]], str],
    model: Optional[Union[str, ModelName]] = None,
    config: Optional[GenerationConfig] = None,
    api_key: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None,
    **kwargs,
) -> Iterator[StreamChunk]:
    """Generate streaming response from any LLM provider.

    Args:
        messages: Input messages (can be string, list of dicts, or list of Message objects)
        model: Model to use (ModelName enum or string for custom models).
               If not provided, must be specified in config.
        config: Generation configuration
        api_key: API key (if not provided, uses environment variable)
        provider: LLMProvider enum specifying which API to use
        callback: Optional callback function for each chunk
        **kwargs: Additional config parameters

    Yields:
        StreamChunk: Chunks of the generated response
    """
    # Determine the model to use
    if config is not None and model is None:
        # Use model from config
        model_str = config.model
        model_for_detection = config.model
    elif model is not None:
        # Convert ModelName enum to string for internal use
        model_str = model.value if isinstance(model, ModelName) else model
        model_for_detection = model
    else:
        raise ValueError(
            "Either 'model' parameter or 'config' with a model must be provided"
        )

    if isinstance(messages, str):
        messages = [Message(role=MessageRole.USER, content=messages)]
    elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
        messages = [
            Message(role=m.get("role", "user"), content=m["content"]) for m in messages
        ]

    if config is None:
        config = GenerationConfig(model=model_str, stream=True)
    else:
        if model is not None:
            config.model = model_str
        config.stream = True

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Validate provider
    if provider is None:
        raise ValueError(
            "Provider must be specified. Pass the provider parameter.\\n"
            "Example: generate_stream('Hello', model='gpt-4o-mini', provider=LLMProvider.OPENAI)"
        )

    if provider == LLMProvider.OPENAI:
        yield from _generate_stream_openai(messages, config, api_key, callback)
    elif provider == LLMProvider.ANTHROPIC:
        yield from _generate_stream_anthropic(messages, config, api_key, callback)
    elif provider == LLMProvider.GOOGLE:
        yield from _generate_stream_google(messages, config, api_key, callback)
    elif provider == LLMProvider.COHERE:
        yield from _generate_stream_cohere(messages, config, api_key, callback)
    elif provider == LLMProvider.MISTRAL:
        yield from _generate_stream_mistral(messages, config, api_key, callback)
    else:
        response = generate(
            messages, model, config, api_key, provider=provider, **kwargs
        )
        chunk = StreamChunk(
            content=response.content,
            finish_reason=response.finish_reason,
            model=model_str,
        )
        if callback:
            callback(chunk)
        yield chunk


def generate_batch(
    prompts: List[Union[str, List[Message]]],
    model: Optional[Union[str, ModelName]] = None,
    config: Optional[GenerationConfig] = None,
    api_key: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    max_concurrent: int = 5,
    show_progress: bool = False,
    **kwargs,
) -> List[GenerationResponse]:
    """Generate batch responses.

    Args:
        prompts: List of prompts to process
        model: Model to use (ModelName enum or string for custom models).
               If not provided, must be specified in config.
        config: Generation configuration
        api_key: API key (if not provided, uses environment variable)
        provider: LLMProvider enum specifying which API to use
        max_concurrent: Maximum concurrent requests
        show_progress: Whether to show progress
        **kwargs: Additional config parameters

    Returns:
        List[GenerationResponse]: List of generated responses
    """
    if model is None and config is None:
        raise ValueError(
            "Either 'model' parameter or 'config' with a model must be provided"
        )

    async def _batch():
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(prompt):
            async with sem:
                return await asyncio.to_thread(
                    generate,
                    prompt,
                    model=model,
                    config=config,
                    api_key=api_key,
                    provider=provider,
                    **kwargs,
                )

        tasks = [_one(p) for p in prompts]

        if show_progress:
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                results.append(await task)
                print(f"Completed {i+1}/{len(prompts)}", end="\r")
            print()
            return results
        return await asyncio.gather(*tasks)

    return asyncio.run(_batch())


async def generate_async(
    messages: Union[List[Message], List[Dict[str, str]], str],
    model: Optional[Union[str, ModelName]] = None,
    config: Optional[GenerationConfig] = None,
    api_key: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    use_cache: bool = True,
    cost_tracker: Optional[CostTracker] = None,
    track_cost: bool = False,
    max_retries: int = 3,
    **kwargs,
) -> GenerationResponse:
    """Async generation.

    Args:
        messages: Input messages (can be string, list of dicts, or list of Message objects)
        model: Model to use (ModelName enum or string for custom models).
               If not provided, must be specified in config.
        config: Generation configuration
        api_key: API key (if not provided, uses environment variable)
        provider: LLMProvider enum specifying which API to use
        use_cache: Whether to use response caching
        cost_tracker: Optional cost tracker instance
        track_cost: Whether to track costs in global tracker
        max_retries: Maximum retry attempts for failed requests
        **kwargs: Additional config parameters

    Returns:
        GenerationResponse: The generated response
    """
    return await asyncio.to_thread(
        generate,
        messages,
        model=model,
        config=config,
        api_key=api_key,
        provider=provider,
        use_cache=use_cache,
        cost_tracker=cost_tracker,
        track_cost=track_cost,
        max_retries=max_retries,
        **kwargs,
    )


def _generate_mock(
    messages: List[Message], config: GenerationConfig, provider: LLMProvider
) -> GenerationResponse:
    """Mock generation.

    Args:
        messages: Input messages
        config: Generation configuration
        provider: The detected provider to use in the response

    Returns:
        GenerationResponse: Mock response with the specified provider
    """
    content = f"Mock response for model {config.model}"
    prompt_tokens = sum(len(m.content.split()) * 1.3 for m in messages)
    completion_tokens = len(content.split()) * 1.3

    usage = Usage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(prompt_tokens + completion_tokens),
    )

    return GenerationResponse(
        content=content,
        model=config.model,
        provider=provider,  # Use the provider passed in (already detected)
        usage=usage,
        finish_reason="stop",
        metadata={"mock": True},
    )


class Generator:
    """Universal LLM generator - easily switch between models and providers.

    This class provides a convenient stateful interface for LLM generation with
    support for both enum-based and string-based model specification. It makes
    it easy to switch between different models and providers without changing
    your code structure.

    Examples:
        >>> # Using ModelName enum
        >>> gen = Generator(model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI)
        >>> response = gen.generate("Hello!")

        >>> # Using custom model name
        >>> gen = Generator(model="my-custom-model", provider=LLMProvider.OPENAI)
        >>> response = gen.generate("Hello!")

        >>> # Easy model switching
        >>> gen_gpt = Generator(model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI, temperature=0.7)
        >>> gen_claude = Generator(model=ModelName.CLAUDE_35_HAIKU, provider=LLMProvider.ANTHROPIC, temperature=0.7)
    """

    def __init__(
        self,
        model: Union[str, ModelName],
        api_key: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        cost_tracker: Optional[CostTracker] = None,
        **default_config,
    ):
        """Initialize the universal Generator.

        Args:
            model: Model to use (ModelName enum or string for custom models)
            api_key: API key (if not provided, uses environment variable)
            provider: LLMProvider enum specifying which API to use
            cost_tracker: Optional cost tracker instance
            **default_config: Default configuration parameters (temperature, max_tokens, etc.)
        """
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.cost_tracker = cost_tracker
        self.default_config = default_config

    def generate(
        self, messages: Union[List[Message], List[Dict[str, str]], str], **kwargs
    ) -> GenerationResponse:
        """Generate a response.

        Args:
            messages: Input messages
            **kwargs: Override default config parameters

        Returns:
            GenerationResponse: The generated response
        """
        config = {**self.default_config, **kwargs}
        return generate(
            messages,
            model=self.model,
            api_key=self.api_key,
            provider=self.provider,
            cost_tracker=self.cost_tracker,
            **config,
        )

    def stream(
        self, messages: Union[List[Message], List[Dict[str, str]], str], **kwargs
    ) -> Iterator[StreamChunk]:
        """Generate a streaming response.

        Args:
            messages: Input messages
            **kwargs: Override default config parameters

        Yields:
            StreamChunk: Chunks of the generated response
        """
        config = {**self.default_config, **kwargs}
        return generate_stream(
            messages,
            model=self.model,
            api_key=self.api_key,
            provider=self.provider,
            **config,
        )

    def batch(
        self, prompts: List[Union[str, List[Message]]], **kwargs
    ) -> List[GenerationResponse]:
        """Generate batch responses.

        Args:
            prompts: List of prompts to process
            **kwargs: Override default config parameters

        Returns:
            List[GenerationResponse]: List of generated responses
        """
        config = {**self.default_config, **kwargs}
        return generate_batch(
            prompts,
            model=self.model,
            api_key=self.api_key,
            provider=self.provider,
            **config,
        )
