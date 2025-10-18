"""OpenAI provider implementation for LLM generation.

This module provides OpenAI-specific generation functionality.
"""

import os
from typing import List, Optional, Iterator, Callable
from abc import ABC, abstractmethod

from kerb.core.types import Message

from ..config import GenerationConfig, GenerationResponse, Usage, StreamChunk
from ..enums import LLMProvider


class OpenAIGenerator:
    """OpenAI generator with simplified interface.
    
    This is a convenience class for OpenAI-specific generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = kwargs
    
    def generate(
        self,
        messages: List[Message],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> GenerationResponse:
        """Generate using OpenAI API.
        
        Args:
            messages: Conversation messages
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_openai(messages, config, self.api_key)
    
    def stream(
        self,
        messages: List[Message],
        model: str = "gpt-4o-mini",
        callback: Optional[Callable[[StreamChunk], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Stream from OpenAI API.
        
        Args:
            messages: Conversation messages
            model: Model name
            callback: Optional callback for each chunk
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator of StreamChunks
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_stream_openai(messages, config, self.api_key, callback)


# ============================================================================
# Internal OpenAI Functions
# ============================================================================

def _generate_openai(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None
) -> GenerationResponse:
    """Generate using OpenAI API.
    
    Args:
        messages: Conversation messages
        config: Generation configuration
        api_key: OpenAI API key
        
    Returns:
        GenerationResponse
    """
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    # Get API key
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Build request
    request_params = {
        "model": config.model,
        "messages": [m.to_dict() for m in messages],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
        "n": config.n,
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    if config.stop_sequences:
        request_params["stop"] = config.stop_sequences
    if config.logprobs:
        request_params["logprobs"] = True
        request_params["top_logprobs"] = config.logprobs
    if config.seed is not None:
        request_params["seed"] = config.seed
    if config.response_format:
        request_params["response_format"] = config.response_format
    if config.tools:
        request_params["tools"] = config.tools
    if config.tool_choice:
        request_params["tool_choice"] = config.tool_choice
    
    # Make request
    response = client.chat.completions.create(**request_params)
    
    # Parse response
    choice = response.choices[0]
    content = choice.message.content or ""
    
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )
    
    return GenerationResponse(
        content=content,
        model=response.model,
        provider=LLMProvider.OPENAI,
        usage=usage,
        finish_reason=choice.finish_reason,
        raw_response=response
    )


def _generate_stream_openai(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None
) -> Iterator[StreamChunk]:
    """Stream from OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided")
    
    client = openai.OpenAI(api_key=api_key)
    
    request_params = {
        "model": config.model,
        "messages": [m.to_dict() for m in messages],
        "temperature": config.temperature,
        "stream": True,
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    
    stream = client.chat.completions.create(**request_params)
    
    for chunk_data in stream:
        if chunk_data.choices and len(chunk_data.choices) > 0:
            choice = chunk_data.choices[0]
            content = choice.delta.content or ""
            finish_reason = choice.finish_reason
            
            if content or finish_reason:
                chunk = StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    model=chunk_data.model
                )
                if callback:
                    callback(chunk)
                yield chunk
