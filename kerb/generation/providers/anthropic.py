"""Anthropic (Claude) provider implementation for LLM generation.

This module provides Anthropic-specific generation functionality.
"""

import os
from typing import List, Optional, Iterator, Callable

from kerb.core.types import Message, MessageRole

from ..config import GenerationConfig, GenerationResponse, Usage, StreamChunk
from ..enums import LLMProvider


class AnthropicGenerator:
    """Anthropic generator with simplified interface.
    
    This is a convenience class for Anthropic-specific generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic generator.
        
        Args:
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.config = kwargs
    
    def generate(
        self,
        messages: List[Message],
        model: str = "claude-3-5-haiku-20241022",
        **kwargs
    ) -> GenerationResponse:
        """Generate using Anthropic API.
        
        Args:
            messages: Conversation messages
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_anthropic(messages, config, self.api_key)
    
    def stream(
        self,
        messages: List[Message],
        model: str = "claude-3-5-haiku-20241022",
        callback: Optional[Callable[[StreamChunk], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Stream from Anthropic API.
        
        Args:
            messages: Conversation messages
            model: Model name
            callback: Optional callback for each chunk
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator of StreamChunks
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_stream_anthropic(messages, config, self.api_key, callback)


# ============================================================================
# Internal Anthropic Functions
# ============================================================================

def _generate_anthropic(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None
) -> GenerationResponse:
    """Generate using Anthropic API.
    
    Args:
        messages: Conversation messages
        config: Generation configuration
        api_key: Anthropic API key
        
    Returns:
        GenerationResponse
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    # Get API key
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY env var not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Separate system messages
    system_message = None
    conversation_messages = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            system_message = msg.content
        else:
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            conversation_messages.append({"role": role, "content": msg.content})
    
    # Build request
    request_params = {
        "model": config.model,
        "messages": conversation_messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens or 4096,
    }
    
    if system_message:
        request_params["system"] = system_message
    if config.stop_sequences:
        request_params["stop_sequences"] = config.stop_sequences
    if config.tools:
        request_params["tools"] = config.tools
    if config.tool_choice:
        request_params["tool_choice"] = config.tool_choice
    
    # Make request
    response = client.messages.create(**request_params)
    
    # Parse response
    content = ""
    if response.content:
        for block in response.content:
            if hasattr(block, 'text'):
                content += block.text
    
    usage = Usage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens
    )
    
    return GenerationResponse(
        content=content,
        model=response.model,
        provider=LLMProvider.ANTHROPIC,
        usage=usage,
        finish_reason=response.stop_reason,
        raw_response=response
    )


def _generate_stream_anthropic(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None
) -> Iterator[StreamChunk]:
    """Stream from Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not provided")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Separate system messages
    system_message = None
    conversation_messages = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            system_message = msg.content
        else:
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            conversation_messages.append({"role": role, "content": msg.content})
    
    request_params = {
        "model": config.model,
        "messages": conversation_messages,
        "max_tokens": config.max_tokens or 4096,
        "stream": True,
    }
    
    if system_message:
        request_params["system"] = system_message
    
    with client.messages.stream(**request_params) as stream:
        for text in stream.text_stream:
            chunk = StreamChunk(content=text, model=config.model)
            if callback:
                callback(chunk)
            yield chunk
