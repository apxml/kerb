"""Mistral AI provider implementation for LLM generation.

This module provides Mistral-specific generation functionality.
"""

import os
from typing import List, Optional, Iterator, Callable

from kerb.core.types import Message, MessageRole

from ..config import GenerationConfig, GenerationResponse, Usage, StreamChunk
from ..enums import LLMProvider


class MistralGenerator:
    """Mistral generator with simplified interface.
    
    This is a convenience class for Mistral-specific generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Mistral generator.
        
        Args:
            api_key: Mistral API key (if None, uses MISTRAL_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.config = kwargs
    
    def generate(
        self,
        messages: List[Message],
        model: str = "mistral-small",
        **kwargs
    ) -> GenerationResponse:
        """Generate using Mistral API.
        
        Args:
            messages: Conversation messages
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_mistral(messages, config, self.api_key)
    
    def stream(
        self,
        messages: List[Message],
        model: str = "mistral-small",
        callback: Optional[Callable[[StreamChunk], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Stream from Mistral API.
        
        Args:
            messages: Conversation messages
            model: Model name
            callback: Optional callback for each chunk
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator of StreamChunks
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_stream_mistral(messages, config, self.api_key, callback)


# ============================================================================
# Internal Mistral Functions
# ============================================================================

def _generate_mistral(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None
) -> GenerationResponse:
    """Generate using Mistral API.
    
    Args:
        messages: Conversation messages
        config: Generation configuration
        api_key: Mistral API key
        
    Returns:
        GenerationResponse
    """
    try:
        from mistralai.client import MistralClient
    except ImportError:
        raise ImportError("Mistral AI package not installed. Install with: pip install mistralai")
    
    # Get API key
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key not provided and MISTRAL_API_KEY env var not set")
    
    client = MistralClient(api_key=api_key)
    
    # Convert messages to Mistral format (similar to OpenAI)
    mistral_messages = []
    for msg in messages:
        role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
        mistral_messages.append({"role": role, "content": msg.content})
    
    # Build request parameters
    request_params = {
        "model": config.model,
        "messages": mistral_messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    if config.stop_sequences:
        request_params["stop"] = config.stop_sequences
    
    # Make request
    response = client.chat(
        **request_params
    )
    
    # Parse response
    content = ""
    if response.choices and len(response.choices) > 0:
        content = response.choices[0].message.content or ""
    
    # Extract usage information
    prompt_tokens = 0
    completion_tokens = 0
    if hasattr(response, 'usage'):
        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
    
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )
    
    # Get finish reason
    finish_reason = None
    if response.choices and len(response.choices) > 0:
        finish_reason = response.choices[0].finish_reason
    
    return GenerationResponse(
        content=content,
        model=response.model if hasattr(response, 'model') else config.model,
        provider=LLMProvider.MISTRAL,
        usage=usage,
        finish_reason=finish_reason,
        raw_response=response
    )


def _generate_stream_mistral(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None
) -> Iterator[StreamChunk]:
    """Stream from Mistral API."""
    try:
        from mistralai.client import MistralClient
    except ImportError:
        raise ImportError("Mistral AI package not installed. Install with: pip install mistralai")
    
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key not provided")
    
    client = MistralClient(api_key=api_key)
    
    # Convert messages to Mistral format
    mistral_messages = []
    for msg in messages:
        role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
        mistral_messages.append({"role": role, "content": msg.content})
    
    request_params = {
        "model": config.model,
        "messages": mistral_messages,
        "temperature": config.temperature,
        "stream": True,
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    
    # Stream response
    stream = client.chat_stream(**request_params)
    
    for chunk_data in stream:
        if chunk_data.choices and len(chunk_data.choices) > 0:
            delta = chunk_data.choices[0].delta
            content = delta.content if hasattr(delta, 'content') and delta.content else ""
            finish_reason = chunk_data.choices[0].finish_reason if hasattr(chunk_data.choices[0], 'finish_reason') else None
            
            if content or finish_reason:
                chunk = StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    model=config.model
                )
                if callback:
                    callback(chunk)
                yield chunk
