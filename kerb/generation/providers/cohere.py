"""Cohere provider implementation for LLM generation.

This module provides Cohere-specific generation functionality.
"""

import os
from typing import List, Optional, Iterator, Callable

from kerb.core.types import Message, MessageRole

from ..config import GenerationConfig, GenerationResponse, Usage, StreamChunk
from ..enums import LLMProvider


class CohereGenerator:
    """Cohere generator with simplified interface.
    
    This is a convenience class for Cohere-specific generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Cohere generator.
        
        Args:
            api_key: Cohere API key (if None, uses COHERE_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.config = kwargs
    
    def generate(
        self,
        messages: List[Message],
        model: str = "command-r",
        **kwargs
    ) -> GenerationResponse:
        """Generate using Cohere API.
        
        Args:
            messages: Conversation messages
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_cohere(messages, config, self.api_key)
    
    def stream(
        self,
        messages: List[Message],
        model: str = "command-r",
        callback: Optional[Callable[[StreamChunk], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Stream from Cohere API.
        
        Args:
            messages: Conversation messages
            model: Model name
            callback: Optional callback for each chunk
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator of StreamChunks
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_stream_cohere(messages, config, self.api_key, callback)


# ============================================================================
# Internal Cohere Functions
# ============================================================================

def _generate_cohere(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None
) -> GenerationResponse:
    """Generate using Cohere API.
    
    Args:
        messages: Conversation messages
        config: Generation configuration
        api_key: Cohere API key
        
    Returns:
        GenerationResponse
    """
    try:
        import cohere
    except ImportError:
        raise ImportError("Cohere package not installed. Install with: pip install cohere")
    
    # Get API key
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Cohere API key not provided and COHERE_API_KEY env var not set")
    
    client = cohere.Client(api_key=api_key)
    
    # Convert messages to Cohere format
    # Cohere uses chat history with system message (preamble) separate
    preamble = None
    chat_history = []
    
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            preamble = msg.content
        elif msg.role in [MessageRole.USER, "user"]:
            # If this is the last message, it goes as the message parameter, not history
            if i < len(messages) - 1:
                chat_history.append({"role": "USER", "message": msg.content})
        elif msg.role in [MessageRole.ASSISTANT, "assistant"]:
            chat_history.append({"role": "CHATBOT", "message": msg.content})
    
    # The last user message should be the current message
    current_message = messages[-1].content if messages and messages[-1].role in [MessageRole.USER, "user"] else ""
    
    # Build request parameters
    request_params = {
        "model": config.model,
        "message": current_message,
        "temperature": config.temperature,
        "p": config.top_p,
    }
    
    if preamble:
        request_params["preamble"] = preamble
    if chat_history:
        request_params["chat_history"] = chat_history
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    if config.stop_sequences:
        request_params["stop_sequences"] = config.stop_sequences
    
    # Make request
    response = client.chat(**request_params)
    
    # Parse response
    content = response.text if response.text else ""
    
    # Extract usage information
    prompt_tokens = 0
    completion_tokens = 0
    if hasattr(response, 'meta') and response.meta:
        if hasattr(response.meta, 'billed_units'):
            prompt_tokens = getattr(response.meta.billed_units, 'input_tokens', 0)
            completion_tokens = getattr(response.meta.billed_units, 'output_tokens', 0)
    
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )
    
    # Get finish reason
    finish_reason = getattr(response, 'finish_reason', None)
    
    return GenerationResponse(
        content=content,
        model=config.model,
        provider=LLMProvider.COHERE,
        usage=usage,
        finish_reason=finish_reason,
        raw_response=response
    )


def _generate_stream_cohere(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None
) -> Iterator[StreamChunk]:
    """Stream from Cohere API."""
    try:
        import cohere
    except ImportError:
        raise ImportError("Cohere package not installed. Install with: pip install cohere")
    
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Cohere API key not provided")
    
    client = cohere.Client(api_key=api_key)
    
    # Convert messages to Cohere format
    preamble = None
    chat_history = []
    
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            preamble = msg.content
        elif msg.role in [MessageRole.USER, "user"]:
            if i < len(messages) - 1:
                chat_history.append({"role": "USER", "message": msg.content})
        elif msg.role in [MessageRole.ASSISTANT, "assistant"]:
            chat_history.append({"role": "CHATBOT", "message": msg.content})
    
    current_message = messages[-1].content if messages and messages[-1].role in [MessageRole.USER, "user"] else ""
    
    request_params = {
        "model": config.model,
        "message": current_message,
        "temperature": config.temperature,
        "stream": True,
    }
    
    if preamble:
        request_params["preamble"] = preamble
    if chat_history:
        request_params["chat_history"] = chat_history
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    
    # Stream response
    stream = client.chat(**request_params)
    
    for event in stream:
        if event.event_type == "text-generation":
            chunk = StreamChunk(content=event.text, model=config.model)
            if callback:
                callback(chunk)
            yield chunk
        elif event.event_type == "stream-end":
            # Final chunk with finish reason
            if hasattr(event, 'finish_reason'):
                chunk = StreamChunk(
                    content="",
                    finish_reason=event.finish_reason,
                    model=config.model
                )
                if callback:
                    callback(chunk)
                yield chunk
