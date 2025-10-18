"""Google (Gemini) provider implementation for LLM generation.

This module provides Google Gemini-specific generation functionality.
"""

import os
from typing import List, Optional, Iterator, Callable

from kerb.core.types import Message, MessageRole

from ..config import GenerationConfig, GenerationResponse, Usage, StreamChunk
from ..enums import LLMProvider


class GoogleGenerator:
    """Google Gemini generator with simplified interface.
    
    This is a convenience class for Google Gemini-specific generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Google Gemini generator.
        
        Args:
            api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.config = kwargs
    
    def generate(
        self,
        messages: List[Message],
        model: str = "gemini-1.5-flash",
        **kwargs
    ) -> GenerationResponse:
        """Generate using Google Gemini API.
        
        Args:
            messages: Conversation messages
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_google(messages, config, self.api_key)
    
    def stream(
        self,
        messages: List[Message],
        model: str = "gemini-1.5-flash",
        callback: Optional[Callable[[StreamChunk], None]] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Stream from Google Gemini API.
        
        Args:
            messages: Conversation messages
            model: Model name
            callback: Optional callback for each chunk
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator of StreamChunks
        """
        config = GenerationConfig(model=model, **kwargs)
        return _generate_stream_google(messages, config, self.api_key, callback)


# ============================================================================
# Internal Google Functions
# ============================================================================

def _generate_google(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None
) -> GenerationResponse:
    """Generate using Google Gemini API.
    
    Args:
        messages: Conversation messages
        config: Generation configuration
        api_key: Google API key
        
    Returns:
        GenerationResponse
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    # Get API key
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not provided and GOOGLE_API_KEY env var not set")
    
    genai.configure(api_key=api_key)
    
    # Convert messages to Gemini format
    # Gemini uses a different format - system instruction separate, then user/model alternating
    system_instruction = None
    conversation_messages = []
    
    for msg in messages:
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            system_instruction = msg.content
        else:
            # Gemini uses "user" and "model" roles (not "assistant")
            role = "model" if msg.role in [MessageRole.ASSISTANT, "assistant"] else "user"
            conversation_messages.append({
                "role": role,
                "parts": [msg.content]
            })
    
    # Create model with configuration
    generation_config = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_output_tokens": config.max_tokens or 2048,
    }
    
    if config.stop_sequences:
        generation_config["stop_sequences"] = config.stop_sequences
    
    model_params = {"model_name": config.model}
    if system_instruction:
        model_params["system_instruction"] = system_instruction
    
    model = genai.GenerativeModel(**model_params)
    
    # Make request
    response = model.generate_content(
        conversation_messages,
        generation_config=generation_config
    )
    
    # Parse response
    content = response.text if response.text else ""
    
    # Extract usage information if available
    prompt_tokens = 0
    completion_tokens = 0
    if hasattr(response, 'usage_metadata'):
        prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
    
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )
    
    # Get finish reason
    finish_reason = None
    if response.candidates and len(response.candidates) > 0:
        finish_reason = str(response.candidates[0].finish_reason.name).lower()
    
    return GenerationResponse(
        content=content,
        model=config.model,
        provider=LLMProvider.GOOGLE,
        usage=usage,
        finish_reason=finish_reason,
        raw_response=response
    )


def _generate_stream_google(
    messages: List[Message],
    config: GenerationConfig,
    api_key: Optional[str] = None,
    callback: Optional[Callable[[StreamChunk], None]] = None
) -> Iterator[StreamChunk]:
    """Stream from Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not provided")
    
    genai.configure(api_key=api_key)
    
    # Convert messages to Gemini format
    system_instruction = None
    conversation_messages = []
    
    for msg in messages:
        if msg.role == MessageRole.SYSTEM or msg.role == "system":
            system_instruction = msg.content
        else:
            role = "model" if msg.role in [MessageRole.ASSISTANT, "assistant"] else "user"
            conversation_messages.append({
                "role": role,
                "parts": [msg.content]
            })
    
    generation_config = {
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens or 2048,
    }
    
    model_params = {"model_name": config.model}
    if system_instruction:
        model_params["system_instruction"] = system_instruction
    
    model = genai.GenerativeModel(**model_params)
    
    # Stream response
    response = model.generate_content(
        conversation_messages,
        generation_config=generation_config,
        stream=True
    )
    
    for chunk_data in response:
        if chunk_data.text:
            chunk = StreamChunk(content=chunk_data.text, model=config.model)
            if callback:
                callback(chunk)
            yield chunk
