"""Multi-modal prompt construction utilities.

This module provides utilities for building multi-modal prompts for different LLM APIs.
"""

import base64
from typing import Any, Dict, List, Optional

from .utilities import get_mime_type
from .vision.processor import image_to_base64
from .audio.processor import transcribe_audio


def build_multimodal_prompt(
    text: str,
    images: Optional[List[str]] = None,
    audio: Optional[List[str]] = None,
    encode_media: bool = True
) -> List[Dict[str, Any]]:
    """Build a multi-modal prompt for LLM APIs.
    
    Args:
        text: Text prompt
        images: List of image file paths
        audio: List of audio file paths (will be transcribed)
        encode_media: Whether to encode media as base64
        
    Returns:
        List of content parts for multi-modal API calls
        
    Examples:
        >>> prompt = build_multimodal_prompt(
        ...     "What's in these images?",
        ...     images=["photo1.jpg", "photo2.jpg"]
        ... )
        >>> len(prompt)
        3
    """
    content = []
    
    # Add text
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    # Add images
    if images:
        for image_path in images:
            if encode_media:
                image_data = image_to_base64(image_path, include_prefix=True)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
            else:
                content.append({
                    "type": "image",
                    "source": image_path
                })
    
    # Add audio (transcribe first)
    if audio:
        for audio_path in audio:
            transcription = transcribe_audio(audio_path)
            content.append({
                "type": "text",
                "text": f"[Audio transcription]: {transcription.text}"
            })
    
    return content


def build_anthropic_multimodal_content(
    text: str,
    images: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Build Anthropic-specific multi-modal content format.
    
    Args:
        text: Text prompt
        images: List of image file paths
        
    Returns:
        List of content blocks in Anthropic format
        
    Examples:
        >>> content = build_anthropic_multimodal_content(
        ...     "Describe this image",
        ...     images=["photo.jpg"]
        ... )
    """
    content = []
    
    # Add images first (Anthropic recommendation)
    if images:
        for image_path in images:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            mime_type = get_mime_type(image_path)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_data
                }
            })
    
    # Add text
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    return content


def build_google_multimodal_content(
    text: str,
    images: Optional[List[str]] = None
) -> List[Any]:
    """Build Google Gemini-specific multi-modal content format.
    
    Args:
        text: Text prompt
        images: List of image file paths
        
    Returns:
        List of content parts for Gemini API
        
    Examples:
        >>> content = build_google_multimodal_content(
        ...     "What's in this image?",
        ...     images=["photo.jpg"]
        ... )
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required for Google multi-modal. Install with: pip install Pillow")
    
    content = []
    
    # Add text
    if text:
        content.append(text)
    
    # Add images
    if images:
        for image_path in images:
            img = Image.open(image_path)
            content.append(img)
    
    return content
