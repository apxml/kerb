"""Vision processing module for images and vision models.

This module provides image processing, vision model analysis, and multimodal embeddings.
"""

from .processor import (
    # Image processing
    load_image,
    get_image_info,
    convert_image_format,
    image_to_base64,
    base64_to_image,
    extract_dominant_colors,
    calculate_image_hash,
    
    # Vision model integration
    analyze_image_with_vision_model,
    
    # Multi-modal embeddings
    embed_multimodal,
    compute_multimodal_similarity,
)

__all__ = [
    # Image processing
    "load_image",
    "get_image_info",
    "convert_image_format",
    "image_to_base64",
    "base64_to_image",
    "extract_dominant_colors",
    "calculate_image_hash",
    
    # Vision model integration
    "analyze_image_with_vision_model",
    
    # Multi-modal embeddings
    "embed_multimodal",
    "compute_multimodal_similarity",
]
