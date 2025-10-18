"""Multi-modal processing utilities for LLM applications.

This module provides comprehensive multi-modal capabilities for working with
images, audio, video, and vision models in LLM applications.

Usage Examples:
    # Common imports - most frequently used
    from kerb.multimodal import (
        # Enums
        MediaType, ImageFormat, AudioFormat, VideoFormat,
        VisionModel, TranscriptionModel,
        # Data classes
        ImageInfo, AudioInfo, VideoInfo, TranscriptionResult, VisionAnalysis,
        # Common utilities
        detect_media_type, validate_media_file,
    )
    
    # Vision - image processing and analysis
    from kerb.multimodal.vision import (
        load_image, get_image_info, image_to_base64,
        analyze_image_with_vision_model, embed_multimodal,
    )
    
    # Audio - transcription and processing
    from kerb.multimodal.audio import (
        transcribe_audio, get_audio_info, convert_audio_format,
    )
    
    # Video - frame extraction and processing
    from kerb.multimodal.video import (
        extract_video_frames, get_video_info, create_video_thumbnail,
    )
    
    # Prompts - multi-modal prompt construction
    from kerb.multimodal.prompts import (
        build_multimodal_prompt,
        build_anthropic_multimodal_content,
        build_google_multimodal_content,
    )

For image editing (resize, crop, rotate, grid), use PIL/Pillow directly:
    from PIL import Image
    img = Image.open("photo.jpg")
    img = img.resize((800, 600))
    img = img.rotate(90)
    img.save("edited.jpg")
"""

# Top-level imports - types and common utilities
from .types import (
    # Enums
    MediaType,
    ImageFormat,
    AudioFormat,
    VideoFormat,
    VisionModel,
    TranscriptionModel,
    EmbeddingModelMultimodal,
    
    # Data classes
    ImageInfo,
    AudioInfo,
    VideoInfo,
    TranscriptionResult,
    VisionAnalysis,
    MultiModalContent,
)

from .utilities import (
    detect_media_type,
    get_mime_type,
    validate_media_file,
    calculate_file_checksum,
)

# Submodule imports - expose commonly used functions at top level
from .vision import (
    load_image,
    get_image_info,
    convert_image_format,
    image_to_base64,
    base64_to_image,
    extract_dominant_colors,
    calculate_image_hash,
    analyze_image_with_vision_model,
    embed_multimodal,
    compute_multimodal_similarity,
)

from .audio import (
    get_audio_info,
    convert_audio_format,
    transcribe_audio,
    transcribe_audio_async,
    extract_audio_from_video,
)

from .video import (
    get_video_info,
    extract_video_frames,
    create_video_thumbnail,
)

from .prompts import (
    build_multimodal_prompt,
    build_anthropic_multimodal_content,
    build_google_multimodal_content,
)

# Import submodules for direct access
from . import vision, audio, video, prompts

__all__ = [
    # Submodules
    "vision",
    "audio",
    "video",
    "prompts",
    
    # Enums
    "MediaType",
    "ImageFormat",
    "AudioFormat",
    "VideoFormat",
    "VisionModel",
    "TranscriptionModel",
    "EmbeddingModelMultimodal",
    
    # Data classes
    "ImageInfo",
    "AudioInfo",
    "VideoInfo",
    "TranscriptionResult",
    "VisionAnalysis",
    "MultiModalContent",
    
    # Utilities
    "detect_media_type",
    "get_mime_type",
    "validate_media_file",
    "calculate_file_checksum",
    
    # Vision/Image processing
    "load_image",
    "get_image_info",
    "convert_image_format",
    "image_to_base64",
    "base64_to_image",
    "extract_dominant_colors",
    "calculate_image_hash",
    "analyze_image_with_vision_model",
    "embed_multimodal",
    "compute_multimodal_similarity",
    
    # Audio processing
    "get_audio_info",
    "convert_audio_format",
    "transcribe_audio",
    "transcribe_audio_async",
    "extract_audio_from_video",
    
    # Video processing
    "get_video_info",
    "extract_video_frames",
    "create_video_thumbnail",
    
    # Prompt construction
    "build_multimodal_prompt",
    "build_anthropic_multimodal_content",
    "build_google_multimodal_content",
]
