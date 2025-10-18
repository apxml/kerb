"""Audio processing and transcription module.

This module provides audio processing, format conversion, and transcription capabilities.
"""

from .processor import (
    get_audio_info,
    convert_audio_format,
    transcribe_audio,
    transcribe_audio_async,
    extract_audio_from_video,
)

__all__ = [
    "get_audio_info",
    "convert_audio_format",
    "transcribe_audio",
    "transcribe_audio_async",
    "extract_audio_from_video",
]
