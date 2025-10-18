"""Audio processing and transcription module.

This module provides audio processing, format conversion, and transcription capabilities.
"""

from .processor import (convert_audio_format, extract_audio_from_video,
                        get_audio_info, transcribe_audio,
                        transcribe_audio_async)

__all__ = [
    "get_audio_info",
    "convert_audio_format",
    "transcribe_audio",
    "transcribe_audio_async",
    "extract_audio_from_video",
]
