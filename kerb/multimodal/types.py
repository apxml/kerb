"""Multimodal type definitions.

This module contains enums and data classes for multimodal processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# ============================================================================
# Enums
# ============================================================================


class MediaType(Enum):
    """Supported media types."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class ImageFormat(Enum):
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    SVG = "svg"


class AudioFormat(Enum):
    """Supported audio formats."""

    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    OPUS = "opus"
    AAC = "aac"


class VideoFormat(Enum):
    """Supported video formats."""

    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"


class VisionModel(Enum):
    """Supported vision models."""

    GPT4_VISION = "gpt-4-vision-preview"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class TranscriptionModel(Enum):
    """Supported transcription models."""

    WHISPER_TINY = "whisper-tiny"
    WHISPER_BASE = "whisper-base"
    WHISPER_SMALL = "whisper-small"
    WHISPER_MEDIUM = "whisper-medium"
    WHISPER_LARGE = "whisper-large"
    WHISPER_LARGE_V3 = "whisper-large-v3"
    OPENAI_WHISPER_1 = "whisper-1"


class EmbeddingModelMultimodal(Enum):
    """Supported multi-modal embedding models."""

    CLIP_VIT_B_32 = "clip-vit-b-32"
    CLIP_VIT_L_14 = "clip-vit-l-14"
    OPENAI_CLIP = "openai/clip-vit-base-patch32"
    IMAGEBIND = "imagebind"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ImageInfo:
    """Information about an image."""

    width: int
    height: int
    format: ImageFormat
    mode: str  # RGB, RGBA, L, etc.
    size_bytes: int
    aspect_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioInfo:
    """Information about an audio file."""

    duration_seconds: float
    sample_rate: int
    channels: int
    format: AudioFormat
    size_bytes: int
    bitrate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoInfo:
    """Information about a video file."""

    width: int
    height: int
    duration_seconds: float
    fps: float
    frame_count: int
    format: VideoFormat
    size_bytes: int
    codec: Optional[str] = None
    has_audio: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionAnalysis:
    """Result of vision model analysis."""

    description: str
    objects: Optional[List[Dict[str, Any]]] = None
    text_content: Optional[str] = None
    emotions: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalContent:
    """Represents multi-modal content for prompts."""

    type: str  # "text", "image", "audio", "video"
    content: Union[str, bytes, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
