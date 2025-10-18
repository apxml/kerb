"""Multimodal utilities.

This module provides utility functions for media type detection, validation,
and file operations.
"""

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional

from .types import MediaType


def detect_media_type(file_path: str) -> MediaType:
    """Detect media type from file extension.

    Args:
        file_path: Path to the media file

    Returns:
        MediaType: Detected media type

    Examples:
        >>> detect_media_type("photo.jpg")
        MediaType.IMAGE

        >>> detect_media_type("audio.mp3")
        MediaType.AUDIO
    """
    ext = Path(file_path).suffix.lower().lstrip(".")

    image_exts = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif", "svg"}
    audio_exts = {"mp3", "wav", "m4a", "flac", "ogg", "opus", "aac"}
    video_exts = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "mpeg", "mpg"}

    if ext in image_exts:
        return MediaType.IMAGE
    elif ext in audio_exts:
        return MediaType.AUDIO
    elif ext in video_exts:
        return MediaType.VIDEO
    else:
        return MediaType.UNKNOWN


def get_mime_type(file_path: str) -> str:
    """Get MIME type for a file.

    Args:
        file_path: Path to the file

    Returns:
        str: MIME type (e.g., "image/jpeg")

    Examples:
        >>> get_mime_type("photo.jpg")
        'image/jpeg'
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def validate_media_file(
    file_path: str, expected_type: Optional[MediaType] = None
) -> bool:
    """Validate that a media file exists and is of expected type.

    Args:
        file_path: Path to the media file
        expected_type: Expected media type (None to skip type check)

    Returns:
        bool: True if valid, False otherwise

    Examples:
        >>> validate_media_file("photo.jpg", MediaType.IMAGE)
        True
    """
    if not os.path.exists(file_path):
        return False

    if expected_type is None:
        return True

    detected_type = detect_media_type(file_path)
    return detected_type == expected_type


def calculate_file_checksum(file_path: str, algorithm: str = "md5") -> str:
    """Calculate checksum of a media file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ("md5", "sha1", "sha256")

    Returns:
        str: Hexadecimal checksum string

    Examples:
        >>> checksum = calculate_file_checksum("video.mp4")
        >>> len(checksum)
        32
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()
