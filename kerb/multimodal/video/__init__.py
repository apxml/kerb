"""Video processing and frame extraction module.

This module provides video processing, frame extraction, and thumbnail creation.
"""

from .processor import (
    get_video_info,
    extract_video_frames,
    create_video_thumbnail,
)

__all__ = [
    "get_video_info",
    "extract_video_frames",
    "create_video_thumbnail",
]
