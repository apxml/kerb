"""Video processing and frame extraction module.

This module provides video processing, frame extraction, and thumbnail creation.
"""

from .processor import (create_video_thumbnail, extract_video_frames,
                        get_video_info)

__all__ = [
    "get_video_info",
    "extract_video_frames",
    "create_video_thumbnail",
]
