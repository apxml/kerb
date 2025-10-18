"""Video processing and frame extraction.

This module provides video processing, frame extraction, and thumbnail creation.
"""

import os
from pathlib import Path
from typing import List, Optional

from ..types import VideoFormat, VideoInfo


def get_video_info(file_path: str) -> VideoInfo:
    """Get detailed information about a video file.

    Args:
        file_path: Path to the video file

    Returns:
        VideoInfo: Video information object

    Raises:
        ImportError: If moviepy is not installed

    Examples:
        >>> info = get_video_info("video.mp4")
        >>> print(f"{info.width}x{info.height} @ {info.fps} FPS")
        1920x1080 @ 30.0 FPS
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. Install with: pip install moviepy"
        )

    video = VideoFileClip(file_path)

    ext = Path(file_path).suffix.lower().lstrip(".")
    format_map = {
        "mp4": VideoFormat.MP4,
        "avi": VideoFormat.AVI,
        "mov": VideoFormat.MOV,
        "mkv": VideoFormat.MKV,
        "webm": VideoFormat.WEBM,
        "flv": VideoFormat.FLV,
    }
    video_format = format_map.get(ext, VideoFormat.MP4)

    info = VideoInfo(
        width=video.w,
        height=video.h,
        duration_seconds=video.duration,
        fps=video.fps,
        frame_count=int(video.fps * video.duration),
        format=video_format,
        size_bytes=os.path.getsize(file_path),
        has_audio=video.audio is not None,
    )

    video.close()
    return info


def extract_video_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[str]:
    """Extract frames from a video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (None for all frames)
        max_frames: Maximum number of frames to extract
        start_time: Start time in seconds
        end_time: End time in seconds (None for end of video)

    Returns:
        List of paths to extracted frame images

    Examples:
        >>> frames = extract_video_frames("video.mp4", "frames/", fps=1)
        >>> len(frames)
        30
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. Install with: pip install moviepy"
        )

    os.makedirs(output_dir, exist_ok=True)

    video = VideoFileClip(video_path)

    if end_time is None:
        end_time = video.duration

    # Calculate frame extraction times
    if fps is not None:
        interval = 1.0 / fps
        times = []
        t = start_time
        while t < end_time:
            times.append(t)
            t += interval
            if max_frames and len(times) >= max_frames:
                break
    else:
        # Extract all frames
        video_fps = video.fps
        interval = 1.0 / video_fps
        times = []
        t = start_time
        while t < end_time:
            times.append(t)
            t += interval
            if max_frames and len(times) >= max_frames:
                break

    frame_paths = []
    for i, t in enumerate(times):
        frame = video.get_frame(t)
        frame_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")

        from PIL import Image

        img = Image.fromarray(frame)
        img.save(frame_path, quality=95)
        frame_paths.append(frame_path)

    video.close()
    return frame_paths


def create_video_thumbnail(
    video_path: str, output_path: Optional[str] = None, time: float = 1.0
) -> str:
    """Create a thumbnail image from a video.

    Args:
        video_path: Path to the video file
        output_path: Output path for thumbnail (auto-generated if None)
        time: Time in seconds to extract frame

    Returns:
        str: Path to the thumbnail image

    Examples:
        >>> thumb = create_video_thumbnail("video.mp4")
        >>> print(thumb)
        'video_thumb.jpg'
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. Install with: pip install moviepy"
        )

    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_thumb.jpg"

    video = VideoFileClip(video_path)

    # Ensure time is within video duration
    time = min(time, video.duration - 0.1)

    frame = video.get_frame(time)

    from PIL import Image

    img = Image.fromarray(frame)
    img.save(output_path, quality=95)

    video.close()
    return output_path
