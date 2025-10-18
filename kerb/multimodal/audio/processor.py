"""Audio processing and transcription.

This module provides audio processing, format conversion, and transcription capabilities.
"""

import os
from pathlib import Path
from typing import Optional, Union

from ..types import (AudioFormat, AudioInfo, TranscriptionModel,
                     TranscriptionResult)


def get_audio_info(file_path: str) -> AudioInfo:
    """Get detailed information about an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        AudioInfo: Audio information object

    Raises:
        ImportError: If required audio library is not installed

    Examples:
        >>> info = get_audio_info("audio.mp3")
        >>> print(f"Duration: {info.duration_seconds}s")
        Duration: 123.5s
    """
    try:
        import mutagen
    except ImportError:
        raise ImportError(
            "mutagen is required for audio processing. Install with: pip install mutagen"
        )

    audio = mutagen.File(file_path)
    if audio is None:
        raise ValueError(f"Could not load audio file: {file_path}")

    # Extract format
    ext = Path(file_path).suffix.lower().lstrip(".")
    format_map = {
        "mp3": AudioFormat.MP3,
        "wav": AudioFormat.WAV,
        "m4a": AudioFormat.M4A,
        "flac": AudioFormat.FLAC,
        "ogg": AudioFormat.OGG,
        "opus": AudioFormat.OPUS,
        "aac": AudioFormat.AAC,
    }
    audio_format = format_map.get(ext, AudioFormat.MP3)

    # Get audio properties
    duration = audio.info.length if hasattr(audio.info, "length") else 0.0
    sample_rate = audio.info.sample_rate if hasattr(audio.info, "sample_rate") else 0
    channels = audio.info.channels if hasattr(audio.info, "channels") else 0
    bitrate = audio.info.bitrate if hasattr(audio.info, "bitrate") else None

    size_bytes = os.path.getsize(file_path)

    # Extract metadata
    metadata = {}
    if audio.tags:
        for key, value in audio.tags.items():
            metadata[str(key)] = str(value)

    return AudioInfo(
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        format=audio_format,
        size_bytes=size_bytes,
        bitrate=bitrate,
        metadata=metadata,
    )


def convert_audio_format(
    file_path: str,
    target_format: Union[str, AudioFormat],
    output_path: Optional[str] = None,
    bitrate: str = "192k",
) -> str:
    """Convert audio to a different format.

    Args:
        file_path: Path to the input audio
        target_format: Target format (e.g., "mp3", "wav")
        output_path: Output path (auto-generated if None)
        bitrate: Bitrate for lossy formats (e.g., "192k")

    Returns:
        str: Path to the converted audio

    Raises:
        ImportError: If pydub is not installed

    Examples:
        >>> convert_audio_format("audio.wav", "mp3")
        'audio.mp3'
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio conversion. Install with: pip install pydub"
        )

    audio = AudioSegment.from_file(file_path)

    if isinstance(target_format, AudioFormat):
        target_format = target_format.value
    else:
        target_format = target_format.lower()

    if output_path is None:
        base = os.path.splitext(file_path)[0]
        output_path = f"{base}.{target_format}"

    audio.export(output_path, format=target_format, bitrate=bitrate)
    return output_path


def transcribe_audio(
    file_path: str,
    model: Union[str, TranscriptionModel] = TranscriptionModel.OPENAI_WHISPER_1,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    return_timestamps: bool = False,
    max_size_mb: float = 25,
    max_duration_minutes: Optional[float] = None,
) -> TranscriptionResult:
    """Transcribe audio to text using various models.

    Args:
        file_path: Path to the audio file
        model: Transcription model to use
        language: Language code (None for auto-detect)
        api_key: API key for cloud models (OpenAI)
        return_timestamps: Whether to return word-level timestamps
        max_size_mb: Maximum file size in MB. Defaults to 25 (OpenAI limit).
        max_duration_minutes: Maximum audio duration in minutes. None for no limit.

    Returns:
        TranscriptionResult: Transcription result with text and metadata

    Raises:
        ValueError: If file exceeds size or duration limits
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> result = transcribe_audio("audio.mp3")
        >>> print(result.text)
        'Hello, this is a test transcription.'

        >>> # With size and duration limits
        >>> result = transcribe_audio("audio.mp3", max_size_mb=50,
        ...                           max_duration_minutes=10)
    """
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"Audio file size ({file_size_mb:.2f} MB) exceeds maximum "
            f"allowed size ({max_size_mb} MB)"
        )

    # Check duration if specified
    if max_duration_minutes is not None:
        try:
            audio_info = get_audio_info(file_path)
            duration_minutes = audio_info.duration_seconds / 60.0
            if duration_minutes > max_duration_minutes:
                raise ValueError(
                    f"Audio duration ({duration_minutes:.2f} min) exceeds maximum "
                    f"allowed duration ({max_duration_minutes} min)"
                )
        except Exception as e:
            # If we can't get duration, log warning but continue
            print(f"Warning: Could not check audio duration: {e}")

    model_str = model.value if isinstance(model, TranscriptionModel) else model

    # OpenAI Whisper API
    if model_str == "whisper-1" or model_str == "openai_whisper_1":
        return _transcribe_openai_whisper(
            file_path, api_key, language, return_timestamps
        )

    # Local Whisper models
    else:
        return _transcribe_local_whisper(
            file_path, model_str, language, return_timestamps
        )


def _transcribe_openai_whisper(
    file_path: str,
    api_key: Optional[str],
    language: Optional[str],
    return_timestamps: bool,
) -> TranscriptionResult:
    """Transcribe using OpenAI Whisper API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI Whisper. Install with: pip install openai"
        )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required (set OPENAI_API_KEY env var or pass api_key)"
        )

    client = openai.OpenAI(api_key=api_key)

    with open(file_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        if return_timestamps:
            kwargs["timestamp_granularities"] = ["word"]
            kwargs["response_format"] = "verbose_json"

        response = client.audio.transcriptions.create(**kwargs)

    # Parse response
    if isinstance(response, str):
        return TranscriptionResult(text=response)

    result = TranscriptionResult(
        text=response.text,
        language=getattr(response, "language", language),
        duration=getattr(response, "duration", None),
    )

    if hasattr(response, "words"):
        result.word_timestamps = response.words

    if hasattr(response, "segments"):
        result.segments = response.segments

    return result


def _transcribe_local_whisper(
    file_path: str, model: str, language: Optional[str], return_timestamps: bool
) -> TranscriptionResult:
    """Transcribe using local Whisper model."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "whisper is required for local transcription. Install with: pip install openai-whisper"
        )

    # Map model names
    model_map = {
        "whisper-tiny": "tiny",
        "whisper-base": "base",
        "whisper-small": "small",
        "whisper-medium": "medium",
        "whisper-large": "large",
        "whisper-large-v3": "large-v3",
    }
    whisper_model = model_map.get(model, model)

    # Load model
    model_obj = whisper.load_model(whisper_model)

    # Transcribe
    kwargs = {"word_timestamps": return_timestamps}
    if language:
        kwargs["language"] = language

    result = model_obj.transcribe(file_path, **kwargs)

    transcription = TranscriptionResult(
        text=result.get("text", ""),
        language=result.get("language"),
        segments=[dict(s) for s in result.get("segments", [])],
    )

    return transcription


async def transcribe_audio_async(
    file_path: str,
    model: Union[str, TranscriptionModel] = TranscriptionModel.OPENAI_WHISPER_1,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    return_timestamps: bool = False,
    max_size_mb: float = 25,
    max_duration_minutes: Optional[float] = None,
) -> TranscriptionResult:
    """Transcribe audio to text asynchronously using API models.

    Args:
        file_path: Path to the audio file
        model: Transcription model to use
        language: Language code (None for auto-detect)
        api_key: API key for cloud models (OpenAI)
        return_timestamps: Whether to return word-level timestamps
        max_size_mb: Maximum file size in MB. Defaults to 25 (OpenAI limit).
        max_duration_minutes: Maximum audio duration in minutes. None for no limit.

    Returns:
        TranscriptionResult: Transcription result with text and metadata

    Note:
        Currently supports async for OpenAI Whisper API only.
        Local Whisper models will run synchronously in a thread pool.

    Examples:
        >>> import asyncio
        >>> result = asyncio.run(transcribe_audio_async("audio.mp3"))
        >>> print(result.text)
    """
    model_str = model.value if isinstance(model, TranscriptionModel) else model

    # For OpenAI Whisper API, run async
    if model_str == "whisper-1" or model_str == "openai_whisper_1":
        return await _transcribe_openai_whisper_async(
            file_path,
            api_key,
            language,
            return_timestamps,
            max_size_mb,
            max_duration_minutes,
        )

    # For local models, run in thread pool
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        transcribe_audio,
        file_path,
        model,
        language,
        api_key,
        return_timestamps,
        max_size_mb,
        max_duration_minutes,
    )


async def _transcribe_openai_whisper_async(
    file_path: str,
    api_key: Optional[str],
    language: Optional[str],
    return_timestamps: bool,
    max_size_mb: float,
    max_duration_minutes: Optional[float],
) -> TranscriptionResult:
    """Transcribe using OpenAI Whisper API asynchronously."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI Whisper. Install with: pip install openai"
        )

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"Audio file size ({file_size_mb:.2f} MB) exceeds maximum "
            f"allowed size ({max_size_mb} MB)"
        )

    # Check duration if specified
    if max_duration_minutes is not None:
        try:
            audio_info = get_audio_info(file_path)
            duration_minutes = audio_info.duration_seconds / 60.0
            if duration_minutes > max_duration_minutes:
                raise ValueError(
                    f"Audio duration ({duration_minutes:.2f} min) exceeds maximum "
                    f"allowed duration ({max_duration_minutes} min)"
                )
        except Exception as e:
            print(f"Warning: Could not check audio duration: {e}")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required (set OPENAI_API_KEY env var or pass api_key)"
        )

    client = AsyncOpenAI(api_key=api_key)

    with open(file_path, "rb") as audio_file:
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        if return_timestamps:
            kwargs["timestamp_granularities"] = ["word"]
            kwargs["response_format"] = "verbose_json"

        response = await client.audio.transcriptions.create(**kwargs)

    # Parse response
    if isinstance(response, str):
        return TranscriptionResult(text=response)

    result = TranscriptionResult(
        text=response.text,
        language=getattr(response, "language", language),
        duration=getattr(response, "duration", None),
    )

    if hasattr(response, "words"):
        result.word_timestamps = response.words

    if hasattr(response, "segments"):
        result.segments = response.segments

    return result


def extract_audio_from_video(
    video_path: str, output_path: Optional[str] = None, audio_format: str = "mp3"
) -> str:
    """Extract audio track from video file.

    Args:
        video_path: Path to the video file
        output_path: Output path for audio (auto-generated if None)
        audio_format: Output audio format

    Returns:
        str: Path to the extracted audio file

    Raises:
        ImportError: If moviepy is not installed

    Examples:
        >>> audio_path = extract_audio_from_video("video.mp4")
        >>> print(audio_path)
        'video.mp3'
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for video processing. Install with: pip install moviepy"
        )

    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}.{audio_format}"

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, logger=None)
    video.close()

    return output_path
