"""Audio Transcription Example

This example demonstrates audio transcription and processing.

Concepts covered:
- Audio transcription with OpenAI Whisper
- Audio format conversion
- Getting audio metadata
- Handling different audio formats
- Extracting audio from video
"""

from kerb.multimodal import (
    transcribe_audio,
    get_audio_info,
    convert_audio_format,
    extract_audio_from_video,
    TranscriptionModel,
    AudioFormat
)
import os
from pathlib import Path


def create_sample_audio():
    """Create a simple test audio file using text-to-speech or sine wave.
    
    Note: This creates a simple tone for testing. In production,
    you would work with real audio files (recordings, podcasts, etc.)
    """
    try:
        # Try to create a simple tone using pydub
        from pydub.generators import Sine
        from pydub import AudioSegment
        
        # Generate a 3-second tone at 440Hz (A note)
        tone = Sine(440).to_audio_segment(duration=3000)
        
        # Export as WAV
        tone.export("sample_audio.wav", format="wav")
        return "sample_audio.wav"
    except ImportError:
        print("pydub not available. Using alternative method...")
        
    # Alternative: Create a very simple WAV file header
    # This won't produce actual audio but will be a valid WAV structure
    import wave
    import array
    
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz
    
    # Create sample data
    import math
    num_samples = sample_rate * duration
    samples = []
    for i in range(num_samples):
        sample = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(sample)
    
    # Write WAV file
    with wave.open("sample_audio.wav", 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(array.array('h', samples).tobytes())
    
    return "sample_audio.wav"


def main():
    """Run audio transcription examples."""
    
    print("="*80)
    print("AUDIO TRANSCRIPTION EXAMPLE")
    print("="*80)
    
    # Create sample audio
    audio_path = create_sample_audio()
    
    if not audio_path:
        print("\nCannot create sample audio.")
        return
    
    print(f"\nCreated sample audio: {audio_path}")
    
    # 1. Get audio information
    print("\n" + "-"*80)
    print("1. AUDIO INFORMATION")
    print("-"*80)
    
    try:
        info = get_audio_info(audio_path)
        print(f"Duration: {info.duration:.2f} seconds")
        print(f"Sample rate: {info.sample_rate} Hz")
        print(f"Channels: {info.channels}")
        print(f"Bit rate: {info.bitrate}")
        print(f"Format: {info.format}")
        print(f"File size: {info.size_bytes:,} bytes ({info.size_bytes/1024:.2f} KB)")
    except Exception as e:
        print(f"Could not get audio info (requires mutagen): {e}")
        print("Install with: pip install mutagen")
    
    # 2. Audio format conversion
    print("\n" + "-"*80)
    print("2. AUDIO FORMAT CONVERSION")
    print("-"*80)
    
    try:
        # Convert WAV to MP3
        mp3_path = convert_audio_format(
            audio_path,
            AudioFormat.MP3,
            bitrate="128k"
        )
        print(f"Converted to MP3: {mp3_path}")
        
        mp3_info = get_audio_info(mp3_path)
        print(f"MP3 size: {mp3_info.size_bytes:,} bytes ({mp3_info.size_bytes/1024:.2f} KB)")
        
        # Convert to M4A (AAC)
        m4a_path = convert_audio_format(
            audio_path,
            AudioFormat.M4A,
            bitrate="96k"
        )
        print(f"Converted to M4A: {m4a_path}")
        
    except Exception as e:
        print(f"Format conversion requires pydub and ffmpeg: {e}")
        print("Install with: pip install pydub")
        print("And install ffmpeg: brew install ffmpeg (macOS)")
    
    # 3. Audio transcription with Whisper
    print("\n" + "-"*80)
    print("3. AUDIO TRANSCRIPTION")
    print("-"*80)
    
    if os.getenv("OPENAI_API_KEY"):
        print("\nNote: This example uses OpenAI Whisper API.")
        print("The sample audio is just a tone, so transcription will be empty.")
        print("\nFor real transcription, use actual speech audio files.")
        print("\nExample transcription (commented out to avoid API costs):")
        print("""
        result = transcribe_audio(
            "speech.mp3",
            model=TranscriptionModel.OPENAI_WHISPER_1,
            language="en",
            return_timestamps=True
        )
        
        print(f"Text: {result.text}")
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration}s")
        
        if result.segments:
            print("\\nSegments with timestamps:")
            for seg in result.segments:
                print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
        """)
    else:
        print("\nOpenAI API key not found. Set OPENAI_API_KEY to use transcription.")
        print("\nExample usage:")
        print("""
        from kerb.multimodal import transcribe_audio, TranscriptionModel
        
        result = transcribe_audio(
            "meeting_recording.mp3",
            model=TranscriptionModel.OPENAI_WHISPER_1,
            language="en",
            return_timestamps=True
        )
        
        print(result.text)  # Full transcription
        print(result.segments)  # Timestamped segments
        """)
    
    # 4. Extracting audio from video
    print("\n" + "-"*80)
    print("4. EXTRACT AUDIO FROM VIDEO")
    print("-"*80)
    print("""
Example usage (requires moviepy and ffmpeg):

    from kerb.multimodal import extract_audio_from_video
    
    # Extract audio from video file
    audio_path = extract_audio_from_video(
        "video.mp4",
        output_format="mp3"
    )
    
    # Then transcribe the extracted audio
    result = transcribe_audio(audio_path)
    print(result.text)
    """)
    
    # Use cases for LLM developers
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Meeting Transcription:
   - Transcribe meeting recordings
   - Extract action items and key points
   - Generate meeting summaries with LLMs

2. Podcast Processing:
   - Transcribe podcast episodes
   - Create searchable text for audio content
   - Build RAG systems with audio knowledge

3. Video Content Analysis:
   - Extract audio from videos
   - Transcribe video content
   - Combine with vision models for full multimodal understanding

4. Voice Command Processing:
   - Transcribe voice commands
   - Build voice-enabled LLM applications
   - Create conversational AI systems

5. Accessibility:
   - Generate captions for videos
   - Create transcripts for audio content
   - Build accessible content platforms

6. Content Moderation:
   - Transcribe audio for content review
   - Detect inappropriate content
   - Automated audio moderation

7. Multilingual Support:
   - Transcribe audio in multiple languages
   - Auto-detect language
   - Build multilingual RAG systems

8. Audio Search:
   - Make audio content searchable
   - Build semantic search over audio
   - Create audio knowledge bases
    """)
    
    # Best practices
    print("\n" + "-"*80)
    print("BEST PRACTICES")
    print("-"*80)
    print("""
1. Audio Quality:
   - Use 16kHz or higher sample rate
   - Minimize background noise
   - Use appropriate bitrate (128-192kbps for speech)

2. File Size Management:
   - Check file size before API calls (OpenAI has 25MB limit)
   - Compress audio when possible
   - Split long audio files if needed

3. Format Selection:
   - MP3: Good compression, widely supported
   - M4A/AAC: Better quality at same bitrate
   - WAV: Lossless but large files
   - OGG: Open format, good compression

4. Transcription Accuracy:
   - Specify language when known
   - Use timestamps for segmentation
   - Post-process with LLMs for better formatting

5. Cost Optimization:
   - Convert to lower bitrate for transcription
   - Use appropriate model (Whisper variants)
   - Cache transcription results
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    files_to_remove = [audio_path]
    if os.path.exists("sample_audio.mp3"):
        files_to_remove.append("sample_audio.mp3")
    if os.path.exists("sample_audio.m4a"):
        files_to_remove.append("sample_audio.m4a")
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
