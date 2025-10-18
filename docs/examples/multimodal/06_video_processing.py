"""Video Processing Example

This example demonstrates video frame extraction and processing.

Concepts covered:
- Extracting video information
- Extracting frames from video
- Creating video thumbnails
- Sampling frames for analysis
- Combining video processing with LLM workflows
"""

from kerb.multimodal import (
    get_video_info,
    extract_video_frames,
    create_video_thumbnail
)
import os
from pathlib import Path


def create_sample_video():
    """Create a simple test video using images.
    
    Note: This creates a basic video for demonstration.
    In production, you would work with real video files.
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Try to use moviepy if available
        try:
            from moviepy.editor import ImageSequenceClip
            
            # Create a sequence of images
            frames = []
            for i in range(30):  # 30 frames (1 second at 30fps)
                img = Image.new('RGB', (400, 300), color='white')
                draw = ImageDraw.Draw(img)
                
                # Draw a moving circle
                x = 50 + (i * 10)
                y = 150
                draw.ellipse([x-20, y-20, x+20, y+20], fill='blue')
                
                # Add frame number
                draw.text((10, 10), f"Frame {i+1}", fill='black')
                
                frames.append(np.array(img))
            
            # Create video clip
            clip = ImageSequenceClip(frames, fps=30)
            clip.write_videofile("sample_video.mp4", verbose=False, logger=None)
            
            return "sample_video.mp4"
            
        except ImportError:
            print("moviepy not available. Skipping video creation.")
            return None
            
    except ImportError:
        print("PIL not available. Skipping video creation.")
        return None


def main():
    """Run video processing examples."""
    
    print("="*80)
    print("VIDEO PROCESSING EXAMPLE")
    print("="*80)
    
    print("\nNote: Video processing requires moviepy and ffmpeg:")
    print("  pip install moviepy")
    print("  brew install ffmpeg  # macOS")
    
    # Create sample video
    video_path = create_sample_video()
    
    if not video_path:
        print("\nShowing example usage only (video creation requires dependencies)...")
        show_examples_only()
        return
    
    print(f"\nCreated sample video: {video_path}")
    
    # 1. Get video information
    print("\n" + "-"*80)
    print("1. VIDEO INFORMATION")
    print("-"*80)
    
    try:
        info = get_video_info(video_path)
        print(f"Duration: {info.duration:.2f} seconds")
        print(f"Width: {info.width}px")
        print(f"Height: {info.height}px")
        print(f"FPS: {info.fps}")
        print(f"Total frames: {info.frame_count}")
        print(f"Format: {info.format}")
        print(f"File size: {info.size_bytes:,} bytes ({info.size_bytes/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"Error getting video info: {e}")
    
    # 2. Extract video frames
    print("\n" + "-"*80)
    print("2. EXTRACT VIDEO FRAMES")
    print("-"*80)
    
    try:
        # Extract frames at 1 second intervals
        frames = extract_video_frames(
            video_path,
            interval=1.0,  # Extract every 1 second
            output_dir="video_frames"
        )
        
        print(f"Extracted {len(frames)} frames to: video_frames/")
        for frame_path in frames:
            print(f"  - {frame_path}")
            
    except Exception as e:
        print(f"Error extracting frames: {e}")
    
    # 3. Create video thumbnail
    print("\n" + "-"*80)
    print("3. CREATE VIDEO THUMBNAIL")
    print("-"*80)
    
    try:
        # Create thumbnail at 0.5 seconds
        thumbnail = create_video_thumbnail(
            video_path,
            timestamp=0.5,
            output_path="video_thumbnail.jpg",
            size=(200, 150)
        )
        
        print(f"Created thumbnail: {thumbnail}")
        print(f"  Size: 200x150px")
        print(f"  Timestamp: 0.5s")
        
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
    
    # 4. Sample frames for analysis
    print("\n" + "-"*80)
    print("4. SAMPLE FRAMES FOR ANALYSIS")
    print("-"*80)
    
    try:
        # Extract specific number of frames evenly distributed
        sample_frames = extract_video_frames(
            video_path,
            max_frames=5,  # Get 5 frames total
            output_dir="video_samples"
        )
        
        print(f"Sampled {len(sample_frames)} frames evenly from video:")
        for frame_path in sample_frames:
            print(f"  - {frame_path}")
            
    except Exception as e:
        print(f"Error sampling frames: {e}")
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Removed: {video_path}")
    
    # Clean up frame directories
    import shutil
    for dir_name in ["video_frames", "video_samples"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed directory: {dir_name}/")
    
    if os.path.exists("video_thumbnail.jpg"):
        os.remove("video_thumbnail.jpg")
        print("Removed: video_thumbnail.jpg")


def show_examples_only():
    """Show example code when dependencies are not available."""
    
    print("\n" + "-"*80)
    print("EXAMPLE 1: GET VIDEO INFORMATION")
    print("-"*80)
    print("""
    from kerb.multimodal import get_video_info
    
    info = get_video_info("meeting_recording.mp4")
    
    print(f"Duration: {info.duration} seconds")
    print(f"Resolution: {info.width}x{info.height}")
    print(f"FPS: {info.fps}")
    print(f"Total frames: {info.frame_count}")
    """)
    
    print("\n" + "-"*80)
    print("EXAMPLE 2: EXTRACT FRAMES AT INTERVALS")
    print("-"*80)
    print("""
    from kerb.multimodal import extract_video_frames
    
    # Extract one frame every 5 seconds
    frames = extract_video_frames(
        "video.mp4",
        interval=5.0,
        output_dir="frames"
    )
    
    print(f"Extracted {len(frames)} frames")
    """)
    
    print("\n" + "-"*80)
    print("EXAMPLE 3: SAMPLE FRAMES FOR ANALYSIS")
    print("-"*80)
    print("""
    # Extract exactly 10 frames evenly distributed
    frames = extract_video_frames(
        "long_video.mp4",
        max_frames=10,
        output_dir="samples"
    )
    
    # Analyze each frame with vision model
    from kerb.multimodal import analyze_image_with_vision_model
    
    for frame in frames:
        analysis = analyze_image_with_vision_model(
            frame,
            prompt="What is happening in this video frame?"
        )
        print(analysis.description)
    """)
    
    print("\n" + "-"*80)
    print("EXAMPLE 4: CREATE VIDEO THUMBNAIL")
    print("-"*80)
    print("""
    from kerb.multimodal import create_video_thumbnail
    
    # Create thumbnail from middle of video
    thumbnail = create_video_thumbnail(
        "video.mp4",
        timestamp=None,  # None = middle of video
        size=(320, 240),
        output_path="thumb.jpg"
    )
    """)
    
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Video Summarization:
   - Extract key frames from videos
   - Analyze frames with vision models
   - Generate video summaries with LLMs
   
   Example workflow:
   frames = extract_video_frames("video.mp4", max_frames=10)
   
   descriptions = []
   for frame in frames:
       analysis = analyze_image_with_vision_model(frame, "Describe this scene")
       descriptions.append(analysis.description)
   
   # Combine with LLM for summary
   summary_prompt = f"Summarize this video based on these scenes: {descriptions}"

2. Video Search & Indexing:
   - Extract frames for embedding generation
   - Build searchable video databases
   - Implement semantic video search
   
   Example:
   frames = extract_video_frames("video.mp4", interval=2.0)
   
   for frame in frames:
       embedding = embed_multimodal(frame, "image")
       # Store embedding with timestamp in vector DB

3. Content Moderation:
   - Sample frames for review
   - Detect inappropriate content
   - Automated video filtering

4. Video Q&A:
   - Extract relevant frames
   - Answer questions about video content
   - Build video understanding systems

5. Lecture/Tutorial Processing:
   - Extract slides from lecture videos
   - Transcribe audio (extract_audio_from_video)
   - Build educational knowledge bases

6. Product Demo Analysis:
   - Extract product screenshots from demos
   - Analyze UI/UX elements
   - Build product documentation

7. Surveillance & Monitoring:
   - Sample frames for analysis
   - Detect events or anomalies
   - Automated alerting systems

8. Video Metadata Extraction:
   - Extract scene information
   - Generate tags and descriptions
   - Build video catalogs
    """)
    
    print("\n" + "-"*80)
    print("BEST PRACTICES")
    print("-"*80)
    print("""
1. Frame Sampling Strategy:
   - Use interval for regular sampling
   - Use max_frames for even distribution
   - Consider video content when choosing strategy

2. Performance Optimization:
   - Don't extract all frames from long videos
   - Use appropriate intervals (1-5 seconds typical)
   - Sample fewer frames for quick analysis

3. Storage Management:
   - Clean up extracted frames after processing
   - Use temporary directories
   - Compress frames if storing long-term

4. Video Quality:
   - Check video resolution and FPS
   - Validate video format compatibility
   - Handle corrupted videos gracefully

5. Integration with Vision Models:
   - Extract frames -> Analyze with vision model -> Summarize with LLM
   - Batch process frames for efficiency
   - Cache analysis results

6. Audio + Video:
   - Use extract_audio_from_video for complete analysis
   - Combine transcription with frame analysis
   - Build multimodal understanding
    """)
    
    print("\n" + "-"*80)
    print("ADVANCED PATTERNS")
    print("-"*80)
    print("""
1. Intelligent Frame Selection:
   - Extract frames only when scene changes
   - Use motion detection algorithms
   - Focus on frames with high information content

2. Multi-resolution Analysis:
   - Extract different frame sizes for different tasks
   - Thumbnails for overview, full-res for details
   - Progressive analysis strategies

3. Temporal Understanding:
   - Maintain frame sequence information
   - Analyze frame transitions
   - Build temporal knowledge graphs

4. Hybrid Processing:
   - Combine frame analysis + audio transcription
   - Cross-reference visual and audio content
   - Build comprehensive video understanding
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
