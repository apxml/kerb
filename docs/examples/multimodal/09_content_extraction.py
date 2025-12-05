"""
Content Extraction Pipeline Example
===================================

This example demonstrates a complete multimodal content extraction pipeline.

Concepts covered:
- Extracting text from images (OCR with vision models)
- Transcribing audio to text
- Processing video (frames + audio)
- Building searchable content from multimodal data
- Creating structured data for LLM applications
"""

from kerb.multimodal import (
    analyze_image_with_vision_model,
    transcribe_audio,
    extract_video_frames,
    get_image_info,
    VisionModel,
    TranscriptionModel
)
import os
import json


def create_sample_document_image():
    """Create a sample document image with text."""
    try:
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((50, 30), "MEETING NOTES - Q4 2024", fill='black')
        
        # Content
        draw.text((50, 80), "Date: October 14, 2024", fill='black')
        draw.text((50, 110), "Attendees: John, Sarah, Mike", fill='black')
        
        draw.text((50, 150), "Key Points:", fill='black')
        draw.text((70, 180), "- Launch new product in Q1 2025", fill='black')
        draw.text((70, 210), "- Increase marketing budget by 20%", fill='black')
        draw.text((70, 240), "- Hire 3 new engineers", fill='black')
        
        draw.text((50, 280), "Action Items:", fill='black')
        draw.text((70, 310), "- Finalize product roadmap", fill='black')
        draw.text((70, 340), "- Review budget proposal", fill='black')
        
        img.save("meeting_notes.jpg")
        return "meeting_notes.jpg"
    except ImportError:
        return None


def extract_text_from_image(image_path: str) -> dict:
    """Extract text and structured information from an image."""

# %%
# Setup and Imports
# -----------------
    
    result = {
        "source": image_path,
        "type": "image",
        "extracted_text": None,
        "structured_data": None
    }
    
    if not os.getenv("OPENAI_API_KEY"):
        result["extracted_text"] = "API key required for text extraction"
        return result
    
    # Extract all text
    ocr_analysis = analyze_image_with_vision_model(
        image_path,
        prompt="Extract all visible text from this image. Preserve the formatting and structure.",
        model=VisionModel.GPT4O
    )
    result["extracted_text"] = ocr_analysis.description
    
    # Extract structured information
    structured_analysis = analyze_image_with_vision_model(
        image_path,
        prompt="""Analyze this image and extract information in JSON format with these fields:
- title: document title
- date: any dates mentioned
- people: names of people mentioned
- key_points: list of main points
- action_items: list of action items or tasks

Return only valid JSON.""",
        model=VisionModel.GPT4O,
        max_tokens=500
    )
    
    try:
        # Try to parse JSON from response
        json_start = structured_analysis.description.find('{')
        json_end = structured_analysis.description.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = structured_analysis.description[json_start:json_end]
            result["structured_data"] = json.loads(json_str)
    except:
        result["structured_data"] = {"raw": structured_analysis.description}
    
    return result



# %%
# Process Audio File
# ------------------

def process_audio_file(audio_path: str) -> dict:
    """Transcribe and process audio content."""
    
    result = {
        "source": audio_path,
        "type": "audio",
        "transcription": None,
        "metadata": {}
    }
    
    if not os.getenv("OPENAI_API_KEY"):
        result["transcription"] = "API key required for transcription"
        return result
    
    # Transcribe audio
    transcription_result = transcribe_audio(
        audio_path,
        model=TranscriptionModel.OPENAI_WHISPER_1,
        return_timestamps=True
    )
    
    result["transcription"] = transcription_result.text
    result["metadata"] = {
        "language": transcription_result.language,
        "duration": transcription_result.duration,
        "segments": transcription_result.segments
    }
    
    return result


def process_video_file(video_path: str) -> dict:
    """Extract and process content from video."""
    
    result = {
        "source": video_path,
        "type": "video",
        "frames": [],
        "frame_descriptions": [],
        "audio_transcription": None
    }
    
    # Extract sample frames
    try:
        frames = extract_video_frames(
            video_path,
            max_frames=5,
            output_dir="video_frames_temp"
        )
        result["frames"] = frames
        
        # Analyze each frame (if vision API available)
        if os.getenv("OPENAI_API_KEY"):
            for frame in frames[:3]:  # Analyze first 3 frames
                analysis = analyze_image_with_vision_model(
                    frame,
                    "Describe what is shown in this video frame.",
                    model=VisionModel.GPT4O
                )
                result["frame_descriptions"].append({
                    "frame": frame,
                    "description": analysis.description
                })
    except Exception as e:
        result["frames"] = [f"Error extracting frames: {e}"]
    
    # Extract and transcribe audio
    # Note: This would use extract_audio_from_video + transcribe_audio
    # Simplified for this example
    result["audio_transcription"] = "Audio transcription would be here"
    
    return result



# %%
# Build Searchable Content
# ------------------------

def build_searchable_content(extraction_results: list) -> dict:
    """Build searchable content from multimodal extractions."""
    
    searchable = {
        "all_text": [],
        "structured_data": [],
        "metadata": {
            "total_sources": len(extraction_results),
            "types": {}
        }
    }
    
    for result in extraction_results:
        # Count types
        doc_type = result.get("type", "unknown")
        searchable["metadata"]["types"][doc_type] = \
            searchable["metadata"]["types"].get(doc_type, 0) + 1
        
        # Collect all text
        if result.get("extracted_text"):
            searchable["all_text"].append({
                "source": result["source"],
                "text": result["extracted_text"]
            })
        
        if result.get("transcription"):
            searchable["all_text"].append({
                "source": result["source"],
                "text": result["transcription"]
            })
        
        # Collect structured data
        if result.get("structured_data"):
            searchable["structured_data"].append({
                "source": result["source"],
                "data": result["structured_data"]
            })
    
    # Combine all text for full-text search
    searchable["combined_text"] = " ".join(
        item["text"] for item in searchable["all_text"]
    )
    
    return searchable


def main():
    """Run content extraction pipeline example."""
    
    print("="*80)
    print("CONTENT EXTRACTION PIPELINE EXAMPLE")
    print("="*80)
    
    # Create sample document
    doc_image = create_sample_document_image()
    
    if not doc_image:
        print("\nCannot create sample document. Install Pillow: pip install Pillow")
        return
    
    print(f"\nCreated sample document: {doc_image}")
    
    # Check API availability
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    print(f"OpenAI API: {'Available' if has_openai else 'Not configured (set OPENAI_API_KEY)'}")
    
    # 1. Extract text from image
    print("\n" + "-"*80)
    print("1. IMAGE TEXT EXTRACTION (OCR)")
    print("-"*80)
    
    if has_openai:
        print(f"\nExtracting text from: {doc_image}")
        image_result = extract_text_from_image(doc_image)
        
        print("\nExtracted Text:")
        print(image_result["extracted_text"])
        
        if image_result["structured_data"]:
            print("\nStructured Data:")
            print(json.dumps(image_result["structured_data"], indent=2))
    else:
        print("\nExample output:")
        print("""
Extracted Text:
MEETING NOTES - Q4 2024
Date: October 14, 2024
Attendees: John, Sarah, Mike

Key Points:
- Launch new product in Q1 2025
- Increase marketing budget by 20%
- Hire 3 new engineers

Action Items:
- Finalize product roadmap
- Review budget proposal

Structured Data:
{
  "title": "Meeting Notes - Q4 2024",
  "date": "October 14, 2024",
  "people": ["John", "Sarah", "Mike"],
  "key_points": [
    "Launch new product in Q1 2025",
    "Increase marketing budget by 20%",
    "Hire 3 new engineers"
  ],
  "action_items": [
    "Finalize product roadmap",
    "Review budget proposal"
  ]
}
        """)
    
    # 2. Audio transcription example
    print("\n" + "-"*80)
    print("2. AUDIO TRANSCRIPTION")
    print("-"*80)
    
    print("\nExample workflow:")
    print("""
    # Transcribe meeting audio
    from kerb.multimodal import transcribe_audio
    
    result = transcribe_audio(
        "meeting_audio.mp3",
        model=TranscriptionModel.OPENAI_WHISPER_1,
        return_timestamps=True
    )
    
    print(f"Transcription: {result.text}")
    
    # Extract structured data with LLM
    prompt = f'''
    From this meeting transcript, extract:
    - Key decisions
    - Action items
    - Participants
    
    Transcript: {result.text}
    '''
    # Send to LLM for structured extraction
    """)
    
    # 3. Video processing example
    print("\n" + "-"*80)
    print("3. VIDEO CONTENT EXTRACTION")
    print("-"*80)
    
    print("\nComplete video processing workflow:")
    print("""
    from kerb.multimodal import (
        extract_video_frames,
        extract_audio_from_video,
        transcribe_audio,
        analyze_image_with_vision_model
    )
    
    # Step 1: Extract frames
    frames = extract_video_frames("presentation.mp4", max_frames=10)
    
    # Step 2: Analyze key frames
    slide_content = []
    for frame in frames:
        analysis = analyze_image_with_vision_model(
            frame,
            "Extract all text and describe visual content"
        )
        slide_content.append(analysis.description)
    
    # Step 3: Extract and transcribe audio
    audio = extract_audio_from_video("presentation.mp4")
    transcription = transcribe_audio(audio)
    
    # Step 4: Combine for complete understanding
    complete_content = {
        "visual_content": slide_content,
        "spoken_content": transcription.text
    }
    """)
    
    # 4. Building searchable content
    print("\n" + "-"*80)
    print("4. BUILDING SEARCHABLE CONTENT")
    print("-"*80)
    
    print("\nExample: Creating a searchable knowledge base")
    print("""
    # Process multiple documents
    documents = [
        "report1.pdf",  # Extract images -> OCR
        "meeting1.mp3",  # Transcribe
        "presentation.mp4"  # Frames + audio
    ]
    
    searchable_content = []
    
    for doc in documents:
        if doc.endswith('.pdf'):
            # Extract images from PDF, OCR each
            images = extract_images_from_pdf(doc)
            for img in images:
                text = extract_text_from_image(img)
                searchable_content.append({
                    "source": doc,
                    "content": text,
                    "type": "document"
                })
        
        elif doc.endswith('.mp3'):
            # Transcribe audio
            result = transcribe_audio(doc)
            searchable_content.append({
                "source": doc,
                "content": result.text,
                "type": "audio"
            })
        
        elif doc.endswith('.mp4'):
            # Extract frames and audio
            frames = extract_video_frames(doc, max_frames=5)
            audio = extract_audio_from_video(doc)
            
            visual = [analyze_image_with_vision_model(f, "Describe") 
                     for f in frames]
            spoken = transcribe_audio(audio).text
            
            searchable_content.append({
                "source": doc,
                "content": {
                    "visual": visual,
                    "audio": spoken
                },
                "type": "video"
            })
    
    # Index in vector database for semantic search
    for item in searchable_content:
        embedding = embed_multimodal(item["content"], "text")
        vector_db.insert(embedding, item)
    """)
    
    # Use cases
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Meeting Intelligence:
   - Transcribe audio/video recordings
   - Extract action items and decisions
   - Build searchable meeting database
   - Generate automated summaries

2. Document Processing:
   - OCR scanned documents
   - Extract structured data from forms
   - Build document search systems
   - Automated data entry

3. Content Management:
   - Process mixed media content
   - Extract searchable text from all formats
   - Build unified knowledge bases
   - Enable cross-format search

4. Education Platforms:
   - Process lecture videos
   - Extract slide content + spoken words
   - Build searchable course materials
   - Generate study guides

5. Legal/Compliance:
   - Process legal documents
   - Transcribe depositions
   - Extract key terms and clauses
   - Build case knowledge bases

6. Media Monitoring:
   - Process news videos
   - Extract quotes and context
   - Build searchable media archives
   - Track topics and entities

7. Customer Support:
   - Process support tickets with images
   - Transcribe call recordings
   - Extract issues and resolutions
   - Build support knowledge base

8. Research & Analysis:
   - Process research papers (text + figures)
   - Extract data from charts/graphs
   - Transcribe interviews
   - Build research databases
    """)
    
    # Best practices
    print("\n" + "-"*80)
    print("BEST PRACTICES")
    print("-"*80)
    print("""
1. Pipeline Design:
   - Process in stages (extract -> analyze -> structure)
   - Cache intermediate results
   - Handle errors gracefully
   - Log processing steps

2. Quality Control:
   - Validate OCR accuracy
   - Check transcription quality
   - Review structured extractions
   - Implement confidence scores

3. Performance:
   - Batch process similar content
   - Parallelize independent operations
   - Use appropriate model sizes
   - Optimize API usage

4. Cost Management:
   - Cache API responses
   - Use cheaper models for initial filtering
   - Process only necessary content
   - Monitor API usage

5. Data Organization:
   - Maintain source references
   - Store metadata with content
   - Version extracted data
   - Enable incremental updates

6. Integration:
   - Store in vector databases for search
   - Index with Elasticsearch for full-text
   - Combine with LLMs for Q&A
   - Build RAG systems
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    if os.path.exists(doc_image):
        os.remove(doc_image)
        print(f"Removed: {doc_image}")
    
    import shutil
    if os.path.exists("video_frames_temp"):
        shutil.rmtree("video_frames_temp")
        print("Removed: video_frames_temp/")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
