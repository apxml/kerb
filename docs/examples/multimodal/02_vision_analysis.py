"""
Vision Model Analysis Example
=============================

This example demonstrates image analysis using vision models (GPT-4V, Claude, Gemini).

Concepts covered:
- Analyzing images with different vision models
- Structured prompts for vision tasks
- Extracting information from images
- Handling API keys and model selection
- Practical use cases for vision models
"""

from kerb.multimodal import (
    analyze_image_with_vision_model,
    VisionModel,
    get_image_info
)
import os


def create_sample_image():
    """Create a test image with text and objects."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image with content
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw shapes
        draw.rectangle([50, 50, 200, 150], fill='red', outline='black')
        draw.ellipse([250, 50, 400, 150], fill='blue', outline='black')
        draw.polygon([(450, 50), (550, 150), (350, 150)], fill='green', outline='black')
        
        # Add text
        draw.text((50, 200), "Sample Image for Vision Analysis", fill='black')
        draw.text((50, 250), "Objects: Rectangle, Circle, Triangle", fill='black')
        draw.text((50, 300), "Colors: Red, Blue, Green", fill='black')
        
        img.save("vision_test.jpg")
        return "vision_test.jpg"
    except ImportError:
        print("PIL/Pillow not available")
        return None


def main():
    """Run vision model analysis examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("VISION MODEL ANALYSIS EXAMPLE")
    print("="*80)
    
    # Create sample image
    image_path = create_sample_image()
    
    if not image_path:
        print("\nCannot create sample image. Please install Pillow:")
        print("  pip install Pillow")
        return
    
    print(f"\nCreated test image: {image_path}")
    
    # Show image info
    info = get_image_info(image_path)
    print(f"Image: {info.width}x{info.height}, {info.format}, {info.size_bytes/1024:.1f}KB")
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    
    print("\n" + "-"*80)
    print("API KEY STATUS")
    print("-"*80)
    print(f"OpenAI API Key: {'Available' if has_openai else 'Not set (export OPENAI_API_KEY=...)'}")
    print(f"Anthropic API Key: {'Available' if has_anthropic else 'Not set (export ANTHROPIC_API_KEY=...)'}")
    print(f"Google API Key: {'Available' if has_google else 'Not set (export GOOGLE_API_KEY=...)'}")
    
    # Example 1: Basic image description
    if has_openai:
        print("\n" + "-"*80)
        print("1. BASIC IMAGE DESCRIPTION (GPT-4V)")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="Describe what you see in this image in detail.",
            model=VisionModel.GPT4O
        )
        
        print(f"Model: {VisionModel.GPT4O.value}")
        print(f"Description: {analysis.description}")
        if analysis.metadata:
            print(f"Metadata: {analysis.metadata}")
    
    # Example 2: Object detection
    if has_openai:
        print("\n" + "-"*80)
        print("2. OBJECT DETECTION")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="List all objects visible in this image. For each object, provide its color and shape.",
            model=VisionModel.GPT4O,
            max_tokens=500
        )
        
        print(f"Objects detected:\n{analysis.description}")
    
    # Example 3: Text extraction (OCR)
    if has_openai:
        print("\n" + "-"*80)
        print("3. TEXT EXTRACTION (OCR)")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="Extract all visible text from this image. Provide the exact text content.",
            model=VisionModel.GPT4O
        )
        
        print(f"Extracted text:\n{analysis.description}")
    
    # Example 4: Structured information extraction
    if has_openai:
        print("\n" + "-"*80)
        print("4. STRUCTURED INFORMATION EXTRACTION")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="""Analyze this image and provide:
1. Number of objects
2. List of colors present
3. List of shapes present
4. Any text content
5. Overall layout description

Format as structured bullet points.""",
            model=VisionModel.GPT4O,
            max_tokens=600
        )
        
        print(f"Structured analysis:\n{analysis.description}")
    
    # Example 5: Using Claude (if available)
    if has_anthropic:
        print("\n" + "-"*80)
        print("5. CLAUDE VISION ANALYSIS")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="Describe this image focusing on the geometric shapes and their arrangement.",
            model=VisionModel.CLAUDE_3_5_SONNET
        )
        
        print(f"Model: {VisionModel.CLAUDE_3_5_SONNET.value}")
        print(f"Analysis: {analysis.description}")
    
    # Example 6: Using Gemini (if available)
    if has_google:
        print("\n" + "-"*80)
        print("6. GEMINI VISION ANALYSIS")
        print("-"*80)
        
        analysis = analyze_image_with_vision_model(
            image_path,
            prompt="What colors and shapes are in this image?",
            model=VisionModel.GEMINI_PRO_VISION
        )
        
        print(f"Model: {VisionModel.GEMINI_PRO_VISION.value}")
        print(f"Analysis: {analysis.description}")
    
    # Use cases for LLM developers
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Document Processing:
   - Extract text from scanned documents, receipts, forms
   - Convert images of documents to structured data
   - Build OCR pipelines for RAG systems

2. Visual Question Answering:
   - Answer questions about image content
   - Build multimodal chatbots
   - Image-based search and retrieval

3. Content Moderation:
   - Detect inappropriate content in images
   - Verify image compliance with guidelines
   - Automated image review systems

4. Product Analysis:
   - Extract product information from images
   - Analyze product features and attributes
   - Build visual search systems

5. Accessibility:
   - Generate alt text for images
   - Create image descriptions for visually impaired users
   - Automated accessibility compliance

6. Data Extraction:
   - Extract structured data from screenshots, charts, diagrams
   - Convert visual data to text/JSON
   - Build multimodal data pipelines

7. Image Understanding for RAG:
   - Generate searchable text from images
   - Create embeddings from visual content
   - Enhance retrieval with visual information
    """)
    
    # Model comparison tips
    print("\n" + "-"*80)
    print("MODEL SELECTION TIPS")
    print("-"*80)
    print("""
GPT-4V (GPT-4O, GPT-4-Turbo):
  - Excellent for detailed analysis and reasoning
  - Good OCR capabilities
  - Supports high-resolution images
  - Fast response times with GPT-4O

Claude 3.5 Sonnet / Claude 3 Opus:
  - Strong visual reasoning
  - Good for complex image understanding
  - Detailed and nuanced descriptions
  - Excellent for document analysis

Gemini Pro Vision (Gemini 1.5):
  - Good general-purpose vision capabilities
  - Fast inference
  - Competitive pricing
  - Good for high-volume processing
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removed: {image_path}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
