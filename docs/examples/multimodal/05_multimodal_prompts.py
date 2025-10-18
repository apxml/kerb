"""Multimodal Prompts Example

This example demonstrates building multimodal prompts for different LLM providers.

Concepts covered:
- Building prompts with images for OpenAI
- Creating Anthropic-specific multimodal content
- Formatting Google Gemini multimodal requests
- Combining multiple images with text
- Best practices for multimodal prompting
"""

from kerb.multimodal import (
    build_multimodal_prompt,
    build_anthropic_multimodal_content,
    build_google_multimodal_content,
    image_to_base64
)
import os
import json


def create_sample_images():
    """Create sample images for demonstration."""
    try:
        from PIL import Image, ImageDraw
        
        # Image 1: Product diagram
        img1 = Image.new('RGB', (400, 300), color='white')
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([50, 50, 350, 250], outline='black', width=2)
        draw1.text((100, 120), "PRODUCT DIAGRAM", fill='black')
        draw1.text((100, 160), "Feature A: Advanced", fill='black')
        draw1.text((100, 190), "Feature B: Premium", fill='black')
        img1.save("product.jpg")
        
        # Image 2: Chart
        img2 = Image.new('RGB', (400, 300), color='white')
        draw2 = ImageDraw.Draw(img2)
        # Simple bar chart
        draw2.rectangle([50, 200, 100, 250], fill='blue')
        draw2.rectangle([120, 150, 170, 250], fill='green')
        draw2.rectangle([190, 100, 240, 250], fill='red')
        draw2.text((80, 270), "Q1  Q2  Q3", fill='black')
        img2.save("chart.jpg")
        
        return ["product.jpg", "chart.jpg"]
    except ImportError:
        return []


def main():
    """Run multimodal prompts examples."""
    
    print("="*80)
    print("MULTIMODAL PROMPTS EXAMPLE")
    print("="*80)
    
    # Create sample images
    images = create_sample_images()
    
    if not images:
        print("\nCannot create images. Install Pillow: pip install Pillow")
        return
    
    print(f"\nCreated {len(images)} sample images: {', '.join(images)}")
    
    # 1. Generic multimodal prompt
    print("\n" + "-"*80)
    print("1. GENERIC MULTIMODAL PROMPT")
    print("-"*80)
    
    prompt = build_multimodal_prompt(
        text="What is shown in these images?",
        images=images
    )
    
    print("\nGeneric format (works with multiple providers):")
    print(f"Number of content blocks: {len(prompt)}")
    for i, block in enumerate(prompt):
        if block['type'] == 'text':
            print(f"  Block {i}: Text - '{block['text'][:50]}...'")
        elif block['type'] == 'image':
            print(f"  Block {i}: Image - {len(block['data'])} chars of base64")
    
    # 2. OpenAI-specific format
    print("\n" + "-"*80)
    print("2. OPENAI GPT-4V FORMAT")
    print("-"*80)
    
    print("\nExample for OpenAI API:")
    print("""
    from openai import OpenAI
    from kerb.multimodal import build_multimodal_prompt
    
    client = OpenAI()
    
    # Build multimodal content
    content = build_multimodal_prompt(
        text="Analyze these product images and extract key features.",
        images=["product1.jpg", "product2.jpg"]
    )
    
    # Use with OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=500
    )
    
    print(response.choices[0].message.content)
    """)
    
    # 3. Anthropic-specific format
    print("\n" + "-"*80)
    print("3. ANTHROPIC CLAUDE FORMAT")
    print("-"*80)
    
    anthropic_content = build_anthropic_multimodal_content(
        text="Describe what you see in these images.",
        images=images
    )
    
    print(f"\nAnthropic-specific format ({len(anthropic_content)} blocks):")
    for i, block in enumerate(anthropic_content):
        if block['type'] == 'text':
            print(f"  Block {i}: Text - '{block['text'][:50]}...'")
        elif block['type'] == 'image':
            source_type = block['source']['type']
            media_type = block['source']['media_type']
            print(f"  Block {i}: Image - {source_type}, {media_type}")
    
    print("\nExample for Anthropic API:")
    print("""
    from anthropic import Anthropic
    from kerb.multimodal import build_anthropic_multimodal_content
    
    client = Anthropic()
    
    # Build Anthropic-specific content
    content = build_anthropic_multimodal_content(
        text="Compare these two charts and identify trends.",
        images=["chart1.jpg", "chart2.jpg"]
    )
    
    # Use with Claude API
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    
    print(response.content[0].text)
    """)
    
    # 4. Google Gemini format
    print("\n" + "-"*80)
    print("4. GOOGLE GEMINI FORMAT")
    print("-"*80)
    
    google_content = build_google_multimodal_content(
        text="What insights can you extract from these images?",
        images=images
    )
    
    print(f"\nGoogle Gemini format ({len(google_content)} parts):")
    for i, part in enumerate(google_content):
        if isinstance(part, str):
            print(f"  Part {i}: Text - '{part[:50]}...'")
        else:
            print(f"  Part {i}: Image data")
    
    print("\nExample for Google Gemini API:")
    print("""
    import google.generativeai as genai
    from kerb.multimodal import build_google_multimodal_content
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Build Gemini-specific content
    content = build_google_multimodal_content(
        text="Analyze these diagrams and summarize the information.",
        images=["diagram1.jpg", "diagram2.jpg"]
    )
    
    # Generate response
    response = model.generate_content(content)
    print(response.text)
    """)
    
    # 5. Multiple images with detailed prompts
    print("\n" + "-"*80)
    print("5. COMPLEX MULTIMODAL PROMPTS")
    print("-"*80)
    
    complex_prompt = """Analyze these images and provide:

1. A brief description of each image
2. Common themes or patterns
3. Key differences between the images
4. Actionable insights based on the visual data

Format your response as structured JSON with keys:
- descriptions: list of image descriptions
- themes: list of common themes
- differences: list of key differences
- insights: list of actionable insights"""
    
    content = build_multimodal_prompt(
        text=complex_prompt,
        images=images
    )
    
    print("\nComplex prompt structure:")
    print(f"  Text: {len(complex_prompt)} chars")
    print(f"  Images: {len(images)}")
    print(f"  Total blocks: {len(content)}")
    
    # Use cases for LLM developers
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Document Analysis:
   - Analyze multiple pages of documents
   - Extract structured data from forms
   - Compare document versions

2. Visual Q&A Systems:
   - Build chatbots with image understanding
   - Create visual search systems
   - Answer questions about images

3. Product Analysis:
   - Compare product images
   - Extract features from product photos
   - Build visual product catalogs

4. Data Visualization Analysis:
   - Analyze charts and graphs
   - Extract insights from dashboards
   - Compare visual metrics

5. Multimodal RAG:
   - Retrieve and analyze relevant images
   - Combine text and image context
   - Build hybrid knowledge systems

6. Content Generation:
   - Generate descriptions from images
   - Create alt text for accessibility
   - Build image captioning systems

7. Quality Control:
   - Compare images for defects
   - Automated visual inspection
   - Product quality analysis

8. Educational Applications:
   - Analyze diagrams and illustrations
   - Explain visual concepts
   - Build visual learning systems
    """)
    
    # Best practices
    print("\n" + "-"*80)
    print("BEST PRACTICES")
    print("-"*80)
    print("""
1. Prompt Design:
   - Be specific about what to analyze
   - Request structured output when needed
   - Use clear, unambiguous language
   - Specify output format (JSON, markdown, etc.)

2. Image Ordering:
   - Order images logically (chronological, spatial, etc.)
   - Reference images by position if needed
   - Consider image-first vs text-first ordering

3. Provider-Specific Tips:
   
   OpenAI GPT-4V:
   - Supports high-resolution images
   - Good for detailed analysis
   - Use "gpt-4o" for best performance
   
   Anthropic Claude:
   - Images should come before text (Anthropic recommendation)
   - Excellent for document analysis
   - Strong reasoning capabilities
   
   Google Gemini:
   - Good for multi-image analysis
   - Fast inference
   - Cost-effective for high volume

4. Performance Optimization:
   - Resize images to optimal size (not too large)
   - Compress images when quality isn't critical
   - Cache vision model responses
   - Batch similar requests when possible

5. Error Handling:
   - Check image file sizes (API limits)
   - Validate image formats
   - Handle API rate limits
   - Implement retry logic

6. Cost Management:
   - Monitor token usage (images count as tokens)
   - Use appropriate model for task
   - Cache results when possible
   - Consider image compression
    """)
    
    # Advanced patterns
    print("\n" + "-"*80)
    print("ADVANCED PATTERNS")
    print("-"*80)
    print("""
1. Chain-of-Thought with Images:
    Prompt: "Analyze this image step by step:
    1. First, identify all objects
    2. Then, determine relationships
    3. Finally, draw conclusions"

2. Few-Shot Learning with Examples:
    Include example images with expected outputs
    Guide the model with visual examples

3. Comparative Analysis:
    "Compare image A and image B focusing on [specific aspect]"
    Structured comparison prompts

4. Multi-Step Reasoning:
    Use images at different steps of reasoning
    Combine with text-based chain-of-thought

5. Structured Extraction:
    Request specific JSON schemas
    Use templates for consistent output
    Validate and parse responses
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    for img_path in images:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Removed: {img_path}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
