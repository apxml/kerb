"""
Image Processing Example
========================

This example demonstrates basic image processing operations.

Concepts covered:
- Getting image information (dimensions, format, size)
- Converting image formats
- Base64 encoding for API usage
- Extracting dominant colors
- Calculating perceptual hashes for similarity
"""

from kerb.multimodal import (
    get_image_info,
    convert_image_format,
    image_to_base64,
    extract_dominant_colors,
    calculate_image_hash,
    ImageFormat
)
import os
from pathlib import Path


def create_sample_image():
    """Create a simple test image."""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple image with colored rectangles
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some colored rectangles
        draw.rectangle([50, 50, 150, 150], fill='red')
        draw.rectangle([200, 50, 300, 150], fill='blue')
        draw.rectangle([125, 175, 225, 250], fill='green')
        
        # Save as PNG
        img.save("sample_image.png")
        return "sample_image.png"
    except ImportError:
        print("PIL/Pillow not available. Skipping image creation.")
        return None


def main():
    """Run image processing examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("IMAGE PROCESSING EXAMPLE")
    print("="*80)
    
    # Create a sample image
    image_path = create_sample_image()
    
    if not image_path:
        print("\nCannot create sample image. Please install Pillow:")
        print("  pip install Pillow")
        return
    
    print(f"\nCreated sample image: {image_path}")
    
    # 1. Get image information
    print("\n" + "-"*80)
    print("1. IMAGE INFORMATION")
    print("-"*80)
    
    info = get_image_info(image_path)
    print(f"Width: {info.width}px")
    print(f"Height: {info.height}px")
    print(f"Format: {info.format}")
    print(f"Mode: {info.mode}")
    print(f"File size: {info.size_bytes:,} bytes ({info.size_bytes/1024:.2f} KB)")
    
    # 2. Format conversion
    print("\n" + "-"*80)
    print("2. FORMAT CONVERSION")
    print("-"*80)
    
    # Convert PNG to JPEG
    jpeg_path = convert_image_format(
        image_path,
        ImageFormat.JPEG,
        quality=90
    )
    print(f"Converted to JPEG: {jpeg_path}")
    
    jpeg_info = get_image_info(jpeg_path)
    print(f"JPEG size: {jpeg_info.size_bytes:,} bytes ({jpeg_info.size_bytes/1024:.2f} KB)")
    
    # Convert to WebP for better compression
    webp_path = convert_image_format(
        image_path,
        ImageFormat.WEBP,
        quality=85
    )
    print(f"Converted to WebP: {webp_path}")
    
    webp_info = get_image_info(webp_path)
    print(f"WebP size: {webp_info.size_bytes:,} bytes ({webp_info.size_bytes/1024:.2f} KB)")
    
    # 3. Base64 encoding (for LLM API usage)
    print("\n" + "-"*80)
    print("3. BASE64 ENCODING FOR API USAGE")
    print("-"*80)
    
    # With data URI prefix (ready for browser/API)
    b64_with_prefix = image_to_base64(image_path, include_prefix=True)
    print(f"Base64 with data URI prefix: {b64_with_prefix[:80]}...")
    print(f"Total length: {len(b64_with_prefix)} characters")
    
    # Without prefix (just the base64 string)
    b64_only = image_to_base64(image_path, include_prefix=False)
    print(f"\nBase64 only: {b64_only[:60]}...")
    print(f"Total length: {len(b64_only)} characters")
    
    # 4. Extract dominant colors
    print("\n" + "-"*80)
    print("4. DOMINANT COLOR EXTRACTION")
    print("-"*80)
    
    colors = extract_dominant_colors(image_path, num_colors=5)
    print("Top 5 dominant colors (RGB):")
    for i, (r, g, b) in enumerate(colors, 1):
        print(f"  {i}. RGB({r:3d}, {g:3d}, {b:3d}) - #{r:02x}{g:02x}{b:02x}")
    
    # 5. Perceptual hashing (for similarity detection)
    print("\n" + "-"*80)
    print("5. PERCEPTUAL HASHING")
    print("-"*80)
    
    hash1 = calculate_image_hash(image_path, hash_size=8)
    print(f"Perceptual hash (8x8): {hash1}")
    
    # Calculate hash of JPEG version
    hash2 = calculate_image_hash(jpeg_path, hash_size=8)
    print(f"JPEG version hash:     {hash2}")
    
    # Calculate Hamming distance (number of different bits)
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    print(f"\nHamming distance: {hamming_distance}")
    print("(Lower distance = more similar images)")
    
    # Use case explanation
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Image Information:
   - Validate image dimensions before processing
   - Check file sizes for API limits (e.g., OpenAI has 20MB limit)
   - Filter images by format/dimensions in RAG pipelines

2. Format Conversion:
   - Convert to JPEG for vision model APIs (many prefer JPEG)
   - Use WebP for efficient storage in vector databases
   - Standardize formats in multimodal pipelines

3. Base64 Encoding:
   - Directly send images to vision LLMs (GPT-4V, Claude, Gemini)
   - Embed images in API requests
   - Store image data in JSON/database

4. Color Extraction:
   - Generate color-based descriptions for image search
   - Create metadata for semantic search
   - Build visual similarity features

5. Perceptual Hashing:
   - Detect duplicate/similar images in datasets
   - Implement image deduplication in RAG systems
   - Find near-duplicate images efficiently (without embeddings)
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    for file in [image_path, jpeg_path, webp_path]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
