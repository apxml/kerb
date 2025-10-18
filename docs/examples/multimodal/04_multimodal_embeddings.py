"""Multimodal Embeddings Example

This example demonstrates multimodal embeddings for image-text similarity.

Concepts covered:
- Generating CLIP embeddings for images and text
- Computing similarity between different modalities
- Building multimodal search systems
- Finding similar images using embeddings
- Cross-modal retrieval (text-to-image, image-to-text)
"""

from kerb.multimodal import (
    embed_multimodal,
    compute_multimodal_similarity,
    EmbeddingModelMultimodal
)
from kerb.core.enums import Device
import os


def create_sample_images():
    """Create sample images for similarity testing."""
    try:
        from PIL import Image, ImageDraw
        
        images = []
        
        # Image 1: Red circle
        img1 = Image.new('RGB', (300, 300), color='white')
        draw1 = ImageDraw.Draw(img1)
        draw1.ellipse([75, 75, 225, 225], fill='red')
        draw1.text((100, 260), "Red Circle", fill='black')
        img1.save("red_circle.jpg")
        images.append("red_circle.jpg")
        
        # Image 2: Blue square
        img2 = Image.new('RGB', (300, 300), color='white')
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([75, 75, 225, 225], fill='blue')
        draw2.text((90, 260), "Blue Square", fill='black')
        img2.save("blue_square.jpg")
        images.append("blue_square.jpg")
        
        # Image 3: Green triangle
        img3 = Image.new('RGB', (300, 300), color='white')
        draw3 = ImageDraw.Draw(img3)
        draw3.polygon([(150, 75), (225, 225), (75, 225)], fill='green')
        draw3.text((80, 260), "Green Triangle", fill='black')
        img3.save("green_triangle.jpg")
        images.append("green_triangle.jpg")
        
        # Image 4: Another red circle (similar to image 1)
        img4 = Image.new('RGB', (300, 300), color='white')
        draw4 = ImageDraw.Draw(img4)
        draw4.ellipse([50, 50, 250, 250], fill='darkred')
        draw4.text((85, 260), "Dark Red Circle", fill='black')
        img4.save("red_circle_2.jpg")
        images.append("red_circle_2.jpg")
        
        return images
    except ImportError:
        print("PIL not available")
        return []


def main():
    """Run multimodal embeddings examples."""
    
    print("="*80)
    print("MULTIMODAL EMBEDDINGS EXAMPLE")
    print("="*80)
    
    # Note about dependencies
    print("\nNote: This example requires PyTorch and transformers:")
    print("  pip install torch transformers")
    print("  (or pip install kerb[multimodal])")
    
    # Create sample images
    print("\nCreating sample images...")
    images = create_sample_images()
    
    if not images:
        print("Cannot create images. Install Pillow: pip install Pillow")
        return
    
    print(f"Created {len(images)} sample images")
    
    # Check if CLIP is available
    has_clip = False
    try:
        import torch
        import transformers
        # Try to actually generate an embedding to ensure models are available
        test_embedding = embed_multimodal(
            images[0],
            content_type="image",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
            device=Device.CPU
        )
        has_clip = True
    except Exception as e:
        print(f"\nCLIP embeddings not available: {type(e).__name__}")
        print("Showing example usage only.")
    
    # 1. Generate embeddings for images
    print("\n" + "-"*80)
    print("1. GENERATE IMAGE EMBEDDINGS")
    print("-"*80)
    
    if has_clip:
        print("\nGenerating CLIP embeddings for images...")
        
        image_embeddings = []
        for img_path in images:
            embedding = embed_multimodal(
                img_path,
                content_type="image",
                model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
                device=Device.CPU  # Use Device.CUDA for GPU
            )
            image_embeddings.append(embedding)
            print(f"  {img_path}: {len(embedding)} dimensions")
    else:
        print("\nExample code:")
        print("""
        from kerb.multimodal import embed_multimodal, EmbeddingModelMultimodal
        from kerb.core.enums import Device
        
        embedding = embed_multimodal(
            "image.jpg",
            content_type="image",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
            device=Device.CPU
        )
        print(f"Embedding dimensions: {len(embedding)}")
        """)
    
    # 2. Generate embeddings for text
    print("\n" + "-"*80)
    print("2. GENERATE TEXT EMBEDDINGS")
    print("-"*80)
    
    text_queries = [
        "a red circular shape",
        "a blue square",
        "a green triangle",
        "geometric shapes",
    ]
    
    if has_clip:
        print("\nGenerating CLIP embeddings for text queries...")
        
        text_embeddings = []
        for text in text_queries:
            embedding = embed_multimodal(
                text,
                content_type="text",
                model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
                device=Device.CPU
            )
            text_embeddings.append(embedding)
            print(f"  '{text}': {len(embedding)} dimensions")
    else:
        print("\nExample code:")
        print("""
        text_embedding = embed_multimodal(
            "a red circular shape",
            content_type="text",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32
        )
        """)
    
    # 3. Compute cross-modal similarity (text-to-image)
    print("\n" + "-"*80)
    print("3. CROSS-MODAL SIMILARITY (Text-to-Image Search)")
    print("-"*80)
    
    if has_clip:
        print("\nFinding images matching text queries...")
        
        for i, query in enumerate(text_queries):
            print(f"\nQuery: '{query}'")
            similarities = []
            
            for j, img_path in enumerate(images):
                similarity = compute_multimodal_similarity(
                    text_embeddings[i],
                    image_embeddings[j]
                )
                similarities.append((img_path, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print("  Top matches:")
            for img_path, sim in similarities[:3]:
                print(f"    {img_path}: {sim:.4f}")
    else:
        print("\nExample code:")
        print("""
        from kerb.multimodal import compute_multimodal_similarity
        
        # Find images matching a text query
        query = "a red circle"
        query_embedding = embed_multimodal(query, "text")
        
        similarities = []
        for image_path in image_paths:
            img_embedding = embed_multimodal(image_path, "image")
            similarity = compute_multimodal_similarity(query_embedding, img_embedding)
            similarities.append((image_path, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"Best match: {similarities[0][0]}")
        """)
    
    # 4. Image-to-image similarity
    print("\n" + "-"*80)
    print("4. IMAGE-TO-IMAGE SIMILARITY")
    print("-"*80)
    
    if has_clip:
        print("\nComparing images with each other...")
        
        # Compare first image with all others
        reference_img = images[0]
        print(f"\nReference: {reference_img}")
        
        for img_path in images[1:]:
            similarity = compute_multimodal_similarity(
                image_embeddings[0],
                image_embeddings[images.index(img_path)]
            )
            print(f"  vs {img_path}: {similarity:.4f}")
        
        print("\nNote: red_circle_2.jpg should have high similarity to red_circle.jpg")
    else:
        print("\nExample code:")
        print("""
        # Find similar images
        reference_embedding = embed_multimodal("reference.jpg", "image")
        
        for candidate_path in candidate_images:
            candidate_embedding = embed_multimodal(candidate_path, "image")
            similarity = compute_multimodal_similarity(reference_embedding, candidate_embedding)
            if similarity > 0.8:  # High similarity threshold
                print(f"Similar image: {candidate_path}")
        """)
    
    # Use cases for LLM developers
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Multimodal Search:
   - Search images using text queries
   - Find images semantically similar to text descriptions
   - Build Google Images-like search systems

2. Image Deduplication:
   - Find duplicate/similar images
   - Clean image datasets
   - Remove near-duplicate images from training data

3. Reverse Image Search:
   - Find similar images in a database
   - Build "find similar" features
   - Visual product search

4. Multimodal RAG:
   - Retrieve images based on text queries
   - Combine text and image retrieval
   - Build hybrid search systems

5. Content Recommendation:
   - Recommend images based on text preferences
   - Find visually similar products
   - Build image-based recommendation engines

6. Zero-shot Classification:
   - Classify images using text labels
   - No training data needed
   - Flexible categorization system

7. Visual Question Answering:
   - Combine with vision models for Q&A
   - Retrieve relevant images for questions
   - Build visual knowledge bases

8. Cross-modal Retrieval:
   - Find images from text, text from images
   - Build multimodal knowledge graphs
   - Unified search across modalities
    """)
    
    # Model selection tips
    print("\n" + "-"*80)
    print("MODEL SELECTION")
    print("-"*80)
    print("""
CLIP (ViT-B/32) - Default:
  - Good general-purpose embeddings
  - Fast inference
  - 512-dimensional embeddings
  - Well-balanced accuracy/speed

CLIP (ViT-L/14):
  - Higher accuracy
  - Larger embeddings (768 dims)
  - Slower inference
  - Better for fine-grained similarity

ImageBind (if available):
  - Multi-modal beyond image/text
  - Supports audio, video, depth
  - More complex setup
  - Research/experimental
    """)
    
    # Performance tips
    print("\n" + "-"*80)
    print("PERFORMANCE OPTIMIZATION")
    print("-"*80)
    print("""
1. Device Selection:
   - Use GPU (Device.CUDA) for batch processing
   - CPU is fine for single images
   - MPS for Apple Silicon

2. Batch Processing:
   - Process images in batches for speed
   - Pre-compute and cache embeddings
   - Store embeddings in vector databases

3. Similarity Search:
   - Use approximate nearest neighbor (ANN) for large datasets
   - Libraries: FAISS, Annoy, Hnswlib
   - Pre-filter with metadata when possible

4. Storage:
   - Store embeddings separately from images
   - Use efficient formats (numpy, parquet)
   - Consider dimensionality reduction for storage
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
