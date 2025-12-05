"""
Image Similarity Search Example
===============================

This example demonstrates finding similar images using perceptual hashing and embeddings.

Concepts covered:
- Perceptual hashing for fast similarity detection
- CLIP embeddings for semantic similarity
- Building an image search system
- Deduplication of image datasets
- Combining multiple similarity metrics
"""

from kerb.multimodal import (
    calculate_image_hash,
    embed_multimodal,
    compute_multimodal_similarity,
    get_image_info,
    EmbeddingModelMultimodal
)
from kerb.core.enums import Device
import os
from typing import List, Tuple


def create_image_dataset():
    """Create a dataset of images with some duplicates and similar images."""
    try:
        from PIL import Image, ImageDraw
        
        images = []
        
        # Image 1: Red square
        img1 = Image.new('RGB', (300, 300), color='white')
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([75, 75, 225, 225], fill='red')
        img1.save("img_red_square.jpg")
        images.append("img_red_square.jpg")
        
        # Image 2: Red square (slightly different - near duplicate)
        img2 = Image.new('RGB', (300, 300), color='white')
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([70, 70, 230, 230], fill='red')
        img2.save("img_red_square_2.jpg")
        images.append("img_red_square_2.jpg")
        
        # Image 3: Blue circle
        img3 = Image.new('RGB', (300, 300), color='white')
        draw3 = ImageDraw.Draw(img3)
        draw3.ellipse([75, 75, 225, 225], fill='blue')
        img3.save("img_blue_circle.jpg")
        images.append("img_blue_circle.jpg")
        
        # Image 4: Red circle (different shape, same color as img1)
        img4 = Image.new('RGB', (300, 300), color='white')
        draw4 = ImageDraw.Draw(img4)
        draw4.ellipse([75, 75, 225, 225], fill='red')
        img4.save("img_red_circle.jpg")
        images.append("img_red_circle.jpg")
        
        # Image 5: Blue square (different color, same shape as img1)
        img5 = Image.new('RGB', (300, 300), color='white')
        draw5 = ImageDraw.Draw(img5)
        draw5.rectangle([75, 75, 225, 225], fill='blue')
        img5.save("img_blue_square.jpg")
        images.append("img_blue_square.jpg")
        
        # Image 6: Green triangle (completely different)
        img6 = Image.new('RGB', (300, 300), color='white')
        draw6 = ImageDraw.Draw(img6)
        draw6.polygon([(150, 75), (225, 225), (75, 225)], fill='green')
        img6.save("img_green_triangle.jpg")
        images.append("img_green_triangle.jpg")
        
        return images
    except ImportError:
        print("PIL not available")
        return []


def find_duplicates_by_hash(image_paths: List[str], threshold: int = 5) -> List[Tuple[str, str, int]]:
    """Find duplicate images using perceptual hashing.

# %%
# Setup and Imports
# -----------------
    
    Args:
        image_paths: List of image file paths
        threshold: Maximum Hamming distance for duplicates (lower = more similar)
        
    Returns:
        List of (image1, image2, distance) tuples for duplicates
    """
    duplicates = []
    
    # Calculate hashes
    hashes = {}
    for img_path in image_paths:
        img_hash = calculate_image_hash(img_path, hash_size=8)
        hashes[img_path] = img_hash
    
    # Compare all pairs
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            hash1, hash2 = hashes[path1], hashes[path2]
            
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            
            if distance <= threshold:
                duplicates.append((path1, path2, distance))
    
    return duplicates


def find_similar_by_embeddings(
    query_image: str,
    candidate_images: List[str],
    threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """Find similar images using CLIP embeddings.
    
    Args:
        query_image: Path to query image
        candidate_images: List of candidate image paths
        threshold: Minimum similarity score (0-1)
        
    Returns:
        List of (image_path, similarity) tuples, sorted by similarity
    """
    try:
        # Generate query embedding
        query_embedding = embed_multimodal(
            query_image,
            content_type="image",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
            device=Device.CPU
        )
        
        # Generate candidate embeddings and compute similarities
        results = []
        for candidate in candidate_images:
            if candidate == query_image:
                continue
                
            candidate_embedding = embed_multimodal(
                candidate,
                content_type="image",
                model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
                device=Device.CPU
            )
            
            similarity = compute_multimodal_similarity(query_embedding, candidate_embedding)
            
            if similarity >= threshold:
                results.append((candidate, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
        
    except ImportError:
        print("PyTorch/Transformers not available for embeddings")
        return []


def main():
    """Run image similarity search examples."""
    
    print("="*80)
    print("IMAGE SIMILARITY SEARCH EXAMPLE")
    print("="*80)
    
    # Create image dataset
    images = create_image_dataset()
    
    if not images:
        print("\nCannot create images. Install Pillow: pip install Pillow")
        return
    
    print(f"\nCreated dataset with {len(images)} images:")
    for img in images:
        print(f"  - {img}")
    
    # 1. Find duplicates using perceptual hashing
    print("\n" + "-"*80)
    print("1. DUPLICATE DETECTION (Perceptual Hashing)")
    print("-"*80)
    
    print("\nFinding near-duplicate images (fast method)...")
    duplicates = find_duplicates_by_hash(images, threshold=5)
    
    if duplicates:
        print(f"\nFound {len(duplicates)} near-duplicate pairs:")
        for img1, img2, distance in duplicates:
            print(f"  {img1} <-> {img2}")
            print(f"    Hamming distance: {distance}")
            print(f"    Status: {'Exact duplicate' if distance == 0 else 'Very similar'}")
    else:
        print("No duplicates found")
    
    # 2. Semantic similarity using embeddings
    print("\n" + "-"*80)
    print("2. SEMANTIC SIMILARITY (CLIP Embeddings)")
    print("-"*80)
    
    # Check if embeddings are available
    has_embeddings = False
    try:
        import torch
        import transformers
        # Try a test embedding
        test_emb = embed_multimodal(
            images[0],
            content_type="image",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
            device=Device.CPU
        )
        has_embeddings = True
    except Exception as e:
        pass
    
    if has_embeddings:
        query_image = images[0]  # Red square
        print(f"\nQuery image: {query_image}")
        print("Finding semantically similar images...")
        
        similar_images = find_similar_by_embeddings(
            query_image,
            images,
            threshold=0.7
        )
        
        print(f"\nFound {len(similar_images)} similar images:")
        for img_path, similarity in similar_images:
            print(f"  {img_path}: {similarity:.4f}")
    else:
        print("\nEmbeddings require PyTorch and transformers:")
        print("  pip install torch transformers")
        print("\nExample output:")
        print("""
  Query image: img_red_square.jpg
  
  Found similar images:
    img_red_square_2.jpg: 0.9856 (near duplicate)
    img_blue_square.jpg: 0.8234 (same shape)
    img_red_circle.jpg: 0.7892 (same color)
        """)
    
    # 3. Combined approach
    print("\n" + "-"*80)
    print("3. COMBINED SIMILARITY DETECTION")
    print("-"*80)
    
    print("""
Best practice: Use perceptual hashing + embeddings for comprehensive similarity

1. Fast Pre-filtering (Perceptual Hash):
   - Quick elimination of obvious duplicates
   - Low computational cost
   - Good for exact/near duplicates
   
2. Semantic Similarity (Embeddings):
   - Understands visual concepts
   - Finds semantically similar images
   - Better for "similar meaning" not just "similar pixels"

Example workflow:
    # Step 1: Fast duplicate check
    duplicates = find_duplicates_by_hash(images, threshold=5)
    
    # Step 2: Semantic similarity for non-duplicates
    non_duplicates = [img for img in images if img not in duplicates]
    similar = find_similar_by_embeddings(query, non_duplicates, threshold=0.8)
    """)
    
    # Use cases
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Dataset Deduplication:
   - Remove duplicate images from training data
   - Clean web-scraped image datasets
   - Ensure dataset diversity
   
   Example:
   duplicates = find_duplicates_by_hash(all_images, threshold=3)
   unique_images = remove_duplicates(all_images, duplicates)

2. Visual Search Systems:
   - Find similar products
   - Reverse image search
   - "Find similar" features
   
   Example:
   similar = find_similar_by_embeddings(
       user_query_image, 
       product_catalog,
       threshold=0.85
   )

3. Content-Based Image Retrieval:
   - Search images by visual similarity
   - Build recommendation engines
   - Image clustering and organization

4. Plagiarism Detection:
   - Find copied/modified images
   - Detect unauthorized use
   - Copyright enforcement

5. Quality Control:
   - Detect duplicate submissions
   - Find similar defects in manufacturing
   - Visual inspection systems

6. Multimodal RAG:
   - Retrieve relevant images for queries
   - Deduplicate retrieved results
   - Build visual knowledge bases
   
   Example:
   # Get query embedding
   query_emb = embed_multimodal(user_query, "text")
   
   # Find relevant images
   for img in image_database:
       img_emb = embed_multimodal(img, "image")
       score = compute_multimodal_similarity(query_emb, img_emb)
       if score > threshold:
           relevant_images.append(img)

7. Image Organization:
   - Auto-tag similar images
   - Create smart albums
   - Group related content

8. A/B Testing:
   - Find visually similar variants
   - Compare design iterations
   - Track visual changes
    """)
    
    # Performance comparison
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    print("""
Perceptual Hashing:
  Pros:
  - Very fast (milliseconds per comparison)
  - No GPU required
  - Low memory usage
  - Good for exact/near duplicates
  
  Cons:
  - Only detects pixel-level similarity
  - Misses semantic similarity
  - Sensitive to crops/rotations
  
  Best for:
  - Large-scale duplicate detection
  - Fast pre-filtering
  - Exact match finding

CLIP Embeddings:
  Pros:
  - Semantic understanding
  - Finds conceptually similar images
  - Works across transformations
  - Enables cross-modal search (text-to-image)
  
  Cons:
  - Slower (requires model inference)
  - Needs GPU for speed
  - Higher memory usage
  - Requires ML libraries
  
  Best for:
  - Semantic search
  - Recommendation systems
  - Content understanding
  - Cross-modal retrieval
    """)
    
    # Advanced patterns
    print("\n" + "-"*80)
    print("ADVANCED PATTERNS")
    print("-"*80)
    print("""
1. Hybrid Search Strategy:
   
   def find_all_similar(query_image, database):
       # Stage 1: Fast hash-based filtering
       hash_duplicates = find_duplicates_by_hash([query_image] + database)
       
       # Stage 2: Embedding-based semantic search
       non_duplicates = exclude_duplicates(database, hash_duplicates)
       semantic_similar = find_similar_by_embeddings(query_image, non_duplicates)
       
       return hash_duplicates, semantic_similar

2. Multi-level Thresholds:
   
   # Exact duplicates: hash distance < 3
   # Near duplicates: hash distance < 8
   # Semantically similar: embedding similarity > 0.85
   # Somewhat similar: embedding similarity > 0.70

3. Incremental Indexing:
   
   # Pre-compute and cache embeddings
   embedding_index = {}
   for img in database:
       embedding_index[img] = embed_multimodal(img, "image")
   
   # Fast lookup at query time
   query_emb = embed_multimodal(query, "image")
   for img, img_emb in embedding_index.items():
       similarity = compute_multimodal_similarity(query_emb, img_emb)

4. Vector Database Integration:
   
   # Use FAISS, Pinecone, or Weaviate for efficient similarity search
   import faiss
   
   # Build index
   index = faiss.IndexFlatL2(embedding_dim)
   index.add(embeddings_matrix)
   
   # Search
   distances, indices = index.search(query_embedding, k=10)
    """)
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    for img in images:
        if os.path.exists(img):
            os.remove(img)
            print(f"Removed: {img}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
