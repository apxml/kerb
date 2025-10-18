"""Multimodal RAG Example

This example demonstrates a complete multimodal Retrieval Augmented Generation pipeline.

Concepts covered:
- Building a multimodal knowledge base
- Indexing images with embeddings
- Cross-modal retrieval (text queries -> image results)
- Combining vision models with LLMs for answers
- Practical RAG workflow for multimodal data
"""

from kerb.multimodal import (
    embed_multimodal,
    compute_multimodal_similarity,
    analyze_image_with_vision_model,
    get_image_info,
    EmbeddingModelMultimodal,
    VisionModel
)
from kerb.core.enums import Device
from typing import List, Dict, Tuple
import os


def create_knowledge_base_images():
    """Create a sample multimodal knowledge base."""
    try:
        from PIL import Image, ImageDraw
        
        images = []
        metadata = {}
        
        # Document 1: Product specs
        img1 = Image.new('RGB', (500, 400), color='white')
        draw1 = ImageDraw.Draw(img1)
        draw1.text((20, 20), "PRODUCT SPECIFICATIONS", fill='black')
        draw1.text((20, 60), "Model: X-2000", fill='black')
        draw1.text((20, 90), "CPU: 8-core processor", fill='black')
        draw1.text((20, 120), "RAM: 16GB DDR4", fill='black')
        draw1.text((20, 150), "Storage: 512GB SSD", fill='black')
        draw1.text((20, 180), "Display: 15.6 inch", fill='black')
        img1.save("doc_product_specs.jpg")
        images.append("doc_product_specs.jpg")
        metadata["doc_product_specs.jpg"] = {
            "type": "specifications",
            "category": "product",
            "content": "Product specifications for Model X-2000"
        }
        
        # Document 2: Sales data chart
        img2 = Image.new('RGB', (500, 400), color='white')
        draw2 = ImageDraw.Draw(img2)
        draw2.text((20, 20), "QUARTERLY SALES 2024", fill='black')
        # Simple bar chart
        draw2.rectangle([50, 250, 120, 350], fill='blue')
        draw2.rectangle([150, 200, 220, 350], fill='green')
        draw2.rectangle([250, 150, 320, 350], fill='red')
        draw2.rectangle([350, 180, 420, 350], fill='orange')
        draw2.text((50, 360), "Q1", fill='black')
        draw2.text((150, 360), "Q2", fill='black')
        draw2.text((250, 360), "Q3", fill='black')
        draw2.text((350, 360), "Q4", fill='black')
        img2.save("doc_sales_chart.jpg")
        images.append("doc_sales_chart.jpg")
        metadata["doc_sales_chart.jpg"] = {
            "type": "chart",
            "category": "sales",
            "content": "Quarterly sales data for 2024"
        }
        
        # Document 3: Installation diagram
        img3 = Image.new('RGB', (500, 400), color='white')
        draw3 = ImageDraw.Draw(img3)
        draw3.text((20, 20), "INSTALLATION GUIDE", fill='black')
        draw3.text((20, 60), "Step 1: Connect power cable", fill='black')
        draw3.text((20, 90), "Step 2: Attach monitor", fill='black')
        draw3.text((20, 120), "Step 3: Insert batteries", fill='black')
        draw3.text((20, 150), "Step 4: Power on device", fill='black')
        # Simple diagram
        draw3.rectangle([300, 100, 450, 250], outline='black', width=2)
        draw3.ellipse([350, 150, 400, 200], fill='green')
        img3.save("doc_installation.jpg")
        images.append("doc_installation.jpg")
        metadata["doc_installation.jpg"] = {
            "type": "guide",
            "category": "installation",
            "content": "Step-by-step installation instructions"
        }
        
        # Document 4: Team organization chart
        img4 = Image.new('RGB', (500, 400), color='white')
        draw4 = ImageDraw.Draw(img4)
        draw4.text((20, 20), "TEAM ORGANIZATION", fill='black')
        draw4.text((20, 60), "CEO: John Smith", fill='black')
        draw4.text((40, 100), "CTO: Sarah Johnson", fill='black')
        draw4.text((40, 130), "CFO: Mike Davis", fill='black')
        draw4.text((60, 170), "Engineering: 15 members", fill='black')
        draw4.text((60, 200), "Finance: 8 members", fill='black')
        img4.save("doc_org_chart.jpg")
        images.append("doc_org_chart.jpg")
        metadata["doc_org_chart.jpg"] = {
            "type": "chart",
            "category": "organization",
            "content": "Company organization structure"
        }
        
        return images, metadata
        
    except ImportError:
        return [], {}


class MultimodalRAG:
    """Simple multimodal RAG system."""
    
    def __init__(self):
        self.documents = {}  # {image_path: metadata}
        self.embeddings = {}  # {image_path: embedding}
        self.has_embeddings = False
        
        try:
            import torch
            import transformers
            # Try a simple test to ensure model can be loaded
            self.has_embeddings = True
        except Exception:
            pass
    
    def index_document(self, image_path: str, metadata: Dict):
        """Add a document to the knowledge base."""
        self.documents[image_path] = metadata
        
        if self.has_embeddings:
            try:
                # Generate and cache embedding
                embedding = embed_multimodal(
                    image_path,
                    content_type="image",
                    model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
                    device=Device.CPU
                )
                self.embeddings[image_path] = embedding
            except Exception:
                # If model loading fails, disable embeddings
                self.has_embeddings = False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for a query.
        
        Returns:
            List of (image_path, score, metadata) tuples
        """
        if not self.has_embeddings:
            # Fallback: simple keyword matching
            results = []
            for img_path, meta in self.documents.items():
                # Simple scoring based on keyword overlap
                score = sum(word.lower() in meta["content"].lower() 
                          for word in query.split())
                if score > 0:
                    results.append((img_path, score, meta))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        # Generate query embedding
        query_embedding = embed_multimodal(
            query,
            content_type="text",
            model=EmbeddingModelMultimodal.CLIP_VIT_B_32,
            device=Device.CPU
        )
        
        # Compute similarities
        results = []
        for img_path, img_embedding in self.embeddings.items():
            similarity = compute_multimodal_similarity(query_embedding, img_embedding)
            results.append((img_path, similarity, self.documents[img_path]))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def answer_question(self, question: str, use_vision_model: bool = False) -> Dict:
        """Answer a question using RAG.
        
        Returns:
            Dict with retrieved documents and answer
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=2)
        
        result = {
            "question": question,
            "retrieved_documents": [],
            "answer": None
        }
        
        # Add retrieved documents
        for img_path, score, metadata in retrieved_docs:
            result["retrieved_documents"].append({
                "image": img_path,
                "score": score,
                "metadata": metadata
            })
        
        # If vision model available, analyze retrieved images
        if use_vision_model and os.getenv("OPENAI_API_KEY"):
            # Analyze top retrieved image
            if retrieved_docs:
                top_image = retrieved_docs[0][0]
                analysis = analyze_image_with_vision_model(
                    top_image,
                    prompt=f"Based on this image, answer the question: {question}",
                    model=VisionModel.GPT4O
                )
                result["answer"] = analysis.description
        else:
            # Simple answer based on metadata
            if retrieved_docs:
                top_doc = retrieved_docs[0][2]
                result["answer"] = f"Based on {top_doc['type']} about {top_doc['category']}: {top_doc['content']}"
        
        return result


def main():
    """Run multimodal RAG example."""
    
    print("="*80)
    print("MULTIMODAL RAG EXAMPLE")
    print("="*80)
    
    # Create knowledge base
    print("\nCreating multimodal knowledge base...")
    images, metadata = create_knowledge_base_images()
    
    if not images:
        print("Cannot create knowledge base. Install Pillow: pip install Pillow")
        return
    
    print(f"Created {len(images)} documents:")
    for img, meta in metadata.items():
        print(f"  - {img}: {meta['type']} ({meta['category']})")
    
    # Initialize RAG system
    print("\n" + "-"*80)
    print("BUILDING RAG INDEX")
    print("-"*80)
    
    rag = MultimodalRAG()
    
    print(f"\nIndexing documents...")
    for img_path in images:
        rag.index_document(img_path, metadata[img_path])
        print(f"  Indexed: {img_path}")
    
    if not rag.has_embeddings:
        print("\nNote: Using simple keyword matching (install torch & transformers for embeddings)")
    else:
        print("\nUsing CLIP embeddings for semantic search")
    
    # Example queries
    queries = [
        "What are the product specifications?",
        "Show me the sales data",
        "How do I install the device?",
        "Who are the team members?",
    ]
    
    print("\n" + "-"*80)
    print("MULTIMODAL RETRIEVAL")
    print("-"*80)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = rag.retrieve(query, top_k=2)
        
        print("  Retrieved documents:")
        for img_path, score, meta in results:
            print(f"    - {img_path} (score: {score:.4f})")
            print(f"      Type: {meta['type']}, Category: {meta['category']}")
    
    # RAG-based question answering
    print("\n" + "-"*80)
    print("RAG QUESTION ANSWERING")
    print("-"*80)
    
    has_vision_api = bool(os.getenv("OPENAI_API_KEY"))
    
    example_question = "What are the CPU specifications?"
    print(f"\nQuestion: '{example_question}'")
    
    result = rag.answer_question(example_question, use_vision_model=has_vision_api)
    
    print("\nRetrieved Documents:")
    for doc in result["retrieved_documents"]:
        print(f"  - {doc['image']} (score: {doc['score']:.4f})")
    
    if result["answer"]:
        print(f"\nAnswer: {result['answer']}")
    else:
        print("\nAnswer: (Vision model not configured - set OPENAI_API_KEY)")
    
    # Use cases
    print("\n" + "-"*80)
    print("USE CASES FOR LLM DEVELOPERS")
    print("-"*80)
    print("""
1. Document Understanding Systems:
   - Index scanned documents, PDFs, diagrams
   - Retrieve relevant pages for questions
   - Extract structured information

2. Visual Knowledge Bases:
   - Store product images with descriptions
   - Enable visual search with text queries
   - Build hybrid text-image databases

3. Technical Documentation:
   - Index diagrams, flowcharts, schematics
   - Answer questions about technical content
   - Find relevant visual examples

4. E-commerce Search:
   - Product catalog with images
   - Search products by description
   - Visual recommendation systems

5. Medical Imaging Systems:
   - Index medical images with metadata
   - Retrieve similar cases
   - Support diagnostic decisions

6. Educational Platforms:
   - Index lecture slides and diagrams
   - Answer questions about visual content
   - Build interactive learning systems

7. News and Media:
   - Index article images
   - Search by event description
   - Build visual news archives

8. Research Databases:
   - Index scientific figures and charts
   - Find relevant visualizations
   - Support literature review
    """)
    
    # Architecture patterns
    print("\n" + "-"*80)
    print("ARCHITECTURE PATTERNS")
    print("-"*80)
    print("""
1. Basic RAG Pipeline:
   
   User Query -> Text Embedding -> Vector Search -> 
   Top-K Images -> Vision Model Analysis -> LLM Answer

2. Hybrid Retrieval:
   
   - Text search on metadata (fast pre-filter)
   - Embedding search on image content (semantic match)
   - Combine and re-rank results

3. Multi-stage Pipeline:
   
   Stage 1: Retrieve candidate images (embedding search)
   Stage 2: Analyze top-K with vision models
   Stage 3: Generate final answer with LLM

4. Caching Strategy:
   
   - Cache image embeddings (expensive to compute)
   - Cache vision model analyses (API cost)
   - Cache frequent query results

5. Real-time Updates:
   
   - Incremental indexing for new images
   - Update embeddings on content change
   - Maintain consistency in vector DB
    """)
    
    # Implementation tips
    print("\n" + "-"*80)
    print("IMPLEMENTATION TIPS")
    print("-"*80)
    print("""
1. Embedding Storage:
   - Use vector databases (Pinecone, Weaviate, Qdrant)
   - Store embeddings with metadata
   - Enable efficient similarity search

2. Metadata Design:
   - Store rich metadata (category, tags, date, etc.)
   - Enable filtering before semantic search
   - Combine keyword + semantic search

3. Vision Model Usage:
   - Pre-analyze images during indexing (optional)
   - Store analysis results as text metadata
   - Use vision models only for top retrieved results

4. Scaling Considerations:
   - Batch process embeddings
   - Use GPU for embedding generation
   - Implement approximate nearest neighbor (ANN) search

5. Quality Improvements:
   - Re-rank results with vision models
   - Use multiple embedding models (ensemble)
   - Implement relevance feedback

6. Cost Optimization:
   - Cache vision model API calls
   - Use smaller models for initial filtering
   - Batch API requests when possible
    """)
    
    # Example integration
    print("\n" + "-"*80)
    print("PRODUCTION EXAMPLE")
    print("-"*80)
    print("""
Example production RAG system with vector database:

```python
from kerb.multimodal import embed_multimodal, analyze_image_with_vision_model
import pinecone

# Initialize vector DB
pinecone.init(api_key="...")
index = pinecone.Index("multimodal-rag")

# Index new document
def index_document(image_path, metadata):
    # Generate embedding
    embedding = embed_multimodal(image_path, "image")
    
    # Optional: Pre-analyze with vision model
    analysis = analyze_image_with_vision_model(
        image_path,
        "Describe this image in detail"
    )
    
    # Store in vector DB
    index.upsert([(
        image_path,
        embedding,
        {
            **metadata,
            "description": analysis.description
        }
    )])

# Query
def search(query, top_k=5):
    # Get query embedding
    query_emb = embed_multimodal(query, "text")
    
    # Search vector DB
    results = index.query(query_emb, top_k=top_k, include_metadata=True)
    
    # Analyze top result with vision model
    top_image = results.matches[0].id
    answer = analyze_image_with_vision_model(
        top_image,
        f"Based on this image, answer: {query}"
    )
    
    return {
        "retrieved": results.matches,
        "answer": answer.description
    }
```
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
