"""Document Retrieval System Example

This example demonstrates building a RAG-like document retrieval system.

Main concepts:
- Creating a document index with embeddings
- Implementing semantic search
- Re-ranking results
- Building a complete RAG pipeline
- Handling document updates
"""

from kerb.embedding import (
    embed,
    embed_batch,
    cosine_similarity,
    top_k_similar,
    mean_pooling
)
import time


class Document:
    """Simple document class."""
    
    def __init__(self, doc_id, title, content, metadata=None):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, title='{self.title}')"


class DocumentRetriever:
    """Semantic document retrieval system."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.doc_index = {}
    
    def add_document(self, document):
        """Add a document to the index."""
        # Create embedding from title and content
        combined_text = f"{document.title}. {document.content}"
        embedding = embed(combined_text)
        
        # Store document and embedding
        doc_idx = len(self.documents)
        self.documents.append(document)
        self.embeddings.append(embedding)
        self.doc_index[document.doc_id] = doc_idx
        
        return doc_idx
    
    def add_documents_batch(self, documents):
        """Add multiple documents efficiently."""
        # Combine title and content for each document
        combined_texts = [
            f"{doc.title}. {doc.content}"
            for doc in documents
        ]
        
        # Generate embeddings in batch
        new_embeddings = embed_batch(combined_texts)
        
        # Add to index
        for doc, embedding in zip(documents, new_embeddings):
            doc_idx = len(self.documents)
            self.documents.append(doc)
            self.embeddings.append(embedding)
            self.doc_index[doc.doc_id] = doc_idx
    
    def search(self, query, top_k=5, min_score=0.0):
        """Search for relevant documents."""
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = embed(query)
        
        # Find top-k similar documents
        top_indices = top_k_similar(query_embedding, self.embeddings, k=min(top_k, len(self.embeddings)))
        
        # Build results with scores
        results = []
        for idx in top_indices:
            score = cosine_similarity(query_embedding, self.embeddings[idx])
            
            if score >= min_score:
                results.append({
                    'document': self.documents[idx],
                    'score': score,
                    'index': idx
                })
        
        return results
    
    def get_document(self, doc_id):
        """Retrieve a specific document by ID."""
        idx = self.doc_index.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None
    
    def update_document(self, doc_id, new_content=None, new_title=None):
        """Update a document and re-compute its embedding."""
        idx = self.doc_index.get(doc_id)
        if idx is None:
            return False
        
        doc = self.documents[idx]
        
        if new_title:
            doc.title = new_title
        if new_content:
            doc.content = new_content
        
        # Re-compute embedding
        combined_text = f"{doc.title}. {doc.content}"
        self.embeddings[idx] = embed(combined_text)
        
        return True
    
    def get_stats(self):
        """Get retriever statistics."""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
        }


def main():
    """Run document retrieval example."""
    
    print("="*80)
    print("DOCUMENT RETRIEVAL SYSTEM EXAMPLE")
    print("="*80)
    
    # 1. Create retriever and add documents
    print("\n1. BUILDING DOCUMENT INDEX")
    print("-"*80)
    
    retriever = DocumentRetriever()
    
    # Create sample documents
    documents = [
        Document(
            doc_id="ml-101",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of AI that enables systems to learn from data. "
                   "Common algorithms include decision trees, neural networks, and support vector machines.",
            metadata={"category": "AI", "difficulty": "beginner"}
        ),
        Document(
            doc_id="nlp-basics",
            title="Natural Language Processing Fundamentals",
            content="NLP helps computers understand and process human language. "
                   "Key tasks include text classification, named entity recognition, and sentiment analysis.",
            metadata={"category": "NLP", "difficulty": "beginner"}
        ),
        Document(
            doc_id="deep-learning",
            title="Deep Learning with Neural Networks",
            content="Deep learning uses multi-layer neural networks to learn complex patterns. "
                   "Popular architectures include CNNs for images and RNNs for sequences.",
            metadata={"category": "AI", "difficulty": "advanced"}
        ),
        Document(
            doc_id="data-prep",
            title="Data Preprocessing Techniques",
            content="Data preprocessing is essential for ML success. "
                   "Steps include cleaning, normalization, feature engineering, and handling missing values.",
            metadata={"category": "Data Science", "difficulty": "intermediate"}
        ),
        Document(
            doc_id="transformers",
            title="Transformer Architecture",
            content="Transformers revolutionized NLP with attention mechanisms. "
                   "Models like BERT and GPT use transformers for various language tasks.",
            metadata={"category": "NLP", "difficulty": "advanced"}
        ),
        Document(
            doc_id="python-ml",
            title="Python for Machine Learning",
            content="Python is the primary language for ML development. "
                   "Libraries like scikit-learn, TensorFlow, and PyTorch are widely used.",
            metadata={"category": "Programming", "difficulty": "beginner"}
        )
    ]
    
    print(f"Adding {len(documents)} documents to index...")
    start_time = time.time()
    retriever.add_documents_batch(documents)
    elapsed = time.time() - start_time
    
    stats = retriever.get_stats()
    print(f"Indexed {stats['total_documents']} documents in {elapsed:.4f}s")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    
    # 2. Perform semantic searches
    print("\n2. SEMANTIC SEARCH")
    print("-"*80)
    
    queries = [
        "How do neural networks work?",
        "Text processing and language understanding",
        "Preparing data for machine learning"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=3)
        
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            print(f"  {i}. [{score:.4f}] {doc.title}")
            print(f"     Category: {doc.metadata['category']}, "
                  f"Difficulty: {doc.metadata['difficulty']}")
    
    # 3. Threshold filtering
    print("\n3. THRESHOLD FILTERING")
    print("-"*80)
    
    query = "quantum computing"
    threshold = 0.3
    
    print(f"Query: '{query}'")
    print(f"Minimum score threshold: {threshold}")
    
    results = retriever.search(query, top_k=5, min_score=threshold)
    
    if results:
        print(f"\nFound {len(results)} results above threshold:")
        for i, result in enumerate(results, 1):
            doc = result['document']
            print(f"  {i}. [{result['score']:.4f}] {doc.title}")
    else:
        print("\nNo results found above threshold")
        # Show best match anyway
        all_results = retriever.search(query, top_k=1, min_score=0.0)
        if all_results:
            best = all_results[0]
            print(f"Best match (below threshold): [{best['score']:.4f}] {best['document'].title}")
    
    # 4. Document lookup
    print("\n4. DOCUMENT LOOKUP BY ID")
    print("-"*80)
    
    doc_id = "transformers"
    print(f"Looking up document: {doc_id}")
    
    doc = retriever.get_document(doc_id)
    if doc:
        print(f"Found: {doc.title}")
        print(f"Content: {doc.content[:100]}...")
    
    # 5. Update document
    print("\n5. UPDATING DOCUMENT")
    print("-"*80)
    
    doc_id = "ml-101"
    print(f"Updating document: {doc_id}")
    
    # Search before update
    query = "supervised learning algorithms"
    print(f"\nBefore update - Query: '{query}'")
    results_before = retriever.search(query, top_k=1)
    if results_before:
        print(f"  Best match: {results_before[0]['document'].title} "
              f"[{results_before[0]['score']:.4f}]")
    
    # Update document content
    new_content = (
        "Machine learning is a subset of AI that enables systems to learn from data. "
        "Supervised learning uses labeled data to train algorithms like linear regression, "
        "logistic regression, and random forests."
    )
    
    retriever.update_document(doc_id, new_content=new_content)
    print("\nDocument updated with new content about supervised learning")
    
    # Search after update
    print(f"\nAfter update - Query: '{query}'")
    results_after = retriever.search(query, top_k=1)
    if results_after:
        print(f"  Best match: {results_after[0]['document'].title} "
              f"[{results_after[0]['score']:.4f}]")
    
    # 6. Multi-query RAG pattern
    print("\n6. RAG PATTERN - MULTI-QUERY")
    print("-"*80)
    
    # Break down complex query into sub-queries
    complex_query = "What are the best practices for implementing NLP models in Python?"
    
    sub_queries = [
        "natural language processing implementation",
        "Python programming for NLP",
        "NLP model best practices"
    ]
    
    print(f"Complex query: '{complex_query}'")
    print(f"\nBreaking into {len(sub_queries)} sub-queries:")
    
    all_results = {}
    for sub_query in sub_queries:
        print(f"  - {sub_query}")
        results = retriever.search(sub_query, top_k=2, min_score=0.2)
        
        for result in results:
            doc_id = result['document'].doc_id
            if doc_id not in all_results or result['score'] > all_results[doc_id]['score']:
                all_results[doc_id] = result
    
    # Re-rank by score
    print(f"\nCombined results ({len(all_results)} unique documents):")
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(sorted_results[:3], 1):
        doc = result['document']
        print(f"  {i}. [{result['score']:.4f}] {doc.title}")
    
    # 7. Context preparation for LLM
    print("\n7. PREPARING CONTEXT FOR LLM")
    print("-"*80)
    
    user_query = "Explain transformers in NLP"
    print(f"User query: '{user_query}'")
    
    # Retrieve relevant documents
    results = retriever.search(user_query, top_k=3)
    
    # Build context
    context_parts = []
    for i, result in enumerate(results, 1):
        doc = result['document']
        context_parts.append(f"[Document {i}: {doc.title}]\n{doc.content}")
    
    context = "\n\n".join(context_parts)
    
    print(f"\nRetrieved {len(results)} documents for context")
    print(f"Total context length: {len(context)} characters")
    print("\nContext preview:")
    print(context[:200] + "...")
    
    # This context would be sent to an LLM with the query
    print("\nThis context would be sent to an LLM for generation")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Document retrieval systems use embeddings for semantic search")
    print("- Batch indexing is efficient for multiple documents")
    print("- Top-k retrieval finds most relevant documents")
    print("- Threshold filtering ensures quality results")
    print("- Documents can be updated and re-indexed")
    print("- Multi-query approach improves recall")
    print("- Retrieved context enhances LLM responses (RAG)")


if __name__ == "__main__":
    main()
