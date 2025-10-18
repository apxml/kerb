"""Document Splitting for Chunking and Embedding

This example demonstrates text splitting strategies for LLM applications.

Main concepts:
- Sentence-based splitting
- Paragraph-based splitting
- Semantic chunking strategies
- Optimizing chunks for embeddings

Use cases:
- Preparing documents for vector embeddings
- Creating chunks for RAG systems
- Splitting long documents for context windows
- Semantic search optimization
"""

import tempfile
import os
from kerb.document import (
    load_document,
    split_into_sentences,
    split_into_paragraphs,
    extract_document_stats,
    Document,
)


def create_sample_documents(temp_dir: str):
    """Create sample documents for splitting demonstration."""
    
    # Short document
    short_doc = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It enables systems to learn from data.

Applications include image recognition, natural language processing, and recommendation systems. These technologies are transforming industries worldwide.

The future of ML looks promising. New techniques emerge regularly."""
    
    short_file = os.path.join(temp_dir, "short.txt")
    with open(short_file, 'w') as f:
        f.write(short_doc)
    
    # Long technical document
    long_doc = """Deep Learning Architectures

Introduction

Deep learning has revolutionized artificial intelligence. Neural networks with multiple layers can learn complex patterns. This capability has led to breakthroughs in computer vision. Natural language processing has also benefited greatly.

Convolutional Neural Networks

CNNs are designed for processing grid-like data. They excel at image recognition tasks. The architecture includes convolutional layers. Pooling layers reduce spatial dimensions. This design captures hierarchical features effectively.

Each convolutional layer applies filters. These filters detect specific patterns. Early layers identify edges and textures. Deeper layers recognize complex objects. The network learns these features automatically.

Recurrent Neural Networks

RNNs process sequential data effectively. They maintain internal state between inputs. This makes them ideal for time series. Language modeling is another key application.

LSTM networks solve the vanishing gradient problem. They use gates to control information flow. This allows learning long-term dependencies. GRU networks provide a simpler alternative. Both architectures power modern NLP systems.

Transformer Architecture

Transformers revolutionized sequence modeling. Self-attention mechanisms enable parallel processing. This dramatically speeds up training. The architecture scales to massive datasets effectively.

BERT and GPT are prominent examples. They achieve state-of-the-art performance. Pre-training on large corpora is crucial. Fine-tuning adapts models to specific tasks. This approach dominates modern NLP.

Attention mechanisms compute relationships between elements. Query, key, and value matrices enable this. Multi-head attention captures diverse patterns. Position encodings preserve sequence information.

Applications and Future Directions

Deep learning applications are everywhere. Computer vision powers autonomous vehicles. Machine translation breaks language barriers. Speech recognition enables voice assistants.

Future research explores several directions. Efficient architectures reduce computational costs. Few-shot learning minimizes data requirements. Explainable AI increases model transparency. These advances will drive the next generation of AI systems.
""" * 2  # Duplicate to make it longer
    
    long_file = os.path.join(temp_dir, "long.txt")
    with open(long_file, 'w') as f:
        f.write(long_doc)
    
    return {'short': short_file, 'long': long_file}


def create_fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Create fixed-size chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary if possible
        if end < text_len:
            last_period = chunk.rfind('.')
            last_exclaim = chunk.rfind('!')
            last_question = chunk.rfind('?')
            sentence_end = max(last_period, last_exclaim, last_question)
            
            if sentence_end > chunk_size * 0.5:  # If we found a sentence end in latter half
                chunk = text[start:start + sentence_end + 1]
                end = start + sentence_end + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def main():
    """Run document splitting examples."""
    
    print("="*80)
    print("DOCUMENT SPLITTING FOR CHUNKING AND EMBEDDING")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        files = create_sample_documents(temp_dir)
        
        # Example 1: Sentence-based splitting
        print("\n1. SENTENCE-BASED SPLITTING")
        print("-" * 80)
        
        doc = load_document(files['short'])
        sentences = split_into_sentences(doc.content)
        
        print(f"Document split into {len(sentences)} sentences")
        print("\nFirst 3 sentences:")
        for i, sentence in enumerate(sentences[:3], 1):
            print(f"  {i}. {sentence}")
        
        # Analyze sentence lengths
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        print(f"\nSentence statistics:")
        print(f"  Average length: {avg_length:.1f} characters")
        print(f"  Shortest: {min(sentence_lengths)} characters")
        print(f"  Longest: {max(sentence_lengths)} characters")
        
        # Example 2: Paragraph-based splitting
        print("\n2. PARAGRAPH-BASED SPLITTING")
        print("-" * 80)
        
        paragraphs = split_into_paragraphs(doc.content)
        
        print(f"Document split into {len(paragraphs)} paragraphs")
        print("\nParagraph preview:")
        for i, para in enumerate(paragraphs, 1):
            words = len(para.split())
            print(f"  {i}. {words} words: {para[:60]}...")
        
        # Example 3: Fixed-size chunks with overlap
        print("\n3. FIXED-SIZE CHUNKS WITH OVERLAP")
        print("-" * 80)
        
        long_doc = load_document(files['long'])
        chunks = create_fixed_size_chunks(long_doc.content, chunk_size=400, overlap=50)
        
        print(f"Document split into {len(chunks)} chunks")
        print(f"Target chunk size: 400 characters, overlap: 50 characters")
        
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i} ({len(chunk)} chars):")
            print(f"  {chunk[:100]}...")
        
        # Show overlap
        if len(chunks) >= 2:
            print("\nOverlap between chunks 1 and 2:")
            chunk1_end = chunks[0][-60:]
            chunk2_start = chunks[1][:60]
            print(f"  End of chunk 1: ...{chunk1_end}")
            print(f"  Start of chunk 2: {chunk2_start}...")
        
        # Example 4: Semantic paragraph chunks
        print("\n4. SEMANTIC PARAGRAPH CHUNKS")
        print("-" * 80)
        
        # Split by paragraphs, but combine small ones
        paragraphs = split_into_paragraphs(long_doc.content)
        min_chunk_size = 200  # Minimum words per chunk
        
        semantic_chunks = []
        current_chunk = []
        current_words = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_words + para_words < min_chunk_size:
                current_chunk.append(para)
                current_words += para_words
            else:
                if current_chunk:
                    semantic_chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_words = para_words
        
        if current_chunk:
            semantic_chunks.append('\n\n'.join(current_chunk))
        
        print(f"Created {len(semantic_chunks)} semantic chunks")
        print(f"Minimum chunk size: {min_chunk_size} words")
        
        print("\nChunk sizes:")
        for i, chunk in enumerate(semantic_chunks, 1):
            word_count = len(chunk.split())
            print(f"  Chunk {i}: {word_count} words")
        
        # Example 5: Optimal chunking for embeddings
        print("\n5. OPTIMAL CHUNKING FOR EMBEDDINGS")
        print("-" * 80)
        
        # Typical embedding model constraints:
        # - Max tokens: 512-8192 (varies by model)
        # - Optimal chunk size: 256-512 tokens (~200-400 words)
        
        target_words = 300
        overlap_words = 50
        
        words = long_doc.content.split()
        embedding_chunks = []
        start = 0
        
        while start < len(words):
            end = start + target_words
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            embedding_chunks.append(chunk_text)
            start = end - overlap_words
        
        print(f"Created {len(embedding_chunks)} chunks for embedding")
        print(f"Target: {target_words} words per chunk")
        print(f"Overlap: {overlap_words} words")
        
        print("\nChunk analysis:")
        for i, chunk in enumerate(embedding_chunks[:3], 1):
            stats = extract_document_stats(chunk)
            print(f"  Chunk {i}:")
            print(f"    Words: {stats['word_count']}")
            print(f"    Characters: {stats['char_count']}")
            print(f"    Sentences: {stats['sentence_count']}")
        
        # Example 6: Creating Document objects for chunks
        print("\n6. CREATING DOCUMENT OBJECTS FOR CHUNKS")
        print("-" * 80)
        
        chunk_documents = []
        for i, chunk_text in enumerate(embedding_chunks[:5], 1):  # First 5 for demo
            chunk_doc = Document(
                id=f"chunk_{i:03d}",
                content=chunk_text,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(embedding_chunks),
                    "source_document": files['long'],
                    "chunk_strategy": "word_overlap",
                    "target_words": target_words,
                    "overlap_words": overlap_words,
                    "stats": extract_document_stats(chunk_text),
                },
                source=files['long']
            )
            chunk_documents.append(chunk_doc)
        
        print(f"Created {len(chunk_documents)} Document objects")
        print("\nSample chunk document:")
        sample = chunk_documents[0]
        print(f"  ID: {sample.id}")
        print(f"  Chunk index: {sample.metadata['chunk_index']}/{sample.metadata['total_chunks']}")
        print(f"  Words: {sample.metadata['stats']['word_count']}")
        print(f"  Strategy: {sample.metadata['chunk_strategy']}")
        
        # Example 7: Chunking strategy comparison
        print("\n7. CHUNKING STRATEGY COMPARISON")
        print("-" * 80)
        
        strategies = {
            "Sentences": split_into_sentences(long_doc.content),
            "Paragraphs": split_into_paragraphs(long_doc.content),
            "Fixed (400 chars)": create_fixed_size_chunks(long_doc.content, 400, 50),
            "Semantic (200 words)": semantic_chunks,
            "Embedding (300 words)": embedding_chunks,
        }
        
        print(f"{'Strategy':<25} {'Chunks':<10} {'Avg Size (words)':<20}")
        print("-" * 60)
        
        for name, chunks in strategies.items():
            chunk_count = len(chunks)
            avg_words = sum(len(c.split()) for c in chunks) / chunk_count if chunk_count > 0 else 0
            print(f"{name:<25} {chunk_count:<10} {avg_words:<20.1f}")
        
        # Recommendations
        print("\n8. CHUNKING RECOMMENDATIONS")
        print("-" * 80)
        
        doc_stats = extract_document_stats(long_doc.content)
        word_count = doc_stats['word_count']
        
        print(f"Document size: {word_count} words")
        print("\nRecommendations:")
        
        if word_count < 200:
            print("  - Use document as single chunk")
            print("  - No splitting needed")
        elif word_count < 1000:
            print("  - Paragraph-based splitting recommended")
            print("  - Preserves semantic coherence")
        else:
            print("  - Use sliding window with overlap")
            print("  - Target: 300-400 words per chunk")
            print("  - Overlap: 50-100 words")
            print("  - Ensures context preservation")
        
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Sentence splitting: Fine-grained, but may lose context")
    print("- Paragraph splitting: Preserves semantic units")
    print("- Fixed-size chunks: Consistent for embedding models")
    print("- Overlap prevents context loss at boundaries")
    print("- Target 300-400 words for most embedding models")
    print("- Choose strategy based on document structure")
    print("- Store chunk metadata for retrieval context")


if __name__ == "__main__":
    main()
