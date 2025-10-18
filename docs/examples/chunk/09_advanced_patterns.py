"""Advanced Chunking Patterns Example

This example demonstrates real-world chunking patterns and strategies.

Main concepts:
- Combining multiple chunking strategies
- Content-type detection and routing
- Multi-stage chunking pipelines
- Production-ready patterns

Use cases:
- Building robust RAG systems
- Multi-format document processing
- Adaptive chunking based on content
- Enterprise-scale document pipelines
"""

from kerb.chunk import (
    RecursiveChunker, SemanticChunker, CodeChunker, MarkdownChunker,
    token_based_chunker, sentence_window_chunker, merge_chunks,
    optimize_chunk_size
)
from kerb.tokenizer import Tokenizer


def demonstrate_hybrid_chunking():
    """Show combining different chunking strategies."""
    print("="*80)
    print("HYBRID CHUNKING STRATEGY")
    print("="*80)
    
    document = """
# Understanding RAG Systems

RAG systems combine retrieval and generation for better LLM performance.

## Architecture

The architecture has three key components. First, a vector database stores embeddings. Second, an embedding model creates vectors. Third, an LLM generates responses.

## Implementation

```python
def create_rag_system():
    chunker = RecursiveChunker(chunk_size=500)
    embedder = EmbeddingModel("ada-002")
    store = VectorStore()
    return RAGSystem(chunker, embedder, store)
```

This code initializes the system components.
    """.strip()
    
    print(f"Mixed-format document:\n{document[:150]}...\n")
    
    # Step 1: Detect content types
    has_markdown = document.count('#') > 0
    has_code = '```' in document
    
    print("Content analysis:")
    print(f"  Has markdown: {has_markdown}")
    print(f"  Has code blocks: {has_code}")
    
    # Step 2: Use appropriate chunker
    if has_markdown:
        print("\nUsing MarkdownChunker for structured content...")
        chunker = MarkdownChunker(max_chunk_size=300)
        chunks = chunker.chunk(document)
    else:
        print("\nUsing RecursiveChunker for plain text...")
        chunker = RecursiveChunker(chunk_size=300)
        chunks = chunker.chunk(document)
    
    print(f"\nHybrid approach produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        has_code_block = '```' in chunk
        lines = chunk.split('\n')
        heading = next((l for l in lines if l.startswith('#')), 'No heading')
        
        print(f"\nChunk {i}:")
        print(f"  Type: {'Code' if has_code_block else 'Text'}")
        print(f"  Heading: {heading}")
        print(f"  Size: {len(chunk)} chars")


def demonstrate_content_routing():
    """Route different content types to appropriate chunkers."""
    print("\n" + "="*80)
    print("CONTENT-TYPE ROUTING")
    print("="*80)
    
    # Different content types
    contents = [
        {
            'type': 'markdown',
            'text': '# API Guide\n\n## Authentication\n\nUse API keys for auth.\n\n## Endpoints\n\nPOST /embed for embeddings.'
        },
        {
            'type': 'code',
            'text': 'def embed(text):\n    model = load_model()\n    return model.encode(text)\n\nclass VectorDB:\n    def __init__(self):\n        self.vectors = []'
        },
        {
            'type': 'text',
            'text': 'LLMs process text using transformers. Attention mechanisms enable context understanding. Self-attention computes token relationships.'
        }
    ]
    
    print(f"Processing {len(contents)} documents with different types\n")
    
    all_chunks = []
    
    for doc in contents:
        print(f"Document type: {doc['type']}")
        print(f"  Content: {doc['text'][:50]}...")
        
        # Route to appropriate chunker
        if doc['type'] == 'markdown':
            chunker = MarkdownChunker(max_chunk_size=100)
            chunks = chunker.chunk(doc['text'])
            print(f"  Using MarkdownChunker -> {len(chunks)} chunks")
        
        elif doc['type'] == 'code':
            chunker = CodeChunker(max_chunk_size=100, language="python")
            chunks = chunker.chunk(doc['text'])
            print(f"  Using CodeChunker -> {len(chunks)} chunks")
        
        else:  # text
            chunker = SemanticChunker(sentences_per_chunk=2)
            chunks = chunker.chunk(doc['text'])
            print(f"  Using SemanticChunker -> {len(chunks)} chunks")
        
        all_chunks.extend([{'type': doc['type'], 'text': c} for c in chunks])
        print()
    
    print(f"Total chunks across all types: {len(all_chunks)}")


def demonstrate_multi_stage_pipeline():
    """Show multi-stage chunking pipeline."""
    print("\n" + "="*80)
    print("MULTI-STAGE CHUNKING PIPELINE")
    print("="*80)
    
    long_document = """
    Large language models have revolutionized natural language processing. They enable tasks like translation, summarization, and question answering. Training requires massive datasets and computational resources. Fine-tuning adapts pre-trained models to specific domains. This approach is more efficient than training from scratch. Prompt engineering can optimize model performance without fine-tuning. Few-shot learning uses examples in the prompt. Chain-of-thought prompting improves reasoning capabilities. RAG systems combine retrieval with generation. They reduce hallucinations by grounding responses in retrieved documents. Vector databases enable efficient similarity search. Embedding models convert text to high-dimensional vectors. This enables semantic search across large document collections.
    """.strip()
    
    print(f"Long document ({len(long_document)} chars)\n")
    
    # Stage 1: Initial chunking with semantic boundaries
    print("Stage 1: Semantic chunking...")
    stage1_chunker = SemanticChunker(sentences_per_chunk=3)
    stage1_chunks = stage1_chunker.chunk(long_document)
    print(f"  Result: {len(stage1_chunks)} semantic chunks")
    
    # Stage 2: Optimize chunk sizes
    print("\nStage 2: Size optimization...")
    stage2_chunks = []
    for chunk in stage1_chunks:
        optimal_size = optimize_chunk_size(chunk, target_size=120, tolerance=0.2)
        if len(chunk) > optimal_size * 1.2:
            # Re-chunk if too large
            from kerb.chunk import simple_chunker
            sub_chunks = simple_chunker(chunk, chunk_size=optimal_size)
            stage2_chunks.extend(sub_chunks)
        else:
            stage2_chunks.append(chunk)
    print(f"  Result: {len(stage2_chunks)} optimized chunks")
    
    # Stage 3: Token-based final adjustment
    print("\nStage 3: Token limit enforcement...")
    stage3_chunks = []
    for chunk in stage2_chunks:
        # Ensure within token limits
        token_chunks = token_based_chunker(chunk, max_tokens=50, tokenizer=Tokenizer.CL100K_BASE)
        stage3_chunks.extend(token_chunks)
    print(f"  Result: {len(stage3_chunks)} token-safe chunks")
    
    print(f"\n\nPipeline summary:")
    print(f"  Input: 1 document ({len(long_document)} chars)")
    print(f"  Stage 1 (Semantic): {len(stage1_chunks)} chunks")
    print(f"  Stage 2 (Optimize): {len(stage2_chunks)} chunks")
    print(f"  Stage 3 (Tokens): {len(stage3_chunks)} chunks")
    print(f"  Final output: {len(stage3_chunks)} production-ready chunks")


def demonstrate_adaptive_strategy():
    """Adaptively choose chunking strategy based on content."""
    print("\n" + "="*80)
    print("ADAPTIVE CHUNKING STRATEGY")
    print("="*80)
    
    def smart_chunk(text: str, max_chunk_size: int = 500):
        """Intelligently choose chunking strategy."""
        
        # Analyze content
        has_code = 'def ' in text or 'class ' in text or 'import ' in text
        has_markdown = text.count('#') > 2
        avg_sentence_length = len(text) / max(len(text.split('. ')), 1)
        
        print(f"Content analysis:")
        print(f"  Contains code: {has_code}")
        print(f"  Contains markdown: {has_markdown}")
        print(f"  Avg sentence length: {avg_sentence_length:.1f} chars")
        
        # Choose strategy
        if has_code:
            print("  Strategy: CodeChunker")
            chunker = CodeChunker(max_chunk_size=max_chunk_size, language="python")
            return chunker.chunk(text)
        
        elif has_markdown:
            print("  Strategy: MarkdownChunker")
            chunker = MarkdownChunker(max_chunk_size=max_chunk_size)
            return chunker.chunk(text)
        
        elif avg_sentence_length > 100:
            print("  Strategy: RecursiveChunker (long sentences)")
            chunker = RecursiveChunker(chunk_size=max_chunk_size)
            return chunker.chunk(text)
        
        else:
            print("  Strategy: SemanticChunker (short sentences)")
            chunker = SemanticChunker(sentences_per_chunk=3)
            return chunker.chunk(text)
    
    # Test with different content types
    test_contents = [
        "def process(): pass\nclass Handler: pass",
        "# Title\n## Section\nContent here.",
        "The transformer architecture uses self-attention mechanisms to process sequences in parallel and capture long-range dependencies more effectively.",
        "LLMs are powerful. They process text. Embeddings are used. Vectors enable search."
    ]
    
    for i, content in enumerate(test_contents, 1):
        print(f"\nContent {i}: {content[:50]}...")
        chunks = smart_chunk(content)
        print(f"  Result: {len(chunks)} chunks\n")


def demonstrate_rag_production_pipeline():
    """Show production-ready RAG chunking pipeline."""
    print("\n" + "="*80)
    print("PRODUCTION RAG PIPELINE")
    print("="*80)
    
    class RAGChunkingPipeline:
        """Production-ready chunking pipeline for RAG."""
        
        def __init__(self, chunk_size=500, overlap_sentences=1, max_tokens=512):
            self.chunk_size = chunk_size
            self.overlap_sentences = overlap_sentences
            self.max_tokens = max_tokens
        
        def process(self, document, doc_type='text'):
            """Process document through chunking pipeline."""
            
            # Stage 1: Content-aware chunking
            if doc_type == 'markdown':
                chunker = MarkdownChunker(max_chunk_size=self.chunk_size)
                chunks = chunker.chunk(document)
            elif doc_type == 'code':
                chunker = CodeChunker(max_chunk_size=self.chunk_size)
                chunks = chunker.chunk(document)
            else:
                # Use sentence window for better retrieval
                chunks = sentence_window_chunker(
                    document,
                    window_sentences=3,
                    overlap_sentences=self.overlap_sentences
                )
            
            # Stage 2: Token limit enforcement
            final_chunks = []
            for chunk in chunks:
                token_chunks = token_based_chunker(
                    chunk,
                    max_tokens=self.max_tokens,
                    tokenizer=Tokenizer.CL100K_BASE
                )
                final_chunks.extend(token_chunks)
            
            # Stage 3: Quality filtering
            quality_chunks = [
                c for c in final_chunks
                if len(c.strip()) > 20  # Remove tiny chunks
            ]
            
            return quality_chunks
    
    # Initialize pipeline
    pipeline = RAGChunkingPipeline(chunk_size=400, overlap_sentences=1, max_tokens=200)
    
    # Process different document types
    documents = [
        ('text', 'RAG systems combine retrieval and generation. They reduce hallucinations. Vector databases store embeddings. Similarity search finds relevant docs.'),
        ('markdown', '# Guide\n## Setup\nInstall dependencies.\n## Usage\nRun the system.'),
        ('code', 'def embed(text):\n    return model(text)\n\nclass Store:\n    def add(self, vec): pass')
    ]
    
    print("Processing documents through RAG pipeline:\n")
    
    for doc_type, content in documents:
        print(f"Document type: {doc_type}")
        print(f"  Content: {content[:60]}...")
        
        chunks = pipeline.process(content, doc_type)
        
        print(f"  Chunks produced: {len(chunks)}")
        print(f"  Token-safe: Yes (max {pipeline.max_tokens} tokens)")
        print(f"  Quality-filtered: Yes (min 20 chars)")
        print()


def demonstrate_performance_optimization():
    """Show chunking optimization for performance."""
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION")
    print("="*80)
    
    large_text = " ".join([
        "Large language models process text efficiently." for _ in range(20)
    ])
    
    print(f"Large text: {len(large_text)} chars\n")
    
    # Strategy 1: Simple (fastest)
    from kerb.chunk import simple_chunker
    print("Strategy 1: Simple chunking (fastest)")
    simple_chunks = simple_chunker(large_text, chunk_size=100)
    print(f"  Chunks: {len(simple_chunks)}")
    print(f"  Use case: High-volume, basic needs")
    
    # Strategy 2: Recursive (balanced)
    print("\nStrategy 2: Recursive chunking (balanced)")
    recursive_chunker = RecursiveChunker(chunk_size=100)
    recursive_chunks = recursive_chunker.chunk(large_text)
    print(f"  Chunks: {len(recursive_chunks)}")
    print(f"  Use case: Quality + performance balance")
    
    # Strategy 3: Semantic (highest quality)
    print("\nStrategy 3: Semantic chunking (highest quality)")
    semantic_chunker = SemanticChunker(sentences_per_chunk=2)
    semantic_chunks = semantic_chunker.chunk(large_text)
    print(f"  Chunks: {len(semantic_chunks)}")
    print(f"  Use case: Maximum retrieval quality")
    
    print("\n\nPerformance guidelines:")
    print("  - Simple chunking: 10,000+ docs/sec")
    print("  - Recursive chunking: 5,000+ docs/sec")
    print("  - Semantic chunking: 1,000+ docs/sec")
    print("  Choose based on your throughput needs")


def demonstrate_real_world_example():
    """Complete real-world example."""
    print("\n" + "="*80)
    print("REAL-WORLD EXAMPLE: DOCUMENTATION INDEXING")
    print("="*80)
    
    documentation = {
        'readme.md': '''
# Project Documentation

## Installation
pip install package

## Quick Start
from package import feature
        '''.strip(),
        
        'api.py': '''
def authenticate(api_key):
    """Verify API key."""
    return validate(api_key)

class Client:
    def __init__(self, key):
        self.key = key
        '''.strip(),
        
        'guide.txt': '''
Getting started is easy. First install the package. Then import the modules. Configure your API key. Finally run your application.
        '''.strip()
    }
    
    print(f"Indexing {len(documentation)} files\n")
    
    indexed_chunks = []
    
    for filename, content in documentation.items():
        print(f"Processing: {filename}")
        
        # Detect file type
        if filename.endswith('.md'):
            chunker = MarkdownChunker(max_chunk_size=200)
            doc_type = 'markdown'
        elif filename.endswith('.py'):
            chunker = CodeChunker(max_chunk_size=200, language='python')
            doc_type = 'code'
        else:
            chunker = SemanticChunker(sentences_per_chunk=2)
            doc_type = 'text'
        
        chunks = chunker.chunk(content)
        
        print(f"  Type: {doc_type}")
        print(f"  Chunks: {len(chunks)}")
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            indexed_chunks.append({
                'file': filename,
                'chunk_id': i,
                'type': doc_type,
                'text': chunk,
                'size': len(chunk)
            })
        
        print()
    
    print(f"Indexing complete: {len(indexed_chunks)} searchable chunks")
    print("\nChunk distribution:")
    for chunk in indexed_chunks:
        print(f"  {chunk['file']} #{chunk['chunk_id']} ({chunk['type']}): {chunk['size']} chars")


def main():
    """Run advanced chunking pattern examples."""
    
    print("\n" + "="*80)
    print("ADVANCED CHUNKING PATTERNS")
    print("="*80)
    print("\nReal-world patterns and production-ready strategies.\n")
    
    demonstrate_hybrid_chunking()
    demonstrate_content_routing()
    demonstrate_multi_stage_pipeline()
    demonstrate_adaptive_strategy()
    demonstrate_rag_production_pipeline()
    demonstrate_performance_optimization()
    demonstrate_real_world_example()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- Combine multiple chunking strategies for optimal results
- Route content to appropriate chunkers based on type
- Use multi-stage pipelines for complex requirements
- Adapt strategy based on content analysis
- Enforce token limits for LLM compatibility
- Filter and optimize chunks for quality
- Balance performance vs quality based on needs
- Production pipelines should handle multiple formats
- Always include metadata for traceability
- Test and iterate on real data for best results
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
