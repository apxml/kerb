"""Token-Based Chunking Example

This example demonstrates chunking based on token limits for LLM applications.

Main concepts:
- Token-based chunking for LLM context windows
- Working with different tokenizers (GPT, Claude, etc.)
- Staying within model token limits
- Character-to-token estimation

Use cases:
- Preparing text for specific LLM models
- Respecting context window limits
- Optimizing token usage and costs
- Multi-model compatibility
"""

from kerb.chunk import token_based_chunker
from kerb.tokenizer import Tokenizer


def demonstrate_basic_token_chunking():
    """Show basic token-based chunking."""
    print("="*80)
    print("BASIC TOKEN-BASED CHUNKING")
    print("="*80)
    
    text = """
Large language models process text as tokens, not characters. A token can be a word, part of a word, or punctuation. Different models use different tokenization schemes. GPT models use byte-pair encoding (BPE). The number of tokens affects both context limits and API costs. Understanding tokenization is crucial for efficient LLM usage. You can estimate roughly 4 characters per token in English. But this varies significantly by language and content type. Always use proper token counting for production applications.
    """.strip()
    
    print(f"\nOriginal text ({len(text)} chars):\n{text}\n")
    
    # Chunk with token limit (default tokenizer is CL100K_BASE for GPT-4/GPT-3.5)
    chunks = token_based_chunker(text, max_tokens=50)
    
    print(f"\nCreated {len(chunks)} chunks (max_tokens=50, tokenizer=CL100K_BASE):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")


def demonstrate_different_tokenizers():
    """Show chunking with different tokenizer types."""
    print("\n" + "="*80)
    print("DIFFERENT TOKENIZERS")
    print("="*80)
    
    text = """
OpenAI's GPT-4 uses the cl100k_base tokenizer. GPT-3.5-turbo also uses cl100k_base. Older GPT-3 models used p50k_base. Claude models use their own tokenizer. Each tokenizer produces different token counts for the same text. This affects context window utilization and costs.
    """.strip()
    
    print(f"\nText to chunk ({len(text)} chars):\n{text}\n")
    
    # GPT-4/3.5 tokenizer
    print("GPT-4/3.5 Tokenizer (CL100K_BASE):")
    chunks_gpt4 = token_based_chunker(text, max_tokens=30, tokenizer=Tokenizer.CL100K_BASE)
    print(f"  Chunks: {len(chunks_gpt4)}")
    for i, chunk in enumerate(chunks_gpt4, 1):
        print(f"  {i}. {chunk[:60]}...")
    
    # GPT-3 tokenizer
    print("\nGPT-3 Tokenizer (P50K_BASE):")
    chunks_gpt3 = token_based_chunker(text, max_tokens=30, tokenizer=Tokenizer.P50K_BASE)
    print(f"  Chunks: {len(chunks_gpt3)}")
    for i, chunk in enumerate(chunks_gpt3, 1):
        print(f"  {i}. {chunk[:60]}...")
    
    # R50K tokenizer
    print("\nR50K Tokenizer (R50K_BASE):")
    chunks_r50k = token_based_chunker(text, max_tokens=30, tokenizer=Tokenizer.R50K_BASE)
    print(f"  Chunks: {len(chunks_r50k)}")
    for i, chunk in enumerate(chunks_r50k, 1):
        print(f"  {i}. {chunk[:60]}...")
    
    print("\nNote: Different tokenizers may produce different numbers of chunks!")


def demonstrate_context_window_limits():
    """Show how to respect different model context windows."""
    print("\n" + "="*80)
    print("RESPECTING MODEL CONTEXT WINDOWS")
    print("="*80)
    
    long_text = """
Context windows determine how much text a model can process at once. GPT-3.5-turbo has a 4K token context window, though newer versions support 16K. GPT-4 offers 8K and 32K variants. Claude 3 models support up to 200K tokens. Llama 2 has 4K context. When chunking for these models, you must stay within their limits. Remember that the context includes both input and output tokens. Always leave room for the model's response. A good practice is to use 70-80% of the context window for input. This ensures the model has space to generate comprehensive responses. For RAG systems, factor in the retrieved context size. If you retrieve 3 chunks of 500 tokens each, that's 1500 tokens before your query. Plan accordingly to avoid context overflow errors.
    """.strip()
    
    print(f"\nLong text ({len(long_text)} chars):\n{long_text[:200]}...\n")
    
    # GPT-3.5-turbo (4K context, use ~3K for input)
    print("GPT-3.5-turbo configuration (4K context):")
    gpt35_chunks = token_based_chunker(
        long_text,
        max_tokens=200,  # Leave room for system prompt and response
        tokenizer=Tokenizer.CL100K_BASE
    )
    print(f"  Max tokens per chunk: 200")
    print(f"  Chunks created: {len(gpt35_chunks)}")
    print(f"  First chunk: {gpt35_chunks[0][:80]}...")
    
    # GPT-4 (8K context, use ~6K for input)
    print("\nGPT-4 configuration (8K context):")
    gpt4_chunks = token_based_chunker(
        long_text,
        max_tokens=400,
        tokenizer=Tokenizer.CL100K_BASE
    )
    print(f"  Max tokens per chunk: 400")
    print(f"  Chunks created: {len(gpt4_chunks)}")
    print(f"  First chunk: {gpt4_chunks[0][:80]}...")
    
    # Claude 3 (200K context, use larger chunks)
    print("\nClaude 3 configuration (200K context):")
    claude_chunks = token_based_chunker(
        long_text,
        max_tokens=1000,
        tokenizer=Tokenizer.CL100K_BASE  # Approximation
    )
    print(f"  Max tokens per chunk: 1000")
    print(f"  Chunks created: {len(claude_chunks)}")
    print(f"  First chunk: {claude_chunks[0][:80]}...")


def demonstrate_cost_optimization():
    """Show how token-based chunking affects API costs."""
    print("\n" + "="*80)
    print("COST OPTIMIZATION WITH TOKEN CHUNKING")
    print("="*80)
    
    document = """
API costs are typically calculated per token. GPT-4 costs significantly more per token than GPT-3.5. Efficient chunking can reduce costs substantially. By staying close to optimal chunk sizes, you minimize wasted tokens. Padding and truncation both increase costs unnecessarily. Token-based chunking ensures you use exactly what you need. This is especially important for large-scale applications processing millions of documents.
    """.strip()
    
    print(f"\nDocument ({len(document)} chars):\n{document}\n")
    
    # Inefficient: Character-based chunking (wastes tokens)
    from kerb.chunk import simple_chunker
    char_chunks = simple_chunker(document, chunk_size=100)
    
    print(f"Character-based chunking (chunk_size=100 chars):")
    print(f"  Chunks created: {len(char_chunks)}")
    print(f"  Estimated tokens (4 chars/token): ~{len(document) // 4}")
    
    # Efficient: Token-based chunking
    token_chunks = token_based_chunker(document, max_tokens=25)
    
    print(f"\nToken-based chunking (max_tokens=25):")
    print(f"  Chunks created: {len(token_chunks)}")
    print(f"  Guaranteed tokens: ≤25 per chunk")
    
    print("\nCost implications:")
    print(f"  - Character chunking may exceed token limits unexpectedly")
    print(f"  - Token chunking guarantees you stay within budget")
    print(f"  - Critical for production applications with cost constraints")


def demonstrate_multilingual_chunking():
    """Show token chunking with multilingual text."""
    print("\n" + "="*80)
    print("MULTILINGUAL TOKEN CHUNKING")
    print("="*80)
    
    # Mix of languages
    multilingual_text = """
English text typically uses about 4 characters per token. 
日本語のテキストは異なるトークン化を使用します。
El texto en español tiene sus propias características.
中文文本的分词方式也不同。
Tokenization varies significantly across languages.
    """.strip()
    
    print(f"\nMultilingual text:\n{multilingual_text}\n")
    print(f"Total characters: {len(multilingual_text)}")
    
    # Token-based chunking handles different languages properly
    chunks = token_based_chunker(multilingual_text, max_tokens=15)
    
    print(f"\nToken-based chunks (max_tokens=15):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
    
    print("\nNote: Token counts vary by language!")
    print("  - English: ~4 chars/token")
    print("  - Japanese/Chinese: ~2 chars/token")
    print("  - Spanish: ~4-5 chars/token")
    print("  Token-based chunking handles this automatically!")


def demonstrate_rag_with_tokens():
    """Show token-based chunking for RAG systems."""
    print("\n" + "="*80)
    print("TOKEN-BASED CHUNKING FOR RAG")
    print("="*80)
    
    kb_article = """
Vector embeddings represent text as high-dimensional vectors. Similar texts have similar vector representations. This enables semantic search across large document collections. Embedding models like text-embedding-ada-002 convert text to 1536-dimensional vectors. The dimensionality affects both storage and search performance. Higher dimensions capture more nuances but require more resources. Cosine similarity measures the angle between vectors. This metric works well for text similarity. When building a RAG system, chunk documents before embedding. Each chunk becomes a searchable unit. The chunk size should balance context and precision. Too small and you lose context. Too large and retrieval becomes less precise.
    """.strip()
    
    print(f"\nKnowledge base article:\n{kb_article[:150]}...\n")
    
    # Chunk for embedding (typical: 200-500 tokens)
    embedding_chunks = token_based_chunker(
        kb_article,
        max_tokens=50,  # Small for demo, use 200-500 in production
        tokenizer=Tokenizer.CL100K_BASE
    )
    
    print(f"Chunked for embeddings (max_tokens=50):")
    print(f"  Total chunks: {len(embedding_chunks)}")
    print("\nChunks for embedding:")
    for i, chunk in enumerate(embedding_chunks, 1):
        print(f"\n  Chunk {i}:")
        print(f"    Text: {chunk[:70]}...")
        print(f"    Chars: {len(chunk)}")
        print(f"    Est. tokens: ≤50")
    
    print("\n\nRAG Pipeline Token Budget:")
    print("  1. System prompt: ~100 tokens")
    print("  2. User query: ~20 tokens")
    print("  3. Retrieved chunks: 3 chunks × 50 tokens = 150 tokens")
    print("  4. Response budget: ~200 tokens")
    print("  Total: ~470 tokens (fits in 4K context)")


def main():
    """Run token-based chunking examples."""
    
    print("\n" + "="*80)
    print("TOKEN-BASED CHUNKING EXAMPLES")
    print("="*80)
    print("\nChunking based on token limits for LLM applications.\n")
    
    demonstrate_basic_token_chunking()
    demonstrate_different_tokenizers()
    demonstrate_context_window_limits()
    demonstrate_cost_optimization()
    demonstrate_multilingual_chunking()
    demonstrate_rag_with_tokens()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- token_based_chunker ensures chunks stay within token limits
- Different models use different tokenizers (CL100K_BASE, P50K_BASE, R50K_BASE)
- Context windows vary: GPT-3.5 (4K), GPT-4 (8K/32K), Claude (200K)
- Always leave room for system prompts and model responses
- Token-based chunking reduces API costs vs character chunking
- Multilingual text has different token densities
- For RAG: typical chunk size is 200-500 tokens
- Use appropriate tokenizer for your target model
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
