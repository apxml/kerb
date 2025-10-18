"""Chunk Utilities Example

This example demonstrates advanced chunking utilities and customization.

Main concepts:
- merge_chunks for combining small chunks
- optimize_chunk_size for finding optimal sizes
- custom_chunker for domain-specific strategies
- Advanced chunk manipulation

Use cases:
- Post-processing chunked content
- Dynamic chunk size optimization
- Custom chunking logic for specific domains
- Chunk quality improvement
"""

from kerb.chunk import merge_chunks, optimize_chunk_size, custom_chunker


def demonstrate_merge_chunks():
    """Show merging small chunks into larger ones."""
    print("="*80)
    print("MERGING CHUNKS")
    print("="*80)
    
    # Start with small chunks
    small_chunks = [
        "LLMs process text.",
        "Tokenization is key.",
        "Context windows limit input size.",
        "Embeddings capture meaning.",
        "Vector databases enable search.",
        "RAG combines retrieval and generation."
    ]
    
    print(f"Original chunks ({len(small_chunks)} chunks):")
    for i, chunk in enumerate(small_chunks, 1):
        print(f"  {i}. '{chunk}' ({len(chunk)} chars)")
    
    # Merge into larger chunks
    merged = merge_chunks(small_chunks, max_size=80, separator=" ")
    
    print(f"\nMerged chunks ({len(merged)} chunks, max_size=80):")
    for i, chunk in enumerate(merged, 1):
        print(f"  {i}. '{chunk}' ({len(chunk)} chars)")
    
    # Try different separator
    print("\nWith paragraph separator (\\n\\n):")
    merged_para = merge_chunks(small_chunks, max_size=100, separator="\n\n")
    
    for i, chunk in enumerate(merged_para, 1):
        print(f"  {i}. ({len(chunk)} chars)")
        print(f"      {chunk}")


def demonstrate_optimize_chunk_size():
    """Show finding optimal chunk size for content."""
    print("\n" + "="*80)
    print("OPTIMIZING CHUNK SIZE")
    print("="*80)
    
    text = """
The attention mechanism is fundamental to transformers. It computes weighted sums of value vectors based on query-key similarities. Self-attention allows each position to attend to all positions in the previous layer. Multi-head attention uses multiple attention mechanisms in parallel, each focusing on different representation subspaces. This enables the model to capture various types of relationships.
    """.strip()
    
    print(f"Text ({len(text)} chars):\n{text}\n")
    
    # Find optimal size with 20% tolerance
    optimal_size = optimize_chunk_size(text, target_size=150, tolerance=0.2)
    
    print(f"Target size: 150 chars")
    print(f"Tolerance: 20%")
    print(f"Optimal size: {optimal_size} chars")
    print(f"Acceptable range: {int(150 * 0.8)}-{int(150 * 1.2)} chars")
    
    # Show chunking with optimal size
    from kerb.chunk import simple_chunker
    chunks = simple_chunker(text, chunk_size=optimal_size)
    
    print(f"\nChunks using optimal size ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        in_range = int(150 * 0.8) <= len(chunk) <= int(150 * 1.2)
        status = "within target" if in_range else "outside target"
        print(f"  {i}. {len(chunk)} chars ({status})")


def demonstrate_custom_chunker():
    """Show custom chunking logic for specific domains."""
    print("\n" + "="*80)
    print("CUSTOM CHUNKING STRATEGIES")
    print("="*80)
    
    # Custom delimiter-based content
    structured_data = """
    entry:user_query=What is RAG?
    entry:response=RAG combines retrieval and generation.
    entry:user_query=How does it work?
    entry:response=It retrieves relevant docs then generates.
    entry:user_query=What are the benefits?
    entry:response=Reduces hallucinations and improves accuracy.
    """.strip()
    
    print(f"Structured data:\n{structured_data}\n")
    
    # Define custom split function
    def split_by_entry(text):
        """Split on 'entry:' delimiter."""
        parts = text.split('entry:')
        return [f"entry:{p.strip()}" for p in parts if p.strip()]
    
    # Use custom chunker
    chunks = custom_chunker(
        structured_data,
        chunk_size=100,
        split_fn=split_by_entry
    )
    
    print(f"Custom chunker (split on 'entry:'):")
    print(f"Chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")


def demonstrate_json_chunking():
    """Show custom chunking for JSON-like data."""
    print("\n" + "="*80)
    print("JSON-LIKE DATA CHUNKING")
    print("="*80)
    
    json_data = """
    {"model": "gpt-4", "temp": 0.7, "max_tokens": 2000}
    {"model": "claude-3", "temp": 0.5, "max_tokens": 1500}
    {"model": "llama-2", "temp": 0.8, "max_tokens": 1800}
    {"model": "gemini-pro", "temp": 0.6, "max_tokens": 2500}
    """.strip()
    
    print(f"JSON data:\n{json_data}\n")
    
    # Custom function to split JSON objects
    def split_json_objects(text):
        """Split into individual JSON objects."""
        import re
        # Split on } followed by optional whitespace and {
        pattern = r'\}\s*(?=\{)'
        parts = re.split(pattern, text)
        # Re-add closing braces except for last part
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + '}')
            else:
                result.append(part)
        return result
    
    chunks = custom_chunker(
        json_data,
        chunk_size=60,
        split_fn=split_json_objects
    )
    
    print(f"JSON object chunks ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk.strip()}")


def demonstrate_merge_and_optimize():
    """Combine merging and optimization."""
    print("\n" + "="*80)
    print("COMBINING UTILITIES")
    print("="*80)
    
    # Start with uneven chunks
    chunks = [
        "LLM",
        "RAG system",
        "Vector database",
        "Embedding model converts text to vectors",
        "Similarity search finds relevant documents",
        "Context window limits input size",
        "Fine-tuning adapts models",
        "Prompt engineering optimizes performance"
    ]
    
    print(f"Initial chunks ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. '{chunk}' ({len(chunk)} chars)")
    
    # Step 1: Merge small chunks
    print("\nStep 1: Merge small chunks (max_size=100)")
    merged = merge_chunks(chunks, max_size=100, separator=". ")
    
    print(f"After merging ({len(merged)} chunks):")
    for i, chunk in enumerate(merged, 1):
        print(f"  {i}. '{chunk}' ({len(chunk)} chars)")
    
    # Step 2: Optimize for target size
    print("\nStep 2: Find optimal chunk size")
    combined_text = " ".join(merged)
    optimal = optimize_chunk_size(combined_text, target_size=80, tolerance=0.15)
    
    print(f"Optimal chunk size: {optimal} chars")
    print(f"Target range: {int(80 * 0.85)}-{int(80 * 1.15)} chars")


def demonstrate_custom_domain_chunking():
    """Show domain-specific custom chunking."""
    print("\n" + "="*80)
    print("DOMAIN-SPECIFIC CHUNKING")
    print("="*80)
    
    # Email-like content
    email_content = """
    From: researcher@ai.lab
    To: team@ai.lab
    Subject: RAG System Results
    
    The new RAG system shows impressive results. Accuracy improved by 23%. Hallucination rate decreased significantly. User satisfaction increased to 87%.
    
    Next steps include scaling to production. We need to optimize chunk sizes. Vector database performance must be monitored. Cost analysis is also required.
    """.strip()
    
    print(f"Email content:\n{email_content[:100]}...\n")
    
    # Custom function for email structure
    def split_email_sections(text):
        """Split email into header and body sections."""
        parts = []
        
        # Extract headers
        header_lines = []
        body_lines = []
        in_body = False
        
        for line in text.split('\n'):
            if not in_body and (line.startswith('From:') or line.startswith('To:') or line.startswith('Subject:')):
                header_lines.append(line)
            elif line.strip() == '' and header_lines:
                in_body = True
            elif in_body:
                body_lines.append(line)
        
        if header_lines:
            parts.append('\n'.join(header_lines))
        if body_lines:
            # Split body by paragraphs
            body = '\n'.join(body_lines)
            parts.extend([p.strip() for p in body.split('\n\n') if p.strip()])
        
        return parts
    
    chunks = custom_chunker(
        email_content,
        chunk_size=150,
        split_fn=split_email_sections
    )
    
    print(f"Email-aware chunks ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        chunk_type = "header" if "From:" in chunk or "To:" in chunk else "body"
        print(f"\nChunk {i} ({chunk_type}):")
        print(f"  {chunk}")


def demonstrate_quality_improvement():
    """Show using utilities to improve chunk quality."""
    print("\n" + "="*80)
    print("IMPROVING CHUNK QUALITY")
    print("="*80)
    
    # Poor initial chunking
    poor_chunks = [
        "a",
        "b",
        "RAG systems are powerful",
        "They combine",
        "retrieval with generation for better accuracy and reduced hallucinations",
        "x",
        "Vector databases store embeddings efficiently"
    ]
    
    print(f"Poor quality chunks ({len(poor_chunks)} chunks):")
    for i, chunk in enumerate(poor_chunks, 1):
        quality = "too short" if len(chunk) < 5 else "good"
        print(f"  {i}. '{chunk}' ({len(chunk)} chars) - {quality}")
    
    # Filter out very short chunks first
    print("\nStep 1: Filter chunks < 5 chars")
    filtered = [c for c in poor_chunks if len(c) >= 5]
    print(f"Remaining: {len(filtered)} chunks")
    
    # Merge to improve quality
    print("\nStep 2: Merge into quality chunks (target ~60 chars)")
    improved = merge_chunks(filtered, max_size=80, separator=" ")
    
    print(f"\nImproved chunks ({len(improved)} chunks):")
    for i, chunk in enumerate(improved, 1):
        print(f"  {i}. '{chunk}' ({len(chunk)} chars)")
    
    print("\nQuality improvements:")
    print(f"  - Removed {len(poor_chunks) - len(improved)} low-quality chunks")
    print(f"  - Merged into {len(improved)} coherent chunks")
    print(f"  - All chunks > 20 characters")


def demonstrate_adaptive_chunking():
    """Show adaptive chunking based on content analysis."""
    print("\n" + "="*80)
    print("ADAPTIVE CHUNKING")
    print("="*80)
    
    mixed_content = """
    # Technical Section
    
    Attention mechanisms compute similarity between queries and keys. The softmax function normalizes attention weights.
    
    # Simple Section
    
    RAG is useful. It helps LLMs. It reduces errors.
    
    # Detailed Section
    
    The transformer architecture revolutionized NLP by introducing self-attention mechanisms that allow models to weigh the importance of different input tokens dynamically. This enables parallel processing and captures long-range dependencies more effectively than recurrent architectures.
    """.strip()
    
    print(f"Mixed content:\n{mixed_content[:150]}...\n")
    
    # Analyze content to determine chunking strategy
    def analyze_and_chunk(text):
        """Adaptive chunking based on content."""
        sections = text.split('# ')
        chunks = []
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            title = lines[0] if lines else "Section"
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            # Simple heuristic: if avg sentence length < 20, merge more
            sentences = content.split('. ')
            avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            
            if avg_len < 20:
                # Short sentences: merge more
                chunks.append(f"# {title}\n{content}")
            else:
                # Long sentences: split more
                for sent in sentences:
                    if sent.strip():
                        chunks.append(f"# {title}\n{sent.strip()}")
        
        return chunks
    
    adaptive_chunks = analyze_and_chunk(mixed_content)
    
    print(f"Adaptive chunks ({len(adaptive_chunks)} chunks):")
    for i, chunk in enumerate(adaptive_chunks, 1):
        lines = chunk.split('\n')
        print(f"\nChunk {i}:")
        print(f"  {lines[0]}")
        print(f"  Content: {lines[1] if len(lines) > 1 else 'N/A'}...")
        print(f"  Size: {len(chunk)} chars")


def main():
    """Run chunk utilities examples."""
    
    print("\n" + "="*80)
    print("CHUNK UTILITIES EXAMPLES")
    print("="*80)
    print("\nAdvanced chunking utilities and customization.\n")
    
    demonstrate_merge_chunks()
    demonstrate_optimize_chunk_size()
    demonstrate_custom_chunker()
    demonstrate_json_chunking()
    demonstrate_merge_and_optimize()
    demonstrate_custom_domain_chunking()
    demonstrate_quality_improvement()
    demonstrate_adaptive_chunking()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- merge_chunks combines small chunks into larger, more useful ones
- optimize_chunk_size finds the best chunk size for your content
- custom_chunker enables domain-specific chunking logic
- Combine utilities for sophisticated chunking pipelines
- Filter and merge to improve chunk quality
- Adapt chunking strategy based on content analysis
- Custom split functions provide unlimited flexibility
- Use these utilities to fine-tune chunking for your specific needs
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
