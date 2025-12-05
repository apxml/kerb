"""
Complete Document Processing Pipeline for RAG
=============================================

This example demonstrates a complete end-to-end pipeline for preparing
documents for RAG (Retrieval-Augmented Generation) systems.

Main concepts:
- Multi-stage document processing
- Load -> Clean -> Extract -> Split -> Enrich
- Preparing documents for vector embeddings
- Quality control and validation

Use cases:
- RAG system document ingestion
- Building searchable knowledge bases
- Document corpus preparation
- Production-ready document pipeline
"""

import tempfile
import os
from typing import List, Dict, Any
from kerb.document import (
    load_document,
    load_directory,
    clean_text,
    preprocess_pdf_text,
    preprocess_html_text,
    extract_document_stats,
    extract_urls,
    extract_emails,
    split_into_sentences,
    split_into_paragraphs,
    Document,
    DocumentFormat,
)


def create_sample_corpus(temp_dir: str):
    """Create a sample document corpus."""
    
    corpus_dir = os.path.join(temp_dir, "corpus")
    os.makedirs(corpus_dir)
    
    # Document 1: Technical guide
    doc1 = """Building RAG Systems: A Technical Guide

# %%
# Setup and Imports
# -----------------

Introduction

Retrieval-Augmented Generation (RAG) systems combine large language models
with external knowledge retrieval. This approach addresses limitations of
standalone LLMs, particularly regarding factual accuracy and knowledge
currency.

Architecture Overview

A typical RAG system consists of three main components:

1. Document Ingestion Pipeline
   - Load documents from various sources
   - Clean and preprocess text
   - Split into chunks
   - Generate embeddings
   - Store in vector database

2. Retrieval Module
   - Process user queries
   - Generate query embeddings
   - Perform similarity search
   - Rank and filter results

3. Generation Module
   - Combine query and retrieved context
   - Generate response using LLM
   - Ensure coherence and relevance

Best Practices

Always implement proper error handling. Monitor embedding quality metrics.
Use appropriate chunk sizes for your embedding model. Test retrieval
relevance regularly. Implement caching for common queries.

For more information, visit https://docs.rag-systems.com or contact
support@rag-systems.com.
"""
    
    # Document 2: Research notes
    doc2 = """Research Notes: Embedding Models

Date: 2024-01-15

Key Findings

Modern embedding models like text-embedding-3 and instructor-xl provide
high-quality semantic representations. Dimensionality varies from 384 to
1536 dimensions.

Performance Comparison
- OpenAI text-embedding-3-large: 3072 dimensions, excellent quality
- Cohere embed-v3: 1024 dimensions, multilingual support
- Local models: Lower cost, privacy benefits

Recommendations

For production systems, consider:
- Cost vs quality tradeoffs
- Latency requirements
- Privacy constraints
- Multilingual needs

Contact researcher@university.edu for full data.
"""
    
    # Document 3: FAQ
    doc3 = """Frequently Asked Questions

Q: How do I choose chunk size?
A: Consider your embedding model's token limit. Most models work well with
   256-512 tokens per chunk. Larger chunks provide more context but may
   dilute relevance signals.

Q: Should I use overlap between chunks?
A: Yes, overlapping chunks (50-100 tokens) prevents context loss at
   boundaries. This is especially important for technical content.

Q: How many documents should I retrieve?
A: Start with 3-5 documents. More isn't always better - too many documents
   can confuse the LLM or exceed context limits.

Q: What about document metadata?
A: Always store metadata like source, timestamp, and document type.
   This enables filtering and helps users verify sources.

Visit https://help.example.com for more FAQs.
"""
    
    docs = {
        "technical_guide.txt": doc1,
        "research_notes.txt": doc2,
        "faq.txt": doc3,
    }
    
    for filename, content in docs.items():
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
    
    return corpus_dir


def stage1_load(directory: str) -> List[Document]:
    """Stage 1: Load documents from directory."""
    print("\nSTAGE 1: LOADING DOCUMENTS")
    print("-" * 80)
    
    documents = load_directory(directory, recursive=True)
    print(f"Loaded {len(documents)} documents")
    
    for doc in documents:
        filename = os.path.basename(doc.source) if doc.source else "unknown"
        print(f"  - {filename}: {len(doc.content)} chars")
    
    return documents



# %%
# Stage2 Clean
# ------------

def stage2_clean(documents: List[Document]) -> List[Document]:
    """Stage 2: Clean and preprocess documents."""
    print("\nSTAGE 2: CLEANING AND PREPROCESSING")
    print("-" * 80)
    
    cleaned_docs = []
    
    for doc in documents:
        # Apply format-specific preprocessing
        if doc.format == DocumentFormat.PDF:
            cleaned_content = preprocess_pdf_text(doc.content)
        elif doc.format == DocumentFormat.HTML:
            cleaned_content = preprocess_html_text(doc.content)
        else:
            cleaned_content = doc.content
        
        # General cleaning
        cleaned_content = clean_text(
            cleaned_content,
            normalize_whitespace=True,
            remove_urls=False,  # Keep URLs for context
            remove_emails=False,  # Keep emails for metadata
        )
        
        # Create cleaned document
        cleaned_doc = Document(
            content=cleaned_content,
            metadata=doc.metadata.copy(),
            source=doc.source,
            format=doc.format,
        )
        
        cleaned_docs.append(cleaned_doc)
        
        original_len = len(doc.content)
        cleaned_len = len(cleaned_content)
        reduction = ((original_len - cleaned_len) / original_len * 100) if original_len > 0 else 0
        
        filename = os.path.basename(doc.source) if doc.source else "unknown"
        print(f"  {filename}: {original_len} -> {cleaned_len} chars ({reduction:.1f}% reduction)")
    
    return cleaned_docs


def stage3_extract_metadata(documents: List[Document]) -> List[Document]:
    """Stage 3: Extract and enrich metadata."""
    print("\nSTAGE 3: EXTRACTING METADATA")
    print("-" * 80)
    
    enriched_docs = []
    
    for doc in documents:
        # Extract statistics
        stats = extract_document_stats(doc.content)
        
        # Extract entities
        urls = extract_urls(doc.content)
        emails = extract_emails(doc.content)
        
        # Determine document category
        content_lower = doc.content.lower()
        if 'faq' in content_lower or 'question' in content_lower:
            category = "faq"
        elif 'research' in content_lower or 'study' in content_lower:
            category = "research"
        elif 'guide' in content_lower or 'tutorial' in content_lower:
            category = "guide"
        else:
            category = "general"
        
        # Enrich metadata
        enriched_metadata = {
            **doc.metadata,
            "stats": stats,
            "urls": urls,
            "emails": emails,
            "category": category,
            "has_code": "```" in doc.content or "def " in doc.content,
            "has_links": len(urls) > 0,
        }
        
        enriched_doc = Document(
            content=doc.content,
            metadata=enriched_metadata,
            source=doc.source,
            format=doc.format,
        )
        
        enriched_docs.append(enriched_doc)
        
        filename = os.path.basename(doc.source) if doc.source else "unknown"
        print(f"  {filename}:")
        print(f"    Category: {category}")
        print(f"    Words: {stats['word_count']}")
        print(f"    URLs: {len(urls)}")
        print(f"    Emails: {len(emails)}")
    
    return enriched_docs



# %%
# Stage4 Split Into Chunks
# ------------------------

def stage4_split_into_chunks(documents: List[Document]) -> List[Document]:
    """Stage 4: Split documents into chunks."""
    print("\nSTAGE 4: SPLITTING INTO CHUNKS")
    print("-" * 80)
    
    chunks = []
    target_words = 300
    overlap_words = 50
    
    for doc_idx, doc in enumerate(documents):
        words = doc.content.split()
        doc_chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = start + target_words
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk document
            chunk_doc = Document(
                id=f"doc{doc_idx:02d}_chunk{chunk_idx:03d}",
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "source_document": doc.source,
                    "chunk_index": chunk_idx,
                    "chunk_start_word": start,
                    "chunk_end_word": min(end, len(words)),
                    "is_first_chunk": chunk_idx == 0,
                    "is_last_chunk": end >= len(words),
                },
                source=doc.source,
                format=doc.format,
            )
            
            chunks.append(chunk_doc)
            doc_chunks.append(chunk_doc)
            
            start = end - overlap_words
            chunk_idx += 1
        
        filename = os.path.basename(doc.source) if doc.source else "unknown"
        print(f"  {filename}: {len(doc_chunks)} chunks")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks


def stage5_validate(chunks: List[Document]) -> List[Document]:
    """Stage 5: Validate and filter chunks."""
    print("\nSTAGE 5: VALIDATION AND QUALITY CONTROL")
    print("-" * 80)
    
    valid_chunks = []
    rejected = []
    
    for chunk in chunks:
        stats = extract_document_stats(chunk.content)
        
        # Quality checks
        is_valid = True
        rejection_reasons = []
        
        # Minimum length check
        if stats['word_count'] < 50:
            is_valid = False
            rejection_reasons.append("too_short")
        
        # Maximum length check
        if stats['word_count'] > 600:
            is_valid = False
            rejection_reasons.append("too_long")
        
        # Content quality check (very basic)
        if chunk.content.strip().count(' ') < 10:
            is_valid = False
            rejection_reasons.append("low_quality")
        
        if is_valid:
            valid_chunks.append(chunk)
        else:
            rejected.append({
                "chunk_id": chunk.id,
                "reasons": rejection_reasons,
                "word_count": stats['word_count']
            })
    
    print(f"Valid chunks: {len(valid_chunks)}")
    print(f"Rejected chunks: {len(rejected)}")
    
    if rejected:
        print("\nRejection summary:")
        for item in rejected[:3]:  # Show first 3
            print(f"  {item['chunk_id']}: {', '.join(item['reasons'])} ({item['word_count']} words)")
    
    return valid_chunks



# %%
# Stage6 Prepare For Embedding
# ----------------------------

def stage6_prepare_for_embedding(chunks: List[Document]) -> List[Dict[str, Any]]:
    """Stage 6: Prepare final format for embedding."""
    print("\nSTAGE 6: PREPARING FOR EMBEDDING")
    print("-" * 80)
    
    embedding_ready = []
    
    for chunk in chunks:
        # Create embedding-ready record
        record = {
            "id": chunk.id,
            "text": chunk.content,
            "metadata": {
                "source": chunk.source,
                "category": chunk.metadata.get("category", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "word_count": chunk.metadata.get("stats", {}).get("word_count", 0),
                "has_links": chunk.metadata.get("has_links", False),
                "is_first_chunk": chunk.metadata.get("is_first_chunk", False),
            }
        }
        
        embedding_ready.append(record)
    
    print(f"Prepared {len(embedding_ready)} records for embedding")
    print("\nSample record:")
    sample = embedding_ready[0]
    print(f"  ID: {sample['id']}")
    print(f"  Text length: {len(sample['text'])} chars")
    print(f"  Metadata keys: {list(sample['metadata'].keys())}")
    print(f"  Category: {sample['metadata']['category']}")
    
    return embedding_ready


def main():
    """Run complete document processing pipeline."""
    
    print("="*80)
    print("COMPLETE DOCUMENT PROCESSING PIPELINE FOR RAG")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample corpus
        corpus_dir = create_sample_corpus(temp_dir)
        
        # Execute pipeline stages
        print("\nPIPELINE EXECUTION")
        print("="*80)
        
        # Stage 1: Load
        documents = stage1_load(corpus_dir)
        
        # Stage 2: Clean
        cleaned_docs = stage2_clean(documents)
        
        # Stage 3: Extract metadata
        enriched_docs = stage3_extract_metadata(cleaned_docs)
        
        # Stage 4: Split into chunks
        chunks = stage4_split_into_chunks(enriched_docs)
        
        # Stage 5: Validate
        valid_chunks = stage5_validate(chunks)
        
        # Stage 6: Prepare for embedding
        embedding_records = stage6_prepare_for_embedding(valid_chunks)
        
        # Pipeline summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        print(f"\nInput: {len(documents)} documents")
        print(f"After cleaning: {len(cleaned_docs)} documents")
        print(f"After enrichment: {len(enriched_docs)} documents")
        print(f"After chunking: {len(chunks)} chunks")
        print(f"After validation: {len(valid_chunks)} valid chunks")
        print(f"Final output: {len(embedding_records)} embedding-ready records")
        
        # Statistics by category
        print("\nRecords by category:")
        categories = {}
        for record in embedding_records:
            cat = record['metadata']['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} chunks")
        
        # Sample output
        print("\n" + "="*80)
        print("SAMPLE OUTPUT FOR VECTOR DATABASE")
        print("="*80)
        
        sample = embedding_records[0]
        print(f"\nRecord ID: {sample['id']}")
        print(f"Text preview: {sample['text'][:150]}...")
        print(f"\nMetadata:")
        for key, value in sample['metadata'].items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Generate embeddings for each record['text']")
        print("2. Store embeddings in vector database with metadata")
        print("3. Create indices for efficient similarity search")
        print("4. Implement retrieval function with filtering")
        print("5. Connect to LLM for generation phase")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Multi-stage pipelines ensure quality and consistency")
    print("- Clean and preprocess before extracting metadata")
    print("- Chunk size should match embedding model constraints")
    print("- Always validate chunks before embedding")
    print("- Rich metadata enables better retrieval filtering")
    print("- Track processing statistics for monitoring")
    print("- Implement error handling at each stage")
    print("- This pipeline is production-ready for RAG systems")


if __name__ == "__main__":
    main()
