"""
Batch Document Processing and Management
========================================

This example demonstrates loading and processing multiple documents efficiently.

Main concepts:
- Loading documents from directories
- Batch processing documents
- Merging multiple documents
- Filtering and organizing document collections

Use cases:
- Building document corpora for RAG systems
- Processing document batches for embedding
- Organizing knowledge bases
- Document collection management
"""

import tempfile
import os
from pathlib import Path
from kerb.document import (
    load_directory,
    load_document,
    merge_documents,
    extract_document_stats,
    Document,
    DocumentFormat,
)


def create_document_collection(temp_dir: str):
    """Create a sample document collection."""
    
    # Create subdirectories
    docs_dir = os.path.join(temp_dir, "documents")
    tech_dir = os.path.join(docs_dir, "technical")
    marketing_dir = os.path.join(docs_dir, "marketing")
    
    os.makedirs(tech_dir)
    os.makedirs(marketing_dir)
    
    # Technical documents
    tech_docs = {
        "api_guide.txt": """API Documentation

# %%
# Setup and Imports
# -----------------
        
This guide covers our REST API endpoints.
Authentication is required via API keys.
Rate limits: 1000 requests per hour.
        """,
        "architecture.md": """# System Architecture

## Overview
Our system uses a microservices architecture.

## Components
- API Gateway
- Authentication Service
- Data Processing Pipeline
        """,
        "config.json": """{
  "service": "api",
  "version": "2.0",
  "endpoints": ["users", "data", "reports"]
}"""
    }
    
    for filename, content in tech_docs.items():
        filepath = os.path.join(tech_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
    
    # Marketing documents
    marketing_docs = {
        "product_brief.txt": """Product Brief
        
Our AI-powered platform helps businesses automate document processing.
Key features include intelligent extraction and classification.
        """,
        "press_release.md": """# New Product Launch

We're excited to announce our latest innovation in AI technology.
This product transforms how organizations handle documents.
        """,
        "customer_data.csv": """customer_id,company,industry
1,TechCorp,Technology
2,FinanceInc,Finance
3,HealthCo,Healthcare"""
    }
    
    for filename, content in marketing_docs.items():
        filepath = os.path.join(marketing_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
    
    return {
        'root': docs_dir,
        'technical': tech_dir,
        'marketing': marketing_dir
    }


def main():
    """Run batch document processing examples."""
    
    print("="*80)
    print("BATCH DOCUMENT PROCESSING")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dirs = create_document_collection(temp_dir)
        
        # Example 1: Load all documents from a directory
        print("\n1. LOAD DIRECTORY (NON-RECURSIVE)")
        print("-" * 80)
        
        tech_docs = load_directory(dirs['technical'])
        print(f"Loaded {len(tech_docs)} technical documents")
        
        for doc in tech_docs:
            print(f"  - {Path(doc.source).name if doc.source else 'Unknown'} "
                  f"({doc.format.value}, {len(doc)} chars)")
        
        # Example 2: Recursive directory loading
        print("\n2. LOAD DIRECTORY (RECURSIVE)")
        print("-" * 80)
        
        all_docs = load_directory(dirs['root'], recursive=True)
        print(f"Loaded {len(all_docs)} total documents")
        
        # Group by directory
        by_category = {}
        for doc in all_docs:
            if doc.source:
                parent = Path(doc.source).parent.name
                if parent not in by_category:
                    by_category[parent] = []
                by_category[parent].append(doc)
        
        for category, docs in by_category.items():
            print(f"  {category}: {len(docs)} documents")
        
        # Example 3: Load with file pattern filtering
        print("\n3. PATTERN-BASED LOADING")
        print("-" * 80)
        
        # Load only markdown files
        md_docs = load_directory(dirs['root'], pattern="*.md", recursive=True)
        print(f"Markdown files: {len(md_docs)}")
        
        # Load only text files
        txt_docs = load_directory(dirs['root'], pattern="*.txt", recursive=True)
        print(f"Text files: {len(txt_docs)}")
        
        # Example 4: Batch document analysis
        print("\n4. BATCH DOCUMENT ANALYSIS")
        print("-" * 80)
        
        total_words = 0
        total_chars = 0
        format_counts = {}
        
        for doc in all_docs:
            stats = extract_document_stats(doc.content)
            total_words += stats['word_count']
            total_chars += stats['char_count']
            
            fmt = doc.format.value
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        print(f"Collection Statistics:")
        print(f"  Total documents: {len(all_docs)}")
        print(f"  Total words: {total_words:,}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Average words per document: {total_words // len(all_docs):,}")
        
        print(f"\nFormat Distribution:")
        for fmt, count in sorted(format_counts.items()):
            print(f"  {fmt}: {count}")
        
        # Example 5: Filter and organize documents
        print("\n5. FILTERING AND ORGANIZING")
        print("-" * 80)
        
        # Filter by format
        text_documents = [d for d in all_docs if d.format == DocumentFormat.TXT]
        md_documents = [d for d in all_docs if d.format == DocumentFormat.MARKDOWN]
        structured_docs = [d for d in all_docs if d.format in [DocumentFormat.JSON, DocumentFormat.CSV]]
        
        print(f"Text documents: {len(text_documents)}")
        print(f"Markdown documents: {len(md_documents)}")
        print(f"Structured documents: {len(structured_docs)}")
        
        # Filter by size
        small_docs = []
        medium_docs = []
        large_docs = []
        
        for doc in all_docs:
            word_count = extract_document_stats(doc.content)['word_count']
            if word_count < 50:
                small_docs.append(doc)
            elif word_count < 200:
                medium_docs.append(doc)
            else:
                large_docs.append(doc)
        
        print(f"\nBy size:")
        print(f"  Small (<50 words): {len(small_docs)}")
        print(f"  Medium (50-200 words): {len(medium_docs)}")
        print(f"  Large (>200 words): {len(large_docs)}")
        
        # Example 6: Merge documents
        print("\n6. MERGING DOCUMENTS")
        print("-" * 80)
        
        # Merge all technical documents
        tech_merged = merge_documents(tech_docs, separator="\n\n" + "="*40 + "\n\n")
        print(f"Merged {len(tech_docs)} technical documents")
        print(f"Merged content length: {len(tech_merged)} characters")
        print(f"Sources: {tech_merged.metadata.get('num_documents')} documents")
        print(f"\nMerged content preview:")
        print(tech_merged.content[:200] + "...")
        
        # Example 7: Create document index
        print("\n7. BUILDING DOCUMENT INDEX")
        print("-" * 80)
        
        # Create a searchable index with metadata
        document_index = []
        
        for i, doc in enumerate(all_docs):
            stats = extract_document_stats(doc.content)
            
            index_entry = {
                "id": f"doc_{i:03d}",
                "source": doc.source,
                "format": doc.format.value,
                "word_count": stats['word_count'],
                "char_count": stats['char_count'],
                "category": Path(doc.source).parent.name if doc.source else "unknown",
                "filename": Path(doc.source).name if doc.source else "unknown",
            }
            document_index.append(index_entry)
        
        print(f"Created index with {len(document_index)} entries")
        print("\nSample index entries:")
        for entry in document_index[:3]:
            print(f"  ID: {entry['id']}")
            print(f"    File: {entry['filename']}")
            print(f"    Category: {entry['category']}")
            print(f"    Format: {entry['format']}")
            print(f"    Words: {entry['word_count']}")
        
        # Example 8: Batch processing with transformation
        print("\n8. BATCH PROCESSING WITH TRANSFORMATION")
        print("-" * 80)
        
        # Process all documents for embedding preparation
        processed_docs = []
        
        for doc in all_docs:
            # Create enhanced document for RAG
            processed = Document(
                id=f"rag_{hash(doc.content) % 10000:04d}",
                content=doc.content,
                metadata={
                    "original_source": doc.source,
                    "format": doc.format.value,
                    "stats": extract_document_stats(doc.content),
                    "processed_for": "embedding",
                    "category": Path(doc.source).parent.name if doc.source else "unknown",
                },
                format=doc.format,
                source=doc.source
            )
            processed_docs.append(processed)
        
        print(f"Processed {len(processed_docs)} documents for embedding")
        print("\nSample processed document:")
        sample = processed_docs[0]
        print(f"  ID: {sample.id}")
        print(f"  Category: {sample.metadata['category']}")
        print(f"  Stats: {sample.metadata['stats']['word_count']} words")
        print(f"  Processed for: {sample.metadata['processed_for']}")
        
        # Example 9: Document collection statistics
        print("\n9. COLLECTION-LEVEL STATISTICS")
        print("-" * 80)
        
        categories = set(d.metadata['category'] for d in processed_docs)
        
        for category in sorted(categories):
            cat_docs = [d for d in processed_docs if d.metadata['category'] == category]
            total_words = sum(d.metadata['stats']['word_count'] for d in cat_docs)
            avg_words = total_words / len(cat_docs) if cat_docs else 0
            
            print(f"\n{category.upper()}:")
            print(f"  Documents: {len(cat_docs)}")
            print(f"  Total words: {total_words:,}")
            print(f"  Average words: {avg_words:.0f}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Use load_directory() for batch document ingestion")
    print("- Pattern filtering helps load specific file types")
    print("- Document statistics guide chunking strategies")
    print("- Merging creates context-rich documents for RAG")
    print("- Index building enables efficient document retrieval")
    print("- Batch processing prepares documents for embeddings")
    print("- Category-based organization improves search relevance")


if __name__ == "__main__":
    main()
