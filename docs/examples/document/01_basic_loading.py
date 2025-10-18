"""Basic Document Loading Example

This example demonstrates how to load documents from various formats for LLM processing.

Main concepts:
- Automatic format detection
- Loading different document types
- Accessing document content and metadata
- Preparing documents for LLM consumption

Use cases:
- Loading documents for RAG (Retrieval-Augmented Generation)
- Processing documents for embedding generation
- Document ingestion pipelines
"""

import tempfile
import os
from pathlib import Path
from kerb.document import (
    load_document,
    load_text,
    load_markdown,
    load_json,
    load_csv,
    Document,
    DocumentFormat
)


def create_sample_files(temp_dir: str):
    """Create sample files for demonstration."""
    
    # Text file
    text_file = os.path.join(temp_dir, "sample.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("This is a sample document for LLM processing.\n")
        f.write("It contains multiple lines of text.\n")
        f.write("LLMs can use this content for various tasks.")
    
    # Markdown file with frontmatter
    markdown_file = os.path.join(temp_dir, "sample.md")
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("---\n")
        f.write("title: Sample Document\n")
        f.write("author: AI Assistant\n")
        f.write("tags: example, llm, rag\n")
        f.write("---\n\n")
        f.write("# Sample Document\n\n")
        f.write("## Introduction\n\n")
        f.write("This document demonstrates markdown loading for LLM applications.\n\n")
        f.write("## Key Points\n\n")
        f.write("- Point 1: Documents can be loaded with metadata\n")
        f.write("- Point 2: Markdown structure is preserved\n")
        f.write("- Point 3: Great for RAG systems\n")
    
    # JSON file
    json_file = os.path.join(temp_dir, "sample.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write('  "product": "AI Assistant",\n')
        f.write('  "description": "An intelligent system for document processing",\n')
        f.write('  "features": ["NLP", "RAG", "Embeddings"],\n')
        f.write('  "version": "1.0.0"\n')
        f.write('}\n')
    
    # CSV file
    csv_file = os.path.join(temp_dir, "sample.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("name,category,relevance\n")
        f.write("Document A,Technical,High\n")
        f.write("Document B,Marketing,Medium\n")
        f.write("Document C,Legal,Low\n")
    
    return {
        'text': text_file,
        'markdown': markdown_file,
        'json': json_file,
        'csv': csv_file
    }


def main():
    """Run basic document loading examples."""
    
    print("="*80)
    print("BASIC DOCUMENT LOADING EXAMPLE")
    print("="*80)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        files = create_sample_files(temp_dir)
        
        # Example 1: Auto-detect format and load
        print("\n1. AUTO-DETECT FORMAT")
        print("-" * 80)
        
        doc = load_document(files['text'])
        print(f"Loaded document: {doc.format.value}")
        print(f"Content length: {len(doc)} characters")
        print(f"Content preview: {doc.content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        
        # Example 2: Load markdown with frontmatter
        print("\n2. LOAD MARKDOWN WITH FRONTMATTER")
        print("-" * 80)
        
        md_doc = load_markdown(files['markdown'])
        print(f"Format: {md_doc.format.value if hasattr(md_doc, 'format') else 'markdown'}")
        print(f"Frontmatter: {md_doc.metadata.get('frontmatter', {})}")
        print(f"Headings found: {len(md_doc.metadata.get('headings', []))}")
        print(f"Content preview:\n{md_doc.content[:150]}...")
        
        # Example 3: Load JSON for structured data
        print("\n3. LOAD JSON FOR STRUCTURED DATA")
        print("-" * 80)
        
        json_doc = load_json(files['json'])
        print(f"JSON content type: {type(json_doc.metadata.get('parsed_content', {}))}")
        print(f"Keys in JSON: {list(json_doc.metadata.get('parsed_content', {}).keys())}")
        print(f"Content (as string):\n{json_doc.content[:200]}")
        
        # Example 4: Load CSV data
        print("\n4. LOAD CSV DATA")
        print("-" * 80)
        
        csv_doc = load_csv(files['csv'])
        print(f"Number of rows: {csv_doc.metadata.get('num_rows', 0)}")
        print(f"Headers: {csv_doc.metadata.get('headers', [])}")
        print(f"First row: {csv_doc.metadata.get('rows', [{}])[0] if csv_doc.metadata.get('rows') else 'N/A'}")
        print(f"Content preview:\n{csv_doc.content[:150]}")
        
        # Example 5: Creating documents programmatically
        print("\n5. CREATE DOCUMENT PROGRAMMATICALLY")
        print("-" * 80)
        
        # This is useful for LLM-generated content or API responses
        custom_doc = Document(
            content="This document was created programmatically for LLM processing.",
            metadata={
                "source": "llm_generation",
                "timestamp": "2025-10-14",
                "model": "gpt-4",
                "purpose": "rag_corpus"
            },
            format=DocumentFormat.TXT,
            id="doc_001"
        )
        
        print(f"Document ID: {custom_doc.id}")
        print(f"Format: {custom_doc.format.value}")
        print(f"Metadata: {custom_doc.metadata}")
        print(f"Content: {custom_doc.content}")
        
        # Example 6: Converting document to dictionary (for storage/serialization)
        print("\n6. DOCUMENT SERIALIZATION")
        print("-" * 80)
        
        doc_dict = custom_doc.to_dict()
        print(f"Document as dict keys: {list(doc_dict.keys())}")
        print(f"Serialized format: {doc_dict['format']}")
        
        # Reconstruct from dict
        restored_doc = Document.from_dict(doc_dict)
        print(f"Restored document ID: {restored_doc.id}")
        print(f"Content matches: {restored_doc.content == custom_doc.content}")
        
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Use load_document() for automatic format detection")
    print("- Metadata provides context for retrieval and filtering")
    print("- Documents can be serialized for vector databases")
    print("- Programmatic creation enables integration with LLM outputs")
    print("- Different formats preserve structure (JSON, CSV, Markdown)")


if __name__ == "__main__":
    main()
