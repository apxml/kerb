"""Markdown Chunking Example

This example demonstrates chunking markdown documents by heading hierarchy.

Main concepts:
- Heading-based document splitting
- Preserving document structure
- MarkdownChunker class
- Section-aware chunking

Use cases:
- Documentation indexing and search
- Knowledge base chunking
- README and wiki processing
- Structured content retrieval
"""

from kerb.chunk import MarkdownChunker


def demonstrate_basic_markdown_chunking():
    """Show basic markdown chunking by headings."""
    print("="*80)
    print("BASIC MARKDOWN CHUNKING")
    print("="*80)
    
    markdown_doc = '''
# RAG System Documentation

This guide covers building retrieval-augmented generation systems.

## Introduction

RAG combines retrieval and generation for better LLM responses. It reduces hallucinations by grounding answers in retrieved documents.

## Components

A RAG system has three main parts.

### Vector Database

Stores document embeddings for fast similarity search. Popular choices include Pinecone, Weaviate, and Chroma.

### Embedding Model

Converts text to vector representations. OpenAI's ada-002 is widely used.

### Language Model

Generates responses using retrieved context. GPT-4 and Claude work well for this.

## Implementation

Follow these steps to build your RAG system.

### Step 1: Document Processing

Chunk your documents into appropriate sizes. Use RecursiveChunker or SemanticChunker.

### Step 2: Embedding

Embed each chunk using your chosen model. Store embeddings in the vector database.

### Step 3: Retrieval

For each query, retrieve the most relevant chunks based on embedding similarity.

### Step 4: Generation

Pass retrieved chunks as context to your LLM for response generation.
    '''.strip()
    
    print(f"\nMarkdown documentation ({len(markdown_doc)} chars):\n{markdown_doc[:200]}...\n")
    
    # Chunk by heading hierarchy
    chunker = MarkdownChunker(max_chunk_size=400)
    chunks = chunker.chunk(markdown_doc)
    
    print(f"\nMarkdownChunker created {len(chunks)} chunks (max_size=400):")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        heading = next((l for l in lines if l.startswith('#')), 'No heading')
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  Heading: {heading}")
        print(f"  Preview: {chunk[:100]}...")


def demonstrate_heading_hierarchy():
    """Show how heading hierarchy affects chunking."""
    print("\n" + "="*80)
    print("HEADING HIERARCHY CHUNKING")
    print("="*80)
    
    hierarchical_doc = '''
# Main Topic

Top-level introduction.

## Subtopic A

Details about subtopic A.

### Detail A1

Specific information about A1.

### Detail A2

Specific information about A2.

## Subtopic B

Details about subtopic B.

### Detail B1

Specific information about B1.
    '''.strip()
    
    print(f"\nHierarchical markdown:\n{hierarchical_doc}\n")
    
    # Chunk with size that forces hierarchy splits
    chunker = MarkdownChunker(max_chunk_size=100)
    chunks = chunker.chunk(hierarchical_doc)
    
    print(f"\nChunks respecting hierarchy ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        heading = next((l for l in lines if l.startswith('#')), 'No heading')
        level = heading.count('#') if heading != 'No heading' else 0
        
        print(f"\nChunk {i}:")
        print(f"  Heading: {heading}")
        print(f"  Level: {level}")
        print(f"  Content: {chunk}")


def demonstrate_readme_chunking():
    """Show chunking a typical README file."""
    print("\n" + "="*80)
    print("README CHUNKING")
    print("="*80)
    
    readme = '''
# Slick Toolkit

A comprehensive toolkit for LLM application development.

## Features

- Text chunking with multiple strategies
- Embedding generation and management
- Vector storage and retrieval
- RAG pipeline components
- Token counting and optimization

## Installation

Install via pip:

```bash
pip install kerb
```

## Quick Start

Here's a simple example:

```python
from kerb.chunk import RecursiveChunker

chunker = RecursiveChunker(chunk_size=500)
chunks = chunker.chunk(your_text)
```

## API Reference

See full documentation at https://docs.example.com

### Chunking

Multiple chunking strategies available.

### Embedding

Support for OpenAI and local models.

### Retrieval

Vector database integration.
    '''.strip()
    
    print(f"README file:\n{readme[:150]}...\n")
    
    chunker = MarkdownChunker(max_chunk_size=300)
    chunks = chunker.chunk(readme)
    
    print(f"README chunked into {len(chunks)} sections:")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        heading = next((l for l in lines if l.startswith('#')), 'Content')
        has_code = '```' in chunk
        
        print(f"\nChunk {i}: {heading}")
        print(f"  Size: {len(chunk)} chars")
        print(f"  Has code block: {has_code}")


def demonstrate_wiki_style():
    """Show chunking wiki-style documentation."""
    print("\n" + "="*80)
    print("WIKI-STYLE DOCUMENTATION")
    print("="*80)
    
    wiki_page = '''
# Vector Databases

Vector databases are specialized systems for storing and querying high-dimensional vectors.

## Overview

Unlike traditional databases, vector databases optimize for similarity search. They use algorithms like HNSW or IVF for efficient nearest neighbor search.

## Popular Options

### Pinecone

**Type:** Managed cloud service

**Pros:** Easy setup, excellent performance, automatic scaling

**Cons:** Cost at scale, vendor lock-in

### Weaviate

**Type:** Open-source, cloud or self-hosted

**Pros:** Flexible deployment, GraphQL API, hybrid search

**Cons:** More complex setup

### Chroma

**Type:** Open-source, embedded

**Pros:** Simple local development, lightweight

**Cons:** Not designed for production scale

## Choosing a Database

Consider these factors:

- **Scale:** How many vectors will you store?
- **Latency:** What are your query speed requirements?
- **Cost:** What's your budget for infrastructure?
- **Deployment:** Cloud, self-hosted, or embedded?
    '''.strip()
    
    print(f"Wiki page:\n{wiki_page[:150]}...\n")
    
    chunker = MarkdownChunker(max_chunk_size=500)
    chunks = chunker.chunk(wiki_page)
    
    print(f"Wiki page chunked into {len(chunks)} sections:")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        headings = [l for l in lines if l.startswith('#')]
        
        print(f"\nChunk {i} ({len(chunk)} chars):")
        if headings:
            print(f"  Headings: {', '.join(headings)}")
        print(f"  Preview: {lines[0][:60]}...")


def demonstrate_documentation_search():
    """Simulate documentation search with markdown chunking."""
    print("\n" + "="*80)
    print("DOCUMENTATION SEARCH PIPELINE")
    print("="*80)
    
    documentation = '''
# API Documentation

## Authentication

All API requests require authentication.

### API Keys

Generate an API key in your dashboard. Include it in request headers:

```
Authorization: Bearer YOUR_API_KEY
```

### Rate Limits

- Free tier: 100 requests/minute
- Pro tier: 1000 requests/minute
- Enterprise: Custom limits

## Endpoints

### POST /embed

Generate embeddings for text.

**Request:**
```json
{
  "text": "Your text here",
  "model": "text-embedding-ada-002"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "dimensions": 1536
}
```

### POST /search

Search for similar documents.

**Request:**
```json
{
  "query": "Your query",
  "top_k": 5
}
```
    '''.strip()
    
    print("API Documentation:")
    print(f"{documentation[:150]}...\n")
    
    # Chunk for search/retrieval
    chunker = MarkdownChunker(max_chunk_size=400)
    doc_chunks = chunker.chunk(documentation)
    
    print(f"Documentation chunked into {len(doc_chunks)} searchable sections:\n")
    
    # Simulate search indexing
    for i, chunk in enumerate(doc_chunks, 1):
        lines = chunk.strip().split('\n')
        heading = next((l for l in lines if l.startswith('#')), 'No heading')
        
        # Extract metadata
        has_code = '```' in chunk
        has_request = 'Request:' in chunk
        has_response = 'Response:' in chunk
        
        print(f"Search Index Entry {i}:")
        print(f"  Section: {heading}")
        print(f"  Size: {len(chunk)} chars")
        print(f"  Contains code: {has_code}")
        print(f"  Contains request: {has_request}")
        print(f"  Contains response: {has_response}")
        print()
    
    # Simulate query
    query = "How do I authenticate API requests?"
    print(f"Query: '{query}'")
    print("\nMatching sections (keyword search for demo):")
    
    for i, chunk in enumerate(doc_chunks, 1):
        if 'authentication' in chunk.lower() or 'api key' in chunk.lower():
            lines = chunk.strip().split('\n')
            heading = next((l for l in lines if l.startswith('#')), 'No heading')
            print(f"  - {heading}")
            print(f"    {chunk[:100]}...")


def demonstrate_knowledge_base():
    """Show chunking for knowledge base construction."""
    print("\n" + "="*80)
    print("KNOWLEDGE BASE CONSTRUCTION")
    print("="*80)
    
    kb_articles = [
        '''
# Fine-Tuning Best Practices

Fine-tuning adapts models to specific tasks.

## Data Preparation

Quality over quantity. Use 50-100 high-quality examples minimum.

## Hyperparameters

- Learning rate: 1e-5 to 5e-5
- Batch size: 4-16
- Epochs: 3-5
        '''.strip(),
        
        '''
# Prompt Engineering Guide

Effective prompts improve LLM performance.

## Techniques

### Few-Shot Learning

Provide examples in the prompt.

### Chain-of-Thought

Ask model to explain reasoning.

### Role Assignment

Define a role for the model.
        '''.strip(),
        
        '''
# RAG Implementation

Build retrieval-augmented generation systems.

## Architecture

Three main components work together.

## Chunking Strategy

Use 500-1000 character chunks with 10% overlap.

## Embedding

Choose between OpenAI or open-source models.
        '''.strip()
    ]
    
    print(f"Knowledge base with {len(kb_articles)} articles\n")
    
    chunker = MarkdownChunker(max_chunk_size=250)
    
    all_chunks = []
    for article_id, article in enumerate(kb_articles, 1):
        chunks = chunker.chunk(article)
        
        print(f"Article {article_id}:")
        lines = article.split('\n')
        title = next((l for l in lines if l.startswith('# ')), 'Untitled')
        print(f"  Title: {title}")
        print(f"  Chunks: {len(chunks)}")
        
        for chunk_id, chunk in enumerate(chunks, 1):
            chunk_lines = chunk.strip().split('\n')
            chunk_heading = next((l for l in chunk_lines if l.startswith('#')), 'No heading')
            
            all_chunks.append({
                'article_id': article_id,
                'chunk_id': chunk_id,
                'heading': chunk_heading,
                'text': chunk,
                'size': len(chunk)
            })
        
        print()
    
    print(f"Total knowledge base: {len(all_chunks)} searchable chunks")
    print("\nChunk distribution:")
    for chunk in all_chunks:
        print(f"  Article {chunk['article_id']}, Chunk {chunk['chunk_id']}: {chunk['heading']} ({chunk['size']} chars)")


def main():
    """Run markdown chunking examples."""
    
    print("\n" + "="*80)
    print("MARKDOWN CHUNKING EXAMPLES")
    print("="*80)
    print("\nHeading-based document splitting for structured content.\n")
    
    demonstrate_basic_markdown_chunking()
    demonstrate_heading_hierarchy()
    demonstrate_readme_chunking()
    demonstrate_wiki_style()
    demonstrate_documentation_search()
    demonstrate_knowledge_base()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- MarkdownChunker splits documents by heading hierarchy
- Preserves document structure and semantic boundaries
- Perfect for documentation, READMEs, and wikis
- Handles code blocks and formatted content
- Maintains context within sections
- Ideal for knowledge base and documentation search
- Use max_chunk_size to control chunk granularity
- Great for building structured content retrieval systems
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
