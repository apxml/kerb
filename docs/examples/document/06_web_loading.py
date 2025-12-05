"""
Web Document Loading and HTML Processing
========================================

This example demonstrates loading and processing web documents for LLM applications.

Main concepts:
- Loading documents from URLs (simulation)
- HTML content extraction
- Web scraping for RAG systems
- Cleaning web content for embedding

Use cases:
- Building knowledge bases from web content
- Scraping documentation sites
- Processing HTML for RAG systems
- Extracting clean text from web pages

Note: URL loading requires the 'requests' package in production.
This example uses simulated content for demonstration.
"""

import tempfile
import os
from kerb.document import (
    load_document,
    load_html,
    extract_text_from_html,
    preprocess_html_text,
    clean_text,
    extract_urls,
    extract_document_stats,
    Document,
)


def create_sample_html_files(temp_dir: str):
    """Create sample HTML files simulating web content."""
    
    # Blog post HTML
    blog_html = """<!DOCTYPE html>

# %%
# Setup and Imports
# -----------------
<html>
<head>
    <title>Introduction to RAG Systems</title>
    <meta name="description" content="Learn about Retrieval-Augmented Generation">
    <script>
        // Analytics code
        console.log('Page view tracked');
    </script>
    <style>
        body { font-family: Arial; }
        .sidebar { display: none; }
    </style>
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/blog">Blog</a>
            <a href="/contact">Contact</a>
        </nav>
    </header>
    
    <main>
        <article>
            <h1>Introduction to RAG Systems</h1>
            
            <div class="metadata">
                <span>Published: 2024-01-15</span>
                <span>Author: AI Researcher</span>
            </div>
            
            <p>Retrieval-Augmented Generation (RAG) combines the power of 
            large language models with external knowledge retrieval. This 
            approach significantly improves accuracy and reduces hallucinations.</p>
            
            <h2>How RAG Works</h2>
            
            <p>The RAG system operates in two main phases:</p>
            
            <ol>
                <li><strong>Retrieval Phase:</strong> Relevant documents are 
                retrieved from a vector database based on semantic similarity.</li>
                <li><strong>Generation Phase:</strong> The LLM generates responses 
                using both the query and retrieved context.</li>
            </ol>
            
            <h2>Key Benefits</h2>
            
            <ul>
                <li>Access to up-to-date information</li>
                <li>Reduced hallucinations</li>
                <li>Cited sources for verification</li>
                <li>Domain-specific knowledge integration</li>
            </ul>
            
            <blockquote>
                "RAG systems represent a significant advancement in making 
                LLMs more reliable and verifiable." - Research Paper, 2023
            </blockquote>
            
            <p>For more information, visit 
            <a href="https://example.com/rag-guide">our comprehensive guide</a> 
            or contact us at info@example.com.</p>
        </article>
        
        <aside class="related">
            <h3>Related Articles</h3>
            <ul>
                <li><a href="/vector-databases">Vector Databases</a></li>
                <li><a href="/embeddings">Understanding Embeddings</a></li>
            </ul>
        </aside>
    </main>
    
    <footer>
        <div class="ads">
            <!-- Advertisement -->
            <script>displayAd();</script>
        </div>
        <p>&copy; 2024 AI Blog. All rights reserved.</p>
    </footer>
</body>
</html>"""
    
    # Documentation page HTML
    docs_html = """<!DOCTYPE html>
<html>
<head>
    <title>API Documentation - Document Loading</title>
</head>
<body>
    <div class="container">
        <nav class="sidebar">
            <ul>
                <li><a href="#intro">Introduction</a></li>
                <li><a href="#usage">Usage</a></li>
                <li><a href="#examples">Examples</a></li>
            </ul>
        </nav>
        
        <main class="content">
            <h1>Document Loading API</h1>
            
            <section id="intro">
                <h2>Introduction</h2>
                <p>The document loading API provides a unified interface for 
                loading various document formats including PDF, DOCX, and HTML.</p>
            </section>
            
            <section id="usage">
                <h2>Usage</h2>
                <p>Import the load_document function:</p>
                <pre><code>from kerb.document import load_document

doc = load_document("path/to/file.pdf")
print(doc.content)</code></pre>
            </section>
            
            <section id="examples">
                <h2>Examples</h2>
                <p>See our <a href="https://github.com/example/examples">
                GitHub repository</a> for more examples.</p>
            </section>
        </main>
    </div>
</body>
</html>"""
    
    # News article with noise
    news_html = """<!DOCTYPE html>
<html>
<body>
    <div class="ad-banner">Advertisement - Click here for deals!</div>
    
    <article>
        <h1>AI Breakthrough in Natural Language Processing</h1>
        
        <p class="byline">By Tech Reporter | January 20, 2024</p>
        
        <p>Researchers have announced a significant breakthrough in 
        natural language processing. The new model achieves state-of-the-art 
        performance on multiple benchmarks.</p>
        
        <div class="newsletter-signup">
            <h3>Subscribe to our newsletter!</h3>
            <input type="email" placeholder="your@email.com">
            <button>Subscribe</button>
        </div>
        
        <p>The research team, led by scientists at the AI Research Lab, 
        developed a novel architecture that improves both efficiency and 
        accuracy. Contact the team at research@ailab.org for more details.</p>
        
        <p>Full paper available at https://arxiv.org/abs/2024.12345</p>
    </article>
    
    <div class="comments">
        <h3>Comments (234)</h3>
        <div class="comment">User1: Great article!</div>
        <div class="comment">User2: Very interesting research.</div>
    </div>
</body>
</html>"""
    
    files = {}
    for name, content in [('blog', blog_html), ('docs', docs_html), ('news', news_html)]:
        filepath = os.path.join(temp_dir, f"{name}.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        files[name] = filepath
    
    return files


def main():
    """Run web document loading examples."""
    
    print("="*80)
    print("WEB DOCUMENT LOADING AND HTML PROCESSING")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        files = create_sample_html_files(temp_dir)
        
        # Example 1: Load and extract text from HTML
        print("\n1. BASIC HTML TEXT EXTRACTION")
        print("-" * 80)
        
        blog_doc = load_html(files['blog'])
        print(f"HTML file loaded: {len(blog_doc.content)} characters")
        print("\nExtracted content preview:")
        print(blog_doc.content[:300] + "...")
        
        # Example 2: Clean HTML extraction
        print("\n2. CLEAN HTML TEXT EXTRACTION")
        print("-" * 80)
        
        with open(files['blog'], 'r') as f:
            html_content = f.read()
        
        # Extract text with script removal
        clean_text_content = extract_text_from_html(html_content, remove_scripts=True)
        print(f"Clean text extracted: {len(clean_text_content)} characters")
        print("\nClean text preview:")
        print(clean_text_content[:300])
        
        # Example 3: Preprocess HTML for LLM consumption
        print("\n3. PREPROCESSING HTML FOR LLM")
        print("-" * 80)
        
        preprocessed = preprocess_html_text(html_content)
        print(f"Preprocessed text: {len(preprocessed)} characters")
        print("\nPreprocessed preview:")
        print(preprocessed[:300])
        
        stats = extract_document_stats(preprocessed)
        print(f"\nDocument statistics:")
        print(f"  Words: {stats['word_count']}")
        print(f"  Sentences: {stats['sentence_count']}")
        
        # Example 4: Extract metadata from HTML
        print("\n4. EXTRACTING METADATA FROM WEB CONTENT")
        print("-" * 80)
        
        urls = extract_urls(clean_text_content)
        print(f"URLs found: {len(urls)}")
        for url in urls[:3]:
            print(f"  - {url}")
        
        # Identify content type
        if 'documentation' in preprocessed.lower() or 'api' in preprocessed.lower():
            content_type = "documentation"
        elif 'published' in preprocessed.lower() or 'author' in preprocessed.lower():
            content_type = "blog_post"
        else:
            content_type = "article"
        
        print(f"\nContent type detected: {content_type}")
        
        # Example 5: Processing documentation HTML
        print("\n5. PROCESSING DOCUMENTATION HTML")
        print("-" * 80)
        
        docs_doc = load_html(files['docs'])
        with open(files['docs'], 'r') as f:
            docs_html = f.read()
        
        docs_text = preprocess_html_text(docs_html)
        print(f"Documentation extracted: {len(docs_text)} characters")
        print("\nContent preview:")
        print(docs_text[:250])
        
        # Extract code examples if present
        if 'import' in docs_text or 'def' in docs_text:
            print("\nCode examples detected in documentation")
        
        # Example 6: News article with noise removal
        print("\n6. CLEANING NOISY WEB CONTENT")
        print("-" * 80)
        
        with open(files['news'], 'r') as f:
            news_html = f.read()
        
        # Raw extraction
        raw_text = extract_text_from_html(news_html)
        print("Raw extraction:")
        print(raw_text[:200] + "...")
        
        # Clean text with URL and special char handling
        cleaned = clean_text(
            raw_text,
            normalize_whitespace=True,
            remove_urls=False  # Keep URLs for reference
        )
        print("\nCleaned text:")
        print(cleaned[:250] + "...")
        
        # Example 7: Create enriched documents for RAG
        print("\n7. CREATING ENRICHED DOCUMENTS FOR RAG")
        print("-" * 80)
        
        # Process blog post for RAG system
        blog_text = preprocess_html_text(html_content)
        blog_urls = extract_urls(blog_text)
        
        rag_document = Document(
            id="web_doc_001",
            content=blog_text,
            metadata={
                "source_type": "web",
                "content_type": "blog_post",
                "title": "Introduction to RAG Systems",
                "urls": blog_urls,
                "stats": extract_document_stats(blog_text),
                "domain": "example.com",
                "scraped_at": "2024-01-15",
            },
            source="https://example.com/blog/rag-intro"
        )
        
        print(f"RAG Document created:")
        print(f"  ID: {rag_document.id}")
        print(f"  Content type: {rag_document.metadata['content_type']}")
        print(f"  Word count: {rag_document.metadata['stats']['word_count']}")
        print(f"  URLs found: {len(rag_document.metadata['urls'])}")
        
        # Example 8: Batch web content processing
        print("\n8. BATCH WEB CONTENT PROCESSING")
        print("-" * 80)
        
        web_documents = []
        
        for name, filepath in files.items():
            with open(filepath, 'r') as f:
                html = f.read()
            
            text = preprocess_html_text(html)
            stats = extract_document_stats(text)
            
            doc = Document(
                id=f"web_{name}",
                content=text,
                metadata={
                    "source_type": "web",
                    "filename": name,
                    "stats": stats,
                    "urls": extract_urls(text),
                },
                source=filepath
            )
            web_documents.append(doc)
        
        print(f"Processed {len(web_documents)} web documents")
        print("\nDocument summary:")
        for doc in web_documents:
            print(f"  {doc.id}:")
            print(f"    Words: {doc.metadata['stats']['word_count']}")
            print(f"    URLs: {len(doc.metadata['urls'])}")
        
        # Example 9: Best practices for web scraping
        print("\n9. WEB SCRAPING BEST PRACTICES")
        print("-" * 80)
        
        print("Preprocessing pipeline for web content:")
        print("  1. Load HTML document")
        print("  2. Extract text (remove scripts/styles)")
        print("  3. Preprocess for LLM (normalize whitespace)")
        print("  4. Extract metadata (URLs, dates)")
        print("  5. Identify content type")
        print("  6. Create enriched Document object")
        print("  7. Store for vector database ingestion")
        
        print("\nContent quality checks:")
        sample_doc = web_documents[0]
        word_count = sample_doc.metadata['stats']['word_count']
        
        if word_count < 50:
            print("  - Warning: Very short content (< 50 words)")
        elif word_count < 200:
            print("  - Notice: Short content (< 200 words)")
        else:
            print("  - Good: Sufficient content length")
        
        if len(sample_doc.metadata['urls']) > 10:
            print("  - Warning: High URL density (possible spam)")
        else:
            print("  - Good: Reasonable URL count")
        
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Remove scripts and styles before text extraction")
    print("- Preprocess HTML to normalize whitespace")
    print("- Extract and preserve important URLs")
    print("- Identify content type for better retrieval")
    print("- Enrich documents with web-specific metadata")
    print("- Implement quality checks for scraped content")
    print("- Use consistent processing pipeline")
    print("- In production: use load_from_url() with requests library")


if __name__ == "__main__":
    main()
