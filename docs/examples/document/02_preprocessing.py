"""Document Preprocessing for LLM Consumption

This example demonstrates text preprocessing and cleaning for LLM applications.

Main concepts:
- Text normalization and cleaning
- Format-specific preprocessing
- Preparing documents for embedding
- Handling noisy text from various sources

Use cases:
- Cleaning web-scraped content for RAG
- Normalizing documents before vectorization
- Preprocessing user-uploaded documents
- Standardizing text for consistent embeddings
"""

import tempfile
import os
from kerb.document import (
    load_document,
    clean_text,
    remove_extra_newlines,
    preprocess_pdf_text,
    preprocess_html_text,
    preprocess_markdown,
    extract_text_from_html,
    strip_markdown,
)


def create_noisy_documents(temp_dir: str):
    """Create sample documents with noise/formatting issues."""
    
    # Simulated PDF text with formatting artifacts
    pdf_text = """This is a sam-
ple document extracted
from a PDF    file.
It has   extra    spaces and
broken    lines    that need
clean-
ing."""
    
    pdf_file = os.path.join(temp_dir, "noisy_pdf.txt")
    with open(pdf_file, 'w') as f:
        f.write(pdf_text)
    
    # HTML with tags and noise
    html_content = """
    <html>
        <head><title>Sample Page</title></head>
        <body>
            <script>console.log('ads');</script>
            <h1>Important Information</h1>
            <p>This is &nbsp; content with   HTML entities.</p>
            <p>Contact us at <a href="mailto:info@example.com">info@example.com</a></p>
            <div class="ads">Advertisement here</div>
            <p>Visit our site at https://example.com for more details.</p>
        </body>
    </html>
    """
    
    html_file = os.path.join(temp_dir, "noisy_html.html")
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Markdown with formatting
    markdown_content = """# Main Title

## Section 1

This is **bold** and *italic* text with `code` snippets.

Here's a [link](https://example.com) and an image ![alt](img.png).

```python
def sample_code():
    pass
```

> This is a blockquote

- List item 1
- List item 2
"""
    
    markdown_file = os.path.join(temp_dir, "formatted.md")
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    
    return {
        'pdf': pdf_file,
        'html': html_file,
        'markdown': markdown_file,
        'html_content': html_content
    }


def main():
    """Run document preprocessing examples."""
    
    print("="*80)
    print("DOCUMENT PREPROCESSING FOR LLM APPLICATIONS")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        files = create_noisy_documents(temp_dir)
        
        # Example 1: Clean noisy PDF text
        print("\n1. CLEANING PDF TEXT")
        print("-" * 80)
        
        pdf_doc = load_document(files['pdf'])
        print("Original PDF text:")
        print(repr(pdf_doc.content[:100]))
        
        cleaned_pdf = preprocess_pdf_text(pdf_doc.content)
        print("\nCleaned PDF text:")
        print(cleaned_pdf)
        
        # Example 2: Extract and clean HTML
        print("\n2. EXTRACTING CLEAN TEXT FROM HTML")
        print("-" * 80)
        
        # Direct HTML text extraction
        html_text = extract_text_from_html(files['html_content'], remove_scripts=True)
        print("Extracted HTML text:")
        print(html_text[:150])
        
        # Further preprocessing
        clean_html = preprocess_html_text(files['html_content'])
        print("\nPreprocessed HTML:")
        print(clean_html[:150])
        
        # Example 3: Process Markdown
        print("\n3. PROCESSING MARKDOWN")
        print("-" * 80)
        
        md_doc = load_document(files['markdown'])
        print("Original Markdown (first 150 chars):")
        print(md_doc.content[:150])
        
        # Keep structure for context-aware processing
        processed_md = preprocess_markdown(md_doc.content, keep_structure=True)
        print("\nWith structure preserved:")
        print(processed_md[:150])
        
        # Strip all formatting for pure text
        stripped_md = strip_markdown(md_doc.content)
        print("\nAll formatting stripped:")
        print(stripped_md[:150])
        
        # Example 4: General text cleaning
        print("\n4. GENERAL TEXT CLEANING")
        print("-" * 80)
        
        noisy_text = """
        Check   out our  website at https://example.com
        Contact: info@example.com  or  support@example.com
        
        
        
        Follow us on Twitter!!! @example #hashtag
        Price: $99.99 (limited time offer!!!)
        """
        
        print("Original text:")
        print(repr(noisy_text[:100]))
        
        # Light cleaning - normalize whitespace only
        lightly_cleaned = clean_text(noisy_text, normalize_whitespace=True)
        print("\nLightly cleaned (whitespace normalized):")
        print(lightly_cleaned[:150])
        
        # Moderate cleaning - remove URLs
        moderately_cleaned = clean_text(
            noisy_text,
            normalize_whitespace=True,
            remove_urls=True
        )
        print("\nModerately cleaned (URLs removed):")
        print(moderately_cleaned[:150])
        
        # Aggressive cleaning - remove URLs, emails, and special chars
        aggressively_cleaned = clean_text(
            noisy_text,
            normalize_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            remove_special_chars=True
        )
        print("\nAggressively cleaned:")
        print(aggressively_cleaned[:150])
        
        # Example 5: Remove excessive newlines
        print("\n5. HANDLING EXCESSIVE WHITESPACE")
        print("-" * 80)
        
        text_with_newlines = "Paragraph 1\n\n\n\n\n\nParagraph 2\n\n\n\nParagraph 3"
        print("Original:")
        print(repr(text_with_newlines))
        
        cleaned_newlines = remove_extra_newlines(text_with_newlines, max_consecutive=2)
        print("\nCleaned (max 2 consecutive newlines):")
        print(repr(cleaned_newlines))
        
        # Example 6: Pipeline for LLM preprocessing
        print("\n6. COMPLETE PREPROCESSING PIPELINE")
        print("-" * 80)
        
        raw_text = """
        # Article Title
        
        Visit https://example.com for more info.
        
        
        
        This is   the main    content with    **formatting**.
        Contact us at contact@example.com!!!
        
        > Some quoted text
        
        - Bullet point 1
        - Bullet point 2
        """
        
        print("Raw text (first 100 chars):")
        print(repr(raw_text[:100]))
        
        # Step 1: Strip markdown
        step1 = strip_markdown(raw_text)
        print("\nStep 1 - Markdown stripped:")
        print(step1[:100])
        
        # Step 2: Clean text
        step2 = clean_text(
            step1,
            normalize_whitespace=True,
            remove_urls=True,
            remove_emails=True
        )
        print("\nStep 2 - Text cleaned:")
        print(step2[:100])
        
        # Step 3: Remove excessive newlines
        step3 = remove_extra_newlines(step2, max_consecutive=2)
        print("\nStep 3 - Newlines normalized:")
        print(repr(step3[:100]))
        
        print("\nFinal result ready for embedding:")
        print(step3)
        
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Use format-specific preprocessors for PDF, HTML, Markdown")
    print("- Apply progressive cleaning based on quality needs")
    print("- Normalize whitespace before embedding generation")
    print("- Consider whether to keep URLs/emails for context")
    print("- Pipeline: format-specific -> general cleaning -> normalization")
    print("- Consistent preprocessing improves embedding quality")


if __name__ == "__main__":
    main()
