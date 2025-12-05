"""
Text and Markdown Extraction
============================

This example demonstrates extracting structured content from LLM text outputs,
including XML tags, markdown sections, lists, and tables.

Main concepts:
- Extracting content from XML-style tags
- Parsing markdown sections by heading
- Extracting list items from markdown
- Parsing markdown tables
"""

from kerb.parsing import (
    extract_xml_tag,
    extract_markdown_sections,
    extract_list_items,
    parse_markdown_table
)


def simulate_llm_text_response(response_type: str) -> str:
    """Simulate LLM text responses with various structures."""
    
    if response_type == "xml_tags":
        return """Let me analyze the document:

# %%
# Setup and Imports
# -----------------

<summary>
This document discusses the implementation of a new authentication system
using OAuth 2.0 and JWT tokens for secure user access.
</summary>

<key_points>
- OAuth 2.0 for authorization
- JWT tokens for stateless authentication
- Refresh token rotation for security
- Multi-factor authentication support
</key_points>

<recommendation>
Implement the OAuth 2.0 flow with PKCE extension for enhanced security,
and use short-lived access tokens (15 minutes) with longer-lived refresh
tokens (7 days) that rotate on each use.
</recommendation>

<next_steps>
1. Set up OAuth provider configuration
2. Implement token generation and validation
3. Add refresh token rotation logic
4. Integrate MFA support
</next_steps>"""
    
    elif response_type == "markdown_sections":
        return """# Project Analysis Report

## Executive Summary

The project is progressing well with 85% completion. Key milestones have been
achieved and the team is on track for the Q4 delivery deadline.

## Technical Details

The architecture follows a microservices pattern with the following components:
- API Gateway for request routing
- Authentication service for user management
- Data processing service for ETL operations
- Storage service for file management

## Risk Assessment

### High Priority Risks

Resource constraints may impact timeline. The team needs two additional
developers to maintain the current velocity.

### Medium Priority Risks

Third-party API dependencies could cause integration delays. Recommend
implementing fallback mechanisms.

## Recommendations

Based on the analysis, we recommend:
1. Hiring additional developers immediately
2. Implementing API fallback strategies
3. Increasing test coverage to 90%
4. Setting up automated deployment pipeline"""
    
    elif response_type == "lists":
        return """Here are the requirements for the project:

**Functional Requirements:**
- User registration and authentication
- Profile management with photo upload
- Real-time messaging between users
- Notification system for important events
- Search functionality across all content

**Non-Functional Requirements:**
1. System must handle 10,000 concurrent users
2. API response time under 200ms for 95th percentile
3. 99.9% uptime SLA
4. Data encrypted at rest and in transit
5. GDPR and CCPA compliance

**Technical Stack:**
* Frontend: React with TypeScript
* Backend: Node.js with Express
* Database: PostgreSQL with Redis cache
* Deployment: Kubernetes on AWS
* Monitoring: Prometheus and Grafana"""
    
    elif response_type == "table":
        return """Here's the performance comparison:

| Model | Accuracy | Latency (ms) | Cost ($) | Memory (GB) |
|-------|----------|--------------|----------|-------------|
| GPT-4 | 0.95 | 1200 | 0.03 | 8 |
| GPT-3.5 | 0.87 | 400 | 0.002 | 4 |
| Claude-2 | 0.92 | 800 | 0.015 | 6 |
| Llama-2-70B | 0.88 | 600 | 0.001 | 12 |
| Mistral-7B | 0.83 | 200 | 0.0005 | 3 |

Based on these metrics, choose the model that best fits your requirements
for accuracy, latency, and cost."""
    
    return "No content available."


def main():
    """Run text and markdown extraction examples."""
    
    print("="*80)
    print("TEXT AND MARKDOWN EXTRACTION")
    print("="*80)
    
    # Example 1: Extract XML-style tags
    print("\nExample 1: Extract XML-Style Tags")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("xml_tags")
    print(f"LLM Output:\n{llm_output[:200]}...\n")
    
    # Extract summary
    summaries = extract_xml_tag(llm_output, "summary")
    print("Extracted Summary:")
    for summary in summaries:
        print(f"  {summary}")
    
    # Extract recommendations
    recommendations = extract_xml_tag(llm_output, "recommendation")
    print("\nExtracted Recommendation:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Example 2: Extract multiple tag types
    print("\n\nExample 2: Extract Multiple Tag Types")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("xml_tags")
    
    tags_to_extract = ["summary", "key_points", "recommendation", "next_steps"]
    
    for tag in tags_to_extract:
        content = extract_xml_tag(llm_output, tag)
        if content:
            print(f"\n<{tag}>:")
            print(f"  {content[0][:100]}..." if len(content[0]) > 100 else f"  {content[0]}")
    
    # Example 3: Extract markdown sections
    print("\n\nExample 3: Extract Markdown Sections (H2)")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("markdown_sections")
    
    sections = extract_markdown_sections(llm_output, heading_level=2)
    
    print(f"Found {len(sections)} section(s):\n")
    for heading, content in sections.items():
        print(f"## {heading}")
        print(f"  {content[:150]}..." if len(content) > 150 else f"  {content}")
        print()
    
    # Example 4: Extract subsections (H3)
    print("\n\nExample 4: Extract Subsections (H3)")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("markdown_sections")
    
    subsections = extract_markdown_sections(llm_output, heading_level=3)
    
    if subsections:
        print(f"Found {len(subsections)} subsection(s):\n")
        for heading, content in subsections.items():
            print(f"### {heading}")
            print(f"  {content.strip()}")
            print()
    else:
        print("No H3 subsections found")
    
    # Example 5: Extract unordered list items
    print("\n\nExample 5: Extract Unordered List Items")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("lists")
    
    items = extract_list_items(llm_output, ordered=False)
    
    print(f"Found {len(items)} unordered list item(s):\n")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    
    # Example 6: Extract ordered list items
    print("\n\nExample 6: Extract Ordered List Items")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("lists")
    
    items = extract_list_items(llm_output, ordered=True)
    
    print(f"Found {len(items)} ordered list item(s):\n")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    
    # Example 7: Parse markdown table
    print("\n\nExample 7: Parse Markdown Table")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("table")
    print(f"LLM Output:\n{llm_output[:300]}...\n")
    
    table_data = parse_markdown_table(llm_output)
    
    if table_data:
        print(f"Parsed {len(table_data)} row(s):\n")
        
        # Display headers
        headers = table_data[0].keys()
        print(" | ".join(headers))
        print("-" * 60)
        
        # Display rows
        for row in table_data:
            print(" | ".join(str(row.get(h, "")) for h in headers))
    
    # Example 8: Analyze table data
    print("\n\nExample 8: Analyze Parsed Table Data")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("table")
    table_data = parse_markdown_table(llm_output)
    
    if table_data:
        # Find model with best accuracy
        best_accuracy = max(table_data, key=lambda x: float(x.get('Accuracy', 0)))
        print(f"Best Accuracy: {best_accuracy.get('Model')} ({best_accuracy.get('Accuracy')})")
        
        # Find fastest model
        fastest = min(table_data, key=lambda x: float(x.get('Latency (ms)', float('inf'))))
        print(f"Fastest: {fastest.get('Model')} ({fastest.get('Latency (ms)')} ms)")
        
        # Find cheapest model
        cheapest = min(table_data, key=lambda x: float(x.get('Cost ($)', float('inf'))))
        print(f"Cheapest: {cheapest.get('Model')} (${cheapest.get('Cost ($)')})")
    
    # Example 9: Practical use case - structured data extraction
    print("\n\nExample 9: Practical Use Case - Extract Analysis Report")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("xml_tags")
    
    # Extract structured data
    summary = extract_xml_tag(llm_output, "summary")
    key_points = extract_xml_tag(llm_output, "key_points")
    recommendation = extract_xml_tag(llm_output, "recommendation")
    next_steps = extract_xml_tag(llm_output, "next_steps")
    
    # Structure it
    analysis_report = {
        "summary": summary[0] if summary else None,
        "key_points": key_points[0] if key_points else None,
        "recommendation": recommendation[0] if recommendation else None,
        "next_steps": next_steps[0] if next_steps else None
    }
    
    print("Structured Analysis Report:")
    for key, value in analysis_report.items():
        if value:
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  {value[:100]}..." if len(value) > 100 else f"  {value}")
    
    # Example 10: Combined extraction
    print("\n\nExample 10: Combined Extraction Workflow")
    print("-"*80)
    
    llm_output = simulate_llm_text_response("markdown_sections")
    
    # Extract main sections
    sections = extract_markdown_sections(llm_output, heading_level=2)
    
    # Extract lists from specific section
    if "Recommendations" in sections:
        rec_text = sections["Recommendations"]
        rec_items = extract_list_items(rec_text, ordered=True)
        
        print("Recommendations:")
        for i, item in enumerate(rec_items, 1):
            print(f"  {i}. {item}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
