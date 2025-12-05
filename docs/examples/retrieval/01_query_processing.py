"""
Query Processing Example
========================

This example demonstrates query processing techniques for RAG systems.

Main concepts:
- Rewriting queries for better retrieval
- Expanding queries into variations
- Breaking down complex queries into sub-queries
- Optimizing query formulation for different search methods
"""

from kerb.retrieval import (
    rewrite_query,
    expand_query,
    generate_sub_queries
)


def main():
    """Run query processing examples."""
    
    print("="*80)
    print("QUERY PROCESSING FOR RAG SYSTEMS")
    print("="*80)
    
    # 1. Query Rewriting
    print("\n1. QUERY REWRITING")
    print("-"*80)
    print("Transform user queries into optimized forms for retrieval.\n")
    
    user_query = "how do I use async in python?"
    print(f"Original query: '{user_query}'\n")
    
    # Different rewriting styles
    styles = {
        "clear": "Remove filler words, simplify",
        "detailed": "Add context and specificity",
        "keyword": "Extract key terms only",
        "concise": "Make more compact",
        "natural": "Convert to natural question"
    }
    
    for style, description in styles.items():
        rewritten = rewrite_query(user_query, style=style)
        print(f"  {style.upper():12} | {description}")
        print(f"               -> '{rewritten}'")
    
    
    # 2. Query Expansion
    print("\n\n2. QUERY EXPANSION")
    print("-"*80)
    print("Generate multiple query variations for broader retrieval coverage.\n")
    
    query = "machine learning models"
    print(f"Original query: '{query}'\n")
    
    # Expand with synonyms
    print("Synonym expansion:")
    expanded = expand_query(query, method="synonyms")
    for i, variant in enumerate(expanded[:5], 1):
        print(f"  {i}. {variant}")
    
    # Expand with related terms
    print("\nRelated terms expansion:")
    expanded = expand_query(query, method="related")
    for i, variant in enumerate(expanded[:5], 1):
        print(f"  {i}. {variant}")
    
    # Expand with specificity levels
    print("\nSpecificity expansion:")
    expanded = expand_query(query, method="specificity")
    for i, variant in enumerate(expanded[:5], 1):
        print(f"  {i}. {variant}")
    
    
    # 3. Sub-Query Generation
    print("\n\n3. SUB-QUERY GENERATION")
    print("-"*80)
    print("Break complex queries into simpler sub-queries.\n")
    
    complex_query = "Compare supervised and unsupervised learning methods and their applications"
    print(f"Complex query:\n  '{complex_query}'\n")
    
    print("Generated sub-queries:")
    sub_queries = generate_sub_queries(complex_query, max_queries=4)
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    
    # 4. RAG-Specific Query Optimization
    print("\n\n4. RAG-SPECIFIC QUERY OPTIMIZATION")
    print("-"*80)
    print("Optimize queries specifically for retrieval-augmented generation.\n")
    
    # Conversational query
    conversational = "tell me about transformers"
    print(f"Conversational: '{conversational}'")
    print(f"  -> Keyword:   '{rewrite_query(conversational, style='keyword')}'")
    print(f"  -> Detailed:  '{rewrite_query(conversational, style='detailed')}'")
    
    # Technical query
    technical = "API endpoints REST authentication"
    print(f"\nTechnical: '{technical}'")
    print(f"  -> Natural:   '{rewrite_query(technical, style='natural')}'")
    print(f"  -> Clear:     '{rewrite_query(technical, style='clear')}'")
    
    # Ambiguous query
    ambiguous = "python performance"
    print(f"\nAmbiguous: '{ambiguous}'")
    print("  Expanded variations for context:")
    variations = expand_query(ambiguous, method="specificity")
    for i, var in enumerate(variations[:4], 1):
        print(f"    {i}. {var}")
    
    
    # 5. Multi-Stage Query Processing
    print("\n\n5. MULTI-STAGE QUERY PROCESSING")
    print("-"*80)
    print("Combine multiple techniques for optimal retrieval.\n")
    
    user_input = "what's the best way to handle errors in async code?"
    print(f"User input: '{user_input}'\n")
    
    # Stage 1: Rewrite for clarity
    stage1 = rewrite_query(user_input, style="clear")
    print(f"Stage 1 (Clear):    '{stage1}'")
    
    # Stage 2: Extract keywords
    stage2 = rewrite_query(stage1, style="keyword")
    print(f"Stage 2 (Keywords): '{stage2}'")
    
    # Stage 3: Expand
    stage3 = expand_query(stage2, method="related")
    print(f"Stage 3 (Expand):   {len(stage3)} variations")
    for i, var in enumerate(stage3[:3], 1):
        print(f"  {i}. {var}")
    
    # Stage 4: Generate sub-queries for comprehensive coverage
    sub_q = generate_sub_queries(stage1, max_queries=3)
    print(f"\nStage 4 (Sub-queries): {len(sub_q)} focused queries")
    for i, sq in enumerate(sub_q, 1):
        print(f"  {i}. {sq}")
    
    
    # 6. Use Case: Query Optimization Pipeline
    print("\n\n6. QUERY OPTIMIZATION PIPELINE")
    print("-"*80)
    print("A complete pipeline for production RAG systems.\n")
    
    def optimize_query_for_rag(user_query: str) -> dict:
        """Optimize a query for RAG retrieval."""

# %%
# Setup and Imports
# -----------------
        return {
            'original': user_query,
            'primary': rewrite_query(user_query, style="clear"),
            'keywords': rewrite_query(user_query, style="keyword"),
            'expanded': expand_query(user_query, method="synonyms")[:3],
            'sub_queries': generate_sub_queries(user_query, max_queries=2)
        }
    
    test_query = "how to scale microservices with kubernetes"
    optimized = optimize_query_for_rag(test_query)
    
    print(f"Original:      {optimized['original']}")
    print(f"Primary:       {optimized['primary']}")
    print(f"Keywords:      {optimized['keywords']}")
    print(f"Expanded:      {len(optimized['expanded'])} variants")
    for variant in optimized['expanded']:
        print(f"               - {variant}")
    print(f"Sub-queries:   {len(optimized['sub_queries'])} queries")
    for sq in optimized['sub_queries']:
        print(f"               - {sq}")
    
    
    print("\n" + "="*80)
    print("Query processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
