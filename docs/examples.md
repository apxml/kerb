# Examples

Comprehensive examples demonstrating Kerb's capabilities across all modules.

## By Module

### Generation

Text generation examples with multiple providers.

- [01_basic_generation.py](examples/generation/01_basic_generation.py) - Basic text generation
- [02_streaming_generation.py](examples/generation/02_streaming_generation.py) - Streaming responses
- [03_batch_generation.py](examples/generation/03_batch_generation.py) - Batch processing
- [04_multi_provider_comparison.py](examples/generation/04_multi_provider_comparison.py) - Compare providers
- [05_cost_tracking.py](examples/generation/05_cost_tracking.py) - Track API costs
- [06_structured_output.py](examples/generation/06_structured_output.py) - Structured outputs
- [07_retry_and_error_handling.py](examples/generation/07_retry_and_error_handling.py) - Error handling
- [08_rate_limiting.py](examples/generation/08_rate_limiting.py) - Rate limiting
- [09_response_caching.py](examples/generation/09_response_caching.py) - Response caching
- [10_model_switching.py](examples/generation/10_model_switching.py) - Dynamic model switching

### Context Management

Context window and token management examples.

- [01_basic_context_window.py](examples/context/01_basic_context_window.py) - Basic context windows
- [02_truncation_strategies.py](examples/context/02_truncation_strategies.py) - Truncation strategies
- [03_priority_management.py](examples/context/03_priority_management.py) - Priority management
- [04_context_compression.py](examples/context/04_context_compression.py) - Context compression
- [05_sliding_windows.py](examples/context/05_sliding_windows.py) - Sliding window patterns
- [06_context_optimization.py](examples/context/06_context_optimization.py) - Context optimization
- [07_context_formatting.py](examples/context/07_context_formatting.py) - Context formatting
- [08_conversational_context.py](examples/context/08_conversational_context.py) - Conversational contexts
- [09_rag_context.py](examples/context/09_rag_context.py) - RAG context management

### Parsing

Output parsing and validation examples.

- [01_json_extraction.py](examples/parsing/01_json_extraction.py) - JSON extraction
- [02_pydantic_models.py](examples/parsing/02_pydantic_models.py) - Pydantic validation
- [03_function_calling.py](examples/parsing/03_function_calling.py) - Function calling
- [04_code_extraction.py](examples/parsing/04_code_extraction.py) - Code extraction
- [05_text_extraction.py](examples/parsing/05_text_extraction.py) - Text extraction
- [06_robust_parsing.py](examples/parsing/06_robust_parsing.py) - Robust parsing
- [07_validation.py](examples/parsing/07_validation.py) - Schema validation
- [08_extraction_pipeline.py](examples/parsing/08_extraction_pipeline.py) - Extraction pipelines

### Preprocessing

Text preprocessing and cleaning examples.

- [01_text_normalization.py](examples/preprocessing/01_text_normalization.py) - Text normalization
- [02_content_filtering.py](examples/preprocessing/02_content_filtering.py) - Content filtering
- [03_deduplication.py](examples/preprocessing/03_deduplication.py) - Deduplication
- [04_language_detection.py](examples/preprocessing/04_language_detection.py) - Language detection
- [05_batch_processing.py](examples/preprocessing/05_batch_processing.py) - Batch processing
- [06_content_analysis.py](examples/preprocessing/06_content_analysis.py) - Content analysis
- [07_transformations.py](examples/preprocessing/07_transformations.py) - Text transformations
- [08_production_pipelines.py](examples/preprocessing/08_production_pipelines.py) - Production pipelines

### Evaluation

Model evaluation and benchmarking examples.

- [01_ground_truth_metrics.py](examples/evaluation/01_ground_truth_metrics.py) - Ground truth metrics
- [02_quality_assessment.py](examples/evaluation/02_quality_assessment.py) - Quality assessment
- [03_llm_as_judge.py](examples/evaluation/03_llm_as_judge.py) - LLM-as-a-judge
- [04_benchmarking.py](examples/evaluation/04_benchmarking.py) - Benchmarking
- [05_ab_testing.py](examples/evaluation/05_ab_testing.py) - A/B testing
- [06_rag_evaluation.py](examples/evaluation/06_rag_evaluation.py) - RAG evaluation
- [07_model_comparison.py](examples/evaluation/07_model_comparison.py) - Model comparison

### Prompt Engineering

Prompt templates and optimization examples.

- [01_template_basics.py](examples/prompt/01_template_basics.py) - Template basics
- [04_optimization.py](examples/prompt/04_optimization.py) - Prompt optimization
- [05_advanced_patterns.py](examples/prompt/05_advanced_patterns.py) - Advanced patterns

### Retrieval

RAG and semantic search examples.

- [04_context_management.py](examples/retrieval/04_context_management.py) - Context management
- [05_rag_pipeline.py](examples/retrieval/05_rag_pipeline.py) - RAG pipeline
- [06_multi_query_retrieval.py](examples/retrieval/06_multi_query_retrieval.py) - Multi-query retrieval
- [08_result_formatting.py](examples/retrieval/08_result_formatting.py) - Result formatting

### Fine-Tuning

Model fine-tuning and dataset preparation examples.

See [Fine-Tuning Documentation](examples/fine_tuning/JSONL_QUICK_REFERENCE.md) for detailed guides:

- [JSONL Quick Reference](examples/fine_tuning/JSONL_QUICK_REFERENCE.md)
- [JSONL Enhancements](examples/fine_tuning/JSONL_ENHANCEMENTS.md)
- [JSONL Performance](examples/fine_tuning/JSONL_PERFORMANCE.md)

## Quick Start Examples

### Basic Generation

```python
from kerb.generation import generate, ModelName, LLMProvider

response = generate(
    "Explain quantum computing",
    model=ModelName.GPT_4O_MINI,
    provider=LLMProvider.OPENAI
)
print(response.content)
```

### RAG Pipeline

```python
from kerb.document import load_document
from kerb.chunk import chunk_text
from kerb.embedding import embed, embed_batch
from kerb.retrieval import semantic_search, Document
from kerb.generation import generate, ModelName

# Load and process
doc = load_document("paper.pdf")
chunks = chunk_text(doc.content, chunk_size=512)
embeddings = embed_batch(chunks)

# Search
query_embedding = embed("What are the findings?")
documents = [Document(content=c) for c in chunks]
results = semantic_search(query_embedding, documents, embeddings, top_k=5)

# Generate
context = "\n".join([r.document.content for r in results])
answer = generate(f"Based on: {context}\n\nQuestion: What are the findings?")
```

### Response Caching

```python
from kerb.cache import create_memory_cache, generate_prompt_key
from kerb.generation import generate, ModelName

cache = create_memory_cache(max_size=1000)

def cached_generate(prompt, model=ModelName.GPT_4O_MINI):
    key = generate_prompt_key(prompt, model=model.value)
    if cached := cache.get(key):
        return cached['response']
    response = generate(prompt, model=model)
    cache.set(key, {'response': response})
    return response
```

## Running Examples

All examples are self-contained and can be run directly:

```bash
# Set your API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Run any example
python docs/examples/generation/01_basic_generation.py

# Or use the run script
python scripts/run_examples.py --module generation
```

## Additional Resources

- [API Reference](api/index.rst) - Complete API documentation
- [Getting Started](getting-started.md) - Quick start guide
- [GitHub Examples](https://github.com/apxml/kerb/tree/main/docs/examples) - All example code
