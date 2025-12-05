# Getting Started

This guide will help you get started with Kerb, the complete toolkit for building LLM applications.

## Installation

Kerb is modular - install only what you need:

```bash
# Install everything
pip install kerb[all]

# Or install specific modules
pip install kerb[generation]
pip install kerb[embeddings]
pip install kerb[evaluation]

# Install multiple modules
pip install kerb[generation,embeddings,retrieval]
```

### Available Extras

- `tokenizers` - Token counting and text splitting
- `embeddings` - Embedding generation with multiple providers
- `generation` - LLM generation (OpenAI, Anthropic, Gemini, Cohere)
- `evaluation` - Metrics and benchmarking (BLEU, ROUGE, BERTScore)
- `documents` - Document loading (PDF, DOCX, web pages)
- `preprocessing` - Text cleaning and language detection
- `fine_tuning` - Model fine-tuning utilities
- `config` - Secure configuration management
- `all` - Everything included

## Quick Examples

### Basic Generation

```python
from kerb.generation import generate, ModelName, LLMProvider

# Generate with OpenAI
response = generate(
    "Explain quantum computing in simple terms",
    model=ModelName.GPT_4O_MINI,
    provider=LLMProvider.OPENAI
)

print(f"Response: {response.content}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.cost:.6f}")
```

### Streaming Responses

```python
from kerb.generation import generate_stream, ModelName, LLMProvider

# Stream responses for better UX
for chunk in generate_stream(
    "Write a short story about a robot",
    model=ModelName.GPT_4O,
    provider=LLMProvider.OPENAI
):
    print(chunk.content, end='', flush=True)
```

### Multi-Provider Support

Switch between providers with a single parameter change:

```python
from kerb.generation import generate, ModelName, LLMProvider

# Use OpenAI
response = generate(
    "Translate 'Hello' to Spanish",
    model=ModelName.GPT_4O_MINI,
    provider=LLMProvider.OPENAI
)

# Use Anthropic
response = generate(
    "Translate 'Hello' to Spanish",
    model=ModelName.CLAUDE_35_SONNET,
    provider=LLMProvider.ANTHROPIC
)

# Use Gemini
response = generate(
    "Translate 'Hello' to Spanish",
    model=ModelName.GEMINI_15_FLASH,
    provider=LLMProvider.GEMINI
)
```

### RAG Pipeline

```python
from kerb.document import load_document
from kerb.chunk import chunk_text
from kerb.embedding import embed, embed_batch
from kerb.retrieval import semantic_search, Document
from kerb.generation import generate, ModelName

# Load and process document
doc = load_document("paper.pdf")
chunks = chunk_text(doc.content, chunk_size=512, overlap=50)

# Create embeddings
chunk_embeddings = embed_batch(chunks)

# Search for relevant chunks
query = "What are the main findings?"
query_embedding = embed(query)
documents = [Document(content=c) for c in chunks]
results = semantic_search(
    query_embedding=query_embedding,
    documents=documents,
    document_embeddings=chunk_embeddings,
    top_k=5
)

# Generate answer with context
context = "\n".join([r.document.content for r in results])
answer = generate(
    f"Based on: {context}\n\nQuestion: {query}",
    model=ModelName.GPT_4O_MINI
)
```

### Response Caching

```python
from kerb.cache import create_memory_cache, generate_prompt_key
from kerb.generation import generate, ModelName

cache = create_memory_cache(max_size=1000, default_ttl=3600)

def cached_generate(prompt, model=ModelName.GPT_4O_MINI):
    cache_key = generate_prompt_key(prompt, model=model.value)
    
    if cached := cache.get(cache_key):
        return cached['response']
    
    response = generate(prompt, model=model)
    cache.set(cache_key, {'response': response})
    return response

# First call - hits API
response1 = cached_generate("Explain Python decorators")

# Second call - hits cache
response2 = cached_generate("Explain Python decorators")
```

### Agent Workflows

```python
from kerb.agent.patterns import ReActAgent

def llm_function(prompt: str) -> str:
    """Your LLM function"""
    from kerb.generation import generate, ModelName
    response = generate(prompt, model=ModelName.GPT_4O)
    return response.content

# Create a ReAct agent
agent = ReActAgent(
    name="ResearchAgent",
    llm_func=llm_function,
    max_iterations=5
)

# Execute multi-step task
result = agent.run("Research AI trends and summarize findings")

print(f"Status: {result.status.value}")
print(f"Output: {result.output}")
print(f"Steps: {len(result.steps)}")
```

### Evaluation Metrics

```python
from kerb.evaluation import (
    calculate_bleu,
    calculate_rouge,
    calculate_f1_score
)

reference = "The quick brown fox jumps over the lazy dog"
candidate = "The fast brown fox jumps over the sleepy dog"

# Calculate metrics
bleu_score = calculate_bleu(candidate, reference)
rouge_scores = calculate_rouge(candidate, reference, rouge_type="rouge-l")
f1 = calculate_f1_score(candidate, reference)

print(f"BLEU: {bleu_score:.3f}")
print(f"ROUGE-L F1: {rouge_scores['fmeasure']:.3f}")
print(f"F1 Score: {f1:.3f}")
```

## Configuration

### API Keys

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-gemini-key"
export COHERE_API_KEY="your-cohere-key"
```

Or use Kerb's config management:

```python
from kerb.config import set_api_key, get_api_key

# Store securely
set_api_key("openai", "your-api-key")

# Retrieve
api_key = get_api_key("openai")
```

## Next Steps

- Browse [Examples](examples.md) for more use cases
- Check the [API Reference](api/index.rst) for detailed documentation
- Explore individual [Modules](modules.rst) for specific features

## Getting Help

- **Documentation**: [https://kerb.readthedocs.io](https://kerb.readthedocs.io)
- **GitHub Issues**: [https://github.com/apxml/kerb/issues](https://github.com/apxml/kerb/issues)
- **Examples**: [https://github.com/apxml/kerb/tree/main/docs/examples](https://github.com/apxml/kerb/tree/main/docs/examples)
