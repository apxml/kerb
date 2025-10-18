# Kerb

The complete toolkit for developers building LLM applications.

Built to drive production ML systems at ApX Machine Learning (apxml.com), now available as open source.

## Overview 

### Simple

Advanced LLM techniques made simple. Clean, easy-to-use interfaces for complex operations.

### Lightweight

Only install what you need. Kerb is modular, no unnecessary dependencies.

### Compatible

Works with any LLM project. Kerb is a toolkit, not a framework. Use it alongside your existing stack.

## Installation

```bash
# Install everything
pip install kerb[all]

# Or install specific modules
pip install kerb[generation] kerb[embeddings] kerb[evaluation]
```

## Quick Start

```python
from kerb.generation import generate, ModelName, LLMProvider
from kerb.prompt import render_template

# Generate with any provider, easy config change.
response = generate(
    "Explain quantum computing",
    model=ModelName.GPT_4O_MINI,
    provider=LLMProvider.OPENAI
)

print(f"Response: {response.content}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.cost:.6f}")
```

## Modules

Everything you need to build LLM applications.

| Module | Description |
|--------|-------------|
| **Agent** | Agent orchestration and execution patterns for multi-step reasoning. |
| **Cache** | Response and embedding caching to reduce costs and latency. |
| **Chunk** | Text chunking utilities for optimal context windows and retrieval. |
| **Config** | Configuration management for models, providers, and application settings. |
| **Context** | Context window management and token budget tracking. |
| **Document** | Document loading and processing for PDFs, web pages, and more. |
| **Embedding** | Embedding generation and similarity search helpers. |
| **Evaluation** | Metrics and benchmarking tools for LLM outputs. |
| **Fine-Tuning** | Model fine-tuning utilities and large dataset preparation. |
| **Generation** | Unified LLM generation with multi-provider support (OpenAI, Anthropic, Gemini). |
| **Memory** | Conversation memory and entity tracking for stateful applications. |
| **Multimodal** | Image, audio, and video processing for multimodal models. |
| **Parsing** | Output parsing and validation (JSON, structured data, function calls). |
| **Preprocessing** | Text cleaning and preprocessing for LLM inputs. |
| **Prompt** | Prompt engineering utilities, templates, and chain-of-thought patterns. |
| **Retrieval** | RAG and vector search utilities for semantic retrieval. |
| **Safety** | Content moderation and safety filters. |
| **Testing** | Testing utilities for LLM outputs and evaluation. |
| **Tokenizer** | Token counting and text splitting for any model. |

## Project Structure

```
kerb/
├── core/           # Shared types and interfaces
├── agent/          # Agent systems and reasoning
├── cache/          # Caching mechanisms
├── chunk/          # Text chunking utilities
├── config/         # Configuration management
├── context/        # Context window management
├── document/       # Document loading
├── embedding/      # Embedding generation
├── evaluation/     # Evaluation metrics
├── fine_tuning/    # Model fine-tuning
├── generation/     # LLM text generation
├── memory/         # Memory systems
├── multimodal/     # Multimodal processing
├── parsing/        # Output parsing
├── preprocessing/  # Text preprocessing
├── prompt/         # Prompt management
├── retrieval/      # RAG and retrieval
├── safety/         # Content safety
├── testing/        # Testing utilities
└── tokenizer/      # Token counting
```

## Examples

### RAG Pipeline
```python
from kerb.document import load_document
from kerb.chunk import chunk_text
from kerb.embedding import embed, embed_batch
from kerb.retrieval import semantic_search, Document
from kerb.generation import generate, ModelName, LLMProvider

# Load and process document
doc = load_document("paper.pdf")
chunks = chunk_text(doc.content, chunk_size=512, overlap=50)

# Create embeddings
chunk_embeddings = embed_batch(chunks)

# Search for relevant chunks
query = "main findings"
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
    f"Based on: {context}\n\nQuestion: What are the main findings?",
    model=ModelName.GPT_4O_MINI,
    provider=LLMProvider.OPENAI
)
```

### Agent Workflow
```python
from kerb.agent.patterns import ReActAgent

def llm_function(prompt: str) -> str:
    """Your LLM function (OpenAI, Anthropic, etc.)"""
    # Implementation here
    return "agent response"

# Create a ReAct agent
agent = ReActAgent(
    name="ResearchAgent",
    llm_func=llm_function,
    max_iterations=5
)

# Execute multi-step task
result = agent.run("Research the latest AI papers and summarize key trends")

print(f"Status: {result.status.value}")
print(f"Output: {result.output}")
print(f"Steps taken: {len(result.steps)}")
```

### Custom Evaluation
```python
from kerb.evaluation import (
    calculate_bleu,
    calculate_rouge,
    calculate_f1_score,
    calculate_semantic_similarity
)

# Evaluate translation quality
reference = "Hello, how are you?"
candidate = "Hi, how are you?"

# Calculate metrics
bleu_score = calculate_bleu(candidate, reference)
rouge_scores = calculate_rouge(candidate, reference, rouge_type="rouge-l")
f1 = calculate_f1_score(candidate, reference)

print(f"BLEU: {bleu_score:.3f}")
print(f"ROUGE-L F1: {rouge_scores['fmeasure']:.3f}")
print(f"F1 Score: {f1:.3f}")
```

### Fine-Tuning Dataset Preparation
```python
from kerb.fine_tuning import (
    write_jsonl,
    read_jsonl,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    to_openai_format,
)
from kerb.fine_tuning.jsonl import (
    append_jsonl,
    merge_jsonl,
    validate_jsonl,
    count_jsonl_lines,
)

# Create training examples
examples = []
for i in range(10):
    examples.append(TrainingExample(
        messages=[
            {"role": "user", "content": f"How do I use Python feature {i}?"},
            {"role": "assistant", "content": f"Here's how to use feature {i}: example_code()"}
        ],
        metadata={"category": "coding", "index": i}
    ))

dataset = TrainingDataset(
    examples=examples,
    format=DatasetFormat.CHAT,
    metadata={"source": "coding_qa"}
)

# Convert to OpenAI format and write to JSONL
data = to_openai_format(dataset)
write_jsonl(data, "training_data.jsonl")

# Validate the JSONL file
result = validate_jsonl("training_data.jsonl")
print(f"Valid: {result.is_valid}, Examples: {result.total_examples}")

# Count lines efficiently
count = count_jsonl_lines("training_data.jsonl")
print(f"Total examples: {count}")
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://apxml.com/references/kerb)
- [GitHub](https://github.com/apxml/kerb)

