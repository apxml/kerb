API Reference
=============

Kerb provides a comprehensive set of modules for building LLM applications. Each module is designed to be lightweight, modular, and easy to integrate into your existing projects.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   core
   agent
   cache
   chunk
   config
   context
   document
   embedding
   evaluation
   fine_tuning
   generation
   memory
   multimodal
   parsing
   preprocessing
   prompt
   retrieval
   safety
   testing
   tokenizer

Module Overview
---------------

Core
~~~~
Shared types and interfaces used across all modules.

Agent
~~~~~
Agent orchestration and execution patterns for multi-step reasoning and autonomous task completion.

Cache
~~~~~
Response and embedding caching mechanisms to reduce API costs and improve latency.

Chunk
~~~~~
Text chunking utilities for optimal context window usage and retrieval performance.

Config
~~~~~~
Configuration management for models, providers, API keys, and application settings.

Context
~~~~~~~
Context window management and token budget tracking for LLM conversations.

Document
~~~~~~~~
Document loading and processing utilities for PDFs, web pages, DOCX, and more.

Embedding
~~~~~~~~~
Embedding generation with support for multiple providers and similarity search helpers.

Evaluation
~~~~~~~~~~
Metrics and benchmarking tools for evaluating LLM outputs (BLEU, ROUGE, BERTScore, etc.).

Fine-Tuning
~~~~~~~~~~~
Model fine-tuning utilities and large dataset preparation for training custom models.

Generation
~~~~~~~~~~
Unified LLM text generation with multi-provider support (OpenAI, Anthropic, Gemini, Cohere).

Memory
~~~~~~
Conversation memory and entity tracking for building stateful applications.

Multimodal
~~~~~~~~~~
Image, audio, and video processing utilities for multimodal LLM applications.

Parsing
~~~~~~~
Output parsing and validation for JSON, structured data, and function calls.

Preprocessing
~~~~~~~~~~~~~
Text cleaning, normalization, and preprocessing utilities for LLM inputs.

Prompt
~~~~~~
Prompt engineering utilities, templates, and chain-of-thought patterns.

Retrieval
~~~~~~~~~
RAG (Retrieval-Augmented Generation) and vector search utilities for semantic retrieval.

Safety
~~~~~~
Content moderation, safety filters, and input validation.

Testing
~~~~~~~
Testing utilities and helpers for LLM outputs and evaluation workflows.

Tokenizer
~~~~~~~~~
Token counting and text splitting utilities compatible with any model.
