Kerb Documentation
==================

The complete toolkit for developers building LLM applications.

Built to drive production ML systems at ApX Machine Learning (`apxml.com <https://apxml.com>`_), available open source.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   examples
   api/index
   modules

Overview
--------

Simple
~~~~~~

Advanced LLM techniques made simple. Clean, easy-to-use interfaces for complex operations.

Lightweight
~~~~~~~~~~~

Only install what you need. Kerb is modular, no unnecessary dependencies.

Compatible
~~~~~~~~~~

Works with any LLM project. Kerb is a toolkit, not a framework. Use it alongside your existing stack.

Installation
------------

.. code-block:: bash

   # Install everything
   pip install kerb[all]

   # Or install specific modules
   pip install kerb[generation] kerb[embeddings] kerb[evaluation]

Quick Start
-----------

.. code-block:: python

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

Modules
-------

Everything you need to build LLM applications.

+-------------------+-------------------------------------------------------------------------+
| Module            | Description                                                             |
+===================+=========================================================================+
| **Agent**         | Agent orchestration and execution patterns for multi-step reasoning.   |
+-------------------+-------------------------------------------------------------------------+
| **Cache**         | Response and embedding caching to reduce costs and latency.            |
+-------------------+-------------------------------------------------------------------------+
| **Chunk**         | Text chunking utilities for optimal context windows and retrieval.     |
+-------------------+-------------------------------------------------------------------------+
| **Config**        | Configuration management for models, providers, and settings.          |
+-------------------+-------------------------------------------------------------------------+
| **Context**       | Context window management and token budget tracking.                   |
+-------------------+-------------------------------------------------------------------------+
| **Document**      | Document loading and processing for PDFs, web pages, and more.         |
+-------------------+-------------------------------------------------------------------------+
| **Embedding**     | Embedding generation and similarity search helpers.                    |
+-------------------+-------------------------------------------------------------------------+
| **Evaluation**    | Metrics and benchmarking tools for LLM outputs.                        |
+-------------------+-------------------------------------------------------------------------+
| **Fine-Tuning**   | Model fine-tuning utilities and large dataset preparation.             |
+-------------------+-------------------------------------------------------------------------+
| **Generation**    | Unified LLM generation with multi-provider support.                    |
+-------------------+-------------------------------------------------------------------------+
| **Memory**        | Conversation memory and entity tracking for stateful applications.     |
+-------------------+-------------------------------------------------------------------------+
| **Multimodal**    | Image, audio, and video processing for multimodal models.              |
+-------------------+-------------------------------------------------------------------------+
| **Parsing**       | Output parsing and validation (JSON, structured data, function calls). |
+-------------------+-------------------------------------------------------------------------+
| **Preprocessing** | Text cleaning and preprocessing for LLM inputs.                        |
+-------------------+-------------------------------------------------------------------------+
| **Prompt**        | Prompt engineering utilities, templates, and chain-of-thought.         |
+-------------------+-------------------------------------------------------------------------+
| **Retrieval**     | RAG and vector search utilities for semantic retrieval.                |
+-------------------+-------------------------------------------------------------------------+
| **Safety**        | Content moderation and safety filters.                                 |
+-------------------+-------------------------------------------------------------------------+
| **Testing**       | Testing utilities for LLM outputs and evaluation.                      |
+-------------------+-------------------------------------------------------------------------+
| **Tokenizer**     | Token counting and text splitting for any model.                       |
+-------------------+-------------------------------------------------------------------------+

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
