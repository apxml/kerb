"""Code Chunking Example

This example demonstrates chunking code while respecting structure.

Main concepts:
- Code-aware chunking that preserves functions/classes
- Language-specific splitting strategies
- Maintaining semantic coherence in code
- CodeChunker class for structured code

Use cases:
- Code documentation and indexing
- Code search and retrieval
- LLM-based code analysis
- Code embedding for semantic search
"""

from kerb.chunk import CodeChunker


def demonstrate_basic_code_chunking():
    """Show basic code chunking with Python."""
    print("="*80)
    print("BASIC CODE CHUNKING")
    print("="*80)
    
    python_code = '''
def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)

def embed_text(text, model="text-embedding-ada-002"):
    """Embed text using OpenAI API."""
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

class VectorStore:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
    
    def add(self, vector, metadata=None):
        """Add a vector to the store."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector must be {self.dimension} dimensions")
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
    
    def search(self, query_vector, top_k=5):
        """Search for similar vectors."""
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = calculate_similarity(query_vector, vec)
            similarities.append((sim, i))
        similarities.sort(reverse=True)
        return similarities[:top_k]
    '''.strip()
    
    print(f"\nPython code ({len(python_code)} chars):\n{python_code[:200]}...\n")
    
    # Chunk code while respecting function/class boundaries
    chunker = CodeChunker(max_chunk_size=400, language="python")
    chunks = chunker.chunk(python_code)
    
    print(f"\nCodeChunker created {len(chunks)} chunks (max_size=400, language=python):")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.split('\n')
        print(f"\nChunk {i} ({len(chunk)} chars, {len(lines)} lines):")
        print(f"  First line: {lines[0]}")
        print(f"  Last line: {lines[-1]}")
        print(f"  Preview:\n{chunk[:150]}...")


def demonstrate_class_preservation():
    """Show how CodeChunker preserves class structure."""
    print("\n" + "="*80)
    print("CLASS STRUCTURE PRESERVATION")
    print("="*80)
    
    code_with_classes = '''
class Embedder:
    """Handles text embedding operations."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.cache = {}
    
    def embed(self, text):
        if text in self.cache:
            return self.cache[text]
        embedding = self._compute_embedding(text)
        self.cache[text] = embedding
        return embedding
    
    def _compute_embedding(self, text):
        # Actual embedding computation
        return [0.0] * 1536

class Retriever:
    """Retrieves relevant documents."""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results
    '''.strip()
    
    print(f"\nCode with multiple classes:\n{code_with_classes[:150]}...\n")
    
    # Chunk with small size to force splitting
    chunker = CodeChunker(max_chunk_size=200, language="python")
    chunks = chunker.chunk(code_with_classes)
    
    print(f"\nChunked into {len(chunks)} pieces (max_size=200):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        # Show first and last lines to demonstrate structure
        lines = chunk.strip().split('\n')
        print(f"  Starts: {lines[0]}")
        if len(lines) > 1:
            print(f"  Ends: {lines[-1]}")


def demonstrate_function_boundaries():
    """Show function boundary preservation."""
    print("\n" + "="*80)
    print("FUNCTION BOUNDARY PRESERVATION")
    print("="*80)
    
    functions_code = '''
async def fetch_embeddings(texts, model="all-MiniLM-L6-v2"):
    """Fetch embeddings for multiple texts asynchronously."""
    embeddings = []
    for text in texts:
        embedding = await get_embedding(text, model)
        embeddings.append(embedding)
    return embeddings

def preprocess_text(text):
    """Clean and normalize text before embedding."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def chunk_document(doc, chunk_size=500):
    """Split document into chunks for embedding."""
    chunks = []
    start = 0
    while start < len(doc):
        end = start + chunk_size
        chunks.append(doc[start:end])
        start = end
    return chunks
    '''.strip()
    
    print(f"\nCode with multiple functions:\n{functions_code[:150]}...\n")
    
    chunker = CodeChunker(max_chunk_size=250, language="python")
    chunks = chunker.chunk(functions_code)
    
    print(f"\nFunction-aware chunks (max_size=250):")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        func_name = "Unknown"
        if lines[0].startswith('def ') or lines[0].startswith('async def '):
            func_name = lines[0].split('(')[0].replace('def ', '').replace('async ', '').strip()
        
        print(f"\nChunk {i} (contains: {func_name}):")
        print(f"  {lines[0]}")
        if len(lines) > 1:
            print(f"  ... ({len(lines)} lines total)")


def demonstrate_code_documentation():
    """Show code chunking for documentation/search."""
    print("\n" + "="*80)
    print("CODE DOCUMENTATION & SEARCH")
    print("="*80)
    
    documented_code = '''
def create_rag_pipeline(docs, chunk_size=500):
    """
    Create a complete RAG pipeline from documents.
    
    Args:
        docs (list): List of document strings
        chunk_size (int): Size of chunks for embedding
    
    Returns:
        dict: Pipeline components (embedder, store, retriever)
    """
    chunker = RecursiveChunker(chunk_size=chunk_size)
    embedder = Embedder(model="text-embedding-ada-002")
    store = VectorStore(dimension=1536)
    
    for doc in docs:
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            embedding = embedder.embed(chunk)
            store.add(embedding, metadata={"text": chunk})
    
    retriever = Retriever(store, embedder)
    
    return {
        "embedder": embedder,
        "store": store,
        "retriever": retriever
    }
    '''.strip()
    
    print(f"\nDocumented code:\n{documented_code[:200]}...\n")
    
    # Chunk for indexing/search
    chunker = CodeChunker(max_chunk_size=600, language="python")
    chunks = chunker.chunk(documented_code)
    
    print(f"\nCode chunks for search/documentation ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        # Extract function name and docstring
        lines = chunk.strip().split('\n')
        func_name = lines[0].split('(')[0].replace('def ', '').strip() if lines[0].startswith('def ') else "code block"
        
        has_docstring = '"""' in chunk or "'''" in chunk
        
        print(f"\nChunk {i}: {func_name}")
        print(f"  Has docstring: {has_docstring}")
        print(f"  Size: {len(chunk)} chars")
        print(f"  Preview: {lines[0]}")


def demonstrate_mixed_content():
    """Show handling of mixed code and comments."""
    print("\n" + "="*80)
    print("MIXED CODE AND COMMENTS")
    print("="*80)
    
    mixed_code = '''
# Configuration for RAG system
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
EMBEDDING_MODEL = "text-embedding-ada-002"

# Vector database settings
VECTOR_DB_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

def initialize_system():
    """Initialize the RAG system with configuration."""
    config = {
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP_SIZE,
        "model": EMBEDDING_MODEL
    }
    
    # Connect to vector database
    client = VectorDBClient(VECTOR_DB_URL)
    
    # Create collection if needed
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            name=COLLECTION_NAME,
            dimension=1536
        )
    
    return config, client
    '''.strip()
    
    print(f"\nMixed code with comments and config:\n{mixed_code[:150]}...\n")
    
    chunker = CodeChunker(max_chunk_size=300, language="python")
    chunks = chunker.chunk(mixed_code)
    
    print(f"\nChunks from mixed content ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        lines = chunk.strip().split('\n')
        content_type = "function" if any(l.startswith('def ') for l in lines) else "config/comments"
        
        print(f"\nChunk {i} ({content_type}):")
        for line in lines[:3]:  # Show first 3 lines
            print(f"  {line}")
        if len(lines) > 3:
            print(f"  ... ({len(lines)} lines total)")


def demonstrate_code_rag_pipeline():
    """Simulate a code search RAG pipeline."""
    print("\n" + "="*80)
    print("CODE SEARCH RAG PIPELINE")
    print("="*80)
    
    # Code repository sample
    codebase = '''
class EmbeddingService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def embed_batch(self, texts):
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [item.embedding for item in response.data]

class DocumentProcessor:
    def __init__(self, chunker, embedder):
        self.chunker = chunker
        self.embedder = embedder
    
    def process(self, documents):
        all_chunks = []
        for doc_id, doc in enumerate(documents):
            chunks = self.chunker.chunk(doc)
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk
                })
        
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding
        
        return all_chunks
    '''.strip()
    
    print("Code Repository Sample:")
    print(f"{codebase[:150]}...\n")
    
    # Chunk codebase for search
    chunker = CodeChunker(max_chunk_size=400, language="python")
    code_chunks = chunker.chunk(codebase)
    
    print(f"Chunked codebase into {len(code_chunks)} searchable units:\n")
    
    for i, chunk in enumerate(code_chunks, 1):
        # Simulate metadata extraction
        lines = chunk.strip().split('\n')
        if lines[0].startswith('class '):
            entity_type = "class"
            entity_name = lines[0].split(':')[0].replace('class ', '').strip()
        elif any(l.strip().startswith('def ') for l in lines):
            entity_type = "method"
            method_line = next(l for l in lines if l.strip().startswith('def '))
            entity_name = method_line.split('(')[0].replace('def ', '').strip()
        else:
            entity_type = "code block"
            entity_name = "unknown"
        
        print(f"Chunk {i}:")
        print(f"  Type: {entity_type}")
        print(f"  Name: {entity_name}")
        print(f"  Size: {len(chunk)} chars")
        print(f"  First line: {lines[0][:60]}...")
        print()
    
    print("Usage in RAG:")
    print("  1. Each chunk would be embedded")
    print("  2. Stored in vector DB with metadata")
    print("  3. Retrieved based on code search query")
    print("  4. Used as context for code generation/explanation")


def main():
    """Run code chunking examples."""
    
    print("\n" + "="*80)
    print("CODE CHUNKING EXAMPLES")
    print("="*80)
    print("\nCode-aware chunking that preserves structure and semantics.\n")
    
    demonstrate_basic_code_chunking()
    demonstrate_class_preservation()
    demonstrate_function_boundaries()
    demonstrate_code_documentation()
    demonstrate_mixed_content()
    demonstrate_code_rag_pipeline()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- CodeChunker splits code while respecting function/class boundaries
- Preserves semantic coherence better than character-based splitting
- Supports Python (extensible to other languages)
- Keeps complete functions/classes together when possible
- Handles mixed content (code, comments, config)
- Ideal for code search, documentation, and LLM-based analysis
- Use max_chunk_size based on your context window needs
- Great for building code retrieval and generation systems
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
