"""
Code Extraction from LLM Outputs
================================

This example demonstrates how to extract code blocks from LLM responses
that include code snippets in markdown format.

Main concepts:
- Extracting code blocks from markdown
- Filtering by programming language
- Handling multiple code blocks
- Practical use cases for code generation
"""

from kerb.parsing import extract_code_blocks


def simulate_llm_code_response(task: str) -> str:
    """Simulate LLM responses containing code."""
    
    if task == "python_function":
        return """Here's a Python function to calculate fibonacci numbers:

# %%
# Setup and Imports
# -----------------

```python

# %%
# Fibonacci
# ---------

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

This uses a recursive approach. For better performance with large numbers, you could use memoization."""
    
    elif task == "multiple_languages":
        return """Here's how to implement a simple HTTP server in different languages:

**Python:**
```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):

# %%
# Do Get
# ------

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, World!')

server = HTTPServer(('localhost', 8000), SimpleHandler)
server.serve_forever()
```

**JavaScript (Node.js):**
```javascript
const http = require('http');

const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end('Hello, World!');
});

server.listen(8000, 'localhost');
console.log('Server running at http://localhost:8000/');
```

**Go:**
```go
package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
}

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8000", nil)
}
```

Each implementation creates a basic HTTP server on port 8000."""
    
    elif task == "sql_queries":
        return """Here are some SQL queries for the database:

**Get all active users:**
```sql
SELECT id, name, email, created_at
FROM users
WHERE status = 'active'
ORDER BY created_at DESC;
```

**Calculate user statistics:**
```sql
SELECT 
    COUNT(*) as total_users,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_users,
    AVG(age) as average_age
FROM users;
```

These queries will help you analyze your user base."""
    
    elif task == "config_files":
        return """Here's the configuration you need:

**Docker configuration:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**YAML configuration:**
```yaml
server:
  host: localhost
  port: 8080
  debug: true

database:
  host: db.example.com
  port: 5432
  name: myapp
  
logging:
  level: INFO
  format: json
```

Apply these configurations to set up your environment."""
    
    elif task == "code_review":
        return """I found several issues in the code. Here's the corrected version:

**Original (problematic):**
```python

# %%
# Process Data
# ------------

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```

**Improved version:**
```python

# %%
# Process Data
# ------------

def process_data(data):
    \"\"\"Process data by doubling positive values.
    
    Args:
        data: List of numeric values
        
    Returns:
        List of doubled positive values
    \"\"\"
    if not data:
        return []
    
    return [item * 2 for item in data if item > 0]
```

The improvements include:
- Added docstring
- Added input validation
- Used list comprehension for better performance
- More concise and readable"""
    
    return "No code available."


def main():
    """Run code extraction examples."""
    
    print("="*80)
    print("CODE EXTRACTION FROM LLM OUTPUTS")
    print("="*80)
    
    # Example 1: Extract single code block
    print("\nExample 1: Extract Single Python Code Block")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("python_function")
    print(f"LLM Output:\n{llm_output}\n")
    
    code_blocks = extract_code_blocks(llm_output)
    
    print(f"Found {len(code_blocks)} code block(s):")
    for i, block in enumerate(code_blocks, 1):
        print(f"\nBlock {i} ({block['language']}):")
        print(block['code'])
    
    # Example 2: Extract and filter by language
    print("\n\nExample 2: Extract Multiple Languages - Filter Python")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("multiple_languages")
    
    # Extract all code blocks
    all_blocks = extract_code_blocks(llm_output)
    print(f"Total code blocks found: {len(all_blocks)}")
    
    # Filter for Python only
    python_blocks = extract_code_blocks(llm_output, language="python")
    print(f"Python code blocks: {len(python_blocks)}\n")
    
    for block in python_blocks:
        print(f"Python Code ({len(block['code'].splitlines())} lines):")
        print(block['code'][:200] + "..." if len(block['code']) > 200 else block['code'])
    
    # Example 3: Extract specific language types
    print("\n\nExample 3: Extract All Languages")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("multiple_languages")
    
    all_blocks = extract_code_blocks(llm_output)
    
    languages_found = {}
    for block in all_blocks:
        lang = block['language']
        if lang not in languages_found:
            languages_found[lang] = 0
        languages_found[lang] += 1
    
    print("Languages found:")
    for lang, count in languages_found.items():
        print(f"  - {lang}: {count} block(s)")
    
    # Example 4: Extract SQL queries
    print("\n\nExample 4: Extract SQL Queries")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("sql_queries")
    
    sql_blocks = extract_code_blocks(llm_output, language="sql")
    
    print(f"Found {len(sql_blocks)} SQL query/queries:\n")
    for i, block in enumerate(sql_blocks, 1):
        print(f"Query {i}:")
        print(block['code'])
        print()
    
    # Example 5: Extract configuration files
    print("\n\nExample 5: Extract Configuration Files")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("config_files")
    
    all_blocks = extract_code_blocks(llm_output)
    
    print(f"Found {len(all_blocks)} configuration block(s):\n")
    for block in all_blocks:
        print(f"Format: {block['language']}")
        print(f"Content ({len(block['code'])} characters):")
        print(block['code'][:150] + "..." if len(block['code']) > 150 else block['code'])
        print()
    
    # Example 6: Code review scenario
    print("\n\nExample 6: Code Review - Extract Before/After")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("code_review")
    
    code_blocks = extract_code_blocks(llm_output, language="python")
    
    if len(code_blocks) >= 2:
        print("Original Code:")
        print(code_blocks[0]['code'])
        print("\n" + "-"*40 + "\n")
        print("Improved Code:")
        print(code_blocks[1]['code'])
    
    # Example 7: Practical use case - code execution
    print("\n\nExample 7: Practical Use Case - Execute Extracted Code")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("python_function")
    
    code_blocks = extract_code_blocks(llm_output, language="python")
    
    if code_blocks:
        code = code_blocks[0]['code']
        print("Extracted code for execution:")
        print(code)
        print("\n" + "-"*40 + "\n")
        
        # In a real scenario, you might execute this code in a sandboxed environment
        print("In a production system, you would:")
        print("1. Validate the code for safety")
        print("2. Execute in a sandboxed environment")
        print("3. Capture output and errors")
        print("4. Return results to the LLM for further processing")
    
    # Example 8: Statistics on extracted code
    print("\n\nExample 8: Code Statistics")
    print("-"*80)
    
    llm_output = simulate_llm_code_response("multiple_languages")
    
    all_blocks = extract_code_blocks(llm_output)
    
    total_lines = sum(len(block['code'].splitlines()) for block in all_blocks)
    total_chars = sum(len(block['code']) for block in all_blocks)
    
    print(f"Code Statistics:")
    print(f"  Total blocks: {len(all_blocks)}")
    print(f"  Total lines: {total_lines}")
    print(f"  Total characters: {total_chars}")
    print(f"  Average lines per block: {total_lines / len(all_blocks):.1f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
