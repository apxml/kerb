#!/usr/bin/env python3
"""
Generate API documentation from docstrings and export as JSON.

This script uses sphinx.ext.autodoc approach: extracts docstrings and metadata
as structured data that can be consumed by documentation tools.

Output:
- One JSON file per subpackage in docs/references/api/
- An index.json with metadata about all modules

Format follows Sphinx's JSON builder format for easy integration.
"""

import inspect
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints


def get_type_annotation(annotation) -> str:
    """Convert a type annotation to a string representation."""
    if annotation is inspect.Parameter.empty:
        return None
    
    if hasattr(annotation, '__module__') and annotation.__module__ == 'typing':
        return str(annotation).replace('typing.', '')
    
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    
    return str(annotation)


def extract_function_info(func, name: str) -> Dict[str, Any]:
    """Extract function information in Sphinx-compatible format."""
    try:
        sig = inspect.signature(func)
        
        # Parse parameters
        params = []
        for param_name, param in sig.parameters.items():
            params.append({
                'name': param_name,
                'type': get_type_annotation(param.annotation),
                'default': str(param.default) if param.default is not inspect.Parameter.empty else None,
                'description': ''  # Would need to parse from docstring
            })
        
        return {
            'name': name,
            'type': 'function',
            'signature': str(sig),
            'docstring': inspect.getdoc(func) or '',
            'parameters': params,
            'returns': {
                'type': get_type_annotation(sig.return_annotation),
                'description': ''  # Would need to parse from docstring
            },
            'source_file': inspect.getsourcefile(func) if hasattr(func, '__code__') else None,
        }
    except Exception as e:
        return {
            'name': name,
            'type': 'function',
            'docstring': inspect.getdoc(func) or '',
            'error': str(e)
        }


def extract_class_info(cls, name: str) -> Dict[str, Any]:
    """Extract class information in Sphinx-compatible format."""
    # Get methods
    methods = []
    for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not method_name.startswith('_') or method_name in ['__init__', '__call__']:
            methods.append(extract_function_info(method, method_name))
    
    # Get bases
    bases = [base.__name__ for base in cls.__bases__ if base.__name__ != 'object']
    
    return {
        'name': name,
        'type': 'class',
        'docstring': inspect.getdoc(cls) or '',
        'bases': bases,
        'methods': methods,
        'attributes': [],  # Could extract from __annotations__
        'source_file': inspect.getsourcefile(cls) if hasattr(cls, '__module__') else None,
    }


def extract_module_docs(module_name: str, base_package: str = 'kerb') -> Dict[str, Any]:
    """Extract documentation from a module in structured JSON format."""
    full_module_name = f'{base_package}.{module_name}'
    
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        print(f"Warning: Could not import {full_module_name}: {e}", file=sys.stderr)
        return None
    
    # Collect functions and classes
    functions = []
    classes = []
    
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
        
        try:
            obj_module = getattr(obj, '__module__', '')
            
            if inspect.isclass(obj) and obj_module.startswith(base_package):
                classes.append(extract_class_info(obj, name))
            elif inspect.isfunction(obj) and obj_module.startswith(base_package):
                functions.append(extract_function_info(obj, name))
        except Exception as e:
            print(f"Warning: Error processing {name} in {module_name}: {e}", file=sys.stderr)
    
    return {
        'module': module_name,
        'fullname': full_module_name,
        'docstring': inspect.getdoc(module) or '',
        'functions': functions,
        'classes': classes
    }


def generate_api_docs(output_dir: Path, base_package: str = 'kerb'):
    """Generate API documentation in structured JSON format (Sphinx-compatible)."""
    
    # List of all submodules
    modules = [
        'core', 'agent', 'cache', 'chunk', 'config', 'context',
        'document', 'embedding', 'evaluation', 'fine_tuning',
        'generation', 'memory', 'multimodal', 'parsing',
        'preprocessing', 'prompt', 'retrieval', 'safety',
        'testing', 'tokenizer',
    ]
    
    # Create output directory
    api_dir = output_dir / 'api'
    api_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate index
    index = {
        'package': base_package,
        'format': 'sphinx-json',
        'version': '1.0',
        'modules': []
    }
    
    # Process each module
    for module_name in modules:
        print(f"Processing {module_name}...", file=sys.stderr)
        
        docs = extract_module_docs(module_name, base_package)
        
        if docs:
            # Save module JSON
            output_file = api_dir / f"{module_name}.json"
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
            
            # Add to index
            index['modules'].append({
                'name': module_name,
                'fullname': docs['fullname'],
                'file': f"api/{module_name}.json",
                'functions_count': len(docs['functions']),
                'classes_count': len(docs['classes'])
            })
            
            print(f"✓ Generated {output_file}", file=sys.stderr)
        else:
            print(f"✗ Skipped {module_name}", file=sys.stderr)
    
    # Save index
    index_file = output_dir / 'index.json'
    with index_file.open('w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated index at {index_file}", file=sys.stderr)
    print(f"✓ Total modules processed: {len(index['modules'])}", file=sys.stderr)


if __name__ == '__main__':
    # Get the script's directory (docs/references/)
    script_dir = Path(__file__).parent
    
    # Add the project root to Python path so we can import kerb
    project_root = script_dir.parent.parent
    sys.path.insert(0, str(project_root))
    
    print(f"Generating API documentation in Sphinx-compatible JSON format...\n", file=sys.stderr)
    generate_api_docs(script_dir)
    print(f"\n✅ Documentation generation complete!", file=sys.stderr)
