"""Few-shot example management for prompt engineering.

This module provides tools for managing, selecting, and formatting
few-shot examples to include in prompts.
"""

import random
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict

from kerb.core.enums import SelectionStrategy


@dataclass
class FewShotExample:
    """A few-shot example with input and output.
    
    Attributes:
        input (str): Example input
        output (str): Example output
        metadata (Dict[str, Any]): Additional metadata (category, difficulty, etc.)
    """
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def format(self, template: str = "Input: {input}\nOutput: {output}") -> str:
        """Format the example using a template.
        
        Args:
            template (str): Format template. Defaults to "Input: {input}\\nOutput: {output}".
            
        Returns:
            str: Formatted example
        """
        return template.format(input=self.input, output=self.output)


class ExampleSelector:
    """Manages and selects few-shot examples for prompts.
    
    Supports multiple selection strategies including random, semantic similarity,
    and diversity-based selection.
    """
    
    def __init__(self, examples: Optional[List[FewShotExample]] = None):
        """Initialize the example selector.
        
        Args:
            examples (Optional[List[FewShotExample]]): Initial examples. Defaults to None.
        """
        self.examples = examples or []
    
    def add(self, example: FewShotExample) -> None:
        """Add an example to the selector.
        
        Args:
            example (FewShotExample): Example to add
        """
        self.examples.append(example)
    
    def select(
        self,
        k: int = 3,
        strategy: Union[str, SelectionStrategy] = SelectionStrategy.RANDOM,
        query: Optional[str] = None,
        filter_fn: Optional[Callable[[FewShotExample], bool]] = None
    ) -> List[FewShotExample]:
        """Select k examples using the specified strategy.
        
        Args:
            k (int): Number of examples to select. Defaults to 3.
            strategy (Union[str, SelectionStrategy]): Selection strategy. Defaults to RANDOM.
            query (Optional[str]): Query text for semantic selection. Defaults to None.
            filter_fn (Optional[Callable]): Filter function to apply before selection.
                Defaults to None.
                
        Returns:
            List[FewShotExample]: Selected examples
        """
        # Apply filter if provided
        candidates = self.examples
        if filter_fn:
            candidates = [ex for ex in candidates if filter_fn(ex)]
        
        if not candidates:
            return []
        
        # Convert strategy to enum if needed
        if isinstance(strategy, str):
            strategy = SelectionStrategy(strategy.lower())
        
        # Limit k to available examples
        k = min(k, len(candidates))
        
        if strategy == SelectionStrategy.RANDOM:
            return random.sample(candidates, k)
        
        elif strategy == SelectionStrategy.FIRST:
            return candidates[:k]
        
        elif strategy == SelectionStrategy.LAST:
            return candidates[-k:]
        
        elif strategy == SelectionStrategy.DIVERSE:
            # Select diverse examples by maximizing dissimilarity
            # Simple heuristic: space out selections evenly
            if k >= len(candidates):
                return candidates
            
            step = len(candidates) / k
            indices = [int(i * step) for i in range(k)]
            return [candidates[i] for i in indices]
        
        elif strategy == SelectionStrategy.SEMANTIC:
            # Semantic selection requires embeddings
            if query is None:
                raise ValueError("Query text is required for semantic selection")
            
            try:
                from ..embedding import embed, cosine_similarity
                
                # Embed query and examples
                query_embedding = embed(query)
                example_embeddings = [embed(ex.input) for ex in candidates]
                
                # Calculate similarities
                similarities = [
                    cosine_similarity(query_embedding, ex_emb)
                    for ex_emb in example_embeddings
                ]
                
                # Select top k by similarity
                sorted_indices = sorted(
                    range(len(similarities)),
                    key=lambda i: similarities[i],
                    reverse=True
                )
                
                return [candidates[i] for i in sorted_indices[:k]]
            
            except ImportError:
                # Fall back to random if embedding is not available
                return random.sample(candidates, k)
        
        return candidates[:k]
    
    def format_examples(
        self,
        examples: List[FewShotExample],
        template: str = "Input: {input}\nOutput: {output}",
        separator: str = "\n\n"
    ) -> str:
        """Format multiple examples into a single string.
        
        Args:
            examples (List[FewShotExample]): Examples to format
            template (str): Format template for each example.
                Defaults to "Input: {input}\\nOutput: {output}".
            separator (str): Separator between examples. Defaults to "\\n\\n".
            
        Returns:
            str: Formatted examples string
        """
        formatted = [ex.format(template) for ex in examples]
        return separator.join(formatted)


def create_example(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> FewShotExample:
    """Create a few-shot example.
    
    Args:
        input_text (str): Example input
        output_text (str): Example output
        metadata (Optional[Dict[str, Any]]): Additional metadata. Defaults to None.
        
    Returns:
        FewShotExample: The created example
        
    Examples:
        >>> ex = create_example(
        ...     input_text="What is 2+2?",
        ...     output_text="4",
        ...     metadata={"difficulty": "easy"}
        ... )
    """
    return FewShotExample(
        input=input_text,
        output=output_text,
        metadata=metadata or {}
    )


def select_examples(
    examples: List[FewShotExample],
    k: int = 3,
    strategy: Union[SelectionStrategy, str] = "random",
    query: Optional[str] = None,
    filter_fn: Optional[Callable[[FewShotExample], bool]] = None
) -> List[FewShotExample]:
    """Select k examples from a list using the specified strategy.
    
    Args:
        examples: Available examples
        k: Number of examples to select
        strategy: Selection strategy (SelectionStrategy enum or string: "random", "similarity", "diverse", "recent", "fixed")
        query: Query text for semantic selection
        filter_fn: Filter function
        
    Returns:
        List[FewShotExample]: Selected examples
        
    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import SelectionStrategy
        >>> selected = select_examples(examples, k=3, strategy=SelectionStrategy.SIMILARITY, query="example query")
        
        >>> # Using string (for backward compatibility)
        >>> selected = select_examples(examples, k=3, strategy="random")
    """
    from kerb.core.enums import validate_enum_or_string
    
    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, SelectionStrategy, "strategy")
    if isinstance(strategy_val, SelectionStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val
    
    selector = ExampleSelector(examples)
    return selector.select(k=k, strategy=strategy_str, query=query, filter_fn=filter_fn)


def format_examples(
    examples: List[FewShotExample],
    template: str = "Input: {input}\nOutput: {output}",
    separator: str = "\n\n"
) -> str:
    """Format multiple examples into a single string.
    
    Args:
        examples (List[FewShotExample]): Examples to format
        template (str): Format template for each example.
            Defaults to "Input: {input}\\nOutput: {output}".
        separator (str): Separator between examples. Defaults to "\\n\\n".
        
    Returns:
        str: Formatted examples string
        
    Examples:
        >>> ex1 = create_example("What is 2+2?", "4")
        >>> ex2 = create_example("What is 3+3?", "6")
        >>> format_examples([ex1, ex2])
        'Input: What is 2+2?\\nOutput: 4\\n\\nInput: What is 3+3?\\nOutput: 6'
    """
    selector = ExampleSelector(examples)
    return selector.format_examples(examples, template, separator)
