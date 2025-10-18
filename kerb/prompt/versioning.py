"""Prompt versioning and A/B testing functionality.

This module provides tools for managing multiple versions of prompts,
enabling A/B testing and comparison.
"""

import random
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .template import render_template, extract_template_variables
from kerb.core.enums import VersionSelectionStrategy, validate_enum_or_string


@dataclass
class PromptVersion:
    """A versioned prompt with metadata for tracking and comparison.
    
    Attributes:
        name (str): Unique name for this prompt
        version (str): Version identifier (e.g., "1.0", "v2", "experimental")
        template (str): The prompt template
        description (str): Description of this version
        created_at (str): Timestamp of creation
        metadata (Dict[str, Any]): Additional metadata (performance metrics, etc.)
        variables (Dict[str, Any]): Default variable values
    """
    name: str
    version: str
    template: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, variables: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Render this prompt version with variables.
        
        Args:
            variables (Optional[Dict[str, Any]]): Variables to use for rendering.
                Will be merged with default variables.
            **kwargs: Additional variables as keyword arguments
            
        Returns:
            str: Rendered prompt
        """
        # Merge default variables with provided variables
        merged_vars = {**self.variables, **(variables or {}), **kwargs}
        return render_template(self.template, merged_vars)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class PromptRegistry:
    """Registry for managing multiple prompt versions and A/B testing.
    
    Enables version tracking, comparison, and selection for prompt experimentation.
    """
    
    def __init__(self):
        """Initialize an empty prompt registry."""
        self._prompts: Dict[str, Dict[str, PromptVersion]] = {}
    
    def register(self, prompt: PromptVersion) -> None:
        """Register a prompt version.
        
        Args:
            prompt (PromptVersion): Prompt version to register
        """
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}
        
        self._prompts[prompt.name][prompt.version] = prompt
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptVersion]:
        """Retrieve a prompt version.
        
        Args:
            name (str): Prompt name
            version (Optional[str]): Version to retrieve. If None, returns latest.
                Defaults to None.
                
        Returns:
            Optional[PromptVersion]: Prompt version if found, None otherwise
        """
        if name not in self._prompts:
            return None
        
        versions = self._prompts[name]
        
        if version is None:
            # Return the most recently created version
            return max(versions.values(), key=lambda p: p.created_at)
        
        return versions.get(version)
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions for a prompt.
        
        Args:
            name (str): Prompt name
            
        Returns:
            List[str]: List of version identifiers
        """
        if name not in self._prompts:
            return []
        
        return list(self._prompts[name].keys())
    
    def list_prompts(self) -> List[str]:
        """List all registered prompt names.
        
        Returns:
            List[str]: List of prompt names
        """
        return list(self._prompts.keys())
    
    def compare(self, name: str, versions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare versions of a prompt.
        
        Args:
            name (str): Prompt name
            versions (Optional[List[str]]): Versions to compare. If None, compares all.
                Defaults to None.
                
        Returns:
            Dict[str, Any]: Comparison data including templates, metadata, and differences
        """
        if name not in self._prompts:
            return {}
        
        prompt_versions = self._prompts[name]
        
        if versions is None:
            versions = list(prompt_versions.keys())
        
        comparison = {
            "name": name,
            "versions": {}
        }
        
        for version in versions:
            if version in prompt_versions:
                prompt = prompt_versions[version]
                comparison["versions"][version] = {
                    "template": prompt.template,
                    "description": prompt.description,
                    "created_at": prompt.created_at,
                    "metadata": prompt.metadata,
                    "variables": extract_template_variables(prompt.template),
                    "length": len(prompt.template)
                }
        
        return comparison
    
    def select_ab_version(self, name: str, strategy: str = "random") -> Optional[PromptVersion]:
        """Select a version for A/B testing.
        
        Args:
            name (str): Prompt name
            strategy (str): Selection strategy ("random", "weighted", "newest", "oldest").
                Defaults to "random".
                
        Returns:
            Optional[PromptVersion]: Selected prompt version
        """
        if name not in self._prompts:
            return None
        
        versions = list(self._prompts[name].values())
        
        if not versions:
            return None
        
        if strategy == "random":
            return random.choice(versions)
        elif strategy == "newest":
            return max(versions, key=lambda p: p.created_at)
        elif strategy == "oldest":
            return min(versions, key=lambda p: p.created_at)
        elif strategy == "weighted":
            # Weight by success rate if available in metadata
            weights = []
            for v in versions:
                weight = v.metadata.get("success_rate", 1.0)
                weights.append(weight)
            
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            return random.choices(versions, weights=weights)[0]
        
        return versions[0]


# Global registry instance
_global_registry = PromptRegistry()


def create_version(
    name: str,
    version: str,
    template: str,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    variables: Optional[Dict[str, Any]] = None
) -> PromptVersion:
    """Create a new prompt version.
    
    Args:
        name (str): Unique name for this prompt
        version (str): Version identifier
        template (str): The prompt template
        description (str): Description of this version. Defaults to "".
        metadata (Optional[Dict[str, Any]]): Additional metadata. Defaults to None.
        variables (Optional[Dict[str, Any]]): Default variable values. Defaults to None.
        
    Returns:
        PromptVersion: The created prompt version
        
    Examples:
        >>> v1 = create_version(
        ...     name="greeting",
        ...     version="1.0",
        ...     template="Hello {{name}}!",
        ...     description="Simple greeting"
        ... )
    """
    return PromptVersion(
        name=name,
        version=version,
        template=template,
        description=description,
        metadata=metadata or {},
        variables=variables or {}
    )


def register_prompt(prompt: PromptVersion) -> None:
    """Register a prompt version in the global registry.
    
    Args:
        prompt (PromptVersion): Prompt version to register
    """
    _global_registry.register(prompt)


def get_prompt(name: str, version: Optional[str] = None) -> Optional[PromptVersion]:
    """Retrieve a prompt from the global registry.
    
    Args:
        name (str): Prompt name
        version (Optional[str]): Version to retrieve. If None, returns latest.
            Defaults to None.
            
    Returns:
        Optional[PromptVersion]: Prompt version if found, None otherwise
    """
    return _global_registry.get(name, version)


def list_versions(name: str) -> List[str]:
    """List all versions for a prompt in the global registry.
    
    Args:
        name (str): Prompt name
        
    Returns:
        List[str]: List of version identifiers
    """
    return _global_registry.list_versions(name)


def compare_versions(name: str, versions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compare versions of a prompt in the global registry.
    
    Args:
        name (str): Prompt name
        versions (Optional[List[str]]): Versions to compare. If None, compares all.
            Defaults to None.
            
    Returns:
        Dict[str, Any]: Comparison data
    """
    return _global_registry.compare(name, versions)


def select_version(
    name: str, 
    strategy: Union[VersionSelectionStrategy, str] = "random"
) -> Optional[PromptVersion]:
    """Select a version for A/B testing from the global registry.
    
    Args:
        name: Prompt name
        strategy: Selection strategy (VersionSelectionStrategy enum or string: "random", "latest", "best_performing", "a_b_test")
            
    Returns:
        Optional[PromptVersion]: Selected prompt version
        
    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import VersionSelectionStrategy
        >>> version = select_version("greeting", strategy=VersionSelectionStrategy.BEST_PERFORMING)
        
        >>> # Using string (for backward compatibility)
        >>> version = select_version("greeting", strategy="random")
    """
    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, VersionSelectionStrategy, "strategy")
    if isinstance(strategy_val, VersionSelectionStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val
    
    return _global_registry.select_ab_version(name, strategy_str)
