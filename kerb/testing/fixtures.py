"""Fixture management and response generators."""

import json
import random
import hashlib
import re
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
from dataclasses import asdict

from .types import FixtureData, FixtureFormat


class FixtureManager:
    """Manage and organize test fixtures."""
    
    def __init__(self, fixtures_dir: Optional[Path] = None):
        """Initialize fixture manager.
        
        Args:
            fixtures_dir: Directory to store fixtures
        """
        self.fixtures_dir = fixtures_dir or Path("tests/fixtures")
        self.fixtures: Dict[str, FixtureData] = {}
    
    def add_fixture(
        self,
        name: str,
        prompt: str,
        response: str,
        **kwargs
    ) -> None:
        """Add a fixture.
        
        Args:
            name: Fixture name/ID
            prompt: Prompt text
            response: Response text
            **kwargs: Additional metadata
        """
        self.fixtures[name] = FixtureData(
            prompt=prompt,
            response=response,
            metadata=kwargs
        )
    
    def get_fixture(self, name: str) -> Optional[FixtureData]:
        """Get a fixture by name."""
        return self.fixtures.get(name)
    
    def save(self, format: FixtureFormat = FixtureFormat.JSON) -> None:
        """Save fixtures to disk."""
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        
        if format == FixtureFormat.JSON:
            filepath = self.fixtures_dir / "fixtures.json"
            with open(filepath, "w") as f:
                json.dump(
                    {k: asdict(v) for k, v in self.fixtures.items()},
                    f,
                    indent=2
                )
        elif format == FixtureFormat.JSONL:
            filepath = self.fixtures_dir / "fixtures.jsonl"
            with open(filepath, "w") as f:
                for name, fixture in self.fixtures.items():
                    data = asdict(fixture)
                    data["name"] = name
                    f.write(json.dumps(data) + "\n")
    
    def load(self, filepath: Path) -> None:
        """Load fixtures from disk."""
        if filepath.suffix == ".json":
            with open(filepath) as f:
                data = json.load(f)
                for name, fixture_dict in data.items():
                    self.fixtures[name] = FixtureData(**fixture_dict)
        elif filepath.suffix == ".jsonl":
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    name = data.pop("name")
                    self.fixtures[name] = FixtureData(**data)


def load_fixtures(filepath: Path) -> Dict[str, FixtureData]:
    """Load fixtures from a file.
    
    Args:
        filepath: Path to fixture file
        
    Returns:
        Dictionary of fixture name to FixtureData
    """
    manager = FixtureManager()
    manager.load(filepath)
    return manager.fixtures


def save_fixtures(
    fixtures: Dict[str, FixtureData],
    filepath: Path,
    format: FixtureFormat = FixtureFormat.JSON
) -> None:
    """Save fixtures to a file.
    
    Args:
        fixtures: Fixtures to save
        filepath: Output filepath
        format: Output format
    """
    manager = FixtureManager()
    manager.fixtures = fixtures
    
    if format == FixtureFormat.JSON:
        with open(filepath, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in fixtures.items()},
                f,
                indent=2
            )


class DeterministicResponseGenerator:
    """Generate deterministic responses based on input hash."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator.
        
        Args:
            seed: Random seed for determinism
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate(self, prompt: str, templates: List[str]) -> str:
        """Generate deterministic response.
        
        Args:
            prompt: Input prompt
            templates: Response templates to choose from
            
        Returns:
            Deterministic response
        """
        # Hash prompt to get consistent index
        hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        index = hash_val % len(templates)
        return templates[index]


class SeededResponseGenerator:
    """Generate responses using seeded randomness."""
    
    def __init__(self, seed: int = 42):
        """Initialize with seed.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.rng = random.Random(seed)
    
    def generate(
        self,
        prompt: str,
        length_range: Tuple[int, int] = (50, 200),
        vocabulary: Optional[List[str]] = None
    ) -> str:
        """Generate seeded random response.
        
        Args:
            prompt: Input prompt (affects generation via hash)
            length_range: Min and max word count
            vocabulary: Word vocabulary to use
            
        Returns:
            Generated response
        """
        # Use prompt hash to influence generation
        prompt_hash = hash(prompt)
        local_rng = random.Random(self.seed + prompt_hash)
        
        vocab = vocabulary or ["test", "response", "data", "value", "result"]
        word_count = local_rng.randint(*length_range)
        
        words = [local_rng.choice(vocab) for _ in range(word_count)]
        return " ".join(words)


class PatternResponseGenerator:
    """Generate responses based on pattern matching."""
    
    def __init__(self, patterns: Dict[str, Union[str, Callable]]):
        """Initialize with patterns.
        
        Args:
            patterns: Dict mapping regex patterns to responses or callables
        """
        self.patterns = patterns
    
    def generate(self, prompt: str, default: str = "Default response") -> str:
        """Generate response based on pattern match.
        
        Args:
            prompt: Input prompt
            default: Default response if no pattern matches
            
        Returns:
            Generated response
        """
        for pattern, response in self.patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                if callable(response):
                    return response(match)
                return response
        return default
