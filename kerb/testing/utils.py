"""Utility functions for testing."""

import os
import random
import warnings
from pathlib import Path
from contextlib import contextmanager


def seed_randomness(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    # Also seed numpy and torch if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def capture_warnings():
    """Context manager to capture warnings.
    
    Yields:
        List to collect warnings
    """
    captured = []
    
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        captured.append({
            "message": str(message),
            "category": category.__name__,
            "filename": filename,
            "lineno": lineno
        })
    
    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler
    
    try:
        yield captured
    finally:
        warnings.showwarning = old_showwarning


@contextmanager
def isolate_test():
    """Context manager for test isolation.
    
    Yields:
        None
    """
    # Save state
    old_env = os.environ.copy()
    
    try:
        yield
    finally:
        # Restore state
        os.environ.clear()
        os.environ.update(old_env)


def cleanup_resources(*paths: Path) -> None:
    """Clean up test resources.
    
    Args:
        *paths: Paths to clean up
    """
    for path in paths:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
