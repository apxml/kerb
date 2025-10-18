"""Snapshot testing utilities."""

import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import asdict

from .types import SnapshotData


class SnapshotManager:
    """Manage response snapshots for testing."""
    
    def __init__(self, snapshot_dir: Path = Path("tests/snapshots")):
        """Initialize snapshot manager.
        
        Args:
            snapshot_dir: Directory to store snapshots
        """
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def create_snapshot(
        self,
        name: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> SnapshotData:
        """Create a snapshot.
        
        Args:
            name: Snapshot name
            content: Content to snapshot
            metadata: Additional metadata
            
        Returns:
            SnapshotData object
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        snapshot = SnapshotData(
            name=name,
            content=content,
            hash=content_hash,
            metadata=metadata or {}
        )
        
        # Save snapshot
        filepath = self.snapshot_dir / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)
        
        return snapshot
    
    def load_snapshot(self, name: str) -> Optional[SnapshotData]:
        """Load a snapshot.
        
        Args:
            name: Snapshot name
            
        Returns:
            SnapshotData object or None
        """
        filepath = self.snapshot_dir / f"{name}.json"
        if not filepath.exists():
            return None
        
        with open(filepath) as f:
            data = json.load(f)
        return SnapshotData(**data)
    
    def compare_snapshot(
        self,
        name: str,
        content: str
    ) -> Tuple[bool, Optional[str]]:
        """Compare content against snapshot.
        
        Args:
            name: Snapshot name
            content: Content to compare
            
        Returns:
            Tuple of (matches, diff)
        """
        from .comparison import diff_responses
        
        snapshot = self.load_snapshot(name)
        if not snapshot:
            return False, "Snapshot not found"
        
        if snapshot.content == content:
            return True, None
        
        diff = diff_responses(snapshot.content, content)
        return False, diff
    
    def update_snapshot(
        self,
        name: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> SnapshotData:
        """Update an existing snapshot.
        
        Args:
            name: Snapshot name
            content: New content
            metadata: Updated metadata
            
        Returns:
            Updated SnapshotData
        """
        return self.create_snapshot(name, content, metadata)


def create_snapshot(
    name: str,
    content: str,
    snapshot_dir: Path = Path("tests/snapshots")
) -> SnapshotData:
    """Create a snapshot (convenience function)."""
    manager = SnapshotManager(snapshot_dir)
    return manager.create_snapshot(name, content)


def compare_snapshot(
    name: str,
    content: str,
    snapshot_dir: Path = Path("tests/snapshots")
) -> Tuple[bool, Optional[str]]:
    """Compare against snapshot (convenience function)."""
    manager = SnapshotManager(snapshot_dir)
    return manager.compare_snapshot(name, content)


def update_snapshot(
    name: str,
    content: str,
    snapshot_dir: Path = Path("tests/snapshots")
) -> SnapshotData:
    """Update snapshot (convenience function)."""
    manager = SnapshotManager(snapshot_dir)
    return manager.update_snapshot(name, content)
