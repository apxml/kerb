"""Response recording and replay functionality."""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

from .mocking import MockLLM
from .types import MockBehavior


class ResponseRecorder:
    """Record actual LLM responses for replay."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize recorder.
        
        Args:
            output_dir: Directory to save recordings
        """
        self.output_dir = output_dir or Path("tests/recordings")
        self.recordings: List[Dict[str, Any]] = []
        self.recording_enabled = False
    
    def start_recording(self) -> None:
        """Start recording responses."""
        self.recording_enabled = True
        self.recordings = []
    
    def stop_recording(self) -> None:
        """Stop recording responses."""
        self.recording_enabled = False
    
    def record(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> None:
        """Record a prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: LLM response
            metadata: Additional metadata
        """
        if self.recording_enabled:
            self.recordings.append({
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            })
    
    def save_recording(self, name: str) -> Path:
        """Save recordings to disk.
        
        Args:
            name: Recording session name
            
        Returns:
            Path to saved recording
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"{name}.json"
        
        with open(filepath, "w") as f:
            json.dump(self.recordings, f, indent=2)
        
        return filepath
    
    def load_recording(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load recordings from disk.
        
        Args:
            filepath: Path to recording file
            
        Returns:
            List of recorded interactions
        """
        with open(filepath) as f:
            self.recordings = json.load(f)
        return self.recordings


@contextmanager
def RecordingSession(
    recorder: ResponseRecorder,
    name: str,
    auto_save: bool = True
):
    """Context manager for recording sessions.
    
    Args:
        recorder: ResponseRecorder instance
        name: Session name
        auto_save: Automatically save on exit
        
    Yields:
        ResponseRecorder instance
    """
    recorder.start_recording()
    try:
        yield recorder
    finally:
        recorder.stop_recording()
        if auto_save:
            recorder.save_recording(name)


def replay_responses(recording_file: Path) -> MockLLM:
    """Create a MockLLM from recorded responses.
    
    Args:
        recording_file: Path to recording file
        
    Returns:
        MockLLM configured with recorded responses
    """
    with open(recording_file) as f:
        recordings = json.load(f)
    
    # Create pattern-based responses
    pattern_responses = {
        rec["prompt"]: rec["response"]
        for rec in recordings
    }
    
    return MockLLM(
        responses=pattern_responses,
        behavior=MockBehavior.PATTERN
    )
