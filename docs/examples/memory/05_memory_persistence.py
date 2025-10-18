"""Memory Persistence Example

This example demonstrates how to save and load conversation memory to/from disk,
enabling conversation continuity across sessions.

Main concepts:
- Saving conversation buffers to JSON files
- Loading conversation history from disk
- Merging multiple conversation sessions
- Backup and recovery strategies
"""

import os
import json
from pathlib import Path
from kerb.memory import (
    ConversationBuffer,
    save_conversation,
    load_conversation
)
from kerb.memory.utils import merge_conversations
from kerb.core.types import Message


def main():
    """Run memory persistence example."""
    
    print("="*80)
    print("MEMORY PERSISTENCE EXAMPLE")
    print("="*80)
    
    # Create temporary directory for examples
    temp_dir = Path("temp_memory_examples")
    temp_dir.mkdir(exist_ok=True)
    
    # Create and populate a conversation buffer
    print("\n" + "-"*80)
    print("CREATING CONVERSATION")
    print("-"*80)
    
    buffer = ConversationBuffer(
        max_messages=100,
        enable_summaries=True,
        enable_entity_tracking=True
    )
    
    # Simulate a conversation
    conversation = [
        ("user", "I'm working on a machine learning project using scikit-learn."),
        ("assistant", "Great! Scikit-learn is excellent for ML. What kind of problem are you solving?"),
        ("user", "I need to classify customer reviews as positive or negative."),
        ("assistant", "That's a sentiment analysis task. I'd recommend starting with TfidfVectorizer for features and LogisticRegression or SVM for classification."),
        ("user", "How do I handle imbalanced data?"),
        ("assistant", "Use techniques like SMOTE for oversampling, class weights in your model, or stratified sampling. Scikit-learn supports class_weight='balanced' parameter."),
    ]
    
    for role, content in conversation:
        buffer.add_message(role, content)
    
    print(f"Created buffer with {len(buffer.messages)} messages")
    if buffer.entities:
        print(f"Tracked {len(buffer.entities)} entities")
    
    # Save conversation to file
    print("\n" + "-"*80)
    print("SAVING CONVERSATION")
    print("-"*80)
    
    save_path = temp_dir / "ml_conversation.json"
    buffer.save(str(save_path))
    
    print(f"Saved conversation to: {save_path}")
    print(f"File size: {save_path.stat().st_size} bytes")
    
    # Inspect saved file
    with open(save_path) as f:
        saved_data = json.load(f)
    
    print(f"\nSaved data structure:")
    print(f"  - messages: {len(saved_data['messages'])}")
    print(f"  - summaries: {len(saved_data['summaries'])}")
    print(f"  - entities: {len(saved_data['entities'])}")
    print(f"  - config: {list(saved_data['config'].keys())}")
    
    # Load conversation from file
    print("\n" + "-"*80)
    print("LOADING CONVERSATION")
    print("-"*80)
    
    loaded_buffer = ConversationBuffer.load(str(save_path))
    
    print(f"Loaded buffer:")
    print(f"  Messages: {len(loaded_buffer.messages)}")
    print(f"  Entities: {len(loaded_buffer.entities)}")
    print(f"  Max messages: {loaded_buffer.max_messages}")
    print(f"  Window size: {loaded_buffer.window_size}")
    
    # Verify data integrity
    print(f"\nData integrity check:")
    print(f"  Original messages == Loaded messages: {len(buffer.messages) == len(loaded_buffer.messages)}")
    print(f"  Original entities == Loaded entities: {len(buffer.entities) == len(loaded_buffer.entities)}")
    
    # Continue conversation from loaded state
    print("\n" + "-"*80)
    print("CONTINUING CONVERSATION")
    print("-"*80)
    
    loaded_buffer.add_message("user", "What about feature engineering for text?")
    loaded_buffer.add_message("assistant", "Consider n-grams, word embeddings like Word2Vec, or modern transformers. For scikit-learn, use CountVectorizer with ngram_range parameter.")
    
    print(f"Added 2 new messages")
    print(f"Total messages now: {len(loaded_buffer.messages)}")
    
    # Save updated conversation
    updated_path = temp_dir / "ml_conversation_updated.json"
    loaded_buffer.save(str(updated_path))
    print(f"Saved updated conversation to: {updated_path}")
    
    # Create multiple conversation files
    print("\n" + "-"*80)
    print("MULTIPLE CONVERSATION SESSIONS")
    print("-"*80)
    
    # Session 1: Morning conversation
    morning_buffer = ConversationBuffer()
    morning_buffer.add_message("user", "Good morning! Let's continue working on the ML project.")
    morning_buffer.add_message("assistant", "Good morning! Ready to help with your sentiment analysis project.")
    morning_buffer.save(str(temp_dir / "session_morning.json"))
    print(f"Saved morning session: {len(morning_buffer.messages)} messages")
    
    # Session 2: Afternoon conversation
    afternoon_buffer = ConversationBuffer()
    afternoon_buffer.add_message("user", "I've trained the model. What metrics should I use?")
    afternoon_buffer.add_message("assistant", "For classification, use accuracy, precision, recall, and F1-score. Use classification_report from scikit-learn.")
    afternoon_buffer.save(str(temp_dir / "session_afternoon.json"))
    print(f"Saved afternoon session: {len(afternoon_buffer.messages)} messages")
    
    # Session 3: Evening conversation
    evening_buffer = ConversationBuffer()
    evening_buffer.add_message("user", "The model works! How do I deploy it?")
    evening_buffer.add_message("assistant", "You can use Flask/FastAPI for a REST API, or serialize with joblib/pickle for later use.")
    evening_buffer.save(str(temp_dir / "session_evening.json"))
    print(f"Saved evening session: {len(evening_buffer.messages)} messages")
    
    # Merge all sessions
    print("\n" + "-"*80)
    print("MERGING CONVERSATION SESSIONS")
    print("-"*80)
    
    merged = merge_conversations(morning_buffer, afternoon_buffer, evening_buffer, sort_by_time=True)
    
    print(f"Merged {3} sessions:")
    print(f"  Total messages: {len(merged.messages)}")
    print(f"\nMerged conversation:")
    for i, msg in enumerate(merged.messages, 1):
        print(f"  [{i}] {msg.role}: {msg.content[:70]}...")
    
    # Save merged conversation
    merged_path = temp_dir / "conversation_full_day.json"
    merged.save(str(merged_path))
    print(f"\nSaved merged conversation to: {merged_path}")
    
    # Backup strategy
    print("\n" + "-"*80)
    print("BACKUP STRATEGY")
    print("-"*80)
    
    backup_dir = temp_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"conversation_backup_{timestamp}.json"
    
    merged.save(str(backup_path))
    print(f"Created backup: {backup_path.name}")
    
    # List all backups
    backups = sorted(backup_dir.glob("*.json"))
    print(f"\nAvailable backups: {len(backups)}")
    for backup in backups:
        print(f"  - {backup.name}")
    
    # Recovery: Load latest backup
    if backups:
        latest_backup = backups[-1]
        recovered = ConversationBuffer.load(str(latest_backup))
        print(f"\nRecovered from latest backup: {len(recovered.messages)} messages")
    
    # Export in different formats
    print("\n" + "-"*80)
    print("EXPORT FORMATS")
    print("-"*80)
    
    # Export as JSON (already done)
    json_path = temp_dir / "export.json"
    merged.save(str(json_path))
    print(f"Exported as JSON: {json_path}")
    
    # Export as text
    from kerb.memory.utils import format_messages
    
    text_export = format_messages(merged.messages, format_style="detailed")
    text_path = temp_dir / "export.txt"
    text_path.write_text(text_export)
    print(f"Exported as text: {text_path}")
    
    # Export metadata
    metadata = {
        "total_messages": len(merged.messages),
        "entities": [e.to_dict() for e in merged.get_entities()],
        "summaries": [s.to_dict() for s in merged.summaries],
    }
    metadata_path = temp_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Exported metadata: {metadata_path}")
    
    # Show all created files
    print("\n" + "-"*80)
    print("CREATED FILES")
    print("-"*80)
    
    all_files = list(temp_dir.rglob("*.json")) + list(temp_dir.rglob("*.txt"))
    print(f"\nCreated {len(all_files)} files:")
    for file in sorted(all_files):
        rel_path = file.relative_to(temp_dir)
        size = file.stat().st_size
        print(f"  - {rel_path} ({size} bytes)")
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    # In a real application, you might want to keep these files
    # For this example, we'll note that cleanup is optional
    print(f"\nTemporary files created in: {temp_dir}")
    print("(In production, implement proper cleanup or retention policies)")
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
