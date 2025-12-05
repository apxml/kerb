"""
Format Conversion Example
=========================

This example demonstrates how to convert datasets between different provider formats.

Main concepts:
- Converting to OpenAI fine-tuning format
- Converting to Anthropic format
- Converting to Google AI format
- Converting to HuggingFace format
- Loading data from CSV/JSON files
- Writing datasets to JSONL files

Use case: Preparing the same dataset for multiple LLM providers or
converting between different data formats for various training platforms.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    to_openai_format,
    to_anthropic_format,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    FineTuningProvider,
    write_jsonl,
    read_jsonl,
)
from kerb.fine_tuning.formats import (
    to_google_format,
    to_huggingface_format,
    to_generic_format,
)
import json
import tempfile
import os


def create_sample_chat_dataset():
    """Create a sample chat dataset."""
    examples = [
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are an expert SQL tutor."},
                {"role": "user", "content": "How do I select all columns from a table?"},
                {"role": "assistant", "content": "Use SELECT * FROM table_name to select all columns."}
            ]
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are an expert SQL tutor."},
                {"role": "user", "content": "How do I filter rows?"},
                {"role": "assistant", "content": "Use WHERE clause: SELECT * FROM table WHERE condition."}
            ]
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are an expert SQL tutor."},
                {"role": "user", "content": "What is a JOIN?"},
                {"role": "assistant", "content": "JOIN combines rows from two or more tables based on a related column."}
            ]
        ),
    ]
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        provider=None  # Will convert to different providers
    )


def main():
    """Run format conversion example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("FORMAT CONVERSION EXAMPLE")
    print("="*80)
    
    # Step 1: Create sample dataset
    print("\nStep 1: Creating sample chat dataset")
    dataset = create_sample_chat_dataset()
    print(f"Created dataset: {len(dataset)} examples")
    print(f"Format: {dataset.format.value}")
    
    # Step 2: Convert to OpenAI format
    print("\n" + "="*80)
    print("OPENAI FORMAT")
    print("="*80)
    
    openai_data = to_openai_format(dataset)
    print(f"\nConverted to OpenAI format: {len(openai_data)} examples")
    print("\nExample OpenAI format:")
    print("-"*80)
    print(json.dumps(openai_data[0], indent=2))
    
    # Step 3: Convert to Anthropic format
    print("\n" + "="*80)
    print("ANTHROPIC FORMAT")
    print("="*80)
    
    anthropic_data = to_anthropic_format(dataset)
    print(f"\nConverted to Anthropic format: {len(anthropic_data)} examples")
    print("\nExample Anthropic format:")
    print("-"*80)
    print(json.dumps(anthropic_data[0], indent=2))
    
    # Step 4: Convert to Google AI format
    print("\n" + "="*80)
    print("GOOGLE AI FORMAT")
    print("="*80)
    
    google_data = to_google_format(dataset)
    print(f"\nConverted to Google format: {len(google_data)} examples")
    print("\nExample Google format:")
    print("-"*80)
    print(json.dumps(google_data[0], indent=2))
    
    # Step 5: Convert to HuggingFace format
    print("\n" + "="*80)
    print("HUGGINGFACE FORMAT")
    print("="*80)
    
    hf_data = to_huggingface_format(dataset)
    print(f"\nConverted to HuggingFace format: {len(hf_data)} examples")
    print("\nExample HuggingFace format:")
    print("-"*80)
    print(json.dumps(hf_data[0], indent=2))
    
    # Step 6: Write to JSONL files
    print("\n" + "="*80)
    print("WRITING TO JSONL FILES")
    print("="*80)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write OpenAI format
        openai_file = os.path.join(temp_dir, "openai_format.jsonl")
        write_jsonl(openai_data, openai_file)
        print(f"\nWrote OpenAI format to: openai_format.jsonl")
        
        # Write Anthropic format
        anthropic_file = os.path.join(temp_dir, "anthropic_format.jsonl")
        write_jsonl(anthropic_data, anthropic_file)
        print(f"Wrote Anthropic format to: anthropic_format.jsonl")
        
        # Write Google format
        google_file = os.path.join(temp_dir, "google_format.jsonl")
        write_jsonl(google_data, google_file)
        print(f"Wrote Google format to: google_format.jsonl")
        
        # Write HuggingFace format
        hf_file = os.path.join(temp_dir, "huggingface_format.jsonl")
        write_jsonl(hf_data, hf_file)
        print(f"Wrote HuggingFace format to: huggingface_format.jsonl")
        
        # Step 7: Read back from JSONL
        print("\n" + "="*80)
        print("READING FROM JSONL")
        print("="*80)
        
        read_data = read_jsonl(openai_file)
        print(f"\nRead {len(read_data)} examples from OpenAI JSONL")
        print("First example messages:")
        if read_data and "messages" in read_data[0]:
            for msg in read_data[0]["messages"][:2]:
                print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Step 8: Completion format conversion
    print("\n" + "="*80)
    print("COMPLETION FORMAT CONVERSION")
    print("="*80)
    
    completion_examples = [
        TrainingExample(
            prompt="Translate to French: Hello",
            completion="Bonjour"
        ),
        TrainingExample(
            prompt="Translate to French: Goodbye",
            completion="Au revoir"
        ),
    ]
    
    completion_dataset = TrainingDataset(
        examples=completion_examples,
        format=DatasetFormat.COMPLETION
    )
    
    print(f"\nCompletion dataset: {len(completion_dataset)} examples")
    
    # Convert completion to chat format for OpenAI
    print("\nConverting completion format to OpenAI chat format:")
    openai_completion = to_openai_format(completion_dataset)
    print(json.dumps(openai_completion[0], indent=2))
    
    # Step 9: Generic format (preserves all metadata)
    print("\n" + "="*80)
    print("GENERIC FORMAT (WITH METADATA)")
    print("="*80)
    
    dataset_with_metadata = TrainingDataset(
        examples=[
            TrainingExample(
                messages=[
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."}
                ],
                label="python",
                metadata={"source": "documentation", "quality": "high"}
            )
        ],
        format=DatasetFormat.CHAT
    )
    
    generic_data = to_generic_format(dataset_with_metadata)
    print("\nGeneric format (preserves metadata):")
    print(json.dumps(generic_data[0], indent=2))
    
    # Step 10: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nSupported conversions:")
    print("  - OpenAI (messages format)")
    print("  - Anthropic (messages format)")
    print("  - Google AI (contents format)")
    print("  - HuggingFace (text format)")
    print("  - Generic JSONL (preserves all data)")
    print("\nAll formats support:")
    print("  - Chat format (multi-turn conversations)")
    print("  - Completion format (prompt-completion pairs)")
    print("  - JSONL file I/O")
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
