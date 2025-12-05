"""
JSONL Utilities Example
=======================

This example demonstrates how to work with JSONL files for fine-tuning.

Main concepts:
- Writing datasets to JSONL format
- Reading JSONL files
- Appending to existing JSONL files
- Merging multiple JSONL files
- Validating JSONL file format
- Streaming large JSONL files efficiently
- Counting lines in JSONL files

Use case: Managing large-scale fine-tuning datasets, exporting data
for different providers, and efficiently processing datasets that don't
fit in memory.
"""

from kerb.fine_tuning import (
    write_jsonl,
    read_jsonl,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    to_openai_format,
)
from kerb.fine_tuning.jsonl import (
    append_jsonl,
    merge_jsonl,
    validate_jsonl,
    count_jsonl_lines,
    stream_jsonl,
)
import tempfile
import os


def create_sample_datasets():
    """Create sample datasets for demonstration."""
    # Dataset 1: Coding Q&A
    coding_examples = []
    for i in range(10):
        coding_examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": f"How do I use Python feature {i}?"},
                {"role": "assistant", "content": f"Here's how to use feature {i}: example_code()"}
            ],
            metadata={"category": "coding", "index": i}
        ))
    
    coding_dataset = TrainingDataset(
        examples=coding_examples,
        format=DatasetFormat.CHAT,
        metadata={"source": "coding_qa"}
    )
    
    # Dataset 2: Math problems
    math_examples = []
    for i in range(10):
        math_examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": f"Solve: {i} + {i} = ?"},
                {"role": "assistant", "content": f"The answer is {i + i}"}
            ],
            metadata={"category": "math", "index": i}
        ))
    
    math_dataset = TrainingDataset(
        examples=math_examples,
        format=DatasetFormat.CHAT,
        metadata={"source": "math_problems"}
    )
    
    return coding_dataset, math_dataset


def main():
    """Run JSONL utilities example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("JSONL UTILITIES EXAMPLE")
    print("="*80)
    
    # Create temporary directory for all file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Step 1: Create sample datasets
        print("\nStep 1: Creating sample datasets")
        coding_ds, math_ds = create_sample_datasets()
        print(f"Coding dataset: {len(coding_ds)} examples")
        print(f"Math dataset: {len(math_ds)} examples")
        
        # Step 2: Write datasets to JSONL
        print("\n" + "="*80)
        print("WRITING TO JSONL")
        print("="*80)
        
        coding_file = os.path.join(temp_dir, "coding_qa.jsonl")
        math_file = os.path.join(temp_dir, "math_problems.jsonl")
        
        # Convert to OpenAI format
        coding_data = to_openai_format(coding_ds)
        math_data = to_openai_format(math_ds)
        
        write_jsonl(coding_data, coding_file)
        print(f"\nWrote {len(coding_data)} examples to: coding_qa.jsonl")
        
        write_jsonl(math_data, math_file)
        print(f"Wrote {len(math_data)} examples to: math_problems.jsonl")
        
        # Step 3: Read JSONL files
        print("\n" + "="*80)
        print("READING FROM JSONL")
        print("="*80)
        
        loaded_coding = read_jsonl(coding_file)
        print(f"\nLoaded {len(loaded_coding)} examples from coding_qa.jsonl")
        
        print("\nFirst example from coding_qa.jsonl:")
        print("-"*40)
        if loaded_coding:
            example = loaded_coding[0]
            if "messages" in example:
                for msg in example["messages"][:2]:
                    content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                    print(f"  {msg['role']}: {content}")
        
        # Step 4: Count lines in JSONL
        print("\n" + "="*80)
        print("COUNTING JSONL LINES")
        print("="*80)
        
        coding_count = count_jsonl_lines(coding_file)
        math_count = count_jsonl_lines(math_file)
        
        print(f"\nLines in coding_qa.jsonl: {coding_count}")
        print(f"Lines in math_problems.jsonl: {math_count}")
        
        # Step 5: Append to JSONL
        print("\n" + "="*80)
        print("APPENDING TO JSONL")
        print("="*80)
        
        # Create additional examples
        new_examples = [
            {"messages": [
                {"role": "user", "content": "New question about Python?"},
                {"role": "assistant", "content": "Here's the answer to the new question."}
            ]}
        ]
        
        print(f"\nAppending {len(new_examples)} examples to coding_qa.jsonl")
        append_jsonl(new_examples, coding_file)
        
        new_count = count_jsonl_lines(coding_file)
        print(f"Original count: {coding_count}")
        print(f"After append: {new_count}")
        print(f"Added: {new_count - coding_count} examples")
        
        # Step 6: Merge JSONL files
        print("\n" + "="*80)
        print("MERGING JSONL FILES")
        print("="*80)
        
        merged_file = os.path.join(temp_dir, "merged_training.jsonl")
        merge_jsonl([coding_file, math_file], merged_file)
        
        merged_count = count_jsonl_lines(merged_file)
        print(f"\nMerged files into: merged_training.jsonl")
        print(f"Coding examples: {new_count}")
        print(f"Math examples: {math_count}")
        print(f"Total in merged file: {merged_count}")
        
        # Step 7: Validate JSONL files
        print("\n" + "="*80)
        print("VALIDATING JSONL FILES")
        print("="*80)
        
        # Validate valid file
        print("\nValidating coding_qa.jsonl:")
        result = validate_jsonl(coding_file)
        print(f"  Is valid: {result.is_valid}")
        print(f"  Total examples: {result.total_examples}")
        print(f"  Valid examples: {result.valid_examples}")
        print(f"  Invalid examples: {result.invalid_examples}")
        
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for error in result.errors[:3]:
                print(f"    - {error}")
        
        # Create invalid JSONL for demonstration
        invalid_file = os.path.join(temp_dir, "invalid.jsonl")
        with open(invalid_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        print("\nValidating invalid.jsonl (contains errors):")
        result = validate_jsonl(invalid_file)
        print(f"  Is valid: {result.is_valid}")
        print(f"  Total examples: {result.total_examples}")
        print(f"  Valid examples: {result.valid_examples}")
        print(f"  Invalid examples: {result.invalid_examples}")
        
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"    - {error}")
        
        # Step 8: Stream large JSONL files
        print("\n" + "="*80)
        print("STREAMING LARGE JSONL FILES")
        print("="*80)
        
        # Create a larger file for streaming demo
        large_file = os.path.join(temp_dir, "large_dataset.jsonl")
        large_examples = []
        for i in range(50):
            large_examples.append({
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"}
                ]
            })
        write_jsonl(large_examples, large_file)
        
        print(f"\nCreated large dataset: {len(large_examples)} examples")
        print("\nStreaming in batches of 10:")
        
        batch_num = 0
        total_processed = 0
        for batch in stream_jsonl(large_file, batch_size=10):
            batch_num += 1
            total_processed += len(batch)
            print(f"  Batch {batch_num}: {len(batch)} examples (total: {total_processed})")
        
        # Step 9: Export for different providers
        print("\n" + "="*80)
        print("PROVIDER-SPECIFIC EXPORTS")
        print("="*80)
        
        # OpenAI format (already done)
        openai_file = os.path.join(temp_dir, "openai_format.jsonl")
        write_jsonl(coding_data, openai_file)
        print(f"\nOpenAI format: openai_format.jsonl ({len(coding_data)} examples)")
        
        # Generic format with metadata
        generic_file = os.path.join(temp_dir, "generic_format.jsonl")
        generic_data = [ex.to_dict() for ex in coding_ds.examples]
        write_jsonl(generic_data, generic_file)
        print(f"Generic format: generic_format.jsonl ({len(generic_data)} examples)")
        
        print("\nGeneric format preserves metadata:")
        loaded_generic = read_jsonl(generic_file)
        if loaded_generic and "metadata" in loaded_generic[0]:
            print(f"  Metadata: {loaded_generic[0]['metadata']}")
        
        # Step 10: Best practices summary
        print("\n" + "="*80)
        print("BEST PRACTICES")
        print("="*80)
        
        print("\n1. File Management:")
        print("   - Use meaningful filenames (train.jsonl, val.jsonl, test.jsonl)")
        print("   - Keep train/validation/test in separate files")
        print("   - Include version or date in filename for tracking")
        
        print("\n2. Large Files:")
        print("   - Use streaming for files > 1GB")
        print("   - Process in batches to avoid memory issues")
        print("   - Validate before starting expensive operations")
        
        print("\n3. Data Integrity:")
        print("   - Always validate JSONL before training")
        print("   - Check line counts after operations")
        print("   - Keep backups of original data")
        
        print("\n4. Provider Formats:")
        print("   - Export in provider-specific format")
        print("   - Test with small sample before full export")
        print("   - Keep generic format for portability")
        
        # Step 11: Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nFiles created: 7")
        print(f"Total examples written: {new_count + math_count + len(large_examples) + len(generic_data)}")
        print(f"Batches streamed: {batch_num}")
        print("\nOperations demonstrated:")
        print("  - Writing JSONL files")
        print("  - Reading JSONL files")
        print("  - Appending to files")
        print("  - Merging multiple files")
        print("  - Validating file format")
        print("  - Counting lines")
        print("  - Streaming large files")
        print("  - Provider-specific exports")


if __name__ == "__main__":
    main()
