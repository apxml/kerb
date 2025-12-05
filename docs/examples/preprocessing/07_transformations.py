"""
Text Transformations for Data Standardization
=============================================

This example demonstrates text transformation techniques to standardize
training data for consistent LLM input formats.

Main concepts:
- Expanding contractions for formal text
- Standardizing numbers and dates
- Sentence and word segmentation
- Entity extraction (basic)
- Text normalization for consistency

Use case: Standardizing diverse text sources for uniform LLM training
"""

from kerb.preprocessing import (
    expand_contractions,
    standardize_numbers,
    standardize_dates,
    segment_sentences,
    segment_words,
)


def main():
    """Run text transformation examples."""
    
    print("="*80)
    print("TEXT TRANSFORMATIONS FOR DATA STANDARDIZATION")
    print("="*80)
    
    # Example 1: Expanding contractions
    print("\n" + "-"*80)
    print("Example 1: Expanding Contractions")
    print("-"*80)
    
    informal_texts = [
        "I'm learning about AI and it's amazing!",
        "They've been working on this, but they can't finish.",
        "We'll see what happens. You're welcome to join.",
        "Don't worry, we won't let you down.",
        "She's here and he's there.",
    ]
    
    print("\nExpanding contractions for formal text:")
    for text in informal_texts:
        expanded = expand_contractions(text)
        if expanded != text:
            print(f"\nOriginal:  {text}")
            print(f"Expanded:  {expanded}")
    
    # Example 2: Standardizing numbers
    print("\n" + "-"*80)
    print("Example 2: Standardizing Numbers")
    print("-"*80)
    
    number_texts = [
        "I have three apples and five oranges.",
        "There were twenty people at the meeting.",
        "She bought twelve books and seven magazines.",
        "The project took fifteen days to complete.",
        "We need one hundred participants for the study.",
    ]
    
    print("\nConverting number words to digits:")
    for text in number_texts:
        standardized = standardize_numbers(text)
        print(f"\nOriginal:     {text}")
        print(f"Standardized: {standardized}")
    
    # Example 3: Standardizing dates
    print("\n" + "-"*80)
    print("Example 3: Standardizing Date Formats")
    print("-"*80)
    
    date_texts = [
        "The meeting is on 12/25/2024 at 3pm.",
        "Deadline: 03/15/2024 for submissions.",
        "Event scheduled for 06/30/2024.",
        "Report due on 11-20-2024.",
    ]
    
    print("\nStandardizing dates to YYYY-MM-DD:")
    for text in date_texts:
        standardized = standardize_dates(text)
        print(f"\nOriginal:     {text}")
        print(f"Standardized: {standardized}")
    
    # Example 4: Sentence segmentation
    print("\n" + "-"*80)
    print("Example 4: Sentence Segmentation")
    print("-"*80)
    
    paragraphs = [
        "Machine learning is powerful. It can solve many problems. Deep learning is a subset.",
        "Natural language processing is important. It helps computers understand text. This enables many applications.",
    ]
    
    print("\nSegmenting text into sentences:")
    for i, para in enumerate(paragraphs, 1):
        sentences = segment_sentences(para)
        print(f"\nParagraph {i}:")
        print(f"Original: {para}")
        print(f"Sentences ({len(sentences)}):")
        for j, sent in enumerate(sentences, 1):
            print(f"  {j}. {sent}")
    
    # Example 5: Word segmentation
    print("\n" + "-"*80)
    print("Example 5: Word Segmentation")
    print("-"*80)
    
    sentences = [
        "Machine learning algorithms process data.",
        "Natural language understanding is challenging.",
    ]
    
    print("\nSegmenting sentences into words:")
    for sent in sentences:
        words = segment_words(sent)
        print(f"\nSentence: {sent}")
        print(f"Words ({len(words)}): {words}")
    
    # Example 6: Combined transformations
    print("\n" + "-"*80)
    print("Example 6: Combined Transformations Pipeline")
    print("-"*80)
    
    messy_training_data = [
        "I'm working on three projects. They're due on 12/25/2024.",
        "We've got twenty datasets. The deadline's 03/15/2024.",
        "She's analyzing five models. Results won't be ready until 06/30/2024.",
    ]
    
    print("\nApplying multiple transformations:")
    for text in messy_training_data:
        # Step 1: Expand contractions
        step1 = expand_contractions(text)
        
        # Step 2: Standardize numbers
        step2 = standardize_numbers(step1)
        
        # Step 3: Standardize dates
        step3 = standardize_dates(step2)
        
        print(f"\nOriginal:   {text}")
        print(f"Step 1:     {step1}")
        print(f"Step 2:     {step2}")
        print(f"Final:      {step3}")
    
    # Example 7: Preparing instruction data
    print("\n" + "-"*80)
    print("Example 7: Standardizing Instruction-Response Pairs")
    print("-"*80)
    
    instruction_pairs = [
        {
            "instruction": "I'm looking for three datasets. Can't find them.",
            "response": "You'll find twenty datasets here. They're updated daily."
        },
        {
            "instruction": "What's the deadline for the submission?",
            "response": "It's on 12/15/2024. Don't miss it!"
        },
    ]
    
    print("\nStandardizing instruction-response pairs:")
    for i, pair in enumerate(instruction_pairs, 1):
        # Transform instruction
        inst = pair["instruction"]
        inst = expand_contractions(inst)
        inst = standardize_numbers(inst)
        inst = standardize_dates(inst)
        
        # Transform response
        resp = pair["response"]
        resp = expand_contractions(resp)
        resp = standardize_numbers(resp)
        resp = standardize_dates(resp)
        
        print(f"\nPair {i}:")
        print(f"  Original Instruction: {pair['instruction']}")
        print(f"  Clean Instruction:    {inst}")
        print(f"  Original Response:    {pair['response']}")
        print(f"  Clean Response:       {resp}")
    
    # Example 8: Creating standardization pipeline
    print("\n" + "-"*80)
    print("Example 8: Reusable Standardization Pipeline")
    print("-"*80)
    
    def standardize_text(text):
        """Apply all standardization transformations."""

# %%
# Setup and Imports
# -----------------
        text = expand_contractions(text)
        text = standardize_numbers(text)
        text = standardize_dates(text)
        return text
    
    raw_samples = [
        "I've got five tasks due on 11/30/2024.",
        "We're using three models. They won't be ready until 12-25-2024.",
        "There're twenty participants. Meeting's on 01/15/2025.",
    ]
    
    print("\nUsing reusable standardization pipeline:")
    for sample in raw_samples:
        standardized = standardize_text(sample)
        print(f"\nRaw:          {sample}")
        print(f"Standardized: {standardized}")
    
    # Example 9: Batch standardization
    print("\n" + "-"*80)
    print("Example 9: Batch Standardization")
    print("-"*80)
    
    batch = [
        "I'm analyzing three datasets from 12/01/2024.",
        "We've trained twenty models. They're ready.",
        "The results'll be published on 01/20/2025.",
        "She's got fifteen papers. They won't fit in one volume.",
    ]
    
    print(f"\nStandardizing batch of {len(batch)} samples:")
    standardized_batch = [standardize_text(text) for text in batch]
    
    for i, (original, standardized) in enumerate(zip(batch, standardized_batch), 1):
        print(f"\n{i}. Original:     {original}")
        print(f"   Standardized: {standardized}")
    
    # Example 10: Sentence splitting for context windows
    print("\n" + "-"*80)
    print("Example 10: Sentence Splitting for Context Windows")
    print("-"*80)
    
    long_text = """
    Machine learning is a subset of artificial intelligence. It enables computers 
    to learn from data. Deep learning is a specialized form of machine learning. 
    It uses neural networks with multiple layers. Natural language processing 
    applies ML to text data. It powers many modern applications.
    """
    
    # Clean up whitespace first
    long_text = " ".join(long_text.split())
    
    print("\nLong text for segmentation:")
    print(long_text)
    
    # Segment into sentences
    sentences = segment_sentences(long_text)
    
    print(f"\nSegmented into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
    
    # Group sentences for context windows (e.g., 2 sentences per chunk)
    context_size = 2
    chunks = [
        " ".join(sentences[i:i+context_size])
        for i in range(0, len(sentences), context_size)
    ]
    
    print(f"\nGrouped into {len(chunks)} chunks ({context_size} sentences each):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  {chunk}")
    
    print("\n" + "="*80)
    print("TEXT TRANSFORMATIONS COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Expand contractions for formal, consistent text")
    print("2. Standardize numbers and dates for uniform format")
    print("3. Segment sentences for chunking and analysis")
    print("4. Combine transformations in pipelines")
    print("5. Apply transformations consistently across datasets")
    print("6. Standardization improves training data quality")
    print("7. Essential for instruction-tuning datasets")


if __name__ == "__main__":
    main()
