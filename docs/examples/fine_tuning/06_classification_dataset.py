"""Classification Dataset Example

This example demonstrates how to prepare datasets for classification fine-tuning.

Main concepts:
- Creating classification training data
- Handling multi-class classification
- Binary classification tasks
- Balancing classification datasets
- Label distribution analysis
- Converting various formats to classification format

Use case: Training models for text classification tasks such as
sentiment analysis, topic categorization, intent detection, or
content moderation.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    balance_dataset,
    split_dataset,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    SplitStrategy,
)
from kerb.fine_tuning.quality import check_label_distribution
from kerb.core.enums import BalanceMethod


def create_sentiment_classification():
    """Create sentiment analysis classification dataset."""
    examples = []
    
    # Positive sentiment examples
    positive_texts = [
        "This product exceeded my expectations! Highly recommend.",
        "Absolutely love it! Best purchase I've made this year.",
        "Outstanding quality and fast shipping. Very satisfied!",
        "Great value for money. Will buy again.",
        "Impressive features and easy to use.",
        "Fantastic customer service and excellent product.",
        "Couldn't be happier with this purchase!",
        "Five stars! Everything I hoped for and more.",
    ]
    
    for text in positive_texts:
        examples.append(TrainingExample(
            prompt=f"Classify the sentiment: {text}",
            completion="positive",
            label="positive",
            metadata={"task": "sentiment"}
        ))
    
    # Negative sentiment examples
    negative_texts = [
        "Terrible quality. Broke after one day.",
        "Waste of money. Very disappointing.",
        "Poor customer service and defective product.",
        "Not as described. Returning immediately.",
        "Worst purchase ever. Don't buy this.",
        "Cheap materials and doesn't work properly.",
    ]
    
    for text in negative_texts:
        examples.append(TrainingExample(
            prompt=f"Classify the sentiment: {text}",
            completion="negative",
            label="negative",
            metadata={"task": "sentiment"}
        ))
    
    # Neutral sentiment examples
    neutral_texts = [
        "It's okay. Nothing special but works fine.",
        "Average product. Does what it's supposed to.",
        "Received as described. No complaints.",
        "Standard quality for the price.",
    ]
    
    for text in neutral_texts:
        examples.append(TrainingExample(
            prompt=f"Classify the sentiment: {text}",
            completion="neutral",
            label="neutral",
            metadata={"task": "sentiment"}
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CLASSIFICATION,
        metadata={"classification_type": "sentiment", "num_classes": 3}
    )


def create_intent_classification():
    """Create intent detection classification dataset."""
    examples = []
    
    # Booking intents
    booking_texts = [
        "I'd like to book a flight to New York",
        "Reserve a table for two at 7pm",
        "Can you schedule an appointment for tomorrow?",
        "I want to make a reservation",
        "Book a hotel room for next week",
    ]
    
    for text in booking_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="booking",
            label="booking",
            metadata={"domain": "intent"}
        ))
    
    # Information intents
    info_texts = [
        "What are your business hours?",
        "Tell me about your refund policy",
        "How much does shipping cost?",
        "Where is your store located?",
        "What payment methods do you accept?",
    ]
    
    for text in info_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="information",
            label="information",
            metadata={"domain": "intent"}
        ))
    
    # Support intents
    support_texts = [
        "My order hasn't arrived yet",
        "I need help with my account",
        "The product is not working",
        "I want to file a complaint",
        "Can someone assist me?",
    ]
    
    for text in support_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="support",
            label="support",
            metadata={"domain": "intent"}
        ))
    
    # Cancellation intents
    cancel_texts = [
        "I want to cancel my order",
        "Please cancel my reservation",
        "I'd like to unsubscribe",
    ]
    
    for text in cancel_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="cancellation",
            label="cancellation",
            metadata={"domain": "intent"}
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CLASSIFICATION,
        metadata={"classification_type": "intent", "num_classes": 4}
    )


def create_topic_classification():
    """Create topic categorization dataset."""
    examples = []
    
    topics = {
        "technology": [
            "New AI model achieves breakthrough in natural language processing",
            "Latest smartphone features advanced camera technology",
            "Cybersecurity threats continue to evolve",
        ],
        "sports": [
            "Local team wins championship in thrilling finale",
            "Olympic athlete breaks world record",
            "Coach announces retirement after successful career",
        ],
        "business": [
            "Stock market reaches new highs amid economic recovery",
            "Tech startup raises $50 million in Series B funding",
            "Merger creates industry giant in pharmaceutical sector",
        ],
        "health": [
            "Study reveals benefits of Mediterranean diet",
            "New treatment shows promise for rare disease",
            "Mental health awareness campaign launches nationwide",
        ],
    }
    
    for topic, texts in topics.items():
        for text in texts:
            examples.append(TrainingExample(
                prompt=f"Categorize this article: {text}",
                completion=topic,
                label=topic,
                metadata={"task": "topic_classification"}
            ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CLASSIFICATION,
        metadata={"classification_type": "topic", "num_classes": 4}
    )


def create_binary_classification():
    """Create binary classification dataset (spam detection)."""
    examples = []
    
    # Spam examples
    spam_texts = [
        "CONGRATULATIONS! You've won $1000! Click here now!",
        "Hot singles in your area! Meet them tonight!",
        "Enlarge your profits with this one weird trick!",
        "You have been selected for a FREE iPhone! Claim now!",
        "URGENT: Your account will be closed. Verify immediately!",
    ]
    
    for text in spam_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="spam",
            label="spam",
            metadata={"binary": True}
        ))
    
    # Ham (legitimate) examples
    ham_texts = [
        "Meeting scheduled for tomorrow at 2pm in conference room B",
        "Your package has been delivered to your doorstep",
        "Reminder: Project deadline is this Friday",
        "Thank you for your purchase. Your order number is #12345",
        "Happy birthday! Hope you have a wonderful day!",
        "Can we reschedule our call to next week?",
    ]
    
    for text in ham_texts:
        examples.append(TrainingExample(
            prompt=text,
            completion="ham",
            label="ham",
            metadata={"binary": True}
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CLASSIFICATION,
        metadata={"classification_type": "spam_detection", "num_classes": 2}
    )


def main():
    """Run classification dataset example."""
    
    print("="*80)
    print("CLASSIFICATION DATASET EXAMPLE")
    print("="*80)
    
    # Step 1: Sentiment classification
    print("\nStep 1: Creating sentiment classification dataset")
    sentiment_ds = create_sentiment_classification()
    print(f"Sentiment examples: {len(sentiment_ds)}")
    
    # Analyze label distribution
    label_stats = check_label_distribution(sentiment_ds)
    print(f"Number of classes: {label_stats['unique_labels']}")
    print("Label distribution:")
    for label, count in label_stats['label_counts'].items():
        print(f"  {label}: {count} ({label_stats['label_percentages'][label]}%)")
    print(f"Is balanced: {label_stats['is_balanced']}")
    
    # Display example
    print("\nExample sentiment classification:")
    print("-"*80)
    example = sentiment_ds.examples[0]
    print(f"Prompt: {example.prompt}")
    print(f"Label: {example.label}")
    print(f"Completion: {example.completion}")
    
    # Step 2: Intent classification
    print("\n" + "="*80)
    print("INTENT CLASSIFICATION")
    print("="*80)
    
    intent_ds = create_intent_classification()
    print(f"\nIntent examples: {len(intent_ds)}")
    
    intent_stats = check_label_distribution(intent_ds)
    print(f"Number of intent classes: {intent_stats['unique_labels']}")
    print("Intent distribution:")
    for label, count in intent_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Step 3: Topic classification
    print("\n" + "="*80)
    print("TOPIC CLASSIFICATION")
    print("="*80)
    
    topic_ds = create_topic_classification()
    print(f"\nTopic examples: {len(topic_ds)}")
    
    topic_stats = check_label_distribution(topic_ds)
    print("Topic distribution:")
    for label, count in topic_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Step 4: Binary classification
    print("\n" + "="*80)
    print("BINARY CLASSIFICATION (SPAM DETECTION)")
    print("="*80)
    
    spam_ds = create_binary_classification()
    print(f"\nSpam detection examples: {len(spam_ds)}")
    
    spam_stats = check_label_distribution(spam_ds)
    print("Binary class distribution:")
    for label, count in spam_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Step 5: Balancing imbalanced dataset
    print("\n" + "="*80)
    print("BALANCING CLASSIFICATION DATASET")
    print("="*80)
    
    print("\nOriginal sentiment distribution (imbalanced):")
    for label, count in label_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Balance using undersampling
    balanced_sentiment = balance_dataset(
        sentiment_ds,
        method=BalanceMethod.UNDERSAMPLE
    )
    
    balanced_stats = check_label_distribution(balanced_sentiment)
    print(f"\nAfter balancing (undersample): {len(balanced_sentiment)} examples")
    print("Balanced distribution:")
    for label, count in balanced_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Step 6: Split classification dataset
    print("\n" + "="*80)
    print("SPLITTING CLASSIFICATION DATASET")
    print("="*80)
    
    # Use stratified split to maintain label distribution
    train_ds, val_ds, test_ds = split_dataset(
        balanced_sentiment,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy=SplitStrategy.STRATIFIED,
        seed=42
    )
    
    print(f"\nTrain set: {len(train_ds)} examples")
    print(f"Validation set: {len(val_ds)} examples")
    print(f"Test set: {len(test_ds)} examples")
    
    # Verify stratification maintained distribution
    train_stats = check_label_distribution(train_ds)
    print("\nTrain set label distribution (should be balanced):")
    for label, count in train_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Step 7: Chat format for classification
    print("\n" + "="*80)
    print("CLASSIFICATION IN CHAT FORMAT")
    print("="*80)
    
    chat_classification = [
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are a sentiment classifier. Respond with only: positive, negative, or neutral."},
                {"role": "user", "content": "This movie was absolutely amazing!"},
                {"role": "assistant", "content": "positive"}
            ],
            label="positive"
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are a sentiment classifier. Respond with only: positive, negative, or neutral."},
                {"role": "user", "content": "I regret buying this product."},
                {"role": "assistant", "content": "negative"}
            ],
            label="negative"
        ),
    ]
    
    chat_clf_ds = TrainingDataset(
        examples=chat_classification,
        format=DatasetFormat.CHAT,
        metadata={"task": "classification_chat_format"}
    )
    
    print(f"\nChat format classification: {len(chat_clf_ds)} examples")
    print("\nExample chat classification format:")
    print("-"*80)
    for msg in chat_clf_ds.examples[0].messages:
        print(f"{msg['role']}: {msg['content']}")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nClassification datasets created:")
    print(f"  Sentiment (3-class): {len(sentiment_ds)} examples")
    print(f"  Intent (4-class): {len(intent_ds)} examples")
    print(f"  Topic (4-class): {len(topic_ds)} examples")
    print(f"  Spam detection (binary): {len(spam_ds)} examples")
    print(f"  Chat format classification: {len(chat_clf_ds)} examples")
    print(f"\nTotal classification examples: {len(sentiment_ds) + len(intent_ds) + len(topic_ds) + len(spam_ds) + len(chat_clf_ds)}")
    print("\nKey techniques demonstrated:")
    print("  - Multi-class classification")
    print("  - Binary classification")
    print("  - Label distribution analysis")
    print("  - Dataset balancing")
    print("  - Stratified splitting")
    print("  - Completion and chat formats")


if __name__ == "__main__":
    main()
