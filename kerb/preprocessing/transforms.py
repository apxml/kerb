"""Text transformation operations."""

import re
from typing import List, Optional


def expand_contractions(text: str) -> str:
    """Expand English contractions.

    Args:
        text: Input text with contractions

    Returns:
        Text with expanded contractions

    Examples:
        >>> expand_contractions("I'm doesn't can't")
        "I am does not cannot"
    """
    if not text:
        return text

    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'll": "I will",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }

    result = text
    for contraction, expansion in contractions.items():
        # Case-insensitive replacement with word boundaries
        pattern = r"\b" + re.escape(contraction) + r"\b"
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

    return result


def standardize_numbers(text: str) -> str:
    """Convert number words to digits.

    Args:
        text: Input text

    Returns:
        Text with standardized numbers

    Examples:
        >>> standardize_numbers("I have three apples and five oranges")
        'I have 3 apples and 5 oranges'
    """
    if not text:
        return text

    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
        "thousand": "1000",
    }

    result = text
    for word, digit in number_words.items():
        pattern = r"\b" + word + r"\b"
        result = re.sub(pattern, digit, result, flags=re.IGNORECASE)

    return result


def standardize_dates(text: str) -> str:
    """Normalize date formats.

    Args:
        text: Input text with dates

    Returns:
        Text with standardized dates (YYYY-MM-DD)

    Examples:
        >>> standardize_dates("Meeting on 12/25/2024")
        'Meeting on 2024-12-25'
    """
    if not text:
        return text

    result = text

    # Match MM/DD/YYYY format
    result = re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
        result,
    )

    # Match DD-MM-YYYY format
    result = re.sub(
        r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",
        lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        result,
    )

    return result


def extract_entities(text: str, entity_type: Optional[str] = None) -> List[str]:
    """Extract named entities (basic).

    Args:
        text: Input text
        entity_type: Type of entities to extract (None for all)

    Returns:
        List of extracted entities

    Examples:
        >>> extract_entities("Apple Inc. is in California")
        ['Apple Inc.', 'California']
    """
    if not text:
        return []

    # Simple pattern-based entity extraction
    entities = []

    # Capitalized words (potential names/places)
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    entities.extend(capitalized)

    # Organizations (with Inc., LLC, etc.)
    orgs = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Ltd)\.?\b", text
    )
    entities.extend(orgs)

    return list(set(entities))


def segment_sentences(text: str) -> List[str]:
    """Sentence segmentation.

    Args:
        text: Input text

    Returns:
        List of sentences

    Examples:
        >>> segment_sentences("Hello world. How are you?")
        ['Hello world.', 'How are you?']
    """
    if not text:
        return []

    # Split on sentence terminators followed by space and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def segment_words(text: str) -> List[str]:
    """Word segmentation (tokenization).

    Args:
        text: Input text

    Returns:
        List of words

    Examples:
        >>> segment_words("Hello, world!")
        ['Hello', 'world']
    """
    if not text:
        return []

    # Split on whitespace and punctuation
    words = re.findall(r"\b\w+\b", text)
    return words
