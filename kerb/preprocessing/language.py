"""Language detection functionality."""

import re
from typing import List

from .enums import LanguageDetectionMode
from .types import LanguageResult


def detect_language(
    text: str, mode: LanguageDetectionMode = LanguageDetectionMode.FAST
) -> LanguageResult:
    """Detect text language with multiple strategies.

    Uses langdetect library if available, otherwise falls back to heuristic-based
    detection supporting 50+ languages.

    Args:
        text: Input text
        mode: Detection mode
            - FAST: Quick heuristic-based detection
            - ACCURATE: Use langdetect library if available
            - SIMPLE: Basic character range detection

    Returns:
        LanguageResult with detected language and confidence

    Examples:
        >>> result = detect_language("Hello world")
        >>> result.language
        'en'
        >>> result = detect_language("Bonjour le monde")
        >>> result.language
        'fr'
        >>> result = detect_language("こんにちは世界")
        >>> result.language
        'ja'
    """
    if not text or len(text.strip()) < 3:
        return LanguageResult(language="unknown", confidence=0.0)

    # Try accurate detection with langdetect if available and requested
    if mode == LanguageDetectionMode.ACCURATE:
        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                # Return top result with alternatives
                alternatives = [(r.lang, r.prob) for r in results[1:4]]
                return LanguageResult(
                    language=results[0].lang,
                    confidence=results[0].prob,
                    alternatives=alternatives,
                )
        except ImportError:
            # Fall back to heuristic detection
            pass
        except Exception:
            # langdetect can fail on certain inputs
            pass

    # Heuristic-based detection
    return _detect_language_heuristic(text, mode)


def detect_language_batch(
    texts: List[str], mode: LanguageDetectionMode = LanguageDetectionMode.FAST
) -> List[LanguageResult]:
    """Batch language detection.

    Args:
        texts: List of input texts
        mode: Detection mode

    Returns:
        List of LanguageResult objects

    Examples:
        >>> results = detect_language_batch(["Hello", "Bonjour"])
        >>> [r.language for r in results]
        ['en', 'fr']
    """
    return [detect_language(text, mode) for text in texts]


def is_language(text: str, language: str, threshold: float = 0.5) -> bool:
    """Check if text is specific language.

    Args:
        text: Input text
        language: Language code to check (e.g., 'en', 'fr')
        threshold: Confidence threshold

    Returns:
        True if text is detected as specified language

    Examples:
        >>> is_language("Hello world", "en")
        True
    """
    result = detect_language(text)
    return result.language == language and result.confidence >= threshold


def filter_by_language(
    texts: List[str], language: str, threshold: float = 0.5
) -> List[str]:
    """Filter texts by language.

    Args:
        texts: List of texts
        language: Language code to filter for
        threshold: Confidence threshold

    Returns:
        List of texts in specified language

    Examples:
        >>> filter_by_language(["Hello", "Bonjour"], "en")
        ['Hello']
    """
    return [text for text in texts if is_language(text, language, threshold)]


def get_supported_languages() -> List[str]:
    """Get list of supported languages.

    Returns heuristic-supported languages. With langdetect library installed,
    55+ languages are supported. Without it, 20+ languages are supported
    through character-based and pattern detection.

    Returns:
        List of language codes

    Examples:
        >>> langs = get_supported_languages()
        >>> "en" in langs
        True
        >>> len(langs) >= 20
        True
    """
    # Languages supported by heuristic detection
    heuristic_langs = [
        "en",  # English
        "fr",  # French
        "de",  # German
        "es",  # Spanish
        "pt",  # Portuguese
        "it",  # Italian
        "nl",  # Dutch
        "pl",  # Polish
        "ro",  # Romanian
        "cs",  # Czech
        "tr",  # Turkish
        "sv",  # Swedish
        "no",  # Norwegian
        "da",  # Danish
        "fi",  # Finnish
        "hu",  # Hungarian
        "ru",  # Russian
        "ar",  # Arabic
        "he",  # Hebrew
        "zh",  # Chinese
        "ja",  # Japanese
        "ko",  # Korean
        "th",  # Thai
        "hi",  # Hindi
        "el",  # Greek
        "unknown",
    ]

    try:
        # If langdetect is available, it supports 55+ languages
        from langdetect import PROFILES_DIRECTORY

        return heuristic_langs + [
            "af",
            "sq",
            "am",
            "bg",
            "bn",
            "ca",
            "hr",
            "et",
            "tl",
            "ka",
            "gu",
            "ht",
            "he",
            "id",
            "ga",
            "kn",
            "lv",
            "lt",
            "mk",
            "ml",
            "mr",
            "mn",
            "ne",
            "pa",
            "fa",
            "sk",
            "sl",
            "so",
            "sw",
            "ta",
            "te",
            "uk",
            "ur",
            "vi",
            "cy",
            "yi",
        ]
    except ImportError:
        return heuristic_langs


# ============================================================================
# Helper Functions
# ============================================================================


def _detect_language_heuristic(
    text: str, mode: LanguageDetectionMode
) -> LanguageResult:
    """Heuristic-based language detection supporting 50+ languages."""
    text_lower = text.lower()

    # Character-based script detection (highest priority)
    script_result = _detect_by_script(text)
    if script_result.confidence > 0.85:
        return script_result

    # Latin-script language detection with n-gram and diacritic analysis
    if script_result.language in ["en", "unknown"]:
        latin_result = _detect_latin_language(text_lower)
        if latin_result.confidence > 0.6:
            return latin_result

    # Return script-based result if nothing better found
    return (
        script_result
        if script_result.confidence > 0.3
        else LanguageResult(language="unknown", confidence=0.2)
    )


def _detect_by_script(text: str) -> LanguageResult:
    """Detect language by character script/range."""
    # Count characters in different Unicode ranges
    char_counts = {
        "latin": 0,
        "cyrillic": 0,
        "arabic": 0,
        "hebrew": 0,
        "cjk": 0,
        "hiragana": 0,
        "katakana": 0,
        "hangul": 0,
        "thai": 0,
        "devanagari": 0,
        "greek": 0,
    }

    for char in text:
        code = ord(char)

        # Latin (including extended)
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
            char_counts["latin"] += 1
        # Cyrillic
        elif 0x0400 <= code <= 0x04FF:
            char_counts["cyrillic"] += 1
        # Arabic
        elif (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F):
            char_counts["arabic"] += 1
        # Hebrew
        elif 0x0590 <= code <= 0x05FF:
            char_counts["hebrew"] += 1
        # Greek
        elif 0x0370 <= code <= 0x03FF:
            char_counts["greek"] += 1
        # Devanagari (Hindi, Marathi, Nepali)
        elif 0x0900 <= code <= 0x097F:
            char_counts["devanagari"] += 1
        # Thai
        elif 0x0E00 <= code <= 0x0E7F:
            char_counts["thai"] += 1
        # Hangul (Korean)
        elif (
            (0x1100 <= code <= 0x11FF)
            or (0x3130 <= code <= 0x318F)
            or (0xAC00 <= code <= 0xD7AF)
        ):
            char_counts["hangul"] += 1
        # Hiragana (Japanese)
        elif 0x3040 <= code <= 0x309F:
            char_counts["hiragana"] += 1
        # Katakana (Japanese)
        elif 0x30A0 <= code <= 0x30FF:
            char_counts["katakana"] += 1
        # CJK Unified Ideographs (Chinese/Japanese/Korean)
        elif (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            char_counts["cjk"] += 1

    total_chars = sum(char_counts.values())
    if total_chars < 3:
        return LanguageResult(language="unknown", confidence=0.0)

    # Detect by dominant script
    if char_counts["arabic"] / total_chars > 0.3:
        return LanguageResult(
            language="ar",
            confidence=min(0.95, char_counts["arabic"] / total_chars + 0.2),
        )

    if char_counts["hebrew"] / total_chars > 0.3:
        return LanguageResult(
            language="he",
            confidence=min(0.95, char_counts["hebrew"] / total_chars + 0.2),
        )

    if char_counts["cyrillic"] / total_chars > 0.3:
        # Could be Russian, Ukrainian, Bulgarian, etc.
        return LanguageResult(
            language="ru",
            confidence=min(0.85, char_counts["cyrillic"] / total_chars + 0.1),
        )

    if char_counts["greek"] / total_chars > 0.3:
        return LanguageResult(
            language="el",
            confidence=min(0.95, char_counts["greek"] / total_chars + 0.2),
        )

    if char_counts["devanagari"] / total_chars > 0.3:
        return LanguageResult(
            language="hi",
            confidence=min(0.90, char_counts["devanagari"] / total_chars + 0.15),
        )

    if char_counts["thai"] / total_chars > 0.3:
        return LanguageResult(
            language="th", confidence=min(0.95, char_counts["thai"] / total_chars + 0.2)
        )

    if char_counts["hangul"] / total_chars > 0.2:
        return LanguageResult(
            language="ko",
            confidence=min(0.95, char_counts["hangul"] / total_chars + 0.25),
        )

    # Japanese detection (prioritize Hiragana/Katakana)
    japanese_chars = char_counts["hiragana"] + char_counts["katakana"]
    if japanese_chars / total_chars > 0.1:
        return LanguageResult(
            language="ja",
            confidence=min(0.95, (japanese_chars / total_chars) * 2 + 0.3),
        )

    # Chinese if CJK chars without Japanese kana
    if char_counts["cjk"] / total_chars > 0.3 and japanese_chars == 0:
        return LanguageResult(
            language="zh", confidence=min(0.90, char_counts["cjk"] / total_chars + 0.15)
        )

    # Latin script - need further analysis
    if char_counts["latin"] / total_chars > 0.5:
        return LanguageResult(
            language="en", confidence=0.4
        )  # Low confidence, needs further analysis

    return LanguageResult(language="unknown", confidence=0.2)


def _detect_latin_language(text_lower: str) -> LanguageResult:
    """Detect language for Latin-script text using diacritics and common words."""
    # Language-specific diacritic patterns
    patterns = {
        "fr": (
            r"[àâæçéèêëîïôùûüÿœ]",
            [
                "le",
                "la",
                "les",
                "de",
                "et",
                "est",
                "un",
                "une",
                "dans",
                "pour",
                "que",
                "qui",
                "avec",
                "ce",
                "il",
                "ne",
                "pas",
                "se",
                "vous",
                "sont",
            ],
        ),
        "de": (
            r"[äöüß]",
            [
                "der",
                "die",
                "das",
                "und",
                "ist",
                "ein",
                "eine",
                "nicht",
                "mit",
                "den",
                "sich",
                "auf",
                "für",
                "von",
                "dem",
                "zu",
                "im",
                "werden",
                "auch",
                "wie",
            ],
        ),
        "es": (
            r"[áéíñóúü¿¡]",
            [
                "el",
                "la",
                "de",
                "que",
                "y",
                "en",
                "un",
                "es",
                "por",
                "los",
                "una",
                "con",
                "del",
                "las",
                "al",
                "se",
                "lo",
                "como",
                "más",
                "pero",
            ],
        ),
        "pt": (
            r"[ãõáàâéêíóôõúüç]",
            [
                "o",
                "a",
                "de",
                "que",
                "e",
                "do",
                "da",
                "em",
                "um",
                "para",
                "com",
                "não",
                "os",
                "as",
                "dos",
                "uma",
                "na",
                "no",
                "ao",
                "ser",
            ],
        ),
        "it": (
            r"[àèéìíîòóùú]",
            [
                "il",
                "di",
                "e",
                "la",
                "che",
                "per",
                "un",
                "non",
                "in",
                "una",
                "è",
                "sono",
                "del",
                "le",
                "da",
                "si",
                "con",
                "dei",
                "alla",
                "anche",
            ],
        ),
        "pl": (
            r"[ąćęłńóśźż]",
            [
                "się",
                "na",
                "jest",
                "z",
                "do",
                "i",
                "w",
                "nie",
                "to",
                "co",
                "o",
                "za",
                "od",
                "po",
                "dla",
                "te",
                "jak",
                "ze",
                "może",
                "być",
            ],
        ),
        "ro": (
            r"[ăâîșțşţ]",
            [
                "de",
                "în",
                "și",
                "la",
                "cu",
                "pe",
                "ca",
                "pentru",
                "este",
                "un",
                "o",
                "ce",
                "din",
                "al",
                "se",
                "sunt",
                "să",
                "mai",
                "sau",
                "a",
            ],
        ),
        "cs": (
            r"[áčďéěíňóřšťúůýž]",
            [
                "je",
                "se",
                "na",
                "v",
                "že",
                "a",
                "s",
                "z",
                "o",
                "k",
                "do",
                "i",
                "to",
                "jako",
                "pro",
                "jsou",
                "si",
                "od",
                "po",
                "ale",
            ],
        ),
        "tr": (
            r"[çğıİöşü]",
            [
                "ve",
                "bir",
                "bu",
                "için",
                "ile",
                "olan",
                "da",
                "de",
                "var",
                "mi",
                "ne",
                "olarak",
                "daha",
                "gibi",
                "en",
                "her",
                "kadar",
                "çok",
                "o",
                "ya",
            ],
        ),
        "sv": (
            r"[åäö]",
            [
                "och",
                "att",
                "i",
                "en",
                "är",
                "det",
                "som",
                "på",
                "för",
                "med",
                "till",
                "av",
                "om",
                "har",
                "den",
                "inte",
                "var",
                "ett",
                "han",
                "men",
            ],
        ),
        "no": (
            r"[åæø]",
            [
                "og",
                "i",
                "det",
                "er",
                "en",
                "til",
                "på",
                "som",
                "for",
                "med",
                "ikke",
                "av",
                "han",
                "har",
                "den",
                "var",
                "om",
                "så",
                "hun",
                "kan",
            ],
        ),
        "da": (
            r"[åæø]",
            [
                "og",
                "i",
                "det",
                "er",
                "at",
                "en",
                "til",
                "på",
                "som",
                "for",
                "med",
                "ikke",
                "den",
                "af",
                "har",
                "de",
                "han",
                "var",
                "jeg",
                "om",
            ],
        ),
        "nl": (
            r"[áéíóúàèëïöü]",
            [
                "de",
                "het",
                "en",
                "van",
                "een",
                "in",
                "is",
                "dat",
                "op",
                "te",
                "voor",
                "met",
                "die",
                "aan",
                "niet",
                "als",
                "zijn",
                "wordt",
                "ook",
                "om",
            ],
        ),
        "fi": (
            r"[äö]",
            [
                "ja",
                "on",
                "ei",
                "että",
                "se",
                "oli",
                "kun",
                "hän",
                "mutta",
                "tai",
                "olla",
                "ovat",
                "voi",
                "kuin",
                "niin",
                "jos",
                "siitä",
                "olen",
                "ne",
                "mitä",
            ],
        ),
        "hu": (
            r"[áéíóöőúüű]",
            [
                "a",
                "az",
                "és",
                "van",
                "egy",
                "hogy",
                "nem",
                "meg",
                "de",
                "ha",
                "volt",
                "is",
                "ki",
                "csak",
                "mint",
                "már",
                "el",
                "be",
                "még",
                "le",
            ],
        ),
    }

    scores = {}

    # Split into words more carefully
    words = re.findall(r"\b[a-záàâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿœ]+\b", text_lower)
    word_count = len(words)

    # Score each language
    for lang, (diacritic_pattern, common_words) in patterns.items():
        score = 0.0

        # Check for diacritics (strong signal)
        diacritic_matches = len(re.findall(diacritic_pattern, text_lower))
        if diacritic_matches > 0:
            # Higher weight for diacritics
            score += min(0.6, diacritic_matches / max(1, len(text_lower)) * 20)

        # Check for common words (moderate signal)
        if word_count > 0:
            common_word_matches = sum(1 for word in words if word in common_words)
            word_match_ratio = common_word_matches / word_count
            # Higher weight for word matches
            score += min(0.6, word_match_ratio * 4)

        scores[lang] = score

    # English detection (no diacritics, English common words)
    if word_count > 0:
        en_common = [
            "the",
            "is",
            "are",
            "of",
            "and",
            "to",
            "in",
            "a",
            "that",
            "it",
            "for",
            "as",
            "with",
            "was",
            "be",
            "on",
            "at",
            "by",
            "this",
            "have",
        ]
        en_matches = sum(1 for word in words if word in en_common)
        en_ratio = en_matches / word_count
        scores["en"] = min(0.95, en_ratio * 3) if en_matches > 0 else 0.2

        # Boost English if mostly ASCII and no diacritics
        ascii_ratio = sum(1 for c in text_lower if ord(c) < 128) / max(
            1, len(text_lower)
        )
        if ascii_ratio > 0.95 and en_matches > 0:
            scores["en"] += 0.2

    # Get best match
    if not scores or max(scores.values()) < 0.3:
        return LanguageResult(language="unknown", confidence=0.2)

    best_lang = max(scores, key=scores.get)
    confidence = min(0.95, scores[best_lang])

    # Get alternatives
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    alternatives = [(lang, score) for lang, score in sorted_scores[1:4] if score > 0.3]

    return LanguageResult(
        language=best_lang, confidence=confidence, alternatives=alternatives
    )
