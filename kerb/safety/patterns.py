"""Pattern definitions for safety checks.

This module contains all regex patterns and pattern lists used for detecting
various types of unsafe content.
"""

# ============================================================================
# PII Patterns
# ============================================================================

EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_PATTERN = r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b"
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
IP_ADDRESS_PATTERN = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
URL_PATTERN = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"


# ============================================================================
# Prompt Injection Patterns
# ============================================================================

# Prompt injection patterns - Multi-layered detection
INJECTION_PATTERNS = [
    # Direct instruction manipulation
    (
        r"ignore\s+(?:(?:all|any)\s+)?(?:previous|above|prior)\s+(?:instructions?|commands?|prompts?|directions?)",
        3.0,
    ),
    (
        r"disregard\s+(?:(?:all|any)\s+)?(?:previous|above|prior)\s+(?:instructions?|commands?|prompts?)",
        3.0,
    ),
    (
        r"forget\s+(?:everything|all|previous|above)(?:\s+(?:instructions?|commands?|prompts?))?",
        2.5,
    ),
    # System prompt targeting
    (r"system\s*:\s*you\s+(?:are|must|should|will)", 3.5),
    (r"(?:new|updated|revised)\s+(?:instructions?|commands?|rules?|directives?)", 2.5),
    (r"(?:instead|rather|now)\s*,?\s*(?:tell|say|write|respond|do|execute)", 2.0),
    # Temporal manipulation
    (r"from\s+(?:now|this\s+point)\s+on(?:wards)?", 2.0),
    (r"starting\s+(?:now|today|immediately)", 1.5),
    # Role manipulation
    (r"act\s+as\s+(?:if|though|a|an)\s+\w+", 2.0),
    (r"pretend\s+(?:you|to\s+be|that)", 2.0),
    (r"roleplay\s+as\s+(?:a|an)\s+\w+", 2.0),
    (r"imagine\s+(?:you\s+are|being)\s+(?:a|an)\s+\w+", 1.5),
    # Output manipulation
    (r"always\s+(?:respond|reply|answer)\s+with", 1.8),
    (r"never\s+(?:say|mention|tell|reveal)", 1.8),
]


# ============================================================================
# Jailbreak Patterns
# ============================================================================

# Jailbreak patterns - More comprehensive
JAILBREAK_PATTERNS = [
    # Known jailbreak modes
    (r"\b(?:DAN|STAN|DUDE)\s+mode\b", 4.0),
    (r"developer\s+mode\s*(?:enabled|activated|on)?", 3.5),
    (r"god\s+mode\s*(?:enabled|activated|on)?", 3.5),
    # Bypass attempts
    (r"bypass\s+(?:all\s+)?(?:restrictions?|limitations?|filters?|safeguards?)", 3.5),
    (
        r"without\s+(?:any\s+)?(?:restrictions?|limitations?|filters?|rules?|ethics?)",
        3.0,
    ),
    (r"unrestricted\s+(?:mode|access|capabilities)", 3.0),
    # Ethical override
    (r"ignore\s+(?:your\s+)?(?:programming|training|guidelines|ethics|safety)", 3.5),
    (r"override\s+(?:all\s+)?(?:safety|ethics|restrictions)", 3.5),
    (r"disable\s+(?:all\s+)?(?:safety|content|ethical)\s+(?:filters?|checks?)", 3.5),
    # Hypothetical scenarios
    (r"hypothetically\s+speaking", 1.5),
    (r"in\s+(?:a\s+)?(?:hypothetical|theoretical)\s+(?:scenario|situation|world)", 1.5),
    (r"for\s+(?:educational|research|academic)\s+purposes\s+only", 1.8),
    (r"this\s+is\s+(?:just\s+)?(?:a\s+)?(?:thought\s+)?experiment", 1.2),
]


# ============================================================================
# Toxicity Patterns
# ============================================================================

# Toxicity patterns - More sophisticated detection with context
TOXICITY_PATTERNS = {
    "severe": [
        # Violent threats with targeting
        (r"\b(?:kill|murder|assassinate|eliminate)\s+(?:you|them|him|her|all)\b", 5.0),
        (r"\b(?:die|death)\s+to\s+(?:you|all|them)\b", 5.0),
        # Extreme harm
        (r"\b(?:rape|torture|mutilate|dismember)\b", 5.0),
        (r"\b(?:should|must|gonna|going\s+to)\s+(?:die|suffer|burn)\b", 4.0),
    ],
    "high": [
        # Strong insults with intensity
        (r"\b(?:fucking|goddamn)\s+(?:idiot|stupid|moron|retard)\b", 3.5),
        (r"\byou\s+(?:piece\s+of\s+shit|worthless\s+piece)\b", 3.5),
        # Hate speech indicators
        (r"\bi\s+(?:hate|despise|detest)\s+(?:you|them|everyone)\b", 3.0),
        (r"\b(?:screw|fuck)\s+(?:you|off|this)\b", 3.0),
    ],
    "medium": [
        # Common insults
        (r"\b(?:idiot|stupid|dumb|moron|fool|loser)\b", 2.0),
        (r"\b(?:pathetic|useless|worthless|incompetent)\b", 2.0),
        (r"\b(?:shut\s+up|shut\s+your)\b", 2.0),
        # Dismissive language
        (r"\byou\s+(?:suck|fail|are\s+trash)\b", 2.0),
    ],
    "low": [
        # Mild insults or criticism
        (r"\b(?:annoying|irritating|boring|lame)\b", 1.0),
        (r"\b(?:weird|strange|odd)(?:\s+person)?\b", 0.8),
        (r"\b(?:bad|terrible|awful|horrible)\b", 0.5),
    ],
}


# ============================================================================
# Profanity Patterns
# ============================================================================

# Profanity with severity levels and context
PROFANITY_PATTERNS = {
    "severe": [
        (r"\b(?:fuck|fucking|fucked|fucker|motherfucker)\b", 3.0),
        (r"\b(?:shit|shitty|bullshit)\b", 2.5),
        (r"\b(?:bitch|bastard|asshole)\b", 2.5),
    ],
    "moderate": [
        (r"\b(?:damn|damned|goddamn)\b", 1.5),
        (r"\b(?:hell|piss|crap)\b", 1.2),
        (r"\b(?:ass|arse)\b", 1.0),
    ],
    "mild": [
        (r"\b(?:suck|sucks)\b", 0.5),
    ],
}


# ============================================================================
# Sexual Content Patterns
# ============================================================================

# Sexual content patterns with context
SEXUAL_PATTERNS = [
    # Explicit content
    (r"\b(?:porn|pornography|xxx|x+\s*rated)\b", 3.5),
    (r"\b(?:nude|naked|undressed)\s+(?:photos?|pics?|images?|videos?|content)\b", 3.0),
    (r"\b(?:explicit|graphic|hardcore)\s+(?:photos?|content|material|videos?)\b", 3.0),
    (r"\b(?:sex|sexual)\s+(?:content|act|activity|intercourse|scene)\b", 2.5),
    (r"\bnsfw\b", 2.5),
    # Sexual services/solicitation
    (r"\b(?:escort|prostitut\w+|sex\s+work|strip|stripper)\b", 2.5),
    (r"\b(?:one\s+night\s+stand|hookup|booty\s+call)\b", 2.0),
    (r"\bsext(?:ing)?\b", 2.5),
    # Mild sexual
    (r"\b(?:adult|mature)\s+(?:content|entertainment|website)\b", 1.5),
    (r"\b(?:innuendo|suggestive|provocative)\b", 1.0),
]


# ============================================================================
# Violence Patterns
# ============================================================================

# Violence patterns with context
VIOLENCE_PATTERNS = [
    # Direct threats
    (
        r"\b(?:kill|murder|assassinate|execute)\s+(?:you|someone|them|him|her|everyone)\b",
        4.0,
    ),
    (r"\b(?:destroy|annihilate|obliterate)\s+(?:you|everyone)\b", 3.5),
    (r"\b(?:will|going\s+to|gonna)\s+(?:kill|hurt|harm|destroy|murder)\b", 3.5),
    # Weapons
    (r"\b(?:gun|knife|weapon|bomb|explosive|grenade)\s+(?:to|at|on|with)\b", 3.0),
    (r"\b(?:shoot|stab|strangle|poison|burn)\s+(?:someone|you|them|him|her)\b", 3.5),
    # Physical violence
    (
        r"\b(?:beat|attack|assault|brutalize)\s+(?:the\s+)?(?:shit|crap|hell)\s+out\s+of\b",
        3.0,
    ),
    (
        r"\b(?:punch|kick|hit|strike|slap|choke)\s+(?:someone|you|them|him|her|your)\b",
        2.5,
    ),
    # General violence
    (
        r"\b(?:violent|violence|brutal|savage|bloody)\s+(?:act|attack|scene|murder)\b",
        2.0,
    ),
    (r"\b(?:torture|maim|dismember)\b", 2.5),
    (r"\b(?:war|combat|fight|battle)\s+(?:scene|sequence)\b", 1.0),
]


# ============================================================================
# Hate Speech Patterns
# ============================================================================

# Hate speech patterns with targeting
HATE_SPEECH_PATTERNS = [
    # Direct hate expressions
    (
        r"\b(?:hate|despise|detest)\s+(?:all|those|these)\s+\w+\s+(?:people|folks|guys)\b",
        4.0,
    ),
    (
        r"\b(?:all|those)\s+(?:those|these)?\s*\w+\s+(?:are|should\s+be)\s+(?:inferior|worthless|garbage|scum)\b",
        4.0,
    ),
    # Hate terminology
    (r"\b(?:racist|sexist|bigot|nazi|fascist|supremacist)\b", 3.5),
    (r"\b(?:discriminat\w+|prejudice|bigotry)\s+against\b", 3.0),
    # Superiority claims
    (
        r"\b(?:inferior|superior)\s+(?:race|gender|ethnicity|religion|group|people)\b",
        3.5,
    ),
    (r"\b(?:pure|true|real)\s+(?:race|blood|heritage)\b", 3.0),
    # Dehumanization
    (r"\b(?:subhuman|animals?|vermin|parasites?)\b.*(?:people|group|race)", 3.5),
    (r"\b(?:deserve|deserves)\s+to\s+(?:die|suffer|be\s+eliminated)\b", 3.5),
]


# ============================================================================
# Self-Harm Patterns
# ============================================================================

# Self-harm patterns
SELF_HARM_PATTERNS = [
    # Suicidal ideation
    (r"\b(?:want|going|gonna)\s+to\s+(?:kill|end)\s+(?:myself|my\s+life)\b", 4.5),
    (r"\b(?:suicide|suicidal)\s+(?:thoughts?|ideation|plans?|attempt)\b", 4.0),
    (r"\b(?:end|take|ending)\s+(?:my|it\s+all|my\s+own)\s+(?:life)?\b", 4.0),
    (r"\bcan\'?t\s+(?:go|live)\s+on\b", 3.5),
    # Self-harm
    (r"\b(?:cut|harm|hurt|mutilate)\s+(?:myself|my\s+(?:self|body))\b", 3.5),
    (r"\b(?:cutting|self[-\s]harm|self[-\s]injury)\b", 3.5),
    # Hopelessness
    (r"\b(?:no|nothing)\s+(?:point|reason)\s+(?:in|to)\s+(?:living|life)\b", 3.0),
    (r"\b(?:better\s+off|world\s+would\s+be\s+better)\s+(?:dead|without\s+me)\b", 3.5),
]
