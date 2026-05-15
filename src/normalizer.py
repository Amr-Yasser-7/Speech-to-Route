"""
Arabic Text Normalizer for Route Extraction Pipeline.

Handles:
- Hamza/Alif normalization
- Ta Marbuta/Ha equivalence
- Egyptian filler word removal
- Arabic prefix handling with landmark protection
- Whitespace normalization
"""

import re

# ============================================================
# Protected words that start with common prefix letters
# ============================================================
PROTECTED_PREFIXES = [
    "لبنان", "لوران", "لوتس", "لوكاندة",
    "بورسعيد", "بورفؤاد", "بني سويف", "بولاق", "بنها",
]

# ============================================================
# Egyptian noise/filler words to strip
# ============================================================
NOISE_PHRASES = [
    "لو سمحت", "من فضلك", "يا ريس", "يا باشا", "يا حج",
    "يا كابتن", "يا معلم", "يا عم", "الله يخليك",
    "عايز اروح", "عايزة اروح", "محتاج اوصل",
    "بسرعة", "دلوقتي", "ممكن", "خدني", "وصلني",
    "عايزة", "عايز", "يعني", "كده", "بقى", "بالراحة",
    "طب", "يلا", "بقولك", "اسمع", "شوف",
]


def normalize_arabic(text: str) -> str:
    """
    Full Arabic text normalization pipeline.
    
    Normalizes hamza variants, ta marbuta, diacritics, and whitespace.
    Does NOT remove noise words (that's a separate step).
    """
    if not text:
        return ""

    t = text.strip()

    # Remove diacritics (tashkeel)
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)

    # Normalize Alif variants
    t = t.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # Normalize Alif Maqsura to Ya
    t = t.replace('ى', 'ي')

    # Normalize Ta Marbuta to Ha (for matching purposes)
    # Note: We keep the original in the output, only normalize for matching
    # t = t.replace('ة', 'ه')  # Only used in matching, not preprocessing

    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t


def remove_noise(text: str) -> str:
    """Remove Egyptian filler/noise words from text."""
    cleaned = text
    for noise in NOISE_PHRASES:
        cleaned = cleaned.replace(noise, '')

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def strip_prefixes(text: str) -> str:
    """
    Strip Arabic definite article and preposition prefixes.
    Protects known landmark names that naturally start with these letters.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Check if word is protected
    for protected in PROTECTED_PREFIXES:
        if cleaned.startswith(protected):
            return cleaned

    # Strip leading prepositions/articles: لل, لـ, ل, ب, ال
    cleaned = re.sub(r'^(?:لل|لـ|ل|بـ|ب)(?![\s])', '', cleaned)

    return cleaned.strip()


def normalize_for_matching(text: str) -> str:
    """
    Aggressive normalization specifically for fuzzy matching.
    Strips articles, normalizes ta marbuta, etc.
    """
    if not text:
        return ""

    t = normalize_arabic(text)

    # Strip definite article ال
    t = re.sub(r'^ال', '', t)

    # Normalize Ta Marbuta → Ha
    t = t.replace('ة', 'ه')

    # Remove common prefixes (شارع, ميدان, محطة)
    t = re.sub(r'^(?:شارع|ميدان|محطة|موقف)\s*', '', t)

    return t.strip()


def full_preprocess(text: str) -> str:
    """
    Full preprocessing pipeline for incoming transcripts.
    
    1. Normalize Arabic characters
    2. Remove noise/filler words
    3. Normalize whitespace
    """
    t = normalize_arabic(text)
    t = remove_noise(t)
    return t
