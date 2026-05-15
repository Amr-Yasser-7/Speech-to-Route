"""
Gazetteer-based Location Resolver for Egyptian Arabic Route Extraction.

Provides multi-strategy location matching:
1. Exact match (canonical name + aliases)
2. Normalized match (strip articles, normalize hamza/ta-marbuta)
3. Fuzzy match (Levenshtein distance)
4. Phonetic match (Arabic-adapted character grouping)

Loads from locations.json and builds optimized lookup indices at startup.
"""

import json
import os
import re
from typing import Optional, Tuple, List, Dict

# ============================================================
# Arabic Phonetic Groups (for phonetic matching)
# ============================================================
# Characters that sound similar in Egyptian Arabic
PHONETIC_GROUPS = {
    'ا': 'A', 'أ': 'A', 'إ': 'A', 'آ': 'A', 'ع': 'A',
    'ة': 'H', 'ه': 'H',
    'ي': 'Y', 'ى': 'Y',
    'و': 'W', 'ؤ': 'W',
    'ئ': 'Y',
    'ء': 'A',
    'ض': 'D', 'ظ': 'D',
    'ذ': 'Z', 'ز': 'Z',
    'ث': 'S', 'س': 'S', 'ص': 'S',
    'ق': 'Q', 'ك': 'Q', 'غ': 'Q',
    'ت': 'T', 'ط': 'T',
    'د': 'D',
    'ر': 'R',
    'ش': 'SH',
    'ج': 'G',
    'ح': 'H',
    'خ': 'KH',
    'ب': 'B',
    'ن': 'N',
    'م': 'M',
    'ل': 'L',
    'ف': 'F',
}


def arabic_phonetic_key(text: str) -> str:
    """Generate a phonetic key for Arabic text (similar to Soundex)."""
    if not text:
        return ""

    # Remove diacritics, spaces, and articles
    clean = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    clean = re.sub(r'^ال', '', clean)
    clean = clean.replace(' ', '')

    key = []
    prev = None
    for ch in clean:
        code = PHONETIC_GROUPS.get(ch, ch)
        if code != prev:  # Skip consecutive duplicates
            key.append(code)
            prev = code

    return ''.join(key)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def fuzzy_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity score (0.0 to 1.0) between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def normalize_for_lookup(text: str) -> str:
    """Normalize text for gazetteer lookup."""
    if not text:
        return ""

    t = text.strip()
    # Remove diacritics
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)
    # Normalize Alif
    t = t.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # Normalize Ya
    t = t.replace('ى', 'ي')
    # Normalize Ta Marbuta
    t = t.replace('ة', 'ه')
    # Strip definite article
    t = re.sub(r'^ال', '', t)
    # Lowercase (for English aliases)
    t = t.lower()
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t


class Gazetteer:
    """
    Location gazetteer with multi-strategy matching.
    
    Loads locations from JSON and builds lookup indices for fast resolution
    of noisy/ASR-corrupted location mentions to canonical entries.
    """

    def __init__(self, locations_path: str):
        """Initialize gazetteer from locations JSON file."""
        with open(locations_path, 'r', encoding='utf-8') as f:
            self.locations = json.load(f)

        # Filter out route entries
        self.locations = [loc for loc in self.locations if loc.get('type') != 'route']

        # Build lookup indices
        self._exact_index: Dict[str, dict] = {}       # exact name → location
        self._normalized_index: Dict[str, dict] = {}   # normalized name → location
        self._phonetic_index: Dict[str, List[dict]] = {}  # phonetic key → [locations]
        self._all_names: List[Tuple[str, dict]] = []   # (name, location) for fuzzy search

        self._build_indices()
        print(f"Gazetteer loaded: {len(self.locations)} locations, {len(self._exact_index)} lookup entries")

    def _build_indices(self):
        """Build all lookup indices from location data."""
        for loc in self.locations:
            # Collect all names for this location
            all_names = [loc['name']]
            if loc.get('name_en'):
                all_names.append(loc['name_en'])
            if loc.get('aliases'):
                all_names.extend(loc['aliases'])

            for name in all_names:
                if not name:
                    continue

                # Exact index
                self._exact_index[name] = loc

                # Normalized index
                norm = normalize_for_lookup(name)
                if norm:
                    self._normalized_index[norm] = loc

                # Phonetic index
                # Only for Arabic names
                if any('\u0600' <= c <= '\u06FF' for c in name):
                    pkey = arabic_phonetic_key(name)
                    if pkey:
                        if pkey not in self._phonetic_index:
                            self._phonetic_index[pkey] = []
                        if loc not in self._phonetic_index[pkey]:
                            self._phonetic_index[pkey].append(loc)

                # All names list for fuzzy search
                self._all_names.append((name, loc))

    def resolve(self, text: str, min_fuzzy_score: float = 0.65) -> Optional[dict]:
        """
        Resolve a text mention to a canonical location.
        
        Returns dict with:
        - location: the matched location entry
        - canonical_name: the canonical name
        - match_type: 'exact', 'normalized', 'fuzzy', 'phonetic'
        - confidence: 0.0-1.0
        
        Returns None if no match found above threshold.
        """
        if not text or len(text.strip()) < 2:
            return None

        text = text.strip()

        # Strategy 1: Exact match
        if text in self._exact_index:
            loc = self._exact_index[text]
            return {
                'location': loc,
                'canonical_name': loc['name'],
                'match_type': 'exact',
                'confidence': 1.0,
            }

        # Strategy 2: Normalized match
        norm = normalize_for_lookup(text)
        if norm and norm in self._normalized_index:
            loc = self._normalized_index[norm]
            return {
                'location': loc,
                'canonical_name': loc['name'],
                'match_type': 'normalized',
                'confidence': 0.95,
            }

        # Strategy 3: Fuzzy match (Levenshtein)
        best_score = 0.0
        best_loc = None
        best_name = None

        for name, loc in self._all_names:
            # Quick length filter — skip if lengths are too different
            if abs(len(text) - len(name)) > max(len(text), len(name)) * 0.5:
                continue

            score = fuzzy_similarity(normalize_for_lookup(text), normalize_for_lookup(name))
            if score > best_score:
                best_score = score
                best_loc = loc
                best_name = name

        if best_score >= min_fuzzy_score and best_loc:
            return {
                'location': best_loc,
                'canonical_name': best_loc['name'],
                'match_type': 'fuzzy',
                'confidence': best_score,
            }

        # Strategy 4: Phonetic match
        pkey = arabic_phonetic_key(text)
        if pkey and pkey in self._phonetic_index:
            candidates = self._phonetic_index[pkey]
            if len(candidates) == 1:
                loc = candidates[0]
                return {
                    'location': loc,
                    'canonical_name': loc['name'],
                    'match_type': 'phonetic',
                    'confidence': 0.7,
                }
            elif len(candidates) > 1:
                # Multiple phonetic matches — pick best fuzzy score
                best_phon_score = 0.0
                best_phon_loc = None
                for loc in candidates:
                    score = fuzzy_similarity(normalize_for_lookup(text), normalize_for_lookup(loc['name']))
                    if score > best_phon_score:
                        best_phon_score = score
                        best_phon_loc = loc
                if best_phon_loc and best_phon_score > 0.5:
                    return {
                        'location': best_phon_loc,
                        'canonical_name': best_phon_loc['name'],
                        'match_type': 'phonetic',
                        'confidence': best_phon_score * 0.85,
                    }

        # No match found
        return None

    def get_all_location_names(self) -> List[str]:
        """Get all canonical location names (useful for NER training)."""
        return [loc['name'] for loc in self.locations]
