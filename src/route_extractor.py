"""
Route Extractor v2: NER + Gazetteer Hybrid Pipeline

Replaces the QA-based extraction with a proper NER architecture:

Pipeline:
  Transcript → Normalizer → NER Model → Gazetteer Resolver → Validated Output

Components:
  1. Text Normalizer: Hamza, diacritics, filler word removal
  2. NER Model: EgyBERT/MARBERTv2 fine-tuned for B-ORIGIN/I-ORIGIN/B-DEST/I-DEST
  3. Gazetteer: Multi-strategy location resolution (exact/normalized/fuzzy/phonetic)
  4. Syntactic Fallback: Regex patterns as last resort
  5. Confidence Scorer: Combined NER + gazetteer confidence
"""

import os
import re
import sys

# Ensure src directory is in path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from normalizer import normalize_arabic, remove_noise, full_preprocess, normalize_for_matching
from gazetteer import Gazetteer

# ============================================================
# Configuration
# ============================================================
NER_CONFIDENCE_THRESHOLD = 0.3  # Minimum NER confidence to trust extraction
GAZ_MIN_FUZZY_SCORE = 0.65      # Minimum fuzzy match score for gazetteer
COMBINED_CONFIDENCE_WEIGHTS = (0.6, 0.4)  # (NER weight, Gazetteer weight)


# ============================================================
# Syntactic Fallback (from v1, refined)
# ============================================================

def syntactic_extraction(text: str) -> dict:
    """
    Fallback: Egyptian Arabic syntactic pattern matching.
    Only used when NER confidence is below threshold.
    """
    text_norm = normalize_arabic(text)

    # Pattern 0: انا في [Origin] عايز/رايح [Destination]
    match_loc = re.search(
        r'(?:انا\s+)?(?:في|واقف\s+في)\s+(.+?)\s+(?:(?:و\s*)?عايز\s+(?:اروح)?|(?:و\s*)?رايح|(?:و\s*)?محتاج)\s+(.+)',
        text_norm
    )
    if match_loc:
        return {'origin': match_loc.group(1).strip(), 'destination': match_loc.group(2).strip()}

    # Pattern 1: من [Origin] MARKER [Destination]
    match_word = re.search(
        r'من\s+(.+?)\s+(?:و\s*)?(?:الي|الى|علي|على|رايح|رايحة|رايحه|اروح|اوصل|روح|هروح)\s+(.+)',
        text_norm
    )
    if match_word:
        return {'origin': match_word.group(1).strip(), 'destination': match_word.group(2).strip()}

    # Pattern 1B: من [Origin] ل/لل[Destination]
    match_pref = re.search(r'من\s+(.+?)\s+(?:و\s*)?(?:لل|لـ|ل)(\S+.*)', text_norm)
    if match_pref:
        return {'origin': match_pref.group(1).strip(), 'destination': match_pref.group(2).strip()}

    # Pattern 2: MARKER [Destination] من [Origin]
    match_rev = re.search(
        r'(?:و\s*)?(?:رايح|رايحة|اروح|اوصل|عايز\s+اروح|خدني|وصلني|هروح|نازل)\s+(.+?)\s+من\s+(.+)',
        text_norm
    )
    if match_rev:
        return {'origin': match_rev.group(2).strip(), 'destination': match_rev.group(1).strip()}

    # Pattern 3: من [Origin] [multi-word Destination]
    match_from = re.search(r'من\s+(.+?)\s+(.+)', text_norm)
    if match_from:
        return {'origin': match_from.group(1).strip(), 'destination': match_from.group(2).strip()}

    return None


# ============================================================
# Route Extractor v2
# ============================================================

class RouteExtractor:
    """
    Production route extraction pipeline.
    
    Combines NER model + Gazetteer + Syntactic fallback
    for robust origin/destination extraction from Egyptian Arabic.
    """

    def __init__(self, ner_model_path: str = None, locations_path: str = None,
                 use_onnx: bool = False):
        """
        Initialize the route extractor.
        
        Args:
            ner_model_path: Path to fine-tuned NER model directory
            locations_path: Path to locations.json for gazetteer
            use_onnx: Use ONNX Runtime for NER inference
        """
        self.ner = None
        self.gazetteer = None

        # Load NER model if available
        if ner_model_path and os.path.exists(ner_model_path):
            try:
                from ner_model import NERModel
                self.ner = NERModel(ner_model_path, use_onnx=use_onnx)
            except Exception as e:
                print(f"⚠ Failed to load NER model: {e}")
                print("  Falling back to syntactic-only extraction")

        # Load Gazetteer
        if locations_path and os.path.exists(locations_path):
            try:
                self.gazetteer = Gazetteer(locations_path)
            except Exception as e:
                print(f"⚠ Failed to load Gazetteer: {e}")
        else:
            # Try default paths
            candidates = [
                os.path.join(os.path.dirname(_src_dir), 'data', 'locations.json'),
                os.path.join(_src_dir, 'locations.json'),
            ]
            for path in candidates:
                if os.path.exists(path):
                    self.gazetteer = Gazetteer(path)
                    break

        if not self.ner:
            print("⚠ NER model not loaded — using syntactic extraction + gazetteer only")

    def extract(self, text: str) -> dict:
        """
        Extract origin and destination from Arabic routing text.
        
        Returns:
            {
                'origin': str or None,
                'origin_score': float,
                'origin_canonical': str or None,
                'destination': str or None,
                'destination_score': float,
                'destination_canonical': str or None,
                'method': str,  # 'ner', 'syntactic', 'gazetteer_only'
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        # Step 1: Preprocess
        # Normalize characters (hamza, ya, etc.)
        normalized = normalize_arabic(text)
        
        # Light noise removal: strip politeness fillers but KEEP routing markers
        # These are words that interfere with extraction but are NOT routing signals
        politeness_noise = [
            "لو سمحت", "من فضلك", "يا ريس", "يا باشا", "يا حج",
            "يا كابتن", "يا معلم", "يا عم", "الله يخليك",
            "بسرعة", "دلوقتي", "بالراحة", "يعني", "كده", "بقي",
            "طب", "يلا", "بقولك", "اسمع", "شوف",
        ]
        light_cleaned = normalized
        for noise in politeness_noise:
            light_cleaned = light_cleaned.replace(noise, '')
        light_cleaned = re.sub(r'\s+', ' ', light_cleaned).strip()
        
        # Full noise removal for NER (also removes routing words)
        preprocessed = remove_noise(normalized)

        # Step 2: Try NER extraction
        ner_result = None
        if self.ner:
            try:
                ner_result = self.ner.predict(preprocessed)
            except Exception as e:
                print(f"⚠ NER prediction failed: {e}")

        # Step 3: Check NER confidence
        ner_origin = None
        ner_dest = None
        ner_origin_score = 0
        ner_dest_score = 0

        if ner_result:
            ner_origin = ner_result.get('origin')
            ner_dest = ner_result.get('destination')
            ner_origin_score = ner_result.get('origin_score', 0)
            ner_dest_score = ner_result.get('destination_score', 0)

        # Step 4: If NER confidence is low, try syntactic fallback
        # NOTE: Syntactic runs on NORMALIZED text (not noise-stripped) to see routing markers
        method = 'ner'
        if not ner_origin or not ner_dest or min(ner_origin_score, ner_dest_score) < NER_CONFIDENCE_THRESHOLD:
            syn_result = syntactic_extraction(light_cleaned)
            if syn_result:
                method = 'syntactic'
                if not ner_origin or ner_origin_score < NER_CONFIDENCE_THRESHOLD:
                    ner_origin = syn_result.get('origin')
                    ner_origin_score = 0.7  # Syntactic confidence
                if not ner_dest or ner_dest_score < NER_CONFIDENCE_THRESHOLD:
                    ner_dest = syn_result.get('destination')
                    ner_dest_score = 0.7

        # Step 5: Resolve through gazetteer
        origin_canonical = None
        dest_canonical = None
        origin_gaz_score = 0
        dest_gaz_score = 0

        if self.gazetteer:
            if ner_origin:
                gaz_match = self.gazetteer.resolve(ner_origin, min_fuzzy_score=GAZ_MIN_FUZZY_SCORE)
                if gaz_match:
                    origin_canonical = gaz_match['canonical_name']
                    origin_gaz_score = gaz_match['confidence']

            if ner_dest:
                gaz_match = self.gazetteer.resolve(ner_dest, min_fuzzy_score=GAZ_MIN_FUZZY_SCORE)
                if gaz_match:
                    dest_canonical = gaz_match['canonical_name']
                    dest_gaz_score = gaz_match['confidence']

        # Step 6: Compute combined confidence
        w_ner, w_gaz = COMBINED_CONFIDENCE_WEIGHTS
        origin_final_score = (w_ner * ner_origin_score + w_gaz * origin_gaz_score) if ner_origin else 0
        dest_final_score = (w_ner * ner_dest_score + w_gaz * dest_gaz_score) if ner_dest else 0

        # Use canonical name if available, otherwise raw NER output
        origin_final = origin_canonical or ner_origin
        dest_final = dest_canonical or ner_dest

        return {
            'origin': origin_final,
            'origin_score': round(origin_final_score, 4),
            'origin_canonical': origin_canonical,
            'destination': dest_final,
            'destination_score': round(dest_final_score, 4),
            'destination_canonical': dest_canonical,
            'method': method,
        }

    def _empty_result(self) -> dict:
        return {
            'origin': None, 'origin_score': 0, 'origin_canonical': None,
            'destination': None, 'destination_score': 0, 'destination_canonical': None,
            'method': 'none',
        }


# ============================================================
# Backward-compatible API (drop-in replacement for v1)
# ============================================================

# Module-level instance (loaded on import)
_extractor = None


def _get_extractor():
    """Lazy-load the extractor singleton."""
    global _extractor
    if _extractor is None:
        # Determine paths
        project_root = os.path.dirname(_src_dir)
        locations_path = os.path.join(project_root, 'data', 'locations.json')

        # Try to find a trained NER model
        ner_candidates = [
            os.path.join(project_root, 'models', 'bert-base-arabic-camelbert-mix_best'),
            os.path.join(project_root, 'models', 'ner_best'),
            os.path.join(project_root, 'models', 'egybert_best'),
            os.path.join(project_root, 'models', 'marbert_best'),
        ]

        ner_model_path = None
        for candidate in ner_candidates:
            if os.path.exists(candidate):
                ner_model_path = candidate
                break

        # Check for ONNX version
        use_onnx = False
        onnx_candidates = [
            os.path.join(project_root, 'models', 'egybert_onnx_int8'),
            os.path.join(project_root, 'models', 'ner_onnx_int8'),
            os.path.join(project_root, 'models', 'ner_onnx'),
        ]
        for candidate in onnx_candidates:
            if os.path.exists(candidate):
                ner_model_path = candidate
                use_onnx = True
                break

        _extractor = RouteExtractor(
            ner_model_path=ner_model_path,
            locations_path=locations_path,
            use_onnx=use_onnx,
        )

    return _extractor


def advanced_extract_route(text: str) -> dict:
    """
    Backward-compatible extraction function.
    Drop-in replacement for the v1 QA-based extractor.
    """
    extractor = _get_extractor()
    result = extractor.extract(text)

    # Return in v1-compatible format
    return {
        'origin': result['origin'],
        'origin_score': result['origin_score'],
        'destination': result['destination'],
        'destination_score': result['destination_score'],
    }


# ============================================================
# Quick Test
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Route Extractor v2 — Quick Test")
    print("="*60)

    # Test with gazetteer-only mode (no NER model yet)
    test_cases = [
        "من مدينة نصر الى المعادي",
        "عايز اروح التجمع الخامس من الهرم",
        "رايح الاسكندرية من القاهرة",
        "انا في شبرا عايز اروح الدقي",
        "خدني من المهندسين على الزمالك لو سمحت",
        "من فضلك وصلني من رمسيس للمعادي يا باشا",
    ]

    extractor = _get_extractor()

    for text in test_cases:
        result = extractor.extract(text)
        print(f"\n📝 Input:  {text}")
        print(f"   Origin: {result['origin']} (score: {result['origin_score']:.2f}, canonical: {result['origin_canonical']})")
        print(f"   Dest:   {result['destination']} (score: {result['destination_score']:.2f}, canonical: {result['destination_canonical']})")
        print(f"   Method: {result['method']}")
