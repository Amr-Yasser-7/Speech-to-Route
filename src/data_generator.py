"""
Synthetic NER Training Data Generator for Egyptian Arabic Route Extraction.

Generates BIO-annotated training samples from locations.json using:
- Template-based utterance generation with Egyptian dialect patterns
- ASR noise injection (character substitution, deletion, insertion)
- Data augmentation (synonym replacement, noise word injection, word dropout)
- Balanced origin/destination coverage
"""

import json
import random
import re
import os
import csv

# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 42
NUM_SAMPLES = 2000  # Total samples to generate
NOISE_RATIO = 0.3   # 30% of samples get ASR noise
FILLER_RATIO = 0.4  # 40% of samples get Egyptian filler words

random.seed(RANDOM_SEED)

# ============================================================
# Egyptian Arabic Sentence Templates
# ============================================================
# {ORIGIN} and {DEST} are placeholders for location names
# These cover the full range of Egyptian colloquial routing patterns

TEMPLATES_ORIGIN_DEST = [
    # Standard patterns: من [origin] الى/لـ [dest]
    "من {ORIGIN} الى {DEST}",
    "من {ORIGIN} إلى {DEST}",
    "من {ORIGIN} ل{DEST}",
    "من {ORIGIN} لل{DEST}",
    "من {ORIGIN} على {DEST}",
    "من {ORIGIN} رايح {DEST}",
    "من {ORIGIN} رايحة {DEST}",
    "من {ORIGIN} روح {DEST}",

    # Request patterns
    "عايز اروح من {ORIGIN} لـ {DEST}",
    "عايز اروح {DEST} من {ORIGIN}",
    "عايزة اروح من {ORIGIN} الى {DEST}",
    "محتاج اوصل من {ORIGIN} لـ {DEST}",
    "محتاج اوصل {DEST} من {ORIGIN}",
    "خدني من {ORIGIN} على {DEST}",
    "وصلني من {ORIGIN} لـ {DEST}",
    "نازل من {ORIGIN} رايح {DEST}",
    "طالع من {ORIGIN} على {DEST}",
    "ماشي من {ORIGIN} لـ {DEST}",
    "هروح من {ORIGIN} لـ {DEST}",
    "رايح من {ORIGIN} لـ {DEST}",
    "انا رايح {DEST} من {ORIGIN}",
    "انا عايز اروح {DEST} من {ORIGIN}",
    "انا في {ORIGIN} عايز اروح {DEST}",
    "انا في {ORIGIN} رايح {DEST}",
    "انا واقف في {ORIGIN} عايز {DEST}",

    # Question patterns
    "ازاي اروح من {ORIGIN} لـ {DEST}",
    "اوصل {DEST} من {ORIGIN} ازاي",
    "فيه اتوبيس من {ORIGIN} لـ {DEST}",
    "ايه المواصلات من {ORIGIN} لـ {DEST}",
    "المترو من {ORIGIN} لـ {DEST}",

    # Short/informal patterns
    "{ORIGIN} {DEST}",
    "من {ORIGIN} {DEST}",
    "{DEST} من {ORIGIN}",
    "رايح {DEST} من {ORIGIN}",
    "اروح {DEST} من {ORIGIN}",
]

# Templates with only destination (no explicit origin)
TEMPLATES_DEST_ONLY = [
    "عايز اروح {DEST}",
    "عايزة اروح {DEST}",
    "رايح {DEST}",
    "محتاج اوصل {DEST}",
    "خدني {DEST}",
    "وصلني {DEST}",
    "اروح {DEST} ازاي",
    "عايز {DEST}",
    "على {DEST}",
    "ل{DEST}",
    "هروح {DEST}",
    "نازل {DEST}",
]

# Templates with only origin (no explicit destination)
TEMPLATES_ORIGIN_ONLY = [
    "انا في {ORIGIN}",
    "انا واقف في {ORIGIN}",
    "طالع من {ORIGIN}",
    "نازل من {ORIGIN}",
    "ماشي من {ORIGIN}",
    "من {ORIGIN}",
]

# Egyptian filler/noise words that appear naturally in speech
FILLER_PREFIXES = [
    "لو سمحت ", "من فضلك ", "يا ريس ", "يا باشا ", "يا حج ",
    "يا كابتن ", "ممكن ", "بسرعة ", "يا معلم ", "يا عم ",
    "طب ", "يلا ", "بقولك ", "اسمع ", "شوف ",
]

FILLER_SUFFIXES = [
    " لو سمحت", " من فضلك", " يا ريس", " بسرعة", " دلوقتي",
    " يا معلم", " يا باشا", " بالراحة", " الله يخليك",
    " يعني", " كده", " بقى",
]

# ============================================================
# ASR Noise Simulation
# ============================================================
# Common Arabic character confusions in ASR output
CHAR_CONFUSIONS = {
    'ا': ['أ', 'إ', 'آ', 'ء'],
    'أ': ['ا', 'إ', 'آ'],
    'إ': ['ا', 'أ', 'آ'],
    'آ': ['ا', 'أ', 'إ'],
    'ة': ['ه', 'ت'],
    'ه': ['ة'],
    'ى': ['ي', 'ا'],
    'ي': ['ى'],
    'ؤ': ['و', 'ء'],
    'ئ': ['ي', 'ء'],
    'ض': ['ظ'],
    'ظ': ['ض'],
    'ذ': ['ز', 'د'],
    'ث': ['س', 'ت'],
    'ص': ['س'],
    'ق': ['ك', 'أ', 'ء'],
    'ع': ['ا'],
    'غ': ['ق'],
}


def inject_asr_noise(text: str, noise_level: float = 0.15) -> str:
    """Simulate ASR transcription errors on Arabic text."""
    chars = list(text)
    result = []
    for ch in chars:
        if ch in CHAR_CONFUSIONS and random.random() < noise_level:
            result.append(random.choice(CHAR_CONFUSIONS[ch]))
        elif ch != ' ' and random.random() < noise_level * 0.3:
            # Character deletion
            continue
        elif ch != ' ' and random.random() < noise_level * 0.2:
            # Character duplication
            result.append(ch)
            result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


# ============================================================
# BIO Tag Generator
# ============================================================

def tokenize_arabic(text: str) -> list:
    """Simple whitespace tokenizer for Arabic text."""
    return text.split()


def create_bio_tags(text: str, origin: str, destination: str) -> list:
    """
    Create BIO tags for a text given origin and destination spans.
    Returns list of (token, tag) tuples.
    """
    tokens = tokenize_arabic(text)
    tags = ['O'] * len(tokens)

    if origin:
        origin_tokens = tokenize_arabic(origin)
        _mark_span(tokens, tags, origin_tokens, 'ORIGIN')

    if destination:
        dest_tokens = tokenize_arabic(destination)
        _mark_span(tokens, tags, dest_tokens, 'DEST')

    return list(zip(tokens, tags))


def _mark_span(tokens, tags, entity_tokens, entity_type):
    """Find and mark an entity span in the token list."""
    entity_len = len(entity_tokens)
    for i in range(len(tokens) - entity_len + 1):
        # Check if this position matches the entity
        if tokens[i:i + entity_len] == entity_tokens:
            # Only mark if all positions are currently 'O' (avoid double-tagging)
            if all(tags[j] == 'O' for j in range(i, i + entity_len)):
                tags[i] = f'B-{entity_type}'
                for j in range(i + 1, i + entity_len):
                    tags[j] = f'I-{entity_type}'
                break  # Mark first occurrence only


# ============================================================
# Data Generator
# ============================================================

def load_locations(filepath: str) -> list:
    """Load and filter locations (exclude route types)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        locations = json.load(f)

    # Filter out route entries (M5, M8, etc.) — these aren't locations
    return [loc for loc in locations if loc.get('type') != 'route']


def get_location_name(location: dict) -> str:
    """Get a location name, randomly choosing from canonical name or aliases."""
    names = [location['name']]
    if location.get('aliases'):
        # Only include Arabic aliases (skip English)
        arabic_aliases = [a for a in location['aliases'] if any('\u0600' <= c <= '\u06FF' for c in a)]
        names.extend(arabic_aliases)
    return random.choice(names)


def generate_sample(locations: list, template_type: str = 'both') -> dict:
    """Generate a single training sample."""
    origin_loc = random.choice(locations)
    dest_loc = random.choice(locations)

    # Ensure origin != destination
    while dest_loc['name'] == origin_loc['name']:
        dest_loc = random.choice(locations)

    origin_name = get_location_name(origin_loc)
    dest_name = get_location_name(dest_loc)

    if template_type == 'both':
        template = random.choice(TEMPLATES_ORIGIN_DEST)
        text = template.replace('{ORIGIN}', origin_name).replace('{DEST}', dest_name)
        origin_span = origin_name
        dest_span = dest_name
    elif template_type == 'dest_only':
        template = random.choice(TEMPLATES_DEST_ONLY)
        text = template.replace('{DEST}', dest_name)
        origin_span = None
        dest_span = dest_name
    elif template_type == 'origin_only':
        template = random.choice(TEMPLATES_ORIGIN_ONLY)
        text = template.replace('{ORIGIN}', origin_name)
        origin_span = origin_name
        dest_span = None
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    # Add filler words
    if random.random() < FILLER_RATIO:
        if random.random() < 0.5:
            text = random.choice(FILLER_PREFIXES) + text
        else:
            text += random.choice(FILLER_SUFFIXES)

    # Create clean BIO tags BEFORE noise injection
    bio_tags = create_bio_tags(text, origin_span, dest_span)

    # Optionally inject ASR noise (on the text only, tags remain aligned to original)
    noisy_text = None
    if random.random() < NOISE_RATIO:
        noisy_text = inject_asr_noise(text, noise_level=0.12)
        # Re-tokenize and re-tag with noisy text
        # For noisy samples, we inject noise into the entity spans too
        noisy_origin = inject_asr_noise(origin_span, 0.12) if origin_span else None
        noisy_dest = inject_asr_noise(dest_span, 0.12) if dest_span else None
        bio_tags_noisy = create_bio_tags(noisy_text, noisy_origin, noisy_dest)

    return {
        'text': text,
        'origin': origin_span,
        'destination': dest_span,
        'bio_tags': bio_tags,
        'noisy_text': noisy_text,
        'noisy_bio_tags': bio_tags_noisy if noisy_text else None,
    }


def generate_dataset(locations_path: str, output_dir: str, num_samples: int = NUM_SAMPLES):
    """Generate the full NER training dataset."""
    locations = load_locations(locations_path)
    print(f"Loaded {len(locations)} locations (excluding routes)")

    samples = []
    # Distribution: 70% both, 20% dest_only, 10% origin_only
    for i in range(num_samples):
        r = random.random()
        if r < 0.70:
            template_type = 'both'
        elif r < 0.90:
            template_type = 'dest_only'
        else:
            template_type = 'origin_only'

        sample = generate_sample(locations, template_type)
        samples.append(sample)

        # If noisy variant exists, add it as a separate sample
        if sample['noisy_text']:
            samples.append({
                'text': sample['noisy_text'],
                'origin': sample['origin'],
                'destination': sample['destination'],
                'bio_tags': sample['noisy_bio_tags'],
                'noisy_text': None,
                'noisy_bio_tags': None,
            })

    # Shuffle
    random.shuffle(samples)

    # Split: 80% train, 10% val, 10% test
    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    os.makedirs(output_dir, exist_ok=True)

    # Save in CoNLL-style format (token\ttag per line, blank line between sentences)
    for split_name, split_data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        filepath = os.path.join(output_dir, f'{split_name}.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in split_data:
                for token, tag in sample['bio_tags']:
                    f.write(f'{token}\t{tag}\n')
                f.write('\n')  # Blank line separates sentences

        print(f"  {split_name}: {len(split_data)} samples -> {filepath}")

    # Also save as JSON for easier inspection
    json_path = os.path.join(output_dir, 'dataset_full.json')
    json_samples = []
    for s in samples:
        json_samples.append({
            'text': s['text'],
            'origin': s['origin'],
            'destination': s['destination'],
            'tokens': [t for t, _ in s['bio_tags']],
            'tags': [tag for _, tag in s['bio_tags']],
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_samples, f, ensure_ascii=False, indent=2)
    print(f"  Full dataset JSON -> {json_path}")

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Dataset Generation Complete")
    print(f"{'='*50}")
    print(f"Total samples: {len(samples)}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")

    # Count tag distribution
    tag_counts = {}
    for s in samples:
        for _, tag in s['bio_tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"\nTag Distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == '__main__':
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Try to find locations.json
    locations_candidates = [
        os.path.join(project_root, 'data', 'locations.json'),
        os.path.join(script_dir, 'locations.json'),
        r'c:\Users\tabark\.gemini\antigravity\scratch\signn\hf_space\src\bridge\locations.json',
    ]

    locations_path = None
    for candidate in locations_candidates:
        if os.path.exists(candidate):
            locations_path = candidate
            break

    if not locations_path:
        print("ERROR: Could not find locations.json")
        print("Searched:", locations_candidates)
        exit(1)

    output_dir = os.path.join(project_root, 'data', 'ner_dataset')
    print(f"Using locations: {locations_path}")
    print(f"Output directory: {output_dir}")
    print()

    generate_dataset(locations_path, output_dir)
