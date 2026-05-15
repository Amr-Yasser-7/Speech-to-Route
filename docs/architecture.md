# Architecture Details

The Arabic Speech-to-Route project utilizes a multi-stage NLP pipeline designed specifically for the phonetic and morphological characteristics of Egyptian Arabic.

## 1. Speech-to-Text (ASR)
- **Model**: `itshamdi404/Egy_Arabic_whisper-small`
- **Purpose**: Converts 16kHz audio arrays into Arabic text transcripts.

## 2. Text Normalization
- **Module**: `src/normalizer.py`
- **Purpose**: ASR output is often inconsistent. The normalizer:
  - Standardizes Alif (أ, إ, آ -> ا)
  - Standardizes Yaa/Alif Maqsura (ي/ى)
  - Standardizes Ta Marbuta/Haa (ة/ه)
  - Strips diacritics (Tashkeel)
  - Removes common politeness fillers ("لو سمحت", "يا كابتن") before NER processing to reduce sequence noise.

## 3. Route Extraction (Hybrid Pipeline)
- **Module**: `src/route_extractor.py`

### 3A. NER (Primary)
- **Model Architecture**: Token Classification (e.g., CAMeLBERT)
- **Output**: BIO tagging (`B-ORIGIN`, `I-ORIGIN`, `B-DEST`, `I-DEST`)
- **Advantage**: Understands spatial prepositions (من, لـ, على) contextually.

### 3B. Syntactic Fallback (Secondary)
- Used only if the NER model fails to load or returns low confidence (<30%).
- Employs regex patterns specific to Egyptian syntax (e.g., `من [Origin] لـ [Destination]`).

## 4. Entity Resolution (Gazetteer)
- **Module**: `src/gazetteer.py`
- **Purpose**: Translates raw text strings into canonical location names.
- **Process**:
  1. Exact Match (e.g., "المعادي" -> "المعادي")
  2. Normalized Match (e.g., "معادى" -> "المعادي")
  3. Fuzzy String Match (Levenshtein distance)
  4. Phonetic Match (Arabic Soundex variant for handling severe ASR misspellings)

## 5. Inference Architecture
The system uses **Direct Loading** (Option B) for modern API standards.
- Models are initialized globally upon API startup (`main.py`).
- No pickle serialization is used in the production server, ensuring security and update flexibility.
- For legacy or heavily constrained environments, a `FORCE_PICKLE_FALLBACK` environment variable allows loading from a pre-serialized `Speech.pkl` artifact.
