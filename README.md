# 🗺️ Arabic Speech-to-Route

> **A production-ready NLP pipeline for extracting transportation routing intent from Egyptian dialect speech.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## Overview

**Arabic Speech-to-Route** solves a critical challenge in regional mobility apps: allowing users to dictate their origin and destination in natural, noisy, highly colloquial Egyptian Arabic.

Unlike standard MSA (Modern Standard Arabic) processing, this engine handles the morphological complexity, phonetic variability, and ASR (Automatic Speech Recognition) noise inherent to local dialects.

### Why This Is Hard (And How We Solved It)

1. **Dialectical Variability:** Egyptians say "عايز اروح" (I want to go) instead of "أريد الذهاب". 
   *Solution:* We fine-tuned a CAMeLBERT Transformer model specifically on Egyptian routing structures.
2. **ASR Orthographic Noise:** Speech-to-text might output "مدينه نصر" instead of "مدينة نصر" (Ta Marbuta vs. Haa).
   *Solution:* A custom `normalizer.py` and phonetic Soundex gazetteer resolves fuzzy matches.
3. **Implicit Markers:** Sometimes users just say "من المعادي التحرير" (From Maadi Tahrir).
   *Solution:* Our hybrid pipeline combines NER sequence tagging with syntactic regex fallbacks.

## 🧠 Architecture: The Extraction Pipeline

The system uses a direct-loading architecture (Option B) for modern, secure, standard API deployment.

```text
Audio Input 
  │
  ▼
[ Whisper (ASR) ] ──▶ "من المعادي للتحرير"
  │
  ▼
[ Text Normalizer ] ──▶ Noise removal, Arabic character normalization
  │
  ▼
[ NER Model ] ──▶ Predicts B-ORIGIN, I-ORIGIN, B-DEST, I-DEST tokens
  │
  ▼
[ Gazetteer ] ──▶ Fuzzy matching against known locations (e.g., "maadi" -> "المعادي")
  │
  ▼
JSON Response { "origin": "المعادي", "destination": "التحرير" }
```

*For more details, see [docs/architecture.md](docs/architecture.md).*

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/arabic-speech-to-route.git
cd arabic-speech-to-route
pip install -r requirements.txt
```

### 2. Download Models
This script fetches the Whisper model from the Hugging Face Hub:
```bash
python scripts/download_models.py
```
*(Note: Place your fine-tuned NER model in the `models/` directory).*

### 3. Run the Server
```bash
uvicorn speech_recognition_api:app --reload
```

## 🔌 API Reference

### Extract from Text
**POST** `/extract`
```json
// Request
{
  "text": "عايز اروح التجمع الخامس من الهرم"
}

// Response
{
  "transcription": "عايز اروح التجمع الخامس من الهرم",
  "origin": "الهرم",
  "destination": "التجمع الخامس",
  "status": "Success"
}
```

### Extract from Audio
**POST** `/predict`
* Accepts a multipart/form-data upload with a `file` field containing a `.wav` or `.mp3` audio file.
* Returns the same JSON schema as `/extract`, with the transcribed text.

## 🐳 Docker Deployment

To build and run via Docker (optimized for Hugging Face Spaces):
```bash
docker build -t arabic-speech-to-route .
docker run -p 7860:7860 arabic-speech-to-route
```

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to add locations, improve the normalizer, or retrain the NER model.

---
*Developed by [Amr Yasser]*
