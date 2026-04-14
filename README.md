# Speech-to-Route
### Egyptian Arabic Origin/Destination Extractor
[![Author](https://img.shields.io/badge/Author-Amr%20Yasser-blue)](https://github.com/Amr-Yasser-7)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)
A production-ready NLP engine that extracts routing data — **Origin & Destination** — from colloquial Egyptian Arabic (Ammiya) speech. Built on a **Pure AI (Transformer-based)** architecture engineered specifically for the nuances of colloquial dialects.
```python
from src.main import extract_route_from_text
extract_route_from_text("عايز أروح من المعادي للتحرير")
# → {"origin": "المعادي", "destination": "التحرير"}
extract_route_from_text("ودّيني من شبرا للمطار")
# → {"origin": "شبرا", "destination": "المطار"}
```
---
## Why This Is Hard
Standard NLP pipelines fail on Egyptian Arabic for three compounding reasons:
- **Orthographic instability** — the same word can be spelled 3–4 ways (`المعادي` / `المعدي` / `المعاده`)
- **No fixed word order** — origin and destination swap positions freely across dialects
- **Conversational noise** — filler phrases like *"يا كابتن"* or *"لو سمحت"* pollute span extraction
This project solves all three.
---
## Architecture
```
Audio Input
    │
    ▼
┌─────────────────────────────────┐
│  Speech-to-Text (STT)           │  Fine-tuned Whisper-Small
│  Dialect-accurate transcription │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Neural Extractor           │  AraElectra QA model
│  Context: "رايح فين؟"      │  Identifies location spans
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Neural Intent Extractor    │ 
└────────────┬────────────────┘
             │
             ▼
       {"origin": X, "destination": Y}
```
The **Pure AI design** is the key engineering decision: the transformer model handles the complexity of colloquial inputs and role assignment end-to-end, providing a scalable and unified extraction layer.
---
## Core Technical Achievements
- **Transformer-based Intent Extraction** — End-to-end extraction using AraElectra (Arabic Electra), fine-tuned for high-accuracy span identification in colloquial speech.
- **Linguistic Engineering** — Custom normalization for Egyptian Arabic: Hamza/Ta-Marbuta orthographic variants, prepositional span resolution (`من`/`لـ`/`إلى`), and conversational filler removal.
- **Fully Offline** — Zero external API dependencies. All models are serialized locally into a single `Speech.pkl` brain for sub-second startup and complete data privacy.
- **Production Infrastructure** — Dockerized FastAPI service with optimized memory management.
---
## Getting Started
### 1. Clone & Install
```bash
git clone https://github.com/Amr-Yasser-7/Speech-to-Route.git
cd Speech-to-Route
pip install -r requirements.txt
```
### 2. Cold Start — Download Models
The repository is kept lightweight (~50KB). The 1.5GB models are downloaded once via the setup script:
```bash
python Scripts/save_models.py    # Downloads & serializes AI models locally
python src/main.py               # Generates Speech.pkl inference brain
```
> **Note:** This step only needs to run once. All subsequent startups use the local `models/Speech.pkl` file.
### 3. Run the API
```bash
# Local development
python speech_recognition_api.py
# Via Docker
docker build -t speech-to-route .
docker run -p 8000:8000 speech-to-route
```
API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
---
## Project Structure
```text
Speech-to-Route/
├── src/                      # Core logic and NLP functions
│   ├── main.py
│   └── route_extractor.py    # AI Extraction logic
├── models/                   # Heavy model files (ignored in Git)
│   └── Speech.pkl
├── speech_recognition_api.py   # Main Entry Point (API)
├── Docs/
│   └── walkthrough.md          # Deep-dive architecture notes
├── Scripts/
│   └── save_models.py          # One-time model download
├── requirements.txt
├── Dockerfile
└── README.md
```
---
## Author
Developed by **Amr Yasser**
- **GitHub**: [@Amr-Yasser-7](https://github.com/Amr-Yasser-7)
- **Focus**: Machine Learning Engineering · NLP · Arabic Dialect Processing
---
*Built for real-world deployment in Arabic-speaking transport and ride-hailing contexts.*
