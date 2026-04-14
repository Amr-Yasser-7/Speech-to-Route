# ARA-Route Parser
### Egyptian Arabic Origin/Destination Extractor

[![Author](https://img.shields.io/badge/Author-Amr%20Yasser-blue)](https://github.com/Amr-Yasser-7)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)

A production-ready NLP engine that extracts routing data — **Origin & Destination** — from colloquial Egyptian Arabic (Ammiya) speech. Built on a **Neural-Syntactic Hybrid** architecture engineered specifically for the nuances of low-resource Arabic dialects.

```python
from app.engine import dispatch

dispatch("عايز أروح من المعادي للتحرير")
# → {"origin": "المعادي", "destination": "التحرير"}

dispatch("ودّيني من شبرا للمطار")
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
│  Syntactic Guardrails       │  Rule-based engine
│  Normalization + Role Fix   │  Strips noise, assigns O/D roles
└────────────┬────────────────┘
             │
             ▼
       {"origin": X, "destination": Y}
```

The **hybrid design** is the key engineering decision: the neural model handles fuzzy, colloquial inputs; the deterministic layer guarantees role-assignment correctness on all rule-matched patterns — giving the best of both worlds.

---

## Core Technical Achievements

- **Hybrid AI Pipeline** — Transformer-based extractive QA (AraElectra) backed by deterministic syntactic guardrails, delivering near-deterministic role assignment on structured inputs.
- **Linguistic Engineering** — Custom normalization for Egyptian Arabic: Hamza/Ta-Marbuta orthographic variants, prepositional span resolution (`من`/`لـ`/`إلى`), and conversational filler removal.
- **Fully Offline** — Zero external API dependencies. All models are serialized locally into a single `Speech.pkl` brain for sub-second startup and complete data privacy.
- **Production Infrastructure** — Dockerized FastAPI service with optimized memory management.

---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/Amr-Yasser-7/ara-route-parser.git
cd ara-route-parser
pip install -r requirements.txt
```

### 2. Cold Start — Download Models

The repository is kept lightweight (~50KB). The 1.5GB models are downloaded once via the setup script:

```bash
python scripts/save_models.py   # Downloads & serializes AI models locally
python app/main.py              # Generates Speech.pkl inference brain
```

> **Note:** This step only needs to run once. All subsequent startups use the local `Speech.pkl` file.

### 3. Run the API

```bash
# Local development
python app/main.py

# Via Docker
docker build -t ara-route-parser .
docker run -p 8000:80 ara-route-parser
```

API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Project Structure

```
ara-route-parser/
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── engine.py             # Unified dispatch logic
│   ├── advanced_qa.py        # AraElectra QA model interface
├── docs/
│   └── walkthrough.md        # Deep-dive architecture notes
├── scripts/
│   └── save_models.py        # One-time model download & serialization
├── .gitignore               
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Author

Developed by **Amr Yasser**

- **GitHub**: [@Amr-Yasser-7](https://github.com/Amr-Yasser-7)
- **Focus**: Machine Learning Engineering · NLP · Arabic Dialect Processing

---

*Built for real-world deployment in Arabic-speaking transport and ride-hailing contexts.*