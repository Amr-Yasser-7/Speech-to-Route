# 🗺️ Arabic Speech-to-Route (مُسْتَخْرِج المسارات)

> **A smart, dialect-aware AI engine that listens to Egyptian Arabic speech and figures out exactly where you want to go.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![Hugging Face Space](https://img.shields.io/badge/🤗_Hugging_Face-Space-blue.svg?style=flat-square)](https://huggingface.co/spaces/Amr-Yasserr/ara-route-parser)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## 🚀 Try the Live Demo

You can test the model immediately without installing anything! 
👉 **[Click here to try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Amr-Yasserr/ara-route-parser)**

## 👋 Hello! Welcome to the Project

Building mobility and ride-hailing apps for the Middle East is challenging. When a user in Cairo opens a ride app and speaks to it, they don't use formal, textbook Arabic (Modern Standard Arabic). Instead, they speak in rapid, informal Egyptian dialect. 

They might say: *"عايز اروح التجمع الخامس من الهرم"* (I wanna go to the 5th Settlement from Haram).

**Arabic Speech-to-Route** is an open-source NLP pipeline built specifically to solve this problem. It takes raw, noisy voice input, transcribes it, and extracts the exact `Origin` and `Destination` using a blend of modern Deep Learning and linguistic rules.

---

## 🧩 Why is this a hard problem?

If you've worked with NLP, you might wonder why we can't just use standard tools. Here is why Egyptian Arabic routing requires a custom approach:

1. **The "Dialect" Problem:** Standard models expect formal grammar ("أريد الذهاب إلى"). Egyptians use colloquial structures ("خدني على", "رايح", "عايز اروح").
   * 👉 **Our Solution:** We fine-tuned a CAMeLBERT Transformer model on thousands of synthetically generated Egyptian routing phrases to understand these exact structures.
2. **The "Speech-to-Text" Problem (ASR Noise):** Speech models often misspell locations. They might hear "مدينه نصر" instead of the correct "مدينة نصر" (missing the dots on the Taa Marbuta).
   * 👉 **Our Solution:** We built a custom text normalizer and a phonetic (Soundex) gazetteer. Even if the AI severely misspells a location, our fuzzy-matching engine catches it and snaps it to the official map name.
3. **The "Implicit Context" Problem:** People speak lazily. Sometimes they say "من المعادي التحرير" (From Maadi Tahrir) without a preposition for the destination.
   * 👉 **Our Solution:** Our hybrid architecture uses AI (NER) as the brain, backed by syntactic regex fallbacks as a safety net.

---

## 🧠 How the Brain Works (Architecture)

We use a modern **Direct Loading** architecture for optimal API performance and security.

```text
🎤 Voice Input 
  │
  ▼
[ Whisper ASR ] ──▶ Translates voice to text ("من المعادي للتحرير")
  │
  ▼
[ Text Normalizer ] ──▶ Cleans up Arabic characters, removes "ums" and "ahs"
  │
  ▼
[ NER Model ] ──▶ AI reads the text and tags the locations (B-ORIGIN, B-DEST)
  │
  ▼
[ Gazetteer ] ──▶ Double-checks the AI's answer against a real database
  │
  ▼
✅ JSON Output { "origin": "المعادي", "destination": "التحرير" }
```

*Curious about the technical deep dive? Read our [Architecture Documentation](docs/architecture.md).*

---

## 🚀 Get It Running on Your Machine

Want to test it out? It's easy to spin up locally.

### 1. Clone & Install
```bash
git clone https://github.com/Amr-Yasser-7/Speech-to-Route.git
cd Speech-to-Route
pip install -r requirements.txt
```

### 2. Download the Models
Because AI models are huge, we don't store them in this GitHub repo. Run this script to download the Whisper Speech-to-Text model automatically:
```bash
python scripts/download_models.py
```
*(Note: If you want to run the NER model locally, place your fine-tuned `model.onnx` inside the `models/` folder).*

### 3. Start the Engine
```bash
uvicorn speech_recognition_api:app --reload
```
You can now open `http://localhost:8000/docs` to see the beautiful Swagger UI and test the API!

---

## 🔌 API Quick Reference

### Test with Text
**POST** `/extract`
```json
// You send:
{
  "text": "عايز اروح التجمع الخامس من الهرم"
}

// The API replies:
{
  "transcription": "عايز اروح التجمع الخامس من الهرم",
  "origin": "الهرم",
  "destination": "التجمع الخامس",
  "status": "Success"
}
```

### Test with Voice
**POST** `/predict`
Simply upload a `.wav` or `.mp3` file as multipart/form-data. The API will transcribe the audio and return the exact same JSON format!

---

## 🐳 Docker (Production Ready)

If you're deploying to the cloud (like AWS, GCP, or Hugging Face Spaces), use our optimized Dockerfile:
```bash
docker build -t arabic-speech-to-route .
docker run -p 7860:7860 arabic-speech-to-route
```

---

## 🛡️ Security

We take security seriously. Please review our [Security Policy](SECURITY.md) for reporting vulnerabilities. We use automated dependency scanning to keep the project safe.

## 🤝 Let's Build Together!

This project is entirely open-source. Whether you want to add more locations to the database, improve the text normalizer, or retrain the NER model on a different Arabic dialect (like Gulf or Levantine), we welcome your contributions! 

Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---
*Built with ❤️ by [Amr Yasser]*
