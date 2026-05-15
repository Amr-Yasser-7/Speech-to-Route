# Contributing to Arabic Speech-to-Route

Thank you for your interest in contributing! This project is an open-source demonstration of Egyptian dialect speech-to-route extraction.

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download standard models: `python scripts/download_models.py`
4. Place any fine-tuned NER models in the `models/` directory.

## Project Structure

- `src/`: Core pipeline logic (Normalization, NER, Gazetteer, Syntactic Fallback)
- `speech_recognition_api.py`: FastAPI server
- `scripts/`: Utility scripts (e.g., model downloading)

## Guidelines

- Keep code modular and strictly typed where possible.
- Avoid committing large files (e.g., `.pkl`, `.bin`, `.onnx`). Use `scripts/download_models.py` or document where to fetch them.
- Ensure the API response schema remains backward compatible.
