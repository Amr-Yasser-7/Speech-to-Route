"""
Model Download Script for arabic-speech-to-route
Downloads necessary Hugging Face models so they are cached locally.
"""
import os
import argparse

def download_models():
    print("Downloading Whisper model (itshamdi404/Egy_Arabic_whisper-small)...")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    WHISPER_MODEL_ID = "itshamdi404/Egy_Arabic_whisper-small"
    
    # This will download and cache the models
    WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
    
    print("\n✅ Hugging Face models downloaded successfully.")
    print("\nNote: For the fine-tuned NER model (CAMeLBERT/EgyBERT), please place ")
    print("the model files in the 'models/' directory if not already present.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download required ML models")
    args = parser.parse_args()
    
    download_models()
