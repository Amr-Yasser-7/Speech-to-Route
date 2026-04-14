import os
import shutil
import re
import json
import torch
import pickle

# STAGE 1: Egyptian Arabic Speech-to-Text 

try:
    print("Loading Egyptian Arabic Whisper model... ")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    WHISPER_MODEL_ID = "itshamdi404/Egy_Arabic_whisper-small"
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID, language="ar", task="transcribe")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
    whisper_model.eval()  # Set to inference mode
    print("Egyptian Whisper model loaded successfully.")
except Exception as e:
    print(f"Failed to load Egyptian Whisper model: {e}")
    whisper_processor = None
    whisper_model = None

# STAGE 2: Transportation Route Extraction 

try:
    from route_extractor import advanced_extract_route
    
    def extract_route_from_text(text: str) -> dict:
        ai_res = advanced_extract_route(text)
        return {"origin": ai_res.get("origin"), "destination": ai_res.get("destination")}
        
    print("AI Dispatcher  integrated.")

except Exception as e:
    print(f"Failed to load AI Dispatcher: {e}")
    
    def extract_route_from_text(text: str) -> dict:
        return {"origin": None, "destination": None}


# Aggregate models and functions into the Speech object for pickling
Speech = {
    "whisper_model": whisper_model,
    "whisper_processor": whisper_processor,
    "extract_route": extract_route_from_text
}

# Save to models directory
os.makedirs("../models", exist_ok=True)
with open("../models/Speech.pkl", "wb") as pickle_out:
    pickle.dump(Speech, pickle_out)
