"""
Arabic Speech-to-Route
Model initialization and pipeline orchestration.
Supports both Direct Loading (Option B) and Pickle loading (Option A) as fallback.
"""
import os
import pickle
import sys

# Ensure src directory is in path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

def load_models_direct():
    """Option B: Load models directly into memory (Modern Standard)"""
    print("Loading models directly...")
    try:
        # 1. Load Whisper
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        WHISPER_MODEL_ID = "itshamdi404/Egy_Arabic_whisper-small"
        whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID, language="ar", task="transcribe")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
        whisper_model.eval()

        # 2. Load Extractor
        from route_extractor import advanced_extract_route
        extract_route = advanced_extract_route

        print("Models loaded directly successfully.")
        return whisper_processor, whisper_model, extract_route

    except Exception as e:
        print(f"Failed to load models directly: {e}")
        return None, None, None

def load_models_pickle():
    """Option A: Load models from Pickle (Fallback)"""
    print("Loading models from Speech.pkl (fallback)...")
    pickle_path = os.path.join(os.path.dirname(_src_dir), "models", "Speech.pkl")
    try:
        with open(pickle_path, "rb") as f:
            Speech = pickle.load(f)
        print("Models loaded from pickle successfully.")
        return Speech["whisper_processor"], Speech["whisper_model"], Speech["extract_route"]
    except Exception as e:
        print(f"Failed to load Speech.pkl from {pickle_path}: {e}")
        return None, None, None

def initialize_pipeline(force_pickle=False):
    """
    Initializes the NLP pipeline.
    Prioritizes Direct Loading unless force_pickle is True or Direct Loading fails.
    """
    if force_pickle:
        return load_models_pickle()
        
    proc, mod, ext = load_models_direct()
    if proc is not None and mod is not None and ext is not None:
        return proc, mod, ext
        
    print("Direct loading failed, attempting pickle fallback...")
    return load_models_pickle()
