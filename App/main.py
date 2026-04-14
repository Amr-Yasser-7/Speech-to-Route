import os
import shutil
import re
import json
import torch
import pickle

# STAGE 1: Egyptian Arabic Speech-to-Text (Whisper fine-tuned)

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

# STAGE 2: Transportation Route Extraction (Hybrid AI-Rule Dispatcher)

try:
    from advanced_qa_dispatcher import advanced_extract_route
    from syntactic_dispatcher import syntactic_extract_route
    
    def extract_route_from_text(text: str) -> dict:
        ai_res = advanced_extract_route(text)
        rule_res = syntactic_extract_route(text)
        
        # Default to AI results
        final_route = {"origin": ai_res.get("origin"), "destination": ai_res.get("destination")}
        
        # Use Rules as safety net if AI confidence is low or values are missing
        if not final_route["origin"] or ai_res.get("origin_score", 0) < 0.6:
             final_route["origin"] = rule_res["origin"]
             
        if not final_route["destination"] or ai_res.get("destination_score", 0) < 0.6:
             final_route["destination"] = rule_res["destination"]
             
        if rule_res["origin"] and rule_res["destination"]:
            return rule_res

        return final_route
        

except Exception as e:
    print(f"Failed to load Hybrid Dispatcher: {e}")
    try:
            from syntactic_dispatcher import syntactic_extract_route
            def extract_route_from_text(text: str) -> dict:
                return syntactic_extract_route(text)
    except:
        def extract_route_from_text(text: str) -> dict:
            return {"origin": None, "destination": None}

# Aggregate models and functions into the Speech object for pickling
Speech = {
    "whisper_model": whisper_model,
    "whisper_processor": whisper_processor,
    "extract_route": extract_route_from_text
}

pickle_out = open("Speech.pkl","wb")
pickle.dump(Speech,pickle_out)
pickle_out.close()

with open("Speech.pkl", "wb") as pickle_out:
        pickle.dump(Speech, pickle_out)