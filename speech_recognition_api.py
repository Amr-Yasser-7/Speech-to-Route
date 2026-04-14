import uvicorn
import os
import io
import torch
import librosa
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
# This import is CRITICAL for pickle to resolve the function reference
from src.main import extract_route_from_text

# 2. Create the app object
app = FastAPI(title="Speech Recognition")

# Load the brain from the pickle file
pickle_in = open("models/Speech.pkl", "rb")
Speech = pickle.load(pickle_in)

# Extract components from pickle
whisper_model = Speech["whisper_model"]
whisper_processor = Speech["whisper_processor"]
extract_route = Speech["extract_route"]

# 3. Index route
@app.get('/')
def index():
    return {'message': 'Speech Recognition is Running'}

# 4. Name parameter route (Match example style)
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

# 5. Expose the prediction functionality
@app.post('/predict')
async def predict_route(file: UploadFile = File(...)):
    try:
        # Load audio from upload
        content = await file.read()
        audio, sr = librosa.load(io.BytesIO(content), sr=16000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio Load Error: {str(e)}")

    # STAGE 1: Neural Transcription
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = whisper_model.generate(inputs["input_features"], language="arabic", task="transcribe")
    transcription = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]

    # STAGE 2: Neural Route Extraction (Loading from Pickle)
    route = extract_route(transcription)

    return {
        'transcription': transcription,
        'origin': route.get('origin'),
        'destination': route.get('destination'),
        'status': 'Success'
    }

# 6. Run the API
if __name__ == '__main__':
    # Use 127.0.0.1 for local accessibility on Windows
    uvicorn.run(app, host='127.0.0.1', port=8000)