import uvicorn
import os
import io
import torch
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")

# Import the new initialization logic
from src.main import initialize_pipeline

app = FastAPI(title="Arabic Speech-to-Route API", description="Production API for Arabic Voice Route Extraction")

# Add CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (Direct Loading with Pickle fallback)
whisper_processor, whisper_model, extract_route = initialize_pipeline()

if not all([whisper_processor, whisper_model, extract_route]):
    print("WARNING: Models could not be loaded. API will start but predictions will fail.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get('/')
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {'message': 'Arabic Speech-to-Route API is Running'}

@app.get('/health')
def health():
    return {
        'status': 'healthy', 
        'models_loaded': all([whisper_processor, whisper_model, extract_route])
    }

class TextRequest(BaseModel):
    text: str

@app.post('/extract')
async def extract_route_text(request: TextRequest):
    """Extract route directly from text input (no audio processing)."""
    if not extract_route:
        raise HTTPException(status_code=500, detail="Extractor model not loaded")
        
    route = extract_route(request.text)
    
    return {
        'transcription': request.text,
        'origin': route.get('origin'),
        'destination': route.get('destination'),
        'status': 'Success'
    }

@app.post('/predict')
async def predict_route(file: UploadFile = File(...)):
    """Extract route from audio file upload."""
    if not all([whisper_processor, whisper_model, extract_route]):
        raise HTTPException(status_code=500, detail="Models not fully loaded")

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

    # STAGE 2: Route Extraction
    route = extract_route(transcription)

    return {
        'transcription': transcription,
        'origin': route.get('origin'),
        'destination': route.get('destination'),
        'status': 'Success'
    }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("speech_recognition_api:app", host='0.0.0.0', port=port, reload=True)
