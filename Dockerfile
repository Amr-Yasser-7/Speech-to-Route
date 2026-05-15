FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download standard models during build to optimize cold starts
RUN python scripts/download_models.py

# Expose port (7860 is default for Hugging Face Spaces)
EXPOSE 7860
ENV PORT=7860

# Run the FastAPI server
CMD ["uvicorn", "speech_recognition_api:app", "--host", "0.0.0.0", "--port", "7860"]
