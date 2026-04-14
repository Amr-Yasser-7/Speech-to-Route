FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python save_models.py && python main.py

ENV PORT=80
EXPOSE 80

CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker speech_recognition_api:app --bind 0.0.0.0:$PORT --timeout 600
