FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for opencv)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure thumbnail dir exists
RUN mkdir -p uploads/thumbnails

EXPOSE 8000

ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
