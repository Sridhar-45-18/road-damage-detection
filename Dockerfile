FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Install required system dependencies (ffmpeg and libraries for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (using cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model weights
COPY . .

# Ensure upload and result directories exist
RUN mkdir -p webapp/uploads webapp/results_web

EXPOSE 10000

# Start with Gunicorn: 1 worker + 4 threads to fit comfortably within free tier memory
CMD exec gunicorn --chdir webapp app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}
