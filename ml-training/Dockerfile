# Hugging Face Spaces Dockerfile
# Deploys PyTorch DQN inference server on HF Spaces (16 GB RAM free tier)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY inference_server.py .
COPY models/ ./models/
COPY checkpoints/best_dqn_model.pth ./checkpoints/best_dqn_model.pth

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

# Run with gunicorn (production-grade WSGI server)
# - Single worker (PyTorch models aren't thread-safe)
# - 120s timeout (ML inference can be slow)
# - Bind to 0.0.0.0:7860 (HF Spaces requirement)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "inference_server:app"]
