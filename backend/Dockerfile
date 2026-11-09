# ============================================================
# Smart ICU Backend Dockerfile
# Pinned Python 3.11 slim base image for deterministic reproducibility
# ============================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/backend" \
    SMART_ICU_SERIAL_PORT="NONE"

# Set working directory to project root
WORKDIR /app

# Install system dependencies needed for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source code
COPY backend /app/backend

# Copy tracked data assets and baseline ML models into image
COPY data/processed/eicu_population_reference_v1.json /app/data/processed/eicu_population_reference_v1.json
COPY ml/models /app/ml/models
COPY ml/candidate_artifacts_v2 /app/ml/candidate_artifacts_v2

# Create directories for dynamic data and persistence
RUN mkdir -p /app/data/processed

# Set working directory to backend module
WORKDIR /app/backend

# Expose FastAPI port
EXPOSE 8000

# Health check using curl against /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Start FastAPI application via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
