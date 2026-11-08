# =============================================================================
# Task 6.2 & Task 6.4: Optimized Production Dockerfile for VisionGuard Web App
# Project: L231014_A2_Ops | Author: Junaid Muhammad (23L-1014)
# =============================================================================
# 1. Slim base image (Debian Bookworm slim + Python 3.12)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/mobilenetv2_finetuned_best.keras \
    DECISION_THRESHOLD=0.48

WORKDIR /app

# 2. Install system library needed by curl healthcheck, then clean apt cache in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy ONLY requirements first so Docker caches the heavy pip install layer
#    unless dependencies change (Task 6.2 Layer Cache Optimization)
COPY requirements-docker.txt /app/requirements-docker.txt

# 4. Install CPU-only TensorFlow (tensorflow-cpu) with --no-cache-dir (Task 6.2 & 6.4)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-docker.txt

# 5. Create a dedicated non-root user and group (Task 6.2 Security Requirement)
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# 6. Copy application source code, templates, static assets, and the champion model
COPY src/inference.py src/app.py /app/src/
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY models/mobilenetv2_finetuned_best.keras /app/models/mobilenetv2_finetuned_best.keras

# Ensure non-root user owns the working directory (for SQLite fallback or runtime logs)
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# 7. Container-level Health Check probing Flask's /health endpoint (Task 6.2)
HEALTHCHECK --interval=20s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run production WSGI server with debug mode off
CMD ["python", "src/app.py"]
