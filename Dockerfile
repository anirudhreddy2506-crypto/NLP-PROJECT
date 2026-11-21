# VisionCaption AI - Dockerfile
# Production-ready containerized deployment for Streamlit & PyTorch

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies for Pillow, PyTorch, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, demo data, and assets
COPY app.py /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY data/ /app/data/
COPY outputs/ /app/outputs/
COPY models/ /app/models/
COPY docs/ /app/docs/

# Create required output and cache directories with appropriate permissions
RUN mkdir -p /app/outputs/logs /app/outputs/eval_results /app/models/fine_tuned

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]

