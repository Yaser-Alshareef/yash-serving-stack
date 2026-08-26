FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation in linux
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home app


# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir \
    -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache /app

USER app

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]