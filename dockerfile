FROM python:3.11-slim

RUN useradd --create-home app
WORKDIR /code

# Install deps first so code edits don't invalidate this layer.
COPY app/requirements.txt app/requirements.txt

# Use the CPU wheel index for torch to avoid pulling CUDA wheels into a
# CPU-only image. Drop --index-url if you build a GPU image instead.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r app/requirements.txt

# Copy the app package, keeping the nested layout main.py depends on
# ("from app.schemas import ..." and "uvicorn app.main:app").
COPY app/ app/

RUN mkdir -p /code/model && chown -R app:app /code
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=90s \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]