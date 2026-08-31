# Yash Serving Stack

This repo is our current progress in building a local model serving stack. We are learning how to serve a Qwen model using FastAPI, Pydantic, Docker, and Hugging Face tools, and we are testing different setups on CPU and GPU.

This is not a final production project yet. It is a working progress repo where each lab adds a new step to the system.

---

## Current status

We have completed the following stages so far:

- Model loading on CPU
- FastAPI server setup
- OpenAI-like API structure
- Pydantic validation and error testing
- Docker container setup
- GPU detection with CUDA
- Docker Compose configuration
- API key protection layer
- model serving experiments and benchmarking

---

## Lab progress

### Lab 1 in Week 2
We installed the Qwen weights on CPU and compared different model formats such as fp16 and other formats. We checked how many tokens each format produced and compared the results between formats.

### Lab 2 in Week 2
We built the service with FastAPI, Pydantic, and OpenAI-like endpoint structure. We used Uvicorn to serve the app and tested requests. We also changed the schemas to check validation behavior, including 422 errors from Pydantic.

### Lab 3 in Week 2
We created a Dockerfile and ran the app inside a container. We mounted the model from a local drive and also considered loading it from Hugging Face. After building the image, we pushed it to a container registry and tested the service again after pulling and running it.

### Lab 4 in Week 2
We implemented GPU handling using torch.cuda. The app checks whether CUDA is available; if yes, it uses GPU, otherwise it falls back to CPU. We also used a CUDA base image for the GPU setup.

### Lab 5 in Week 2
We improved the CPU service with the v2 setup. This version handled the env file, Docker Compose config, and API protection. The service was shaped more like an OpenAI-compatible model API, but still served our own local model.

### Lab 1 in Week 3
We used Google Colab with a T4 GPU and investigated the model behavior in terms of GPU utilization and token generation. We saved the results in JSON format in different data types for comparison and analysis.

---

## What is inside this repo

- app/main.py — FastAPI app and inference logic
- app/schemas.py — schemas for requests and responses
- app/registry.json — registry metadata for models
- model/ — local model files
- dockerfile — main Docker image
- dockerfile.cpuV1 / dockerfile.cpuV2 — CPU image variants
- dockerfile.gpu — GPU image
- compose.yaml — Docker Compose setup
- requirements.txt — Python dependencies
- env.example — sample environment configuration
- client.py — client for testing the service
- fuzz_client.py — fuzz test client
- load_gen.py — load generation helper
- verify.py / verify_cell.py — verification scripts
- chaos/ — chaos-related tests
- docs/ — docs and notes
- notebooks/ — experiments and notebook work
- lab_results/ — saved benchmark and measurement outputs

---

## Current technical setup

The project is using:

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- Transformers
- PyTorch
- Docker
- Docker Compose

The app currently loads a local Qwen model, checks whether CUDA is available, and serves the model through routes such as:

- /health
- /v1/models
- /registry
- /v1/chat/completions
- /v1/embeddings

---

## Local running flow

1. Copy env.example to .env
2. Set the model path, API key, and token limit
3. Install dependencies
4. Run the server with Uvicorn
5. Test endpoints with a client

Example:

```bash
cp env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker flow

The repo contains different Docker builds for different runtime targets:

- dockerfile.cpuV1 and dockerfile.cpuV2 — CPU-only serving images
- dockerfile.gpu — GPU-enabled container setup with CUDA support
- dockerfile — main image used for the current setup

Examples:

```bash
# CPU image
docker build -f dockerfile.cpuV2 -t yash-serving:cpu .

# GPU image
docker build -f dockerfile.gpu -t yash-serving:gpu .
```

The project also supports containerized deployment with Compose:

```bash
docker compose up --build
```

This lets the app run in a more isolated environment, with the model mounted from the local folder.

---

## API protection layer

The app includes an API key protection layer for the protected routes. The server checks the X-API-Key header before allowing access to model and serving endpoints.

This is important for the /v1 routes and other protected endpoints, so requests without the correct key are rejected with a 401 error.

---

## Example curl calls

### Health check

```bash
curl http://localhost:8000/health
```

### Model list with API key

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/v1/models
```

### Chat completion request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello there"}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": false,
    "require_gpu": false
  }'
```

This shows how the service is called in practice and how the API protection layer is used in real requests.

---

## Important notes

- Keep .env local and do not commit secrets
- GPU usage depends on CUDA availability in the environment
- CPU and GPU Docker variants are built for different hardware targets
- Protected endpoints require the X-API-Key header

---

## Summary

This project is still in progress, but the core flow is already working: we are loading a Qwen model, serving it through FastAPI, validating requests with Pydantic, packaging it with Docker, and testing it across different hardware setups. Each lab added a new layer to the stack, and the repo reflects that progress clearly.

This is the current state of the project, and it is continuing to evolve as we add more experiments and deployment improvements.
