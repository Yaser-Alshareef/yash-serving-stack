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

## KV cache estimate (RAM perspective)

We also estimated the memory cost of the KV cache during generation. This is a RAM estimate, not GPU memory. The idea is that the cache grows with the number of layers, key/value heads, and hidden dimension.

For this model we used:

- Layers: 28
- KV heads: 2
- Head dim: hidden_size / num_attention_heads = 1536 / 12 = 128
- Data type: fp16 = 2 bytes per value

So the rough per-token cache size is:

- KV cache per token ≈ 2 × 28 × 2 × 128 × 2 bytes
- ≈ 28,672 bytes
- ≈ 28 KB/token

For a long generation window of 16,384 tokens:

- 28 KB × 16,384 ≈ 0.44 GB

This means that the KV cache can use roughly 0.44 GB of RAM in the worst-case generation window.

To estimate how many conversations could fit in the remaining RAM, we divide the usable RAM by the KV-cache footprint:

- 7.6 GB / 0.44 GB ≈ 17 conversations

So the rough estimate is that around 17 conversations could fit under this RAM-based worst-case assumption, if the rest of the memory is available for the cache and not consumed by other processes.

---

## Notebook investigation

The notebook work and saved result files are the main evidence for the performance and memory checks in this project.

### Week 3 Day 1 investigation

The main notebook in this section is the T4 GPU profiling work. It loaded the Qwen2.5-1.5B model in both fp16 and int8 modes and measured token throughput and GPU memory usage under different context lengths.

The saved results in [lab_results/w3d1/profile.json](lab_results/w3d1/profile.json) show:

- fp16 at 512 tokens: 3.113 GB VRAM, 54.0% utilization, 29.7 tokens/s
- fp16 at 2048 tokens: 3.295 GB VRAM, 58.6% utilization, 24.5 tokens/s
- fp16 at 4096 tokens: 3.568 GB VRAM, 89.6% utilization, 26.4 tokens/s
- int8 at 512 tokens: 1.805 GB VRAM, 25.9% utilization, 5.9 tokens/s
- int8 at 2048 tokens: 2.035 GB VRAM, 29.5% utilization, 5.6 tokens/s
- int8 at 4096 tokens: 2.309 GB VRAM, 34.8% utilization, 5.4 tokens/s

This shows that fp16 gives much better throughput, but it uses more memory. int8 is much lighter on VRAM, but the generation speed is noticeably lower.

We also checked for a memory leak in the reload path. The results in [lab_results/w3d1/leak_report.json](lab_results/w3d1/leak_report.json) show:

- leaky run slope: 211.248 MB per iteration
- fixed run slope: 0.286 MB per iteration
- leak status: true for the leaky run, false for the fixed run

This was a clear demonstration that a bad unload or reload pattern could leak memory across repeated runs.

### Week 3 Day 2 investigation

The second notebook set focused on streaming latency and KV-cache behavior. The notebooks and saved reports show how prompt length, memory footprint, and chunked allocation affect concurrency.

The KV-cache result file [lab_results/w3d2/kv_check.json](lab_results/w3d2/kv_check.json) shows:

- formula KB/token: 28.0
- measured KB/token: 28.0
- peak KB/token: 87.6

This means the theoretical cache estimate matched the measured value, but the peak could still be higher under real generation conditions.

The simulation report in [lab_results/w3d2/kv_sim_report.json](lab_results/w3d2/kv_sim_report.json) shows:

- slab peak concurrent: 18
- slab admitted: 18
- slab rejected: 42
- block pool peak concurrent: 60
- block pool admitted: 60
- block pool rejected: 0
- blockpool advantage: 3.33x

This indicates that the block-pool allocation method handled memory more efficiently and allowed much higher concurrency than the simple slab approach.

### Notebook files in the repo

The investigation notebooks are stored in:

- [notebooks/w3d1/w3d1_Lab.ipynb](notebooks/w3d1/w3d1_Lab.ipynb)
- [notebooks/w3d1/w3d1_leak.ipynb](notebooks/w3d1/w3d1_leak.ipynb)
- [notebooks/w3d1/w3d1_leak_bug.ipynb](notebooks/w3d1/w3d1_leak_bug.ipynb)
- [notebooks/w3d2/w3d2.ipynb](notebooks/w3d2/w3d2.ipynb)
- [notebooks/w3d2/w3d2_bug.ipynb](notebooks/w3d2/w3d2_bug.ipynb)
- [notebooks/w3d2/w3d2_KV.ipynb](notebooks/w3d2/w3d2_KV.ipynb)

These notebooks and the saved JSON outputs together form the main evidence for the memory, throughput, and KV-cache experiments in this project.

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
