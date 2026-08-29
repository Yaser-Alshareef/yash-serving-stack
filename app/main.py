#Libraries
from __future__ import annotations

import json
import os
import time
import uuid

from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader


from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)


BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY = BASE_DIR/"app"/"registry.json"
ENV_FILE = BASE_DIR/"app"/".env"

load_dotenv(ENV_FILE)


#-------------

#FastAPI app instance
app = FastAPI()

API_KEY = os.environ.get("API_KEY", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen2.5-1.5B-Instruct"
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    f"{BASE_DIR}/model/Qwen2.5-1.5b"
)

HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} from {MODEL_PATH}")
print(f"Running on: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


print(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32, # torch.bfloat16 if DEVICE == "cuda" else torch.float32 it cause me a problem here );
)

model.to(DEVICE)
model.eval()

print("Model ready")


#---------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        # if you never set an API_KEY in .env, don't accidentally lock yourself out
        raise HTTPException(500, "Server misconfigured: no API_KEY set")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

#----------------


# Endpoints Using FastAPI()
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE
    }


@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
def list_models() -> ModelList:
    return ModelList(
        data=[
            ModelCard(
                id=MODEL_ID,
                created=int(time.time()),
                owned_by="aidc"
            )
        ]
    )


def load_registry_data():
    with open(REGISTRY, "r") as f:
        return json.load(f)

@app.get("/registry", dependencies=[Depends(verify_api_key)])
def list_registry():
    registry_data = load_registry_data()
    return {"models": list(registry_data.keys())}

@app.get("/registry/{name}", dependencies=[Depends(verify_api_key)])
def get_model(name: str):
    registry_data = load_registry_data()

    if name not in registry_data:
        raise HTTPException(status_code=404, detail=f"no such model: {name}")

    return registry_data[name]


@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
def embeddings(payload: dict):
    if DEVICE != "cuda":
        raise HTTPException(400, "Embeddings require a GPU-backed instance; this instance is running in CPU-fallback mode.")
    return {"vector": [0.1] * 8, "device_used": DEVICE}



def _build_inputs(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = inputs["input_ids"]

    return input_ids, input_ids.shape[1]

def _generate(
    input_ids,
    req: ChatCompletionRequest
):
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=(
                req.temperature
                if req.temperature > 0
                else None
            ),
            pad_token_id=tokenizer.eos_token_id,
        )

    return out[0][input_ids.shape[1]:]


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
def chat_completions(
    req: ChatCompletionRequest
):
    if req.require_gpu and DEVICE != "cuda":
        raise HTTPException(
            status_code=400,
            detail=(
                "This request set require_gpu=true, but this instance is "
                "running in CPU-fallback mode (no CUDA device available). "
                "Retry against a GPU-backed instance, or drop require_gpu."
            ),
        )

    if req.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"model '{req.model}' not found"
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            },
        )

    input_ids, prompt_tokens = _build_inputs(req)

    if req.stream:
        return _stream(
            input_ids,
            prompt_tokens,
            req
        )

    new_tokens = _generate(
        input_ids,
        req
    )

    completion_tokens = int(
        new_tokens.shape[0]
    )

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=text
                ),
                finish_reason=(
                    "length"
                    if completion_tokens >= req.max_tokens
                    else "stop"
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens + completion_tokens
            ),
        ),
    )


def _stream(
    input_ids,
    prompt_tokens: int,
    req: ChatCompletionRequest
):
    new_tokens = _generate(
        input_ids,
        req
    )

    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    def chunk(delta: dict, finish=None):
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish
                }
            ],
        }

        return (
            "data: "
            + json.dumps(payload)
            + "\n\n"
        )

    def events():
        yield chunk(
            {
                "role": "assistant",
                "content": ""
            }
        )

        for tok in new_tokens:
            piece = tokenizer.decode(
                [tok],
                skip_special_tokens=True
            )

            if piece:
                yield chunk(
                    {"content": piece}
                )

        yield chunk(
            {},
            finish="stop"
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream"
    )