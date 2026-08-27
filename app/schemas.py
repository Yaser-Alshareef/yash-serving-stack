from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    prompt: str
    require_gpu: bool = False


class ChatCompletionRequest(BaseModel):
    model: str

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1
    )

    max_tokens: int = Field(
        default=256,
        ge=1
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0
    )

    stream: bool = False

    require_gpu: bool = False

    # The chat endpoint expects the conversation to end with
    # a user or system message so the model can generate the assistant reply.
    @field_validator("messages")
    @classmethod
    def last_message_must_be_user_or_system(cls, value):
        if value and value[-1].role == "assistant":
            raise ValueError(
                "the last message must be from "
                "'user' or 'system', not 'assistant'"
            )

        return value


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "aidc"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    # Includes the device because /health returns both the model
    # and the device currently being used (CPU or CUDA).
    status: Literal["ok"] = "ok"
    model: str
    device: str