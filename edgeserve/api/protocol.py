"""OpenAI-compatible request models (the supported subset) and response builders.

Unknown fields are rejected (HTTP 400) instead of silently ignored, so a typo or an unsupported
feature (tools, response_format, images, ...) never produces a result the client did not ask for.
Extensions beyond the OpenAI API: ``request_id`` (client ID, same as the X-Request-Id header),
``top_k`` and ``return_record`` (include the full structured record in the response).
"""

from __future__ import annotations

import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_MESSAGES = 64  # Edge-LLM input limits (cpp/common/inputLimits.h)
MAX_CONTENT_BYTES = 128 * 1024
MAX_STOP_SEQUENCES = 4
MAX_ID_LENGTH = 512
MAX_INT32 = 2**31 - 1  # token counts and top_k are 32-bit in the worker
MAX_TOP_K = 1024  # Edge-LLM's top-k sampler sizes shared memory and its workspace by top_k


def _utf8_size(text: str, what: str) -> int:
    """Encoded size; text without a UTF-8 form (an unpaired surrogate from a JSON escape) is refused: neither the
    worker nor a JSON response or record lookup could carry it."""
    try:
        return len(text.encode())
    except UnicodeEncodeError:
        raise ValueError(f"{what} is not valid Unicode text (it contains an unpaired surrogate)") from None


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


# Message fields of unsupported features; null means "not used" (clients replaying earlier responses send them).
_UNSUPPORTED_NULLABLE_IN_MESSAGES = frozenset({"tool_calls", "function_call", "audio", "refusal", "annotations"})


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _unused_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not (v is None and k in _UNSUPPORTED_NULLABLE_IN_MESSAGES)}
        return data

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else "\n".join(p.text for p in self.content)

    @model_validator(mode="after")
    def _size(self) -> ChatMessage:
        if _utf8_size(self.text(), "message content") > MAX_CONTENT_BYTES:
            raise ValueError(f"message content is larger than {MAX_CONTENT_BYTES // 1024} KB")
        return self


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


# OpenAI fields this server does not support. Some clients send every field, with null for the unused ones: a null
# means "not used" and is accepted; any other value is refused as an unsupported parameter.
_UNSUPPORTED_NULLABLE = frozenset({
    "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call", "response_format",
    "logit_bias", "top_logprobs", "modalities", "audio", "prediction", "reasoning_effort", "service_tier",
    "store", "web_search_options", "suffix", "best_of", "verbosity", "moderation", "prompt_cache_options",
    "prompt_cache_retention",
})


class _GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    max_tokens: int | None = Field(None, ge=1, le=MAX_INT32)
    max_completion_tokens: int | None = Field(None, ge=1, le=MAX_INT32)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=0, le=MAX_INT32)
    seed: int | None = Field(None, ge=0, le=2**64 - 1)
    stop: str | list[str] | None = None
    n: int | None = Field(1, ge=1, le=1)  # OpenAI allows null for n, stream and echo (= the default)
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None  # an end-user ID (OpenAI: abuse monitoring), accepted and ignored, as are its successors:
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None  # a prompt-caching hint
    metadata: dict[str, str] | None = None
    frequency_penalty: float | None = Field(None, ge=0, le=0)
    presence_penalty: float | None = Field(None, ge=0, le=0)
    logprobs: bool | None = Field(None)
    request_id: str | None = None
    return_record: bool = False

    @model_validator(mode="before")
    @classmethod
    def _unused_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not (v is None and k in _UNSUPPORTED_NULLABLE)}
        return data

    @field_validator("stream", mode="before")
    @classmethod
    def _stream(cls, value: object) -> object:
        return False if value is None else value

    @field_validator("stop")
    @classmethod
    def _stop(cls, value: str | list[str] | None) -> list[str] | None:
        if value is None:
            return None
        values = [value] if isinstance(value, str) else value
        if len(values) > MAX_STOP_SEQUENCES or any(not v for v in values):
            raise ValueError(f"stop must be a non-empty string or up to {MAX_STOP_SEQUENCES} non-empty strings")
        for v in values:
            _utf8_size(v, "stop")
        return values

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is not None and (len(value) > 16 or any(len(k) > 64 or len(v) > 512 for k, v in value.items())):
            raise ValueError("metadata allows at most 16 pairs, keys up to 64 and values up to 512 characters")
        for key, text in (value or {}).items():
            _utf8_size(key, "metadata")
            _utf8_size(text, "metadata")
        return value

    @field_validator("logprobs")
    @classmethod
    def _no_logprobs(cls, value: bool | None) -> bool | None:
        if value:
            raise ValueError("logprobs are not supported by this server")
        return value

    @model_validator(mode="after")
    def _token_limits(self):
        if self.max_tokens and self.max_completion_tokens and self.max_tokens != self.max_completion_tokens:
            raise ValueError("max_tokens and max_completion_tokens disagree; send only one")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options is only allowed when stream is true")
        return self

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class ChatCompletionRequest(_GenerationParams):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)


class CompletionRequest(_GenerationParams):
    prompt: str | list[str]
    echo: bool | None = Field(False)

    @field_validator("prompt")
    @classmethod
    def _one_prompt(cls, value: str | list[str]) -> str:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("send one prompt per request (batch files handle many)")
            value = value[0]
        if not value:
            raise ValueError("prompt must not be empty")
        if _utf8_size(value, "prompt") > MAX_CONTENT_BYTES * 8:
            raise ValueError("prompt is too large")
        return value

    @field_validator("echo")
    @classmethod
    def _no_echo(cls, value: bool | None) -> bool | None:
        if value:
            raise ValueError("echo is not supported")
        return value


def valid_request_id(value: str) -> bool:
    """1-512 printable ASCII characters, not starting or ending with a space (HTTP strips those from headers)."""
    return (0 < len(value) <= MAX_ID_LENGTH and all(0x20 <= ord(c) <= 0x7E for c in value)
            and value[0] != " " and value[-1] != " ")


def resolve_sampling(request: _GenerationParams, defaults: dict) -> dict:
    """Effective sampling parameters: request values, else the model's generation_config.json, else greedy.

    A greedy generation config (do_sample false) holds no sampling settings (Hugging Face ignores its top_k / top_p
    then), so a request that asks for sampling (temperature > 0) samples from the full distribution unless it
    sets top_p / top_k itself.
    """
    if request.temperature is not None:
        temperature = request.temperature
    elif defaults.get("do_sample") is False:
        temperature = 0.0
    else:
        temperature = _default(defaults, "temperature", 0.0, 0.0, sys.float_info.max)
    if temperature < 1e-3:  # Edge-LLM treats this as greedy; make the parameters say so too
        return {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    sampling_defaults = {} if defaults.get("do_sample") is False else defaults
    top_p = (request.top_p if request.top_p is not None
             else _default(sampling_defaults, "top_p", 1.0, sys.float_info.min, 1.0))
    top_k = request.top_k if request.top_k is not None else _default(sampling_defaults, "top_k", 0, 0, MAX_TOP_K)
    return {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)}


def _default(defaults: dict, key: str, fallback: float, low: float, high: float) -> float:
    """A generation_config.json value; null, non-numbers, NaN and out-of-range values count as absent."""
    value = defaults.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not low <= value <= high:
        return fallback
    return value


# --------------------------------------------------------------------------- responses

def chat_response(response_id: str, model: str, text: str, finish_reason: str | None, usage: dict,
                  fingerprint: str, created: int) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                     "finish_reason": finish_reason}],
        "usage": usage,
    }


def completion_response(response_id: str, model: str, text: str, finish_reason: str | None, usage: dict,
                        fingerprint: str, created: int) -> dict:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }


def chat_chunk(response_id: str, model: str, created: int, fingerprint: str, delta: dict | None,
               finish_reason: str | None = None, usage: dict | None = None) -> dict:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]
    return {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "system_fingerprint": fingerprint, "choices": choices, "usage": usage}


def completion_chunk(response_id: str, model: str, created: int, fingerprint: str, text: str | None,
                     finish_reason: str | None = None, usage: dict | None = None) -> dict:
    choices = [] if text is None else [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}]
    return {"id": response_id, "object": "text_completion", "created": created, "model": model,
            "system_fingerprint": fingerprint, "choices": choices, "usage": usage}


def now_s() -> int:
    return int(time.time())


__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "chat_chunk",
    "chat_response",
    "completion_chunk",
    "completion_response",
    "resolve_sampling",
    "valid_request_id",
]
