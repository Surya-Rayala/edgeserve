"""OpenAI-style error bodies: {"error": {"message", "type", "param", "code"}}."""

from __future__ import annotations

import re

from fastapi.responses import JSONResponse
from pydantic import ValidationError

from edgeserve.worker import WorkerError

# pydantic names the union member that failed inside the location ("content.str", "prompt.list[str]").
_UNION_MEMBER = re.compile(r"(str|int|float|bool|list\[.*\]|dict\[.*\])")
_CLASS_NAME = re.compile(r" or instance of \w+")


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None, err_type: str = "invalid_request_error",
                 param: str | None = None, retry_after: int | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.err_type = err_type
        self.param = param
        self.retry_after = retry_after
        self.headers = dict(headers or {})

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.err_type, "param": self.param, "code": self.code}}

    def response(self, headers: dict[str, str] | None = None) -> JSONResponse:
        headers = self.headers | dict(headers or {})
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return JSONResponse(self.body(), status_code=self.status, headers=headers)


def from_validation(error: ValidationError) -> APIError:
    errors = error.errors(include_url=False, include_input=False)
    # An unsupported feature (tools, an image part, ...) is the most useful thing to report; otherwise the error
    # that got furthest into the input (for a union, the member the client was evidently trying to send).
    extra = next((e for e in errors if e.get("type") == "extra_forbidden"), None)
    first = extra or max(errors, key=lambda e: len(e.get("loc", ())))
    loc = list(first.get("loc", ()))
    head, key = (loc[:-1], loc[-1:]) if extra else (loc, [])  # an unknown key is the client's text, kept as is
    parts = [p for p in head if not (isinstance(p, str) and _UNION_MEMBER.fullmatch(p))] + key
    location = _printable(".".join(str(part) for part in parts))
    if extra:
        return APIError(400, f"unsupported parameter '{location}'", code="unsupported_parameter", param=location)
    message = _printable(_CLASS_NAME.sub("", first.get("msg", "invalid request").removeprefix("Value error, ")))
    return APIError(400, f"{location}: {message}" if location else message, code="invalid_value",
                    param=location or None)


def _printable(text: str) -> str:
    """Client-supplied keys can hold unpaired surrogates, which no UTF-8 response body can carry."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


# Worker error codes -> HTTP status (worker restarts and timeouts are temporary: clients may retry).
_WORKER_STATUS = {
    "input_too_long": (400, "context_length_exceeded", "invalid_request_error"),
    "bad_request": (400, "invalid_value", "invalid_request_error"),
    "queue_timeout": (503, "queue_timeout", "server_error"),
    "time_limit_exceeded": (504, "time_limit_exceeded", "server_error"),
    "worker_exited": (503, "worker_restarted", "server_error"),
    "worker_unavailable": (503, "server_unavailable", "server_error"),
    "cuda_error": (503, "worker_restarted", "server_error"),
}


def from_worker(error: WorkerError, retry_after: int | None = None) -> APIError:
    status, code, err_type = _WORKER_STATUS.get(error.code, (500, "engine_error", "server_error"))
    return APIError(status, str(error) or error.code, code=code, err_type=err_type,
                    retry_after=retry_after if status == 503 else None)


__all__ = ["APIError", "from_validation", "from_worker"]
