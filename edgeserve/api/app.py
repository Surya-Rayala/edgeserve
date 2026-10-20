"""HTTP API: OpenAI-compatible generation endpoints, health probes, Prometheus metrics, record lookup.

Every check that can fail (authentication, validation, client ID, readiness, memory, queue space,
prompt length) happens before the response starts, so overload and bad input are real HTTP
statuses even for streamed requests. A client that disconnects cancels its generation at the
next decode step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from edgeserve import __version__
from edgeserve import settings as limits
from edgeserve.api.errors import APIError, from_validation, from_worker
from edgeserve.api.protocol import (
    MAX_TOP_K,
    ChatCompletionRequest,
    CompletionRequest,
    chat_chunk,
    chat_response,
    completion_chunk,
    completion_response,
    now_s,
    resolve_sampling,
    valid_request_id,
)
from edgeserve.keys import DEFAULT_CLIENT
from edgeserve.metrics import response_metrics
from edgeserve.scheduler import AdmissionError, Job
from edgeserve.service import Service
from edgeserve.settings import is_loopback
from edgeserve.worker import WorkerError

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
ID_RULE = "request ids must be 1-512 printable ASCII characters, without a leading or trailing space"
log = logging.getLogger("edgeserve")


def create_app(service: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="edgeserve", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None, description="OpenAI-compatible server for TensorRT Edge-LLM engines")
    background: set[asyncio.Task] = set()

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return exc.response(_echo_id(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Routing errors (unknown path, wrong method) in OpenAI's error format too."""
        code, message = _ROUTING_ERRORS.get(exc.status_code, (None, str(exc.detail)))
        headers = dict(exc.headers or {})
        allow = [m.strip() for m in headers.get("Allow", "").split(",") if m.strip()]
        if "GET" in allow and "HEAD" not in allow:  # HEAD routes are registered apart from the GET ones
            headers["Allow"] = ", ".join(allow + ["HEAD"])
        error = APIError(exc.status_code, message, code=code, headers=headers)
        return error.response(_echo_id(request))

    def authorize(request: Request) -> str:
        """The client the request comes from: its API key's name, or its address on a server without keys (this
        machine counts as one client, "local")."""
        if not service.keys.required:
            host = request.client.host if request.client else ""
            client = "local" if not host or is_loopback(host) else host
        else:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            client = service.keys.client(token.strip()) if scheme.lower() == "bearer" else None
            if client is None:
                raise APIError(401, "missing or invalid API key (send 'Authorization: Bearer <key>')",
                               code="invalid_api_key", err_type="authentication_error",
                               headers={"WWW-Authenticate": "Bearer"})
        request.state.client = client
        return client

    def operator(request: Request) -> None:
        """/metrics and the API docs: with keys, only for a key, or for this machine itself."""
        if service.keys.required and not _loopback(request):
            authorize(request)

    # ------------------------------------------------------------------ probes and metadata

    @app.get("/health")
    async def health() -> JSONResponse:
        failed = service.state == "failed"
        return JSONResponse({"status": "failed" if failed else "ok", "state": service.state, "detail": service.detail},
                            status_code=503 if failed else 200)

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        if service.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": service.state, "detail": service.detail}, status_code=503,
                            headers={"Retry-After": "10"})

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        operator(request)
        return Response(service.prometheus.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_schema(request: Request) -> JSONResponse:
        operator(request)
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    async def docs(request: Request) -> HTMLResponse:
        operator(request)
        return get_swagger_ui_html(openapi_url="/openapi.json", title="edgeserve")

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        authorize(request)
        return JSONResponse({"object": "list", "data": [service.models_entry()]})

    @app.get("/v1/models/{model_id:path}")
    async def model(request: Request, model_id: str) -> JSONResponse:
        authorize(request)
        entry = service.models_entry()
        if model_id != entry["id"]:
            raise APIError(404, f"model {model_id!r} is not served here; this server serves {entry['id']!r}",
                           code="model_not_found", param="model")
        return JSONResponse(entry)

    @app.get("/v1/records/{custom_id:path}")
    async def record(request: Request, custom_id: str) -> JSONResponse:
        client = authorize(request)
        if service.output is None:
            raise APIError(404, "this server does not keep records (start it with --output-dir)", code="not_found")
        if not valid_request_id(custom_id):  # no record can have such an ID, and it may not be echoed in a header
            raise APIError(400, ID_RULE, code="invalid_request_id", param="custom_id")
        stored = await asyncio.to_thread(service.read_record, custom_id)
        if stored is None or not _may_read(service, client, stored):  # another client's record: as if absent
            raise APIError(404, f"no record for id {custom_id!r}", code="not_found")
        return JSONResponse(stored, headers={"X-Request-Id": custom_id})

    # HEAD as well (monitors and load balancers use it), kept out of the schema: one operation per path there
    for path, endpoint in (("/health", health), ("/health/ready", ready), ("/metrics", metrics),
                           ("/openapi.json", openapi_schema), ("/docs", docs),
                           ("/v1/models", models), ("/v1/models/{model_id:path}", model),
                           ("/v1/records/{custom_id:path}", record)):
        app.add_api_route(path, endpoint, methods=["HEAD"], include_in_schema=False)

    # ------------------------------------------------------------------ generation

    @app.post(CHAT)
    async def chat_completions(request: Request) -> Response:
        return await generate(request, ChatCompletionRequest, CHAT)

    @app.post(COMPLETIONS)
    async def completions(request: Request) -> Response:
        return await generate(request, CompletionRequest, COMPLETIONS)

    async def generate(request: Request, model_cls: type, endpoint: str) -> Response:
        header = request.headers.get("x-request-id")
        if header is not None and valid_request_id(header):
            request.state.custom_id = header  # echoed on errors raised before the ID is fully checked
        client = authorize(request)  # requests without a valid key are not saved: anyone could send them
        t_arrival_unix = time.time_ns()
        body = custom_id = None
        raw = b""
        try:
            try:
                service.rate_limiter.take(client)
            except AdmissionError as e:
                reject(_rejection_reason(e))
                raise APIError(e.status, str(e), code=e.code, err_type=e.err_type, retry_after=e.retry_after) from e
            raw = await _read_body(request)
            t_arrival, t_arrival_unix = time.monotonic_ns(), time.time_ns()  # upload time is network time
            body = _parse(raw, model_cls)
            custom_id = _client_id(request, body)
            request.state.custom_id = custom_id
            job = await admit(body, endpoint, custom_id, t_arrival, t_arrival_unix, client)
        except APIError as e:
            refused_id = custom_id or getattr(request.state, "custom_id", None)
            # a running request's record must stay, whatever this refusal was
            in_progress = e.code == "duplicate_request_id" or bool(
                refused_id and service.scheduler is not None and service.scheduler.is_inflight(refused_id))
            named = header is not None or getattr(body, "request_id", None) is not None  # chosen by the client
            service.record_rejection(
                endpoint=endpoint, client=client, custom_id=refused_id, request=_request_echo(body, len(raw)),
                metadata=getattr(body, "metadata", None), http_status=e.status, error=e.body()["error"],
                t_arrival_unix_ns=t_arrival_unix, in_progress=in_progress, named=named)
            raise
        headers = {"X-Request-Id": custom_id}
        if body.stream:
            return stream(job, body, endpoint, headers)
        return await complete(request, job, body, endpoint, headers)

    async def admit(body, endpoint: str, custom_id: str, t_arrival: int, t_arrival_unix: int, client: str) -> Job:
        scheduler, model, engine = service.scheduler, service.model, service.engine
        if body.model and body.model != service.model_name:
            raise APIError(404, f"model {body.model!r} is not served here; this server serves "
                                f"{service.model_name!r}", code="model_not_found", param="model")
        if body.stream and body.return_record:
            raise APIError(400, "return_record is only available for non-streaming requests", code="invalid_value",
                           param="return_record")
        if service.state == "draining":
            reject("shutting_down")
            raise APIError(503, "the server is shutting down; retry later", code="server_shutting_down",
                           err_type="server_error", retry_after=10)
        if not service.ready or scheduler is None or model is None or engine is None:
            reject("not_ready")
            raise APIError(503, f"the model is not ready yet ({service.detail})", code="server_unavailable",
                           err_type="server_error", retry_after=10)
        sample = service.telemetry.memory() if service.telemetry else None
        if sample is not None and sample.mem_available_bytes < limits.MEMORY_FLOOR_BYTES:
            reject("low_memory")
            raise APIError(503, "the system is low on memory; retry later", code="low_memory",
                           err_type="server_error", retry_after=5)

        if endpoint == CHAT:
            messages = [{"role": "system" if m.role == "developer" else m.role, "content": m.text()}
                        for m in body.messages]
            item: dict = {"messages": messages}
            request_echo: dict = {"messages": messages}
        else:
            item = {"prompt": body.prompt}
            request_echo = {"prompt": body.prompt}
        if body.stop:
            item["stop"] = body.stop

        if service.output is not None and service.keys.required:
            owner = await asyncio.to_thread(service.read_record, custom_id)
            if owner is not None and owner.get("client", DEFAULT_CLIENT) != client:
                reject("duplicate_request_id")
                raise APIError(409, f"request id {custom_id!r} belongs to another client; use another id",
                               code="duplicate_request_id", headers=_NO_RETRY)
        if scheduler.is_inflight(custom_id):
            reject("duplicate_request_id")
            raise APIError(409, f"a request with id {custom_id!r} is already in progress",
                           code="duplicate_request_id", headers=_NO_RETRY)
        try:
            scheduler.check_client(client)
            scheduler.check_capacity()
        except AdmissionError as e:
            reject(_rejection_reason(e))
            raise APIError(e.status, str(e), code=e.code, err_type=e.err_type, retry_after=e.retry_after) from e

        try:
            prompt_tokens = await service.count_tokens(item)
        except WorkerError as e:
            raise from_worker(e, retry_after=10) from e
        profile = engine.profile
        if prompt_tokens > profile.max_input_len:
            reject("context_length")
            raise APIError(400, f"This model's maximum input length is {profile.max_input_len} tokens. However, your "
                                f"prompt has {prompt_tokens} tokens. Please shorten the prompt.",
                           code="context_length_exceeded", param="messages" if endpoint == CHAT else "prompt")
        room = profile.kv_capacity - prompt_tokens - 1
        requested = body.requested_max_tokens
        if requested is not None and requested > room:
            reject("context_length")
            raise APIError(400, f"This model's maximum context length is {profile.kv_capacity - 1} tokens. However, "
                                f"you requested {prompt_tokens + requested} tokens ({prompt_tokens} in the prompt, "
                                f"{requested} in the completion). Please reduce the prompt or max_tokens.",
                           code="context_length_exceeded", param="max_tokens")
        max_new = requested or min(profile.max_new_tokens, room)
        sampling = resolve_sampling(body, model.generation_defaults) | {"max_new_tokens": max_new}
        if sampling["top_k"] >= model.output_vocab_size:
            sampling["top_k"] = 0  # every token: the same as no limit
        if sampling["top_k"] > MAX_TOP_K:
            raise APIError(400, f"top_k must be at most {MAX_TOP_K}, or 0 (or at least the vocabulary size, "
                                f"{model.output_vocab_size}) for no limit", code="invalid_value", param="top_k")
        seed = body.seed
        if seed is None and sampling["temperature"] > 0:
            # Without a seed Edge-LLM draws the same random number at every step (so sampling would match greedy
            # decoding); a fresh seed per request makes it random, and the record keeps it for reproduction.
            seed = secrets.randbits(63)
        if seed is not None:
            item["seed"] = seed
        params_echo = {"max_tokens": max_new, "temperature": sampling["temperature"], "top_p": sampling["top_p"],
                       "top_k": sampling["top_k"], "seed": seed, "stop": body.stop, "stream": body.stream}
        job = Job(custom_id=custom_id, attempt_id=service.new_attempt_id(), endpoint=endpoint, item=item,
                  sampling=sampling, stream=body.stream, prompt_tokens=prompt_tokens, t_arrival_ns=t_arrival,
                  t_arrival_unix_ns=t_arrival_unix, request_echo=request_echo, params_echo=params_echo,
                  metadata=body.metadata, client=client)
        try:
            scheduler.submit(job)
        except AdmissionError as e:  # e.g. the same ID admitted meanwhile by a concurrent request
            reject(_rejection_reason(e))
            raise APIError(e.status, str(e), code=e.code, err_type=e.err_type, retry_after=e.retry_after,
                           headers=_NO_RETRY if e.status == 409 else None) from e
        return job

    def reject(reason: str) -> None:
        service.prometheus.rejected(reason)

    def response_id(endpoint: str, custom_id: str) -> str:
        return ("chatcmpl-" if endpoint == CHAT else "cmpl-") + custom_id

    def final_body(job: Job, endpoint: str, result: dict, record: dict, created: int) -> dict:
        assert service.ctx is not None
        build = chat_response if endpoint == CHAT else completion_response
        body = build(response_id(endpoint, job.custom_id), service.ctx.served_name, result.get("text", ""),
                     result.get("finish_reason"), dict(record["usage"]), service.ctx.system_fingerprint, created)
        body["request_id"] = job.custom_id
        body["metrics"] = response_metrics(record)
        return body

    def cancelled_by_server(job: Job, event: dict | None = None, result: dict | None = None) -> APIError:
        """Record a request the server stopped while its client was still there (shutdown): safe to retry."""
        error = APIError(503, "the server is shutting down; retry later", code="server_shutting_down",
                         err_type="server_error", retry_after=10)
        service.commit(job, service.make_record(job, status="cancelled", http_status=503, done=event, result=result,
                                                error=error.body()["error"]))
        return error

    def timed_out(job: Job, event: dict | None = None, result: dict | None = None) -> APIError:
        """Record a request stopped by --request-timeout (with what it had generated)."""
        error = APIError(504, f"the request did not finish within {service.settings.request_timeout} s of its "
                              "arrival (the server's --request-timeout)", code="time_limit_exceeded",
                         err_type="server_error", headers=_NO_RETRY)  # it would only run into the limit again
        service.commit(job, service.make_record(job, status="failed", http_status=504, done=event, result=result,
                                                error=error.body()["error"]))
        return error

    def abandon(job: Job) -> None:
        """Nobody will read the job's final event (client gone, response failed, request cancelled): stop the
        job, then record it as cancelled and release its ID once the worker has let go of it."""
        assert service.scheduler is not None
        service.scheduler.cancel(job)
        task = asyncio.ensure_future(_record_cancelled(job))
        background.add(task)
        task.add_done_callback(background.discard)

    async def complete(request: Request, job: Job, body, endpoint: str, headers: dict) -> Response:
        scheduler = service.scheduler
        assert scheduler is not None
        try:
            (kind, *payload), disconnected = await _wait_final(request, job, scheduler)
        except BaseException:  # e.g. cancelled by the server's shutdown timeout
            abandon(job)
            raise
        try:
            if kind == "done":
                event, result = payload
                finish = result.get("finish_reason")
                if disconnected:
                    record = service.make_record(job, status="cancelled", http_status=499, done=event, result=result)
                    service.commit(job, record)
                    return Response(status_code=499)
                if finish == "cancelled":
                    raise timed_out(job, event, result) if job.timed_out else cancelled_by_server(job, event, result)
                if finish == "error":
                    error = APIError(500, "generation failed in the engine", code="engine_error", err_type="server_error")
                    record = service.make_record(job, status="failed", http_status=500, done=event, result=result,
                                                 error=error.body()["error"])
                    service.commit(job, record)
                    raise error
                record = service.make_record(job, status="completed", http_status=200, done=event, result=result)
                response = final_body(job, endpoint, result, record, now_s())
                service.commit(job, record, response_body=response)
                if body.return_record:
                    response = response | {"record": record}
                return JSONResponse(response, headers=headers)
            if kind == "cancelled":
                if not disconnected:
                    raise cancelled_by_server(job)
                service.commit(job, service.make_record(job, status="cancelled", http_status=499))
                return Response(status_code=499)
            error = from_worker(payload[0], retry_after=scheduler.retry_after_s())
            if job.timed_out and job.t_dispatch_ns is not None:  # stopped while running in a batch
                error.headers.update(_NO_RETRY)  # it would only run into the limit again
            service.commit(job, service.make_record(job, status="failed", http_status=error.status,
                                                    error=error.body()["error"]))
            raise error
        finally:
            scheduler.release(job)

    def stream(job: Job, body, endpoint: str, headers: dict) -> Response:
        scheduler, ctx = service.scheduler, service.ctx
        assert scheduler is not None and ctx is not None
        created = now_s()
        rid = response_id(endpoint, job.custom_id)
        name, fp = ctx.served_name, ctx.system_fingerprint
        finished = False  # the stream took the job's final event (and recorded it)

        def delta(text: str) -> str:
            return _sse(chat_chunk(rid, name, created, fp, {"content": text}) if endpoint == CHAT
                        else completion_chunk(rid, name, created, fp, text))

        async def events() -> AsyncGenerator[str, None]:
            nonlocal finished
            sent = ""  # text streamed so far
            # Chat streams open with the assistant role, sent when the first token is (as OpenAI and vLLM do), so
            # clients that time the first chunk measure the real time to first token.
            opened = endpoint != CHAT
            role = _sse(chat_chunk(rid, name, created, fp, {"role": "assistant", "content": ""}))
            while True:
                try:
                    kind, *payload = await asyncio.wait_for(job.events.get(), timeout=limits.SSE_KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # keeps proxies from closing a request that is still queued
                    continue
                if kind == "text":
                    text = payload[0].get("text", "")
                    if text:
                        if not opened:
                            opened = True
                            yield role
                        sent += text
                        yield delta(text)
                    continue
                finished = True
                if kind == "done":
                    event, result = payload
                    finish = result.get("finish_reason")
                    if finish == "cancelled":  # the client is still reading, so the server stopped it
                        stopped = timed_out(job, event, result) if job.timed_out else cancelled_by_server(job, event,
                                                                                                            result)
                        yield _sse(stopped.body())
                        yield "data: [DONE]\n\n"
                        return
                    if finish == "error":
                        error = APIError(500, "generation failed in the engine", code="engine_error",
                                         err_type="server_error")
                        service.commit(job, service.make_record(job, status="failed", http_status=500, done=event,
                                                                result=result, error=error.body()["error"]))
                        yield _sse(error.body())
                        yield "data: [DONE]\n\n"
                        return
                    record = service.make_record(job, status="completed", http_status=200, done=event, result=result)
                    final = final_body(job, endpoint, result, record, created)
                    service.commit(job, record, response_body=final)
                    if not opened:
                        yield role
                    # Edge-LLM can hold back the end of a stream (its stop-string look-ahead, a character cut by
                    # max_tokens); the final text is authoritative, so send whatever the stream has not shown.
                    text = result.get("text", "")
                    if text.startswith(sent):
                        if len(text) > len(sent):
                            yield delta(text[len(sent):])
                    else:
                        log.warning("request %s: the streamed text differs from the final text", job.custom_id)
                    last = (chat_chunk(rid, name, created, fp, {}, finish_reason=finish) if endpoint == CHAT
                            else completion_chunk(rid, name, created, fp, "", finish_reason=finish))
                    yield _sse(last)
                    if body.include_usage:
                        usage_chunk = (chat_chunk(rid, name, created, fp, None, usage=final["usage"]) if endpoint == CHAT
                                       else completion_chunk(rid, name, created, fp, None, usage=final["usage"]))
                        usage_chunk["metrics"] = final["metrics"]
                        yield _sse(usage_chunk)
                    yield "data: [DONE]\n\n"
                    return
                if kind == "cancelled":
                    yield _sse(cancelled_by_server(job).body())
                    yield "data: [DONE]\n\n"
                    return
                error = from_worker(payload[0], retry_after=scheduler.retry_after_s())
                service.commit(job, service.make_record(job, status="failed", http_status=error.status,
                                                        error=error.body()["error"]))
                yield _sse(error.body())
                yield "data: [DONE]\n\n"
                return

        def close() -> None:
            if finished:
                scheduler.release(job)
            else:  # the client went away mid-stream, or the response could not even start
                abandon(job)

        return _EventStream(events(), close, headers | {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _record_cancelled(job: Job) -> None:
        """After the client went away: wait for the worker to stop the request, then record it."""
        scheduler = service.scheduler
        try:
            while True:
                kind, *payload = await asyncio.wait_for(job.events.get(), timeout=120)
                if kind == "text":
                    continue
                if kind == "done":
                    event, result = payload
                    service.commit(job, service.make_record(job, status="cancelled", http_status=499, done=event,
                                                            result=result))
                else:
                    service.commit(job, service.make_record(job, status="cancelled", http_status=499))
                return
        except asyncio.TimeoutError:
            service.commit(job, service.make_record(job, status="cancelled", http_status=499))
        finally:
            if scheduler:
                scheduler.release(job)

    app.openapi = lambda: _openapi(app)
    return app


def _openapi(app: FastAPI) -> dict:
    """FastAPI's schema plus what it cannot infer here (the handlers read their bodies themselves and check the key
    by hand): the request bodies and the bearer key, so /docs can send real requests."""
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        components = schema.setdefault("components", {})
        components["securitySchemes"] = {"apiKey": {"type": "http", "scheme": "bearer"}}
        models = components.setdefault("schemas", {})
        for path, model_cls in ((CHAT, ChatCompletionRequest), (COMPLETIONS, CompletionRequest)):
            body = model_cls.model_json_schema(ref_template="#/components/schemas/{model}")
            models.update(body.pop("$defs", {}))
            models[model_cls.__name__] = body
            schema["paths"][path]["post"]["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{model_cls.__name__}"}}},
            }
        for path, operations in schema["paths"].items():
            for operation in operations.values():
                operation.get("responses", {}).pop("422", None)  # validation errors are 400s here
                if path.startswith("/v1/"):
                    operation["security"] = [{"apiKey": []}]
        app.openapi_schema = schema
    return app.openapi_schema


class _EventStream(StreamingResponse):
    """Server-Sent Events whose ``on_close`` runs exactly once, however the response ends: completed, client
    gone, request task cancelled, or failed before the first byte (when the generator never started, so it
    could not have cleaned up after itself)."""

    def __init__(self, events: AsyncGenerator[str, None], on_close: Callable[[], None],
                 headers: dict[str, str]) -> None:
        super().__init__(events, media_type="text/event-stream", headers=headers)
        self._events = events
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self._events.aclose()
            finally:
                self._on_close()


async def _read_body(request: Request) -> bytes:
    """The request body, refused as soon as it exceeds the limit (Content-Length may be absent or chunked)."""
    too_large = APIError(413, f"request body larger than {limits.MAX_BODY_BYTES // 2**20} MiB", code="body_too_large")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limits.MAX_BODY_BYTES:
        raise too_large

    async def collect() -> bytes:
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > limits.MAX_BODY_BYTES:
                raise too_large
        return bytes(raw)

    try:
        return await asyncio.wait_for(collect(), timeout=limits.BODY_TIMEOUT_S)
    except ClientDisconnect:
        raise APIError(400, "the client disconnected before sending the whole body", code="invalid_json") from None
    except asyncio.TimeoutError:
        raise APIError(408, f"the request body did not arrive within {limits.BODY_TIMEOUT_S:.0f} s",
                       code="request_timeout") from None


def _parse(raw: bytes, model_cls: type):
    try:
        payload = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:  # RecursionError: nesting deeper than the parser's limit
        raise APIError(400, "the request body is not valid JSON", code="invalid_json") from e
    if not isinstance(payload, dict):
        raise APIError(400, "the request body must be a JSON object", code="invalid_json")
    try:
        return model_cls.model_validate(payload)
    except ValidationError as e:
        raise from_validation(e) from e


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _client_id(request: Request, body) -> str:
    if len(request.headers.getlist("x-request-id")) > 1:
        raise APIError(400, "send one X-Request-Id header", code="invalid_request_id", param="request_id")
    header = request.headers.get("x-request-id")
    if header is not None and body.request_id is not None and header != body.request_id:
        raise APIError(400, "the X-Request-Id header and the request_id field disagree", code="invalid_request_id",
                       param="request_id")
    custom_id = header if header is not None else body.request_id
    if custom_id is None:
        return "req_" + secrets.token_hex(12)
    if not valid_request_id(custom_id):
        raise APIError(400, ID_RULE, code="invalid_request_id", param="request_id")
    return custom_id


# The OpenAI SDK retries a 409 unless told not to; a hidden retry would run the request a second time.
_NO_RETRY = {"x-should-retry": "false"}
_ROUTING_ERRORS = {404: ("not_found", "no such endpoint"),
                   405: ("method_not_allowed", "method not allowed for this endpoint")}


def _echo_id(request: Request) -> dict[str, str]:
    """X-Request-Id for a response: the resolved ID, else the client's header when it is a valid ID."""
    custom_id = getattr(request.state, "custom_id", None)
    if custom_id is None:
        header = request.headers.get("x-request-id")
        custom_id = header if header is not None and valid_request_id(header) else None
    return {"X-Request-Id": custom_id} if custom_id else {}


def _rejection_reason(e: AdmissionError) -> str:
    return {"server_is_overloaded": "queue_full", "server_shutting_down": "shutting_down",
            "too_many_requests": "client_limit", "rate_limit_exceeded": "rate_limited"}.get(e.code, e.code)


_FORWARDED = ("forwarded", "x-forwarded-for", "x-real-ip")
_MAX_SAVED_TEXT = 64 * 1024  # a refused request's own text is saved up to this body size


def _loopback(request: Request) -> bool:
    """Whether the request comes from this machine itself (also through an SSH tunnel to 127.0.0.1): a loopback
    peer, a loopback Host (not a DNS-rebinding page) and no proxy in between (a proxy on this machine relays
    other machines' requests)."""
    if request.client is None or any(name in request.headers for name in _FORWARDED):
        return False
    return is_loopback(request.client.host) and is_loopback((request.url.hostname or "").strip("[]"))


def _may_read(service: Service, client: str, record: dict) -> bool:
    """With keys, a client reads the records of its own requests; the default key (the operator's) reads all."""
    return not service.keys.required or client == DEFAULT_CLIENT or record.get("client", DEFAULT_CLIENT) == client


def _request_echo(body, size: int) -> dict | None:
    """What a refused request asked for, for its record (None when its body was invalid or not read); the text of
    a large one is left out."""
    if body is None:
        return None
    big = size > _MAX_SAVED_TEXT
    echo = body.model_dump(mode="json", exclude_none=True,
                           exclude={"request_id", "metadata"} | ({"messages", "prompt"} if big else set()))
    if big:
        echo["omitted"] = f"the text of a {size}-byte request"
    return echo


async def _wait_final(request: Request, job: Job, scheduler) -> tuple[tuple, bool]:
    """Wait for the job's final event; if the client disconnects first, cancel and wait for the worker to stop.

    When this raises (the request was cancelled), nothing reads the job's events any more and a final event
    that was already taken is back in the queue, so whoever takes the job over receives it.
    """

    async def final() -> tuple:
        while (event := await job.events.get())[0] == "text":
            pass
        return event

    async def disconnected() -> None:
        # The body has been read, so receive() now blocks until the client goes away (ASGI spec 2.3; Starlette's
        # StreamingResponse listens the same way).
        while (await request.receive())["type"] != "http.disconnect":
            await asyncio.sleep(0.5)  # not expected after the body; never spin

    final_task = asyncio.ensure_future(final())
    watch_task = asyncio.ensure_future(disconnected())
    try:
        done, _ = await asyncio.wait({final_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        if final_task in done:
            return final_task.result(), False
        scheduler.cancel(job)
        return await final_task, True
    except BaseException:
        if final_task.done() and not final_task.cancelled():
            job.events.put_nowait(final_task.result())
        raise
    finally:
        watch_task.cancel()
        final_task.cancel()  # a cancelled Queue.get() leaves its event in the queue


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n\n"


__all__ = ["create_app"]
