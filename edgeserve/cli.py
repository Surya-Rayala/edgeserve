"""Command line: ``edgeserve setup | export | serve | build | info | batch | check | service``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from edgeserve import __version__, config
from edgeserve import settings as cfg

# Python's own default TLS 1.2 ciphers, stated explicitly: older uvicorn releases replace them with "TLSv1",
# which enables unencrypted (NULL) and non-forward-secret ciphers. TLS 1.3 suites are not affected.
_TLS_CIPHERS = "@SECLEVEL=2:ECDH+AESGCM:ECDH+CHACHA20:ECDH+AES:DHE+AES:!aNULL:!eNULL:!aDSS:!SHA1:!AESCCM"
# Requests are drained before uvicorn's own graceful shutdown (see _server), which then only lets the
# final responses go out.
_RESPONSE_GRACE_S = 5


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": round(record.created, 3), "level": record.levelname.lower(), "logger": record.name}
        message = record.getMessage()
        if record.name == "edgeserve.requests" and message.startswith("{"):
            entry.update(json.loads(message))  # request lines are already structured
        else:
            entry["message"] = message
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class _ProbeFilter(logging.Filter):
    """Keep health checks and Prometheus scrapes (every few seconds) out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(f'"{method} /{path}' in message for method in ("GET", "HEAD") for path in ("health", "metrics"))


def setup_logging() -> None:
    """JSON lines under systemd / pipes, readable lines in a terminal."""
    handler = logging.StreamHandler(sys.stderr)
    if sys.stderr.isatty():
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    else:
        handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").addFilter(_ProbeFilter())


def _version() -> str:
    from edgeserve.nativebuild import EDGELLM_VERSION, edgellm_commit

    try:
        commit = edgellm_commit()[:12]
    except OSError:
        commit = "unknown"
    return f"edgeserve {__version__} (TensorRT Edge-LLM {EDGELLM_VERSION}, commit {commit})"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgeserve", description="OpenAI-compatible server for TensorRT Edge-LLM models on Jetson and NVIDIA "
        "GPUs. Settings: flags, else EDGESERVE_* environment variables, else ~/.config/edgeserve/edgeserve.env.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    setup = sub.add_parser("setup", allow_abbrev=False,
                           help="build the GPU runtime for this machine and write the settings (run once, and "
                                "again after upgrading edgeserve, CUDA or TensorRT)")
    setup.add_argument("--network", action="store_true",
                       help="serve other machines too: listen on all interfaces, require an API key, use HTTPS "
                            "(creates the key and the certificates)")
    setup.add_argument("--model", type=Path, metavar="DIR",
                       help="model folder to serve by default (stored as EDGESERVE_MODEL)")

    export = sub.add_parser("export", allow_abbrev=False,
                            help="turn a Hugging Face checkpoint (an ID, or your own folder: fine-tuned, pruned or "
                                 "INT4) into a model folder to serve; options for Edge-LLM's exporter go after --",
                            epilog="Options after -- go to tensorrt-edgellm-export unchanged, e.g. "
                                   "edgeserve export CHECKPOINT MODEL_DIR -- --reduced-vocab-dir DIR")
    export.add_argument("checkpoint", metavar="CHECKPOINT",
                        help="a Hugging Face model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct-AWQ) or a local checkpoint "
                             "folder (config.json, *.safetensors, tokenizer files)")
    export.add_argument("model_dir", metavar="MODEL_DIR", type=Path,
                        help="the model folder to create (it must not exist yet, or be empty)")

    serve = sub.add_parser("serve", help="run the server (converts the model to a TensorRT engine on first start)",
                           allow_abbrev=False)
    cfg.add_engine_arguments(serve)
    cfg.add_server_arguments(serve)
    build = sub.add_parser("build", help="convert a model to a TensorRT engine for this GPU (or check the cached "
                                         "one), then exit", allow_abbrev=False)
    cfg.add_engine_arguments(build)
    info = sub.add_parser("info", help="show the settings, the GPU runtime, the hardware and (with a model) the "
                                       "model, its memory estimate and its engine", allow_abbrev=False)
    cfg.add_engine_arguments(info)
    sub.add_parser("batch", help="run a JSONL file of requests against a server (see 'edgeserve batch -h')",
                   add_help=False)
    check = sub.add_parser("check", help="check a running server end to end", allow_abbrev=False)
    check.add_argument("--server", metavar="URL",
                       help="server base URL, e.g. https://192.168.1.20:8000 (default: this machine's server, from "
                            "its settings)")
    check.add_argument("--ca-cert", metavar="FILE",
                       help="CA certificate to trust for HTTPS (the server's edgeserve-ca.crt; default: this "
                            "machine's, for its own server)")
    service = sub.add_parser("service", help="start the server at boot (systemd); run as your user, it uses sudo "
                                             "for the system steps", allow_abbrev=False)
    service.add_argument("action", choices=["install", "remove"],
                         help="install: write the unit, enable and (re)start it; remove: stop and delete it")
    service.add_argument("--print", action="store_true",
                         help="with install: only show the unit file and the commands, change nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["batch"]:
        from edgeserve.batch import main as batch_main

        return batch_main(argv[1:])
    extra: list[str] = []
    if argv[:1] == ["export"] and "--" in argv:  # the rest goes to tensorrt-edgellm-export unchanged
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = _parser().parse_args(argv)
    try:
        config.read()  # report a broken settings file once, clearly
    except config.ConfigError as e:
        print(f"edgeserve: {e}", file=sys.stderr)
        return 2
    if args.command == "setup":
        from edgeserve import setupcmd

        return setupcmd.run(network=args.network, model=args.model)
    if args.command == "export":
        from edgeserve import exporter

        return exporter.run(args.checkpoint, args.model_dir, extra)
    if args.command == "check":
        from edgeserve import check

        return check.run(args.server, args.ca_cert)
    if args.command == "service":
        from edgeserve import systemd

        return systemd.remove() if args.action == "remove" else systemd.install(print_only=args.print)
    setup_logging()
    if args.command == "info":
        return _info(args)
    try:
        settings = cfg.from_args(args)
    except ValueError as e:
        print(f"edgeserve: {e}", file=sys.stderr)
        return 2
    problem = _native_problem(settings)
    if problem:
        print(f"edgeserve: {problem}", file=sys.stderr)
        return 2
    if args.command == "serve":
        return _serve(settings)
    return _build(settings)


def _native_problem(settings: cfg.Settings) -> str | None:
    """Why the installed GPU runtime cannot be used (None: fine, or a runtime given with EDGESERVE_NATIVE_DIR)."""
    if config.value("EDGESERVE_NATIVE_DIR"):
        return None
    from edgeserve import nativebuild

    state = nativebuild.status(settings.native_dir)
    return None if state.ok else f"{state.reason}: run edgeserve setup"


def _serve(settings: cfg.Settings) -> int:
    import uvicorn

    from edgeserve.api.app import create_app
    from edgeserve.service import Service

    log = logging.getLogger("edgeserve")
    if not cfg.is_loopback(settings.host) and not settings.tls:
        log.warning("serving on %s without TLS: prompts and outputs cross the network unencrypted "
                    "(edgeserve setup --network creates a certificate; see README, 'Access from other machines')",
                    settings.host)
    service = Service(settings)
    config = uvicorn.Config(
        create_app(service),
        host=settings.host,
        port=settings.port,
        ssl_certfile=str(settings.tls_cert) if settings.tls_cert else None,
        ssl_keyfile=str(settings.tls_key) if settings.tls_key else None,
        ssl_ciphers=_TLS_CIPHERS,
        log_config=None,
        timeout_graceful_shutdown=_RESPONSE_GRACE_S,
        timeout_keep_alive=30,
        limit_concurrency=cfg.MAX_CONNECTIONS,
        http=_head_timeout_protocol(),
    )
    try:
        config.load()  # reads the TLS certificate and key
    except OSError as e:
        if not settings.tls:
            raise
        print(f"edgeserve: cannot use the TLS certificate {settings.tls_cert} and key {settings.tls_key}: {e} "
              "(EDGESERVE_TLS_CERT / EDGESERVE_TLS_KEY; edgeserve setup --network creates them)", file=sys.stderr)
        return 2
    scheme = "https" if settings.tls else "http"
    log.info("edgeserve %s listening on %s://%s:%d (model %s)", __version__, scheme, settings.host, settings.port,
             settings.model)
    try:
        _server(config, service).run()
    except KeyboardInterrupt:  # uvicorn re-raises Ctrl-C after its graceful shutdown
        pass
    return 1 if service.state == "failed" else 0


def _head_timeout_protocol():
    """uvicorn's HTTP/1.1 protocol with the deadlines it lacks, so no client can hold a connection forever.

    * A connection must send a complete request head within HEAD_TIMEOUT_S, after connecting and after each
      response (uvicorn times out idle keep-alive connections only after a response, and partial data resets that).
    * A client that stops reading for WRITE_STALL_S while a response is being sent is dropped.
    * Idle connections are aborted rather than closed, and every close is aborted after CLOSE_GRACE_S: over TLS,
      asyncio's close() first sends what is still queued, then waits for the peer's close_notify without a timeout
      (Python 3.10), so a peer that never answers would keep its connection, and its place under
      limit_concurrency, forever.
    * One client address may hold at most MAX_CONNECTIONS_PER_CLIENT connections, so a single host cannot take
      all MAX_CONNECTIONS (which uvicorn counts server-wide).
    """
    from uvicorn.protocols.http.h11_impl import H11Protocol

    per_client: dict[str, int] = {}

    class _Transport:
        """The connection's transport, with every close() bounded by an abort()."""

        def __init__(self, transport, loop) -> None:
            self._transport, self._loop, self.abort_timer = transport, loop, None

        def close(self) -> None:
            self._transport.close()
            if self.abort_timer is None:
                self.abort_timer = self._loop.call_later(cfg.CLOSE_GRACE_S, self._transport.abort)

        def __getattr__(self, name):
            return getattr(self._transport, name)

    class DeadlineH11Protocol(H11Protocol):
        _head_timer = None
        _write_timer = None
        _client = None

        def connection_made(self, transport) -> None:
            peer = transport.get_extra_info("peername")
            client = peer[0] if isinstance(peer, tuple) and peer else "?"
            if per_client.get(client, 0) >= cfg.MAX_CONNECTIONS_PER_CLIENT:
                transport.abort()  # never registered with uvicorn: connection_lost below ignores it
                return
            per_client[client] = per_client.get(client, 0) + 1
            self._client = client
            super().connection_made(_Transport(transport, self.loop))
            self._arm_head_timer()

        def connection_lost(self, exc) -> None:
            if self._client is None:
                return
            per_client[self._client] -= 1
            if not per_client[self._client]:
                del per_client[self._client]
            self._client = None
            try:
                super().connection_lost(exc)  # may close() the transport once more, which arms its abort timer
            finally:
                for timer in (self._head_timer, self._write_timer, self.transport.abort_timer):
                    if timer is not None:
                        timer.cancel()

        def handle_events(self) -> None:
            cycle = self.cycle
            super().handle_events()
            if self.cycle is not cycle:  # a complete request head arrived
                self._cancel_head_timer()

        def on_response_complete(self) -> None:
            self._arm_head_timer()  # first: the call below may already start a pipelined request
            super().on_response_complete()

        def pause_writing(self) -> None:
            super().pause_writing()
            if self._write_timer is None:
                self._write_timer = self.loop.call_later(cfg.WRITE_STALL_S, self._abort)

        def resume_writing(self) -> None:
            super().resume_writing()
            if self._write_timer is not None:
                self._write_timer.cancel()
                self._write_timer = None

        def timeout_keep_alive_handler(self) -> None:
            if self._idle():
                self._abort()  # uvicorn would close(), which over TLS waits for the peer

        def shutdown(self) -> None:
            if self._idle():
                self._abort()  # (uvicorn waits for connections to go away before it exits)
            else:
                super().shutdown()

        def _idle(self) -> bool:
            """No request in progress and no response still being sent."""
            return ((self.cycle is None or self.cycle.response_complete) and not self.transport.is_closing()
                    and not self.flow.write_paused)

        def _arm_head_timer(self) -> None:
            self._cancel_head_timer()
            self._head_timer = self.loop.call_later(cfg.HEAD_TIMEOUT_S, self._head_timeout)

        def _cancel_head_timer(self) -> None:
            if self._head_timer is not None:
                self._head_timer.cancel()
                self._head_timer = None

        def _head_timeout(self) -> None:
            self._head_timer = None
            if self._idle():
                self._abort()
            elif self.flow.write_paused and not self.transport.is_closing():
                self._arm_head_timer()  # the last response is still being sent: look again later

        def _abort(self) -> None:
            self._write_timer = None
            self.transport.abort()

    return DeadlineH11Protocol


def _server(config, service):
    """A uvicorn server whose shutdown first drains the service, while the request handlers are still there to
    answer and record what is cancelled; uvicorn alone would wait for (or cancel) them before the lifespan
    shutdown even starts the drain."""
    import uvicorn

    log = logging.getLogger("edgeserve")

    class Server(uvicorn.Server):
        async def shutdown(self, sockets=None) -> None:
            for listener in getattr(self, "servers", []):
                listener.close()  # stop accepting connections, as uvicorn's shutdown does first
            drain = asyncio.ensure_future(service.drain())
            while not drain.done() and not self.force_exit:  # a second Ctrl-C skips the wait
                await asyncio.wait({drain}, timeout=0.2)
            if not drain.done():
                drain.cancel()
            elif not drain.cancelled() and drain.exception() is not None:
                log.error("draining failed: %s", drain.exception())
            await super().shutdown(sockets)

    return Server(config)


def _build(settings: cfg.Settings) -> int:
    from edgeserve.service import Service

    service = Service(settings)

    async def run() -> int:
        try:
            engine = await service.prepare_engine()
        except Exception as e:  # noqa: BLE001 - report every failure as a message, not a traceback
            print(f"edgeserve build: {e}", file=sys.stderr)
            return 1
        finally:
            if service.telemetry:
                service.telemetry.stop()
        build = engine.manifest.get("build", {})
        print(json.dumps({"engine": str(engine.path), "built_now": engine.built_now,
                          "fingerprint": engine.fingerprint, "profile": engine.manifest.get("profile"),
                          "build": build}, indent=2))
        return 0

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:  # the build was stopped: llm_build killed, its staging folder removed
        print("edgeserve build: interrupted", file=sys.stderr)
        return 130


def _info(args: argparse.Namespace) -> int:
    from edgeserve import nativebuild
    from edgeserve.engine import BuildProfile, EngineCache, NativeBuild, engine_key
    from edgeserve.hardware import probe, probe_gpu
    from edgeserve.model import estimate_memory, fingerprint, load_model
    from edgeserve.service import available_memory, check_compatibility, check_runtime, default_max_new_tokens

    gib = 2**30
    report: dict = {"edgeserve": _version(), "settings_file": str(config.config_file())}
    try:  # the server settings too (info has no flags for them): what serve would use
        resolved = cfg.resolve(argparse.Namespace(**{"host": None, "port": None, "output_dir": None, **vars(args)}))
    except ValueError as e:  # a setting that is not a number: a usage error, as for serve and build
        print(f"edgeserve: {e}", file=sys.stderr)
        return 2
    if resolved["cache_dir"][0] is None:
        resolved["cache_dir"] = (config.default_cache_dir(), "default")
    settings_report = {attr: {"value": None if v is None else str(v), "from": origin}
                       for attr, (v, origin) in resolved.items()}
    for name in ("EDGESERVE_API_KEY", "EDGESERVE_TLS_CERT", "EDGESERVE_TLS_KEY", "EDGESERVE_NATIVE_DIR"):
        value = config.value(name)
        shown = ("set" if value else None) if name == "EDGESERVE_API_KEY" else value
        if name == "EDGESERVE_NATIVE_DIR" and not value:
            shown = str(config.native_current())
        settings_report[name.removeprefix("EDGESERVE_").lower()] = {"value": shown,
                                                                    "from": config.source(name) or "default"}
    report["settings"] = settings_report
    try:
        if config.value("EDGESERVE_NATIVE_DIR"):
            native = NativeBuild.locate(Path(config.value("EDGESERVE_NATIVE_DIR") or ""))
            report["gpu_runtime"] = {"path": str(native.root), "given_with": "EDGESERVE_NATIVE_DIR"}
        else:
            state = nativebuild.status()
            report["gpu_runtime"] = {"ok": state.ok, "problem": state.reason or None,
                                     "path": str(state.root) if state.root else None}
            if not state.ok:
                report["gpu_runtime"]["fix"] = "run edgeserve setup"
                print(json.dumps(report, indent=2))
                return 1
            native = NativeBuild.locate(config.native_current())
        report["gpu_runtime"].update({k: native.info.get(k) for k in ("edgellm_commit", "cuda", "tensorrt", "sm",
                                                                      "built_at", "edgeserve_version", "self_test")})
        gpu = probe_gpu(native.worker, native.env())
        try:
            check_runtime(native, gpu)
        except Exception as e:  # noqa: BLE001
            report["gpu_runtime"]["problem"] = str(e)
        platform = probe(gpu)
        report["platform"] = platform.as_dict()
        report["memory_available_gib"] = round(available_memory(platform, gpu) / gib, 2)
        if resolved["model"][0] is None:
            report["model"] = None
            report["note"] = "pass --model <folder> (or set EDGESERVE_MODEL) for model, memory and engine facts"
            print(json.dumps(report, indent=2))
            return 0
        settings = cfg.from_args(args)
        model = load_model(settings.model)
        report["model"] = model.as_dict()
        profile = BuildProfile(settings.max_batch_size, settings.max_input_len, default_max_new_tokens(settings, model))
        estimate = estimate_memory(model, profile.max_batch_size, profile.kv_capacity, profile.max_input_len)
        report["memory_estimate_gib"] = {k: round(v / gib, 2) for k, v in estimate.items()}
        report["engine_profile"] = {"max_batch_size": profile.max_batch_size, "max_input_len": profile.max_input_len,
                                    "max_new_tokens": profile.max_new_tokens, "kv_capacity": profile.kv_capacity}
        try:
            check_compatibility(model, gpu)
            report["compatible"] = True
        except Exception as e:  # noqa: BLE001
            report["compatible"] = False
            report["compatibility_problem"] = str(e)
        started = time.monotonic()
        key, _ = engine_key(fingerprint(model.path), profile, native, gpu)
        cached = EngineCache(settings.cache_dir).lookup(model, key, profile, str(gpu.get("tensorrt")))
        report["engine_cache"] = {"key": key[:16], "cached": cached is not None,
                                  "path": str(cached.path) if cached else None,
                                  "digest_seconds": round(time.monotonic() - started, 2)}
    except Exception as e:  # noqa: BLE001
        report["error"] = str(e)
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


__all__ = ["main", "setup_logging"]
