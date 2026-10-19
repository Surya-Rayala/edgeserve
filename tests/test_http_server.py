"""The HTTP server's own limits: every connection must send a complete request head in time."""

from __future__ import annotations

import asyncio
import socket
import ssl
import time

import pytest
import uvicorn

from edgeserve import settings as cfg
from edgeserve.cli import _head_timeout_protocol


async def _app(scope, receive, send) -> None:
    assert scope["type"] == "http"
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.fixture
async def server(monkeypatch):
    monkeypatch.setattr(cfg, "HEAD_TIMEOUT_S", 0.3)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(_app, http=_head_timeout_protocol(), lifespan="off", log_config=None,
                            timeout_keep_alive=30)
    instance = uvicorn.Server(config)
    task = asyncio.create_task(instance.serve(sockets=[sock]))
    while not instance.started:
        await asyncio.sleep(0.01)
    yield sock.getsockname()[1]
    instance.should_exit = True
    await asyncio.wait_for(task, 5)


async def _closed_after(port: int, send: bytes) -> tuple[bytes, float]:
    """What the server sends on a connection that writes ``send`` and then goes quiet, and when it closes it."""
    loop = asyncio.get_running_loop()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    if send:
        writer.write(send)
        await writer.drain()
    started = loop.time()
    data = await asyncio.wait_for(reader.read(), 5)  # read() returns at EOF: the server closed the connection
    writer.close()
    return data, loop.time() - started


async def test_a_silent_connection_is_closed(server):
    data, elapsed = await _closed_after(server, b"")
    assert data == b"" and elapsed < 2  # uvicorn alone keeps it open forever


async def test_a_partial_request_head_is_closed(server):
    data, elapsed = await _closed_after(server, b"GET / HTTP/1.1\r\nHost: test\r\n")  # no final CRLF
    assert data == b"" and elapsed < 2


async def test_an_idle_keep_alive_connection_gets_the_same_deadline(server):
    data, elapsed = await _closed_after(server, b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
    assert data.startswith(b"HTTP/1.1 200") and data.endswith(b"ok")
    assert elapsed < 2  # instead of the 30 s keep-alive, which partial data would have reset


async def test_a_slow_but_complete_request_is_served(server):
    reader, writer = await asyncio.open_connection("127.0.0.1", server)
    writer.write(b"GET / HTTP/1.1\r\n")
    await asyncio.sleep(0.15)
    writer.write(b"Host: test\r\n\r\n")
    await writer.drain()
    head = await asyncio.wait_for(reader.readuntil(b"ok"), 5)
    assert head.startswith(b"HTTP/1.1 200")
    writer.close()


def test_the_server_caps_connections():
    assert cfg.MAX_CONNECTIONS >= 16  # several batch clients and users at once; bounded file descriptors


# ---------------------------------------------------------------------------- TLS and connection slots

async def _big_app(scope, receive, send) -> None:
    body = b"x" * (8 * 2**20)  # far more than the socket buffers hold
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


@pytest.fixture
def certificate(tmp_path):
    import subprocess

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                    "-days", "1", "-subj", "/CN=localhost"], check=True, capture_output=True)
    return cert, key


@pytest.fixture
async def tls_server(monkeypatch, certificate):
    monkeypatch.setattr(cfg, "HEAD_TIMEOUT_S", 0.3)
    monkeypatch.setattr(cfg, "WRITE_STALL_S", 0.3)
    monkeypatch.setattr(cfg, "CLOSE_GRACE_S", 0.3)
    monkeypatch.setattr(cfg, "MAX_CONNECTIONS_PER_CLIENT", 2)
    servers = []

    async def start(app=_app):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        config = uvicorn.Config(app, http=_head_timeout_protocol(), lifespan="off", log_config=None,
                                ssl_certfile=str(certificate[0]), ssl_keyfile=str(certificate[1]))
        instance = uvicorn.Server(config)
        task = asyncio.create_task(instance.serve(sockets=[sock]))
        while not instance.started:
            await asyncio.sleep(0.01)
        servers.append((instance, task))
        return instance, sock.getsockname()[1]

    yield start
    for instance, task in servers:
        instance.should_exit = True
        await asyncio.wait_for(task, 10)


def _silent_tls_client(port: int):
    """A client that completes the TLS handshake and then never reads: it never answers the server's close_notify."""
    import ssl

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    raw = socket.create_connection(("127.0.0.1", port))
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    return context.wrap_socket(raw, server_hostname="localhost")


async def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True


async def test_an_idle_tls_client_that_never_answers_the_close_loses_its_slot(tls_server):
    instance, port = await tls_server()
    client = await asyncio.to_thread(_silent_tls_client, port)
    try:
        assert await _until(lambda: len(instance.server_state.connections) == 1)
        # Over TLS, close() alone waits for the client's close_notify forever and the slot stays taken.
        assert await _until(lambda: not instance.server_state.connections), "the slot was never released"
    finally:
        client.close()


async def test_a_client_that_stops_reading_a_response_loses_its_slot(tls_server):
    instance, port = await tls_server(_big_app)
    client = await asyncio.to_thread(_silent_tls_client, port)
    try:
        await asyncio.to_thread(client.sendall, b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
        assert await _until(lambda: len(instance.server_state.connections) == 1)
        assert await _until(lambda: not instance.server_state.connections and not instance.server_state.tasks)
    finally:
        client.close()


async def test_one_client_cannot_take_every_connection(tls_server, monkeypatch):
    monkeypatch.setattr(cfg, "HEAD_TIMEOUT_S", 30.0)  # the two allowed connections stay open meanwhile
    instance, port = await tls_server()
    clients = [await asyncio.to_thread(_silent_tls_client, port) for _ in range(2)]
    try:
        assert await _until(lambda: len(instance.server_state.connections) == 2)
        # (Over TLS the check runs once the handshake is done; asyncio limits handshakes to 60 s.)
        refused = await asyncio.to_thread(_silent_tls_client, port)
        refused.settimeout(5)
        try:
            assert await asyncio.to_thread(refused.recv, 1) == b""  # dropped at once
        except (ConnectionResetError, OSError):
            pass  # dropped at once (reset)
        refused.close()
        assert len(instance.server_state.connections) == 2
    finally:
        for client in clients:
            client.close()


async def test_closing_does_not_cut_off_a_large_response_that_is_still_being_read(tls_server, monkeypatch):
    # Connection: close makes uvicorn close() right after queueing the body; all of it must still arrive.
    monkeypatch.setattr(cfg, "CLOSE_GRACE_S", 60.0)
    monkeypatch.setattr(cfg, "WRITE_STALL_S", 10.0)  # a slow reader is not a stalled one
    instance, port = await tls_server(_big_app)

    def reader() -> int:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        client = context.wrap_socket(socket.create_connection(("127.0.0.1", port)), server_hostname="localhost")
        client.sendall(b"GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
        received = 0
        while chunk := client.recv(65536):
            received += len(chunk)
            time.sleep(0.002)  # slower than the server writes: most of the body is still queued at close()
        client.close()
        return received

    assert await asyncio.wait_for(asyncio.to_thread(reader), 60) >= 8 * 2**20  # the headers and the whole body


async def test_shutdown_does_not_wait_for_idle_tls_clients(tls_server):
    instance, port = await tls_server()
    client = await asyncio.to_thread(_silent_tls_client, port)
    try:
        assert await _until(lambda: len(instance.server_state.connections) == 1)
        started = asyncio.get_running_loop().time()
        instance.should_exit = True
        assert await _until(lambda: not instance.server_state.connections, timeout=3)
        assert asyncio.get_running_loop().time() - started < 2  # uvicorn's close() would wait out its 5 s grace
    finally:
        client.close()
