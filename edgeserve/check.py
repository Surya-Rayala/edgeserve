"""``edgeserve check``: check a running server end to end.

Waits until the server is ready (the first start may build an engine), then checks the model list,
one chat completion with a request ID, streaming, and the stored record. Without ``--server`` it
checks this machine's server with the settings file's address, API key and CA certificate.
"""

from __future__ import annotations

import json
import ssl
import time

import httpx

from edgeserve import config

READY_WAIT_S = 900.0
MESSAGES = [{"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
            {"role": "user", "content": "What does a compiler do?"}]


def run(server: str | None = None, ca_cert: str | None = None) -> int:
    target = config.client_target(server, ca_cert)
    headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    verify: ssl.SSLContext | bool = (ssl.create_default_context(cafile=str(target.ca_cert)) if target.ca_cert
                                     else True)
    print(f"checking {target.url}")
    with httpx.Client(base_url=target.url, headers=headers, verify=verify,
                      timeout=httpx.Timeout(600, connect=10)) as client:
        return _checks(client, target.url)


def _checks(client: httpx.Client, url: str) -> int:
    deadline = time.monotonic() + READY_WAIT_S
    last = ""
    while True:
        try:
            r = client.get("/health/ready")
            if r.status_code == 200:
                break
            try:
                status = r.json()
            except ValueError:  # e.g. a proxy's error page
                status = {"status": f"HTTP {r.status_code}", "detail": r.text[:100]}
            message = f"{status.get('status')}: {status.get('detail')}"
            if status.get("status") == "failed":
                print(f"FAIL the server failed to start: {status.get('detail')}")
                return 1
        except httpx.HTTPError as e:
            if (problem := _permanent(e, url)) is not None:  # waiting would not help
                print(f"FAIL {problem}")
                return 1
            message = f"not reachable ({e.__class__.__name__})"
        if message != last:
            print(f"waiting for the server: {message}")
            last = message
        if time.monotonic() > deadline:
            print("FAIL the server was not ready in time")
            return 1
        time.sleep(2)

    r = client.get("/v1/models")
    if r.status_code != 200:
        hint = " (set EDGESERVE_API_KEY to the server's key)" if r.status_code == 401 else ""
        print(f"FAIL /v1/models: HTTP {r.status_code} {r.text[:300]}{hint}")
        return 1
    model = r.json()["data"][0]
    print(f"ok   model {model['id']}: max input {model.get('max_input_tokens')} tokens, "
          f"batch {model.get('max_batch_size')}, {model.get('quantization')}")

    custom_id = f"check-{int(time.time())}"
    r = client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 64},
                    headers={"X-Request-Id": custom_id})
    if r.status_code != 200 or r.headers.get("x-request-id") != custom_id:
        print(f"FAIL chat completion: HTTP {r.status_code} {r.text[:300]}")
        return 1
    body = r.json()
    m = body["metrics"]
    print(f"ok   chat completion id={body['id']} ({body['usage']['completion_tokens']} tokens): "
          f"TTFT {m['ttft_ms']:.0f} ms, {m['decode_tokens_per_s']} tok/s, TPOT {m['tpot_ms']} ms, "
          f"MBU {m['decode_mbu']}, energy {m['energy_j']} J")
    print(f"     text: {body['choices'][0]['message']['content'][:200]!r}")

    pieces = []
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": MESSAGES, "max_tokens": 32, "stream": True,
                             "stream_options": {"include_usage": True}}) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices"):
                    pieces.append(chunk["choices"][0].get("delta", {}).get("content") or "")
    if not "".join(pieces).strip():
        print("FAIL streaming returned no text")
        return 1
    print(f"ok   streaming ({len(pieces)} chunks)")

    r = client.get(f"/v1/records/{custom_id}")
    if r.status_code == 200:
        record = r.json()
        print(f"ok   record {custom_id}: {record['steps']['count']} steps, status {record['status']}")
    else:
        print(f"note record lookup returned HTTP {r.status_code} (the server keeps records only with --output-dir)")
    print("all checks passed")
    return 0


def _permanent(error: BaseException, server: str) -> str | None:
    """A connection problem that waiting for the server does not fix (certificate, wrong scheme)."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, ssl.SSLCertVerificationError):
            return (f"the server certificate is not accepted ({getattr(cause, 'verify_message', cause)}): pass "
                    "--ca-cert edgeserve-ca.crt and use a host name the certificate lists")
        if isinstance(cause, ssl.SSLError) and getattr(cause, "reason", None) == "WRONG_VERSION_NUMBER":
            return f"{server} does not answer TLS: use http://"
        cause = cause.__cause__ or cause.__context__
    if isinstance(error, (httpx.RemoteProtocolError, httpx.ReadError)) and server.startswith("http://"):
        return f"{server} closed the connection without answering: if it serves HTTPS, use https://"
    return None


__all__ = ["run"]
