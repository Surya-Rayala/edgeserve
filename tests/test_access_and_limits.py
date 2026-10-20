"""Client keys, the operator endpoints, per-client limits, the queue options, the request time limit and saved
refusals: through the HTTP API with the fake worker, and the scheduler and keys on their own."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import time

import httpx
import pytest

from conftest import edgeserve_server
from edgeserve import batch as batch_module
from edgeserve import cli, keys
from edgeserve import settings as cfg
from edgeserve.api.app import create_app
from edgeserve.batch import BatchItem, run_batch
from edgeserve.records import OutputDir
from edgeserve.scheduler import AdmissionError, RateLimiter, Scheduler
from edgeserve.service import Service
from test_service_api import CHAT, MESSAGES, ASGIRequest, eventually, make_settings, running, serving, wait_for_file

REMOTE = ("192.0.2.7", 40000)  # a client on another machine (RFC 5737 documentation address)


@contextlib.asynccontextmanager
async def serve(settings, client_address=("127.0.0.1", 123)):
    service = Service(settings)
    app = create_app(service)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=client_address)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", timeout=30) as client:
            await eventually(lambda: service.state in ("ready", "failed"), timeout=20)
            assert service.state == "ready", service.detail
            yield client, service


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def results(tmp_path) -> list[dict]:
    path = tmp_path / "out" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def read_record(tmp_path, custom_id: str) -> dict:
    path = tmp_path / "out" / "records" / f"{custom_id}.json"
    await wait_for_file(path)
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- edgeserve keys

def test_keys_are_shown_once_and_stored_as_hashes_only(capsys):
    assert cli.main(["keys", "add", "lab-laptop"]) == 0
    key = next(word for word in capsys.readouterr().out.split() if word.startswith("es-"))
    stored = keys.path().read_text()
    assert key not in stored and keys.digest(key) in stored
    assert stat.S_IMODE(keys.path().stat().st_mode) == 0o600
    assert cli.main(["keys", "list"]) == 0
    listed = capsys.readouterr().out
    assert "lab-laptop" in listed and key not in listed
    assert cli.main(["keys", "add", "lab-laptop"]) == 2 and "already has a key" in capsys.readouterr().err
    for bad in ("../etc", "", ".hidden", "has space", "a" * 65, "default"):
        assert cli.main(["keys", "add", bad]) == 2
    assert cli.main(["keys", "remove", "default"]) == 2 and "EDGESERVE_API_KEY" in capsys.readouterr().err
    assert cli.main(["keys", "remove", "lab-laptop"]) == 0
    assert cli.main(["keys", "remove", "lab-laptop"]) == 1
    assert keys.load() == {}


def test_the_key_store_follows_the_file(tmp_path):
    file = tmp_path / "client-keys.json"
    store = keys.KeyStore(None, always_required=False, file=file)
    assert not store.required  # this machine only, and no keys: open, as before
    key = keys.add("alice", file)
    assert store.required and store.client(key) == "alice" and store.client(key + "x") is None
    keys.remove("alice", file)
    assert store.client(key) is None and store.required  # the last key gone: still closed until a restart
    assert keys.KeyStore(None, always_required=True, file=file).required  # on the network: never open
    both = keys.KeyStore("main-key", always_required=True, file=file)
    assert both.client("main-key") == keys.DEFAULT_CLIENT


def test_a_broken_keys_file_keeps_only_the_keys_still_in_it(tmp_path):
    file = tmp_path / "client-keys.json"
    alice, bob = keys.add("alice", file), keys.add("bob", file)
    store = keys.KeyStore(None, always_required=False, file=file)
    assert store.client(alice) == "alice" and store.client(bob) == "bob"
    data = json.loads(file.read_text())
    del data["keys"]["bob"]  # a hand edit that removed bob, and left the file broken
    file.write_text(json.dumps(data)[:-2] + ",}")
    with pytest.raises(keys.KeysError, match="cannot read"):
        keys.load(file)
    assert store.client(alice) == "alice" and store.client(bob) is None and store.required


def test_the_key_store_sees_a_replaced_key(tmp_path):
    file = tmp_path / "client-keys.json"
    old = keys.add("alice", file)
    store = keys.KeyStore(None, always_required=True, file=file)
    before = file.stat()
    keys.remove("alice", file)
    new = keys.add("alice", file)  # the same name: the file has the same size again
    after = file.stat()
    if after.st_ino != before.st_ino:
        pytest.skip("this file system gave the new file another inode")
    os.utime(file, ns=(after.st_atime_ns, before.st_mtime_ns))  # the same clock tick: the same stamp
    assert store.client(new) == "alice" and store.client(old) is None


def test_keys_commands_do_not_lose_changes_made_at_the_same_time():
    import threading

    keys.add("alice")
    errors = []

    def change(action, name):
        try:
            action(name)
        except keys.KeysError as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=change, args=(keys.remove, "alice"))]
    threads += [threading.Thread(target=change, args=(keys.add, f"bob{i}")) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and sorted(keys.load()) == sorted(f"bob{i}" for i in range(8))


async def test_every_client_key_is_its_own_client(tmp_path, tiny_model, fake_native):
    alice = keys.add("alice")
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="main-key", host="0.0.0.0")
    async with serve(settings, REMOTE) as (client, _):
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice) | {"X-Request-Id": "a1"})
        assert r.status_code == 200
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer("main-key") | {"X-Request-Id": "m1"})
        assert r.status_code == 200
        assert (await read_record(tmp_path, "a1"))["client"] == "alice"
        assert (await read_record(tmp_path, "m1"))["client"] == keys.DEFAULT_CLIENT
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer("wrong") | {"X-Request-Id": "w1"})
        assert r.status_code == 401
        keys.remove("alice")  # revoked at once, without a restart
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice) | {"X-Request-Id": "a2"})
        assert r.status_code == 401
    lines = results(tmp_path)
    assert {(line["custom_id"], line["client"]) for line in lines} == {("a1", "alice"), ("m1", "default")}
    assert not (tmp_path / "out" / "records" / "w1.json").exists()  # unauthorized requests are never saved


async def test_a_network_server_without_any_key_lets_nobody_in(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0"), REMOTE) as (client, _):
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 401


# --------------------------------------------------------------------------- /metrics, /docs, /openapi.json

async def test_operator_pages_need_a_key_except_on_this_machine(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="main-key", host="0.0.0.0")
    async with serve(settings, REMOTE) as (client, _):
        for path in ("/metrics", "/docs", "/openapi.json"):
            assert (await client.get(path)).status_code == 401, path
            assert (await client.get(path, headers=bearer("main-key"))).status_code == 200, path
        assert (await client.head("/metrics")).status_code == 401
        for path in ("/health", "/health/ready"):
            assert (await client.get(path)).status_code == 200, path
    async with serve(settings, ("127.0.0.1", 50000)) as (client, _):
        for path in ("/metrics", "/docs", "/openapi.json"):
            assert (await client.get(path)).status_code == 200, path  # on the server itself, or an SSH tunnel


async def test_operator_pages_stay_open_without_keys(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native), REMOTE) as (client, _):
        assert (await client.get("/metrics")).status_code == 200
        r = await client.get("/docs")
        assert r.status_code == 200 and "swagger" in r.text.lower()


# --------------------------------------------------------------------------- per-client limits

async def test_max_client_requests_limits_each_client_alone(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "150")
    alice, bob = keys.add("alice"), keys.add("bob")
    settings = make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0", max_client_requests=1)
    async with serve(settings, REMOTE) as (client, _):
        first = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice)))
        await asyncio.sleep(0.2)
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice) | {"X-Request-Id": "busy"})
        assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_requests"
        assert 1 <= int(r.headers["retry-after"]) <= 60
        assert (await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(bob))).status_code == 200
        assert (await first).status_code == 200
        assert (await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice))).status_code == 200
    refused = await read_record(tmp_path, "busy")
    assert (refused["status"], refused["http_status"], refused["client"]) == ("rejected", 429, "alice")
    assert refused["request"]["messages"] == MESSAGES


async def test_client_rate_limit_counts_requests_per_minute(tmp_path, tiny_model, fake_native):
    alice, bob = keys.add("alice"), keys.add("bob")
    settings = make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0", client_rate_limit=2)
    async with serve(settings, REMOTE) as (client, _):
        codes = [(await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice))).status_code
                 for _ in range(2)]
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice) | {"X-Request-Id": "fast"})
        assert codes == [200, 200] and r.status_code == 429
        assert r.json()["error"]["code"] == "rate_limit_exceeded" and 1 <= int(r.headers["retry-after"]) <= 30
        assert (await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(bob))).status_code == 200
    assert (await read_record(tmp_path, "fast"))["status"] == "rejected"


def test_the_rate_limiter_refills_over_the_minute():
    now = [0.0]
    limiter = RateLimiter(3, clock=lambda: now[0])
    for _ in range(3):
        limiter.take("alice")
    with pytest.raises(AdmissionError) as refused:
        limiter.take("alice")
    assert refused.value.status == 429 and refused.value.retry_after == 20  # one more every 60/3 s
    limiter.take("bob")  # every client has its own allowance
    now[0] = 20.0
    limiter.take("alice")
    RateLimiter(0).take("anyone")  # 0: no limit


# --------------------------------------------------------------------------- the queue

async def test_queue_size_minus_one_keeps_every_request(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "10")
    async with serve(make_settings(tmp_path, tiny_model, fake_native, queue_size=-1)) as (client, _):
        replies = await asyncio.gather(*(client.post(CHAT, json={"messages": MESSAGES}) for _ in range(40)))
    assert [r.status_code for r in replies] == [200] * 40


async def test_a_full_queue_is_refused_saved_and_hints_a_short_retry(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "100")
    async with serve(make_settings(tmp_path, tiny_model, fake_native, queue_size=1)) as (client, _):
        warm = await client.post(CHAT, json={"messages": MESSAGES})  # one measured batch: about 0.6 s
        assert warm.status_code == 200
        running = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}))
        await asyncio.sleep(0.15)
        waiting = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}))
        await asyncio.sleep(0.15)
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "third"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "server_is_overloaded"
        assert int(r.headers["retry-after"]) <= 2  # the running batch's remaining time, not minutes
        assert (await running).status_code == 200 and (await waiting).status_code == 200
    record = await read_record(tmp_path, "third")
    assert (record["status"], record["http_status"], record["error"]["code"]) == ("rejected", 503,
                                                                                 "server_is_overloaded")
    assert any(line["custom_id"] == "third" and line["response"]["status_code"] == 503 for line in results(tmp_path))


@pytest.mark.parametrize(("timeout", "status"), [(1, 503), (0, 200)])
async def test_queue_timeout(tmp_path, tiny_model, fake_native, monkeypatch, timeout, status):
    monkeypatch.setenv("FAKE_STEP_MS", "400")  # 6 tokens: about 2.4 s per request
    settings = make_settings(tmp_path, tiny_model, fake_native, queue_timeout=timeout)
    async with serve(settings) as (client, _):
        first = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}))
        await asyncio.sleep(0.2)
        r = await client.post(CHAT, json={"messages": MESSAGES})
        assert r.status_code == status
        if status == 503:
            assert r.json()["error"]["code"] == "queue_timeout"
        assert (await first).status_code == 200


# --------------------------------------------------------------------------- --request-timeout

async def test_request_timeout_stops_a_running_request(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "500")  # about 3 s for 6 tokens
    async with serve(make_settings(tmp_path, tiny_model, fake_native, request_timeout=1)) as (client, _):
        started = time.monotonic()
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "slow"})
        assert r.status_code == 504 and r.json()["error"]["code"] == "time_limit_exceeded"
        assert time.monotonic() - started < 2.9
        async with client.stream("POST", CHAT, json={"messages": MESSAGES, "stream": True}) as stream:
            body = "".join([chunk async for chunk in stream.aiter_text()])
        assert "time_limit_exceeded" in body and body.rstrip().endswith("data: [DONE]")
    record = await read_record(tmp_path, "slow")
    assert (record["status"], record["http_status"], record["error"]["code"]) == ("failed", 504,
                                                                                 "time_limit_exceeded")


async def test_request_timeout_also_covers_the_wait(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "400")
    settings = make_settings(tmp_path, tiny_model, fake_native, request_timeout=1, queue_timeout=0)
    async with serve(settings) as (client, _):
        first = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}))
        await asyncio.sleep(0.1)
        r = await client.post(CHAT, json={"messages": MESSAGES})
        assert r.status_code == 504 and r.json()["error"]["code"] == "time_limit_exceeded"
        assert (await first).status_code == 504


# --------------------------------------------------------------------------- saved refusals

async def test_a_refusal_never_replaces_an_earlier_record(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        assert (await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "done-1"})).status_code \
            == 200
        await wait_for_file(tmp_path / "out" / "records" / "done-1.json")
        r = await client.post(CHAT, json={"messages": MESSAGES, "temperature": 9}, headers={"X-Request-Id": "done-1"})
        assert r.status_code == 400
        r = await client.post(CHAT, json={"messages": MESSAGES, "max_tokens": 0}, headers={"X-Request-Id": "bad-1"})
        assert r.status_code == 400
        await eventually(lambda: len(results(tmp_path)) == 3)
    assert (await read_record(tmp_path, "done-1"))["status"] == "completed"
    assert (await read_record(tmp_path, "bad-1"))["status"] == "rejected"
    statuses = [(line["custom_id"], line["response"]["status_code"]) for line in results(tmp_path)]
    assert statuses == [("done-1", 200), ("done-1", 400), ("bad-1", 400)]


# --------------------------------------------------------------------------- the scheduler and the settings

async def test_retry_after_is_the_running_batchs_remaining_time():
    scheduler = Scheduler(lambda: None, max_batch_size=1, kv_capacity=128, queue_size=1, queue_timeout_s=600)
    assert scheduler.retry_after_s() == 5  # nothing measured yet
    scheduler._batch_s, scheduler._batch_started = 10.0, time.monotonic() - 3
    assert scheduler.retry_after_s() == 7
    scheduler._batch_started = None  # nothing running: room again at once
    assert scheduler.retry_after_s() == 1
    scheduler._batch_s, scheduler._batch_started = 500.0, time.monotonic()
    assert scheduler.retry_after_s() == 60  # never more than the OpenAI SDK waits


@pytest.mark.parametrize(("flags", "message"), [
    (["--queue-size", "0"], "--queue-size"), (["--queue-size", "-2"], "--queue-size"),
    (["--queue-timeout", "-1"], "--queue-timeout"), (["--request-timeout", "-5"], "--request-timeout"),
    (["--max-client-requests", "-1"], "--max-client-requests"), (["--client-rate-limit", "-1"], "--client-rate-limit"),
])
def test_bad_limits_are_refused_with_the_flag_named(tmp_path, flags, message):
    with pytest.raises(ValueError, match=message):
        cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), *flags]))


def test_limits_come_from_flags_the_environment_or_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGESERVE_QUEUE_SIZE", "-1")
    monkeypatch.setenv("EDGESERVE_CLIENT_RATE_LIMIT", "30")
    settings = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--queue-timeout", "0",
                                                       "--request-timeout", "120", "--max-client-requests", "4"]))
    assert (settings.queue_size, settings.queue_timeout, settings.request_timeout, settings.max_client_requests,
            settings.client_rate_limit) == (-1, 0, 120, 4, 30)
    defaults = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path)]))
    assert (defaults.queue_size, defaults.queue_timeout, defaults.request_timeout, defaults.max_client_requests,
            defaults.client_rate_limit) == (-1, 0, 0, 0, 30)  # the environment above; -1 also lifts the wait limit


def test_queue_size_minus_one_alone_keeps_every_request_until_its_turn(tmp_path):
    settings = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--queue-size", "-1"]))
    assert settings.queue_timeout == 0  # else a request that waits 10 minutes for its turn fails with 503
    explicit = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--queue-size", "-1",
                                                       "--queue-timeout", "30"]))
    assert explicit.queue_timeout == 30  # a wait limit asked for still applies


def test_build_and_info_do_not_need_a_readable_keys_file(tmp_path, tiny_model, fake_native, monkeypatch, capsys):
    monkeypatch.setenv("EDGESERVE_NATIVE_DIR", str(fake_native))
    keys.add("alice")
    keys.path().write_text("{not json")  # e.g. edited by hand
    settings = cfg.from_args(cli._parser().parse_args(["build", "--model", str(tiny_model)]))
    assert settings.model == tiny_model.resolve()
    assert cli.main(["info", "--model", str(tiny_model)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "problem" in report["client_keys"] and report["model"]["num_layers"] > 0


def test_client_keys_alone_allow_serving_on_the_network(tmp_path):
    with pytest.raises(ValueError, match="without an API key"):
        cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--host", "0.0.0.0"]))
    keys.add("alice")
    settings = cfg.from_args(cli._parser().parse_args(["serve", "--model", str(tmp_path), "--host", "0.0.0.0"]))
    assert settings.client_keys and settings.api_key is None


# --------------------------------------------------------------------------- from the review: isolation and caps

async def test_a_client_key_reads_and_reuses_only_its_own_ids(tmp_path, tiny_model, fake_native):
    alice, bob = keys.add("alice"), keys.add("bob")
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="main-key", host="0.0.0.0")
    async with serve(settings, REMOTE) as (client, _):
        r = await client.post(CHAT, json={"messages": [{"role": "user", "content": "alice's private text"}]},
                              headers=bearer(alice) | {"X-Request-Id": "request-1"})
        assert r.status_code == 200
        await wait_for_file(tmp_path / "out" / "records" / "request-1.json")
        assert (await client.get("/v1/records/request-1", headers=bearer(alice))).status_code == 200
        assert (await client.get("/v1/records/request-1", headers=bearer(bob))).status_code == 404  # as if absent
        assert (await client.get("/v1/records/request-1", headers=bearer("main-key"))).status_code == 200  # operator
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(bob) | {"X-Request-Id": "request-1"})
        assert r.status_code == 409 and "another client" in r.json()["error"]["message"]
        assert r.headers["x-should-retry"] == "false"
    record = json.loads((tmp_path / "out" / "records" / "request-1.json").read_text())
    assert record["client"] == "alice" and record["status"] == "completed"  # bob replaced nothing


async def test_operator_pages_stay_closed_through_a_proxy_or_another_host_name(tmp_path, tiny_model, fake_native):
    settings = make_settings(tmp_path, tiny_model, fake_native, api_key="main-key", host="0.0.0.0")
    async with serve(settings, ("127.0.0.1", 50000)) as (client, _):
        assert (await client.get("/metrics")).status_code == 200
        assert (await client.get("/metrics", headers={"X-Forwarded-For": "192.0.2.9"})).status_code == 401
        assert (await client.get("/metrics", headers={"Host": "rebind.example:8000"})).status_code == 401


async def test_this_machine_is_one_client_on_a_server_without_keys(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native, client_rate_limit=1)) as (client, _):
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 200
        r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Forwarded-For": "198.51.100.1"})
        assert r.status_code == 429  # a forwarding header does not make it another client


async def test_saved_refusals_are_capped_per_client(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native, client_rate_limit=1)) as (client, _):
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 200
        for i in range(80):
            r = await client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": f"flood-{i}"})
            assert r.status_code == 429
        r = await client.post(CHAT, json={"messages": MESSAGES})  # no ID of its own: a line, no record file
        assert r.status_code == 429
        await asyncio.sleep(0.5)
    refused = [line for line in results(tmp_path) if line["response"]["status_code"] == 429]
    assert len(refused) <= 60  # at most 60 saved refusals a minute per client
    files = sorted(p.name for p in (tmp_path / "out" / "records").iterdir())
    assert all(name.startswith("flood-") for name in files if not name.startswith("req_")) and \
        not any(name.startswith("req_") for name in files if json.loads((tmp_path / "out" / "records" / name)
                                                                       .read_text())["status"] == "rejected")


async def test_the_text_of_a_large_refused_request_is_not_saved(tmp_path, tiny_model, fake_native):
    alice = keys.add("alice")
    settings = make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0", max_client_requests=1)
    big = "word " * 20000  # 100 KB: valid (messages may have 128 KB), but too large to keep in a refusal
    async with serve(settings, REMOTE) as (client, service):
        service.scheduler.check_client = lambda client: (_ for _ in ()).throw(
            AdmissionError(429, "too_many_requests", "busy", retry_after=1, err_type="rate_limit_error"))
        r = await client.post(CHAT, json={"messages": [{"role": "user", "content": big}], "max_tokens": 3},
                              headers=bearer(alice) | {"X-Request-Id": "big-1"})
        assert r.status_code == 429
    record = await read_record(tmp_path, "big-1")
    assert "messages" not in record["request"] and record["request"]["max_tokens"] == 3
    assert "byte request" in record["request"]["omitted"]


async def test_a_refusal_leaves_a_running_request_of_its_id_alone(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "300")
    async with serve(make_settings(tmp_path, tiny_model, fake_native)) as (client, service):
        first = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "busy-1"}))
        await eventually(lambda: service.scheduler.running == 1)
        r = await client.post(CHAT, content=b"{not json",
                              headers={"X-Request-Id": "busy-1", "content-type": "application/json"})
        assert r.status_code == 400
        await eventually(lambda: len(results(tmp_path)) == 1)  # the refusal's results.jsonl line
        assert (await client.get("/v1/records/busy-1")).status_code == 404  # running: no record yet, not "rejected"
        assert (await first).status_code == 200
    assert (await read_record(tmp_path, "busy-1"))["status"] == "completed"


async def test_an_unusable_output_dir_fails_the_startup_with_its_reason(tmp_path, tiny_model, fake_native):
    (tmp_path / "a-file").write_text("")
    settings = make_settings(tmp_path, tiny_model, fake_native, output_dir=tmp_path / "a-file" / "out")
    async with running(settings, expect_ready=False) as (client, service):
        assert service.state == "failed" and "a-file" in service.detail
        assert (await client.get("/health")).status_code == 503


async def test_the_api_pages_answer_head_like_every_get_endpoint(tmp_path, tiny_model, fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native)) as (client, _):
        for path in ("/docs", "/openapi.json"):
            r = await client.head(path)
            assert r.status_code == 200 and r.content == b"", path


# --------------------------------------------------------------------------- from the review: paths that free places

async def test_a_refused_stream_is_saved_like_any_refusal(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "150")
    alice = keys.add("alice")
    settings = make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0", max_client_requests=1)
    async with serve(settings, REMOTE) as (client, service):
        first = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}, headers=bearer(alice)))
        await eventually(lambda: service.scheduler.running == 1)
        r = await client.post(CHAT, json={"messages": MESSAGES, "stream": True},
                              headers=bearer(alice) | {"X-Request-Id": "s-busy"})
        assert r.status_code == 429 and r.headers["content-type"] == "application/json"
        assert (await first).status_code == 200
    record = await read_record(tmp_path, "s-busy")
    assert (record["status"], record["client"], record["schema_version"]) == ("rejected", "alice", "1.1")
    assert record["request"]["stream"] is True


async def test_a_request_refused_before_its_body_is_read_is_saved_under_its_header_id(tmp_path, tiny_model,
                                                                                       fake_native):
    async with serve(make_settings(tmp_path, tiny_model, fake_native, client_rate_limit=1)) as (client, _):
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 200
        r = await client.post(CHAT, json={"messages": MESSAGES, "metadata": {"k": "v"}},
                              headers={"X-Request-Id": "early"})
        assert r.status_code == 429 and r.headers["x-request-id"] == "early"
    record = await read_record(tmp_path, "early")
    assert (record["status"], record["request"], record["metadata"], record["client"]) == ("rejected", None, None,
                                                                                         "local")


async def test_a_client_that_went_away_frees_its_place(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_TOKENS", "40")
    monkeypatch.setenv("FAKE_STEP_MS", "50")
    settings = make_settings(tmp_path, tiny_model, fake_native, max_client_requests=1, max_new_tokens=40)
    async with serving(settings) as (app, client, service):
        for payload in ({"messages": MESSAGES, "max_tokens": 40, "stream": True}, {"messages": MESSAGES,
                                                                                   "max_tokens": 40}):
            call = ASGIRequest(app, CHAT, payload)
            task = asyncio.create_task(call.run())
            await eventually(lambda: service.scheduler.running == 1)
            await asyncio.sleep(0.15)
            call.gone.set()
            await asyncio.wait_for(task, 5)
            await eventually(lambda: not service.scheduler._per_client)
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 200


async def test_a_worker_crash_frees_the_clients_places(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_FILE", str(tmp_path / "crashed-once"))
    async with serve(make_settings(tmp_path, tiny_model, fake_native, max_client_requests=1)) as (client, service):
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 503
        await eventually(lambda: service.state == "ready", timeout=10)
        assert (await client.post(CHAT, json={"messages": MESSAGES})).status_code == 200


async def test_request_timeout_stops_every_request_of_a_batch_at_once(tmp_path, tiny_model, fake_native,
                                                                       monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "400")
    settings = make_settings(tmp_path, tiny_model, fake_native, max_batch_size=2, request_timeout=2)
    async with serve(settings) as (client, service):
        scheduler, get_worker = service.scheduler, service.scheduler._get_worker
        scheduler._get_worker = lambda: None  # both wait, then leave the queue together
        plain = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES}, headers={"X-Request-Id": "t0"}))
        streamed = asyncio.create_task(client.post(CHAT, json={"messages": MESSAGES, "stream": True},
                                                   headers={"X-Request-Id": "t1"}))
        await eventually(lambda: scheduler.waiting == 2)
        scheduler._get_worker = get_worker
        scheduler.wake()
        r, s = await plain, await streamed
        assert r.status_code == 504 and "time_limit_exceeded" in s.text and s.text.endswith("data: [DONE]\n\n")
        assert r.headers.get("x-should-retry") == "false"  # stopped while running: not to be sent again
        await eventually(lambda: not scheduler._per_client)
    for custom_id in ("t0", "t1"):
        record = await read_record(tmp_path, custom_id)
        assert (record["http_status"], record["metrics"]["batch_size"]) == (504, 2)


async def test_a_request_stopped_by_the_time_limit_is_not_sent_again(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "500")  # about 3 s, over the 1 s limit
    real_sleep = asyncio.sleep

    async def no_wait(seconds):  # the batch runner's back-off, skipped
        await real_sleep(0)

    monkeypatch.setattr(batch_module, "_sleep", no_wait)
    service = Service(make_settings(tmp_path, tiny_model, fake_native, request_timeout=1))
    app = create_app(service)
    async with app.router.lifespan_context(app):
        await eventually(lambda: service.state == "ready", timeout=20)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(CHAT, json={"messages": MESSAGES})
        assert r.status_code == 504 and r.headers.get("x-should-retry") == "false"
        await run_batch([BatchItem("long-1", CHAT, {"messages": MESSAGES}, 1)], OutputDir(tmp_path / "batch-out"),
                        "http://test", progress=lambda line: None, _transport=transport)
        await eventually(lambda: any(line["custom_id"] == "long-1" for line in results(tmp_path)))
    assert len([line for line in results(tmp_path) if line["custom_id"] == "long-1"]) == 1


async def test_a_batch_run_stays_within_the_servers_client_limit(tmp_path, tiny_model, fake_native, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_MS", "100")  # about 0.6 s per request
    alice = keys.add("alice")
    service = Service(make_settings(tmp_path, tiny_model, fake_native, host="0.0.0.0", max_client_requests=1))
    app = create_app(service)
    async with app.router.lifespan_context(app):
        await eventually(lambda: service.state == "ready", timeout=20)
        items = [BatchItem(f"seg-{i:02d}", CHAT, {"messages": MESSAGES}, i + 1) for i in range(12)]
        run = await run_batch(items, OutputDir(tmp_path / "batch-out"), "http://test", api_key=alice,
                              progress=lambda line: None, _transport=httpx.ASGITransport(app=app, client=REMOTE))
    assert run["request_counts"]["failed"] == 0  # the server runs them all, one at a time


def test_queue_size_minus_one_keeps_every_request_of_one_machine(tiny_model, fake_native, tmp_path):
    with edgeserve_server(tiny_model, fake_native, EDGESERVE_QUEUE_SIZE="-1",
                          EDGESERVE_CACHE_DIR=str(tmp_path / "cache")) as url:
        async def burst():
            async with httpx.AsyncClient(base_url=url, timeout=120,
                                         limits=httpx.Limits(max_connections=100)) as client:
                return await asyncio.gather(*(client.post(CHAT, json={"messages": MESSAGES}) for _ in range(40)),
                                            return_exceptions=True)

        replies = asyncio.run(burst())
    outcomes = [r.status_code if isinstance(r, httpx.Response) else type(r).__name__ for r in replies]
    assert outcomes == [200] * 40, {o: outcomes.count(o) for o in set(outcomes)}
