"""Shared fixtures: a tiny Edge-LLM-shaped model folder and a fake native build (fake worker + llm_build)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """No test sees the developer's settings file, built runtime, engines or EDGESERVE_* variables, and the
    logging setup of cli.main() is undone after each test."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for name in list(os.environ):
        if name.startswith(("EDGESERVE_", "XDG_")):
            monkeypatch.delenv(name)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    access = logging.getLogger("uvicorn.access")
    filters = access.filters[:]
    yield home
    root.handlers[:] = handlers
    root.setLevel(level)
    access.filters[:] = filters

HIDDEN, VOCAB, INTER = 64, 128, 96


def _safetensors(path: Path, name: str, dtype: str, shape: list[int], itemsize: int) -> None:
    size = itemsize
    for dim in shape:
        size *= dim
    header = json.dumps({name: {"dtype": dtype, "shape": shape, "data_offsets": [0, size]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(size))


def make_tiny_model(root: Path, name: str = "tiny_model", chat_template: bool = True) -> Path:
    """A 1-layer Llama-shaped Edge-LLM export: one INT4 plugin linear per projection plus an fp16 lm_head."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    folder = root / name
    folder.mkdir(parents=True)
    config = {
        "model": "llama", "engine_role": "llm", "spec_decode_type": "none", "edgellm_version": "0.11.0",
        "vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "kv_cache_dtype": "fp16",
        "eos_token_id": [2], "external_weight_files": [],
    }
    (folder / "config.json").write_text(json.dumps(config))

    # The graph is never executed, only inspected, so the projections are simply chained.
    nodes, inits = [], []
    current = "inputs_embeds"
    shapes = [("q", HIDDEN, HIDDEN), ("k", 32, HIDDEN), ("v", 32, HIDDEN), ("o", HIDDEN, HIDDEN),
              ("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER)]
    for label, n, k in shapes:
        qweight = numpy_helper.from_array(np.zeros((max(1, n * k // 2 // 64), 64), dtype=np.int8), f"{label}.qweight")
        scales = numpy_helper.from_array(np.ones((max(1, k // 128), n), dtype=np.float16), f"{label}.scales")
        inits += [qweight, scales]
        nodes.append(helper.make_node("Int4GroupwiseGemmPluginV2", [current, qweight.name, scales.name],
                                      [f"{label}_out"], domain="trt_edgellm", gemm_n=n, gemm_k=k, group_size=128))
        current = f"{label}_out"
    lm = numpy_helper.from_array(np.zeros((VOCAB, HIDDEN), dtype=np.float16), "lm_head.weight")
    inits.append(lm)
    nodes.append(helper.make_node("Gemm", [current, "lm_head.weight"], ["logits_fp16"], transB=1))
    nodes.append(helper.make_node("Cast", ["logits_fp16"], ["logits"], to=TensorProto.FLOAT))
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, ["tokens", HIDDEN])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["sequences", VOCAB])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("trt_edgellm", 1)])
    onnx.save(model, str(folder / "model.onnx"))

    _safetensors(folder / "embedding.safetensors", "embedding", "F16", [VOCAB, HIDDEN], 2)
    (folder / "tokenizer.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>"}))
    if chat_template:
        (folder / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    (folder / "generation_config.json").write_text(json.dumps({"do_sample": False, "max_new_tokens": 6}))
    return folder


def _script(path: Path, target: Path) -> None:
    path.write_text(f"#!/bin/sh\nexec {sys.executable} {target} \"$@\"\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_fake_native(root: Path) -> Path:
    native = root / "native"
    (native / "examples" / "llm").mkdir(parents=True)
    _script(native / "edgeserve-worker", TESTS / "fake_worker.py")
    _script(native / "examples" / "llm" / "llm_build", TESTS / "fake_llm_build.py")
    (native / "libNvInfer_edgellm_plugin.so").write_bytes(b"")
    (native / "build_info.json").write_text(json.dumps({
        "edgellm_commit": "fake", "patches": {}, "cuda": "12.6", "tensorrt": "10.3.0.30", "sm": 87,
        "arch": "aarch64", "embedded_target": None, "plugin_sha256": "0" * 64,
        "cute_dsl": {"groups": [], "cuda_version": "12.6", "cutlass_dsl_version": "4.7.0"},
    }))
    return native


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    return make_tiny_model(tmp_path / "models")


@pytest.fixture
def fake_native(tmp_path: Path) -> Path:
    return make_fake_native(tmp_path)


@contextlib.contextmanager
def edgeserve_server(model: Path, native: Path, network: bool = False, **settings: str) -> Iterator[str]:
    """``edgeserve serve`` in a subprocess with the fake worker, configured through the settings file as
    ``edgeserve setup`` writes it (with ``network``: API key and HTTPS, but listening on 127.0.0.1 only).
    Yields the server's URL once it is ready."""
    import httpx

    from edgeserve import config, tls

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    values = {"EDGESERVE_MODEL": str(model), "EDGESERVE_PORT": str(port), "EDGESERVE_HOST": "127.0.0.1", **settings}
    verify: object = True
    if network:
        folder = config.config_dir()
        tls.ensure(folder)
        values.update(EDGESERVE_API_KEY=tls.new_api_key(), EDGESERVE_TLS_CERT=str(folder / tls.SERVER_CERT),
                      EDGESERVE_TLS_KEY=str(folder / tls.SERVER_KEY))
        import ssl

        verify = ssl.create_default_context(cafile=str(folder / tls.CA_CERT))
    config.write(values)
    env = {name: value for name, value in os.environ.items() if not name.startswith("EDGESERVE_")}
    env["EDGESERVE_NATIVE_DIR"] = str(native)
    url = f"{'https' if network else 'http'}://127.0.0.1:{port}"
    log = open(config.config_dir() / "serve.log", "w")  # noqa: SIM115 (closed below)
    proc = subprocess.Popen([sys.executable, "-m", "edgeserve", "serve"], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 120
        while True:
            assert proc.poll() is None, (config.config_dir() / "serve.log").read_text()[-3000:]
            assert time.monotonic() < deadline, "the server did not become ready"
            try:
                if httpx.get(f"{url}/health/ready", verify=verify, timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        yield url
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
