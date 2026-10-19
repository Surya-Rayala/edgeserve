#!/usr/bin/env python3
"""Stand-in for Edge-LLM's llm_build: writes a minimal engine folder (FAKE_LLM_BUILD_FAIL=1 simulates an OOM)."""

from __future__ import annotations

import json
import os
import shutil
import sys


def main() -> int:
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    if os.environ.get("FAKE_LLM_BUILD_FAIL") == "1":
        print("[ERROR] [TensorRT] out of memory while building", flush=True)
        return 1
    src, dst = args["--onnxDir"], args["--engineDir"]
    os.makedirs(dst, exist_ok=True)
    header = b"ftrt" + bytes(20) + bytes([10, 3, 0, 30]) + bytes(36)
    with open(os.path.join(dst, "llm.engine"), "wb") as f:
        f.write(header)
    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    config["builder_config"] = {"max_batch_size": int(args["--maxBatchSize"]), "max_input_len": int(args["--maxInputLen"]),
                                "max_kv_cache_capacity": int(args["--maxKVCacheCapacity"])}
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), dst)
    with open(os.path.join(dst, "invocations.log"), "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")
    print("[INFO] LLM engine built successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
