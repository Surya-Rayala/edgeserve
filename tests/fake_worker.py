#!/usr/bin/env python3
"""Stand-in for edgeserve-worker that speaks the same JSON-lines protocol, for tests without a GPU.

Behaviour knobs (environment):
  FAKE_TOKENS            tokens generated per sequence when max_new_tokens allows (default 6)
  FAKE_STEP_MS           delay per step (default 5)
  FAKE_LOAD_FAIL=1       report load_failed and exit 2
  FAKE_CRASH_FILE=path   crash (exit 3) mid-generation once: only while <path> does not exist yet
  FAKE_STREAM_HOLD=N     streamed text holds back the last N characters, as Edge-LLM's stop-string look-ahead does
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

PROBE = {
    "ev": "probe", "gpu_name": "FakeGPU", "sm": 87, "sm_count": 8, "integrated": False,
    "total_memory_bytes": 64 * 2**30, "free_memory_bytes": 60 * 2**30, "memory_bus_width_bits": 128,
    "l2_cache_bytes": 2**21, "clock_rate_khz": 1_000_000, "memory_clock_rate_khz": 3_000_000,
    "cuda_runtime": "12.6", "cuda_driver": "12.6", "tensorrt": "10.3.0.30", "tensorrt_runtime": "10.3.0",
    "edgellm": "0.11.0",
}
_out_lock = threading.Lock()


def emit(event: dict) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


def count_tokens(item: dict) -> int:
    if "prompt" in item:
        return len(item["prompt"].split()) + 1
    return sum(len(m["content"].split()) for m in item["messages"]) + 10


class Worker:
    def __init__(self) -> None:
        self.cancelled: dict[int, set[int]] = {}
        self.lock = threading.Lock()

    def generate(self, cmd: dict, received_ns: int) -> None:
        job_id = cmd["id"]
        items = cmd["requests"]
        max_new = int(cmd["sampling"]["max_new_tokens"])
        want = min(max_new, int(os.environ.get("FAKE_TOKENS", "6")))
        step_s = float(os.environ.get("FAKE_STEP_MS", "5")) / 1000
        crash_file = os.environ.get("FAKE_CRASH_FILE")
        hold = int(os.environ.get("FAKE_STREAM_HOLD", "0"))
        start = time.monotonic_ns()
        t_ns, gpu_ms, free, active = [], [], [], []
        outputs = [[] for _ in items]
        finished = [None] * len(items)
        streamed = [0] * len(items)  # characters of each sequence's text sent so far
        for step in range(want):
            time.sleep(step_s)
            if crash_file and step == 1 and not os.path.exists(crash_file):
                open(crash_file, "w").close()
                os._exit(3)
            t_ns.append(time.monotonic_ns())
            gpu_ms.append(step_s * 1000 * 0.9)
            free.append(30 * 2**30)
            live = 0
            for r in range(len(items)):
                if finished[r]:
                    continue
                with self.lock:
                    if r in self.cancelled.get(job_id, set()):
                        finished[r] = "cancelled"
                        continue
                live += 1
                outputs[r].append(100 + step)
                if cmd.get("stream"):
                    text = "".join(f"tok{i} " for i in range(len(outputs[r])))
                    upto = max(streamed[r], len(text) - hold)
                    if upto > streamed[r]:
                        emit({"ev": "text", "id": job_id, "req": r, "t_ns": t_ns[-1], "text": text[streamed[r]:upto]})
                        streamed[r] = upto
                if step == want - 1:
                    finished[r] = "length" if want == max_new else "stop"
            active.append(max(1, live))
            if all(finished):
                break
        results = []
        for r, item in enumerate(items):
            ids = outputs[r]
            results.append({"req": r, "finish_reason": finished[r] or "stop", "prompt_tokens": count_tokens(item),
                            "text": "".join(f"tok{i} " for i in range(len(ids))), "token_ids": ids,
                            "pieces": [f"tok{i} " for i in range(len(ids))]})
        n_steps = max((len(o) for o in outputs), default=0)
        emit({"ev": "done", "id": job_id, "t_received_ns": received_ns, "t_start_ns": start,
              "t_end_ns": time.monotonic_ns(),
              "steps": {"t_ns": t_ns[:n_steps], "gpu_ms": gpu_ms[:n_steps], "free_bytes": free[:n_steps],
                        "active": active[:n_steps]},
              "results": results})
        with self.lock:
            self.cancelled.pop(job_id, None)


def main() -> int:
    args = sys.argv[1:]
    if args == ["--probe"]:
        emit(PROBE)
        return 0
    if "--engine-dir" not in args:
        print("--engine-dir is required", file=sys.stderr)
        return 64
    engine_dir = args[args.index("--engine-dir") + 1]
    print(f"[edgeserve-worker] info: fake worker loading {engine_dir}", file=sys.stderr, flush=True)
    if os.environ.get("FAKE_LOAD_FAIL") == "1":
        emit({"ev": "error", "id": None, "code": "load_failed", "message": "fake load failure", "fatal": True})
        return 2
    if not os.path.exists(os.path.join(engine_dir, "llm.engine")):
        emit({"ev": "error", "id": None, "code": "load_failed", "message": "no llm.engine", "fatal": True})
        return 2
    emit({"ev": "ready", "engine_dir": engine_dir, "max_batch_size": 4, "cuda_graph": True, "load_ms": 1.0,
          "warmup_ms": 1.0, "device_free_before_load_bytes": 40 * 2**30, "device_free_after_load_bytes": 36 * 2**30,
          "device": PROBE})
    worker = Worker()
    jobs: list[tuple[dict, int]] = []
    cv = threading.Condition()
    stopping = False

    def run_jobs() -> None:
        while True:
            with cv:
                while not jobs and not stopping:
                    cv.wait()
                if stopping and not jobs:
                    return
                cmd, received = jobs.pop(0)
            worker.generate(cmd, received)

    runner = threading.Thread(target=run_jobs, daemon=True)
    runner.start()
    for line in sys.stdin:
        if not line.strip():
            continue
        cmd = json.loads(line)
        op = cmd.get("op")
        if op == "generate":
            with cv:
                jobs.append((cmd, time.monotonic_ns()))
                cv.notify()
        elif op == "count":
            emit({"ev": "count", "id": cmd["id"], "prompt_tokens": [count_tokens(i) for i in cmd["requests"]]})
        elif op == "cancel":
            with worker.lock:
                reqs = worker.cancelled.setdefault(cmd["id"], set())
                reqs.update([cmd["req"]] if "req" in cmd else range(16))
        elif op == "shutdown":
            break
    with cv:
        stopping = True
        cv.notify()
    runner.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
