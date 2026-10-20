from __future__ import annotations

import hashlib
import json
import math
import os
import random
import stat
import threading
import urllib.parse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from edgeserve import records
from edgeserve.records import OutputDir, safe_filename, summarize


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def strict_umask() -> Iterator[None]:
    """A umask that strips owner write/exec bits too: only explicit chmods can produce 0700/0600."""
    old = os.umask(0o277)
    try:
        yield
    finally:
        os.umask(old)


# --- safe_filename -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("custom_id", "expected"),
    [
        ("seg-001", "seg-001"),
        ("A.b_c-9", "A.b_c-9"),
        ("a b/c", "a%20b%2Fc"),
        ("100%", "100%25"),
        ("é", "%C3%A9"),
        ("x\x00y\n", "x%00y%0A"),
        ("~*?:", "%7E%2A%3F%3A"),
        (".hidden", "%2Ehidden"),
        (".", "%2E"),
        ("..", "%2E."),
        ("a..b.", "a..b."),
    ],
)
def test_safe_filename_encodes_unsafe_bytes(custom_id: str, expected: str) -> None:
    assert safe_filename(custom_id) == expected


@pytest.mark.parametrize("custom_id", ["seg-001", "a/b\\c", "%41", ".x", "..", "日本語 🚀", "\t\r\n", "a" * 66])
def test_safe_filename_is_reversible(custom_id: str) -> None:
    stem = safe_filename(custom_id)
    assert urllib.parse.unquote(stem, errors="strict") == custom_id
    assert not stem.startswith(".") and "/" not in stem


def test_safe_filename_hashes_long_stems() -> None:
    assert safe_filename("a" * 200) == "a" * 200  # exactly at the limit: kept
    long_id = "a" * 201
    assert safe_filename(long_id) == "a" * 183 + "-" + hashlib.sha256(long_id.encode()).hexdigest()[:16]
    # The limit applies to the encoded stem: 66 spaces encode to 198 chars, 67 to 201.
    assert safe_filename(" " * 66) == "%20" * 66
    spaced = safe_filename(" " * 67)
    assert len(spaced) == 200 and spaced.startswith("%20" * 61 + "-")
    # Long IDs sharing a prefix stay distinct, and the file name fits NAME_MAX.
    assert safe_filename("b" * 300 + "1") != safe_filename("b" * 300 + "2")
    assert len(f"{safe_filename('ü' * 500)}.json".encode()) <= 255


def test_safe_filename_rejects_bad_ids() -> None:
    with pytest.raises(ValueError):
        safe_filename("")
    with pytest.raises(ValueError):
        safe_filename("bad \ud800 surrogate")
    with pytest.raises(TypeError):
        safe_filename(None)  # type: ignore[arg-type]


# --- OutputDir: folders and records ------------------------------------------------------------


def test_output_dir_creates_private_folders(tmp_path: Path, strict_umask: None) -> None:
    root = tmp_path / "a" / "b"
    out = OutputDir(root)
    assert out.root == root
    assert _mode(tmp_path / "a") == 0o700
    assert _mode(root) == 0o700
    assert _mode(root / "records") == 0o700


def test_output_dir_leaves_existing_folder_mode(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    os.chmod(root, 0o750)
    OutputDir(root)
    OutputDir(root)  # reopening an existing output folder is fine
    assert _mode(root) == 0o750
    assert _mode(root / "records") == 0o700


def test_output_dir_on_a_file_fails(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(OSError):
        OutputDir(target)


def test_write_record_is_private_utf8_and_indented(tmp_path: Path, strict_umask: None) -> None:
    out = OutputDir(tmp_path / "out")
    record = {"custom_id": "seg/001", "status": "completed", "output": "naïve ✓", "steps": {"itl_ms": [None, 1.5]}}
    path = out.write_record(record)
    assert path == out.record_path("seg/001") == tmp_path / "out" / "records" / "seg%2F001.json"
    assert path.read_text("utf-8") == json.dumps(record, indent=1, ensure_ascii=False) + "\n"
    assert _mode(path) == 0o600
    assert out.read_record("seg/001") == record

    out.write_record({**record, "status": "failed"})
    assert out.read_record("seg/001")["status"] == "failed"
    assert [p.name for p in (tmp_path / "out" / "records").iterdir()] == ["seg%2F001.json"]  # no temp files


def test_write_record_with_unpaired_surrogate_round_trips(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    record = {"custom_id": "s1", "output": "broken \ud800 text"}
    out.write_record(record)
    assert out.read_record("s1") == record


@pytest.mark.parametrize("record", [{}, {"custom_id": ""}, {"custom_id": 5}, {"custom_id": None}])
def test_write_record_requires_a_custom_id(tmp_path: Path, record: dict) -> None:
    with pytest.raises(ValueError):
        OutputDir(tmp_path).write_record(record)


def test_failed_write_keeps_the_previous_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = OutputDir(tmp_path)
    out.write_record({"custom_id": "a", "status": "completed"})

    def broken_fsync(fd: int) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(records.os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        out.write_record({"custom_id": "a", "status": "failed"})
    monkeypatch.undo()
    assert out.read_record("a") == {"custom_id": "a", "status": "completed"}
    assert [p.name for p in (tmp_path / "records").iterdir()] == ["a.json"]


def test_read_record_returns_none_when_absent_or_unreadable(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    assert out.read_record("missing") is None
    assert out.read_record("") is None
    out.record_path("corrupt").write_text('{"custom_id": "corrupt", ')
    out.record_path("array").write_text("[1, 2]")
    out.record_path("binary").write_bytes(b"\xff\xfe\x00\x81garbage")
    out.record_path("other").write_text(json.dumps({"custom_id": "someone-else", "status": "completed"}))
    out.record_path("folder").mkdir()
    for custom_id in ("corrupt", "array", "binary", "other", "folder"):
        assert out.read_record(custom_id) is None, custom_id


def test_completed_ids(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    long_id = "L" * 300
    statuses = {"done": "completed", "x/y": "completed", long_id: "completed", "bad": "failed", "stop": "cancelled"}
    for custom_id, status in statuses.items():
        out.write_record({"custom_id": custom_id, "status": status})
    folder = tmp_path / "records"
    (folder / "corrupt.json").write_text("{")
    (folder / "renamed.json").write_text(json.dumps({"custom_id": "elsewhere", "status": "completed"}))
    (folder / "anonymous.json").write_text(json.dumps({"status": "completed"}))
    (folder / "notes.txt").write_text(json.dumps({"custom_id": "notes", "status": "completed"}))
    (folder / "dir.json").mkdir()
    assert out.completed_ids() == {"done", "x/y", long_id}


def test_completed_ids_without_records_folder(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    (tmp_path / "records").rmdir()
    assert out.completed_ids() == set()


def test_write_run_is_atomic_and_private(tmp_path: Path, strict_umask: None) -> None:
    out = OutputDir(tmp_path)
    out.write_run({"object": "batch_run", "n": 1})
    out.write_run({"object": "batch_run", "n": 2, "note": "ü"})
    path = tmp_path / "run.json"
    assert json.loads(path.read_text("utf-8")) == {"object": "batch_run", "n": 2, "note": "ü"}
    assert _mode(path) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records", "run.json"]


# --- OutputDir: results.jsonl ------------------------------------------------------------------


def test_append_result_writes_lines_immediately(tmp_path: Path, strict_umask: None) -> None:
    out = OutputDir(tmp_path)
    out.append_result({"custom_id": "a", "n": 1})
    out.append_result({"custom_id": "b", "text": "ü"})
    path = tmp_path / "results.jsonl"
    assert _mode(path) == 0o600
    # Visible before close(): every line is handed to the kernel when appended.
    assert [json.loads(line) for line in path.read_text("utf-8").splitlines()] == [
        {"custom_id": "a", "n": 1},
        {"custom_id": "b", "text": "ü"},
    ]
    out.close()
    out.close()  # idempotent
    out.append_result({"custom_id": "c"})  # reopens after close()
    out.close()
    assert len(path.read_text().splitlines()) == 3


def test_append_result_continues_an_existing_log(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"custom_id":"old"}\n{"custom_id":"cut sho')  # a previous run died mid-line
    out = OutputDir(tmp_path)
    out.append_result({"custom_id": "new"})
    out.close()
    assert path.read_text().split("\n") == ['{"custom_id":"old"}', '{"custom_id":"cut sho', '{"custom_id":"new"}', ""]


def test_works_without_fchmod_and_pread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows (before Python 3.13) has neither os.fchmod nor os.pread."""
    results = tmp_path / "results.jsonl"
    results.write_bytes(b'{"custom_id":"old"}\n{"custom_id":"cut sho')  # a previous run died mid-line
    monkeypatch.delattr(os, "fchmod")
    monkeypatch.delattr(os, "pread")
    out = OutputDir(tmp_path)
    path = out.write_record({"custom_id": "a", "status": "completed"})
    assert _mode(path) == 0o600 and out.read_record("a") == {"custom_id": "a", "status": "completed"}
    out.append_result({"custom_id": "new"})
    out.append_result({"custom_id": "newer"})
    out.close()
    lines = results.read_text().split("\n")
    assert lines == ['{"custom_id":"old"}', '{"custom_id":"cut sho', '{"custom_id":"new"}', '{"custom_id":"newer"}', ""]


def test_new_files_are_private_without_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict_umask: None
) -> None:
    monkeypatch.delattr(os, "fchmod")
    out = OutputDir(tmp_path / "out")
    assert _mode(out.write_record({"custom_id": "a"})) == 0o600
    out.append_result({"custom_id": "a"})
    out.write_run({"object": "batch_run"})
    out.close()
    assert _mode(tmp_path / "out" / "results.jsonl") == 0o600
    assert _mode(tmp_path / "out" / "run.json") == 0o600


def test_append_result_is_thread_safe(tmp_path: Path) -> None:
    out = OutputDir(tmp_path)
    payload = "x" * 5000

    def work(thread: int) -> None:
        for i in range(100):
            out.append_result({"thread": thread, "i": i, "payload": payload})

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    out.close()
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert sorted((p["thread"], p["i"]) for p in parsed) == [(t, i) for t in range(8) for i in range(100)]
    assert all(p["payload"] == payload for p in parsed)


def test_append_result_fsyncs_at_most_every_5_s_and_on_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(records, "_monotonic", lambda: clock[0])
    file_syncs: list[int] = []
    real_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):  # ignore directory syncs
            file_syncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(records.os, "fsync", counting_fsync)
    out = OutputDir(tmp_path)
    for now, synced in [(1000.0, 0), (1003.0, 0), (1005.0, 1), (1006.0, 1), (1009.9, 1), (1010.0, 2)]:
        clock[0] = now
        out.append_result({"t": now})
        assert len(file_syncs) == synced, now
    out.close()
    assert len(file_syncs) == 3
    out.close()
    assert len(file_syncs) == 3


# --- summarize ---------------------------------------------------------------------------------


def _record(
    status: str = "completed",
    arrival: int | None = None,
    finished: int | None = None,
    usage: dict | None = None,
    itl: list | None = None,
    **metrics: object,
) -> dict:
    record: dict = {"custom_id": "x", "status": status, "metrics": metrics}
    if arrival is not None or finished is not None:
        record["timestamps_ns"] = {"arrival": arrival, "finished": finished}
    if usage is not None:
        record["usage"] = usage
    if itl is not None:
        record["steps"] = {"itl_ms": itl}
    return record


def test_summarize_empty() -> None:
    summary = summarize([])
    assert summary == {
        "request_counts": {"total": 0, "completed": 0, "failed": 0, "cancelled": 0, "rejected": 0},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "window_s": None,
        "busy_s": None,
        "request_throughput_per_s": None,
        "output_tokens_per_s": None,
        "total_tokens_per_s": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "queue_ms": None,
        "decode_tokens_per_s": None,
        "itl_ms": None,
        "energy_j": None,
        "output_tokens_per_j": None,
    }


def test_summarize_mixed_records() -> None:
    a = _record(
        arrival=1_000_000_000,
        finished=3_000_000_000,
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        itl=[None, 40.0, 42.0],
        ttft_ms=200.0,
        tpot_ms=40.0,
        e2e_ms=2000.0,
        queue_ms=10.0,
        decode_tokens_per_s=25.0,
        energy_j=10.0,
    )
    b = _record(
        arrival=2_000_000_000,
        finished=5_000_000_000,
        usage={"prompt_tokens": 200, "completion_tokens": 100},
        itl=[None, 30.0, None, 31.0],
        ttft_ms=400.0,
        tpot_ms=30.0,
        e2e_ms=3000.0,
        queue_ms=None,
        decode_tokens_per_s=35.0,
    )
    failed = _record(
        "failed",
        arrival=1_500_000_000,
        finished=2_000_000_000,
        usage={"prompt_tokens": 50, "completion_tokens": 5},
        itl=[None, 999.0],
        ttft_ms=999.0,
        energy_j=1.0,
    )
    cancelled = {"custom_id": "c", "status": "cancelled"}
    unknown = {"custom_id": "u", "status": "weird", "usage": {"prompt_tokens": 1}}
    summary = summarize(iter([a, b, failed, cancelled, unknown, "not a record"]))  # type: ignore[list-item]

    assert summary["request_counts"] == {"total": 5, "completed": 2, "failed": 1, "cancelled": 1, "rejected": 0}
    assert summary["usage"] == {"prompt_tokens": 351, "completion_tokens": 155, "total_tokens": 506}
    assert summary["window_s"] == 4.0  # first arrival 1.0 s (a), last finished 5.0 s (b)
    assert summary["busy_s"] == 4.0  # the requests overlap: no idle time
    assert summary["request_throughput_per_s"] == 0.5  # completed records only
    assert summary["output_tokens_per_s"] == 155 / 4
    assert summary["total_tokens_per_s"] == 506 / 4
    # Latency statistics come from completed records only (the failed record's 999 is ignored).
    expected_ttft = {"mean": 300.0, "p50": 300.0, "p90": 380.0, "p99": 398.0, "min": 200.0, "max": 400.0}
    assert summary["ttft_ms"] == pytest.approx(expected_ttft)
    assert summary["queue_ms"] == {"mean": 10.0, "p50": 10.0, "p90": 10.0, "p99": 10.0, "min": 10.0, "max": 10.0}
    # Pooled gaps [30, 31, 40, 42]: nulls and the failed record's gaps are skipped.
    expected_itl = {"mean": 35.75, "p50": 35.5, "p90": 41.4, "p99": 41.94, "min": 30.0, "max": 42.0}
    assert summary["itl_ms"] == pytest.approx(expected_itl)
    # Energy over every record that reports it; tokens per joule only over those records.
    assert summary["energy_j"] == 11.0
    assert summary["output_tokens_per_j"] == pytest.approx(55 / 11.0)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 10, 101, 1000])
def test_summarize_percentiles_match_numpy(size: int) -> None:
    rng = random.Random(size)
    values = [rng.lognormvariate(3.0, 1.0) for _ in range(size)]
    stats = summarize([_record(e2e_ms=value) for value in values])["e2e_ms"]
    for name, q in (("p50", 50), ("p90", 90), ("p99", 99)):
        assert stats[name] == pytest.approx(float(np.percentile(values, q)), rel=1e-12, abs=1e-12), name
    assert stats["mean"] == pytest.approx(float(np.mean(values)), rel=1e-12)
    assert stats["min"] == min(values) and stats["max"] == max(values)


def test_summarize_ignores_unusable_values() -> None:
    bad_values = [None, True, "12", math.nan, math.inf, [1], {"v": 1}]
    recs = [_record(ttft_ms=value, energy_j=value, usage={"prompt_tokens": value}) for value in bad_values]
    malformed = {"metrics": [1, 2], "usage": "lots", "timestamps_ns": None, "steps": {"itl_ms": "fast"}}
    recs.append({"custom_id": "m", "status": "completed", **malformed})
    recs.append(_record(itl=[None, "1", True, math.nan, 5]))
    summary = summarize(recs)
    assert summary["request_counts"]["total"] == len(bad_values) + 2
    assert summary["ttft_ms"] is None
    assert summary["energy_j"] is None and summary["output_tokens_per_j"] is None
    assert summary["usage"]["prompt_tokens"] == 0
    assert summary["itl_ms"] == {"mean": 5.0, "p50": 5.0, "p90": 5.0, "p99": 5.0, "min": 5.0, "max": 5.0}
    json.dumps(summary, allow_nan=False)


def test_summarize_ignores_integers_beyond_float_range() -> None:
    huge = 10**400  # valid JSON, but float(huge) overflows
    record = _record(
        arrival=0,
        finished=huge,
        usage={"prompt_tokens": huge, "completion_tokens": 7},
        itl=[None, huge, 3.0],
        ttft_ms=huge,
        e2e_ms=1000.0,
        energy_j=huge,
    )
    summary = summarize([record, _record(ttft_ms=5.0)])
    assert summary["ttft_ms"] == {"mean": 5.0, "p50": 5.0, "p90": 5.0, "p99": 5.0, "min": 5.0, "max": 5.0}
    assert summary["usage"] == {"prompt_tokens": 0, "completion_tokens": 7, "total_tokens": 7}
    assert summary["itl_ms"]["max"] == 3.0
    assert summary["energy_j"] is None and summary["window_s"] is None
    assert summary["e2e_ms"]["mean"] == 1000.0
    json.dumps(summary, allow_nan=False)


def test_summarize_window_edge_cases() -> None:
    # A zero-length window is known but gives no rates.
    same = summarize([_record(arrival=5, finished=5, usage={"completion_tokens": 3})])
    assert same["window_s"] == 0.0 and same["output_tokens_per_s"] is None and same["request_throughput_per_s"] is None
    # Records with only one timestamp do not take part; an inverted window is unknown.
    partial = summarize([_record(arrival=1_000_000_000), _record(finished=9_000_000_000)])
    assert partial["window_s"] is None and partial["total_tokens_per_s"] is None
    inverted = summarize([_record(arrival=3_000_000_000, finished=1_000_000_000)])
    assert inverted["window_s"] is None
    # Large monotonic nanosecond values keep full precision.
    base = 10**18
    precise = summarize([_record(arrival=base, finished=base + 1_500_000_001)])
    assert precise["window_s"] == 1.500000001


def test_summarize_rates_leave_out_idle_time_between_runs() -> None:
    # An interrupted batch resumed later: two bursts 20 s apart. Rates count the time requests ran, not the pause.
    first = [_record(arrival=s * 10**9, finished=(s + 1) * 10**9, usage={"completion_tokens": 10}) for s in (0, 1)]
    later = [_record(arrival=s * 10**9, finished=(s + 1) * 10**9, usage={"completion_tokens": 10}) for s in (22, 23)]
    summary = summarize(first + later)
    assert summary["window_s"] == 24.0 and summary["busy_s"] == 4.0
    assert summary["request_throughput_per_s"] == 1.0 and summary["output_tokens_per_s"] == 10.0


def test_summarize_zero_energy_has_no_efficiency() -> None:
    summary = summarize([_record(energy_j=0.0, usage={"completion_tokens": 10})])
    assert summary["energy_j"] == 0.0
    assert summary["output_tokens_per_j"] is None
