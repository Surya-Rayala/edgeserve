"""Output folder: one structured record per request ID, an OpenAI-Batch-style results log and a run summary.

Layout::

    <root>/records/<safe_filename(custom_id)>.json   full record of the latest attempt for that ID
    <root>/results.jsonl                             one line per attempt (OpenAI BatchRequestOutput shape)
    <root>/run.json                                  aggregate summary (see summarize())

Records contain prompts and outputs, which can be sensitive, so directories are
created with mode 0700 and files with mode 0600. Writes are atomic (temporary file in the same
directory, flush + fsync, os.replace) so a crash never leaves a truncated record.

A record is a dict produced by edgeserve.metrics (schema "edgeserve.record" v1); the fields used
here are: custom_id, status ("completed" | "failed" | "cancelled" | "rejected"), timestamps_ns.arrival,
timestamps_ns.finished, usage.prompt_tokens, usage.completion_tokens, metrics.ttft_ms,
metrics.tpot_ms, metrics.e2e_ms, metrics.queue_ms, metrics.decode_tokens_per_s, metrics.energy_j
and steps.itl_ms (per-token gaps; first entry null). Any of them may be missing or null.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path

_SAFE_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_MAX_STEM_CHARS = 200
_HASHED_PREFIX_CHARS = 183
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_O_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation
_FSYNC_INTERVAL_S = 5.0
_STATUSES = ("completed", "failed", "cancelled", "rejected")
_LATENCY_KEYS = ("ttft_ms", "tpot_ms", "e2e_ms", "queue_ms", "decode_tokens_per_s")
_PERCENTILES = (("p50", 50.0), ("p90", 90.0), ("p99", 99.0))

_monotonic = time.monotonic  # module-level so tests can drive the fsync throttle


def safe_filename(custom_id: str) -> str:
    """Reversible, collision-free file stem for an ID.

    Percent-encodes (``%XX``, upper-case hex, over the UTF-8 bytes) every byte outside ``[A-Za-z0-9._-]``, and
    also a leading ``.`` (so the result is never hidden, "." or ".."). If the encoded stem is longer than 200
    characters it becomes ``<first 183 chars>-<first 16 hex chars of sha256(custom_id)>``. Raises ValueError for
    an empty ID.
    """
    if not isinstance(custom_id, str):
        raise TypeError(f"custom_id must be a str, not {type(custom_id).__name__}")
    if not custom_id:
        raise ValueError("custom_id must not be empty")
    try:
        data = custom_id.encode("utf-8")
    except UnicodeEncodeError as exc:  # unpaired surrogates have no UTF-8 form
        raise ValueError(f"custom_id is not valid Unicode text: {exc}") from None
    stem = "".join(
        chr(byte) if byte in _SAFE_BYTES and not (index == 0 and byte == 0x2E) else f"%{byte:02X}"
        for index, byte in enumerate(data)
    )
    if len(stem) > _MAX_STEM_CHARS:
        stem = f"{stem[:_HASHED_PREFIX_CHARS]}-{hashlib.sha256(data).hexdigest()[:16]}"
    return stem


class OutputDir:
    """Thread-safe writer for one output folder (creates it if needed)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._records_dir = self._root / "records"
        _make_private_dirs(self._records_dir)
        self._lock = threading.Lock()
        self._results_fd: int | None = None
        self._last_fsync = 0.0

    @property
    def root(self) -> Path:
        return self._root

    def record_path(self, custom_id: str) -> Path:
        return self._records_dir / f"{safe_filename(custom_id)}.json"

    def write_record(self, record: dict) -> Path:
        """Atomically write ``record`` to records/<safe_filename(record['custom_id'])>.json (indent=1, UTF-8)."""
        custom_id = record.get("custom_id") if isinstance(record, dict) else None
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError("a record needs a non-empty string 'custom_id'")
        path = self.record_path(custom_id)
        _atomic_write(path, _encode(record, indent=1) + b"\n")
        return path

    def read_record(self, custom_id: str) -> dict | None:
        """The stored record for an ID, or None if absent or unreadable."""
        try:
            path = self.record_path(custom_id)
        except (TypeError, ValueError):
            return None
        record = _load_json(path)
        # The ID check also rules out a (deliberately crafted) collision with a hashed long-ID stem.
        if isinstance(record, dict) and record.get("custom_id") == custom_id:
            return record
        return None

    def append_result(self, line: dict) -> None:
        """Append one JSON line to results.jsonl and flush (fsync at most every 5 s and on close())."""
        data = _encode(line) + b"\n"
        with self._lock:
            if self._results_fd is None:
                self._results_fd = self._open_results()
                self._last_fsync = _monotonic()
            try:
                _write_all(self._results_fd, data)  # unbuffered: the line reaches the kernel at once
            except OSError:
                # Reopen next time, which also terminates the partial line this write may have left.
                with contextlib.suppress(OSError):
                    os.close(self._results_fd)
                self._results_fd = None
                raise
            now = _monotonic()
            if now - self._last_fsync >= _FSYNC_INTERVAL_S:
                os.fsync(self._results_fd)
                self._last_fsync = now

    def completed_ids(self) -> set[str]:
        """IDs whose stored record has status "completed" (for resuming batches; ignores corrupt files)."""
        try:
            with os.scandir(self._records_dir) as entries:
                names = [e.name for e in entries if e.name.endswith(".json") and not e.name.startswith(".")]
        except OSError:
            return set()
        ids: set[str] = set()
        for name in names:
            record = _load_json(self._records_dir / name)
            if not isinstance(record, dict) or record.get("status") != "completed":
                continue
            custom_id = record.get("custom_id")
            # Only files read_record() would find for that ID (not renamed or copied ones).
            if isinstance(custom_id, str) and _stem_or_none(custom_id) == name[: -len(".json")]:
                ids.add(custom_id)
        return ids

    def write_run(self, run: dict) -> None:
        """Atomically write run.json."""
        _atomic_write(self._root / "run.json", _encode(run, indent=1) + b"\n")

    def close(self) -> None:
        """Flush and fsync results.jsonl (idempotent)."""
        with self._lock:
            fd, self._results_fd = self._results_fd, None
            if fd is None:
                return
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _open_results(self) -> int:
        path = self._root / "results.jsonl"
        flags = os.O_RDWR | os.O_APPEND | _O_BINARY
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        except FileExistsError:
            fd = os.open(path, flags)
            try:
                size = os.fstat(fd).st_size
                if size and _byte_at(fd, size - 1) != b"\n":
                    _write_all(fd, b"\n")  # a previous run died mid-line: keep the new lines parseable
            except BaseException:
                os.close(fd)
                raise
            return fd
        try:
            _chmod_private(fd, path)  # the creation mode is masked by the umask
        except BaseException:
            os.close(fd)
            raise
        _fsync_dir(self._root)
        return fd


def summarize(records: Iterable[dict]) -> dict:
    """Aggregate statistics over records.

    Returns::

        {"request_counts": {"total", "completed", "failed", "cancelled", "rejected"},
         "usage": {"prompt_tokens", "completion_tokens", "total_tokens"},
         "window_s": last finished - first arrival (seconds; None if unknown),
         "busy_s": time with at least one request in progress (seconds; None if unknown): window_s without the
             idle gaps, e.g. between an interrupted run and its resumption,
         "request_throughput_per_s", "output_tokens_per_s", "total_tokens_per_s",   # over busy_s; None if unknown
         "ttft_ms", "tpot_ms", "e2e_ms", "queue_ms", "decode_tokens_per_s":          # over completed records
             {"mean", "p50", "p90", "p99", "min", "max"} or None when no values,
         "itl_ms": same stats pooled over every token gap of every completed record (steps.itl_ms, nulls skipped),
         "energy_j": sum (None if no record has energy), "output_tokens_per_j": completion tokens / energy_j}

    Percentiles use linear interpolation between closest ranks (numpy's default "linear" method).

    Usage and energy are summed over records of every status (partial work of failed or cancelled requests
    counts); request_throughput_per_s counts completed records only. output_tokens_per_j divides the
    completion tokens of the records that report energy by their energy, so records without a measurement
    do not inflate it. Non-numeric, boolean and non-finite values (and integers beyond the float range) are
    ignored like None.
    """
    counts = {"total": 0, "completed": 0, "failed": 0, "cancelled": 0, "rejected": 0}
    prompt_tokens: int | float = 0
    completion_tokens: int | float = 0
    first_arrival: int | float | None = None
    last_finished: int | float | None = None
    intervals: list[tuple[int | float, int | float]] = []
    series: dict[str, list[float]] = {key: [] for key in _LATENCY_KEYS}
    gaps: list[float] = []
    energy_j: float | None = None
    energy_tokens: int | float = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        counts["total"] += 1
        status = record.get("status")
        if status in _STATUSES:
            counts[status] += 1

        usage = _mapping(record.get("usage"))
        record_completion = _number(usage.get("completion_tokens")) or 0
        prompt_tokens += _number(usage.get("prompt_tokens")) or 0
        completion_tokens += record_completion

        timestamps = _mapping(record.get("timestamps_ns"))
        arrival, finished = _number(timestamps.get("arrival")), _number(timestamps.get("finished"))
        if arrival is not None and finished is not None:
            first_arrival = arrival if first_arrival is None else min(first_arrival, arrival)
            last_finished = finished if last_finished is None else max(last_finished, finished)
            if finished >= arrival:
                intervals.append((arrival, finished))

        metrics = _mapping(record.get("metrics"))
        energy = _number(metrics.get("energy_j"))
        if energy is not None:
            energy_j = energy if energy_j is None else energy_j + energy
            energy_tokens += record_completion

        if status == "completed":
            for key in _LATENCY_KEYS:
                value = _number(metrics.get(key))
                if value is not None:
                    series[key].append(value)
            itl = _mapping(record.get("steps")).get("itl_ms")
            if isinstance(itl, list):
                gaps.extend(value for value in map(_number, itl) if value is not None)

    window_s: float | None = None
    if first_arrival is not None and last_finished is not None and last_finished >= first_arrival:
        window_s = (last_finished - first_arrival) / 1e9  # subtract first: exact for int nanoseconds
    busy_s = _covered_ns(intervals) / 1e9 if intervals else None
    total_tokens = prompt_tokens + completion_tokens

    def per_second(amount: int | float) -> float | None:
        return amount / busy_s if busy_s else None

    summary: dict = {
        "request_counts": counts,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "window_s": window_s,
        "busy_s": busy_s,
        "request_throughput_per_s": per_second(counts["completed"]),
        "output_tokens_per_s": per_second(completion_tokens),
        "total_tokens_per_s": per_second(total_tokens),
    }
    for key in _LATENCY_KEYS:
        summary[key] = _stats(series[key])
    summary["itl_ms"] = _stats(gaps)
    summary["energy_j"] = energy_j
    summary["output_tokens_per_j"] = energy_tokens / energy_j if energy_j is not None and energy_j > 0 else None
    return summary


def _covered_ns(intervals: list[tuple[int | float, int | float]]) -> int | float:
    """Total length of the union of [start, end] intervals."""
    covered: int | float = 0
    current: tuple[int | float, int | float] | None = None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                covered += current[1] - current[0]
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    if current is not None:
        covered += current[1] - current[0]
    return covered


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    stats = {"mean": math.fsum(ordered) / len(ordered)}
    for name, q in _PERCENTILES:
        stats[name] = _percentile(ordered, q)
    stats["min"] = float(ordered[0])
    stats["max"] = float(ordered[-1])
    return stats


def _percentile(ordered: list[float], q: float) -> float:
    """numpy.percentile(..., method="linear") of already sorted values."""
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    a, b = float(ordered[lower]), float(ordered[upper])
    fraction = position - lower
    # Same two-sided form as numpy's _lerp, which keeps the result exact at both ends.
    return b - (b - a) * (1.0 - fraction) if fraction >= 0.5 else a + (b - a) * fraction


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:  # an int beyond the float range would break every average and rate
        return None


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _stem_or_none(custom_id: str) -> str | None:
    try:
        return safe_filename(custom_id)
    except (TypeError, ValueError):
        return None


def _encode(obj: object, indent: int | None = None) -> bytes:
    """UTF-8 JSON; falls back to ASCII escapes when ``obj`` holds unpaired surrogates (not encodable as UTF-8)."""
    separators = None if indent is not None else (",", ":")
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, separators=separators).encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(obj, indent=indent, separators=separators).encode("ascii")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            _chmod_private(file.fileno(), Path(tmp))
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def _chmod_private(fd: int, path: Path) -> None:
    """Mode 0600 for an open file (os.fchmod is missing on Windows before Python 3.13)."""
    if hasattr(os, "fchmod"):
        os.fchmod(fd, _FILE_MODE)
    else:
        os.chmod(path, _FILE_MODE)


def _byte_at(fd: int, offset: int) -> bytes:
    """One byte at ``offset`` (os.pread is missing on Windows)."""
    if hasattr(os, "pread"):
        return os.pread(fd, 1, offset)
    os.lseek(fd, offset, os.SEEK_SET)  # harmless with O_APPEND: writes still go to the end
    return os.read(fd, 1)


def _fsync_dir(path: Path) -> None:
    """Make a rename or creation in ``path`` durable (best effort: not every filesystem supports it)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _make_private_dirs(path: Path) -> None:
    """Create ``path`` and any missing parents with mode 0700; existing directories are left as they are."""
    missing: list[Path] = []
    current = path
    while not current.is_dir():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, _DIR_MODE)
        except FileExistsError:
            if directory.is_dir():  # created concurrently
                continue
            raise
        os.chmod(directory, _DIR_MODE)  # mkdir's mode is masked by the umask


__all__ = ["OutputDir", "safe_filename", "summarize"]
