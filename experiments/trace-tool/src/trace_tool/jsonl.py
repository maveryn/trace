"""Append-only JSONL files that survive a crash mid-write."""

from __future__ import annotations

import fcntl
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """All decodable records, plus the number of lines skipped (torn writes, bad bytes)."""
    records: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return records, bad
    for raw in path.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    if bad:
        print(f"warning: skipped {bad} undecodable lines in {path}", file=sys.stderr)
    return records, bad


def repair_tail(path: Path) -> None:
    """Drop a partial last line left by a crash mid-write, so appends start on a fresh line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        keep = handle.read().rfind(b"\n") + 1
        handle.truncate(keep)
    print(f"warning: dropped a partial last line from {path}", file=sys.stderr)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Refuse to run if another process holds the lock (two writers on one output file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{path} is locked: another process is writing to {path.parent}") from None
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
