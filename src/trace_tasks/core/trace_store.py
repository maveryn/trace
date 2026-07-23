"""Sidecar trace shard writer/reader utilities."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import zstandard as zstd

from .canonical import canonical_json_bytes
from .hash_utils import blake3_hex
from .types import TraceRef


class TraceShardWriter:
    """Append-only writer for one compressed JSONL trace shard."""

    def __init__(self, root: Path, shard_id: str = "trace_shard_0001.jsonl.zst") -> None:
        self.root = root
        self.shard_id = shard_id
        self.path = self.root / "traces" / self.shard_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._line_index = 0
        self._fh = self.path.open("wb")
        self._compressor = zstd.ZstdCompressor(level=9).stream_writer(self._fh)

    def append(self, record: Dict[str, Any]) -> TraceRef:
        """Write one trace record and return the corresponding trace ref."""
        ref = self.preview_ref(record)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self._compressor.write(line.encode("utf-8"))
        self._compressor.write(b"\n")

        self._line_index += 1
        return ref

    def preview_ref(self, record: Dict[str, Any]) -> TraceRef:
        """Return the trace ref that would be assigned to the next appended record."""
        canonical = canonical_json_bytes(record)
        record_hash = blake3_hex(canonical)
        return TraceRef(
            shard_id=self.shard_id,
            line_index=self._line_index,
            trace_record_hash=record_hash,
        )

    def close(self) -> None:
        """Flush and close the underlying compressed stream."""
        if self._compressor is not None:
            self._compressor.flush(zstd.FLUSH_FRAME)
            self._compressor.close()
            self._compressor = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TraceShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_trace_shard(path: str | Path) -> List[Dict[str, Any]]:
    """Read and parse all JSONL records from a compressed trace shard."""
    in_path = Path(path)
    compressed = in_path.read_bytes()
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
        raw = reader.read()
    lines = [line for line in raw.splitlines() if line.strip()]
    return [json.loads(line.decode("utf-8")) for line in lines]
