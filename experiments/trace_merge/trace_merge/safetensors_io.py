"""Bounded-memory safetensors reading and a deterministic streaming writer.

The merge never holds a whole checkpoint (or a whole embedding matrix) in memory: tensors are read in row slices from
the parents and written slice by slice into a pre-computed safetensors layout. The writer produces the standard format
(8-byte little-endian header length, JSON header padded with spaces to an 8-byte boundary, then one contiguous byte
buffer with tensors in header order), so ``safetensors.safe_open`` and vLLM read it like any other shard.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
                "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
                "BOOL": torch.bool}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path  # resolved file to read from
    begin: int  # absolute byte offset of the tensor data in its file
    end: int
    shard: str = ""  # shard file name as it appears in the checkpoint directory (may be a symlink)

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def read_header(path: Path) -> tuple[dict[str, dict], dict[str, str] | None, int]:
    """Return (tensor entries, __metadata__, data start offset) of one safetensors file."""

    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    return header, meta, 8 + n


def index_checkpoint(model_dir: Path) -> dict[str, TensorInfo]:
    """Every tensor of a (possibly sharded) safetensors checkpoint; duplicate names across shards fail."""

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{model_dir}: no safetensors files")
    out: dict[str, TensorInfo] = {}
    for path in files:
        header, _, start = read_header(path)
        for name, spec in header.items():
            if name in out:
                raise ValueError(f"{model_dir}: tensor {name} appears in more than one shard")
            dtype = str(spec["dtype"])
            shape = tuple(int(x) for x in spec["shape"])
            b, e = (int(x) for x in spec["data_offsets"])
            info = TensorInfo(name, dtype, shape, path.resolve(), start + b, start + e, path.name)
            if dtype not in DTYPE_BYTES:
                raise ValueError(f"{path}: unsupported dtype {dtype} for {name}")
            if info.nbytes != info.numel * DTYPE_BYTES[dtype]:
                raise ValueError(f"{path}: {name} byte range does not match its shape/dtype")
            out[name] = info
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
        if set(weight_map) != set(out):
            missing = sorted(set(weight_map) - set(out))[:5]
            extra = sorted(set(out) - set(weight_map))[:5]
            raise ValueError(f"{model_dir}: index weight_map disagrees with shard headers (missing {missing}, extra {extra})")
        for name, shard in weight_map.items():
            if out[name].shard != Path(shard).name:
                raise ValueError(f"{model_dir}: index puts {name} in {shard}, headers in {out[name].shard}")
    return out


def iter_row_slices(info: TensorInfo, max_elements: int) -> Iterator[tuple[int, int]]:
    """(row_start, row_stop) slices along dim 0 holding at most ``max_elements`` elements (at least one row)."""

    if not info.shape:
        yield 0, 1
        return
    rows = info.shape[0]
    per_row = max(1, info.numel // max(rows, 1))
    step = max(1, max_elements // per_row)
    for r in range(0, rows, step):
        yield r, min(rows, r + step)


def read_rows(info: TensorInfo, r0: int, r1: int) -> torch.Tensor:
    """Rows [r0, r1) of a tensor as a torch tensor of its stored dtype (reads only that byte range)."""

    if not info.shape:
        r0, r1 = 0, 1
        per_row = 1
        shape: tuple[int, ...] = ()
    else:
        per_row = info.numel // info.shape[0] if info.shape[0] else 0
        shape = (r1 - r0, *info.shape[1:])
    esize = DTYPE_BYTES[info.dtype]
    off = info.begin + r0 * per_row * esize
    n = (r1 - r0) * per_row * esize
    with open(info.file, "rb") as f:
        f.seek(off)
        buf = f.read(n)
    if len(buf) != n:
        raise IOError(f"{info.file}: short read for {info.name}")
    t = torch.frombuffer(bytearray(buf), dtype=TORCH_DTYPES[info.dtype])
    return t.reshape(shape) if shape else t.reshape(())


class StreamingShardWriter:
    """Write one safetensors shard whose layout is fixed up front; tensors arrive in ``order``, slice by slice.

    Tensors are laid out by descending element size (a stable sort, so a single-dtype shard keeps the given order). That
    puts every tensor at an offset aligned to its dtype without padding (the format forbids holes), as ``save_file`` does.
    """

    def __init__(self, path: Path, tensors: list[tuple[str, str, tuple[int, ...]]], metadata: dict[str, str] | None):
        self.path = path
        tensors = sorted(tensors, key=lambda t: -DTYPE_BYTES[t[1]])
        self.order = [name for name, _, _ in tensors]
        header: dict[str, object] = {}
        offset = 0
        self.sizes: dict[str, int] = {}
        for name, dtype, shape in tensors:
            nbytes = (math.prod(shape) if shape else 1) * DTYPE_BYTES[dtype]
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
            self.sizes[name] = nbytes
            offset += nbytes
        if metadata is not None:
            header["__metadata__"] = dict(metadata)
        raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
        raw += b" " * ((8 - len(raw) % 8) % 8)
        self.header_bytes = struct.pack("<Q", len(raw)) + raw
        self.total = len(self.header_bytes) + offset
        self._f = open(path, "wb")
        self._f.write(self.header_bytes)
        self._current: str | None = None
        self._written = 0
        self._next = 0

    def begin(self, name: str) -> None:
        if self._current is not None:
            raise RuntimeError(f"tensor {self._current} not finished")
        if self._next >= len(self.order) or self.order[self._next] != name:
            raise RuntimeError(f"out-of-order tensor {name}; expected {self.order[self._next] if self._next < len(self.order) else None}")
        self._current, self._written = name, 0

    def write(self, data: bytes) -> None:
        if self._current is None:
            raise RuntimeError("write outside a tensor")
        self._f.write(data)
        self._written += len(data)
        if self._written > self.sizes[self._current]:
            raise RuntimeError(f"{self._current}: wrote more bytes than its layout")

    def end(self) -> None:
        if self._current is None or self._written != self.sizes[self._current]:
            raise RuntimeError(f"{self._current}: wrote {self._written} of {self.sizes.get(self._current or '', 0)} bytes")
        self._current = None
        self._next += 1

    def close(self) -> None:
        if self._next != len(self.order):
            raise RuntimeError(f"{self.path}: only {self._next} of {len(self.order)} tensors written")
        self._f.flush()
        self._f.close()
        if self.path.stat().st_size != self.total:
            raise RuntimeError(f"{self.path}: size {self.path.stat().st_size} != layout {self.total}")


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a contiguous CPU tensor (bf16 via an int16 view)."""

    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()
