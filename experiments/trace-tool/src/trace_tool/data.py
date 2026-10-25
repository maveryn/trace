"""Load TRACE splits from the Hugging Face Hub (or a local parquet)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

log = logging.getLogger(__name__)

REPO_ID = "maveryn/trace"
# `dataset-v1` resolves to this immutable commit; the same pin as TRACE's
# rlvr/evaluation/trace_validation/prepare_dataset.py.
REVISION = "4e5b54361360296a855542b40cfd8b7f81b355fe"
SPLIT_FILES = {
    "validation": "data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet",
}
SPLIT_SHA256 = {
    "validation": "051d1441b3f65f291841384962f16d6b5063f236072805f6c330f49afc02c4d1",
}
TRAIN_FILES = [
    f"data/train/trace_rlvr_train_64000_all1000_seed42-{i:05d}-of-00016.parquet" for i in range(16)
]

COLUMNS = [
    "images",
    "image_sizes",
    "prompt_answer",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
]


@dataclass
class TraceRow:
    row_index: int
    instance_id: str
    domain: str
    task: str
    scene_id: str
    query_id: str
    prompt: str
    image_bytes: list[bytes]
    image_sizes: list[tuple[int, int]]
    answer_gt: dict[str, Any]
    annotation_gt: dict[str, Any]
    reward_contract: dict[str, Any]

    @property
    def answer_type(self) -> str:
        return str(self.answer_gt["type"])

    def images(self) -> list[Image.Image]:
        out = []
        for raw in self.image_bytes:
            image = Image.open(io.BytesIO(raw))
            image.load()
            out.append(to_rgb(image))
        return out


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB conversion matching vLLM's ``convert_image_mode``: alpha is composited on white.

    TRACE's protocol sends the raw PNG and lets vLLM convert it; 53 validation
    images are RGBA (7 with real transparency), so a plain ``convert("RGB")``
    would show the model different pixels.
    """
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        converted = Image.new("RGB", rgba.size, (255, 255, 255))
        converted.paste(rgba, mask=rgba.split()[3])
        return converted
    return image.convert("RGB")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_split_path(split: str, local_path: str | None = None, revision: str = REVISION) -> Path:
    if local_path:
        return Path(local_path)
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)} (training shards: use rl/prepare_verl_data.py)")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, SPLIT_FILES[split], repo_type="dataset", revision=revision))


def load_split(
    split: str = "validation",
    *,
    local_path: str | None = None,
    revision: str = REVISION,
    verify_sha256: bool = True,
) -> list[TraceRow]:
    path = resolve_split_path(split, local_path, revision)
    if verify_sha256 and split in SPLIT_SHA256 and revision == REVISION:
        actual = file_sha256(path)
        if actual != SPLIT_SHA256[split]:
            message = f"{path} sha256 {actual} != pinned {SPLIT_SHA256[split]}"
            if local_path is None:
                raise RuntimeError(message)
            log.warning("%s (local --dataset-path; continuing)", message)
    return rows_from_parquet(path)


def rows_from_parquet(path: str | Path) -> list[TraceRow]:
    table = pq.read_table(str(path), columns=COLUMNS)
    rows = []
    for i, record in enumerate(table.to_pylist()):
        rows.append(
            TraceRow(
                row_index=i,
                instance_id=record["instance_id"],
                domain=record["domain"],
                task=record["task"],
                scene_id=record["scene_id"],
                query_id=record["query_id"],
                prompt=record["prompt_answer"],
                image_bytes=[img["bytes"] for img in record["images"]],
                image_sizes=[(int(s["width"]), int(s["height"])) for s in record["image_sizes"]],
                answer_gt=json.loads(record["answer_gt"]),
                annotation_gt=json.loads(record["annotation_gt"]),
                reward_contract=json.loads(record["reward_contract"]),
            )
        )
    return rows


def select_rows(
    rows: list[TraceRow],
    *,
    per_task: int | None = None,
    domains: list[str] | None = None,
    limit: int | None = None,
    shard: tuple[int, int] | None = None,
) -> list[TraceRow]:
    """Deterministic subsets: ``per_task`` keeps the first K rows of each task in file order."""
    out = rows
    if domains:
        wanted = set(domains)
        out = [r for r in out if r.domain in wanted]
    if per_task is not None:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for row in out:
            if seen[row.task] < per_task:
                kept.append(row)
                seen[row.task] += 1
        out = kept
    if shard is not None:
        index, count = shard
        out = [r for i, r in enumerate(out) if i % count == index]
    if limit is not None:
        out = out[:limit]
    return out
