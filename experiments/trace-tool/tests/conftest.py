from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from trace_tool.data import TraceRow

REPO = Path(__file__).resolve().parents[1]


def make_image(width: int = 600, height: int = 400, text: str | None = None) -> Image.Image:
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 150, 120], fill=(200, 30, 30))
    draw.ellipse([400, 250, 460, 310], fill=(30, 30, 200))
    if text:
        draw.text((60, 200), text, fill=(0, 0, 0), font_size=36)
    return image


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_row(answer=3, answer_type="integer", prompt=None, image=None, row_index=0) -> TraceRow:
    image = image or make_image()
    return TraceRow(
        row_index=row_index,
        instance_id=f"synthetic-{row_index}",
        domain="charts",
        task="task_synthetic",
        scene_id="scene",
        query_id="q",
        prompt=prompt or '<image>How many shapes are there?\nExample JSON:\n{"answer":2}',
        image_bytes=[png_bytes(image)],
        image_sizes=[image.size],
        answer_gt={"type": answer_type, "value": answer},
        annotation_gt={"type": "bbox_set", "value": [[50, 50, 150, 120], [400, 250, 460, 310]]},
        reward_contract={
            "annotation": {"id": "bbox_set_soft_iou_v0", "type": "bbox_set"},
            "answer": {"id": "answer_exact_match_v0", "type": answer_type},
            "reward_contract_version": "v0",
        },
    )


@pytest.fixture
def row() -> TraceRow:
    return make_row()


def validation_parquet() -> str | None:
    """Path to a local copy of the TRACE validation parquet, if one is available."""
    path = os.environ.get("TRACE_VALIDATION_PARQUET")
    return path if path and Path(path).exists() else None
