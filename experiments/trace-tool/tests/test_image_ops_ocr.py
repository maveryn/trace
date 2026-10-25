import numpy as np
import pytest

from trace_tool.tools import build_tools
from trace_tool.tools.base import ToolContext
from trace_tool.models import QWEN3_VL, QWEN25_VL

from .conftest import make_image


def test_rotate_flip_draw():
    tools = build_tools(["visual_ops"], QWEN3_VL)
    ctx = ToolContext([make_image(600, 400)])
    rotated = tools["image_rotate_tool"]({"angle": 90, "img_idx": 0}, ctx)
    assert rotated.ok and rotated.images[0].size[1] > rotated.images[0].size[0]
    flipped = tools["image_flip_tool"]({"direction": "horizontal", "img_idx": 0}, ctx)
    assert flipped.ok
    # the red box at x=50..150 moves to the right side after a horizontal flip
    arr = np.asarray(flipped.images[0].resize((600, 400)))
    assert arr[85, 500, 0] > 150 and arr[85, 100, 0] < 255
    drawn = tools["image_draw_tool"]({"lines": [[0, 500, 1000, 500]], "points": [[500, 500]], "color": "blue", "img_idx": 0}, ctx)
    assert drawn.ok and drawn.meta == {"n_lines": 1, "n_boxes": 0, "n_points": 1}
    assert not tools["image_draw_tool"]({"img_idx": 0}, ctx).ok
    assert not tools["image_flip_tool"]({"direction": "diagonal", "img_idx": 0}, ctx).ok
    assert not tools["image_rotate_tool"]({"angle": "abc", "img_idx": 0}, ctx).ok


@pytest.fixture(scope="module")
def ocr_tool():
    pytest.importorskip("rapidocr")
    return build_tools(["ocr"], QWEN3_VL)["ocr"]


def test_ocr_reads_text_and_maps_region_boxes(ocr_tool):
    image = make_image(800, 400, text="TOTAL 1234")
    ctx = ToolContext([image])
    full = ocr_tool({"img_idx": 0}, ctx)
    assert full.ok and "1234" in full.text
    # region OCR returns boxes in the full image's frame
    region = ocr_tool({"img_idx": 0, "bbox_2d": [0, 400, 1000, 750]}, ctx)
    assert "1234" in region.text
    line = next(l for l in region.text.splitlines() if "1234" in l)
    box = [int(v) for v in line[line.rindex("[") + 1 : line.rindex("]")].split(",")]
    assert 400 <= box[1] <= box[3] <= 750
    empty = ocr_tool({"img_idx": 0, "bbox_2d": [700, 0, 1000, 100]}, ctx)
    assert empty.ok and empty.text == "No text detected."


def test_ocr_cache_hits(ocr_tool):
    calls = []
    real = ocr_tool.backend

    def counting(image):
        calls.append(image.size)
        return real(image)

    ocr_tool.backend = counting
    try:
        ctx = ToolContext([make_image(800, 400, text="CACHE 77")])
        first = ocr_tool({"img_idx": 0}, ctx)
        second = ocr_tool({"img_idx": 0}, ToolContext([make_image(800, 400, text="CACHE 77")]))
        assert first.text == second.text and len(calls) == 1
        ocr_tool({"img_idx": 0, "bbox_2d": [0, 0, 500, 500]}, ctx)
        assert len(calls) == 2
    finally:
        ocr_tool.backend = real
