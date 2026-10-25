import random

import numpy as np
import pytest

from trace_tool.tools import build_tools
from trace_tool.tools.base import ToolContext
from trace_tool.tools.zoom import ImageZoomInTool, zoom_crop
from trace_tool.models import QWEN3_VL, QWEN25_VL

from .conftest import make_image
from .reference.qwen_agent_zoom import ReferenceZoom


def _random_boxes(n=300, seed=0):
    rng = random.Random(seed)
    boxes = []
    for _ in range(n):
        x1, y1 = rng.uniform(-50, 1000), rng.uniform(-50, 1000)
        w, h = rng.choice([rng.uniform(0.5, 20), rng.uniform(20, 600)]), rng.choice([rng.uniform(0.5, 20), rng.uniform(20, 600)])
        boxes.append([x1, y1, x1 + w, y1 + h])
    return boxes


@pytest.mark.parametrize("size", [(1200, 800), (583, 583), (151, 300), (1730, 760)])
def test_crop_matches_qwen_agent(size):
    image = make_image(*size)
    reference = ReferenceZoom()
    for box in _random_boxes():
        try:
            ref_img, ref_box = reference.crop(image, box)
        except Exception:
            with pytest.raises(Exception):
                zoom_crop(image, box)
            continue
        ours, our_box = zoom_crop(image, box)
        assert our_box == ref_box
        assert ours.size == ref_img.size
        assert np.array_equal(np.asarray(ours), np.asarray(ref_img))


def test_schema_matches_qwen_agent():
    schema = ImageZoomInTool().schema()
    assert schema["function"]["name"] == "image_zoom_in_tool"
    params = schema["function"]["parameters"]
    assert list(params["properties"]) == ["bbox_2d", "label", "img_idx"]
    assert params["required"] == ["bbox_2d", "label", "img_idx"]
    assert params["properties"]["bbox_2d"]["minItems"] == 4


def test_tool_call_and_image_indexing():
    tool = build_tools(["zoom"], QWEN3_VL)["image_zoom_in_tool"]
    ctx = ToolContext([make_image(1200, 800)])
    first = tool({"bbox_2d": [0, 0, 500, 500], "label": "x", "img_idx": 0}, ctx)
    assert first.ok and len(first.images) == 1
    assert first.images[0].size[0] % 32 == 0 and first.images[0].size[1] % 32 == 0
    ctx.add_images(first.images)
    second = tool({"bbox_2d": [0, 0, 500, 500], "label": "x", "img_idx": 1.0}, ctx)
    assert second.ok
    bad = tool({"bbox_2d": [0, 0, 500, 500], "label": "x", "img_idx": 5}, ctx)
    assert not bad.ok and "out of range" in bad.text
    degenerate = tool({"bbox_2d": [600, 600, 100, 100], "label": "x", "img_idx": 0}, ctx)
    assert not degenerate.ok and degenerate.text.startswith("Tool Execution Error")
    missing = tool({"label": "x", "img_idx": 0}, ctx)
    assert not missing.ok


def test_small_crop_upscaled_to_min_pixels():
    crop, _ = zoom_crop(make_image(1200, 800), [100, 100, 130, 130])
    assert crop.size[0] * crop.size[1] >= 256 * 32 * 32
