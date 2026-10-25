"""Qwen2.5-VL (TRACE's base models): absolute pixel coordinates on a 28-px grid."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import jinja2
import numpy as np
import pytest

from trace_tool.agent import Completion, build_initial_messages, run_episode
from trace_tool.conditions import build_episode_config, load_conditions
from trace_tool.models import QWEN3_VL, QWEN25_VL, detect_family
from trace_tool.prompting import TOOLS_FOOTER, TOOLS_HEADER, system_message, text_part, tool_response_content, user_prompt_content
from trace_tool.tools import build_tools
from trace_tool.tools.base import ToolContext

from .conftest import REPO, make_image, make_row

TEMPLATE = (REPO / "tests" / "data" / "qwen2_5_vl_7b_instruct_chat_template.jinja").read_text()


@pytest.mark.parametrize(
    "model_id, family",
    [
        ("Qwen/Qwen2.5-VL-7B-Instruct", QWEN25_VL),
        ("Qwen/Qwen2.5-VL-3B-Instruct", QWEN25_VL),
        ("qwen2.5-vl-7b-instruct", QWEN25_VL),
        ("maveryn/trace-qwen2.5-vl-3b", QWEN25_VL),
        ("Qwen/Qwen3-VL-8B-Instruct", QWEN3_VL),
        ("/ckpt/qwen2_5_vl_7b_grpo", QWEN25_VL),
        ("qwen2_5_vl", QWEN25_VL),
        ("qwen3_vl_8b", QWEN3_VL),
    ],
)
def test_detect_family(model_id, family):
    assert detect_family(model_id) is family


def test_detect_family_unknown():
    with pytest.raises(ValueError):
        detect_family("org/some-model")


def test_rendering_matches_qwen_agent_nous_prompt():
    """Qwen2.5-VL's template has no tool support; Qwen-Agent's NousFnCallPrompt defines the format."""
    tools = build_tools(["zoom"], QWEN25_VL)
    schemas = [t.schema() for t in tools.values()]
    image = make_image()
    call = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 300, 200], "label": "box", "img_idx": 0}}\n</tool_call>'
    parts = lambda ps: [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image_url", "image_url": {"url": "x"}} for p in ps]
    messages = [
        system_message("SYS", schemas),
        {"role": "user", "content": parts(user_prompt_content("<image>Q?", [(image, 0)]))},
        {"role": "assistant", "content": call},
        {"role": "user", "content": parts(tool_response_content([[{"type": "image", "image": image, "index": 1}]]))},
    ]
    rendered = jinja2.Environment().from_string(TEMPLATE).render(messages=messages, add_generation_prompt=True)
    tools_block = TOOLS_HEADER + "".join("\n" + json.dumps(s, ensure_ascii=False) for s in schemas) + TOOLS_FOOTER
    expected = (
        f"<|im_start|>system\nSYS\n\n{tools_block}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Q?<|im_end|>\n"
        f"<|im_start|>assistant\n{call}<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert rendered == expected


def test_shown_image_on_28px_grid_and_tool_descriptions():
    cfg = build_episode_config(load_conditions()["all"], seed=0, family=QWEN25_VL)
    messages, ctx, shown = build_initial_messages(make_row(image=make_image(1200, 800)), cfg)
    assert shown[0].size == (1204, 812)  # smart_resize(800, 1200, factor=28)
    assert ctx.shown_sizes[0] == (1204, 812) and ctx.images[0].size == (1200, 800)
    schemas = {t.name: json.dumps(t.schema()) for t in cfg.tools.values()}
    assert "pixel coordinates of that image as you see it" in schemas["ocr"]
    assert "pixel coordinates of that image as you see it" in schemas["image_draw_tool"]
    assert "[0, 1000]" not in messages[0]["content"]


def test_absolute_zoom_maps_shown_pixels_to_source():
    zoom = build_tools(["zoom"], QWEN25_VL)["image_zoom_in_tool"]
    ctx = ToolContext([make_image(1200, 800)], shown_sizes=[(1204, 812)])
    # the red box is drawn at source pixels (50, 50)-(150, 120); in the shown frame that is ~(50, 51)-(150, 122)
    result = zoom({"bbox_2d": [50, 51, 150, 122], "label": "red box", "img_idx": 0}, ctx)
    assert result.ok
    left, top, right, bottom = result.meta["abs_box"]
    assert abs(left - 50) < 1 and abs(top - 50) < 1.5 and abs(right - 150) < 1 and abs(bottom - 120) < 1.5
    w, h = result.images[0].size
    assert w % 28 == 0 and h % 28 == 0 and w * h >= 262144
    crop = np.asarray(result.images[0])
    assert crop[h // 2, w // 2, 0] > 150 and crop[h // 2, w // 2, 2] < 100  # the crop is the red box


def test_lowres_zoom_recovers_full_resolution_detail():
    """The *_lowres conditions: the model sees a small copy; zoom crops the full-resolution source."""
    cfg = build_episode_config(load_conditions()["zoom_lowres"], seed=0, family=QWEN25_VL)
    _, ctx, shown = build_initial_messages(make_row(image=make_image(1600, 1200)), cfg)
    sw, sh = shown[0].size
    assert sw * sh <= 401408 and sw % 28 == 0
    result = cfg.tools["image_zoom_in_tool"]({"bbox_2d": [0, 0, sw // 4, sh // 4], "label": "corner", "img_idx": 0}, ctx)
    left, top, right, bottom = result.meta["abs_box"]
    assert (right - left) > sw // 4 * 1.5  # a quarter of the shown frame is ~400 source pixels wide


def test_draw_and_ocr_boxes_use_shown_frame():
    draw = build_tools(["image_draw_tool"], QWEN25_VL)["image_draw_tool"]
    ctx = ToolContext([make_image(1200, 800)], shown_sizes=[(600, 400)])
    result = draw({"points": [[300, 200]], "color": "blue", "img_idx": 0}, ctx)
    arr = np.asarray(result.images[0].resize((1200, 800)))
    assert arr[400, 600, 2] > 150 and arr[400, 600, 0] < 100  # shown (300, 200) is source (600, 400)
    assert ctx.to_model_box([600, 400, 1200, 800], 0, "abs") == [300, 200, 600, 400]
    # tool outputs are shown at their own size
    (idx,) = ctx.add_images(result.images)
    assert ctx.to_source_box([0, 0, 10, 10], idx, "abs") == [0, 0, 10, 10]


def test_episode_with_qwen25_family_end_to_end():
    class Scripted:
        def __init__(self):
            self.replies = [
                '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [40, 40, 160, 130], "label": "box", "img_idx": 0}}\n</tool_call>',
                '{"answer": 3}',
            ]

        async def complete(self, messages, **kw):
            return Completion(self.replies.pop(0), "stop", 10, 5)

    cfg = build_episode_config(load_conditions()["zoom"], seed=0, family=QWEN25_VL)
    with ThreadPoolExecutor(2) as pool:
        record = asyncio.run(run_episode(make_row(), cfg, Scripted(), pool))
    assert record["tool_calls"][0]["ok"] and record["final_response"] == '{"answer": 3}'
    assert record["shown_image_size"] == [644, 420]  # 600x400 is below 262144 px: upscaled onto the 28-px grid


def test_integer_boxes_map_exactly_and_28px_boxes_are_not_regrown():
    from PIL import Image

    zoom = build_tools(["zoom"], QWEN25_VL)["image_zoom_in_tool"]
    ctx = ToolContext([Image.new("RGB", (1204, 1204))])  # shown == stored, as for tool outputs
    result = zoom({"bbox_2d": [133, 133, 161, 161], "label": "x", "img_idx": 0}, ctx)
    assert result.meta["abs_box"] == [133.0, 133.0, 161.0, 161.0]
    ctx = ToolContext([Image.new("RGB", (644, 644))], shown_sizes=[(644, 644)])
    assert ctx.to_source_box([91, 91, 119, 119], 0, "abs") == [91.0, 91.0, 119.0, 119.0]


def test_tool_family_decides_the_frame():
    """A Qwen2.5-VL tool reads pixels even from a context built without any frame hint."""
    zoom = build_tools(["zoom"], QWEN25_VL)["image_zoom_in_tool"]
    result = zoom({"bbox_2d": [50, 50, 150, 120], "label": "x", "img_idx": 0}, ToolContext([make_image(644, 420)]))
    assert result.meta["abs_box"] == [50.0, 50.0, 150.0, 120.0]


def test_ocr_schema_wording_per_family():
    pytest.importorskip("rapidocr")
    qwen3 = json.dumps(build_tools(["ocr"], QWEN3_VL)["ocr"].schema())
    assert "in relative coordinates [0, 1000] of that image" in qwen3  # unchanged Qwen3-VL wording
    qwen25 = json.dumps(build_tools(["ocr"], QWEN25_VL)["ocr"].schema())
    assert "in pixel coordinates of that image as you see it" in qwen25 and "[0, 1000]" not in qwen25
