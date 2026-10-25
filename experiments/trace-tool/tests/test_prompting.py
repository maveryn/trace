"""Our flattened messages must render exactly like the native tools= / role=tool path."""

import json

import jinja2
import pytest

from trace_tool.prompting import system_message, text_part, tool_response_content, user_prompt_content
from trace_tool.tools import build_tools
from trace_tool.models import QWEN3_VL, QWEN25_VL

from .conftest import REPO, make_image

TEMPLATE = (REPO / "tests" / "data" / "qwen3_vl_8b_instruct_chat_template.jinja").read_text()


def _env():
    # transformers' chat-template environment: trim/lstrip blocks and its own tojson
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )
    return env.from_string(TEMPLATE)


def _openai_parts(parts):
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image_url", "image_url": {"url": "x"}} for p in parts]


def _render(messages, tools=None):
    return _env().render(messages=messages, tools=tools, add_generation_prompt=True)


@pytest.mark.parametrize("n_responses", [1, 2])
def test_flattened_matches_native(n_responses):
    tools = build_tools(["zoom", "calculator"], QWEN3_VL)
    schemas = [t.schema() for t in tools.values()]
    image = make_image()
    prompt = '<image>Count the red boxes.\nExample JSON:\n{"answer":2}'
    sys_text = "You are a helpful assistant."
    assistant = 'I will zoom.\n<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "box", "img_idx": 0}}\n</tool_call>'

    user = _openai_parts(user_prompt_content(prompt, [(image, 0)]))
    bodies = [[{"type": "image", "image": image, "index": 1}], [text_part("42")]][:n_responses]

    native = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    native += [{"role": "tool", "content": _openai_parts(b)} for b in bodies]
    ours = [
        system_message(sys_text, schemas),
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": _openai_parts(tool_response_content(bodies))},
    ]
    native_text = _render(native, tools=schemas)
    assert native_text == _render(ours)
    assert "<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>" in native_text


def test_no_tools_system_is_plain():
    assert system_message("abc", []) == {"role": "system", "content": "abc"}
    rendered = _render([system_message("abc", []), {"role": "user", "content": "hi"}])
    assert rendered == "<|im_start|>system\nabc<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_image_marker_split():
    image = make_image()
    parts = user_prompt_content("<image>What is shown?", [(image, 0)])
    assert [p["type"] for p in parts] == ["image", "text"]
    with pytest.raises(ValueError):
        user_prompt_content("no marker", [(image, 0)])
