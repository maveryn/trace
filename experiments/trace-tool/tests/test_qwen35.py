"""Qwen3.5: XML tool calls, the tool block before the system prompt, thinking switched off.

Our flattened messages (tool block in the system message, tool results in user
turns) must render through the Qwen3.5 chat template exactly like the native
``tools=`` + ``tool_calls`` + ``role: tool`` path. The template is vendored from
Qwen/Qwen3.5-4B (identical in -9B; Apache-2.0) at the pinned revisions.
"""

import hashlib
import json
import re

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from trace_tool.agent import build_initial_messages
from trace_tool.conditions import build_episode_config, load_conditions
from trace_tool.models import (
    FAMILIES,
    PINNED_REVISIONS,
    QWEN3_VL,
    QWEN25_VL,
    QWEN35,
    chat_template_kwargs,
    detect_family,
    get_family,
)
from trace_tool.parsing import parse_xml_tool_calls
from trace_tool.prompting import (
    FINAL_ANSWER_NOTICE,
    QWEN35_TOOLS_FOOTER,
    QWEN35_TOOLS_HEADER,
    calls_left_notice,
    render_tools_block,
    system_message,
    text_part,
    tool_response_content,
    user_prompt_content,
)
from trace_tool.tools import build_tools

from .conftest import REPO, make_image, make_row

TEMPLATE_PATH = REPO / "tests" / "data" / "qwen3_5_chat_template.jinja"
TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
TEMPLATE = TEMPLATE_PATH.read_text()
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _template():
    # transformers' chat-template environment (utils/chat_template_utils.py)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(TEMPLATE)


def _render(messages, tools=None, enable_thinking=False):
    return _template().render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=enable_thinking)


def _parts(parts):
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image_url", "image_url": {"url": "x"}} for p in parts]


def _schemas(names=("zoom", "calculator")):
    return [t.schema() for t in build_tools(list(names), QWEN35).values()]


def _param_schemas(schemas):
    return {s["function"]["name"]: s["function"]["parameters"] for s in schemas}


def xml_calls(content, calls):
    """The text the model writes for ``calls``, laid out as the template renders ``tool_calls``."""
    out = content
    for i, (name, args) in enumerate(calls):
        lead = ("\n\n" if content.strip() else "") if i == 0 else "\n"
        out += f"{lead}<tool_call>\n<function={name}>\n"
        for key, value in args.items():
            shown = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            out += f"<parameter={key}>\n{shown}\n</parameter>\n"
        out += "</function>\n</tool_call>"
    return out


def _native_assistant(content, calls):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls],
    }


ZOOM_ARGS = ("image_zoom_in_tool", {"bbox_2d": [0, 0, 500, 500], "label": "box", "img_idx": 0})
CALC_ARGS = ("calculator", {"expression": "1 + 2"})
SYS = "You are a helpful assistant."
PROMPT = '<image>Count the red boxes.\nExample JSON:\n{"answer":2}'


def _conversations(turns, sys_text=SYS):
    """(native, ours) for a conversation of ``turns`` = [(assistant content, calls, bodies), ...]."""
    schemas = _schemas()
    user = _parts(user_prompt_content(PROMPT, [(make_image(), 0)]))
    native = ([{"role": "system", "content": sys_text}] if sys_text is not None else []) + [{"role": "user", "content": user}]
    ours = [system_message(sys_text, schemas, family=QWEN35), {"role": "user", "content": user}]
    for content, calls, bodies in turns:
        native.append(_native_assistant(content, calls))
        native += [{"role": "tool", "content": _parts(body)} for body in bodies]
        ours.append({"role": "assistant", "content": xml_calls(content, calls)})
        ours.append({"role": "user", "content": _parts(tool_response_content(bodies, family=QWEN35))})
    return native, ours, schemas


# --- family -------------------------------------------------------------------


def test_family_fields_and_registry():
    assert (QWEN35.name, QWEN35.patch_factor, QWEN35.coords, QWEN35.zoom_min_side) == ("qwen3.5", 32, "rel1000", 32)
    assert QWEN35.tool_format == "qwen3_xml" and QWEN35.thinking_off is True
    for family in (QWEN25_VL, QWEN3_VL):  # the older families keep the defaults
        assert family.tool_format == "hermes" and family.thinking_off is False
    assert FAMILIES == {"qwen2.5-vl": QWEN25_VL, "qwen3-vl": QWEN3_VL, "qwen3.5": QWEN35}
    assert get_family("qwen3.5") is QWEN35
    assert PINNED_REVISIONS["Qwen/Qwen3.5-4B"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert PINNED_REVISIONS["Qwen/Qwen3.5-9B"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert chat_template_kwargs(QWEN35) == {"enable_thinking": False}
    assert chat_template_kwargs(QWEN25_VL) is None and chat_template_kwargs(QWEN3_VL) is None


@pytest.mark.parametrize(
    "model_id, family",
    [
        ("Qwen/Qwen3.5-4B", QWEN35),
        ("Qwen/Qwen3.5-9B", QWEN35),
        ("qwen3.5-4b", QWEN35),
        ("/ckpt/qwen3_5_9b_grpo", QWEN35),
        ("Qwen/Qwen3.5-27B", QWEN35),
        ("Qwen/Qwen2.5-VL-7B-Instruct", QWEN25_VL),
        ("maveryn/trace-qwen2.5-vl-3b", QWEN25_VL),
        ("Qwen/Qwen3-VL-8B-Instruct", QWEN3_VL),
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", QWEN3_VL),
        # names with both a VL family and "qwen3.5": the VL checks run first and win
        ("org/qwen2.5-vl-7b-distilled-from-qwen3.5", QWEN25_VL),
        ("org/qwen3-vl-8b-from-qwen3.5-teacher", QWEN3_VL),
        ("/ckpt/Qwen3.5-teacher_qwen2_5_vl_3b", QWEN25_VL),
    ],
)
def test_detect_family(model_id, family):
    assert detect_family(model_id) is family


@pytest.mark.parametrize("model_id", ["org/qwen2.5-vl-7b-distilled-from-qwen3.5", "org/qwen3-vl-8b-from-qwen3.5-teacher", "/ckpt/Qwen3.5-teacher_qwen2_5_vl_3b"])
def test_detect_family_cases_pin_the_check_order(model_id):
    # these names normalise (as in detect_family) to keys holding two families' markers, so only
    # the order of the checks decides them
    key = re.sub(r"[^a-z0-9]", "", model_id.lower())
    assert "qwen35" in key and ("qwen25vl" in key or "qwen3vl" in key)


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B", "org/some-model"])
def test_detect_family_unknown(model_id):
    with pytest.raises(ValueError, match=r"qwen3\.5"):
        detect_family(model_id)


def test_vendored_template_is_the_pinned_one():
    assert hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() == TEMPLATE_SHA256


# --- prompt layout ----------------------------------------------------------------


def test_tools_block_is_verbatim_and_precedes_the_system_prompt():
    schemas = _schemas()
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": "hi"}]
    native = _render(messages, tools=schemas)
    block = render_tools_block(schemas, QWEN35)
    assert block.startswith(QWEN35_TOOLS_HEADER) and block.endswith(QWEN35_TOOLS_FOOTER)
    assert native.startswith(f"<|im_start|>system\n{block}\n\n{SYS}<|im_end|>\n")
    assert system_message(SYS, schemas, family=QWEN35) == {"role": "system", "content": f"{block}\n\n{SYS}"}
    assert _render([system_message(SYS, schemas, family=QWEN35), messages[1]]) == native


@pytest.mark.parametrize("sys_text", ["  padded system prompt \n\n", "", None])
def test_system_prompt_is_trimmed_or_omitted(sys_text):
    native, ours, schemas = _conversations([], sys_text=sys_text)
    block = render_tools_block(schemas, QWEN35)
    assert ours[0]["content"] == (f"{block}\n\npadded system prompt" if sys_text else block)
    assert _render(ours) == _render(native, tools=schemas)


def test_without_tools_the_system_prompt_is_plain():
    assert system_message("abc", [], family=QWEN35) == {"role": "system", "content": "abc"}
    assert system_message("", [], family=QWEN35) is None
    rendered = _render([system_message("abc", [], family=QWEN35), {"role": "user", "content": "hi"}])
    assert rendered == f"<|im_start|>system\nabc<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n{EMPTY_THINK}"


def test_hermes_families_are_unchanged_by_the_family_argument():
    schemas = [t.schema() for t in build_tools(["zoom", "calculator"], QWEN3_VL).values()]
    image = make_image()
    bodies = [[text_part("  edged \n")], [{"type": "image", "image": image, "index": 1}, text_part(" caption \n")]]
    for family in (QWEN25_VL, QWEN3_VL):
        assert render_tools_block(schemas, family) == render_tools_block(schemas)
        for sys_text in ("SYS", "", None):
            assert system_message(sys_text, schemas, family=family) == system_message(sys_text, schemas)
        assert tool_response_content(bodies, family=family) == tool_response_content(bodies)
    # hermes keeps the bodies as they are; qwen3_xml trims them as the Qwen3.5 template does
    assert tool_response_content(bodies)[0]["text"].startswith("<tool_response>\n  edged \n\n</tool_response>")
    assert tool_response_content(bodies, family=QWEN35)[0]["text"].startswith("<tool_response>\nedged\n</tool_response>")


# --- byte identity with the native path ------------------------------------------


@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize("n_responses", [1, 2])
def test_flattened_matches_native(n_responses, enable_thinking):
    calls = [ZOOM_ARGS, CALC_ARGS][:n_responses]
    bodies = [[{"type": "image", "image": make_image(), "index": 1}], [text_part("3")]][:n_responses]
    native, ours, schemas = _conversations([("I will zoom.", calls, bodies)])
    native_text = _render(native, tools=schemas, enable_thinking=enable_thinking)
    assert _render(ours, enable_thinking=enable_thinking) == native_text
    # the image inside the tool response, and the generation prompt for each thinking mode
    assert f"<|im_start|>user\n<tool_response>\n{VISION}\n</tool_response>" in native_text
    assert native_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n" if not enable_thinking else "<|im_start|>assistant\n<think>\n")
    # history turns after the question keep their (empty) think block in both modes
    assert f"{EMPTY_THINK}I will zoom.\n\n<tool_call>\n<function=image_zoom_in_tool>" in native_text


def test_no_preamble_call_matches_native():
    native, ours, schemas = _conversations([("", [ZOOM_ARGS], [[text_part("ok")]])])
    assert ours[2]["content"].startswith("<tool_call>\n<function=")
    assert _render(ours) == _render(native, tools=schemas)


@pytest.mark.parametrize(
    "bodies",
    [
        [[text_part("  \n 42 \n\n")]],
        [[text_part("\n"), {"type": "image", "image": make_image(), "index": 1}, text_part(" caption\t\n")]],
        [[text_part("   ")], [text_part("\tsecond  ")]],
        [[text_part("  a"), text_part("  b  "), text_part("c \n")]],  # unmerged text parts
        # whitespace-only end parts: only merging the text parts before trimming strips the
        # notice's leading "\n\n" (or the text's trailing "\n") as the template's |trim does
        [[text_part("  "), text_part("\n\n" + calls_left_notice(8))]],
        [[text_part("x\n"), text_part(" \n")]],
    ],
)
def test_whitespace_edged_bodies_match_native(bodies):
    calls = [ZOOM_ARGS, CALC_ARGS][: len(bodies)]
    native, ours, schemas = _conversations([("Zooming.", calls, bodies)])
    assert _render(ours) == _render(native, tools=schemas)


def test_reasoning_in_history_matches_native_with_thinking_on():
    """With thinking on, the model's reasoning stays in our assistant text; the template splits it out."""
    schemas = _schemas()
    user = _parts(user_prompt_content(PROMPT, [(make_image(), 0)]))
    reply = xml_calls("I will zoom.", [ZOOM_ARGS])
    native = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
        {**_native_assistant("I will zoom.", [ZOOM_ARGS]), "reasoning_content": "The boxes are small."},
        {"role": "tool", "content": [{"type": "text", "text": "done"}]},
    ]
    ours = [
        system_message(SYS, schemas, family=QWEN35),
        {"role": "user", "content": user},
        {"role": "assistant", "content": "The boxes are small.\n</think>\n\n" + reply},  # the text after the "<think>\n" prompt
        {"role": "user", "content": _parts(tool_response_content([[text_part("done")]], family=QWEN35))},
    ]
    native_text = _render(native, tools=schemas, enable_thinking=True)
    assert _render(ours, enable_thinking=True) == native_text
    assert "<|im_start|>assistant\n<think>\nThe boxes are small.\n</think>\n\nI will zoom." in native_text


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_forced_notice_inside_last_response_keeps_history(enable_thinking):
    """Forced-mode notices go inside the last <tool_response>: the turn stays a tool turn, so the
    question stays the last query and earlier assistant turns keep their empty think blocks."""
    image = make_image()
    turn1 = ("I will zoom.", [ZOOM_ARGS], [[{"type": "image", "image": image, "index": 1}, text_part("\n\n" + calls_left_notice(8))]])
    turn2 = ("Now compute.", [CALC_ARGS], [[text_part("3"), text_part("\n\n" + FINAL_ANSWER_NOTICE)]])
    native, ours, schemas = _conversations([turn1, turn2])
    native_text = _render(native, tools=schemas, enable_thinking=enable_thinking)
    assert _render(ours, enable_thinking=enable_thinking) == native_text
    assert f"<tool_response>\n{VISION}\n\n[Tool calls left: 8]\n</tool_response>" in native_text
    assert f"<tool_response>\n3\n\n{FINAL_ANSWER_NOTICE}\n</tool_response><|im_end|>\n" in native_text
    assert native_text.count(EMPTY_THINK + "I will zoom.") == 1 and native_text.count(EMPTY_THINK + "Now compute.") == 1

    # counter-example: the same notice after the closing tag makes that turn a new query,
    # and the template re-renders the earlier assistant turns without their think blocks
    outside = ours[:-1] + [
        {"role": "user", "content": _parts(tool_response_content([[text_part("3")]], family=QWEN35) + [text_part("\n\n" + FINAL_ANSWER_NOTICE)])}
    ]
    rewritten = _render(outside, enable_thinking=enable_thinking)
    assert EMPTY_THINK + "I will zoom." not in rewritten and "<|im_start|>assistant\nI will zoom." in rewritten


def test_template_environment_matches_transformers():
    """Our jinja environment renders like transformers' own (the path vLLM uses)."""
    pytest.importorskip("transformers")
    from transformers.utils.chat_template_utils import render_jinja_template

    turn = ("I will zoom.", [ZOOM_ARGS, CALC_ARGS], [[{"type": "image", "image": make_image(), "index": 1}], [text_part(" 3 \n")]])
    native, ours, schemas = _conversations([turn])
    for enable_thinking in (False, True):
        for messages, tools in ((native, schemas), (ours, None)):
            rendered = render_jinja_template(
                conversations=[messages], tools=tools, chat_template=TEMPLATE, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            text = rendered[0][0] if isinstance(rendered, tuple) else rendered[0]
            assert text == _render(messages, tools=tools, enable_thinking=enable_thinking)


# --- XML calls round-trip through the template -----------------------------------


def test_template_rendered_calls_parse_back_to_the_native_arguments():
    schemas = _schemas(["all"])
    calls = [
        ZOOM_ARGS,
        CALC_ARGS,
        ("image_rotate_tool", {"angle": 90.5, "img_idx": -1}),
        ("image_flip_tool", {"direction": "horizontal", "img_idx": 1}),
        ("image_draw_tool", {"boxes": [[1, 2, 3, 4]], "points": [[5.5, 6]], "color": "red", "img_idx": 0}),
        ("ocr", {"img_idx": 0, "bbox_2d": [10, 20, 300, 400]}),
    ]
    native = [{"role": "user", "content": "q"}, _native_assistant("Several calls.", calls)]
    rendered = _render(native, tools=schemas)
    text = rendered.split("<|im_start|>assistant\n<think>\n\n</think>\n\n")[1].split("<|im_end|>")[0]
    assert text == xml_calls("Several calls.", calls)
    parsed = parse_xml_tool_calls(text, _param_schemas(schemas))
    assert [(c.name, c.arguments, c.error) for c in parsed] == [(name, args, None) for name, args in calls]
    assert isinstance(parsed[0].arguments["img_idx"], int) and isinstance(parsed[2].arguments["angle"], float)


# --- initial messages ----------------------------------------------------------------


def test_initial_messages_put_the_tool_block_first():
    cfg = build_episode_config(load_conditions()["visual_ops"], seed=0, family=QWEN35)
    messages, ctx, shown = build_initial_messages(make_row(image=make_image(1200, 800)), cfg)
    system = messages[0]["content"]
    assert system.startswith(QWEN35_TOOLS_HEADER)
    assert system.endswith("\n</IMPORTANT>\n\n" + cfg.system_prompt.strip())
    assert '"name": "image_zoom_in_tool"' in system and "relative coordinates [0, 1000]" in system  # rel1000 wording
    assert shown[0].size[0] % 32 == 0 and shown[0].size[1] % 32 == 0
