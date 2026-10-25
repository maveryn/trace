"""Tool prompts in Qwen-Agent's Nous format, for Qwen2.5-VL and Qwen3-VL.

We do not send ``tools=`` to the server (that would need vLLM's tool parser and
image support in ``role: tool`` messages). Instead, like Qwen-Agent's
``NousFnCallPrompt``, the tool block is appended to the system message and tool
results are sent as user messages wrapped in ``<tool_response>`` tags. With the
Qwen3-VL chat template this renders to the same string as the native
``tools=`` + ``role: tool`` path (tests/test_prompting.py). Qwen2.5-VL's
template has no tool path; this is the layout Qwen-Agent prompts it with
(tests/test_qwen25.py, checked against the real tokenizers).

Qwen3.5 (``family.tool_format == "qwen3_xml"``) is flattened the same way, in
its own layout: the XML tool block comes first and the system prompt after it,
and each tool body is trimmed as its template trims ``role: tool`` content, so
our messages render exactly like the native path (tests/test_qwen35.py).
Functions that take ``family`` keep the Hermes layout when it is None or a
Hermes family.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from .models import ModelFamily

# Verbatim from the Qwen3-VL chat template (Qwen/Qwen3-VL-8B-Instruct), and
# identical to Qwen-Agent's FN_CALL_TEMPLATE.
TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

# Verbatim from the Qwen3.5 chat template (Qwen/Qwen3.5-4B and -9B, tests/data/qwen3_5_chat_template.jinja,
# lines 47-53): the tool block opens the system turn and the system prompt follows it.
QWEN35_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
QWEN35_TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>"
)

TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"

IMAGE_MARKER = "<image>"

# Forced conditions (``forced_answer``): a call budget stated up front, and a notice at
# the end of the last tool response of each tool turn, inside its <tool_response> tags
# (see agent.run_episode): the countdown, or the final-answer notice once the turns,
# calls or tokens run out. A turn's earlier tool responses carry no notice; the
# model-facing "Each tool response ends ..." wording is kept as fixed by the design.
# Their sha256 is a resume key of the forced conditions (conditions.describe).
BUDGET_STATEMENT = (
    "You can make at most {budget} tool calls in total, and at most {per_turn} per turn. Each tool response ends "
    "with the number of tool calls you have left. When none are left, give your final answer without calling any tools."
)
CALLS_LEFT_NOTICE = "[Tool calls left: {left}]"
# Mirrors the required ending of prompts/trace_system_prompt.txt. Not a format string.
FINAL_ANSWER_NOTICE = (
    "[No tool calls left. Do not call any more tools. Give your final answer now, ending your response with a "
    'JSON object in this format: {"answer": ...}]'
)
FORCED_PROMPT_SHA256 = hashlib.sha256(
    (BUDGET_STATEMENT + "\n" + CALLS_LEFT_NOTICE + "\n" + FINAL_ANSWER_NOTICE).encode()
).hexdigest()


def hf_tojson(value: Any) -> str:
    """transformers' chat-template ``tojson`` filter with default arguments."""
    return json.dumps(value, ensure_ascii=False)


def budget_statement(budget: int, per_turn: int) -> str:
    return BUDGET_STATEMENT.format(budget=budget, per_turn=per_turn)


def calls_left_notice(left: int) -> str:
    return CALLS_LEFT_NOTICE.format(left=left)


def _is_xml(family: "ModelFamily | None") -> bool:
    return family is not None and family.tool_format == "qwen3_xml"


def render_tools_block(tool_schemas: list[dict[str, Any]], family: "ModelFamily | None" = None) -> str:
    header, footer = (QWEN35_TOOLS_HEADER, QWEN35_TOOLS_FOOTER) if _is_xml(family) else (TOOLS_HEADER, TOOLS_FOOTER)
    return header + "".join("\n" + hf_tojson(schema) for schema in tool_schemas) + footer


def system_message(
    system_prompt: str | None, tool_schemas: list[dict[str, Any]], family: "ModelFamily | None" = None
) -> dict[str, Any] | None:
    if tool_schemas:
        block = render_tools_block(tool_schemas, family)
        if _is_xml(family):
            # the Qwen3.5 template: tool block, then "\n\n" + the trimmed system prompt if it is not empty
            prompt = (system_prompt or "").strip()
            return {"role": "system", "content": block + ("\n\n" + prompt if prompt else "")}
        content = f"{system_prompt}\n\n{block}" if system_prompt else block
        return {"role": "system", "content": content}
    if system_prompt:
        return {"role": "system", "content": system_prompt}
    return None


# Internal message format: content is a str or a list of parts, where a part is
# {"type": "text", "text": str} or {"type": "image", "image": PIL.Image, "index": int}.
# `index` is the image's position in ToolContext.images.


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image: Image.Image, index: int) -> dict[str, Any]:
    return {"type": "image", "image": image, "index": index}


def merge_text_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            if not part["text"]:
                continue
            if merged and merged[-1]["type"] == "text":
                merged[-1] = text_part(merged[-1]["text"] + part["text"])
                continue
        merged.append(part)
    return merged


def user_prompt_content(prompt: str, images: list[tuple[Image.Image, int]]) -> list[dict[str, Any]]:
    """Split a TRACE prompt on its ``<image>`` markers, as TRACE's generator does."""
    segments = prompt.split(IMAGE_MARKER)
    if len(segments) - 1 != len(images):
        raise ValueError(f"prompt has {len(segments) - 1} {IMAGE_MARKER} markers for {len(images)} images")
    parts: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        if segment:
            parts.append(text_part(segment))
        if i < len(images):
            parts.append(image_part(*images[i]))
    return parts


def _trim_body(body: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``render_content(content)|trim`` of the Qwen3.5 template, on parts: strip the text at both ends."""
    body = merge_text_parts(body)
    if body and body[0]["type"] == "text":
        body[0] = text_part(body[0]["text"].lstrip())
    if body and body[-1]["type"] == "text":
        body[-1] = text_part(body[-1]["text"].rstrip())
    return merge_text_parts(body)


def tool_response_content(responses: list[list[dict[str, Any]]], family: "ModelFamily | None" = None) -> list[dict[str, Any]]:
    """One user turn holding consecutive tool responses, as the template renders ``role: tool`` runs."""
    if _is_xml(family):  # the Qwen3.5 template trims each role: tool message
        responses = [_trim_body(body) for body in responses]
    parts: list[dict[str, Any]] = []
    for i, body in enumerate(responses):
        parts.append(text_part(("\n" if i else "") + TOOL_RESPONSE_OPEN))
        parts.extend(body)
        parts.append(text_part(TOOL_RESPONSE_CLOSE))
    return merge_text_parts(parts)


def encode_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ImageURLCache:
    """Encodes each image once per episode; images are re-sent on every turn."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Image.Image, str]] = {}

    def url(self, image: Image.Image) -> str:
        hit = self._cache.get(id(image))
        if hit is None or hit[0] is not image:
            hit = (image, encode_png_data_url(image))
            self._cache[id(image)] = hit
        return hit[1]


def to_openai_messages(messages: list[dict[str, Any]], cache: ImageURLCache) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": cache.url(part["image"])}})
        out.append({"role": message["role"], "content": parts})
    return out


def to_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-serialisable copy with images replaced by their index and size."""
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image", "index": part["index"], "size": list(part["image"].size)})
        out.append({"role": message["role"], "content": parts})
    return out


def render_text(messages: list[dict[str, Any]]) -> str:
    """Plain-text rendering for debugging; images become ``<image k>``."""
    chunks = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            body = content
        else:
            body = "".join(p["text"] if p["type"] == "text" else f"<image {p['index']}>" for p in content)
        chunks.append(f"<|{message['role']}|>\n{body}")
    return "\n".join(chunks)
