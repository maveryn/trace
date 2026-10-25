"""Model families: the conventions tools must follow to match how each model sees images.

| family | vision patch x merge | grounding coordinates | zoom tool convention from | tool calls | thinking |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-vl | 14 x 2 = 28 | absolute pixels of the image as the model sees it (after resizing) | DeepEyes (Qwen2.5-VL-7B): absolute pixel boxes, 28-px minimum side | Hermes JSON | none |
| qwen3-vl | 16 x 2 = 32 | relative [0, 1000] | Qwen3-VL cookbook ``image_zoom_in_tool`` (ZwZ "official tool use") | Hermes JSON | none (Instruct) |
| qwen3.5 | 16 x 2 = 32 | relative [0, 1000] (confirmed by probe, see ``QWEN35``) | as qwen3-vl | Qwen3 XML (``<function=...>``) | off (``enable_thinking: false``) |

Other Qwen2.5-VL agents differ: Mini-o3 and Pixel-Reasoner zoom with relative /
normalised boxes and Thyme crops through generated code. We follow DeepEyes
because it matches Qwen2.5-VL's native grounding frame.

Tools take coordinates in the family's native grounding frame, so the model can
use the same boxes it would output for grounding. The zoom tool grows tiny
boxes to one patch-merge unit per side: 28 px for Qwen2.5-VL (as DeepEyes
does), 32 px for Qwen3-VL (as Qwen-Agent does).

Qwen3.5 (Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B; Qwen3_5ForConditionalGeneration) has
the Qwen3-VL vision grid, but its chat template calls tools in XML, puts the
tool block before the system prompt and thinks unless ``enable_thinking`` is
false. ``tool_format`` and ``thinking_off`` carry those differences; the
defaults describe the older families.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# TRACE's pinned checkpoints (rlvr/evaluation/trace_validation/suite.v1.json).
PINNED_REVISIONS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "maveryn/trace-qwen2.5-vl-3b": "2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5",
    "maveryn/trace-qwen2.5-vl-7b": "4d0f1ae8ee25022058090dbdbff61957ece7331d",
    # Qwen3.5 snapshots whose (identical) chat templates were checked (tests/data/qwen3_5_chat_template.jinja).
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    patch_factor: int  # images are resized to multiples of this
    coords: str  # "abs": pixels of the image as shown; "rel1000": relative [0, 1000]
    zoom_min_side: int  # zoom boxes are grown to at least this many pixels per side
    # "hermes": {"name": ..., "arguments": {...}} JSON in <tool_call> (trace_tool.parsing.parse_tool_calls);
    # "qwen3_xml": <tool_call><function=NAME><parameter=KEY>value</parameter>... (parse_xml_tool_calls)
    tool_format: str = "hermes"
    thinking_off: bool = False  # the chat template thinks by default; send enable_thinking=False


QWEN25_VL = ModelFamily("qwen2.5-vl", patch_factor=28, coords="abs", zoom_min_side=28)
QWEN3_VL = ModelFamily("qwen3-vl", patch_factor=32, coords="rel1000", zoom_min_side=32)
# Grounding frame confirmed by probe (2026-10-05, vLLM 0.30.0, thinking off, greedy): on synthetic
# 1600x800, 800x1600 and 1920x640 images Qwen3.5-4B and -9B answered boxes within a few units of the
# relative [0, 1000] box in 12/12 cases, e.g. pixels (1000, 200, 1300, 500) on 1600x800 ->
# [623, 250, 814, 625] (4B) and [620, 234, 828, 634] (9B), where rel1000 is [625, 250, 812, 625].
QWEN35 = ModelFamily("qwen3.5", patch_factor=32, coords="rel1000", zoom_min_side=32, tool_format="qwen3_xml", thinking_off=True)
FAMILIES = {f.name: f for f in (QWEN25_VL, QWEN3_VL, QWEN35)}


def detect_family(model_id: str) -> ModelFamily:
    """Family from an HF id, served name or checkpoint path, e.g. Qwen/Qwen2.5-VL-7B-Instruct,
    maveryn/trace-qwen2.5-vl-3b, /ckpt/qwen2_5_vl_7b_grpo, Qwen/Qwen3.5-4B."""
    key = re.sub(r"[^a-z0-9]", "", model_id.lower())
    if "qwen25vl" in key:
        return QWEN25_VL
    if "qwen3vl" in key:
        return QWEN3_VL
    if "qwen35" in key:  # after the VL checks: "Qwen/Qwen3.5-4B" -> "qwenqwen354b"
        return QWEN35
    raise ValueError(
        f"cannot tell the model family of {model_id!r} (expected qwen2.5-vl, qwen3-vl or qwen3.5 in the name); "
        f"pass --family {'/'.join(FAMILIES)}"
    )


def chat_template_kwargs(family: ModelFamily) -> dict[str, Any] | None:
    """Extra chat-template arguments the endpoint must render this family's prompts with (None: none)."""
    return {"enable_thinking": False} if family.thinking_off else None


def get_family(name: str) -> ModelFamily:
    if name not in FAMILIES:
        raise ValueError(f"unknown model family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name]
