"""Tool registry."""

from __future__ import annotations

from typing import Callable

from ..models import ModelFamily
from .base import Tool, ToolContext, ToolError, ToolResult
from .calculator import CalculatorTool
from .image_ops import ImageDrawTool, ImageFlipTool, ImageRotateTool
from .zoom import ImageZoomInTool

TOOL_FACTORIES: dict[str, Callable[[ModelFamily], Tool]] = {
    ImageZoomInTool.name: ImageZoomInTool,
    ImageRotateTool.name: ImageRotateTool,
    ImageFlipTool.name: ImageFlipTool,
    ImageDrawTool.name: ImageDrawTool,
    CalculatorTool.name: CalculatorTool,
}


def _ocr_factory(family: ModelFamily) -> Tool:
    from .ocr import OCRTool  # numpy/onnx imports only when OCR is requested

    return OCRTool(family)


TOOL_FACTORIES["ocr"] = _ocr_factory

# Named groups used in configs/conditions.yaml.
TOOL_GROUPS: dict[str, list[str]] = {
    # ZwZ "Qwen3-VL with official tool use" baseline.
    "zoom": ["image_zoom_in_tool"],
    # ZwZ Table 9 information-neutral operations we implement.
    "visual_ops": ["image_zoom_in_tool", "image_rotate_tool", "image_flip_tool", "image_draw_tool"],
    "calculator": ["calculator"],
    "ocr": ["ocr"],
    "all": ["image_zoom_in_tool", "image_rotate_tool", "image_flip_tool", "image_draw_tool", "calculator", "ocr"],
}


def build_tools(names: list[str], family: ModelFamily) -> dict[str, Tool]:
    """Tools in the coordinate frame and patch grid of ``family`` (see trace_tool.models)."""
    expanded: list[str] = []
    for name in names:
        for item in TOOL_GROUPS.get(name, [name]):
            if item not in TOOL_FACTORIES:
                raise ValueError(f"unknown tool {item!r}; known tools: {sorted(TOOL_FACTORIES)}")
            if item not in expanded:
                expanded.append(item)
    return {name: TOOL_FACTORIES[name](family) for name in expanded}


__all__ = ["TOOL_FACTORIES", "TOOL_GROUPS", "Tool", "ToolContext", "ToolError", "ToolResult", "build_tools"]
