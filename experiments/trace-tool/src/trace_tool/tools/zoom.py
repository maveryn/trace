"""Qwen3-VL's official zoom-in tool.

This is the tool behind the "Qwen3-VL with official tool use" baseline in
Zooming without Zooming (arXiv:2602.11858, Sec. 5 / Table 4), which follows the
Qwen3-VL ``think_with_images`` cookbook. The schema and the crop/resize
arithmetic are ported from Qwen-Agent's ``qwen_agent/tools/image_zoom_in_qwen3vl.py``
(Apache-2.0); tests/test_zoom.py checks numerical parity against that code.

For Qwen2.5-VL (TRACE's base models) the same tool takes ``bbox_2d`` in
absolute pixels of the image as the model sees it and grows boxes to 28 px,
the convention of DeepEyes (the Qwen2.5-VL-7B zoom agent ZwZ compares
against) and of Qwen2.5-VL grounding; crops are resized to the 28-px grid.
The schema text is the same for both families: like the cookbook and DeepEyes,
it relies on the model's native grounding frame instead of naming units.

Arguments are validated like Qwen-Agent's jsonschema check: ``bbox_2d`` (4
numbers), ``label`` (string) and ``img_idx`` (number) are required, and
``img_idx`` indexes the conversation's images like a Python list (-1 = latest).
Deliberate deviations, all on error paths:
- an integral float ``img_idx`` such as 1.0 is accepted (Qwen-Agent's list
  indexing raises and returns its generic invalid-image error);
- error messages name the bad argument;
- a crop more elongated than 200:1 (a sliver at an image edge cannot be grown
  to the minimum side) becomes a tool error (``Tool.__call__``) instead of an image the
  server would reject, which would end the episode.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Any

from PIL import Image

from .base import Tool, ToolContext, ToolError, ToolResult, as_bbox, rel_to_abs_box, smart_resize

ZOOM_MIN_SIDE = 32
ZOOM_FACTOR = 32
ZOOM_MIN_PIXELS = 256 * 32 * 32
ZOOM_MAX_PIXELS = 12845056


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def maybe_resize_bbox(
    left: float, top: float, right: float, bottom: float, img_width: int, img_height: int, min_side: int = ZOOM_MIN_SIDE
) -> list[float]:
    """Clamp the box to the image and grow it to >= min_side px per side around its centre."""
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    height = bottom - top
    width = right - left
    if height < min_side or width < min_side:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = min_side / min(height, width)
        new_half_height = ceil(height * ratio * 0.5)
        new_half_width = ceil(width * ratio * 0.5)
        new_left = floor(center_x - new_half_width)
        new_right = ceil(center_x + new_half_width)
        new_top = floor(center_y - new_half_height)
        new_bottom = ceil(center_y + new_half_height)

        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_width, new_right)
        new_bottom = min(img_height, new_bottom)

        new_height = new_bottom - new_top
        new_width = new_right - new_left
        if new_height > min_side and new_width > min_side:
            return [new_left, new_top, new_right, new_bottom]
    return [left, top, right, bottom]


def zoom_crop_pixels(
    image: Image.Image, box: list[float], *, factor: int = ZOOM_FACTOR, min_side: int = ZOOM_MIN_SIDE
) -> tuple[Image.Image, list[float]]:
    """Crop a pixel box (grown to min_side) and upscale it to >= 256 patches as Qwen-Agent does.

    Returns the resized crop and the pixel box that was cropped.
    """
    img_width, img_height = image.size
    left, top, right, bottom = maybe_resize_bbox(*box, img_width, img_height, min_side=min_side)
    cropped = image.crop((left, top, right, bottom))
    new_h, new_w = smart_resize(
        bottom - top,
        right - left,
        factor=factor,
        min_pixels=ZOOM_MIN_PIXELS,
        max_pixels=ZOOM_MAX_PIXELS,
    )
    return cropped.resize((new_w, new_h), resample=Image.BICUBIC), [left, top, right, bottom]


def zoom_crop(image: Image.Image, rel_bbox: list[float]) -> tuple[Image.Image, list[float]]:
    """Qwen-Agent's Qwen3-VL zoom exactly: ``rel_bbox`` in [0, 1000], factor and minimum side 32."""
    return zoom_crop_pixels(image, rel_to_abs_box(rel_bbox, *image.size))


class ImageZoomInTool(Tool):
    name = "image_zoom_in_tool"
    description = (
        "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) "
        "and an optional object label"
    )
    parameters = {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) "
                    "is the top-left corner and (x2, y2) is the bottom-right corner"
                ),
            },
            "label": {
                "type": "string",
                "description": "The name or label of the object in the specified bounding box",
            },
            "img_idx": {
                "type": "number",
                "description": "The index of the zoomed-in image (starting from 0)",
            },
        },
        "required": ["bbox_2d", "label", "img_idx"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        missing = [k for k in self.parameters["required"] if k not in args]
        if missing:
            raise ToolError(f"Error: missing required argument(s) {', '.join(missing)}")
        box = args["bbox_2d"]
        if not isinstance(box, list) or len(box) != 4 or not all(_is_json_number(v) for v in box):
            raise ToolError("Error: bbox_2d must be a list of 4 numbers [x1, y1, x2, y2]")
        if not isinstance(args["label"], str):
            raise ToolError("Error: label must be a string")
        if not _is_json_number(args["img_idx"]):
            raise ToolError("Error: img_idx must be a number")
        image = ctx.get_image(args["img_idx"])
        pixel_box = ctx.to_source_box(as_bbox(box), args["img_idx"], self.family.coords)
        try:
            crop, abs_box = zoom_crop_pixels(image, pixel_box, factor=self.family.patch_factor, min_side=self.family.zoom_min_side)
        except Exception as exc:  # degenerate boxes, as in Qwen-Agent
            return ToolResult.error(f"Tool Execution Error {exc}")
        return ToolResult(images=[crop], meta={"abs_box": [float(v) for v in abs_box], "out_size": list(crop.size)})
