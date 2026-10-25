"""Information-neutral image operations listed in ZwZ Table 9.

Zooming without Zooming (arXiv:2602.11858, Table 9) lists the image actions
used by "thinking-with-images" agents: zoom in (crop), zoom out, flip, rotate,
denoise, 2D grounding, 3D grounding and drawing (plus web search). The paper
gives no API for any of them except zoom in (via the Qwen3-VL cookbook), so the
schemas below are ours and follow the zoom tool's conventions: ``img_idx``
selects an image from the conversation and coordinates are in the model
family's grounding frame (relative [0, 1000] for Qwen3-VL, pixels of the image
as shown for Qwen2.5-VL; see trace_tool.models).

Implemented: rotate, flip, draw (auxiliary lines / boxes / points).
Not implemented, by design:
- zoom out / denoise: TRACE images are clean renders shown at native
  resolution, so neither changes what the model can see;
- 2D grounding: Qwen3-VL grounds natively in the same [0, 1000] frame;
- 3D grounding / depth: needs an extra perception model, which would make the
  comparison about that model rather than about tool use;
- web search: no external information is relevant to synthetic scenes.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from ..models import QWEN3_VL, ModelFamily
from .base import COORD_PHRASE, Tool, ToolContext, ToolError, ToolResult, as_bbox, as_number, fit_image

# Outputs get the same pixel bounds as the zoom tool's minimum and the input cap.
OUT_MIN_PIXELS = 256 * 32 * 32
OUT_MAX_PIXELS = 4194304

_IMG_IDX = {"type": "number", "description": "The index of the image to operate on (starting from 0)"}

_COLORS = {
    "red": (230, 25, 75),
    "green": (60, 180, 75),
    "blue": (0, 130, 200),
    "yellow": (255, 225, 25),
    "magenta": (240, 50, 230),
    "cyan": (70, 240, 240),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "orange": (245, 130, 48),
}

MAX_DRAW_ITEMS = 32


def _fit(image: Image.Image, factor: int) -> Image.Image:
    return fit_image(image, min_pixels=OUT_MIN_PIXELS, max_pixels=OUT_MAX_PIXELS, factor=factor)


class ImageRotateTool(Tool):
    name = "image_rotate_tool"
    description = (
        "Rotate an image counterclockwise by the given angle in degrees; the canvas is expanded so "
        "nothing is cropped"
    )
    parameters = {
        "type": "object",
        "properties": {
            "angle": {
                "type": "number",
                "description": "Counterclockwise rotation angle in degrees, e.g. 90, 180, -45",
            },
            "img_idx": _IMG_IDX,
        },
        "required": ["angle", "img_idx"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        image = ctx.get_image(args.get("img_idx", 0))
        if "angle" not in args:
            raise ToolError("Error: missing required argument angle")
        angle = as_number(args["angle"], "angle")
        if not -360.0 <= angle <= 360.0:
            raise ToolError("Error: angle must be between -360 and 360 degrees")
        rgb = image.convert("RGB")
        rotated = rgb.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        return ToolResult(images=[_fit(rotated, self.family.patch_factor)], meta={"angle": angle})


class ImageFlipTool(Tool):
    name = "image_flip_tool"
    description = "Mirror an image horizontally (left-right) or vertically (top-bottom)"
    parameters = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["horizontal", "vertical"],
                "description": "'horizontal' mirrors left-right, 'vertical' mirrors top-bottom",
            },
            "img_idx": _IMG_IDX,
        },
        "required": ["direction", "img_idx"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        image = ctx.get_image(args.get("img_idx", 0))
        direction = str(args.get("direction", "")).strip().lower()
        if direction not in {"horizontal", "vertical"}:
            raise ToolError("Error: direction must be 'horizontal' or 'vertical'")
        method = Image.Transpose.FLIP_LEFT_RIGHT if direction == "horizontal" else Image.Transpose.FLIP_TOP_BOTTOM
        return ToolResult(images=[_fit(image.convert("RGB").transpose(method), self.family.patch_factor)], meta={"direction": direction})


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ToolError(f"Error: each entry of {name} must be [x, y]")
    x, y = (as_number(v, name) for v in value)
    return x, y


class ImageDrawTool(Tool):
    name = "image_draw_tool"
    description = (
        "Draw auxiliary marks (straight lines, rectangles, points) on a copy of an image, e.g. a "
        "horizontal guide line to compare heights or a box around a region"
    )

    def __init__(self, family: ModelFamily = QWEN3_VL) -> None:
        super().__init__(family)
        frame = COORD_PHRASE[family.coords]
        self.parameters = {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "description": f"Line segments as [x1, y1, x2, y2] in {frame}",
                },
                "boxes": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "description": f"Rectangles as [x1, y1, x2, y2] in {frame}",
                },
                "points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "description": f"Points as [x, y] in {frame}",
                },
                "color": {
                    "type": "string",
                    "enum": sorted(_COLORS),
                    "description": "Color of the marks (default red)",
                },
                "img_idx": _IMG_IDX,
            },
            "required": ["img_idx"],
        }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        img_idx = args.get("img_idx", 0)
        image = ctx.get_image(img_idx).convert("RGB").copy()
        width, height = image.size
        color = _COLORS.get(str(args.get("color", "red")).strip().lower())
        if color is None:
            raise ToolError(f"Error: color must be one of {sorted(_COLORS)}")
        lines = args.get("lines") or []
        boxes = args.get("boxes") or []
        points = args.get("points") or []
        for name, items in (("lines", lines), ("boxes", boxes), ("points", points)):
            if not isinstance(items, list):
                raise ToolError(f"Error: {name} must be a list")
        if not (lines or boxes or points):
            raise ToolError("Error: provide at least one of lines, boxes or points")
        if len(lines) + len(boxes) + len(points) > MAX_DRAW_ITEMS:
            raise ToolError(f"Error: at most {MAX_DRAW_ITEMS} marks per call")

        stroke = max(2, round(min(width, height) / 250))
        radius = max(3, stroke * 2)
        draw = ImageDraw.Draw(image)
        for line in lines:
            x1, y1, x2, y2 = ctx.to_source_box(as_bbox(line, "lines"), img_idx, self.family.coords)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=stroke)
        for box in boxes:
            x1, y1, x2, y2 = ctx.to_source_box(as_bbox(box, "boxes"), img_idx, self.family.coords)
            draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=color, width=stroke)
        for point in points:
            x, y = ctx.to_source_point(_point(point, "points"), img_idx, self.family.coords)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        meta = {"n_lines": len(lines), "n_boxes": len(boxes), "n_points": len(points)}
        return ToolResult(images=[_fit(image, self.family.patch_factor)], meta=meta)
