"""Framework-agnostic tool interface.

Tools are plain callables over a per-episode :class:`ToolContext`, so the same
implementations back the evaluation agent loop (``trace_tool.agent``) and the
verl adapters for RL (``trace_tool.rl.verl_tools``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..models import QWEN3_VL, ModelFamily

# Relative coordinate frame used by Qwen3-VL grounding and by the official
# `image_zoom_in_tool`: every bbox/point coordinate is in [0, 1000].
REL_SCALE = 1000.0

# How tool descriptions name each family's coordinate frame.
COORD_PHRASE = {
    "rel1000": "relative coordinates [0, 1000]",
    "abs": "pixel coordinates of that image as you see it",
}

# Qwen2-VL / Qwen3-VL image processors (transformers and qwen-vl-utils
# ``smart_resize``) raise for images whose long side exceeds 200x the short side.
MAX_ASPECT_RATIO = 200


class ToolError(Exception):
    """Raised for invalid arguments; the message is shown to the model."""


@dataclass
class ToolResult:
    """Output of one tool call.

    ``images`` are appended to the episode's image list (and shown to the
    model) in order; ``text`` is shown after the images.
    """

    images: list[Image.Image] = field(default_factory=list)
    text: str | None = None
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str, **meta: Any) -> "ToolResult":
        return cls(text=message, ok=False, meta=meta)


class ToolContext:
    """Per-episode state shared by the tools.

    ``images`` lists every image in the conversation in order of appearance,
    matching Qwen-Agent's ``extract_images_from_messages`` indexing: index 0 is
    the input image (at full resolution, even when the model is shown a
    downsampled copy), and each image returned by a tool is appended.
    ``shown_sizes`` holds the size the model sees for each image (the input is
    resized to the patch grid; tool outputs are sent as they are), which is the
    frame of absolute (Qwen2.5-VL) coordinates. The coordinate convention itself
    belongs to the calling tool's model family, so a tool and the context can
    never disagree about it.
    """

    def __init__(self, images: list[Image.Image], shown_sizes: list[tuple[int, int]] | None = None):
        self.images: list[Image.Image] = list(images)
        self.shown_sizes: list[tuple[int, int]] = list(shown_sizes) if shown_sizes else [im.size for im in self.images]
        if len(self.shown_sizes) != len(self.images):
            raise ValueError("one shown size per image")

    def _index(self, img_idx: Any) -> int:
        idx = as_int(img_idx, "img_idx")
        if not -len(self.images) <= idx < len(self.images):
            raise ToolError(
                f"Error: img_idx {idx} is out of range; valid indices are 0..{len(self.images) - 1}"
            )
        return idx % len(self.images)

    def to_source_box(self, box: list[float], img_idx: Any, coords: str) -> list[float]:
        """Model-frame box -> pixel box in the stored (full-resolution) image."""
        idx = self._index(img_idx)
        sw, sh = self.images[idx].size
        x1, y1, x2, y2 = box
        if coords == "rel1000":  # Qwen-Agent's arithmetic order, kept for pixel parity
            return [x1 / REL_SCALE * sw, y1 / REL_SCALE * sh, x2 / REL_SCALE * sw, y2 / REL_SCALE * sh]
        if coords != "abs":
            raise ValueError(f"coords must be one of {sorted(COORD_PHRASE)}")
        fw, fh = self.shown_sizes[idx]
        if (fw, fh) == (sw, sh):  # tool outputs: shown as stored, the box is already in pixels
            return [float(v) for v in box]
        # multiply first, so integer boxes map exactly when the scale is integral
        return [x1 * sw / fw, y1 * sh / fh, x2 * sw / fw, y2 * sh / fh]

    def to_source_point(self, point: tuple[float, float], img_idx: Any, coords: str) -> tuple[float, float]:
        x1, y1, _, _ = self.to_source_box([point[0], point[1], point[0], point[1]], img_idx, coords)
        return x1, y1

    def to_model_box(self, box: list[float], img_idx: Any, coords: str) -> list[int]:
        """Pixel box in the stored image -> rounded model-frame box."""
        idx = self._index(img_idx)
        sw, sh = self.images[idx].size
        x1, y1, x2, y2 = box
        if coords == "rel1000":  # same arithmetic as before the family change
            return [round(x1 / sw * REL_SCALE), round(y1 / sh * REL_SCALE), round(x2 / sw * REL_SCALE), round(y2 / sh * REL_SCALE)]
        fw, fh = self.shown_sizes[idx]
        return [round(x1 * fw / sw), round(y1 * fh / sh), round(x2 * fw / sw), round(y2 * fh / sh)]

    def get_image(self, img_idx: Any) -> Image.Image:
        """Python list indexing, as Qwen-Agent does (``-1`` is the latest image)."""
        return self.images[self._index(img_idx)]

    def add_images(self, images: list[Image.Image]) -> list[int]:
        """Append images shown to the model at their own size (tool outputs, oracle crops)."""
        start = len(self.images)
        self.images.extend(images)
        self.shown_sizes.extend(im.size for im in images)
        return list(range(start, len(self.images)))


class Tool:
    """Base class: subclasses set ``name``/``description``/``parameters`` and implement ``run``.

    ``family`` fixes the coordinate frame and patch factor; tools whose
    descriptions mention coordinates build them in ``__init__`` from it.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def __init__(self, family: ModelFamily = QWEN3_VL) -> None:
        self.family = family

    def schema(self) -> dict[str, Any]:
        """OpenAI/Qwen function schema, in the key order Qwen-Agent emits."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not isinstance(args, dict):
            return ToolResult.error(f"Error: arguments for {self.name} must be a JSON object")
        try:
            result = self.run(args, ctx)
        except ToolError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # tool bugs must not kill an episode
            return ToolResult.error(f"Tool Execution Error {exc}")
        for image in result.images:
            width, height = image.size
            if max(width, height) > MAX_ASPECT_RATIO * min(width, height):
                # the server would reject the whole request; report it as a tool error instead
                return ToolResult.error(
                    f"Error: the resulting image is {width}x{height}, more elongated than the "
                    f"{MAX_ASPECT_RATIO}:1 the model accepts; choose a less elongated region"
                )
        return result


def as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ToolError(f"Error: {name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ToolError(f"Error: {name} must be an integer, got {value!r}")


def as_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ToolError(f"Error: {name} must be a number")
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            raise ToolError(f"Error: {name} must be a number, got {value!r}") from None
    else:
        raise ToolError(f"Error: {name} must be a number, got {value!r}")
    if not math.isfinite(out):
        raise ToolError(f"Error: {name} must be finite")
    return out


def as_bbox(value: Any, name: str = "bbox_2d") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ToolError(f"Error: {name} must be a list of 4 numbers [x1, y1, x2, y2]")
    return [as_number(v, name) for v in value]


def rel_to_abs_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 / REL_SCALE * width, y1 / REL_SCALE * height, x2 / REL_SCALE * width, y2 / REL_SCALE * height]


def abs_to_rel_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        round(x1 / width * REL_SCALE),
        round(y1 / height * REL_SCALE),
        round(x2 / width * REL_SCALE),
        round(y2 / height * REL_SCALE),
    ]


# --- qwen-vl-utils resize helpers (as copied into Qwen-Agent's zoom tool) ---


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: float,
    width: float,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 12845056,
) -> tuple[int, int]:
    """Return (height, width) rounded to ``factor`` with area in [min_pixels, max_pixels]."""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fit_image(
    image: Image.Image,
    *,
    min_pixels: int,
    max_pixels: int,
    factor: int = 32,
) -> Image.Image:
    """Resize like the Qwen-VL processors (factor 28 for Qwen2.5-VL, 32 for Qwen3-VL),
    so the server applies no further resize."""
    width, height = image.size
    new_h, new_w = smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), resample=Image.BICUBIC)
