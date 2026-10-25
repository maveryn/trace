"""OCR tool: detected text lines with boxes in the model family's grounding frame
(absolute pixels of the image as shown for Qwen2.5-VL, relative [0, 1000] for Qwen3-VL).

Backends (selected with ``TRACE_TOOL_OCR_BACKEND`` or ``OCRTool(backend=...)``):
- ``rapidocr`` (default): RapidOCR 3.x on onnxruntime, PP-OCR models bundled
  with the wheel, CPU only. Apache-2.0.
- ``tesseract``: pytesseract + the ``tesseract`` system binary. Apache-2.0.

Engines are created one per thread (tool calls run in a thread pool) and shared
by every OCRTool in the process; ``TRACE_TOOL_OCR_THREADS`` (default 4) sets
onnxruntime threads per engine. Results are cached process-wide by image content
and region, and concurrent identical requests run the engine once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ..models import QWEN3_VL, ModelFamily
from .base import COORD_PHRASE, Tool, ToolContext, ToolError, ToolResult, as_bbox, smart_resize
from .zoom import maybe_resize_bbox

MAX_LINES = 120
DEFAULT_MIN_SCORE = 0.5
# Small crops are upscaled before recognition, like the zoom tool does.
OCR_MIN_PIXELS = 256 * 32 * 32
OCR_MAX_PIXELS = 4194304


@dataclass(frozen=True)
class OCRLine:
    text: str
    box: tuple[float, float, float, float]  # absolute pixels in the OCR'd image
    score: float


class OCRBackend(Protocol):
    def __call__(self, image: Image.Image) -> list[OCRLine]: ...


class RapidOCRBackend:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(f"OCR backend 'rapidocr' cannot be imported ({exc})") from exc
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        # one engine per tool thread, so keep each onnxruntime session small
        threads = int(os.environ.get("TRACE_TOOL_OCR_THREADS", "4"))
        self._engine = RapidOCR(
            params={
                "Global.log_level": "warning",
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    def __call__(self, image: Image.Image) -> list[OCRLine]:
        out = self._engine(np.asarray(image.convert("RGB")))
        if out is None or out.boxes is None or out.txts is None:
            return []
        lines = []
        for quad, text, score in zip(out.boxes, out.txts, out.scores):
            quad = np.asarray(quad, dtype=float)
            box = (float(quad[:, 0].min()), float(quad[:, 1].min()), float(quad[:, 0].max()), float(quad[:, 1].max()))
            lines.append(OCRLine(text=str(text), box=box, score=float(score)))
        return lines


class TesseractBackend:
    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("OCR backend 'tesseract' needs `pip install trace-tool[tesseract]`") from exc
        self._tess = pytesseract

    def __call__(self, image: Image.Image) -> list[OCRLine]:
        data = self._tess.image_to_data(image.convert("RGB"), output_type=self._tess.Output.DICT)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(data["text"]):
            if word.strip() and float(data["conf"][i]) >= 0:
                groups.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
        lines = []
        for idxs in groups.values():
            x1 = min(data["left"][i] for i in idxs)
            y1 = min(data["top"][i] for i in idxs)
            x2 = max(data["left"][i] + data["width"][i] for i in idxs)
            y2 = max(data["top"][i] + data["height"][i] for i in idxs)
            text = " ".join(data["text"][i] for i in idxs)
            score = float(np.mean([float(data["conf"][i]) for i in idxs])) / 100.0
            lines.append(OCRLine(text=text, box=(x1, y1, x2, y2), score=score))
        return lines


_BACKENDS = {"rapidocr": RapidOCRBackend, "tesseract": TesseractBackend}


_BACKEND_HINTS = {
    # vLLM pins opencv-python-headless and RapidOCR pulls opencv-python; both own cv2/,
    # so remove both before reinstalling the headless build.
    "rapidocr": "pip install 'trace-tool[ocr]'; on a headless Linux server (or next to vLLM) run "
    "`pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless`",
    "tesseract": "pip install 'trace-tool[tesseract]' and install the tesseract binary",
}


class _ThreadLocalBackend:
    def __init__(self, name: str) -> None:
        if name not in _BACKENDS:
            raise ValueError(f"unknown OCR backend {name!r}; choose from {sorted(_BACKENDS)}")
        self.name = name
        self._local = threading.local()
        # Build one engine now: a broken install (e.g. OpenCV without libGL) must stop
        # the run at startup instead of turning every OCR call into an error message.
        try:
            self._local.engine = _BACKENDS[name]()
        except Exception as exc:
            raise RuntimeError(f"OCR backend {name!r} is unusable ({type(exc).__name__}: {exc}). {_BACKEND_HINTS[name]}") from exc

    def __call__(self, image: Image.Image) -> list[OCRLine]:
        engine = getattr(self._local, "engine", None)
        if engine is None:
            engine = self._local.engine = _BACKENDS[self.name]()
        return engine(image)


_SHARED_BACKENDS: dict[str, _ThreadLocalBackend] = {}
_SHARED_LOCK = threading.Lock()


def shared_backend(name: str) -> _ThreadLocalBackend:
    """One set of per-thread engines per backend for the whole process."""
    with _SHARED_LOCK:
        if name not in _SHARED_BACKENDS:
            _SHARED_BACKENDS[name] = _ThreadLocalBackend(name)
        return _SHARED_BACKENDS[name]


def backend_versions(name: str) -> dict[str, str]:
    """Package versions recorded in run_config.json (OCR output depends on them)."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {"rapidocr": ["rapidocr", "onnxruntime", "opencv-python", "opencv-python-headless"], "tesseract": ["pytesseract"]}[name]
    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            continue
    return out


class _SingleFlightCache:
    """LRU of OCR results; concurrent requests for the same key wait for one computation."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.lock = threading.Lock()
        self.done: OrderedDict[tuple, list] = OrderedDict()
        self.pending: dict[tuple, Future] = {}

    def get(self, key: tuple, compute):
        with self.lock:
            if key in self.done:
                self.done.move_to_end(key)
                return self.done[key]
            future = self.pending.get(key)
            owner = future is None
            if owner:
                future = self.pending[key] = Future()
        if not owner:
            return future.result()
        try:
            value = compute()
        except BaseException as exc:
            with self.lock:
                self.pending.pop(key, None)
            future.set_exception(exc)
            raise
        with self.lock:
            self.done[key] = value
            while len(self.done) > self.size:
                self.done.popitem(last=False)
            self.pending.pop(key, None)
        future.set_result(value)
        return value


# The same image is OCR'd repeatedly (seeds, conditions, GRPO groups): share results process-wide.
_RESULT_CACHE = _SingleFlightCache(8192)


def _reading_order(lines: list[OCRLine]) -> list[OCRLine]:
    """Sort top-to-bottom, grouping lines whose vertical centres are close, then left-to-right."""
    if not lines:
        return []
    heights = sorted(line.box[3] - line.box[1] for line in lines)
    row_tol = max(4.0, 0.5 * heights[len(heights) // 2])
    rows: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        cy = (line.box[1] + line.box[3]) / 2
        if rows and abs(cy - np.mean([(l.box[1] + l.box[3]) / 2 for l in rows[-1]])) <= row_tol:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda l: l.box[0])]


def run_ocr(
    backend: OCRBackend,
    image: Image.Image,
    pixel_box: list[float] | None = None,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_side: int = 32,
) -> list[tuple[str, list[float], float]]:
    """OCR ``image`` (optionally only ``pixel_box``, grown to min_side); boxes come back in pixels of ``image``."""
    width, height = image.size
    left, top, right, bottom = 0.0, 0.0, float(width), float(height)
    if pixel_box is not None:
        left, top, right, bottom = maybe_resize_bbox(*pixel_box, width, height, min_side=min_side)
        if right - left < 1 or bottom - top < 1:
            raise ToolError("Error: bbox_2d is empty after clipping to the image")
    region = image.crop((left, top, right, bottom)) if pixel_box is not None else image
    rw, rh = region.size
    new_h, new_w = smart_resize(rh, rw, factor=1, min_pixels=OCR_MIN_PIXELS, max_pixels=OCR_MAX_PIXELS)
    scale_x, scale_y = new_w / rw, new_h / rh
    if (new_w, new_h) != (rw, rh):
        region = region.resize((new_w, new_h), resample=Image.BICUBIC)

    results = []
    for line in _reading_order(backend(region)):
        if line.score < min_score or not line.text.strip():
            continue
        x1, y1, x2, y2 = line.box
        box = [left + x1 / scale_x, top + y1 / scale_y, left + x2 / scale_x, top + y2 / scale_y]
        results.append((line.text.strip(), box, line.score))
    return results


class OCRTool(Tool):
    name = "ocr"

    def __init__(self, family: ModelFamily = QWEN3_VL, backend: str | None = None, min_score: float = DEFAULT_MIN_SCORE) -> None:
        super().__init__(family)
        frame = COORD_PHRASE[family.coords]
        # Qwen3-VL keeps the original wording; "of that image" is already part of the pixel phrase
        of_image = " of that image" if family.coords == "rel1000" else ""
        self.description = (
            "Read the text in an image (or in one region of it) with an OCR engine. Returns each detected "
            f"text line with its bounding box [x1, y1, x2, y2] in {frame}{of_image}"
        )
        self._header_frame = "in [0, 1000]" if family.coords == "rel1000" else "in pixels of that image as you see it"
        self.parameters = {
            "type": "object",
            "properties": {
                "img_idx": {
                    "type": "number",
                    "description": "The index of the image to read (starting from 0)",
                },
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": f"Optional region to read, as [x1, y1, x2, y2] in {frame}; omit to read the whole image",
                },
            },
            "required": ["img_idx"],
        }
        self.backend_name = backend or os.environ.get("TRACE_TOOL_OCR_BACKEND", "rapidocr")
        self.backend: OCRBackend = shared_backend(self.backend_name)
        self.min_score = min_score

    def _ocr_cached(self, image: Image.Image, pixel_box: list[float] | None) -> list[tuple[str, list[float], float]]:
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        key = (
            self.backend_name,
            self.min_score,
            digest,
            image.size,
            image.mode,
            None if pixel_box is None else tuple(round(v, 1) for v in pixel_box),
            self.family.zoom_min_side,
        )
        backend = self.backend  # tests swap in a counting wrapper
        return _RESULT_CACHE.get(
            key, lambda: run_ocr(backend, image, pixel_box, min_score=self.min_score, min_side=self.family.zoom_min_side)
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        img_idx = args.get("img_idx", 0)
        image = ctx.get_image(img_idx)
        model_box = as_bbox(args["bbox_2d"]) if args.get("bbox_2d") is not None else None
        pixel_box = ctx.to_source_box(model_box, img_idx, self.family.coords) if model_box is not None else None
        lines = self._ocr_cached(image, pixel_box)
        if not lines:
            return ToolResult(text="No text detected.", meta={"n_lines": 0})
        shown = lines[:MAX_LINES]
        header = f"Detected {len(lines)} text lines in image {img_idx}"
        if model_box is not None:
            header += f" region {[round(v) for v in model_box]}"
        header += f" (text, bbox_2d {self._header_frame}):"
        body = "\n".join(
            f"{i + 1}. {json.dumps(text, ensure_ascii=False)} {ctx.to_model_box(box, img_idx, self.family.coords)}"
            for i, (text, box, _) in enumerate(shown)
        )
        if len(lines) > MAX_LINES:
            body += f"\n... {len(lines) - MAX_LINES} more lines omitted; pass bbox_2d to read a smaller region."
        return ToolResult(text=f"{header}\n{body}", meta={"n_lines": len(lines)})
