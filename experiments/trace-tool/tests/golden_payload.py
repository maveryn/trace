"""Fingerprints of a request payload for the golden tests (tests/test_conditions.py).

The images in a payload are PNG data URLs, and PNG bytes are the deflate output of the
host's zlib: Pillow links the system libz.so.1, and zlib-ng-compat, for example, writes
different bytes for the same pixels. So the raw payload hash depends on the host, not
only on this code. Next to it, :func:`fingerprint` hashes the payload with every PNG data
URL replaced by the decoded image's mode, size and pixels (what the server decodes),
which pins every text byte and every image without depending on the host.

This module imports nothing from trace_tool, so the goldens can be captured with the
code of another commit.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from collections.abc import Iterator
from typing import Any

from PIL import Image

PNG_DATA_URL = "data:image/png;base64,"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dumps(payload: list[dict[str, Any]]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _png_parts(payload: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for message in payload:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url" and part["image_url"]["url"].startswith(PNG_DATA_URL):
                    yield part


def _png_bytes(part: dict[str, Any]) -> bytes:
    return base64.b64decode(part["image_url"]["url"][len(PNG_DATA_URL) :], validate=True)


def fingerprint(payload: list[dict[str, Any]]) -> dict[str, Any]:
    """``payload_sha256``: the payload exactly as sent (PNG bytes depend on the host's zlib);
    ``payload_pixels_sha256``: the same with each PNG data URL replaced by its decoded mode,
    size and pixel sha256 (host-independent); ``png_sha256``: each PNG as sent, in order."""
    pixels = copy.deepcopy(payload)
    png = []
    for part in _png_parts(pixels):
        data = _png_bytes(part)
        png.append(_sha256(data))
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            part["image_url"]["url"] = f"png-pixels:{image.mode}:{image.width}x{image.height}:{_sha256(image.tobytes())}"
    return {
        "payload_sha256": _sha256(_dumps(payload).encode()),
        "payload_pixels_sha256": _sha256(_dumps(pixels).encode()),
        "png_sha256": png,
    }


def reencoded_png_sha256(payload: list[dict[str, Any]]) -> list[str]:
    """Each PNG decoded and encoded again by this host's Pillow with its default settings,
    independently of the code under test. On a host whose encoder writes the capture
    host's bytes this equals the captured ``png_sha256``."""
    out = []
    for part in _png_parts(payload):
        with Image.open(io.BytesIO(_png_bytes(part))) as image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            out.append(_sha256(buffer.getvalue()))
    return out
