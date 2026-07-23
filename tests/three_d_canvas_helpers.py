"""Shared assertions for 3D canonical canvas policy."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.three_d.shared.canvas import CANONICAL_CANVAS_PRESETS, MAX_FINAL_PIXELS


CANONICAL_THREE_D_CANVAS_SIZES = set(CANONICAL_CANVAS_PRESETS.values())


def assert_three_d_canvas_contract(output: Any) -> None:
    """Assert a 3D task used a canonical source canvas and capped final image."""

    render_spec = output.trace_payload["render_spec"]
    source_size = (
        int(render_spec["scene_canvas_width"]),
        int(render_spec["scene_canvas_height"]),
    )
    final_size = (int(output.image.width), int(output.image.height))
    final_pixels = int(output.image.width * output.image.height)

    assert str(render_spec["scene_canvas_preset"]) in CANONICAL_CANVAS_PRESETS
    assert source_size in CANONICAL_THREE_D_CANVAS_SIZES
    assert int(render_spec["final_canvas_width"]) == final_size[0]
    assert int(render_spec["final_canvas_height"]) == final_size[1]
    assert int(render_spec["final_canvas_pixels"]) == final_pixels
    assert final_pixels <= MAX_FINAL_PIXELS
