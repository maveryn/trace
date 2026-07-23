"""Tests for procedural named icon rendering."""

from __future__ import annotations

import pytest

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.icons.shared.annotation import (
    ICON_OBJECT_ANNOTATION_MIN_SIDE_PX,
    icon_bbox_set_annotation,
)
from trace_tasks.tasks.icons.shared.procedural_named_icon_field_scene import rotation_for_named_shape
from trace_tasks.tasks.icons.shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_DISPLAY_NAMES,
    PROCEDURAL_NAMED_ICON_FILL_STYLES,
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    render_procedural_named_icon_rgba,
)


def test_procedural_named_icon_vocabulary_has_clear_shapes() -> None:
    assert len(PROCEDURAL_NAMED_ICON_SHAPES) == 100
    assert len(set(PROCEDURAL_NAMED_ICON_SHAPES)) == len(PROCEDURAL_NAMED_ICON_SHAPES)
    assert set(PROCEDURAL_NAMED_ICON_SHAPES) == set(PROCEDURAL_NAMED_ICON_DISPLAY_NAMES)
    assert procedural_named_icon_display_name("lightning_bolt") == "lightning bolt"
    assert procedural_named_icon_display_name("rocket") == "rocket"
    assert procedural_named_icon_display_name("magnifying_glass") == "magnifying glass"
    assert procedural_named_icon_display_name("shuriken") == "shuriken"
    assert procedural_named_icon_display_name("tree") == "tree"
    assert procedural_named_icon_display_name("rugby_ball") == "rugby ball"
    assert procedural_named_icon_display_name("broccoli") == "broccoli"
    assert procedural_named_icon_display_name("cactus") == "cactus"
    assert procedural_named_icon_display_name("guitar") == "guitar"
    assert procedural_named_icon_display_name("acorn") == "acorn"


def test_procedural_named_icon_renderer_outputs_nonempty_cropped_rgba() -> None:
    for shape_id in PROCEDURAL_NAMED_ICON_SHAPES:
        image = render_procedural_named_icon_rgba(
            shape_id=shape_id,
            size_px=80,
            tint_rgb=(20, 90, 180),
        )
        assert image.mode == "RGBA"
        assert 10 <= image.size[0] <= 80
        assert 10 <= image.size[1] <= 80
        assert image.getchannel("A").getbbox() is not None


def test_procedural_named_icon_renderer_outputs_supported_fill_styles() -> None:
    for fill_style in PROCEDURAL_NAMED_ICON_FILL_STYLES:
        image = render_procedural_named_icon_rgba(
            shape_id="star",
            size_px=96,
            tint_rgb=(20, 90, 180),
            fill_style=str(fill_style),
        )
        assert image.mode == "RGBA"
        assert image.getchannel("A").getbbox() is not None


def test_procedural_named_icon_thin_canonical_shapes_have_usable_48px_bboxes() -> None:
    for shape_id in ("plug", "fork", "spoon", "toothbrush", "key", "knife"):
        image = render_procedural_named_icon_rgba(
            shape_id=shape_id,
            size_px=48,
            tint_rgb=(20, 90, 180),
        )
        assert min(image.size) >= ICON_OBJECT_ANNOTATION_MIN_SIDE_PX


def test_named_icon_nonsemantic_rotation_jitter_is_small() -> None:
    rng = spawn_rng(2026061901, "named_icon_jitter")
    values = [rotation_for_named_shape(rng, shape_id) for shape_id in PROCEDURAL_NAMED_ICON_SHAPES for _ in range(3)]
    assert all(-15 <= int(value) <= 15 for value in values)
    assert any(int(value) < 0 for value in values)
    assert any(int(value) > 0 for value in values)


def test_icon_bbox_set_annotation_expands_small_icon_witnesses_to_min_side() -> None:
    annotation = icon_bbox_set_annotation([[10, 20, 20, 30], [40, 50, 72, 78]])
    assert annotation["annotation_type"] == "bbox_set"
    for bbox in annotation["annotation_value"]:
        assert min(int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])) >= ICON_OBJECT_ANNOTATION_MIN_SIDE_PX


def test_procedural_named_icon_renderer_rejects_unknown_shape() -> None:
    with pytest.raises(KeyError):
        render_procedural_named_icon_rgba(
            shape_id="not_a_shape",
            size_px=80,
            tint_rgb=(20, 90, 180),
        )
