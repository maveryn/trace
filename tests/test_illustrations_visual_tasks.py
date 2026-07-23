"""Tests for reusable illustration visual reconstruction mechanics."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from trace_tasks.tasks.autoload import task_module_exists
from trace_tasks.tasks.illustrations.shared.cutouts import (
    JIGSAW_BOARD_STYLES,
    PATCH_FRAME_STYLES,
    PATCH_MODE_PLAIN,
    ROTATED_GRID_STYLES,
    compose_jigsaw_arrangement_options,
    compose_jigsaw_board,
    compose_patch_options,
    compose_rotated_tile_grid,
    downscale_jigsaw_arrangement_artifacts,
    option_content_order,
    sample_style,
    tile_is_usable,
)
from trace_tasks.tasks.illustrations.shared.canvas_profiles import (
    MAX_RECONSTRUCTION_OUTPUT_PIXELS,
    reconstruction_grid_for_size,
    reconstruction_option_labels,
    resize_to_max_pixels,
    resolve_canvas_profile,
)


RETIRED_TASK_ID_PARTS = (
    ("illustrations", "image_cutout_board", "jigsaw_piece_order"),
    ("illustrations", "image_cutout_board", "rotated_tile_label"),
    ("illustrations", "missing_patch", "missing_patch_label"),
)


def _source_image(width: int = 640, height: int = 420) -> Image.Image:
    image = Image.new("RGB", (int(width), int(height)), (238, 242, 246))
    draw = ImageDraw.Draw(image)
    colors = [(210, 58, 70), (40, 120, 210), (54, 160, 98), (230, 174, 42)]
    for index, color in enumerate(colors):
        x0 = 24 + index * (width // 5)
        y0 = 36 + (index % 2) * 110
        draw.rounded_rectangle((x0, y0, x0 + 110, y0 + 86), radius=14, fill=color, outline=(35, 39, 46), width=4)
        draw.line((x0 + 12, y0 + 14, x0 + 96, y0 + 70), fill=(255, 255, 255), width=5)
    for x in range(0, width, 40):
        draw.line((x, 0, width - x // 2, height), fill=(90, 100, 120), width=1)
    return image


def test_retired_visual_public_tasks_are_not_registered() -> None:
    for domain, scene_id, objective in RETIRED_TASK_ID_PARTS:
        task_id = f"task_{domain}__{scene_id}__{objective}"
        assert not task_module_exists(task_id)


def test_jigsaw_artifacts_bind_answer_order_and_bbox_sequence() -> None:
    rng = random.Random(17)
    style = sample_style(rng, JIGSAW_BOARD_STYLES)
    display_order = option_content_order(
        option_permutation_index=3,
        rng=rng,
        remaining_content_indices=(1, 2, 3),
    )
    artifacts = compose_jigsaw_board(
        source_image=_source_image(),
        rows=2,
        cols=2,
        display_order=display_order,
        board_style=style,
        label_font_family=None,  # type: ignore[arg-type]
    )

    assert artifacts.display_grid_shape == (2, 2)
    assert artifacts.anchored_content_index == 0
    assert artifacts.display_order_content_indices == display_order
    assert set(artifacts.option_bboxes) == {"1", "2", "3"}
    assert len(artifacts.answer_labels) == 3
    assert set(artifacts.answer_labels) == {"1", "2", "3"}
    assert [artifacts.option_bboxes[label] for label in artifacts.answer_labels]
    assert artifacts.image.width > 0 and artifacts.image.height > 0


def test_jigsaw_arrangement_options_bind_single_correct_option() -> None:
    rng = random.Random(19)
    style = sample_style(rng, JIGSAW_BOARD_STYLES)
    artifacts = compose_jigsaw_arrangement_options(
        source_image=_source_image(520, 366),
        rows=2,
        cols=2,
        correct_index=2,
        rng=rng,
        board_style=style,
        label_font_family=None,  # type: ignore[arg-type]
    )

    assert artifacts.grid_shape == (2, 2)
    assert artifacts.option_layout_shape == (2, 2)
    assert set(artifacts.option_bboxes) == {"A", "B", "C", "D"}
    assert artifacts.selected_label == "C"
    assert artifacts.selected_index == 2
    assert artifacts.option_permutations[2] == artifacts.correct_permutation
    assert artifacts.correct_permutation == (0, 1, 2, 3)
    assert sum(perm == artifacts.correct_permutation for perm in artifacts.option_permutations) == 1
    assert len(set(artifacts.option_permutations)) == 4
    for index, permutation in enumerate(artifacts.option_permutations):
        if index == artifacts.selected_index:
            continue
        assert sum(left != right for left, right in zip(permutation, artifacts.correct_permutation)) == 2
    assert artifacts.selected_option_bbox == artifacts.option_bboxes["C"]
    assert artifacts.image.size == (1136, 892)


def test_rotated_tile_artifacts_select_one_labeled_tile() -> None:
    source = _source_image(600, 600)
    rng = random.Random(23)
    style = sample_style(rng, ROTATED_GRID_STYLES)
    artifacts = compose_rotated_tile_grid(
        source_image=source,
        correct_index=4,
        rotation_degrees=90,
        grid_style=style,
        label_font_family=None,  # type: ignore[arg-type]
    )
    pieces = source.crop((200, 200, 400, 400))
    rotated = pieces.rotate(-90, expand=False)

    assert artifacts.grid_shape == (3, 3)
    assert artifacts.selected_label == "E"
    assert artifacts.selected_index == 4
    assert artifacts.rotation_degrees == 90
    assert artifacts.selected_bbox == artifacts.tile_bboxes["E"]
    assert len(artifacts.tile_bboxes) == 9
    assert tile_is_usable(pieces, rotated, min_detail_score=1.0, min_rotation_delta=1.0)


def test_rotated_tile_grid_draws_no_perimeter_border_when_frameless() -> None:
    source = _source_image(600, 600)
    style = {"style_id": "test", **dict(ROTATED_GRID_STYLES["slate_badges"])}
    artifacts = compose_rotated_tile_grid(
        source_image=source,
        correct_index=4,
        rotation_degrees=90,
        grid_style=style,
        label_font_family=None,  # type: ignore[arg-type]
        render_margin=0,
    )

    assert artifacts.image.size == source.size
    assert artifacts.image.getpixel((0, 0)) == source.getpixel((0, 0))
    assert artifacts.image.getpixel((source.width - 1, source.height - 1)) == source.getpixel(
        (source.width - 1, source.height - 1)
    )
    assert artifacts.image.getpixel((200, 100)) == style["grid_rgb"]
    assert artifacts.image.getpixel((100, 200)) == style["grid_rgb"]


def test_rotated_tile_rejects_non_square_quarter_turn_cells() -> None:
    with pytest.raises(ValueError, match="square cells"):
        compose_rotated_tile_grid(
            source_image=_source_image(960, 720),
            correct_index=2,
            rotation_degrees=90,
            grid_style=ROTATED_GRID_STYLES["slate_badges"],
            label_font_family=None,  # type: ignore[arg-type]
            rows=2,
            cols=3,
        )


def test_canvas_profiles_resolve_profile_grids_and_labels() -> None:
    defaults = {"canvas_profile_support": ["landscape", "square", "portrait"]}
    assert resolve_canvas_profile(
        params={"canvas_profile": "landscape"},
        defaults=defaults,
        fallback_width=640,
        fallback_height=420,
    ).size == (1200, 800)
    assert resolve_canvas_profile(
        params={"canvas_profile": "square"},
        defaults=defaults,
        fallback_width=640,
        fallback_height=420,
    ).size == (960, 960)
    assert resolve_canvas_profile(
        params={"canvas_profile": "portrait"},
        defaults=defaults,
        fallback_width=640,
        fallback_height=420,
    ).size == (800, 1200)
    assert reconstruction_grid_for_size(1200, 800) == (2, 3)
    assert reconstruction_grid_for_size(960, 960) == (3, 3)
    assert reconstruction_grid_for_size(800, 1200) == (3, 2)
    assert reconstruction_option_labels(2, 2) == ("A", "B", "C", "D")
    assert reconstruction_option_labels(2, 3) == ("A", "B", "C", "D", "E", "F")
    assert reconstruction_option_labels(3, 3) == ("A", "B", "C", "D", "E", "F", "G", "H", "I")


def test_downscale_helpers_scale_jigsaw_bboxes_under_pixel_cap() -> None:
    rng = random.Random(29)
    artifacts = compose_jigsaw_arrangement_options(
        source_image=_source_image(1200, 800),
        rows=2,
        cols=3,
        correct_index=1,
        rng=rng,
        board_style=JIGSAW_BOARD_STYLES["pale_cross"],
        label_font_family=None,  # type: ignore[arg-type]
        labels=("A", "B", "C", "D"),
    )

    assert artifacts.image.width * artifacts.image.height > MAX_RECONSTRUCTION_OUTPUT_PIXELS
    scaled = downscale_jigsaw_arrangement_artifacts(
        artifacts,
        max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
    )
    assert scaled.image.width * scaled.image.height <= MAX_RECONSTRUCTION_OUTPUT_PIXELS
    assert scaled.pre_downscale_canvas_size == artifacts.image.size
    assert scaled.output_scale_xy[0] < 1.0
    assert scaled.selected_option_bbox == scaled.option_bboxes[scaled.selected_label]
    resized, scale_x, scale_y = resize_to_max_pixels(artifacts.image, max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS)
    assert resized.size == scaled.image.size
    assert scaled.output_scale_xy == (scale_x, scale_y)


def test_patch_option_artifacts_use_keyed_visual_witnesses() -> None:
    rng = random.Random(31)
    style = sample_style(rng, PATCH_FRAME_STYLES)
    artifacts = compose_patch_options(
        source_image=_source_image(),
        rng=rng,
        patch_mode=PATCH_MODE_PLAIN,
        correct_index=2,
        option_count=4,
        patch_size=(128, 100),
        crop_margin_px=32,
        frame_style=style,
        label_font_family=None,  # type: ignore[arg-type]
    )

    assert artifacts.option_grid_shape == (2, 2)
    assert artifacts.selected_label == "C"
    assert artifacts.selected_index == 2
    assert set(artifacts.option_bboxes) == {"A", "B", "C", "D"}
    assert artifacts.selected_option_bbox == artifacts.option_bboxes["C"]
    assert len(artifacts.missing_region_bbox) == 4
    assert len(artifacts.selected_option_bbox) == 4
    assert artifacts.selected_transform == "none"
    assert artifacts.option_source_crop_boxes.count(artifacts.source_crop_box) == 1
    missing_w = artifacts.missing_region_bbox[2] - artifacts.missing_region_bbox[0]
    missing_h = artifacts.missing_region_bbox[3] - artifacts.missing_region_bbox[1]
    for bbox in artifacts.option_bboxes.values():
        assert bbox[2] - bbox[0] == missing_w
        assert bbox[3] - bbox[1] == missing_h


def test_patch_option_candidate_crop_boxes_are_honored() -> None:
    candidates = (
        (36, 40, 164, 140),
        (220, 42, 348, 142),
        (408, 48, 536, 148),
        (52, 232, 180, 332),
        (252, 244, 380, 344),
        (438, 250, 566, 350),
    )
    artifacts = compose_patch_options(
        source_image=_source_image(),
        rng=random.Random(37),
        patch_mode=PATCH_MODE_PLAIN,
        correct_index=1,
        option_count=4,
        patch_size=(128, 100),
        crop_margin_px=32,
        frame_style=PATCH_FRAME_STYLES["slate_cards"],
        label_font_family=None,  # type: ignore[arg-type]
        candidate_crop_boxes=candidates,
    )

    assert artifacts.candidate_crop_count == len(candidates)
    assert artifacts.source_crop_box in candidates
    assert len(artifacts.option_source_crop_boxes) == 4
    assert set(artifacts.option_source_crop_boxes).issubset(set(candidates))
    assert artifacts.option_source_crop_boxes[1] == artifacts.source_crop_box


def test_visual_shared_helpers_do_not_reference_public_task_plumbing() -> None:
    forbidden = {
        "TaskOutput",
        "create_task",
        "register_task",
        "source_task_id",
        "SOURCE_TASK_IDS",
    }
    for path in (
        Path("src/trace_tasks/tasks/illustrations/shared/cutouts.py"),
        Path("src/trace_tasks/tasks/illustrations/shared/option_rendering.py"),
    ):
        text = path.read_text(encoding="utf-8")
        present = sorted(value for value in forbidden if value in text)
        assert not present, f"{path} contains forbidden public plumbing: {present}"


def test_patch_mode_validation_rejects_legacy_query_names() -> None:
    with pytest.raises(ValueError):
        compose_patch_options(
            source_image=_source_image(),
            rng=random.Random(47),
            patch_mode="plain_patch_label",
            correct_index=0,
            option_count=4,
            patch_size=(128, 100),
            crop_margin_px=32,
            frame_style=PATCH_FRAME_STYLES["slate_cards"],
            label_font_family=None,  # type: ignore[arg-type]
        )
