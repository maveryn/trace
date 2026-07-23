"""Rendering primitives for mirror-grid icon scenes."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageOps

from ....shared.text_rendering import load_font
from ....shared.text_legibility import draw_text_traced
from ...shared.icon_assets import render_icon_rgba, resolve_icon_pool
from ...shared.icon_grid_scene import centered_square_bbox, resolve_fixed_grid_cell_slots
from ...shared.icon_labeled_grid_scene import prepare_two_panel_labeled_grid_scene
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import panel_geometry_to_trace
from ...shared.icon_task_rendering import sample_icon_instance_noise

from .sampling import (
    NONSYMMETRIC_KIND,
    SYMMETRY_KINDS,
    fixed_grid_labels,
    sample_distractor_symmetry_kinds,
    sample_matching_indices,
)
from .state import MirrorGridCompletionScenePayload, MirrorGridScenePayload
from .styles import sample_mirror_grid_palette


_EXACT_SYMMETRY_SIGNATURES: Dict[str, Tuple[bool, bool, bool, bool]] = {
    "vertical": (True, False, False, False),
    "horizontal": (False, True, False, False),
    "diagonal_main": (False, False, True, False),
    "diagonal_anti": (False, False, False, True),
    "both_axes": (True, True, False, False),
}


def _reflection_name(symmetry_kind: str) -> str:
    """Return the trace name for the image operation applied to a paired sprite."""

    if str(symmetry_kind) == "vertical":
        return "horizontal_flip"
    if str(symmetry_kind) == "horizontal":
        return "vertical_flip"
    if str(symmetry_kind) == "diagonal_main":
        return "diagonal_main_flip"
    if str(symmetry_kind) == "diagonal_anti":
        return "diagonal_anti_flip"
    if str(symmetry_kind) == "both_axes":
        return "both_axes_flip"
    raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")


def _flip_image(image: Image.Image, symmetry_kind: str) -> Image.Image:
    """Return one image flipped across the requested symmetry operation."""

    if str(symmetry_kind) == "vertical":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if str(symmetry_kind) == "horizontal":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    if str(symmetry_kind) == "diagonal_main":
        return image.transpose(Image.TRANSPOSE)
    if str(symmetry_kind) == "diagonal_anti":
        return image.transpose(Image.TRANSVERSE)
    if str(symmetry_kind) == "both_axes":
        return image.transpose(Image.ROTATE_180)
    raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")


def _has_mirror_symmetry(image: Image.Image, symmetry_kind: str) -> bool:
    """Return true when the rendered patch is exactly symmetric under one operation."""

    if str(symmetry_kind) in {"diagonal_main", "diagonal_anti"} and int(image.size[0]) != int(image.size[1]):
        return False
    flipped = _flip_image(image, str(symmetry_kind))
    return ImageChops.difference(image, flipped).getbbox() is None


def _symmetry_signature(image: Image.Image) -> Tuple[bool, bool, bool, bool]:
    """Return the exact `(vertical, horizontal, main diagonal, anti diagonal)` flags."""

    return (
        bool(_has_mirror_symmetry(image, "vertical")),
        bool(_has_mirror_symmetry(image, "horizontal")),
        bool(_has_mirror_symmetry(image, "diagonal_main")),
        bool(_has_mirror_symmetry(image, "diagonal_anti")),
    )


def _is_exact_symmetry_kind(image: Image.Image, symmetry_kind: str) -> bool:
    """Return true only when a patch has exactly the requested symmetry kind."""

    expected = _EXACT_SYMMETRY_SIGNATURES.get(str(symmetry_kind))
    if expected is None:
        raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")
    return tuple(bool(value) for value in _symmetry_signature(image)) == tuple(bool(value) for value in expected)


def _rects_intersect(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int], pad: int = 0) -> bool:
    """Return true when two `xywh` rectangles overlap after gap padding."""

    lx, ly, lw, lh = [int(value) for value in left]
    rx, ry, rw, rh = [int(value) for value in right]
    padding = int(max(0, pad))
    return not (
        lx + lw + padding <= rx
        or rx + rw + padding <= lx
        or ly + lh + padding <= ry
        or ry + rh + padding <= ly
    )


def _scale_bounds_for_count(icon_count: int) -> Tuple[float, float]:
    """Return icon-size scale bounds tuned to the patch icon count."""

    if int(icon_count) <= 4:
        return (0.22, 0.38)
    if int(icon_count) <= 6:
        return (0.18, 0.32)
    return (0.14, 0.28)


def _random_position_within_patch(
    rng,
    *,
    width: int,
    height: int,
    sprite_w: int,
    sprite_h: int,
    inner_margin_px: int,
) -> Tuple[int, int] | None:
    """Sample one random top-left position inside a patch."""

    x_min = int(inner_margin_px)
    x_max = int(width - inner_margin_px - sprite_w)
    y_min = int(inner_margin_px)
    y_max = int(height - inner_margin_px - sprite_h)
    if x_min > x_max or y_min > y_max:
        return None
    return int(rng.randint(int(x_min), int(x_max))), int(rng.randint(int(y_min), int(y_max)))


def _sample_nonoverlapping_position(
    rng,
    *,
    width: int,
    height: int,
    sprite_w: int,
    sprite_h: int,
    inner_margin_px: int,
    min_gap_px: int,
    rects_xywh: Sequence[Tuple[int, int, int, int]],
    attempts: int,
) -> Tuple[int, int] | None:
    """Sample a top-left sprite position that does not collide with existing sprites."""

    for _ in range(max(1, int(attempts))):
        position = _random_position_within_patch(
            rng,
            width=int(width),
            height=int(height),
            sprite_w=int(sprite_w),
            sprite_h=int(sprite_h),
            inner_margin_px=int(inner_margin_px),
        )
        if position is None:
            return None
        rect_xywh = (int(position[0]), int(position[1]), int(sprite_w), int(sprite_h))
        if any(_rects_intersect(rect_xywh, other, pad=int(min_gap_px)) for other in rects_xywh):
            continue
        return int(position[0]), int(position[1])
    return None


def _sample_symmetric_seed_position(
    rng,
    *,
    width: int,
    height: int,
    sprite_w: int,
    sprite_h: int,
    symmetry_kind: str,
    inner_margin_px: int,
    min_gap_px: int,
) -> Tuple[int, int] | None:
    """Sample a seed placement whose reflected counterpart stays inside the patch."""

    margin = int(inner_margin_px)
    gap = int(min_gap_px)
    if str(symmetry_kind) == "vertical":
        x_min = int(margin)
        x_max = int((int(width) - (2 * int(sprite_w)) - int(gap)) // 2)
        y_min = int(margin)
        y_max = int(height - margin - sprite_h)
    elif str(symmetry_kind) == "horizontal":
        x_min = int(margin)
        x_max = int(width - margin - sprite_w)
        y_min = int(margin)
        y_max = int((int(height) - (2 * int(sprite_h)) - int(gap)) // 2)
    elif str(symmetry_kind) in {"diagonal_main", "diagonal_anti"}:
        for _ in range(32):
            position = _random_position_within_patch(
                rng,
                width=int(width),
                height=int(height),
                sprite_w=int(sprite_w),
                sprite_h=int(sprite_h),
                inner_margin_px=int(inner_margin_px),
            )
            if position is None:
                return None
            x, y = int(position[0]), int(position[1])
            if str(symmetry_kind) == "diagonal_main":
                if int(x + sprite_w + gap) <= int(y) or int(y + sprite_h + gap) <= int(x):
                    return int(x), int(y)
            else:
                anti_diagonal = int(width - sprite_w)
                if int(x + y) <= int(anti_diagonal - sprite_w - gap) or int(x + y) >= int(anti_diagonal + gap):
                    return int(x), int(y)
        return None
    else:
        raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")
    if x_min > x_max or y_min > y_max:
        return None
    return int(rng.randint(int(x_min), int(x_max))), int(rng.randint(int(y_min), int(y_max)))


def _mirrored_rect(
    rect_xywh: Tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    symmetry_kind: str,
) -> Tuple[int, int, int, int]:
    """Return one reflected `xywh` rectangle within a patch."""

    x, y, sprite_w, sprite_h = [int(value) for value in rect_xywh]
    if str(symmetry_kind) == "vertical":
        return (int(width - x - sprite_w), int(y), int(sprite_w), int(sprite_h))
    if str(symmetry_kind) == "horizontal":
        return (int(x), int(height - y - sprite_h), int(sprite_w), int(sprite_h))
    if str(symmetry_kind) == "diagonal_main":
        return (int(y), int(x), int(sprite_h), int(sprite_w))
    if str(symmetry_kind) == "diagonal_anti":
        return (
            int(width - y - sprite_h),
            int(height - x - sprite_w),
            int(sprite_h),
            int(sprite_w),
        )
    raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")


def _flip_both_axes_rect(
    rect_xywh: Tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Return one rectangle reflected across vertical and horizontal axes."""

    x, y, sprite_w, sprite_h = [int(value) for value in rect_xywh]
    return (
        int(width - x - sprite_w),
        int(height - y - sprite_h),
        int(sprite_w),
        int(sprite_h),
    )


def _sample_nonoverlapping_symmetric_seed_position(
    rng,
    *,
    width: int,
    height: int,
    sprite_w: int,
    sprite_h: int,
    symmetry_kind: str,
    inner_margin_px: int,
    min_gap_px: int,
    rects_xywh: Sequence[Tuple[int, int, int, int]],
    attempts: int,
) -> Tuple[int, int] | None:
    """Sample a mirrored-pair seed position that avoids all existing sprites."""

    for _ in range(max(1, int(attempts))):
        position = _sample_symmetric_seed_position(
            rng,
            width=int(width),
            height=int(height),
            sprite_w=int(sprite_w),
            sprite_h=int(sprite_h),
            symmetry_kind=str(symmetry_kind),
            inner_margin_px=int(inner_margin_px),
            min_gap_px=int(min_gap_px),
        )
        if position is None:
            continue
        rect_xywh = (int(position[0]), int(position[1]), int(sprite_w), int(sprite_h))
        mirrored_rect_xywh = _mirrored_rect(
            rect_xywh,
            width=int(width),
            height=int(height),
            symmetry_kind=str(symmetry_kind),
        )
        if _rects_intersect(rect_xywh, mirrored_rect_xywh, pad=int(min_gap_px)):
            continue
        if any(_rects_intersect(rect_xywh, other, pad=int(min_gap_px)) for other in rects_xywh):
            continue
        if any(_rects_intersect(mirrored_rect_xywh, other, pad=int(min_gap_px)) for other in rects_xywh):
            continue
        return int(position[0]), int(position[1])
    return None


def _sample_nonoverlapping_both_axes_seed_position(
    rng,
    *,
    width: int,
    height: int,
    sprite_w: int,
    sprite_h: int,
    inner_margin_px: int,
    min_gap_px: int,
    rects_xywh: Sequence[Tuple[int, int, int, int]],
    attempts: int,
) -> Tuple[int, int] | None:
    """Sample a base position whose four reflected copies are collision-free."""

    margin = int(inner_margin_px)
    gap = int(min_gap_px)
    x_min = int(margin)
    x_max = int((int(width) - (2 * int(sprite_w)) - int(gap)) // 2)
    y_min = int(margin)
    y_max = int((int(height) - (2 * int(sprite_h)) - int(gap)) // 2)
    if x_min > x_max or y_min > y_max:
        return None

    for _ in range(max(1, int(attempts))):
        x = int(rng.randint(int(x_min), int(x_max)))
        y = int(rng.randint(int(y_min), int(y_max)))
        base_rect = (int(x), int(y), int(sprite_w), int(sprite_h))
        orbit_rects = [
            base_rect,
            _mirrored_rect(base_rect, width=int(width), height=int(height), symmetry_kind="vertical"),
            _mirrored_rect(base_rect, width=int(width), height=int(height), symmetry_kind="horizontal"),
            _flip_both_axes_rect(base_rect, width=int(width), height=int(height)),
        ]
        if len({tuple(int(value) for value in rect) for rect in orbit_rects}) != 4:
            continue
        has_collision = any(
            _rects_intersect(rect, other, pad=int(gap))
            for rect in orbit_rects
            for other in rects_xywh
        )
        if has_collision:
            continue
        has_internal_collision = False
        for left_index, left_rect in enumerate(orbit_rects):
            for right_rect in orbit_rects[int(left_index) + 1 :]:
                if _rects_intersect(left_rect, right_rect, pad=int(gap)):
                    has_internal_collision = True
                    break
            if has_internal_collision:
                break
        if has_internal_collision:
            continue
        return int(x), int(y)
    return None


def _sprite_record(
    *,
    icon_id: str,
    tint_rgb: Tuple[int, int, int],
    rotation_degrees: int,
    noise_edits: Sequence[Mapping[str, Any]],
    noise_seed: int | None,
    bbox_xyxy: Sequence[int],
    relation_to_pair: str,
    mirrored_from_index: int | None,
    reflection_applied: str,
) -> Dict[str, Any]:
    """Build one trace-friendly icon-placement record."""

    return {
        "icon_id": str(icon_id),
        "tint_rgb": [int(value) for value in tint_rgb],
        "rotation_degrees": int(rotation_degrees) % 360,
        "noise_edits": [dict(edit) for edit in noise_edits],
        "noise_seed": None if noise_seed is None else int(noise_seed),
        "bbox_xyxy": [int(value) for value in bbox_xyxy],
        "relation_to_pair": str(relation_to_pair),
        "mirrored_from_index": None if mirrored_from_index is None else int(mirrored_from_index),
        "reflection_applied": str(reflection_applied),
    }


def _render_symmetric_patch(
    rng,
    *,
    instance_seed: int,
    noise_namespace: str,
    width: int,
    height: int,
    symmetry_kind: str,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    rotation_candidates: Sequence[int],
    render_params: Mapping[str, Any],
) -> Tuple[Image.Image, List[Dict[str, Any]], int]:
    """Render one exact single-axis or diagonal mirror-symmetric patch."""

    choices = [int(value) for value in render_params["symmetric_icon_count_choices"]]
    if not choices:
        raise ValueError("symmetric_icon_count_choices resolved no values")
    final_count = int(rng.choice(choices))
    seed_count = max(1, int(final_count // 2))
    scale_min, scale_max = _scale_bounds_for_count(int(final_count))
    patch_width = int(width)
    patch_height = int(height)
    inner_margin = int(render_params["patch_inner_margin_px"])
    min_gap = int(render_params["patch_min_gap_px"])
    max_attempts = max(1, int(render_params["patch_sampling_attempts"]))

    for _ in range(max_attempts):
        placements: List[Tuple[Tuple[int, int, int, int], Image.Image, Dict[str, Any], Image.Image]] = []
        rects_xywh: List[Tuple[int, int, int, int]] = []
        failed = False
        for seed_index in range(seed_count):
            icon_id = str(rng.choice(pool))
            tint_rgb = tuple(int(value) for value in rng.choice(palette))
            rotation_degrees = int(rng.choice(rotation_candidates))
            target_size = max(
                18,
                int(
                    round(
                        min(float(patch_width), float(patch_height))
                        * float(rng.uniform(float(scale_min), float(scale_max)))
                    )
                ),
            )
            noise_edits, noise_seed = sample_icon_instance_noise(
                instance_seed=int(instance_seed),
                namespace=f"{noise_namespace}:seed_{int(seed_index)}",
                render_params=render_params,
            )
            sprite = render_icon_rgba(
                icon_id=str(icon_id),
                size_px=int(target_size),
                tint_rgb=tuple(int(value) for value in tint_rgb),
                rotation_degrees=int(rotation_degrees),
                mirror_x=False,
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            sprite_w, sprite_h = sprite.size
            position = _sample_nonoverlapping_symmetric_seed_position(
                rng,
                width=int(patch_width),
                height=int(patch_height),
                sprite_w=int(sprite_w),
                sprite_h=int(sprite_h),
                symmetry_kind=str(symmetry_kind),
                inner_margin_px=int(inner_margin),
                min_gap_px=int(min_gap),
                rects_xywh=tuple(rects_xywh),
                attempts=max(24, int(max_attempts // 2)),
            )
            if position is None:
                failed = True
                break
            rect_xywh = (int(position[0]), int(position[1]), int(sprite_w), int(sprite_h))
            mirrored_rect_xywh = _mirrored_rect(
                rect_xywh,
                width=int(patch_width),
                height=int(patch_height),
                symmetry_kind=str(symmetry_kind),
            )
            rects_xywh.extend((tuple(int(value) for value in rect_xywh), tuple(int(value) for value in mirrored_rect_xywh)))
            placements.append(
                (
                    tuple(int(value) for value in rect_xywh),
                    sprite,
                    {
                        "icon_id": str(icon_id),
                        "tint_rgb": tuple(int(value) for value in tint_rgb),
                        "rotation_degrees": int(rotation_degrees),
                        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(noise_edits))],
                        "noise_seed": int(noise_seed),
                    },
                    _flip_image(sprite, str(symmetry_kind)),
                )
            )
        if failed or len(placements) != int(seed_count):
            continue

        patch = Image.new("RGBA", (int(patch_width), int(patch_height)), (255, 255, 255, 0))
        placement_records: List[Dict[str, Any]] = []
        for pair_index, (rect_xywh, sprite, sprite_meta, mirrored_sprite) in enumerate(placements):
            x, y, sprite_w, sprite_h = [int(value) for value in rect_xywh]
            patch.alpha_composite(sprite, (int(x), int(y)))
            mirrored_rect_xywh = _mirrored_rect(
                rect_xywh,
                width=int(patch_width),
                height=int(patch_height),
                symmetry_kind=str(symmetry_kind),
            )
            mx, my, mw, mh = [int(value) for value in mirrored_rect_xywh]
            patch.alpha_composite(mirrored_sprite, (int(mx), int(my)))
            placement_records.append(
                _sprite_record(
                    icon_id=str(sprite_meta["icon_id"]),
                    tint_rgb=tuple(int(value) for value in sprite_meta["tint_rgb"]),
                    rotation_degrees=int(sprite_meta["rotation_degrees"]),
                    noise_edits=tuple(sprite_meta["noise_edits"]),
                    noise_seed=int(sprite_meta["noise_seed"]),
                    bbox_xyxy=(int(x), int(y), int(x + sprite_w), int(y + sprite_h)),
                    relation_to_pair="base",
                    mirrored_from_index=None,
                    reflection_applied="none",
                )
            )
            placement_records.append(
                _sprite_record(
                    icon_id=str(sprite_meta["icon_id"]),
                    tint_rgb=tuple(int(value) for value in sprite_meta["tint_rgb"]),
                    rotation_degrees=int(sprite_meta["rotation_degrees"]),
                    noise_edits=tuple(sprite_meta["noise_edits"]),
                    noise_seed=int(sprite_meta["noise_seed"]),
                    bbox_xyxy=(int(mx), int(my), int(mx + mw), int(my + mh)),
                    relation_to_pair="mirrored",
                    mirrored_from_index=int(pair_index),
                    reflection_applied=str(_reflection_name(str(symmetry_kind))),
                )
            )
        if _is_exact_symmetry_kind(patch, str(symmetry_kind)):
            return patch, placement_records, int(final_count)
    raise ValueError("failed to sample an exact mirror-symmetric patch")


def _render_both_axes_patch(
    rng,
    *,
    instance_seed: int,
    noise_namespace: str,
    width: int,
    height: int,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    rotation_candidates: Sequence[int],
    render_params: Mapping[str, Any],
) -> Tuple[Image.Image, List[Dict[str, Any]], int]:
    """Render one exact patch symmetric across vertical and horizontal axes."""

    if int(width) != int(height):
        raise ValueError("both-axis symmetry requires a square patch")
    choices = [int(value) for value in render_params["both_axes_icon_count_choices"]]
    if not choices:
        raise ValueError("both_axes_icon_count_choices resolved no values")
    final_count = int(rng.choice(choices))
    if int(final_count) % 4 != 0:
        raise ValueError("both_axes icon counts must be multiples of four")
    orbit_count = max(1, int(final_count // 4))
    scale_min, scale_max = _scale_bounds_for_count(int(final_count))
    patch_width = int(width)
    patch_height = int(height)
    inner_margin = int(render_params["patch_inner_margin_px"])
    min_gap = int(render_params["patch_min_gap_px"])
    max_attempts = max(1, int(render_params["patch_sampling_attempts"]))

    for _ in range(max_attempts):
        patch = Image.new("RGBA", (int(patch_width), int(patch_height)), (255, 255, 255, 0))
        rects_xywh: List[Tuple[int, int, int, int]] = []
        placement_records: List[Dict[str, Any]] = []
        failed = False
        for orbit_index in range(int(orbit_count)):
            icon_id = str(rng.choice(pool))
            tint_rgb = tuple(int(value) for value in rng.choice(palette))
            rotation_degrees = int(rng.choice(rotation_candidates))
            target_size = max(
                18,
                int(
                    round(
                        min(float(patch_width), float(patch_height))
                        * float(rng.uniform(float(scale_min), float(scale_max)))
                    )
                ),
            )
            noise_edits, noise_seed = sample_icon_instance_noise(
                instance_seed=int(instance_seed),
                namespace=f"{noise_namespace}:orbit_{int(orbit_index)}",
                render_params=render_params,
            )
            sprite = render_icon_rgba(
                icon_id=str(icon_id),
                size_px=int(target_size),
                tint_rgb=tint_rgb,
                rotation_degrees=int(rotation_degrees),
                mirror_x=False,
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            sprite_w, sprite_h = sprite.size
            position = _sample_nonoverlapping_both_axes_seed_position(
                rng,
                width=int(patch_width),
                height=int(patch_height),
                sprite_w=int(sprite_w),
                sprite_h=int(sprite_h),
                inner_margin_px=int(inner_margin),
                min_gap_px=int(min_gap),
                rects_xywh=tuple(rects_xywh),
                attempts=max(24, int(max_attempts // 2)),
            )
            if position is None:
                failed = True
                break
            base_rect = (int(position[0]), int(position[1]), int(sprite_w), int(sprite_h))
            orbit_rects = [
                base_rect,
                _mirrored_rect(base_rect, width=int(patch_width), height=int(patch_height), symmetry_kind="vertical"),
                _mirrored_rect(base_rect, width=int(patch_width), height=int(patch_height), symmetry_kind="horizontal"),
                _flip_both_axes_rect(base_rect, width=int(patch_width), height=int(patch_height)),
            ]
            sprite_variants = [
                (sprite, "base", None, "none"),
                (_flip_image(sprite, "vertical"), "mirrored", int(len(placement_records)), "horizontal_flip"),
                (_flip_image(sprite, "horizontal"), "mirrored", int(len(placement_records)), "vertical_flip"),
                (_flip_image(sprite, "both_axes"), "mirrored", int(len(placement_records)), "both_axes_flip"),
            ]
            rects_xywh.extend(tuple(int(value) for value in rect) for rect in orbit_rects)
            for rect_xywh, (variant_sprite, relation_to_pair, mirrored_from_index, reflection_applied) in zip(
                orbit_rects,
                sprite_variants,
            ):
                x, y, rect_w, rect_h = [int(value) for value in rect_xywh]
                patch.alpha_composite(variant_sprite, (int(x), int(y)))
                placement_records.append(
                    _sprite_record(
                        icon_id=str(icon_id),
                        tint_rgb=tuple(int(value) for value in tint_rgb),
                        rotation_degrees=int(rotation_degrees),
                        noise_edits=tuple(dict(edit) for edit in serialize_icon_noise_edits(tuple(noise_edits))),
                        noise_seed=int(noise_seed),
                        bbox_xyxy=(int(x), int(y), int(x + rect_w), int(y + rect_h)),
                        relation_to_pair=str(relation_to_pair),
                        mirrored_from_index=int(mirrored_from_index) if mirrored_from_index is not None else None,
                        reflection_applied=str(reflection_applied),
                    )
                )
        if failed or len(placement_records) != int(final_count):
            continue
        if _is_exact_symmetry_kind(patch, "both_axes"):
            return patch, placement_records, int(final_count)
    raise ValueError("failed to sample an exact both-axes-symmetric patch")


def _render_nonsymmetric_patch(
    rng,
    *,
    instance_seed: int,
    noise_namespace: str,
    width: int,
    height: int,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    rotation_candidates: Sequence[int],
    render_params: Mapping[str, Any],
) -> Tuple[Image.Image, List[Dict[str, Any]], int]:
    """Render one patch with none of the supported mirror symmetries."""

    choices = [int(value) for value in render_params["nonsymmetric_icon_count_choices"]]
    if not choices:
        raise ValueError("nonsymmetric_icon_count_choices resolved no values")
    icon_count = int(rng.choice(choices))
    scale_min, scale_max = _scale_bounds_for_count(int(icon_count))
    patch_width = int(width)
    patch_height = int(height)
    inner_margin = int(render_params["patch_inner_margin_px"])
    min_gap = int(render_params["patch_min_gap_px"])
    max_attempts = max(1, int(render_params["patch_sampling_attempts"]))

    for _ in range(max_attempts):
        patch = Image.new("RGBA", (int(patch_width), int(patch_height)), (255, 255, 255, 0))
        rects_xywh: List[Tuple[int, int, int, int]] = []
        placement_records: List[Dict[str, Any]] = []
        failed = False
        for icon_index in range(int(icon_count)):
            icon_id = str(rng.choice(pool))
            tint_rgb = tuple(int(value) for value in rng.choice(palette))
            rotation_degrees = int(rng.choice(rotation_candidates))
            target_size = max(
                18,
                int(
                    round(
                        min(float(patch_width), float(patch_height))
                        * float(rng.uniform(float(scale_min), float(scale_max)))
                    )
                ),
            )
            noise_edits, noise_seed = sample_icon_instance_noise(
                instance_seed=int(instance_seed),
                namespace=f"{noise_namespace}:icon_{int(icon_index)}",
                render_params=render_params,
            )
            sprite = render_icon_rgba(
                icon_id=str(icon_id),
                size_px=int(target_size),
                tint_rgb=tint_rgb,
                rotation_degrees=int(rotation_degrees),
                mirror_x=False,
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            position = _sample_nonoverlapping_position(
                rng,
                width=int(patch_width),
                height=int(patch_height),
                sprite_w=int(sprite.size[0]),
                sprite_h=int(sprite.size[1]),
                inner_margin_px=int(inner_margin),
                min_gap_px=int(min_gap),
                rects_xywh=tuple(rects_xywh),
                attempts=max(24, int(max_attempts // 2)),
            )
            if position is None:
                failed = True
                break
            rect_xywh = (int(position[0]), int(position[1]), int(sprite.size[0]), int(sprite.size[1]))
            rects_xywh.append(tuple(int(value) for value in rect_xywh))
            x, y, sprite_w, sprite_h = rect_xywh
            patch.alpha_composite(sprite, (int(x), int(y)))
            placement_records.append(
                _sprite_record(
                    icon_id=str(icon_id),
                    tint_rgb=tuple(int(value) for value in tint_rgb),
                    rotation_degrees=int(rotation_degrees),
                    noise_edits=tuple(dict(edit) for edit in serialize_icon_noise_edits(tuple(noise_edits))),
                    noise_seed=int(noise_seed),
                    bbox_xyxy=(int(x), int(y), int(x + sprite_w), int(y + sprite_h)),
                    relation_to_pair="single",
                    mirrored_from_index=None,
                    reflection_applied="none",
                )
            )
        if failed or len(placement_records) != int(icon_count):
            continue
        if tuple(bool(value) for value in _symmetry_signature(patch)) == (False, False, False, False):
            return patch, placement_records, int(icon_count)
    raise ValueError("failed to sample a non-symmetric patch")


def _offset_patch_records(
    placements: Sequence[Mapping[str, Any]],
    *,
    offset_x: int,
    offset_y: int,
) -> List[Dict[str, Any]]:
    """Offset patch-local placement boxes into final image coordinates."""

    projected: List[Dict[str, Any]] = []
    for record in placements:
        bbox = record["bbox_xyxy"]
        projected.append(
            {
                **dict(record),
                "bbox_xyxy": [
                    int(bbox[0]) + int(offset_x),
                    int(bbox[1]) + int(offset_y),
                    int(bbox[2]) + int(offset_x),
                    int(bbox[3]) + int(offset_y),
                ],
            }
        )
    return projected


def _render_patch_for_kind(
    rng,
    *,
    instance_seed: int,
    noise_namespace: str,
    symmetry_kind: str,
    width: int,
    height: int,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    rotation_candidates: Sequence[int],
    render_params: Mapping[str, Any],
) -> Tuple[Image.Image, List[Dict[str, Any]], int]:
    """Render one patch for the requested neutral symmetry kind."""

    if str(symmetry_kind) == "both_axes":
        return _render_both_axes_patch(
            rng,
            instance_seed=int(instance_seed),
            noise_namespace=str(noise_namespace),
            width=int(width),
            height=int(height),
            pool=pool,
            palette=palette,
            rotation_candidates=rotation_candidates,
            render_params=render_params,
        )
    if str(symmetry_kind) in set(SYMMETRY_KINDS):
        return _render_symmetric_patch(
            rng,
            instance_seed=int(instance_seed),
            noise_namespace=str(noise_namespace),
            width=int(width),
            height=int(height),
            symmetry_kind=str(symmetry_kind),
            pool=pool,
            palette=palette,
            rotation_candidates=rotation_candidates,
            render_params=render_params,
        )
    if str(symmetry_kind) == NONSYMMETRIC_KIND:
        return _render_nonsymmetric_patch(
            rng,
            instance_seed=int(instance_seed),
            noise_namespace=str(noise_namespace),
            width=int(width),
            height=int(height),
            pool=pool,
            palette=palette,
            rotation_candidates=rotation_candidates,
            render_params=render_params,
        )
    raise ValueError(f"unsupported symmetry kind: {symmetry_kind}")


def _completion_mirror_cell(row: int, col: int, *, mirror_axis: str) -> Tuple[int, int]:
    """Return the paired 4x4 grid cell for one vertical or horizontal mirror axis."""

    if str(mirror_axis) == "vertical":
        return int(row), int(3 - int(col))
    if str(mirror_axis) == "horizontal":
        return int(3 - int(row)), int(col)
    raise ValueError(f"unsupported completion mirror axis: {mirror_axis}")


def _reflect_completion_sprite(sprite: Image.Image, *, mirror_axis: str) -> Image.Image:
    """Return the reflected icon image needed for the paired completion cell."""

    if str(mirror_axis) == "vertical":
        return ImageOps.mirror(sprite)
    if str(mirror_axis) == "horizontal":
        return ImageOps.flip(sprite)
    raise ValueError(f"unsupported completion mirror axis: {mirror_axis}")


def _sprite_digest(sprite: Image.Image) -> str:
    """Return a stable digest for one rendered option sprite."""

    rgba = sprite.convert("RGBA")
    payload = f"{rgba.size[0]}x{rgba.size[1]}:".encode("ascii") + rgba.tobytes()
    return hashlib.md5(payload).hexdigest()


def _center_paste_rgba(image: Image.Image, sprite: Image.Image, bbox_xyxy: Sequence[int]) -> Tuple[int, int, int, int]:
    """Paste an RGBA sprite centered in a target box and return its image bbox."""

    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
    paste_x = int(round(float(x0 + x1 - int(sprite.size[0])) / 2.0))
    paste_y = int(round(float(y0 + y1 - int(sprite.size[1])) / 2.0))
    image.alpha_composite(sprite, (int(paste_x), int(paste_y)))
    return (
        int(paste_x),
        int(paste_y),
        int(paste_x + int(sprite.size[0])),
        int(paste_y + int(sprite.size[1])),
    )


def _draw_completion_missing_mark(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_xyxy: Sequence[int],
    render_params: Mapping[str, Any],
) -> None:
    """Draw a centered question mark inside the missing grid cell."""

    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
    side = max(1, min(int(x1 - x0), int(y1 - y0)))
    font_size = max(24, int(round(float(side) * 0.56)))
    font = load_font(int(font_size), bold=True)
    text = "?"
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    text_w = int(text_bbox[2] - text_bbox[0])
    text_h = int(text_bbox[3] - text_bbox[1])
    draw_text_traced(
        draw,
        (
            int(round(float(x0 + x1 - text_w) / 2.0)),
            int(round(float(y0 + y1 - text_h) / 2.0)) - 2,
        ),
        text,
        font=font,
        fill=tuple(int(v) for v in render_params["header_text_rgb"]),
        stroke_fill=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        stroke_width=1,
        role="missing_mirror_grid_cell_text",
        required=False,
    )


def _sample_completion_sprite(
    rng,
    *,
    instance_seed: int,
    noise_namespace: str,
    icon_size_px: int,
    pool: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    rotation_candidates: Sequence[int],
    render_params: Mapping[str, Any],
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Sample and render one icon sprite for a completion grid cell or option."""

    icon_id = str(rng.choice(pool))
    tint_rgb = tuple(int(value) for value in rng.choice(palette))
    rotation_degrees = int(rng.choice(rotation_candidates))
    noise_edits, noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=str(noise_namespace),
        render_params=render_params,
    )
    sprite = render_icon_rgba(
        icon_id=str(icon_id),
        size_px=int(icon_size_px),
        tint_rgb=tuple(int(value) for value in tint_rgb),
        rotation_degrees=int(rotation_degrees),
        mirror_x=False,
        noise_edits=tuple(noise_edits),
        noise_seed=int(noise_seed),
    )
    return sprite, {
        "icon_id": str(icon_id),
        "tint_rgb": [int(value) for value in tint_rgb],
        "rotation_degrees": int(rotation_degrees) % 360,
        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(noise_edits))],
        "noise_seed": int(noise_seed),
    }


def sample_and_render_missing_mirror_cell_scene(
    rng,
    *,
    instance_seed: int,
    mirror_axis: str,
    option_count: int,
    answer_index: int,
    render_params: Mapping[str, Any],
    pool_manifest: str,
    noise_namespace: str,
) -> Tuple[MirrorGridCompletionScenePayload, Image.Image]:
    """Sample and render a 4x4 mirror grid with one missing icon cell."""

    axis = str(mirror_axis)
    if axis not in {"vertical", "horizontal"}:
        raise ValueError(f"unsupported mirror_axis: {axis}")
    if int(option_count) not in {4, 6}:
        raise ValueError("missing mirror-cell options must use 4 or 6 choices")
    if int(answer_index) < 0 or int(answer_index) >= int(option_count):
        raise ValueError("answer_index outside visible option range")

    pool = tuple(str(icon_id) for icon_id in resolve_icon_pool(str(pool_manifest)))
    if not pool:
        raise ValueError("mirror-grid completion scene resolved an empty icon pool")
    sampled_palette_rgb = sample_mirror_grid_palette(rng, render_params)
    option_labels = fixed_grid_labels(int(option_count))

    prepared = prepare_two_panel_labeled_grid_scene(
        scene_labels=option_labels,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        panel_title_font_size_px=int(render_params["panel_title_font_size_px"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        cell_padding_px=int(render_params["cell_padding_px"]),
        cell_border_rgb=tuple(int(v) for v in render_params["cell_border_rgb"]),
        cell_label_color_rgb=tuple(int(v) for v in render_params["cell_label_color_rgb"]),
        cell_label_stroke_rgb=tuple(int(v) for v in render_params["cell_label_stroke_rgb"]),
        cell_label_stroke_width_px=1,
        cell_label_font_size_px=int(render_params["cell_label_font_size_px"]),
        reference_square_cell=True,
        scene_square_cells=True,
        reference_title="Grid",
        scene_title="Options",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    image = prepared.image
    draw = ImageDraw.Draw(image)
    grid_bbox = centered_square_bbox(tuple(int(value) for value in prepared.reference_cell.content_bbox_xyxy))
    cell_slots = resolve_fixed_grid_cell_slots(
        grid_bbox,
        rows=4,
        cols=4,
        cell_padding_px=max(2, int(render_params["cell_padding_px"]) // 2),
    )
    if len(cell_slots) != 16:
        raise RuntimeError("mirror-grid completion expected 16 grid cells")
    icon_size_px = max(
        24,
        int(
            round(
                min(
                    min(int(slot[2] - slot[0]), int(slot[3] - slot[1])) for slot in cell_slots
                )
                * 0.66
            )
        ),
    )

    eligible_missing = [
        int(index)
        for index in range(16)
        if _completion_mirror_cell(int(index // 4), int(index % 4), mirror_axis=axis)
        != (int(index // 4), int(index % 4))
    ]
    missing_index = int(rng.choice(eligible_missing))
    missing_row, missing_col = int(missing_index // 4), int(missing_index % 4)
    counterpart_row, counterpart_col = _completion_mirror_cell(
        int(missing_row),
        int(missing_col),
        mirror_axis=axis,
    )
    counterpart_index = int(counterpart_row) * 4 + int(counterpart_col)

    answer_sprite, answer_meta = _sample_completion_sprite(
        rng,
        instance_seed=int(instance_seed),
        noise_namespace=f"{noise_namespace}:answer",
        icon_size_px=int(icon_size_px),
        pool=pool,
        palette=sampled_palette_rgb,
        rotation_candidates=render_params["rotation_candidates_degrees"],
        render_params=render_params,
    )
    counterpart_sprite = _reflect_completion_sprite(answer_sprite, mirror_axis=axis)
    pair_sprites: Dict[Tuple[int, int], Tuple[Image.Image, Image.Image, Dict[str, Any]]] = {}
    grid_cells: List[Dict[str, Any]] = []

    for index, slot in enumerate(cell_slots):
        row, col = int(index // 4), int(index % 4)
        slot_bbox = tuple(int(value) for value in slot)
        draw.rounded_rectangle(
            slot_bbox,
            radius=8,
            outline=tuple(int(v) for v in render_params["cell_border_rgb"]),
            width=2,
            fill=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        )
        content_bbox = (
            int(slot_bbox[0] + 4),
            int(slot_bbox[1] + 4),
            int(slot_bbox[2] - 4),
            int(slot_bbox[3] - 4),
        )
        cell_payload: Dict[str, Any] = {
            "panel": "grid",
            "row": int(row),
            "col": int(col),
            "cell_index": int(index),
            "cell_bbox_xyxy": list(slot_bbox),
            "content_bbox_xyxy": list(content_bbox),
            "is_missing": bool(index == missing_index),
            "is_counterpart": bool(index == counterpart_index),
        }
        if int(index) == int(missing_index):
            _draw_completion_missing_mark(draw, bbox_xyxy=slot_bbox, render_params=render_params)
            cell_payload["icon_bbox_xyxy"] = None
            cell_payload["placements"] = []
            grid_cells.append(cell_payload)
            continue

        if int(index) == int(counterpart_index):
            sprite = counterpart_sprite
            meta = {
                **dict(answer_meta),
                "relation_to_missing": "mirror_counterpart",
                "reflection_applied": "horizontal_flip" if axis == "vertical" else "vertical_flip",
            }
        else:
            mirror_row, mirror_col = _completion_mirror_cell(row, col, mirror_axis=axis)
            mirror_index = int(mirror_row) * 4 + int(mirror_col)
            pair_key = tuple(sorted((int(index), int(mirror_index))))
            if pair_key not in pair_sprites:
                base_sprite, base_meta = _sample_completion_sprite(
                    rng,
                    instance_seed=int(instance_seed),
                    noise_namespace=f"{noise_namespace}:pair_{pair_key[0]}_{pair_key[1]}",
                    icon_size_px=int(icon_size_px),
                    pool=pool,
                    palette=sampled_palette_rgb,
                    rotation_candidates=render_params["rotation_candidates_degrees"],
                    render_params=render_params,
                )
                pair_sprites[pair_key] = (
                    base_sprite,
                    _reflect_completion_sprite(base_sprite, mirror_axis=axis),
                    dict(base_meta),
                )
            base_sprite, reflected_sprite, base_meta = pair_sprites[pair_key]
            sprite = base_sprite if int(index) == int(pair_key[0]) else reflected_sprite
            meta = {
                **dict(base_meta),
                "relation_to_missing": "context_pair",
                "reflection_applied": "none" if int(index) == int(pair_key[0]) else (
                    "horizontal_flip" if axis == "vertical" else "vertical_flip"
                ),
            }
        icon_bbox = _center_paste_rgba(image, sprite, content_bbox)
        cell_payload["icon_bbox_xyxy"] = list(icon_bbox)
        cell_payload["placements"] = [
            _sprite_record(
                icon_id=str(meta["icon_id"]),
                tint_rgb=tuple(int(v) for v in meta["tint_rgb"]),
                rotation_degrees=int(meta["rotation_degrees"]),
                noise_edits=tuple(dict(edit) for edit in meta["noise_edits"]),
                noise_seed=int(meta["noise_seed"]),
                bbox_xyxy=tuple(int(value) for value in icon_bbox),
                relation_to_pair=str(meta["relation_to_missing"]),
                mirrored_from_index=None,
                reflection_applied=str(meta["reflection_applied"]),
            )
        ]
        grid_cells.append(cell_payload)

    axis_color = tuple(int(value) for value in render_params["header_text_rgb"])
    if axis == "vertical":
        x = int(round(float(grid_bbox[0] + grid_bbox[2]) / 2.0))
        draw.line((x, int(grid_bbox[1]) + 2, x, int(grid_bbox[3]) - 2), fill=axis_color, width=4)
    else:
        y = int(round(float(grid_bbox[1] + grid_bbox[3]) / 2.0))
        draw.line((int(grid_bbox[0]) + 2, y, int(grid_bbox[2]) - 2, y), fill=axis_color, width=4)

    answer_label = str(option_labels[int(answer_index)])
    option_sprites: List[Tuple[Image.Image, Dict[str, Any], bool]] = []
    used_digests = {_sprite_digest(answer_sprite)}
    for option_index, _label in enumerate(option_labels):
        if int(option_index) == int(answer_index):
            option_sprites.append((answer_sprite, dict(answer_meta), True))
            continue
        for attempt in range(80):
            candidate_sprite, candidate_meta = _sample_completion_sprite(
                rng,
                instance_seed=int(instance_seed),
                noise_namespace=f"{noise_namespace}:option_{option_index}_{attempt}",
                icon_size_px=int(icon_size_px),
                pool=pool,
                palette=sampled_palette_rgb,
                rotation_candidates=render_params["rotation_candidates_degrees"],
                render_params=render_params,
            )
            digest = _sprite_digest(candidate_sprite)
            if digest in used_digests:
                continue
            used_digests.add(str(digest))
            option_sprites.append((candidate_sprite, dict(candidate_meta), False))
            break
        else:
            raise ValueError("failed to sample unique mirror-grid completion distractor option")

    option_cells: List[Dict[str, Any]] = []
    for prepared_cell, (sprite, meta, is_answer) in zip(prepared.scene_cells, option_sprites):
        icon_bbox = _center_paste_rgba(image, sprite, prepared_cell.content_bbox_xyxy)
        option_cells.append(
            {
                "panel": "options",
                "label": str(prepared_cell.label),
                "cell_bbox_xyxy": [int(value) for value in prepared_cell.cell_bbox_xyxy],
                "content_bbox_xyxy": [int(value) for value in prepared_cell.content_bbox_xyxy],
                "icon_bbox_xyxy": [int(value) for value in icon_bbox],
                "is_answer": bool(is_answer),
                "placements": [
                    _sprite_record(
                        icon_id=str(meta["icon_id"]),
                        tint_rgb=tuple(int(v) for v in meta["tint_rgb"]),
                        rotation_degrees=int(meta["rotation_degrees"]),
                        noise_edits=tuple(dict(edit) for edit in meta["noise_edits"]),
                        noise_seed=int(meta["noise_seed"]),
                        bbox_xyxy=tuple(int(value) for value in icon_bbox),
                        relation_to_pair="answer_option" if bool(is_answer) else "distractor_option",
                        mirrored_from_index=None,
                        reflection_applied="none",
                    )
                ],
            }
        )

    return (
        MirrorGridCompletionScenePayload(
            object_count=16,
            option_count=int(option_count),
            mirror_axis=str(axis),
            missing_row=int(missing_row),
            missing_col=int(missing_col),
            counterpart_row=int(counterpart_row),
            counterpart_col=int(counterpart_col),
            answer_label=str(answer_label),
            cell_labels=tuple(str(index) for index in range(16)),
            option_labels=tuple(str(value) for value in option_labels),
            sampled_palette_rgb=tuple(sampled_palette_rgb),
            panel_geometry=panel_geometry_to_trace(prepared.layout),
            grid_panel={
                "panel": "grid",
                "cell_bbox_xyxy": [int(value) for value in prepared.reference_cell.cell_bbox_xyxy],
                "content_bbox_xyxy": [int(value) for value in prepared.reference_cell.content_bbox_xyxy],
                "grid_bbox_xyxy": [int(value) for value in grid_bbox],
            },
            grid_cells=tuple(dict(item) for item in grid_cells),
            option_cells=tuple(dict(item) for item in option_cells),
        ),
        image.convert("RGB"),
    )


def sample_and_render_mirror_grid_scene(
    rng,
    *,
    instance_seed: int,
    reference_symmetry_kind: str,
    object_count: int,
    target_count: int,
    matching_indices: Sequence[int] | None = None,
    render_params: Mapping[str, Any],
    pool_manifest: str,
    noise_namespace: str,
) -> Tuple[MirrorGridScenePayload, Image.Image]:
    """Sample and render one two-panel mirror-grid scene."""

    reference_kind = str(reference_symmetry_kind)
    if reference_kind not in set(SYMMETRY_KINDS):
        raise ValueError(f"unsupported reference symmetry kind: {reference_kind}")
    pool = tuple(str(icon_id) for icon_id in resolve_icon_pool(str(pool_manifest)))
    if not pool:
        raise ValueError("mirror-grid scene resolved an empty icon pool")

    sampled_palette_rgb = sample_mirror_grid_palette(rng, render_params)
    labels = fixed_grid_labels(int(object_count))
    if matching_indices is None:
        resolved_match_indices = sample_matching_indices(
            rng,
            object_count=int(object_count),
            target_count=int(target_count),
        )
    else:
        resolved_match_indices = tuple(sorted(int(index) for index in matching_indices))
        if len(set(resolved_match_indices)) != len(resolved_match_indices):
            raise ValueError("matching_indices must be unique")
        if len(resolved_match_indices) != int(target_count):
            raise ValueError("matching_indices length must match target_count")
        if any(int(index) < 0 or int(index) >= int(object_count) for index in resolved_match_indices):
            raise ValueError("matching_indices contains an index outside the scene cell range")
    match_indices = set(resolved_match_indices)
    distractor_count = int(object_count) - int(target_count)
    distractor_kinds = sample_distractor_symmetry_kinds(
        rng,
        reference_symmetry_kind=str(reference_kind),
        distractor_count=int(distractor_count),
    )

    prepared = prepare_two_panel_labeled_grid_scene(
        scene_labels=labels,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        panel_title_font_size_px=int(render_params["panel_title_font_size_px"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        cell_padding_px=int(render_params["cell_padding_px"]),
        cell_border_rgb=tuple(int(v) for v in render_params["cell_border_rgb"]),
        cell_label_color_rgb=tuple(int(v) for v in render_params["cell_label_color_rgb"]),
        cell_label_stroke_rgb=tuple(int(v) for v in render_params["cell_label_stroke_rgb"]),
        cell_label_stroke_width_px=1,
        cell_label_font_size_px=int(render_params["cell_label_font_size_px"]),
        reference_square_cell=True,
        scene_square_cells=True,
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    image = prepared.image

    ref_content_bbox = tuple(int(value) for value in prepared.reference_cell.content_bbox_xyxy)
    reference_patch, reference_placements, reference_icon_count = _render_patch_for_kind(
        rng,
        instance_seed=int(instance_seed),
        noise_namespace=f"{noise_namespace}:reference",
        symmetry_kind=str(reference_kind),
        width=int(ref_content_bbox[2] - ref_content_bbox[0]),
        height=int(ref_content_bbox[3] - ref_content_bbox[1]),
        pool=pool,
        palette=sampled_palette_rgb,
        rotation_candidates=render_params["rotation_candidates_degrees"],
        render_params=render_params,
    )
    image.alpha_composite(reference_patch, (int(ref_content_bbox[0]), int(ref_content_bbox[1])))
    reference_signature = _symmetry_signature(reference_patch)
    reference_payload = {
        "panel": "reference",
        "symmetry_kind": str(reference_kind),
        "cell_bbox_xyxy": list(prepared.reference_cell.cell_bbox_xyxy),
        "content_bbox_xyxy": list(prepared.reference_cell.content_bbox_xyxy),
        "icon_count": int(reference_icon_count),
        "has_vertical_symmetry": bool(reference_signature[0]),
        "has_horizontal_symmetry": bool(reference_signature[1]),
        "has_diagonal_main_symmetry": bool(reference_signature[2]),
        "has_diagonal_anti_symmetry": bool(reference_signature[3]),
        "placements": _offset_patch_records(
            reference_placements,
            offset_x=int(ref_content_bbox[0]),
            offset_y=int(ref_content_bbox[1]),
        ),
    }

    matching_labels: List[str] = []
    scene_cell_symmetry_kinds: List[str] = []
    scene_cells: List[Dict[str, Any]] = []
    distractor_cursor = 0
    for index, prepared_cell in enumerate(prepared.scene_cells):
        if int(index) in match_indices:
            cell_kind = str(reference_kind)
            is_match = True
            matching_labels.append(str(prepared_cell.label))
        else:
            cell_kind = str(distractor_kinds[int(distractor_cursor)])
            distractor_cursor += 1
            is_match = False

        content_bbox = tuple(int(value) for value in prepared_cell.content_bbox_xyxy)
        patch, placements, icon_count = _render_patch_for_kind(
            rng,
            instance_seed=int(instance_seed),
            noise_namespace=f"{noise_namespace}:scene_{int(index)}",
            symmetry_kind=str(cell_kind),
            width=int(content_bbox[2] - content_bbox[0]),
            height=int(content_bbox[3] - content_bbox[1]),
            pool=pool,
            palette=sampled_palette_rgb,
            rotation_candidates=render_params["rotation_candidates_degrees"],
            render_params=render_params,
        )
        image.alpha_composite(patch, (int(content_bbox[0]), int(content_bbox[1])))
        cell_signature = _symmetry_signature(patch)
        scene_cells.append(
            {
                "panel": "scene",
                "label": str(prepared_cell.label),
                "cell_bbox_xyxy": list(prepared_cell.cell_bbox_xyxy),
                "content_bbox_xyxy": list(prepared_cell.content_bbox_xyxy),
                "symmetry_kind": str(cell_kind),
                "icon_count": int(icon_count),
                "has_vertical_symmetry": bool(cell_signature[0]),
                "has_horizontal_symmetry": bool(cell_signature[1]),
                "has_diagonal_main_symmetry": bool(cell_signature[2]),
                "has_diagonal_anti_symmetry": bool(cell_signature[3]),
                "is_match": bool(is_match),
                "placements": _offset_patch_records(
                    placements,
                    offset_x=int(content_bbox[0]),
                    offset_y=int(content_bbox[1]),
                ),
            }
        )
        scene_cell_symmetry_kinds.append(str(cell_kind))

    return (
        MirrorGridScenePayload(
            object_count=int(object_count),
            target_count=int(target_count),
            distractor_count=int(distractor_count),
            reference_symmetry_kind=str(reference_kind),
            cell_labels=tuple(str(value) for value in labels),
            matching_labels=tuple(sorted(str(value) for value in matching_labels)),
            scene_cell_symmetry_kinds=tuple(str(value) for value in scene_cell_symmetry_kinds),
            sampled_palette_rgb=tuple(sampled_palette_rgb),
            panel_geometry=panel_geometry_to_trace(prepared.layout),
            reference_cell=dict(reference_payload),
            scene_cells=tuple(dict(item) for item in scene_cells),
        ),
        image.convert("RGB"),
    )


__all__ = [
    "sample_and_render_mirror_grid_scene",
    "sample_and_render_missing_mirror_cell_scene",
]
