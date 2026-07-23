"""Single-panel procedural named-icon field scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.config_defaults import group_default
from .icon_noise import serialize_icon_noise_edits
from .icon_scene import (
    BBox,
    draw_single_panel,
    resolve_single_panel_layout,
    single_panel_geometry_to_trace,
    sort_bboxes_reading_order,
)
from .procedural_named_icons import (
    DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS,
    procedural_named_icon_display_name,
    procedural_named_icon_fill_style_probability_map,
    render_procedural_named_icon_rgba,
    validate_procedural_named_icon_fill_style_support,
)


SCENE_ID = "named_field"

DEFAULT_NAMED_ICON_ROTATION_JITTER_DEGREES = 15


@dataclass(frozen=True)
class NamedIconFieldSpec:
    """One icon instance to render in a named-icon field."""

    shape_id: str
    tint_rgb: Tuple[int, int, int]
    nominal_size_px: int
    color_name: str = ""
    fill_style: str = "solid"
    rotation_degrees: int = 0
    placement_group: str = ""
    noise_edits: Tuple[Any, ...] = ()
    noise_seed: int | None = None


@dataclass(frozen=True)
class RenderedNamedIconInstance:
    """Rendered procedural named icon metadata."""

    instance_id: str
    shape_id: str
    shape_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    color_name: str
    fill_style: str
    layout_row: int
    layout_col: int
    placement_group: str = ""
    noise_edits: Tuple[Dict[str, Any], ...] = ()
    noise_seed: int | None = None


@dataclass(frozen=True)
class RenderedNamedIconFieldScene:
    """Rendered named-icon field plus trace metadata."""

    image: Image.Image
    instances: Tuple[RenderedNamedIconInstance, ...]
    panel_geometry: Dict[str, Any]
    layout_mode: str


def resolve_named_icon_int_bounds(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, int]:
    """Resolve non-negative inclusive integer bounds for named-icon tasks."""

    low = int(params.get(low_key, group_default(gen_defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(gen_defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a uniform probability map over string support."""

    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): probability for value in support}


def resolve_named_icon_fill_style_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    fallback_support: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve renderable procedural named-icon fill styles."""

    key = "named_icon_fill_style_support"
    fallback = tuple(fallback_support)
    raw = params.get(key, group_default(gen_defaults, key, fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = fallback
    support = validate_procedural_named_icon_fill_style_support(
        tuple(str(value) for value in raw),
    )
    return support


def resolve_named_icon_fill_style_probabilities(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support: Sequence[str],
    *,
    default_weights: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Resolve procedural named-icon fill-style probabilities."""

    raw = params.get(
        "named_icon_fill_style_weights",
        group_default(gen_defaults, "named_icon_fill_style_weights", default_weights),
    )
    if not isinstance(raw, Mapping):
        raw = default_weights
    return procedural_named_icon_fill_style_probability_map(
        tuple(str(value) for value in support),
        dict(raw) if raw is not None else None,
    )


def rotation_for_named_shape(
    rng,
    shape_id: str,
    *,
    rotatable_shapes: Sequence[str] | None = None,
    jitter_degrees: int = DEFAULT_NAMED_ICON_ROTATION_JITTER_DEGREES,
) -> int:
    """Return a small naturalistic pose jitter for named-icon scenes.

    ``shape_id`` and ``rotatable_shapes`` are retained for call-site
    compatibility. Current named-icon counting/spatial scenes use the same
    non-semantic jitter for every named icon, while stack layouts opt out at the
    call site by passing/setting a zero rotation.
    """

    del shape_id, rotatable_shapes
    span = max(0, int(jitter_degrees))
    if span <= 0:
        return 0
    return int(rng.randint(-span, span))


def bbox_center_float(bbox: Sequence[int | float]) -> Tuple[float, float]:
    """Return the center of one xyxy bbox."""

    return (0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3])))


def bbox_from_center_and_size(center_xy: Sequence[float], sprite_size: Sequence[int]) -> Tuple[int, int, int, int]:
    """Return an xyxy bbox centered on `center_xy` with `sprite_size`."""

    cx, cy = float(center_xy[0]), float(center_xy[1])
    w, h = int(sprite_size[0]), int(sprite_size[1])
    x0 = int(round(cx - 0.5 * float(w)))
    y0 = int(round(cy - 0.5 * float(h)))
    return (int(x0), int(y0), int(x0 + w), int(y0 + h))


def bbox_from_center_dimensions(center: Tuple[float, float], *, width: int, height: int) -> BBox:
    """Return an xyxy bbox centered on `center` with explicit dimensions."""

    cx, cy = float(center[0]), float(center[1])
    w, h = int(width), int(height)
    x0 = int(round(cx - 0.5 * float(w)))
    y0 = int(round(cy - 0.5 * float(h)))
    return (int(x0), int(y0), int(x0 + w), int(y0 + h))


def bbox_inside(inner: BBox, outer: BBox) -> bool:
    """Return true when `inner` is fully inside `outer`."""

    return int(inner[0]) >= int(outer[0]) and int(inner[1]) >= int(outer[1]) and int(inner[2]) <= int(outer[2]) and int(inner[3]) <= int(outer[3])


def boxes_overlap(left: BBox, right: BBox, *, gap_px: int) -> bool:
    """Return true when two bboxes overlap after applying a minimum gap."""

    gap = max(0, int(gap_px))
    return (
        int(left[0]) < int(right[2]) + gap
        and int(left[2]) + gap > int(right[0])
        and int(left[1]) < int(right[3]) + gap
        and int(left[3]) + gap > int(right[1])
    )


def label_bbox_for_icon(
    *,
    icon_bbox: BBox,
    label: str,
    content_bbox: BBox,
    font,
    padding_px: int,
    gap_px: int,
) -> BBox:
    """Return a label bbox placed above or below one icon."""

    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=0)
    text_w = max(1, int(text_bbox[2] - text_bbox[0]))
    text_h = max(1, int(text_bbox[3] - text_bbox[1]))
    pad = max(1, int(padding_px))
    width = text_w + 2 * pad
    height = text_h + 2 * pad
    ix0, iy0, ix1, iy1 = tuple(int(value) for value in icon_bbox)
    cx = int(round(0.5 * float(ix0 + ix1)))
    x0 = int(max(int(content_bbox[0]), min(int(content_bbox[2]) - width, cx - width // 2)))
    above_y0 = int(iy0 - int(gap_px) - height)
    if above_y0 >= int(content_bbox[1]):
        y0 = above_y0
    else:
        y0 = int(min(int(content_bbox[3]) - height, int(iy1 + int(gap_px))))
    return (int(x0), int(y0), int(x0 + width), int(y0 + height))


def union_bbox(left: BBox, right: BBox) -> BBox:
    """Return the union of two xyxy bboxes."""

    return (
        min(int(left[0]), int(right[0])),
        min(int(left[1]), int(right[1])),
        max(int(left[2]), int(right[2])),
        max(int(left[3]), int(right[3])),
    )


def render_planned_named_icon_sprite(plan: Any) -> Image.Image:
    """Render a sprite from any plan object with the named-icon plan fields."""

    return render_procedural_named_icon_rgba(
        shape_id=str(plan.shape_id),
        size_px=int(plan.nominal_size_px),
        tint_rgb=tuple(int(value) for value in plan.tint_rgb),
        fill_style=str(plan.fill_style),
        rotation_degrees=int(plan.rotation_degrees),
        mirror_x=False,
        noise_edits=tuple(plan.noise_edits),
        noise_seed=plan.noise_seed,
    )


def _resolve_grid(*, count: int, content_bbox: BBox) -> Tuple[int, int]:
    x0, y0, x1, y1 = content_bbox
    content_w = max(1, int(x1) - int(x0))
    content_h = max(1, int(y1) - int(y0))
    aspect = float(content_w) / float(content_h)
    cols = max(1, int(round((float(count) * aspect) ** 0.5)))
    while int(cols) * max(1, (int(count) + int(cols) - 1) // int(cols)) < int(count):
        cols += 1
    rows = max(1, (int(count) + int(cols) - 1) // int(cols))
    return int(rows), int(cols)


def _grid_slots(content_bbox: BBox, *, rows: int, cols: int, inner_padding_px: int) -> List[Tuple[BBox, int, int]]:
    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    cell_w = float(max(1, x1 - x0)) / float(max(1, int(cols)))
    cell_h = float(max(1, y1 - y0)) / float(max(1, int(rows)))
    pad = max(0, int(inner_padding_px))
    slots: List[Tuple[BBox, int, int]] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            sx0 = int(round(float(x0) + float(col) * cell_w)) + pad
            sy0 = int(round(float(y0) + float(row) * cell_h)) + pad
            sx1 = int(round(float(x0) + float(col + 1) * cell_w)) - pad
            sy1 = int(round(float(y0) + float(row + 1) * cell_h)) - pad
            if sx1 > sx0 and sy1 > sy0:
                slots.append(((sx0, sy0, sx1, sy1), int(row), int(col)))
    return slots


def _centered_paste_xy(
    *,
    sprite_size: Tuple[int, int],
    slot_bbox: BBox,
    jitter_px: int,
    rng,
) -> Tuple[int, int]:
    sx0, sy0, sx1, sy1 = tuple(int(value) for value in slot_bbox)
    sprite_w, sprite_h = int(sprite_size[0]), int(sprite_size[1])
    if sprite_w > sx1 - sx0 or sprite_h > sy1 - sy0:
        raise ValueError("procedural named icon does not fit slot")
    slack_x = max(0, int(sx1 - sx0 - sprite_w))
    slack_y = max(0, int(sy1 - sy0 - sprite_h))
    jitter_x = min(max(0, int(jitter_px)), slack_x // 2)
    jitter_y = min(max(0, int(jitter_px)), slack_y // 2)
    x = int(sx0 + (slack_x // 2))
    y = int(sy0 + (slack_y // 2))
    if jitter_x:
        x += int(rng.randint(-jitter_x, jitter_x))
    if jitter_y:
        y += int(rng.randint(-jitter_y, jitter_y))
    return int(x), int(y)


def _draw_layout_guides(image: Image.Image, *, content_bbox: BBox, rows: int, layout_mode: str) -> None:
    if str(layout_mode) != "shelf_rows":
        return
    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    draw = ImageDraw.Draw(image)
    row_h = float(max(1, y1 - y0)) / float(max(1, int(rows)))
    color = (218, 224, 235, 255)
    for row in range(1, int(rows)):
        y = int(round(float(y0) + float(row) * row_h))
        draw.line((int(x0), y, int(x1), y), fill=color, width=2)


def _overlaps_existing(bbox: BBox, existing: Sequence[BBox], *, gap_px: int) -> bool:
    x0, y0, x1, y1 = tuple(int(value) for value in bbox)
    gap = max(0, int(gap_px))
    for other in existing:
        ox0, oy0, ox1, oy1 = tuple(int(value) for value in other)
        if x0 < ox1 + gap and x1 + gap > ox0 and y0 < oy1 + gap and y1 + gap > oy0:
            return True
    return False


def _group_key(spec: NamedIconFieldSpec) -> str:
    value = str(spec.placement_group or "").strip()
    return value or str(spec.shape_id)


def _group_indices(icon_specs: Sequence[NamedIconFieldSpec]) -> List[Tuple[str, List[int]]]:
    grouped: Dict[str, List[int]] = {}
    order: List[str] = []
    for index, spec in enumerate(icon_specs):
        key = _group_key(spec)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(int(index))
    return sorted(((key, grouped[key]) for key in order), key=lambda item: (-len(item[1]), item[0]))


def _sprite_for_spec(spec: NamedIconFieldSpec, *, nominal_size: int) -> Image.Image:
    return render_procedural_named_icon_rgba(
        shape_id=str(spec.shape_id),
        size_px=max(12, int(nominal_size)),
        tint_rgb=tuple(int(value) for value in spec.tint_rgb),
        fill_style=str(spec.fill_style or "solid"),
        rotation_degrees=int(spec.rotation_degrees),
        mirror_x=False,
        noise_edits=tuple(spec.noise_edits),
        noise_seed=spec.noise_seed,
    )


def _compact_stack_grid(*, count: int, bbox: BBox) -> Tuple[int, int]:
    """Resolve rows/cols for a compact icon stack inside one group bbox."""

    rows, cols = _resolve_grid(count=int(count), content_bbox=bbox)
    # Prefer block-like stacks over long rows so counts can be read as rows x columns.
    while int(cols) - int(rows) > 2 and int(rows) * max(1, int(cols) - 1) >= int(count):
        cols -= 1
        rows = max(1, (int(count) + int(cols) - 1) // int(cols))
    return int(rows), int(cols)


def render_procedural_named_icon_field_scene(
    *,
    rng,
    instance_seed: int,
    task_id: str,
    icon_specs: Sequence[NamedIconFieldSpec],
    render_params: Mapping[str, Any],
    layout_modes: Sequence[str],
    slot_padding_px: int,
    slot_jitter_px: int,
    stack_gap_px: int = 1,
) -> RenderedNamedIconFieldScene:
    """Render a deterministic single-panel field of procedural named icons."""

    if not icon_specs:
        raise ValueError("icon_specs must contain at least one item")
    layout_mode_values = tuple(str(value) for value in layout_modes if str(value).strip())
    if not layout_mode_values:
        layout_mode_values = ("jittered_grid",)
    layout_mode = str(layout_mode_values[int(rng.randrange(0, len(layout_mode_values)))])

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    rows, cols = _resolve_grid(count=len(icon_specs), content_bbox=layout.scene_content_xyxy)
    _draw_layout_guides(
        image,
        content_bbox=tuple(int(value) for value in layout.scene_content_xyxy),
        rows=int(rows),
        layout_mode=str(layout_mode),
    )
    slots = _grid_slots(
        tuple(int(value) for value in layout.scene_content_xyxy),
        rows=int(rows),
        cols=int(cols),
        inner_padding_px=int(slot_padding_px),
    )
    if len(slots) < len(icon_specs):
        raise ValueError("not enough named-icon field slots")
    slot_order = list(slots)
    rng.shuffle(slot_order)
    if str(layout_mode) == "ordered_grid":
        slot_order = list(slots)

    rendered: List[RenderedNamedIconInstance] = []

    def add_instance(index: int, spec: NamedIconFieldSpec, sprite: Image.Image, paste_x: int, paste_y: int, row: int, col: int, nominal_size: int) -> None:
        image.alpha_composite(sprite, (int(paste_x), int(paste_y)))
        rendered.append(
            RenderedNamedIconInstance(
                instance_id=f"named_icon_{int(index):02d}",
                shape_id=str(spec.shape_id),
                shape_name=procedural_named_icon_display_name(str(spec.shape_id)),
                bbox_xyxy=(int(paste_x), int(paste_y), int(paste_x + sprite.size[0]), int(paste_y + sprite.size[1])),
                nominal_size_px=int(nominal_size),
                rotation_degrees=int(spec.rotation_degrees) % 360,
                tint_rgb=tuple(int(value) for value in spec.tint_rgb),
                color_name=str(spec.color_name or ""),
                fill_style=str(spec.fill_style or "solid"),
                layout_row=int(row),
                layout_col=int(col),
                placement_group=str(spec.placement_group or ""),
                noise_edits=serialize_icon_noise_edits(tuple(spec.noise_edits)),
                noise_seed=None if spec.noise_seed is None else int(spec.noise_seed),
            )
        )

    if str(layout_mode) == "free_scatter":
        content_x0, content_y0, content_x1, content_y1 = tuple(int(value) for value in layout.scene_content_xyxy)
        existing_bboxes: List[BBox] = []
        for index, spec in enumerate(icon_specs):
            nominal_size = int(spec.nominal_size_px)
            placed = False
            for shrink_round in range(8):
                candidate_size = max(20, int(round(float(nominal_size) * (0.92 ** int(shrink_round)))))
                sprite = _sprite_for_spec(spec, nominal_size=int(candidate_size))
                max_x = int(content_x1 - sprite.size[0])
                max_y = int(content_y1 - sprite.size[1])
                if max_x <= content_x0 or max_y <= content_y0:
                    continue
                for _ in range(220):
                    paste_x = int(rng.randint(int(content_x0), int(max_x)))
                    paste_y = int(rng.randint(int(content_y0), int(max_y)))
                    bbox = (paste_x, paste_y, paste_x + int(sprite.size[0]), paste_y + int(sprite.size[1]))
                    if _overlaps_existing(bbox, existing_bboxes, gap_px=max(2, int(slot_padding_px) // 2)):
                        continue
                    existing_bboxes.append(bbox)
                    add_instance(index, spec, sprite, paste_x, paste_y, index // max(1, cols), index % max(1, cols), int(candidate_size))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                raise ValueError("could not place free-scatter named icon without collision")
    elif str(layout_mode) in {"clustered_by_shape", "shape_stacks", "target_stack_with_oddballs", "mixed_stacks", "rows_by_shape"}:
        compact_stack_modes = {"shape_stacks", "target_stack_with_oddballs", "mixed_stacks"}
        groups = _group_indices(icon_specs)
        group_rows, group_cols = _resolve_grid(count=len(groups), content_bbox=layout.scene_content_xyxy)
        group_slots = _grid_slots(
            tuple(int(value) for value in layout.scene_content_xyxy),
            rows=int(group_rows),
            cols=int(group_cols),
            inner_padding_px=max(8, int(slot_padding_px)),
        )
        if len(group_slots) < len(groups):
            raise ValueError("not enough named-icon group slots")
        for group_index, (_group_name, indices) in enumerate(groups):
            group_bbox, group_row, group_col = group_slots[int(group_index)]
            if str(layout_mode) in compact_stack_modes:
                inner_rows, inner_cols = _compact_stack_grid(count=len(indices), bbox=group_bbox)
                gap_px = max(0, int(stack_gap_px))
                gx0, gy0, gx1, gy1 = tuple(int(value) for value in group_bbox)
                available_w = max(1, gx1 - gx0)
                available_h = max(1, gy1 - gy0)
                max_cell_w = (available_w - max(0, int(inner_cols) - 1) * gap_px) // max(1, int(inner_cols))
                max_cell_h = (available_h - max(0, int(inner_rows) - 1) * gap_px) // max(1, int(inner_rows))
                nominal_group_size = max(int(icon_specs[int(icon_index)].nominal_size_px) for icon_index in indices)
                stack_size = max(12, min(int(nominal_group_size), int(max_cell_w), int(max_cell_h)))
                group_sprites: List[Tuple[int, NamedIconFieldSpec, Image.Image]] = []
                max_sprite_w = 0
                max_sprite_h = 0
                for shrink_round in range(8):
                    candidate_size = max(12, int(round(float(stack_size) * (0.92 ** int(shrink_round)))))
                    candidate_sprites = [
                        (
                            int(icon_index),
                            icon_specs[int(icon_index)],
                            _sprite_for_spec(icon_specs[int(icon_index)], nominal_size=int(candidate_size)),
                        )
                        for icon_index in indices
                    ]
                    candidate_max_w = max(int(sprite.size[0]) for _icon_index, _spec, sprite in candidate_sprites)
                    candidate_max_h = max(int(sprite.size[1]) for _icon_index, _spec, sprite in candidate_sprites)
                    block_w = int(inner_cols) * int(candidate_max_w) + max(0, int(inner_cols) - 1) * gap_px
                    block_h = int(inner_rows) * int(candidate_max_h) + max(0, int(inner_rows) - 1) * gap_px
                    if block_w <= available_w and block_h <= available_h:
                        stack_size = int(candidate_size)
                        group_sprites = candidate_sprites
                        max_sprite_w = int(candidate_max_w)
                        max_sprite_h = int(candidate_max_h)
                        break
                if not group_sprites:
                    raise ValueError("could not fit compact named-icon stack")
                block_w = int(inner_cols) * int(max_sprite_w) + max(0, int(inner_cols) - 1) * gap_px
                block_h = int(inner_rows) * int(max_sprite_h) + max(0, int(inner_rows) - 1) * gap_px
                start_x = int(gx0 + max(0, available_w - block_w) // 2)
                start_y = int(gy0 + max(0, available_h - block_h) // 2)
                for inner_index, (icon_index, spec, sprite) in enumerate(group_sprites):
                    inner_row = int(inner_index) // max(1, int(inner_cols))
                    inner_col = int(inner_index) % max(1, int(inner_cols))
                    paste_x = int(start_x + int(inner_col) * (int(max_sprite_w) + gap_px) + max(0, int(max_sprite_w) - int(sprite.size[0])) // 2)
                    paste_y = int(start_y + int(inner_row) * (int(max_sprite_h) + gap_px) + max(0, int(max_sprite_h) - int(sprite.size[1])) // 2)
                    add_instance(
                        int(icon_index),
                        spec,
                        sprite,
                        int(paste_x),
                        int(paste_y),
                        int(group_row) * 100 + int(inner_row),
                        int(group_col) * 100 + int(inner_col),
                        int(stack_size),
                    )
                continue

            if str(layout_mode) == "rows_by_shape":
                inner_rows, inner_cols = 1, len(indices)
            else:
                inner_rows, inner_cols = _resolve_grid(count=len(indices), content_bbox=group_bbox)
            inner_slots = _grid_slots(
                tuple(int(value) for value in group_bbox),
                rows=int(inner_rows),
                cols=int(inner_cols),
                inner_padding_px=2 if str(layout_mode) != "clustered_by_shape" else 4,
            )
            if len(inner_slots) < len(indices):
                raise ValueError("not enough named-icon stack slots")
            if str(layout_mode) == "clustered_by_shape":
                rng.shuffle(inner_slots)
            for inner_index, icon_index in enumerate(indices):
                spec = icon_specs[int(icon_index)]
                slot_bbox, inner_row, inner_col = inner_slots[int(inner_index)]
                max_slot_size = max(12, min(int(slot_bbox[2] - slot_bbox[0]), int(slot_bbox[3] - slot_bbox[1])))
                nominal_size = min(int(spec.nominal_size_px), int(max_slot_size))
                sprite = _sprite_for_spec(spec, nominal_size=int(nominal_size))
                jitter = int(slot_jitter_px) if str(layout_mode) == "clustered_by_shape" else 0
                paste_x, paste_y = _centered_paste_xy(
                    sprite_size=sprite.size,
                    slot_bbox=slot_bbox,
                    jitter_px=int(jitter),
                    rng=rng,
                )
                add_instance(
                    int(icon_index),
                    spec,
                    sprite,
                    int(paste_x),
                    int(paste_y),
                    int(group_row) * 100 + int(inner_row),
                    int(group_col) * 100 + int(inner_col),
                    int(nominal_size),
                )
    else:
        for index, spec in enumerate(icon_specs):
            slot_bbox, row, col = slot_order[int(index)]
            max_slot_size = max(12, min(int(slot_bbox[2] - slot_bbox[0]), int(slot_bbox[3] - slot_bbox[1])))
            nominal_size = min(int(spec.nominal_size_px), int(max_slot_size))
            sprite = _sprite_for_spec(spec, nominal_size=int(nominal_size))
            paste_x, paste_y = _centered_paste_xy(
                sprite_size=sprite.size,
                slot_bbox=slot_bbox,
                jitter_px=int(slot_jitter_px),
                rng=rng,
            )
            add_instance(index, spec, sprite, int(paste_x), int(paste_y), int(row), int(col), int(nominal_size))

    return RenderedNamedIconFieldScene(
        image=image.convert("RGB"),
        instances=tuple(rendered),
        panel_geometry=single_panel_geometry_to_trace(layout),
        layout_mode=str(layout_mode),
    )


def serialize_named_icon_instance(instance: RenderedNamedIconInstance) -> Dict[str, Any]:
    """Serialize one rendered named icon for Trace payloads."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(instance.instance_id),
        "shape_id": str(instance.shape_id),
        "shape_name": str(instance.shape_name),
        "bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "nominal_size_px": int(instance.nominal_size_px),
        "rotation_degrees": int(instance.rotation_degrees),
        "tint_rgb": [int(value) for value in instance.tint_rgb],
        "color_name": str(instance.color_name),
        "fill_style": str(instance.fill_style),
        "layout_row": int(instance.layout_row),
        "layout_col": int(instance.layout_col),
        "placement_group": str(instance.placement_group),
        "noise_edits": [dict(edit) for edit in instance.noise_edits],
        "noise_seed": None if instance.noise_seed is None else int(instance.noise_seed),
    }


def named_icon_bboxes_for_shape(
    instances: Sequence[RenderedNamedIconInstance],
    *,
    shape_id: str,
) -> List[List[int]]:
    """Return reading-order bboxes for all instances of a target shape."""

    return sort_bboxes_reading_order(
        tuple(instance.bbox_xyxy for instance in instances if str(instance.shape_id) == str(shape_id))
    )


__all__ = [
    "SCENE_ID",
    "NamedIconFieldSpec",
    "DEFAULT_NAMED_ICON_ROTATION_JITTER_DEGREES",
    "RenderedNamedIconFieldScene",
    "RenderedNamedIconInstance",
    "named_icon_bboxes_for_shape",
    "render_procedural_named_icon_field_scene",
    "serialize_named_icon_instance",
]
