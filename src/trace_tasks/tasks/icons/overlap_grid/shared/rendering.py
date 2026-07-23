"""Reusable reference-overlap plus labeled scene-grid rendering for icon occlusion tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ....shared.text_legibility import draw_text_traced
from ...shared.icon_assets import render_icon_rgba
from ...shared.icon_grid_scene import resolve_grid_cell_slots
from ...shared.icon_noise import NoiseEdit, serialize_icon_noise_edits
from ...shared.icon_scene import IconInstanceSpec, IconPanelLayout, draw_two_panel_panels, resolve_two_panel_layout
from ...shared.scene_style import IconCanvasStyle


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class IconOverlapPairSpec:
    """One overlapping icon pair with an explicit front/back order."""

    icon_a: IconInstanceSpec
    icon_b: IconInstanceSpec
    front_role: str
    offset_dx_frac: float
    offset_dy_frac: float
    overlap_ratio: float


@dataclass(frozen=True)
class RenderedReferenceOverlapPair:
    """Rendered metadata for one reference overlap pair."""

    icon_a_id: str
    icon_b_id: str
    front_role: str
    overlap_ratio: float
    icon_a_bbox_xyxy: BBox
    icon_b_bbox_xyxy: BBox
    icon_a_tint_rgb: Tuple[int, int, int]
    icon_b_tint_rgb: Tuple[int, int, int]
    icon_a_noise_edits: Tuple[Dict[str, Any], ...]
    icon_a_noise_seed: int | None
    icon_b_noise_edits: Tuple[Dict[str, Any], ...]
    icon_b_noise_seed: int | None


@dataclass(frozen=True)
class RenderedSceneOverlapCell:
    """Rendered metadata for one labeled overlap-order scene cell."""

    label: str
    icon_a_id: str
    icon_b_id: str
    front_role: str
    overlap_ratio: float
    cell_bbox_xyxy: BBox
    icon_a_bbox_xyxy: BBox
    icon_b_bbox_xyxy: BBox
    icon_a_tint_rgb: Tuple[int, int, int]
    icon_b_tint_rgb: Tuple[int, int, int]
    icon_a_noise_edits: Tuple[Dict[str, Any], ...]
    icon_a_noise_seed: int | None
    icon_b_noise_edits: Tuple[Dict[str, Any], ...]
    icon_b_noise_seed: int | None


@dataclass(frozen=True)
class RenderedIconOverlapGridScene:
    """Complete rendered output for one reference-overlap plus scene-grid image."""

    image: Image.Image
    layout: IconPanelLayout
    reference_pair: RenderedReferenceOverlapPair
    scene_cells: Tuple[RenderedSceneOverlapCell, ...]


def _render_spec_icon(spec: IconInstanceSpec, *, nominal_size_px: int) -> Image.Image:
    """Render one icon instance from the shared instance spec."""

    return render_icon_rgba(
        icon_id=str(spec.icon_id),
        size_px=int(spec.nominal_size_px if spec.nominal_size_px is not None else nominal_size_px),
        tint_rgb=tuple(int(v) for v in spec.tint_rgb),
        rotation_degrees=int(spec.rotation_degrees),
        mirror_x=bool(spec.mirror_x),
        noise_edits=tuple(spec.noise_edits),
        noise_seed=spec.noise_seed,
    )


def _draw_overlap_pair_in_box(
    *,
    image: Image.Image,
    pair_spec: IconOverlapPairSpec,
    outer_bbox: BBox,
    icon_size_px: int,
) -> Tuple[BBox, BBox]:
    """Draw one overlapping icon pair inside a target box and return both icon boxes."""

    if str(pair_spec.front_role) not in {"a", "b"}:
        raise ValueError("front_role must be 'a' or 'b'")
    icon_a = _render_spec_icon(pair_spec.icon_a, nominal_size_px=int(icon_size_px))
    icon_b = _render_spec_icon(pair_spec.icon_b, nominal_size_px=int(icon_size_px))
    dx = int(round(float(pair_spec.offset_dx_frac) * float(icon_size_px)))
    dy = int(round(float(pair_spec.offset_dy_frac) * float(icon_size_px)))
    ax0_rel = 0
    ay0_rel = 0
    bx0_rel = dx
    by0_rel = dy
    min_x = min(ax0_rel, bx0_rel)
    min_y = min(ay0_rel, by0_rel)
    max_x = max(ax0_rel + int(icon_a.size[0]), bx0_rel + int(icon_b.size[0]))
    max_y = max(ay0_rel + int(icon_a.size[1]), by0_rel + int(icon_b.size[1]))
    total_w = int(max_x - min_x)
    total_h = int(max_y - min_y)
    x0, y0, x1, y1 = outer_bbox
    if total_w > int(x1 - x0) or total_h > int(y1 - y0):
        raise ValueError("overlap pair does not fit inside target box")
    pair_x0 = int(x0 + max(0, ((x1 - x0) - total_w) // 2))
    pair_y0 = int(y0 + max(0, ((y1 - y0) - total_h) // 2))
    icon_a_x0 = int(pair_x0 + ax0_rel - min_x)
    icon_a_y0 = int(pair_y0 + ay0_rel - min_y)
    icon_b_x0 = int(pair_x0 + bx0_rel - min_x)
    icon_b_y0 = int(pair_y0 + by0_rel - min_y)
    icon_a_box = (
        int(icon_a_x0),
        int(icon_a_y0),
        int(icon_a_x0 + icon_a.size[0]),
        int(icon_a_y0 + icon_a.size[1]),
    )
    icon_b_box = (
        int(icon_b_x0),
        int(icon_b_y0),
        int(icon_b_x0 + icon_b.size[0]),
        int(icon_b_y0 + icon_b.size[1]),
    )
    if str(pair_spec.front_role) == "a":
        image.alpha_composite(icon_b, (int(icon_b_box[0]), int(icon_b_box[1])))
        image.alpha_composite(icon_a, (int(icon_a_box[0]), int(icon_a_box[1])))
    else:
        image.alpha_composite(icon_a, (int(icon_a_box[0]), int(icon_a_box[1])))
        image.alpha_composite(icon_b, (int(icon_b_box[0]), int(icon_b_box[1])))
    return icon_a_box, icon_b_box


def render_two_panel_icon_overlap_grid_scene(
    *,
    reference_pair: IconOverlapPairSpec,
    scene_pairs: Sequence[IconOverlapPairSpec],
    scene_labels: Sequence[str],
    canvas_width: int,
    canvas_height: int,
    reference_panel_width_px: int,
    outer_margin_px: int,
    panel_gap_px: int,
    panel_padding_px: int,
    panel_corner_radius_px: int,
    cell_padding_px: int,
    scene_icon_size_min_px: int,
    scene_icon_size_max_px: int,
    reference_icon_size_px: int,
    cell_label_font_size_px: int,
    panel_title_font_size_px: int,
    background_rgb: Tuple[int, int, int],
    panel_fill_rgb: Tuple[int, int, int],
    panel_border_rgb: Tuple[int, int, int],
    title_color_rgb: Tuple[int, int, int],
    cell_border_rgb: Tuple[int, int, int],
    cell_label_color_rgb: Tuple[int, int, int],
    cell_label_stroke_rgb: Tuple[int, int, int] | None = None,
    cell_label_stroke_width_px: int = 1,
    icon_canvas_style: IconCanvasStyle | None = None,
) -> RenderedIconOverlapGridScene:
    """Render one reference-overlap plus labeled scene-grid image."""

    if len(scene_pairs) != len(scene_labels):
        raise ValueError("scene pair specs and labels must have the same length")
    if not scene_pairs:
        raise ValueError("scene_pairs must contain at least one cell")

    layout = resolve_two_panel_layout(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        reference_panel_width_px=int(reference_panel_width_px),
        outer_margin_px=int(outer_margin_px),
        panel_gap_px=int(panel_gap_px),
        panel_padding_px=int(panel_padding_px),
        title_font_size_px=int(panel_title_font_size_px),
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_two_panel_panels(
        image=image,
        layout=layout,
        background_rgb=background_rgb,
        panel_fill_rgb=panel_fill_rgb,
        panel_border_rgb=panel_border_rgb,
        title_color_rgb=title_color_rgb,
        corner_radius_px=int(panel_corner_radius_px),
        title_font_size_px=int(panel_title_font_size_px),
        reference_title="Reference",
        scene_title="Scene",
        icon_canvas_style=icon_canvas_style,
    )

    reference_a_box, reference_b_box = _draw_overlap_pair_in_box(
        image=image,
        pair_spec=reference_pair,
        outer_bbox=tuple(int(value) for value in layout.reference_content_xyxy),
        icon_size_px=int(reference_icon_size_px),
    )

    slots = resolve_grid_cell_slots(
        tuple(int(value) for value in layout.scene_content_xyxy),
        cell_count=len(scene_pairs),
        cell_padding_px=int(cell_padding_px),
    )
    rendered_cells: List[RenderedSceneOverlapCell] = []
    label_font = load_font(int(cell_label_font_size_px), bold=True)
    draw = ImageDraw.Draw(image)
    scene_min_size = max(20, int(scene_icon_size_min_px))
    scene_max_size = max(scene_min_size, int(scene_icon_size_max_px))
    for label, pair_spec, cell_bbox in zip(scene_labels, scene_pairs, slots):
        draw.rounded_rectangle(
            cell_bbox,
            radius=12,
            outline=tuple(int(v) for v in cell_border_rgb),
            width=2,
            fill=tuple(int(v) for v in panel_fill_rgb),
        )
        draw_text_traced(draw,
            (int(cell_bbox[0] + 16), int(cell_bbox[1] + 14)),
            str(label),
            font=label_font,
            fill=tuple(int(v) for v in cell_label_color_rgb),
            stroke_fill=tuple(int(v) for v in (cell_label_stroke_rgb or panel_fill_rgb)),
            stroke_width=max(0, int(cell_label_stroke_width_px)),
            role="icon_cell_label_text",
            required=False,
        )
        pair_bbox = (
            int(cell_bbox[0] + 12),
            int(cell_bbox[1] + 42),
            int(cell_bbox[2] - 12),
            int(cell_bbox[3] - 12),
        )
        pair_span_w = max(1, int(pair_bbox[2] - pair_bbox[0]))
        pair_span_h = max(1, int(pair_bbox[3] - pair_bbox[1]))
        icon_cap_from_width = max(
            20,
            int(
                pair_span_w
                / max(1.0, 1.0 + abs(float(pair_spec.offset_dx_frac)))
            ),
        )
        icon_cap_from_height = max(
            20,
            int(
                pair_span_h
                / max(1.0, 1.0 + abs(float(pair_spec.offset_dy_frac)))
            ),
        )
        icon_size = min(scene_max_size, icon_cap_from_width, icon_cap_from_height)
        icon_size = max(scene_min_size, icon_size)
        icon_a_box, icon_b_box = _draw_overlap_pair_in_box(
            image=image,
            pair_spec=pair_spec,
            outer_bbox=pair_bbox,
            icon_size_px=int(icon_size),
        )
        rendered_cells.append(
            RenderedSceneOverlapCell(
                label=str(label),
                icon_a_id=str(pair_spec.icon_a.icon_id),
                icon_b_id=str(pair_spec.icon_b.icon_id),
                front_role=str(pair_spec.front_role),
                overlap_ratio=float(pair_spec.overlap_ratio),
                cell_bbox_xyxy=tuple(int(v) for v in cell_bbox),
                icon_a_bbox_xyxy=tuple(int(v) for v in icon_a_box),
                icon_b_bbox_xyxy=tuple(int(v) for v in icon_b_box),
                icon_a_tint_rgb=tuple(int(v) for v in pair_spec.icon_a.tint_rgb),
                icon_b_tint_rgb=tuple(int(v) for v in pair_spec.icon_b.tint_rgb),
                icon_a_noise_edits=serialize_icon_noise_edits(pair_spec.icon_a.noise_edits),
                icon_a_noise_seed=None if pair_spec.icon_a.noise_seed is None else int(pair_spec.icon_a.noise_seed),
                icon_b_noise_edits=serialize_icon_noise_edits(pair_spec.icon_b.noise_edits),
                icon_b_noise_seed=None if pair_spec.icon_b.noise_seed is None else int(pair_spec.icon_b.noise_seed),
            )
        )

    rendered_reference = RenderedReferenceOverlapPair(
        icon_a_id=str(reference_pair.icon_a.icon_id),
        icon_b_id=str(reference_pair.icon_b.icon_id),
        front_role=str(reference_pair.front_role),
        overlap_ratio=float(reference_pair.overlap_ratio),
        icon_a_bbox_xyxy=tuple(int(v) for v in reference_a_box),
        icon_b_bbox_xyxy=tuple(int(v) for v in reference_b_box),
        icon_a_tint_rgb=tuple(int(v) for v in reference_pair.icon_a.tint_rgb),
        icon_b_tint_rgb=tuple(int(v) for v in reference_pair.icon_b.tint_rgb),
        icon_a_noise_edits=serialize_icon_noise_edits(reference_pair.icon_a.noise_edits),
        icon_a_noise_seed=None if reference_pair.icon_a.noise_seed is None else int(reference_pair.icon_a.noise_seed),
        icon_b_noise_edits=serialize_icon_noise_edits(reference_pair.icon_b.noise_edits),
        icon_b_noise_seed=None if reference_pair.icon_b.noise_seed is None else int(reference_pair.icon_b.noise_seed),
    )
    return RenderedIconOverlapGridScene(
        image=image,
        layout=layout,
        reference_pair=rendered_reference,
        scene_cells=tuple(rendered_cells),
    )


__all__ = [
    "IconOverlapPairSpec",
    "RenderedIconOverlapGridScene",
    "RenderedReferenceOverlapPair",
    "RenderedSceneOverlapCell",
    "render_two_panel_icon_overlap_grid_scene",
]
