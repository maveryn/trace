"""Renderer and projection helpers for organic-structure notation scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ....shared.bbox_projection import round_bbox
from ....shared.text_rendering import load_font

from .state import OrganicProjection, OrganicRenderParams, OrganicRenderedStructure, OrganicStructureSpec, RenderedOrganicScene


def project_organic_structure(
    *,
    spec: OrganicStructureSpec,
    panel_bbox: Tuple[int, int, int, int],
    structure_width_px: int,
    structure_height_px: int,
    instance_seed: int,
    sampling_scope: str,
) -> OrganicProjection:
    """Project structure coordinates into final pixel coordinates."""

    rng = spawn_rng(int(instance_seed), f"{sampling_scope}.layout_projection")
    min_x = min(atom.x for atom in spec.atoms)
    max_x = max(atom.x for atom in spec.atoms)
    min_y = min(atom.y for atom in spec.atoms)
    max_y = max(atom.y for atom in spec.atoms)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    available_w = min(int(structure_width_px), int(panel_bbox[2] - panel_bbox[0]) - 110)
    available_h = min(int(structure_height_px), int(panel_bbox[3] - panel_bbox[1]) - 120)
    scale = min(float(available_w) / width, float(available_h) / height)
    scale *= rng.choice((0.82, 0.88, 0.94))
    cx = (panel_bbox[0] + panel_bbox[2]) / 2.0 + rng.uniform(-26.0, 26.0)
    cy = (panel_bbox[1] + panel_bbox[3]) / 2.0 + rng.uniform(-18.0, 18.0)
    source_cx = (min_x + max_x) / 2.0
    source_cy = (min_y + max_y) / 2.0
    points = tuple((cx + (atom.x - source_cx) * scale, cy + (atom.y - source_cy) * scale) for atom in spec.atoms)
    text_points = tuple(
        (cx + (label.x - source_cx) * scale, cy + (label.y - source_cy) * scale)
        for label in spec.text_labels
    )
    return OrganicProjection(
        atom_points_px=points,
        metadata={
            "projected_atoms_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
            "projected_text_labels_px": [[round(float(x), 3), round(float(y), 3)] for x, y in text_points],
            "atom_count": int(len(points)),
            "bond_count": int(len(spec.bonds)),
            "text_label_count": int(len(spec.text_labels)),
            "projection_scale": round(float(scale), 6),
            "scaffold_id": str(spec.scaffold_id),
            "scaffold_family": str(spec.scaffold_family),
        },
        text_label_points_px=text_points,
    )


def _bond_bbox(p0: Tuple[float, float], p1: Tuple[float, float], pad: float) -> Tuple[float, float, float, float]:
    return round_bbox((min(p0[0], p1[0]) - pad, min(p0[1], p1[1]) - pad, max(p0[0], p1[0]) + pad, max(p0[1], p1[1]) + pad))


def _shortened_points(p0: Tuple[float, float], p1: Tuple[float, float], shorten_px: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    length = math.hypot(dx, dy) or 1.0
    amount = min(float(shorten_px), length * 0.18)
    ux = dx / length
    uy = dy / length
    return (p0[0] + ux * amount, p0[1] + uy * amount), (p1[0] - ux * amount, p1[1] - uy * amount)


def _draw_offset_line(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    offset: float,
    fill: Tuple[int, int, int],
    width: int,
) -> None:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    length = math.hypot(dx, dy) or 1.0
    nx = -dy / length
    ny = dx / length
    draw.line(
        (p0[0] + nx * offset, p0[1] + ny * offset, p1[0] + nx * offset, p1[1] + ny * offset),
        fill=fill,
        width=int(width),
        joint="curve",
    )


def _ring_centers(spec: OrganicStructureSpec, points: Sequence[Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    centers: Dict[int, Tuple[float, float]] = {}
    for ring_index, ring_atoms in enumerate(spec.ring_atom_sets):
        xs = [float(points[int(idx)][0]) for idx in ring_atoms]
        ys = [float(points[int(idx)][1]) for idx in ring_atoms]
        centers[int(ring_index)] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centers


def _draw_bond(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    order: str,
    ring_center: Tuple[float, float] | None,
    bond_rgb: Tuple[int, int, int],
    bond_width_px: int,
    bond_gap_px: int,
) -> None:
    """Render one skeletal bond while preserving order-specific visual witnesses.

    Ring double bonds draw the second stroke toward the ring interior; acyclic
    double and triple bonds use symmetric offsets so projected bond segments
    remain aligned with the canonical atom-to-atom witness.
    """

    if str(order) == "double" and ring_center is not None:
        inner0, inner1 = _shortened_points(p0, p1, max(8.0, float(bond_gap_px)))
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=bond_rgb, width=int(bond_width_px), joint="curve")
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        length = math.hypot(dx, dy) or 1.0
        nx = -dy / length
        ny = dx / length
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        if ((ring_center[0] - mx) * nx + (ring_center[1] - my) * ny) < 0:
            nx *= -1.0
            ny *= -1.0
        offset = float(bond_gap_px)
        draw.line(
            (inner0[0] + nx * offset, inner0[1] + ny * offset, inner1[0] + nx * offset, inner1[1] + ny * offset),
            fill=bond_rgb,
            width=int(bond_width_px),
            joint="curve",
        )
        return
    if str(order) == "double":
        line0, line1 = _shortened_points(p0, p1, max(5.0, float(bond_gap_px) * 0.75))
        gap = float(bond_gap_px) / 2.0
        _draw_offset_line(draw, line0, line1, offset=-gap, fill=bond_rgb, width=int(bond_width_px))
        _draw_offset_line(draw, line0, line1, offset=gap, fill=bond_rgb, width=int(bond_width_px))
        return
    if str(order) == "triple":
        line0, line1 = _shortened_points(p0, p1, max(7.0, float(bond_gap_px)))
        gap = float(bond_gap_px)
        _draw_offset_line(draw, line0, line1, offset=-gap, fill=bond_rgb, width=int(bond_width_px))
        _draw_offset_line(draw, line0, line1, offset=0.0, fill=bond_rgb, width=int(bond_width_px))
        _draw_offset_line(draw, line0, line1, offset=gap, fill=bond_rgb, width=int(bond_width_px))
        return
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=bond_rgb, width=int(bond_width_px), joint="curve")


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    text: str,
    *,
    font_size_px: int,
    fill: Tuple[int, int, int],
) -> Tuple[float, float, float, float]:
    font = load_font(int(font_size_px), bold=True)
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    x0 = float(center[0]) - text_w / 2.0 - 4.0
    y0 = float(center[1]) - text_h / 2.0 - 3.0
    x1 = float(center[0]) + text_w / 2.0 + 4.0
    y1 = float(center[1]) + text_h / 2.0 + 3.0
    draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=(253, 252, 247))
    draw.text(
        (
            float(center[0]) - text_w / 2.0 - float(text_bbox[0]),
            float(center[1]) - text_h / 2.0 - float(text_bbox[1]),
        ),
        str(text),
        fill=fill,
        font=font,
        stroke_width=1,
        stroke_fill=fill,
    )
    return round_bbox((x0, y0, x1, y1))


def draw_organic_structure(
    draw: ImageDraw.ImageDraw,
    *,
    spec: OrganicStructureSpec,
    projection: OrganicProjection,
    bond_rgb: Tuple[int, int, int],
    bond_width_px: int,
    bond_gap_px: int,
) -> OrganicRenderedStructure:
    """Draw one organic structure and return final geometry maps."""

    points = projection.atom_points_px
    ring_centers = _ring_centers(spec, points)
    item_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    item_segments: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    item_points: Dict[str, Tuple[float, float]] = {}
    entities: list[Dict[str, Any]] = []

    for bond in spec.bonds:
        p0 = points[int(bond.atom_a)]
        p1 = points[int(bond.atom_b)]
        _draw_bond(
            draw,
            p0,
            p1,
            order=str(bond.order),
            ring_center=ring_centers.get(int(bond.ring_index)) if bond.ring_index is not None else None,
            bond_rgb=bond_rgb,
            bond_width_px=int(bond_width_px),
            bond_gap_px=int(bond_gap_px),
        )
        bbox = _bond_bbox(p0, p1, pad=max(10.0, float(bond_gap_px + bond_width_px + 4)))
        segment = (
            (round(float(p0[0]), 3), round(float(p0[1]), 3)),
            (round(float(p1[0]), 3), round(float(p1[1]), 3)),
        )
        item_bboxes[str(bond.item_id)] = bbox
        item_segments[str(bond.item_id)] = segment
        entities.append(
            {
                "entity_id": str(bond.item_id),
                "entity_type": "organic_bond",
                "bbox_px": list(bbox),
                "segment_px": [list(point) for point in segment],
                "from_atom": int(bond.atom_a),
                "to_atom": int(bond.atom_b),
                "bond_order": str(bond.order),
                "bond_role": str(bond.role),
                "ring_index": None if bond.ring_index is None else int(bond.ring_index),
            }
        )

    for atom_index, point in enumerate(points):
        atom = spec.atoms[int(atom_index)]
        point_value = (round(float(point[0]), 3), round(float(point[1]), 3))
        item_points[str(atom.item_id)] = point_value
        atom_label_bbox = None
        if not bool(atom.implicit) or str(atom.element) != "C":
            atom_label_bbox = _draw_centered_label(
                draw,
                point,
                str(atom.element),
                font_size_px=max(20, min(36, int(float(bond_width_px) * 5.8))),
                fill=bond_rgb,
            )
            item_bboxes[str(atom.item_id)] = atom_label_bbox
        entities.append(
            {
                "entity_id": str(atom.item_id),
                "entity_type": "organic_line_angle_vertex",
                "center_px": [float(point_value[0]), float(point_value[1])],
                "element": str(atom.element),
                "implicit": bool(atom.implicit),
                "label_bbox_px": None if atom_label_bbox is None else list(atom_label_bbox),
            }
        )

    for label_index, label in enumerate(spec.text_labels):
        label_point = projection.text_label_points_px[int(label_index)]
        label_bbox = _draw_centered_label(
            draw,
            label_point,
            str(label.text),
            font_size_px=max(20, min(34, int(float(bond_width_px) * 5.8))),
            fill=bond_rgb,
        )
        item_bboxes[str(label.item_id)] = label_bbox
        entities.append(
            {
                "entity_id": str(label.item_id),
                "entity_type": "organic_text_label",
                "bbox_px": list(label_bbox),
                "center_px": [round(float(label_point[0]), 3), round(float(label_point[1]), 3)],
                "text": str(label.text),
                "label_role": str(label.role),
                "anchor_atom": None if label.anchor_atom is None else int(label.anchor_atom),
            }
        )

    for ring_index, ring_atoms in enumerate(spec.ring_atom_sets, start=1):
        xs = [points[int(idx)][0] for idx in ring_atoms]
        ys = [points[int(idx)][1] for idx in ring_atoms]
        bbox = round_bbox((min(xs), min(ys), max(xs), max(ys)))
        ring_id = f"ring_{ring_index:02d}"
        item_bboxes[ring_id] = bbox
        entities.append(
            {
                "entity_id": ring_id,
                "entity_type": "organic_ring",
                "bbox_px": list(bbox),
                "atom_indices": [int(idx) for idx in ring_atoms],
                "ring_size": int(len(ring_atoms)),
            }
        )

    return OrganicRenderedStructure(
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        item_segments=item_segments,
        item_points=item_points,
        metadata={
            "bond_render_style": "line_angle_shortened_multiple_bonds_v1",
            "text_label_render_style": "centered_label_with_panel_fill_halo_v1",
            "bond_bbox_policy": "bond_segment_bbox_with_multiple_bond_padding_for_debug_only",
            "bond_annotation_policy": "semantic_bond_endpoint_segment",
            "vertex_annotation_policy": "semantic_line_angle_vertex_center_point",
        },
    )


def draw_scene_variant_marks(draw: ImageDraw.ImageDraw, *, panel_bbox: Tuple[int, int, int, int], scene_variant: str, render_params: OrganicRenderParams) -> Dict[str, Any]:
    """Draw notebook/exam panel marks for one scene variant."""

    x0, y0, x1, y1 = panel_bbox
    if str(scene_variant) == "notebook_problem":
        for y in range(y0 + 68, y1 - 22, 34):
            draw.line((x0 + 22, y, x1 - 22, y), fill=(225, 233, 239), width=1)
        draw.line((x0 + 86, y0 + 24, x0 + 86, y1 - 24), fill=(238, 199, 199), width=1)
        return {"variant_marks": "notebook_lines"}
    if str(scene_variant) == "exam_scan":
        draw.rectangle((x0 + 26, y0 + 24, x0 + 92, y0 + 52), outline=render_params.annotation_rgb, width=1)
        draw.line((x0 + 110, y0 + 38, x1 - 34, y0 + 38), fill=(220, 222, 224), width=1)
        return {"variant_marks": "exam_header_rule"}
    return {"variant_marks": "clean_panel"}


def render_organic_scene(
    base_image: Image.Image,
    *,
    structure: OrganicStructureSpec,
    render_params: OrganicRenderParams,
    scene_variant: str,
    instance_seed: int,
    sampling_scope: str,
) -> RenderedOrganicScene:
    """Render a complete organic panel on a prepared background image."""

    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    panel_x0 = int(render_params.panel_padding_px)
    panel_y0 = int(render_params.panel_padding_px)
    panel_x1 = int(render_params.canvas_width - render_params.panel_padding_px)
    panel_y1 = int(render_params.canvas_height - render_params.panel_padding_px)
    panel_bbox = (panel_x0, panel_y0, panel_x1, panel_y1)
    draw.rounded_rectangle(
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=int(render_params.panel_border_width_px),
    )
    style_meta = draw_scene_variant_marks(draw, panel_bbox=panel_bbox, scene_variant=scene_variant, render_params=render_params)
    projection = project_organic_structure(
        spec=structure,
        panel_bbox=panel_bbox,
        structure_width_px=int(render_params.structure_width_px),
        structure_height_px=int(render_params.structure_height_px),
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
    )
    rendered_structure = draw_organic_structure(
        draw,
        spec=structure,
        projection=projection,
        bond_rgb=render_params.bond_rgb,
        bond_width_px=int(render_params.bond_width_px),
        bond_gap_px=int(render_params.bond_gap_px),
    )
    return RenderedOrganicScene(
        image=image,
        entities=tuple(rendered_structure.entities),
        scene_bbox_px=round_bbox(panel_bbox),
        item_bboxes=dict(rendered_structure.item_bboxes),
        item_segments=dict(rendered_structure.item_segments),
        item_points=dict(rendered_structure.item_points),
        layout_jitter=dict(projection.metadata),
        style_metadata={**dict(style_meta), **dict(rendered_structure.metadata)},
    )


__all__ = [
    "draw_organic_structure",
    "project_organic_structure",
    "render_organic_scene",
]
