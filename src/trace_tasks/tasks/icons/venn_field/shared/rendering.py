"""Renderer for one-panel Venn-field icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import (
    BBox,
    draw_single_panel,
    max_overlap_with_existing,
    resolve_single_panel_layout,
    single_panel_geometry_to_trace,
)
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icons import (
    procedural_named_icon_display_name,
    render_procedural_named_icon_rgba,
)

from .spatial_primitives import (
    category_membership,
    sample_center_for_category,
    sample_venn_spec,
)
from .state import RenderedVennIcon, VennIconPlan, VennScenePayload, VennSpec


def render_venn_field_scene(
    rng: Any,
    *,
    instance_seed: int,
    content_namespace: str,
    object_count: int,
    target_count: int,
    plans: Sequence[VennIconPlan],
    counted_categories: Sequence[str],
    render_params: Mapping[str, Any],
) -> VennScenePayload:
    """Render a Venn field from task-owned icon plans."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    venn = sample_venn_spec(rng, content_bbox=content_bbox)
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(
            int(value) for value in render_params["background_color_rgb"]
        ),
        panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(
            int(value) for value in render_params["panel_border_rgb"]
        ),
        title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    _draw_venn_underlay(image, venn=venn, render_params=render_params)

    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    existing_bboxes: list[BBox] = []
    rendered: list[RenderedVennIcon] = []
    margin_px = int(render_params["venn_boundary_margin_px"])
    counted_category_set = {str(value) for value in counted_categories}
    for index, plan in enumerate(plans):
        placed = False
        nominal_size = int(rng.randint(int(min_size), int(max_size)))
        rotation = _rotation_for_shape(rng, str(plan.shape_id))
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{content_namespace}.named_icon_{int(index)}",
            render_params=render_params,
        )
        for shrink_round in range(7):
            candidate_size = max(
                28, int(round(float(nominal_size) * (0.92 ** int(shrink_round))))
            )
            sprite = render_procedural_named_icon_rgba(
                shape_id=str(plan.shape_id),
                size_px=int(candidate_size),
                tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                fill_style=str(plan.fill_style),
                rotation_degrees=int(rotation),
                mirror_x=False,
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            for _ in range(int(render_params["scene_placement_max_attempts"])):
                cx, cy, bbox = sample_center_for_category(
                    rng,
                    content_bbox=content_bbox,
                    sprite_size=tuple(int(value) for value in sprite.size),
                    venn=venn,
                    category=str(plan.venn_category),
                    margin_px=int(margin_px),
                )
                if max_overlap_with_existing(bbox, existing_bboxes) > float(
                    render_params["scene_max_overlap_fraction"]
                ):
                    continue
                image.alpha_composite(sprite, (int(bbox[0]), int(bbox[1])))
                left_inside, right_inside = category_membership(str(plan.venn_category))
                counted = bool(
                    plan.matches_target
                    and str(plan.venn_category) in counted_category_set
                )
                rendered.append(
                    RenderedVennIcon(
                        instance_id=f"named_venn_icon_{int(index):02d}",
                        shape_id=str(plan.shape_id),
                        shape_name=procedural_named_icon_display_name(
                            str(plan.shape_id)
                        ),
                        color_name=str(plan.color_name),
                        bbox_xyxy=tuple(int(value) for value in bbox),
                        center_xy=(float(cx), float(cy)),
                        nominal_size_px=int(candidate_size),
                        rotation_degrees=int(rotation),
                        tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                        fill_style=str(plan.fill_style),
                        venn_category=str(plan.venn_category),
                        inside_left_circle=bool(left_inside),
                        inside_right_circle=bool(right_inside),
                        matches_target=bool(plan.matches_target),
                        counted=bool(counted),
                        is_reference=bool(plan.is_reference),
                        noise_edits=serialize_icon_noise_edits(tuple(noise_edits)),
                        noise_seed=int(noise_seed),
                    )
                )
                existing_bboxes.append(tuple(int(value) for value in bbox))
                placed = True
                break
            if placed:
                break
        if not placed:
            raise ValueError(
                "could not place named Venn icon with requested membership"
            )

    _draw_venn_outline(image, venn=venn, render_params=render_params)
    for instance in rendered:
        if instance.is_reference:
            _draw_reference_marker(
                image, bbox=instance.bbox_xyxy, render_params=render_params
            )
    counted_count = sum(1 for instance in rendered if instance.counted)
    if int(counted_count) != int(target_count):
        raise RuntimeError("rendered Venn count did not match target answer")

    sampled_palette_rgb = tuple(
        tuple(int(channel) for channel in plan.tint_rgb) for plan in plans
    )
    return VennScenePayload(
        image=image.convert("RGB"),
        panel_geometry=single_panel_geometry_to_trace(layout),
        venn=venn,
        target_count=int(target_count),
        object_count=int(object_count),
        instances=tuple(rendered),
        sampled_palette_rgb=tuple(dict.fromkeys(sampled_palette_rgb)),
    )


def render_venn_field_with_retries(
    *,
    instance_seed: int,
    max_attempts: int,
    content_namespace: str,
    object_count: int,
    target_count: int,
    plans: Sequence[VennIconPlan],
    counted_categories: Sequence[str],
    render_params: Mapping[str, Any],
) -> VennScenePayload:
    """Retry deterministic scene rendering for placement-heavy Venn fields."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{content_namespace}.scene", int(attempt))
        try:
            return render_venn_field_scene(
                rng,
                instance_seed=int(instance_seed),
                content_namespace=str(content_namespace),
                object_count=int(object_count),
                target_count=int(target_count),
                plans=tuple(plans),
                counted_categories=tuple(counted_categories),
                render_params=render_params,
            )
        except Exception as exc:  # pragma: no cover - exercised by retry loop.
            last_error = exc
    raise RuntimeError(
        f"could not render Venn-field scene: {last_error}"
    ) from last_error


def serialize_venn_icon(instance: RenderedVennIcon) -> dict[str, Any]:
    """Serialize one rendered procedural icon for trace metadata."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(instance.instance_id),
        "shape_id": str(instance.shape_id),
        "shape_name": str(instance.shape_name),
        "color_name": str(instance.color_name),
        "bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "center_xy": [float(value) for value in instance.center_xy],
        "nominal_size_px": int(instance.nominal_size_px),
        "rotation_degrees": int(instance.rotation_degrees),
        "tint_rgb": [int(value) for value in instance.tint_rgb],
        "fill_style": str(instance.fill_style),
        "venn_category": str(instance.venn_category),
        "inside_left_circle": bool(instance.inside_left_circle),
        "inside_right_circle": bool(instance.inside_right_circle),
        "matches_target": bool(instance.matches_target),
        "counted": bool(instance.counted),
        "is_reference": bool(instance.is_reference),
        "noise_edits": [dict(edit) for edit in instance.noise_edits],
        "noise_seed": None if instance.noise_seed is None else int(instance.noise_seed),
    }


def _rotation_for_shape(rng: Any, shape_id: str) -> int:
    rotatable = {
        "triangle",
        "pentagon",
        "hexagon",
        "octagon",
        "arrow",
        "lightning_bolt",
        "leaf",
        "flag",
        "capsule",
        "teardrop",
        "hourglass",
        "key",
        "ladder",
        "kite",
        "rocket",
        "pencil",
    }
    if str(shape_id) not in rotatable:
        return 0
    return int(rng.choice((0, 90, 180, 270))) % 360


def _draw_venn_underlay(
    image: Image.Image, *, venn: VennSpec, render_params: Mapping[str, Any]
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(render_params["venn_fill_alpha"])
    draw.ellipse(
        tuple(int(value) for value in venn.left_bbox_xyxy),
        fill=tuple(int(value) for value in render_params["venn_left_fill_rgb"])
        + (alpha,),
    )
    draw.ellipse(
        tuple(int(value) for value in venn.right_bbox_xyxy),
        fill=tuple(int(value) for value in render_params["venn_right_fill_rgb"])
        + (alpha,),
    )
    image.alpha_composite(overlay)


def _draw_venn_outline(
    image: Image.Image, *, venn: VennSpec, render_params: Mapping[str, Any]
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(1, int(render_params["venn_outline_width_px"]))
    draw.ellipse(
        tuple(int(value) for value in venn.left_bbox_xyxy),
        outline=tuple(int(value) for value in render_params["venn_left_outline_rgb"])
        + (235,),
        width=width,
    )
    draw.ellipse(
        tuple(int(value) for value in venn.right_bbox_xyxy),
        outline=tuple(int(value) for value in render_params["venn_right_outline_rgb"])
        + (235,),
        width=width,
    )
    image.alpha_composite(overlay)


def _draw_reference_marker(
    image: Image.Image,
    *,
    bbox: Sequence[int | float],
    render_params: Mapping[str, Any],
) -> None:
    """Draw a high-contrast marker around one reference icon."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    pad = max(2, int(render_params.get("reference_marker_padding_px", 7)))
    width = max(2, int(render_params.get("reference_marker_width_px", 4)))
    outline = tuple(
        int(value)
        for value in render_params.get("reference_marker_outline_rgb", (28, 32, 42))
    )
    dot = tuple(
        int(value)
        for value in render_params.get("reference_marker_dot_rgb", (255, 255, 255))
    )
    marker_bbox = (
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(int(image.size[0] - 1), int(x1 + pad)),
        min(int(image.size[1] - 1), int(y1 + pad)),
    )
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        marker_bbox,
        radius=max(
            6,
            int(
                round(
                    0.16
                    * min(
                        marker_bbox[2] - marker_bbox[0], marker_bbox[3] - marker_bbox[1]
                    )
                )
            ),
        ),
        outline=outline + (255,),
        width=width,
    )
    dot_radius = max(4, int(round(float(width) * 1.8)))
    dot_cx = marker_bbox[0] + dot_radius + 1
    dot_cy = marker_bbox[1] + dot_radius + 1
    draw.ellipse(
        (
            int(dot_cx - dot_radius),
            int(dot_cy - dot_radius),
            int(dot_cx + dot_radius),
            int(dot_cy + dot_radius),
        ),
        fill=outline + (255,),
    )
    inner_radius = max(2, int(round(float(dot_radius) * 0.42)))
    draw.ellipse(
        (
            int(dot_cx - inner_radius),
            int(dot_cy - inner_radius),
            int(dot_cx + inner_radius),
            int(dot_cy + inner_radius),
        ),
        fill=dot + (255,),
    )


__all__ = [
    "render_venn_field_scene",
    "render_venn_field_with_retries",
    "serialize_venn_icon",
]
