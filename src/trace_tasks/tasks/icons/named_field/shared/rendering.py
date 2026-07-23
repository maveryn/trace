"""Reusable rendering primitives for named-field icon scenes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import BBox, draw_single_panel, max_overlap_with_existing, resolve_single_panel_layout, single_panel_geometry_to_trace
from ...shared.icon_style import sample_icon_palette
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icon_field_scene import (
    NamedIconFieldSpec,
    bbox_center_float,
    bbox_from_center_and_size,
    bbox_from_center_dimensions,
    bbox_inside,
    boxes_overlap,
    label_bbox_for_icon,
    render_planned_named_icon_sprite,
    rotation_for_named_shape,
)
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    render_procedural_named_icon_rgba,
    sample_procedural_named_icon_fill_style,
)
from ....shared.text_rendering import draw_text_centered, load_font
from ....shared.named_colors import available_named_colors

from .layout import candidate_distances, occupancy_bbox_for_icon, sample_region_icon_center
from .spatial_primitives import axis_radius, point_inside_region
from .state import (
    CloserReferenceRenderedIcon,
    CloserReferenceSampleSpec,
    CloserReferenceScenePayload,
    DistanceRankIconPlan,
    DistanceRankRenderedIcon,
    DistanceRankScenePayload,
    RegionSpec,
    RenderedRegionIcon,
    ScopedRegionScenePayload,
)


def build_named_icon_specs_from_semantics(
    *,
    semantic_specs: Sequence[Any],
    instance_seed: int,
    render_params: Mapping[str, Any],
    rng,
    noise_namespace: str,
) -> Tuple[Tuple[NamedIconFieldSpec, ...], Tuple[Tuple[int, int, int], ...]]:
    """Convert semantic shape/color/style records into renderable icon specs."""

    color_by_name = {
        str(name): tuple(int(channel) for channel in rgb)
        for name, rgb in available_named_colors()
    }
    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    specs: list[NamedIconFieldSpec] = []
    for index, semantic_spec in enumerate(semantic_specs):
        shape_id = str(semantic_spec.shape_id)
        if shape_id not in set(PROCEDURAL_NAMED_ICON_SHAPES):
            raise ValueError(f"unsupported named icon shape: {shape_id}")
        color_name = str(semantic_spec.color_name)
        tint_rgb = tuple(int(channel) for channel in color_by_name[str(color_name)])
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{noise_namespace}:named_icon_{int(index)}",
            render_params=render_params,
        )
        specs.append(
            NamedIconFieldSpec(
                shape_id=str(shape_id),
                tint_rgb=tint_rgb,
                color_name=str(color_name),
                fill_style=str(semantic_spec.fill_style),
                nominal_size_px=int(rng.randint(int(min_size), int(max_size))),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                placement_group="",
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    sampled_palette_rgb = tuple(
        tuple(int(channel) for channel in rgb)
        for _name, rgb in available_named_colors()
    )
    return tuple(specs), sampled_palette_rgb


def build_boolean_scene_specs(
    *,
    run_namespace: str,
    sample: Any,
    instance_seed: int,
    render_params: Mapping[str, Any],
    rng,
) -> Tuple[Tuple[NamedIconFieldSpec, ...], Tuple[Tuple[int, int, int], ...]]:
    """Project Boolean symbolic icon records to renderable named-icon specs."""

    return build_named_icon_specs_from_semantics(
        semantic_specs=sample.semantic_specs,
        instance_seed=int(instance_seed),
        render_params=render_params,
        rng=rng,
        noise_namespace=str(run_namespace),
    )


def build_counterfactual_scene_specs(
    *,
    run_namespace: str,
    sample: Any,
    instance_seed: int,
    render_params: Mapping[str, Any],
    rng,
) -> Tuple[Tuple[NamedIconFieldSpec, ...], Tuple[Tuple[int, int, int], ...]]:
    """Project counterfactual symbolic icon records to renderable named-icon specs."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    specs: list[NamedIconFieldSpec] = []
    for index, semantic_spec in enumerate(sample.semantic_specs):
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{run_namespace}:named_icon_{int(index)}",
            render_params=render_params,
        )
        specs.append(
            NamedIconFieldSpec(
                shape_id=str(semantic_spec.shape_id),
                tint_rgb=tuple(int(value) for value in rng.choice(palette)),
                nominal_size_px=int(rng.randint(int(min_size), int(max_size))),
                fill_style=sample_procedural_named_icon_fill_style(
                    rng,
                    support=sample.fill_style_support,
                    probabilities=sample.fill_style_probabilities,
                ),
                rotation_degrees=rotation_for_named_shape(rng, str(semantic_spec.shape_id)),
                placement_group="",
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    return tuple(specs), tuple(tuple(int(channel) for channel in color) for color in palette)


def build_shape_count_scene_specs(
    *,
    run_namespace: str,
    sample: Any,
    instance_seed: int,
    render_params: Mapping[str, Any],
    rng,
) -> Tuple[Tuple[NamedIconFieldSpec, ...], Tuple[Tuple[int, int, int], ...]]:
    """Convert direct shape-count semantics into renderable icon specs."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    specs: list[NamedIconFieldSpec] = []
    group_styles: dict[str, Tuple[Tuple[int, int, int], int, str]] = {}
    stack_modes = {"shape_stacks", "target_stack_with_oddballs", "mixed_stacks"}
    for index, shape_id in enumerate(sample.shape_ids):
        placement_group = str(sample.placement_groups[int(index)] or "")
        group_key = placement_group or str(shape_id)
        if str(sample.arrangement_mode) in stack_modes and group_key not in group_styles:
            group_styles[group_key] = (
                tuple(int(value) for value in rng.choice(palette)),
                int(rng.randint(int(min_size), int(max_size))),
                sample_procedural_named_icon_fill_style(
                    rng,
                    support=sample.fill_style_support,
                    probabilities=sample.fill_style_probabilities,
                ),
            )
        if str(sample.arrangement_mode) in stack_modes:
            tint_rgb, nominal_size_px, fill_style = group_styles[str(group_key)]
        else:
            tint_rgb = tuple(int(value) for value in rng.choice(palette))
            nominal_size_px = int(rng.randint(int(min_size), int(max_size)))
            fill_style = sample_procedural_named_icon_fill_style(
                rng,
                support=sample.fill_style_support,
                probabilities=sample.fill_style_probabilities,
            )
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{run_namespace}:named_icon_{int(index)}",
            render_params=render_params,
        )
        specs.append(
            NamedIconFieldSpec(
                shape_id=str(shape_id),
                tint_rgb=tuple(int(value) for value in tint_rgb),
                nominal_size_px=int(nominal_size_px),
                fill_style=str(fill_style),
                rotation_degrees=0 if str(sample.arrangement_mode) in stack_modes else rotation_for_named_shape(rng, str(shape_id)),
                placement_group=str(placement_group),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    return tuple(specs), tuple(tuple(int(channel) for channel in color) for color in palette)


def render_closer_reference_scene(
    *,
    rng,
    sample: CloserReferenceSampleSpec,
    render_params: Mapping[str, Any],
) -> CloserReferenceScenePayload:
    """Render two references and target icons while preserving distance margins."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    plans = tuple(sample.plans)
    sprites = [render_planned_named_icon_sprite(plan) for plan in plans]
    axes = tuple(int(value) for value in sample.reference_axis_probabilities)
    axis_degrees = int(rng.choice(axes)) if axes else 0
    if any(abs(float(value) - 1.0) < 1e-9 for value in sample.reference_axis_probabilities.values()):
        axis_degrees = int(next(int(key) for key, value in sample.reference_axis_probabilities.items() if abs(float(value) - 1.0) < 1e-9))

    angle = math.radians(float(axis_degrees))
    axis = (float(math.cos(angle)), float(math.sin(angle)))
    perp = (-float(axis[1]), float(axis[0]))
    center = (
        0.5 * float(content_bbox[0] + content_bbox[2]),
        0.5 * float(content_bbox[1] + content_bbox[3]),
    )
    radius = axis_radius(center, axis, content_bbox)
    max_ref_size = max(int(sprites[0].size[0]), int(sprites[0].size[1]), int(sprites[1].size[0]), int(sprites[1].size[1]))
    half_sep = max(96.0, min(190.0, float(radius) - 0.85 * float(max_ref_size)))
    if half_sep < 90.0:
        raise ValueError("content bbox too small for reference placement")
    ref_centers = {
        "A": (float(center[0]) - float(axis[0]) * half_sep, float(center[1]) - float(axis[1]) * half_sep),
        "B": (float(center[0]) + float(axis[0]) * half_sep, float(center[1]) + float(axis[1]) * half_sep),
    }
    max_proj = max(float(render_params["distance_margin_px"]) + 8.0, float(radius) - 42.0)
    perp_span = max(38.0, min(132.0, 0.42 * float(radius)))
    collision_gap = int(render_params["icon_collision_gap_px"])

    max_attempts = max(1, int(render_params["scene_placement_max_attempts"]))
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            occupancy: list[BBox] = []
            rendered: list[CloserReferenceRenderedIcon] = []
            reference_centers_actual: dict[str, Tuple[float, float]] = {}

            for index, label in enumerate(("A", "B")):
                plan = plans[int(index)]
                sprite = sprites[int(index)]
                bbox = bbox_from_center_dimensions(ref_centers[str(label)], width=int(sprite.size[0]), height=int(sprite.size[1]))
                if not bbox_inside(bbox, content_bbox):
                    raise ValueError("reference outside content")
                if any(boxes_overlap(bbox, other, gap_px=collision_gap) for other in occupancy):
                    raise ValueError("reference overlap")
                occupancy.append(bbox)
                reference_centers_actual[str(label)] = bbox_center_float(bbox)
                rendered.append(
                    CloserReferenceRenderedIcon(
                        instance_id=f"reference_{str(label).lower()}",
                        role="reference",
                        label=str(label),
                        shape_id=str(plan.shape_id),
                        shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                        color_name=str(plan.color_name),
                        tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                        fill_style=str(plan.fill_style),
                        bbox_xyxy=tuple(int(value) for value in bbox),
                        center_xy=bbox_center_float(bbox),
                        nominal_size_px=int(plan.nominal_size_px),
                        rotation_degrees=int(plan.rotation_degrees),
                        distance_to_reference_a_px=None,
                        distance_to_reference_b_px=None,
                        closer_reference_label="",
                        counted=False,
                        label_bbox_xyxy=None,
                        noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                        noise_seed=plan.noise_seed,
                    )
                )

            for target_index, (plan, sprite) in enumerate(zip(plans[2:], sprites[2:])):
                desired_label = str(plan.desired_closer_label)
                sign = -1.0 if desired_label == "A" else 1.0
                placed = False
                for _placement_attempt in range(180):
                    projection = sign * float(rng.uniform(float(render_params["distance_margin_px"]), max_proj))
                    offset = float(rng.uniform(-perp_span, perp_span))
                    candidate_center = (
                        float(center[0]) + float(axis[0]) * projection + float(perp[0]) * offset,
                        float(center[1]) + float(axis[1]) * projection + float(perp[1]) * offset,
                    )
                    bbox = bbox_from_center_dimensions(candidate_center, width=int(sprite.size[0]), height=int(sprite.size[1]))
                    if not bbox_inside(bbox, content_bbox):
                        continue
                    if any(boxes_overlap(bbox, other, gap_px=collision_gap) for other in occupancy):
                        continue
                    distance_a = math.hypot(candidate_center[0] - reference_centers_actual["A"][0], candidate_center[1] - reference_centers_actual["A"][1])
                    distance_b = math.hypot(candidate_center[0] - reference_centers_actual["B"][0], candidate_center[1] - reference_centers_actual["B"][1])
                    closer = "A" if float(distance_a) < float(distance_b) else "B"
                    if str(closer) != desired_label:
                        continue
                    if abs(float(distance_a) - float(distance_b)) < float(render_params["distance_margin_px"]):
                        continue
                    occupancy.append(bbox)
                    rendered.append(
                        CloserReferenceRenderedIcon(
                            instance_id=f"target_{int(target_index):02d}",
                            role="target",
                            label="",
                            shape_id=str(plan.shape_id),
                            shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                            color_name=str(plan.color_name),
                            tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                            fill_style=str(plan.fill_style),
                            bbox_xyxy=tuple(int(value) for value in bbox),
                            center_xy=bbox_center_float(bbox),
                            nominal_size_px=int(plan.nominal_size_px),
                            rotation_degrees=int(plan.rotation_degrees),
                            distance_to_reference_a_px=float(distance_a),
                            distance_to_reference_b_px=float(distance_b),
                            closer_reference_label=str(closer),
                            counted=str(closer) == str(sample.queried_reference_label),
                            label_bbox_xyxy=None,
                            noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                            noise_seed=plan.noise_seed,
                        )
                    )
                    placed = True
                    break
                if not placed:
                    raise ValueError("failed to place target icon with requested closer reference")

            image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
            draw_single_panel(
                image=image,
                layout=layout,
                background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
                panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
                panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
                title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
                corner_radius_px=int(render_params["panel_corner_radius_px"]),
                title_font_size_px=int(render_params["panel_title_font_size_px"]),
                scene_title="",
                icon_canvas_style=render_params.get("_icon_canvas_style_object"),
            )
            for record, sprite in zip(rendered, sprites):
                image.alpha_composite(sprite, (int(record.bbox_xyxy[0]), int(record.bbox_xyxy[1])))
            return CloserReferenceScenePayload(
                image=image.convert("RGB"),
                icons=tuple(rendered),
                panel_geometry=single_panel_geometry_to_trace(layout),
                reference_axis_degrees=int(axis_degrees),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render closer-reference icon scene") from last_error


def _draw_candidate_label(
    *,
    image: Image.Image,
    icon_bbox: BBox,
    label: str,
    content_bbox: BBox,
    label_font,
    render_params: Mapping[str, Any],
) -> BBox:
    label_bbox = label_bbox_for_icon(
        icon_bbox=tuple(int(value) for value in icon_bbox),
        label=str(label),
        content_bbox=tuple(int(value) for value in content_bbox),
        font=label_font,
        padding_px=int(render_params["candidate_label_padding_px"]),
        gap_px=int(render_params["candidate_label_gap_px"]),
    )
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        label_bbox,
        radius=max(4, int(round(0.28 * float(label_bbox[3] - label_bbox[1])))),
        fill=tuple(int(value) for value in render_params["candidate_label_background_rgb"]) + (238,),
        outline=tuple(int(value) for value in render_params["candidate_label_border_rgb"]) + (255,),
        width=1,
    )
    draw_text_centered(
        draw,
        text=str(label),
        center=bbox_center_float(label_bbox),
        font=label_font,
        fill=tuple(int(value) for value in render_params["candidate_label_color_rgb"]),
        stroke_fill=tuple(
            int(value)
            for value in render_params.get("candidate_label_stroke_rgb", render_params["candidate_label_background_rgb"])
        ),
        stroke_width=1,
    )
    return tuple(int(value) for value in label_bbox)


def render_distance_rank_scene(
    *,
    rng,
    query_name: str,
    answer_label: str,
    answer_rank: int,
    plans: Sequence[DistanceRankIconPlan],
    reference_description: str,
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    distractor_count: int,
    render_params: Mapping[str, Any],
    option_labels: Sequence[str],
    angle_pool_degrees: Sequence[int],
) -> Tuple[DistanceRankScenePayload, Image.Image]:
    """Place and render distance-rank icons, labels, and annotation geometry."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    label_font = load_font(int(render_params["candidate_label_font_size_px"]), bold=True)
    sprites = [render_planned_named_icon_sprite(plan) for plan in plans]
    reference_plan = plans[0]
    reference_sprite = sprites[0]
    candidate_plans = [plan for plan in plans if str(plan.role) == "candidate"]
    candidate_sprites = [sprites[index] for index, plan in enumerate(plans) if str(plan.role) == "candidate"]
    distractor_pairs = [(plan, sprites[index]) for index, plan in enumerate(plans) if str(plan.role) == "distractor"]

    collision_gap = int(render_params["icon_collision_gap_px"])
    max_attempts = max(1, int(render_params["scene_placement_max_attempts"]))
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            occupancy: list[BBox] = []
            rx0 = int(content_bbox[0] + int(max(reference_sprite.size)) + 36)
            rx1 = int(content_bbox[2] - int(max(reference_sprite.size)) - 36)
            ry0 = int(content_bbox[1] + int(max(reference_sprite.size)) + 40)
            ry1 = int(content_bbox[3] - int(max(reference_sprite.size)) - 40)
            if rx1 <= rx0 or ry1 <= ry0:
                raise ValueError("content bbox too small for reference icon")
            reference_center = (float(rng.randint(rx0, rx1)), float(rng.randint(ry0, ry1)))
            reference_bbox = bbox_from_center_dimensions(
                reference_center,
                width=int(reference_sprite.size[0]),
                height=int(reference_sprite.size[1]),
            )
            if not bbox_inside(reference_bbox, content_bbox):
                raise ValueError("reference icon outside content")
            occupancy.append(reference_bbox)

            labels_by_rank = [str(plan.label) for plan in candidate_plans]
            plans_by_label = {str(plan.label): plan for plan in candidate_plans}
            sprites_by_label = {str(plan.label): sprite for plan, sprite in zip(candidate_plans, candidate_sprites)}
            distances = candidate_distances(rng, render_params=render_params, option_count=len(tuple(option_labels)))
            angle_values = list(int(value) for value in angle_pool_degrees)
            rng.shuffle(angle_values)
            candidate_records: list[DistanceRankRenderedIcon] = []
            for rank, label in enumerate(labels_by_rank):
                plan = plans_by_label[str(label)]
                sprite = sprites_by_label[str(label)]
                distance = float(distances[int(rank)])
                placed = False
                for angle_degrees in angle_values:
                    angle_degrees = float(angle_degrees)
                    angle = math.radians(angle_degrees)
                    center = (
                        float(reference_center[0]) + distance * math.cos(angle),
                        float(reference_center[1]) + distance * math.sin(angle),
                    )
                    bbox = bbox_from_center_dimensions(center, width=int(sprite.size[0]), height=int(sprite.size[1]))
                    occupancy_bbox = occupancy_bbox_for_icon(
                        icon_bbox=bbox,
                        label=str(label),
                        content_bbox=content_bbox,
                        label_font=label_font,
                        render_params=render_params,
                    )
                    if not bbox_inside(occupancy_bbox, content_bbox):
                        continue
                    if any(boxes_overlap(occupancy_bbox, other, gap_px=collision_gap) for other in occupancy):
                        continue
                    occupancy.append(occupancy_bbox)
                    candidate_records.append(
                        DistanceRankRenderedIcon(
                            instance_id=f"candidate_{str(label)}",
                            role="candidate",
                            label=str(label),
                            shape_id=str(plan.shape_id),
                            shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                            color_name=str(plan.color_name),
                            tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                            fill_style=str(plan.fill_style),
                            bbox_xyxy=tuple(int(value) for value in bbox),
                            center_xy=bbox_center_float(bbox),
                            nominal_size_px=int(plan.nominal_size_px),
                            rotation_degrees=int(plan.rotation_degrees),
                            distance_to_reference_px=float(distance),
                            distance_rank=int(rank),
                            noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                            noise_seed=plan.noise_seed,
                        )
                    )
                    placed = True
                    break
                if not placed:
                    raise ValueError("failed to place distance-ranked candidate")

            distractor_records: list[DistanceRankRenderedIcon] = []
            for index, (plan, sprite) in enumerate(distractor_pairs):
                placed = False
                for _placement_attempt in range(80):
                    cx = float(rng.randint(int(content_bbox[0] + sprite.size[0] // 2), int(content_bbox[2] - sprite.size[0] // 2)))
                    cy = float(rng.randint(int(content_bbox[1] + sprite.size[1] // 2), int(content_bbox[3] - sprite.size[1] // 2)))
                    bbox = bbox_from_center_dimensions((cx, cy), width=int(sprite.size[0]), height=int(sprite.size[1]))
                    if not bbox_inside(bbox, content_bbox):
                        continue
                    if any(boxes_overlap(bbox, other, gap_px=collision_gap) for other in occupancy):
                        continue
                    occupancy.append(bbox)
                    distance = math.hypot(float(cx) - float(reference_center[0]), float(cy) - float(reference_center[1]))
                    distractor_records.append(
                        DistanceRankRenderedIcon(
                            instance_id=f"distractor_{int(index):02d}",
                            role="distractor",
                            label="",
                            shape_id=str(plan.shape_id),
                            shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                            color_name=str(plan.color_name),
                            tint_rgb=tuple(int(value) for value in plan.tint_rgb),
                            fill_style=str(plan.fill_style),
                            bbox_xyxy=tuple(int(value) for value in bbox),
                            center_xy=(float(cx), float(cy)),
                            nominal_size_px=int(plan.nominal_size_px),
                            rotation_degrees=int(plan.rotation_degrees),
                            distance_to_reference_px=float(distance),
                            distance_rank=None,
                            noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                            noise_seed=plan.noise_seed,
                        )
                    )
                    placed = True
                    break
                if not placed:
                    raise ValueError("failed to place distractor icon")

            sorted_candidates = tuple(
                sorted(candidate_records, key=lambda item: (float(item.distance_to_reference_px or 0.0), str(item.label)))
            )
            sorted_labels = tuple(str(item.label) for item in sorted_candidates)
            if sorted_labels[int(answer_rank)] != str(answer_label):
                raise ValueError("constructed candidate distances did not preserve answer rank")
            adjacent_gaps = [
                float(sorted_candidates[index + 1].distance_to_reference_px or 0.0)
                - float(sorted_candidates[index].distance_to_reference_px or 0.0)
                for index in range(len(sorted_candidates) - 1)
            ]
            if int(answer_rank) > 0 and adjacent_gaps[int(answer_rank) - 1] < float(render_params["distance_rank_margin_px"]):
                raise ValueError("distance gap before answer is too small")
            if int(answer_rank) < len(sorted_candidates) - 1 and adjacent_gaps[int(answer_rank)] < float(render_params["distance_rank_margin_px"]):
                raise ValueError("distance gap after answer is too small")

            image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
            draw_single_panel(
                image=image,
                layout=layout,
                background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
                panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
                panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
                title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
                corner_radius_px=int(render_params["panel_corner_radius_px"]),
                title_font_size_px=int(render_params["panel_title_font_size_px"]),
                scene_title="",
                icon_canvas_style=render_params.get("_icon_canvas_style_object"),
            )
            image.alpha_composite(reference_sprite, (int(reference_bbox[0]), int(reference_bbox[1])))
            for record in candidate_records:
                sprite = sprites_by_label[str(record.label)]
                image.alpha_composite(sprite, (int(record.bbox_xyxy[0]), int(record.bbox_xyxy[1])))
            for record, (_plan, sprite) in zip(distractor_records, distractor_pairs):
                image.alpha_composite(sprite, (int(record.bbox_xyxy[0]), int(record.bbox_xyxy[1])))
            for record in candidate_records:
                _draw_candidate_label(
                    image=image,
                    icon_bbox=tuple(int(value) for value in record.bbox_xyxy),
                    label=str(record.label),
                    content_bbox=content_bbox,
                    label_font=label_font,
                    render_params=render_params,
                )

            reference_record = DistanceRankRenderedIcon(
                instance_id="reference",
                role="reference",
                label="",
                shape_id=str(reference_plan.shape_id),
                shape_name=procedural_named_icon_display_name(str(reference_plan.shape_id)),
                color_name=str(reference_plan.color_name),
                tint_rgb=tuple(int(value) for value in reference_plan.tint_rgb),
                fill_style=str(reference_plan.fill_style),
                bbox_xyxy=tuple(int(value) for value in reference_bbox),
                center_xy=bbox_center_float(reference_bbox),
                nominal_size_px=int(reference_plan.nominal_size_px),
                rotation_degrees=int(reference_plan.rotation_degrees),
                distance_to_reference_px=None,
                distance_rank=None,
                noise_edits=tuple(serialize_icon_noise_edits(reference_plan.noise_edits)),
                noise_seed=reference_plan.noise_seed,
            )
            distance_by_label = {
                str(record.label): float(record.distance_to_reference_px or 0.0)
                for record in candidate_records
            }
            return (
                DistanceRankScenePayload(
                    query_key=str(query_name),
                    answer_label=str(answer_label),
                    answer_rank=int(answer_rank),
                    reference_description=str(reference_description),
                    reference_icon=reference_record,
                    candidate_icons=tuple(sorted(candidate_records, key=lambda item: str(item.label))),
                    distractor_icons=tuple(distractor_records),
                    distance_by_label=distance_by_label,
                    sorted_candidate_labels_by_distance=tuple(sorted_labels),
                    panel_geometry=single_panel_geometry_to_trace(layout),
                    sampled_palette_rgb=tuple(sampled_palette_rgb),
                    distractor_count=int(distractor_count),
                ),
                image.convert("RGB"),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render named-reference distance-rank scene") from last_error


def _draw_clipped_polygon(
    image: Image.Image,
    *,
    content_bbox: BBox,
    polygon: Sequence[Sequence[float]],
    fill_rgba: Tuple[int, int, int, int],
    outline_rgba: Tuple[int, int, int, int] | None,
    width: int,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [(float(x), float(y)) for x, y in polygon]
    draw.polygon(points, fill=tuple(int(value) for value in fill_rgba))
    if outline_rgba is not None:
        draw.line(points + [points[0]], fill=tuple(int(value) for value in outline_rgba), width=max(1, int(width)))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(tuple(int(value) for value in content_bbox), fill=255)
    alpha = ImageChops.multiply(overlay.getchannel("A"), mask)
    overlay.putalpha(alpha)
    image.alpha_composite(overlay)


def draw_region_underlay(image: Image.Image, *, region: RegionSpec, content_bbox: BBox, render_params: Mapping[str, Any]) -> None:
    """Draw the translucent fill for a visible scoped-count region."""

    fill = tuple(int(value) for value in render_params["region_fill_rgb"]) + (int(render_params["region_fill_alpha"]),)
    guide = tuple(int(value) for value in render_params["region_guide_rgb"]) + (150,)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if region.region_kind in {"shape", "quadrant", "shelf"}:
        if region.bbox_xyxy is None:
            raise ValueError("box-like region is missing bbox")
        if region.shape_kind == "ellipse":
            draw.ellipse(tuple(int(value) for value in region.bbox_xyxy), fill=fill)
        else:
            draw.rectangle(tuple(int(value) for value in region.bbox_xyxy), fill=fill)
        if region.region_kind == "quadrant":
            x0, y0, x1, y1 = [int(value) for value in content_bbox]
            xm = int(round(0.5 * float(x0 + x1)))
            ym = int(round(0.5 * float(y0 + y1)))
            draw.line((xm, y0, xm, y1), fill=guide, width=2)
            draw.line((x0, ym, x1, ym), fill=guide, width=2)
        if region.region_kind == "shelf":
            x0, y0, x1, y1 = [int(value) for value in content_bbox]
            for row in range(1, int(region.shelf_count)):
                y = int(round(float(y0) + (float(row) * float(y1 - y0) / float(max(1, int(region.shelf_count))))))
                draw.line((x0, y, x1, y), fill=guide, width=2)
        image.alpha_composite(overlay)
        return
    if region.region_kind == "band":
        _draw_clipped_polygon(
            image,
            content_bbox=content_bbox,
            polygon=region.band_polygon_xy,
            fill_rgba=fill,
            outline_rgba=None,
            width=int(render_params["region_outline_width_px"]),
        )
        return
    raise ValueError(f"unsupported region kind: {region.region_kind}")


def draw_region_outline(image: Image.Image, *, region: RegionSpec, content_bbox: BBox, render_params: Mapping[str, Any]) -> None:
    """Draw the visible boundary for a scoped-count region."""

    outline = tuple(int(value) for value in render_params["region_outline_rgb"]) + (230,)
    guide = tuple(int(value) for value in render_params["region_guide_rgb"]) + (170,)
    width = max(1, int(render_params["region_outline_width_px"]))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if region.region_kind in {"shape", "quadrant", "shelf"}:
        if region.bbox_xyxy is None:
            raise ValueError("box-like region is missing bbox")
        if region.region_kind == "quadrant":
            x0, y0, x1, y1 = [int(value) for value in content_bbox]
            xm = int(round(0.5 * float(x0 + x1)))
            ym = int(round(0.5 * float(y0 + y1)))
            draw.line((xm, y0, xm, y1), fill=guide, width=2)
            draw.line((x0, ym, x1, ym), fill=guide, width=2)
        if region.region_kind == "shelf":
            x0, y0, x1, y1 = [int(value) for value in content_bbox]
            for row in range(1, int(region.shelf_count)):
                y = int(round(float(y0) + (float(row) * float(y1 - y0) / float(max(1, int(region.shelf_count))))))
                draw.line((x0, y, x1, y), fill=guide, width=2)
        if region.shape_kind == "ellipse":
            draw.ellipse(tuple(int(value) for value in region.bbox_xyxy), outline=outline, width=width)
        else:
            draw.rectangle(tuple(int(value) for value in region.bbox_xyxy), outline=outline, width=width)
        image.alpha_composite(overlay)
        return
    if region.region_kind == "band":
        _draw_clipped_polygon(
            image,
            content_bbox=content_bbox,
            polygon=region.band_polygon_xy,
            fill_rgba=(0, 0, 0, 0),
            outline_rgba=outline,
            width=width,
        )
        return
    raise ValueError(f"unsupported region kind: {region.region_kind}")


def render_scoped_region_scene(
    *,
    rng,
    instance_seed: int,
    namespace: str,
    region: RegionSpec,
    target_shape_id: str,
    target_shape_name: str,
    target_count: int,
    object_count: int,
    plans: Sequence[Any],
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    fill_style_support: Sequence[str],
    fill_style_probabilities: Mapping[str, float],
    query_probabilities: Mapping[str, float],
    shape_probabilities: Mapping[str, float],
    target_count_probabilities: Mapping[str, float],
    object_count_probabilities: Mapping[str, float],
) -> ScopedRegionScenePayload:
    """Render scoped-region plans after the public task has sampled semantics."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    draw_region_underlay(image, region=region, content_bbox=content_bbox, render_params=render_params)

    counts_inside = bool(region.counts_inside)
    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    existing_bboxes: list[BBox] = []
    rendered: list[RenderedRegionIcon] = []
    margin_px = int(render_params["region_boundary_margin_px"])
    for index, plan in enumerate(plans):
        placed = False
        nominal_size = int(rng.randint(int(min_size), int(max_size)))
        rotation = rotation_for_named_shape(rng, str(plan.shape_id))
        tint_rgb = tuple(int(value) for value in rng.choice(sampled_palette_rgb))
        fill_style = sample_procedural_named_icon_fill_style(
            rng,
            support=fill_style_support,
            probabilities=fill_style_probabilities,
        )
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{namespace}:named_icon_{int(index)}",
            render_params=render_params,
        )
        for shrink_round in range(8):
            candidate_size = max(28, int(round(float(nominal_size) * (0.92 ** int(shrink_round)))))
            sprite = render_procedural_named_icon_rgba(
                shape_id=str(plan.shape_id),
                size_px=int(candidate_size),
                tint_rgb=tint_rgb,
                fill_style=str(fill_style),
                rotation_degrees=int(rotation),
                mirror_x=False,
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            for _ in range(int(render_params["scene_placement_max_attempts"])):
                center = sample_region_icon_center(
                    rng,
                    content_bbox=content_bbox,
                    sprite_size=tuple(int(value) for value in sprite.size),
                    region=region,
                    desired_inside=bool(plan.desired_inside_region),
                    margin_px=int(margin_px),
                )
                bbox = bbox_from_center_and_size(center, sprite.size)
                if max_overlap_with_existing(bbox, existing_bboxes) > float(render_params["scene_max_overlap_fraction"]):
                    continue
                image.alpha_composite(sprite, (int(bbox[0]), int(bbox[1])))
                inside_region = point_inside_region(region, center)
                counted = bool(plan.is_target_shape and inside_region == counts_inside)
                rendered.append(
                    RenderedRegionIcon(
                        instance_id=f"named_region_icon_{int(index):02d}",
                        shape_id=str(plan.shape_id),
                        shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                        bbox_xyxy=tuple(int(value) for value in bbox),
                        center_xy=(float(center[0]), float(center[1])),
                        nominal_size_px=int(candidate_size),
                        rotation_degrees=int(rotation),
                        tint_rgb=tuple(int(value) for value in tint_rgb),
                        fill_style=str(fill_style),
                        inside_region=bool(inside_region),
                        counted=bool(counted),
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
            raise ValueError("could not place named-region icon with requested membership")

    draw_region_outline(image, region=region, content_bbox=content_bbox, render_params=render_params)
    counted_count = sum(1 for instance in rendered if instance.counted)
    if int(counted_count) != int(target_count):
        raise RuntimeError("rendered region count did not match target answer")

    return ScopedRegionScenePayload(
        image=image.convert("RGB"),
        panel_geometry=single_panel_geometry_to_trace(layout),
        region=region,
        target_shape_id=str(target_shape_id),
        target_shape_name=str(target_shape_name),
        target_count=int(target_count),
        object_count=int(object_count),
        instances=tuple(rendered),
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
        query_probabilities=dict(query_probabilities),
        shape_probabilities=dict(shape_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        object_count_probabilities=dict(object_count_probabilities),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )


__all__ = [
    "build_boolean_scene_specs",
    "build_counterfactual_scene_specs",
    "build_named_icon_specs_from_semantics",
    "build_shape_count_scene_specs",
    "draw_region_outline",
    "draw_region_underlay",
    "render_closer_reference_scene",
    "render_distance_rank_scene",
    "render_scoped_region_scene",
]
