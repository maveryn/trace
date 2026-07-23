"""Count scene-grid cells whose front-to-back icon order matches the Reference."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)
from ...shared.counting_sampling import resolve_counting_target_and_distractor_triplet
from ...shared.labeling import LABEL_POOL_A_L
from ...shared.output_metadata import default_task_versions
from ...shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ..shared.icon_assets import resolve_icon_pool
from ..shared.icon_scene import IconInstanceSpec, panel_geometry_to_trace
from ..shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param, sample_icon_instance_noise
from ..shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette
from .shared.annotations import matching_overlap_cell_bbox_set_annotation
from .shared.defaults import DOMAIN, FIXED_RELATION_ID, OverlapGridDefaults, SCENE_ID
from .shared.output import overlap_grid_style_trace
from .shared.prompts import render_overlap_grid_prompt_artifacts
from .shared.rendering import IconOverlapPairSpec, render_two_panel_icon_overlap_grid_scene
from .shared.sampling import order_id_for_front_role, sample_overlap_offsets, sample_tint_pair
from .shared.state import OverlapGridScenePayload


TASK_ID = "task_icons__overlap_grid__occlusion_order_count"

_DEFAULTS = OverlapGridDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> Dict[str, Any]:
    """Resolve render params for the occlusion-order grid task."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=_RENDER_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    render_params["cell_padding_px"] = int(
        params.get("cell_padding_px", group_default(_RENDER_DEFAULTS, "cell_padding_px", _DEFAULTS.cell_padding_px))
    )
    render_params["cell_border_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=_RENDER_DEFAULTS,
        key="cell_border_rgb",
        fallback=_DEFAULTS.cell_border_rgb,
        instance_seed=int(instance_seed),
    )
    render_params["cell_label_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=_RENDER_DEFAULTS,
        key="cell_label_color_rgb",
        fallback=_DEFAULTS.cell_label_color_rgb,
        instance_seed=int(instance_seed),
    )
    render_params["cell_label_font_size_px"] = int(
        params.get(
            "cell_label_font_size_px",
            group_default(_RENDER_DEFAULTS, "cell_label_font_size_px", _DEFAULTS.cell_label_font_size_px),
        )
    )
    cell_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace="icons.overlap_grid.cell_label_text",
        role="icon_cell_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["cell_label_color_rgb"]),),
        required=False,
    )
    render_params["cell_label_color_rgb"] = tuple(int(value) for value in cell_label_style.fill_rgb)
    render_params["cell_label_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    cell_label_record = cell_label_style.metadata()
    cell_label_record["stroke_rgb"] = list(render_params["cell_label_stroke_rgb"])
    previous_legibility = render_params.get("text_legibility")
    previous_records = []
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*previous_records, cell_label_record]
    )
    render_params["pair_min_color_distance"] = float(
        params.get(
            "pair_min_color_distance",
            group_default(_RENDER_DEFAULTS, "pair_min_color_distance", _DEFAULTS.pair_min_color_distance),
        )
    )
    raw_overlap_range = params.get(
        "overlap_ratio_range",
        group_default(_RENDER_DEFAULTS, "overlap_ratio_range", list(_DEFAULTS.overlap_ratio_range)),
    )
    if not isinstance(raw_overlap_range, (list, tuple)) or len(raw_overlap_range) < 2:
        raise ValueError("overlap_ratio_range must contain two numeric bounds")
    overlap_min = max(0.0, min(0.95, float(raw_overlap_range[0])))
    overlap_max = max(overlap_min, min(0.95, float(raw_overlap_range[1])))
    render_params["overlap_ratio_range"] = (float(overlap_min), float(overlap_max))
    return render_params


def _sample_scene(
    rng,
    *,
    instance_seed: int,
    object_count: int,
    target_count: int,
    pool_manifest: str,
    render_params: Mapping[str, Any],
) -> Tuple[OverlapGridScenePayload, Any]:
    """Sample and render one occlusion-order counting scene."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < 2:
        raise ValueError("icon pool is too small for occlusion-order scene")
    icon_a_id, icon_b_id = rng.sample(pool, 2)
    reference_front_role = str(rng.choice(("a", "b")))
    reference_order_id = order_id_for_front_role(reference_front_role)
    labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(object_count)])
    match_indices = set(rng.sample(list(range(int(object_count))), int(target_count)))

    palette_size = int(
        rng.randint(
            int(render_params["palette_size_min"]),
            int(render_params["palette_size_max"]),
        )
    )
    sampled_palette_rgb = tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
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
    )
    if not icon_palette_meets_distance_constraints(
        palette=sampled_palette_rgb,
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
        ):
        raise ValueError("sampled occlusion palette did not satisfy strict distance constraints")

    reference_a_tint, reference_b_tint = sample_tint_pair(
        rng,
        palette=sampled_palette_rgb,
        pair_min_color_distance=float(render_params["pair_min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    reference_dx_frac, reference_dy_frac, reference_overlap_ratio = sample_overlap_offsets(
        rng,
        overlap_ratio_range=render_params["overlap_ratio_range"],
    )
    reference_a_noise_edits, reference_a_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:reference_a",
        render_params=render_params,
    )
    reference_b_noise_edits, reference_b_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:reference_b",
        render_params=render_params,
    )
    reference_pair = IconOverlapPairSpec(
        icon_a=IconInstanceSpec(
            icon_id=str(icon_a_id),
            tint_rgb=tuple(int(v) for v in reference_a_tint),
            noise_edits=tuple(reference_a_noise_edits),
            noise_seed=int(reference_a_noise_seed),
        ),
        icon_b=IconInstanceSpec(
            icon_id=str(icon_b_id),
            tint_rgb=tuple(int(v) for v in reference_b_tint),
            noise_edits=tuple(reference_b_noise_edits),
            noise_seed=int(reference_b_noise_seed),
        ),
        front_role=str(reference_front_role),
        offset_dx_frac=float(reference_dx_frac),
        offset_dy_frac=float(reference_dy_frac),
        overlap_ratio=float(reference_overlap_ratio),
    )

    scene_pairs: List[IconOverlapPairSpec] = []
    cell_order_ids: List[str] = []
    matching_labels: List[str] = []
    for index, label in enumerate(labels):
        front_role = str(reference_front_role if int(index) in match_indices else ("b" if str(reference_front_role) == "a" else "a"))
        if int(index) in match_indices:
            matching_labels.append(str(label))
        icon_a_tint, icon_b_tint = sample_tint_pair(
            rng,
            palette=sampled_palette_rgb,
            pair_min_color_distance=float(render_params["pair_min_color_distance"]),
            distance_space=str(render_params["color_distance_space"]),
        )
        dx_frac, dy_frac, overlap_ratio = sample_overlap_offsets(
            rng,
            overlap_ratio_range=render_params["overlap_ratio_range"],
        )
        icon_a_noise_edits, icon_a_noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:scene_{int(index)}_a",
            render_params=render_params,
        )
        icon_b_noise_edits, icon_b_noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:scene_{int(index)}_b",
            render_params=render_params,
        )
        scene_pairs.append(
            IconOverlapPairSpec(
                icon_a=IconInstanceSpec(
                    icon_id=str(icon_a_id),
                    tint_rgb=tuple(int(v) for v in icon_a_tint),
                    noise_edits=tuple(icon_a_noise_edits),
                    noise_seed=int(icon_a_noise_seed),
                ),
                icon_b=IconInstanceSpec(
                    icon_id=str(icon_b_id),
                    tint_rgb=tuple(int(v) for v in icon_b_tint),
                    noise_edits=tuple(icon_b_noise_edits),
                    noise_seed=int(icon_b_noise_seed),
                ),
                front_role=str(front_role),
                offset_dx_frac=float(dx_frac),
                offset_dy_frac=float(dy_frac),
                overlap_ratio=float(overlap_ratio),
            )
        )
        cell_order_ids.append(order_id_for_front_role(front_role))

    rendered = render_two_panel_icon_overlap_grid_scene(
        reference_pair=reference_pair,
        scene_pairs=scene_pairs,
        scene_labels=labels,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        cell_padding_px=int(render_params["cell_padding_px"]),
        scene_icon_size_min_px=int(render_params["scene_icon_size_min_px"]),
        scene_icon_size_max_px=int(render_params["scene_icon_size_max_px"]),
        reference_icon_size_px=int(render_params["reference_icon_size_px"]),
        cell_label_font_size_px=int(render_params["cell_label_font_size_px"]),
        panel_title_font_size_px=int(render_params["panel_title_font_size_px"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        cell_border_rgb=tuple(int(v) for v in render_params["cell_border_rgb"]),
        cell_label_color_rgb=tuple(int(v) for v in render_params["cell_label_color_rgb"]),
        cell_label_stroke_rgb=tuple(int(v) for v in render_params["cell_label_stroke_rgb"]),
        cell_label_stroke_width_px=1,
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )

    reference_payload = {
        "panel": "reference",
        "icon_a_id": str(rendered.reference_pair.icon_a_id),
        "icon_b_id": str(rendered.reference_pair.icon_b_id),
        "front_role": str(rendered.reference_pair.front_role),
        "order_id": str(reference_order_id),
        "overlap_ratio": float(rendered.reference_pair.overlap_ratio),
        "icon_a_bbox_xyxy": list(rendered.reference_pair.icon_a_bbox_xyxy),
        "icon_b_bbox_xyxy": list(rendered.reference_pair.icon_b_bbox_xyxy),
        "icon_a_tint_rgb": list(rendered.reference_pair.icon_a_tint_rgb),
        "icon_b_tint_rgb": list(rendered.reference_pair.icon_b_tint_rgb),
        "icon_a_noise_edits": [dict(edit) for edit in rendered.reference_pair.icon_a_noise_edits],
        "icon_a_noise_seed": None
        if rendered.reference_pair.icon_a_noise_seed is None
        else int(rendered.reference_pair.icon_a_noise_seed),
        "icon_b_noise_edits": [dict(edit) for edit in rendered.reference_pair.icon_b_noise_edits],
        "icon_b_noise_seed": None
        if rendered.reference_pair.icon_b_noise_seed is None
        else int(rendered.reference_pair.icon_b_noise_seed),
    }
    scene_cells = tuple(
        {
            "panel": "scene",
            "label": str(cell.label),
            "icon_a_id": str(cell.icon_a_id),
            "icon_b_id": str(cell.icon_b_id),
            "front_role": str(cell.front_role),
            "order_id": str(order_id_for_front_role(str(cell.front_role))),
            "overlap_ratio": float(cell.overlap_ratio),
            "cell_bbox_xyxy": list(cell.cell_bbox_xyxy),
            "icon_a_bbox_xyxy": list(cell.icon_a_bbox_xyxy),
            "icon_b_bbox_xyxy": list(cell.icon_b_bbox_xyxy),
            "icon_a_tint_rgb": list(cell.icon_a_tint_rgb),
            "icon_b_tint_rgb": list(cell.icon_b_tint_rgb),
            "icon_a_noise_edits": [dict(edit) for edit in cell.icon_a_noise_edits],
            "icon_a_noise_seed": None if cell.icon_a_noise_seed is None else int(cell.icon_a_noise_seed),
            "icon_b_noise_edits": [dict(edit) for edit in cell.icon_b_noise_edits],
            "icon_b_noise_seed": None if cell.icon_b_noise_seed is None else int(cell.icon_b_noise_seed),
            "is_match": bool(str(cell.label) in set(matching_labels)),
            "index": int(index),
        }
        for index, cell in enumerate(rendered.scene_cells)
    )
    return OverlapGridScenePayload(
        object_count=int(object_count),
        target_count=int(target_count),
        distractor_count=int(object_count) - int(target_count),
        reference_order_id=str(reference_order_id),
        icon_a_id=str(icon_a_id),
        icon_b_id=str(icon_b_id),
        cell_labels=tuple(str(value) for value in labels),
        matching_labels=tuple(sorted(str(value) for value in matching_labels)),
        cell_order_ids=tuple(str(value) for value in cell_order_ids),
        sampled_palette_rgb=tuple(sampled_palette_rgb),
        panel_geometry=panel_geometry_to_trace(rendered.layout),
        reference_pair=reference_payload,
        scene_cells=scene_cells,
    ), rendered.image


@register_task
class IconsOverlapGridOcclusionOrderCountTask:
    """Count labeled scene cells that match the Reference front-to-back icon order."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'spatial_relations')
    domain = DOMAIN
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic icon occlusion-order instance."""

        scene_rng = spawn_rng(int(instance_seed), "scene")
        sampling_params: Dict[str, Any] = dict(params)
        (
            object_count,
            object_count_probabilities,
            target_count,
            target_count_probabilities,
            distractor_count,
            distractor_count_probabilities,
        ) = resolve_counting_target_and_distractor_triplet(
            scene_rng,
            instance_seed=int(instance_seed),
            params=sampling_params,
            gen_defaults=_GEN_DEFAULTS,
            fallback_total_min=_DEFAULTS.object_count_min,
            fallback_total_max=_DEFAULTS.object_count_max,
            fallback_target_min=_DEFAULTS.target_count_min,
            fallback_target_max=_DEFAULTS.target_count_max,
            fallback_distractor_min=_DEFAULTS.distractor_count_min,
            fallback_distractor_max=_DEFAULTS.distractor_count_max,
        )
        render_params = _resolve_render_params(params, instance_seed=int(instance_seed))
        pool_manifest = str(params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest)))

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = _sample_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    object_count=int(object_count),
                    target_count=int(target_count),
                    pool_manifest=str(pool_manifest),
                    render_params=render_params,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError("failed to generate task_icons__overlap_grid__occlusion_order_count instance") from last_error

        prompt_defaults, prompt_artifacts = render_overlap_grid_prompt_artifacts(
            instance_seed=int(instance_seed),
            prompt_defaults=_PROMPT_DEFAULTS,
        )

        annotation_labels = list(scene_payload.matching_labels)
        annotation_artifacts = matching_overlap_cell_bbox_set_annotation(
            scene_cells=scene_payload.scene_cells,
            matching_labels=annotation_labels,
        )
        answer_gt = TypedValue(type="integer", value=int(scene_payload.target_count))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_reference_grid_occlusion_order_count",
                "entities": [dict(scene_payload.reference_pair), *[dict(item) for item in scene_payload.scene_cells]],
                "relations": {
                    "counting_target": "same_front_to_back_order_as_reference",
                    "reference_order_id": str(scene_payload.reference_order_id),
                    "matching_cell_labels": list(scene_payload.matching_labels),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": {
                "query_id": SINGLE_QUERY_ID,
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "fixed_relation_id": FIXED_RELATION_ID,
                    "object_count": int(object_count),
                    "object_count_probabilities": dict(object_count_probabilities),
                    "target_count": int(target_count),
                    "target_count_probabilities": dict(target_count_probabilities),
                    "distractor_count": int(distractor_count),
                    "distractor_count_probabilities": dict(distractor_count_probabilities),
                    "pool_manifest": str(pool_manifest),
                },
            },
            "render_spec": {
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": overlap_grid_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_pair": dict(scene_payload.reference_pair),
                    "matching_cell_labels": list(scene_payload.matching_labels),
                    "scene_cells": [dict(item) for item in scene_payload.scene_cells],
                },
            },
            "execution_trace": {
                "scene_variant": "reference_overlap_grid",
                "query_id": SINGLE_QUERY_ID,
                "fixed_relation_id": FIXED_RELATION_ID,
                "object_count": int(scene_payload.object_count),
                "object_count_probabilities": dict(object_count_probabilities),
                "target_count": int(scene_payload.target_count),
                "target_count_probabilities": dict(target_count_probabilities),
                "distractor_count": int(scene_payload.distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "reference_order_id": str(scene_payload.reference_order_id),
                "icon_a_id": str(scene_payload.icon_a_id),
                "icon_b_id": str(scene_payload.icon_b_id),
                "cell_labels": list(scene_payload.cell_labels),
                "matching_cell_labels": list(scene_payload.matching_labels),
                "cell_order_ids": list(scene_payload.cell_order_ids),
                "question_format": "count_scene_cells_matching_reference_occlusion_order",
            },
            "witness_symbolic": {
                "reference_order_id": str(scene_payload.reference_order_id),
                **dict(annotation_artifacts["witness_symbolic"]),
            },
            "projected_annotation": dict(annotation_artifacts["projected_annotation"]),
        }
        output = TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=SINGLE_QUERY_ID,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )
        return output


__all__ = ["IconsOverlapGridOcclusionOrderCountTask"]
