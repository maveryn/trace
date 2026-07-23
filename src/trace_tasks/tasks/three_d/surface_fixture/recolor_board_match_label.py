"""Select the candidate board matching one hypothetical recolor."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_domain_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import probability_map, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.three_d.shared.canvas import expand_canvas_size_to_pixel_cap, render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import _resolve_render_params

from .shared.metrics import build_recolor_board_match_surface_data
from .shared.option_rendering import render_surface_fixture_recolor_board_match
from .shared.prompts import build_prompt_artifacts, dynamic_slots_for_surface
from .shared.sampling import one_hot_probability_map, resolve_scene_and_element
from .shared.state import COLOR_READOUT_SCENE_VARIANTS, SCENE_ID


TASK_ID = "task_three_d__surface_fixture__recolor_board_match_label"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "recolor_board_match"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "three_d",
    SCENE_ID,
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _attempt_seed(instance_seed: int, *, attempt_index: int) -> int:
    if int(attempt_index) == 0:
        return int(instance_seed)
    return int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))


def _resolve_answer_label(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("answer_label")
    if explicit is not None:
        label = str(explicit).strip().upper()
        if label not in set(OPTION_LABELS):
            raise ValueError(f"unsupported answer_label for {TASK_ID}: {label}")
        return str(label), one_hot_probability_map(OPTION_LABELS, label)
    label = str(spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label").choice(OPTION_LABELS))
    return str(label), probability_map(OPTION_LABELS)


@register_task
class ThreeDSurfaceFixtureRecolorBoardMatchLabelTask:
    """Select the candidate fixture board matching one source-to-destination recolor."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'state_update', 'matching')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one original board, four candidate boards, and a selected-option answer."""

        selected_query, query_probabilities, clean_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        axes = resolve_scene_and_element(
            params=clean_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=TASK_ID,
            supported_scenes=COLOR_READOUT_SCENE_VARIANTS,
        )
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = _attempt_seed(int(instance_seed), attempt_index=int(attempt_index))
            try:
                answer_label, answer_label_probabilities = _resolve_answer_label(
                    clean_params,
                    instance_seed=int(attempt_seed),
                )
                board_data = build_recolor_board_match_surface_data(
                    namespace=f"{TASK_ID}.objective",
                    scene_variant=axes.scene_variant,
                    element_type=axes.element_type,
                    answer_label=str(answer_label),
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    gen_defaults=_GEN_DEFAULTS,
                )
                original_dataset = dict(board_data["original_dataset"])
                option_datasets = {
                    str(label): dict(dataset)
                    for label, dataset in dict(board_data["option_datasets"]).items()
                }
                render_params = _resolve_render_params(
                    clean_params,
                    render_defaults=_RENDER_DEFAULTS,
                    instance_seed=int(attempt_seed),
                    namespace=f"{TASK_ID}.canvas",
                )
                composite_width, composite_height = expand_canvas_size_to_pixel_cap(
                    int(render_params.canvas_width),
                    int(render_params.canvas_height),
                )
                background, background_meta = make_background_canvas(
                    canvas_width=int(composite_width),
                    canvas_height=int(composite_height),
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    default_config=_BACKGROUND_DEFAULTS,
                )
                rendered = render_surface_fixture_recolor_board_match(
                    background,
                    original_dataset=original_dataset,
                    option_datasets=option_datasets,
                    render_params=render_params,
                )
                image, post_noise_meta = apply_post_image_noise(
                    rendered.image,
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    default_config=_NOISE_DEFAULTS,
                )
                selected_panel_bbox = list(rendered.option_panel_bboxes_px[str(answer_label)])
                annotation_artifacts = bbox_annotation_artifacts(selected_panel_bbox)
                prompt_dataset = dict(original_dataset)
                prompt_dataset.update(
                    {
                        "recolor_phrase": str(board_data["recolor_phrase"]),
                        "source_color_name": str(board_data["source_color_name"]),
                        "destination_color_name": str(board_data["destination_color_name"]),
                    }
                )
                _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                    prompt_query_key=PROMPT_QUERY_KEY,
                    dynamic_slot_values=dynamic_slots_for_surface(
                        prompt_dataset,
                        object_description=(
                            f"one original fixture board and four labeled candidate boards with "
                            f"visible {original_dataset['target_element_plural']}"
                        ),
                    ),
                    instance_seed=int(attempt_seed),
                )
                query_spec = build_prompt_query_spec(
                    prompt_artifacts=prompt_artifacts,
                    query_id=str(selected_query),
                    params={
                        "query_id_probabilities": dict(query_probabilities),
                        "scene_variant": str(axes.scene_variant),
                        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                        "target_element_type": str(axes.element_type),
                        "target_element_type_probabilities": dict(axes.element_type_probabilities),
                        "option_labels": list(OPTION_LABELS),
                        "answer_label": str(answer_label),
                        "answer_label_probabilities": dict(answer_label_probabilities),
                        "source_color_name": str(board_data["source_color_name"]),
                        "destination_color_name": str(board_data["destination_color_name"]),
                        "initial_color_counts": dict(board_data["initial_color_counts"]),
                        "final_color_counts": dict(board_data["final_color_counts"]),
                        "option_color_counts_by_label": {
                            str(label): dict(counts)
                            for label, counts in dict(board_data["option_color_counts_by_label"]).items()
                        },
                        "original_color_by_flat_index": dict(board_data["original_color_by_flat_index"]),
                        "final_color_by_flat_index": dict(board_data["final_color_by_flat_index"]),
                        "option_color_by_flat_index_by_label": {
                            str(label): dict(color_map)
                            for label, color_map in dict(board_data["option_color_by_flat_index_by_label"]).items()
                        },
                    },
                )
                answer_gt = TypedValue(type="option_letter", value=str(answer_label))
                trace_payload = {
                    "scene_ir": {
                        "scene_kind": "three_d_surface_fixture_recolor_board_match_label",
                        "entities": [dict(entity) for entity in rendered.entities],
                        "relations": {
                            "scene_variant": str(axes.scene_variant),
                            "target_element_type": str(axes.element_type),
                            "target_element_name": str(original_dataset["target_element_name"]),
                            "target_element_plural": str(original_dataset["target_element_plural"]),
                            "option_labels": list(OPTION_LABELS),
                            "answer_label": str(answer_label),
                            "source_color_name": str(board_data["source_color_name"]),
                            "destination_color_name": str(board_data["destination_color_name"]),
                            "initial_color_counts": dict(board_data["initial_color_counts"]),
                            "final_color_counts": dict(board_data["final_color_counts"]),
                            "option_color_counts_by_label": dict(rendered.option_color_counts_by_label),
                            "original_color_by_flat_index": dict(board_data["original_color_by_flat_index"]),
                            "final_color_by_flat_index": dict(board_data["final_color_by_flat_index"]),
                            "option_color_by_flat_index_by_label": {
                                str(label): dict(color_map)
                                for label, color_map in dict(board_data["option_color_by_flat_index_by_label"]).items()
                            },
                        },
                    },
                    "query_spec": dict(query_spec),
                    "render_spec": {
                        "canvas_width": int(image.width),
                        "canvas_height": int(image.height),
                        "scene_canvas_preset": str(render_params.canvas_preset),
                        "scene_canvas_width": int(render_params.canvas_width),
                        "scene_canvas_height": int(render_params.canvas_height),
                        "scene_canvas_policy": str(render_params.canvas_policy),
                        **render_params_canvas_metadata(render_params),
                        "final_canvas_width": int(image.width),
                        "final_canvas_height": int(image.height),
                        "final_canvas_pixels": int(image.width) * int(image.height),
                        "coord_space": "pixel",
                        "scene_variant": str(axes.scene_variant),
                        "background_style": dict(background_meta),
                        "post_image_noise": dict(post_noise_meta),
                        "projection_model": "synthetic_perspective_panel_recolor_board_match_v0",
                        "option_grid_shape": [2, 2],
                    },
                    "render_map": {
                        "image_id": "img0",
                        "scene_bbox_px": list(rendered.scene_bbox_px),
                        "original_panel_bbox_px": list(rendered.original_panel_bbox_px),
                        "original_label_bbox_px": list(rendered.original_label_bbox_px),
                        "option_panel_bboxes_px": dict(rendered.option_panel_bboxes_px),
                        "option_label_bboxes_px": dict(rendered.option_label_bboxes_px),
                        "selected_option_panel_bbox_px": list(selected_panel_bbox),
                        "element_bboxes_px": dict(rendered.element_bboxes_px),
                        "element_centers_px": dict(rendered.element_centers_px),
                    },
                    "execution_trace": {
                        "query_id": str(selected_query),
                        "scene_variant": str(axes.scene_variant),
                        "target_element_type": str(axes.element_type),
                        "target_element_name": str(original_dataset["target_element_name"]),
                        "target_element_plural": str(original_dataset["target_element_plural"]),
                        "option_labels": list(OPTION_LABELS),
                        "option_records": [dict(record) for record in rendered.option_records],
                        "answer_label": str(answer_label),
                        "source_color_name": str(board_data["source_color_name"]),
                        "destination_color_name": str(board_data["destination_color_name"]),
                        "recolor_phrase": str(board_data["recolor_phrase"]),
                        "active_color_names": list(board_data["active_color_names"]),
                        "initial_color_counts": dict(board_data["initial_color_counts"]),
                        "final_color_counts": dict(board_data["final_color_counts"]),
                        "option_color_counts_by_label": dict(rendered.option_color_counts_by_label),
                        "original_color_by_flat_index": dict(board_data["original_color_by_flat_index"]),
                        "final_color_by_flat_index": dict(board_data["final_color_by_flat_index"]),
                        "option_color_by_flat_index_by_label": {
                            str(label): dict(color_map)
                            for label, color_map in dict(board_data["option_color_by_flat_index_by_label"]).items()
                        },
                        "selected_option_panel_bbox_px": list(selected_panel_bbox),
                        "surface_original_dataset": {
                            "layout_rows": int(original_dataset["layout_rows"]),
                            "layout_columns": int(original_dataset["layout_columns"]),
                            "layout_style": str(original_dataset["layout_style"]),
                            "surface_cells": [dict(cell) for cell in original_dataset["surface_cells"]],
                        },
                        "surface_option_datasets": {
                            str(label): {
                                "layout_rows": int(dataset["layout_rows"]),
                                "layout_columns": int(dataset["layout_columns"]),
                                "layout_style": str(dataset["layout_style"]),
                                "surface_cells": [dict(cell) for cell in dataset["surface_cells"]],
                                "color_counts": dict(dataset["color_counts"]),
                            }
                            for label, dataset in option_datasets.items()
                        },
                        "question_format": str(selected_query),
                        "solver_trace": {
                            "operation": "select_option_matching_single_recolor_fixed_position_color_state",
                            "source_color_name": str(board_data["source_color_name"]),
                            "destination_color_name": str(board_data["destination_color_name"]),
                            "initial_color_counts": dict(board_data["initial_color_counts"]),
                            "final_color_counts": dict(board_data["final_color_counts"]),
                            "option_color_counts_by_label": dict(rendered.option_color_counts_by_label),
                            "original_color_by_flat_index": dict(board_data["original_color_by_flat_index"]),
                            "final_color_by_flat_index": dict(board_data["final_color_by_flat_index"]),
                            "option_color_by_flat_index_by_label": {
                                str(label): dict(color_map)
                                for label, color_map in dict(board_data["option_color_by_flat_index_by_label"]).items()
                            },
                            "answer_label": str(answer_label),
                            "unique_option_fixed_position_color_states": True,
                            "spatial_positions_fixed": True,
                        },
                    },
                    "witness_symbolic": {
                        "type": "selected_surface_fixture_recolor_option_panel",
                        "option_label": str(answer_label),
                        "option_panel_bbox_px": list(selected_panel_bbox),
                    },
                    "projected_annotation": dict(annotation_artifacts.projected_annotation),
                    "background": dict(background_meta),
                    "post_image_noise": dict(post_noise_meta),
                }
                return TaskOutput(
                    prompt=str(prompt_artifacts.prompt),
                    prompt_variants=dict(prompt_artifacts.prompt_variants),
                    answer_gt=answer_gt,
                    annotation_gt=annotation_artifacts.annotation_gt,
                    image=image,
                    image_id="img0",
                    trace_payload=trace_payload,
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=str(selected_query),
                )
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"{TASK_ID} failed to generate after {max_attempts} attempts: {last_error}")


__all__ = [
    "QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSurfaceFixtureRecolorBoardMatchLabelTask",
]
