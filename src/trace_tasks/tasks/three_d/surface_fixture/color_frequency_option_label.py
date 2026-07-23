"""Select the text option naming the most frequent or absent fixture color."""

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
from trace_tasks.tasks.three_d.shared.canvas import render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import _resolve_render_params

from .shared.metrics import (
    COLOR_FREQUENCY_MAXIMUM_PROGRAM,
    COLOR_FREQUENCY_OPTION_LABELS,
    COLOR_FREQUENCY_ZERO_PROGRAM,
    build_color_frequency_option_surface_data,
)
from .shared.option_rendering import render_surface_fixture_color_frequency_options
from .shared.prompts import build_prompt_artifacts, dynamic_slots_for_surface
from .shared.sampling import one_hot_probability_map, resolve_scene_and_element
from .shared.state import COLOR_READOUT_SCENE_VARIANTS, SCENE_ID


TASK_ID = "task_three_d__surface_fixture__color_frequency_option_label"
MOST_QUERY_ID = "most_frequent_color"
ABSENT_QUERY_ID = "absent_color"
DEFAULT_QUERY_ID = MOST_QUERY_ID
SUPPORTED_QUERY_IDS = (MOST_QUERY_ID, ABSENT_QUERY_ID)
PROMPT_QUERY_KEY_BY_QUERY_ID = {
    MOST_QUERY_ID: "most_frequent_color_option",
    ABSENT_QUERY_ID: "absent_color_option",
}
FREQUENCY_PROGRAM_BY_QUERY_ID = {
    MOST_QUERY_ID: COLOR_FREQUENCY_MAXIMUM_PROGRAM,
    ABSENT_QUERY_ID: COLOR_FREQUENCY_ZERO_PROGRAM,
}
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
        if label not in set(COLOR_FREQUENCY_OPTION_LABELS):
            raise ValueError(f"unsupported answer_label for {TASK_ID}: {label}")
        return str(label), one_hot_probability_map(COLOR_FREQUENCY_OPTION_LABELS, label)

    label = str(spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label").choice(COLOR_FREQUENCY_OPTION_LABELS))
    return str(label), probability_map(COLOR_FREQUENCY_OPTION_LABELS)


def _target_projection_maps(*, rendered: Any, target_ids: Tuple[str, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    target_bboxes = {
        str(element_id): list(rendered.element_bboxes_px[str(element_id)])
        for element_id in target_ids
    }
    target_centers = {
        str(element_id): list(rendered.element_centers_px[str(element_id)])
        for element_id in target_ids
    }
    return dict(target_bboxes), dict(target_centers)


def _query_params_for_color_frequency(
    *,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    axes: Any,
    dataset: Mapping[str, Any],
    answer_label: str,
    answer_label_probabilities: Mapping[str, float],
    total_count_probabilities: Mapping[int, float],
) -> Dict[str, Any]:
    """Record prompt/query parameters without leaking generation-only state."""

    return {
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "target_element_type": str(axes.element_type),
        "target_element_type_probabilities": dict(axes.element_type_probabilities),
        "option_labels": list(COLOR_FREQUENCY_OPTION_LABELS),
        "option_color_names": list(dataset["option_color_names"]),
        "option_color_counts": dict(dataset["option_color_counts"]),
        "answer_label": str(answer_label),
        "answer_label_probabilities": dict(answer_label_probabilities),
        "answer_color_name": str(dataset["answer_color_name"]),
        "answer_color_count": int(dataset["answer_color_count"]),
        "total_count_probabilities": dict(total_count_probabilities),
        "program": str(selected_query),
    }


def _color_frequency_trace_payload(
    *,
    image: Any,
    rendered: Any,
    render_params: Any,
    axes: Any,
    dataset: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    selected_query: str,
    answer_label: str,
    answer_color_name: str,
    selected_option_bbox: Tuple[float, ...],
    target_ids: Tuple[str, ...],
    target_bboxes: Mapping[str, Any],
    target_centers: Mapping[str, Any],
    annotation_projected: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the full verifier trace for one color-frequency MCQ.

    This wrapper owns public query identity and prompt-facing option labels.
    The invariant is that the selected text option bbox, typed answer, and
    counted/absent color state all come from the same finalized render trace.
    """

    return {
        "scene_ir": {
            "scene_kind": "three_d_surface_fixture_color_frequency_option_label",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "target_element_type": str(axes.element_type),
                "target_element_name": str(dataset["target_element_name"]),
                "target_element_plural": str(dataset["target_element_plural"]),
                "query_id": str(selected_query),
                "option_labels": list(COLOR_FREQUENCY_OPTION_LABELS),
                "option_color_counts": dict(dataset["option_color_counts"]),
                "answer_label": str(answer_label),
                "answer_color_name": str(answer_color_name),
                "answer_color_count": int(dataset["answer_color_count"]),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
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
            "projection_model": "synthetic_perspective_panel_with_color_name_options_v0",
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "fixture_panel_bbox_px": list(rendered.fixture_panel_bbox_px),
            "option_bboxes_px": dict(rendered.option_bboxes_px),
            "option_label_bboxes_px": dict(rendered.option_label_bboxes_px),
            "option_text_bboxes_px": dict(rendered.option_text_bboxes_px),
            "selected_option_bbox_px": list(selected_option_bbox),
            "element_bboxes_px": dict(rendered.element_bboxes_px),
            "element_centers_px": dict(rendered.element_centers_px),
            "target_element_bboxes_px": dict(target_bboxes),
            "target_element_centers_px": dict(target_centers),
        },
        "execution_trace": {
            "query_id": str(selected_query),
            "scene_variant": str(axes.scene_variant),
            "target_element_type": str(axes.element_type),
            "target_element_name": str(dataset["target_element_name"]),
            "target_element_plural": str(dataset["target_element_plural"]),
            "surface_cells": [dict(cell) for cell in dataset["surface_cells"]],
            "layout_rows": int(dataset["layout_rows"]),
            "layout_columns": int(dataset["layout_columns"]),
            "layout_style": str(dataset["layout_style"]),
            "option_records": [dict(record) for record in rendered.option_records],
            "option_color_names": list(dataset["option_color_names"]),
            "option_color_counts": dict(dataset["option_color_counts"]),
            "answer_label": str(answer_label),
            "answer_color_name": str(answer_color_name),
            "answer_color_count": int(dataset["answer_color_count"]),
            "target_element_ids": list(target_ids),
            "target_element_bboxes_px": dict(target_bboxes),
            "target_element_centers_px": dict(target_centers),
            "selected_option_bbox_px": list(selected_option_bbox),
            "question_format": str(selected_query),
            "solver_trace": dict(dataset["solver_trace"]),
        },
        "witness_symbolic": {
            "type": "selected_surface_fixture_color_text_option",
            "option_label": str(answer_label),
            "option_color_name": str(answer_color_name),
            "option_bbox_px": list(selected_option_bbox),
        },
        "projected_annotation": dict(annotation_projected),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


@register_task
class ThreeDSurfaceFixtureColorFrequencyOptionLabelTask:
    """Select the labeled text option naming a color frequency condition."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one color-frequency option instance with one stable answer schema.

        The public task wrapper resolves query id, scene axes, prompt text, and
        annotation. Shared metrics receive only internal frequency-program names
        so reusable scene code is not coupled to public query identities.
        """

        selected_query, query_probabilities, clean_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
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
                dataset, total_count_probabilities = build_color_frequency_option_surface_data(
                    namespace=f"{TASK_ID}.objective",
                    scene_variant=axes.scene_variant,
                    element_type=axes.element_type,
                    frequency_program=str(FREQUENCY_PROGRAM_BY_QUERY_ID[str(selected_query)]),
                    answer_label=str(answer_label),
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    gen_defaults=_GEN_DEFAULTS,
                )
                render_params = _resolve_render_params(
                    clean_params,
                    render_defaults=_RENDER_DEFAULTS,
                    instance_seed=int(attempt_seed),
                    namespace=f"{TASK_ID}.canvas",
                )
                background, background_meta = make_background_canvas(
                    canvas_width=int(render_params.canvas_width),
                    canvas_height=int(render_params.canvas_height),
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    default_config=_BACKGROUND_DEFAULTS,
                )
                rendered = render_surface_fixture_color_frequency_options(
                    background,
                    dataset=dataset,
                    render_params=render_params,
                )
                image, post_noise_meta = apply_post_image_noise(
                    rendered.image,
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    default_config=_NOISE_DEFAULTS,
                )
                selected_option_bbox = list(rendered.option_bboxes_px[str(answer_label)])
                annotation_artifacts = bbox_annotation_artifacts(selected_option_bbox)
                prompt_query_key = str(PROMPT_QUERY_KEY_BY_QUERY_ID[str(selected_query)])
                _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                    prompt_query_key=prompt_query_key,
                    dynamic_slot_values=dynamic_slots_for_surface(
                        dataset,
                        object_description=(
                            f"a fixture surface with colored {dataset['target_element_plural']} "
                            "and six labeled text options naming colors"
                        ),
                    ),
                    instance_seed=int(attempt_seed),
                )
                answer_color_name = str(dataset["answer_color_name"])
                query_spec = build_prompt_query_spec(
                    prompt_artifacts=prompt_artifacts,
                    query_id=str(selected_query),
                    params=_query_params_for_color_frequency(
                        selected_query=str(selected_query),
                        query_probabilities=query_probabilities,
                        axes=axes,
                        dataset=dataset,
                        answer_label=str(answer_label),
                        answer_label_probabilities=answer_label_probabilities,
                        total_count_probabilities=total_count_probabilities,
                    ),
                )
                answer_gt = TypedValue(type="option_letter", value=str(answer_label))
                target_ids = tuple(str(element_id) for element_id in dataset["target_element_ids"])
                target_bboxes, target_centers = _target_projection_maps(
                    rendered=rendered,
                    target_ids=target_ids,
                )
                trace_payload = _color_frequency_trace_payload(
                    image=image,
                    rendered=rendered,
                    render_params=render_params,
                    axes=axes,
                    dataset=dataset,
                    prompt_query_spec=query_spec,
                    selected_query=str(selected_query),
                    answer_label=str(answer_label),
                    answer_color_name=str(answer_color_name),
                    selected_option_bbox=tuple(float(value) for value in selected_option_bbox),
                    target_ids=target_ids,
                    target_bboxes=target_bboxes,
                    target_centers=target_centers,
                    annotation_projected=annotation_artifacts.projected_annotation,
                    background_meta=background_meta,
                    post_noise_meta=post_noise_meta,
                )
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
    "ABSENT_QUERY_ID",
    "DEFAULT_QUERY_ID",
    "MOST_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSurfaceFixtureColorFrequencyOptionLabelTask",
]
