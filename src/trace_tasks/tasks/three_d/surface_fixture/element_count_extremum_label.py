"""Select the surface-fixture option panel with the highest or lowest count."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

from .shared.metrics import build_repeated_surface_data
from .shared.option_rendering import render_surface_fixture_option_grid
from .shared.prompts import build_prompt_artifacts, dynamic_slots_for_surface
from .shared.sampling import configured_int, one_hot_probability_map, resolve_scene_and_element, uniform_int_probability_map
from .shared.state import SCENE_ID, SUPPORTED_SCENE_VARIANTS


TASK_ID = "task_three_d__surface_fixture__element_count_extremum_label"
HIGHEST_QUERY_ID = "highest_element_count"
LOWEST_QUERY_ID = "lowest_element_count"
DEFAULT_QUERY_ID = HIGHEST_QUERY_ID
SUPPORTED_QUERY_IDS = (HIGHEST_QUERY_ID, LOWEST_QUERY_ID)
PROMPT_QUERY_KEY_BY_QUERY_ID = {
    HIGHEST_QUERY_ID: "highest_element_count_panel",
    LOWEST_QUERY_ID: "lowest_element_count_panel",
}
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


def _resolve_count_support(params: Mapping[str, Any]) -> Tuple[int, int, Dict[str, float]]:
    minimum = max(3, configured_int(params, _GEN_DEFAULTS, "option_count_min", 4))
    maximum = max(minimum + 3, configured_int(params, _GEN_DEFAULTS, "option_count_max", 16))
    maximum = min(32, int(maximum))
    minimum = min(int(minimum), int(maximum) - 3)
    support = tuple(range(int(minimum), int(maximum) + 1))
    return int(minimum), int(maximum), uniform_int_probability_map(support)


def _resolve_answer_label(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("answer_label")
    if explicit is not None:
        label = str(explicit).strip().upper()
        if label not in set(OPTION_LABELS):
            raise ValueError(f"unsupported answer_label for {TASK_ID}: {label}")
        return str(label), one_hot_probability_map(OPTION_LABELS, label)

    label = str(spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label").choice(OPTION_LABELS))
    return str(label), probability_map(OPTION_LABELS)


def _explicit_option_counts(params: Mapping[str, Any]) -> List[int] | None:
    raw = params.get("option_counts")
    if raw is None:
        raw = params.get("counts_by_label")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return [int(raw[str(label)]) for label in OPTION_LABELS]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        counts = [int(value) for value in raw]
        if len(counts) != len(OPTION_LABELS):
            raise ValueError(f"option_counts must contain {len(OPTION_LABELS)} values")
        return list(counts)
    raise ValueError("option_counts must be a sequence or mapping")


def _sample_option_counts(
    *,
    query_id: str,
    answer_label: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Build unique panel counts while pinning the requested extremum to the answer label."""

    minimum, maximum, count_probabilities = _resolve_count_support(params)
    support = tuple(range(int(minimum), int(maximum) + 1))
    margin = max(1, int(params.get("extremum_count_margin_min", 2)))
    explicit_counts = _explicit_option_counts(params)
    if explicit_counts is not None:
        counts_by_label = {label: int(explicit_counts[index]) for index, label in enumerate(OPTION_LABELS)}
        if len(set(counts_by_label.values())) != len(OPTION_LABELS):
            raise ValueError("option_counts must be unique")
        if any(int(count) not in set(support) for count in counts_by_label.values()):
            raise ValueError(f"option_counts must lie in support {minimum}..{maximum}")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.option_counts")
        counts_by_label = {}
        for _attempt in range(200):
            sampled = list(support)
            rng.shuffle(sampled)
            selected_counts = sorted(sampled[: len(OPTION_LABELS)])
            if int(selected_counts[-1] - selected_counts[-2]) < margin:
                continue
            if int(selected_counts[1] - selected_counts[0]) < margin:
                continue
            if str(query_id) == HIGHEST_QUERY_ID:
                extremum_count = int(selected_counts[-1])
                other_counts = list(selected_counts[:-1])
            else:
                extremum_count = int(selected_counts[0])
                other_counts = list(selected_counts[1:])
            rng.shuffle(other_counts)
            counts_by_label = {str(answer_label): int(extremum_count)}
            other_labels = [label for label in OPTION_LABELS if label != str(answer_label)]
            for label, count in zip(other_labels, other_counts):
                counts_by_label[str(label)] = int(count)
            break
        if not counts_by_label:
            raise ValueError("could not sample unique option counts with requested extremum margin")

    sorted_counts = sorted(counts_by_label.items(), key=lambda item: (int(item[1]), str(item[0])))
    expected_label = str(sorted_counts[-1][0] if str(query_id) == HIGHEST_QUERY_ID else sorted_counts[0][0])
    if expected_label != str(answer_label):
        raise ValueError(f"answer_label={answer_label} does not match {query_id} option counts")
    return dict(counts_by_label), dict(count_probabilities)


def _build_option_datasets(
    *,
    scene_variant: str,
    element_type: str,
    counts_by_label: Mapping[str, int],
    instance_seed: int,
    params: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    datasets: Dict[str, Dict[str, Any]] = {}
    minimum, maximum, _count_probabilities = _resolve_count_support(params)
    for label in OPTION_LABELS:
        option_params = dict(params)
        option_params["target_count"] = int(counts_by_label[str(label)])
        option_params["target_count_min"] = int(minimum)
        option_params["target_count_max"] = int(maximum)
        dataset, _answer_probabilities = build_repeated_surface_data(
            namespace=f"{TASK_ID}.option_{label}",
            scene_variant=str(scene_variant),
            element_type=str(element_type),
            instance_seed=int(spawn_rng(int(instance_seed), f"{TASK_ID}.option_seed.{label}").randrange(1, 2**62)),
            params=option_params,
            gen_defaults=_GEN_DEFAULTS,
        )
        dataset = dict(dataset)
        dataset["option_label"] = str(label)
        datasets[str(label)] = dataset
    return datasets


@register_task
class ThreeDSurfaceFixtureElementCountExtremumLabelTask:
    """Select the labeled surface-fixture option panel with the highest or lowest count."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one visual MCQ option grid and bind the selected panel as annotation."""

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
            supported_scenes=SUPPORTED_SCENE_VARIANTS,
        )

        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = _attempt_seed(int(instance_seed), attempt_index=int(attempt_index))
            try:
                answer_label, answer_label_probabilities = _resolve_answer_label(
                    clean_params,
                    instance_seed=int(attempt_seed),
                )
                counts_by_label, count_probabilities = _sample_option_counts(
                    query_id=str(selected_query),
                    answer_label=str(answer_label),
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                )
                option_datasets = _build_option_datasets(
                    scene_variant=axes.scene_variant,
                    element_type=axes.element_type,
                    counts_by_label=counts_by_label,
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                )
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
                rendered = render_surface_fixture_option_grid(
                    background,
                    option_datasets=option_datasets,
                    render_params=render_params,
                )
                image, post_noise_meta = apply_post_image_noise(
                    rendered.image,
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    default_config=_NOISE_DEFAULTS,
                )
                answer_count = int(rendered.option_counts_by_label[str(answer_label)])
                selected_panel_bbox = list(rendered.option_panel_bboxes_px[str(answer_label)])
                annotation_artifacts = bbox_annotation_artifacts(selected_panel_bbox)
                first_dataset = dict(option_datasets[OPTION_LABELS[0]])
                prompt_query_key = str(PROMPT_QUERY_KEY_BY_QUERY_ID[str(selected_query)])
                _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                    prompt_query_key=prompt_query_key,
                    dynamic_slot_values=dynamic_slots_for_surface(
                        first_dataset,
                        object_description=f"four labeled fixture panels with visible {first_dataset['target_element_plural']}",
                    ),
                    instance_seed=int(attempt_seed),
                )
                extremum_kind = "highest" if str(selected_query) == HIGHEST_QUERY_ID else "lowest"
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
                        "option_counts_by_label": dict(rendered.option_counts_by_label),
                        "answer_label": str(answer_label),
                        "answer_label_probabilities": dict(answer_label_probabilities),
                        "answer_count": int(answer_count),
                        "option_count_probabilities": dict(count_probabilities),
                        "extremum_kind": str(extremum_kind),
                    },
                )
                answer_gt = TypedValue(type="option_letter", value=str(answer_label))
                trace_payload = {
                    "scene_ir": {
                        "scene_kind": "three_d_surface_fixture_element_count_extremum_label",
                        "entities": [dict(entity) for entity in rendered.entities],
                        "relations": {
                            "scene_variant": str(axes.scene_variant),
                            "target_element_type": str(axes.element_type),
                            "target_element_name": str(first_dataset["target_element_name"]),
                            "target_element_plural": str(first_dataset["target_element_plural"]),
                            "option_labels": list(OPTION_LABELS),
                            "option_counts_by_label": dict(rendered.option_counts_by_label),
                            "answer_label": str(answer_label),
                            "answer_count": int(answer_count),
                            "extremum_kind": str(extremum_kind),
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
                        "projection_model": "synthetic_perspective_panel_option_grid_v0",
                        "option_grid_shape": [2, 2],
                    },
                    "render_map": {
                        "image_id": "img0",
                        "scene_bbox_px": list(rendered.scene_bbox_px),
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
                        "target_element_name": str(first_dataset["target_element_name"]),
                        "target_element_plural": str(first_dataset["target_element_plural"]),
                        "option_labels": list(OPTION_LABELS),
                        "option_records": [dict(record) for record in rendered.option_records],
                        "option_counts_by_label": dict(rendered.option_counts_by_label),
                        "answer_label": str(answer_label),
                        "answer_count": int(answer_count),
                        "extremum_kind": str(extremum_kind),
                        "selected_option_panel_bbox_px": list(selected_panel_bbox),
                        "surface_option_datasets": {
                            str(label): {
                                "layout_rows": int(dataset["layout_rows"]),
                                "layout_columns": int(dataset["layout_columns"]),
                                "layout_style": str(dataset["layout_style"]),
                                "visual_color_names": list(dataset.get("visual_color_names", [])),
                                "visual_color_counts": dict(dataset.get("visual_color_counts", {})),
                                "color_role": str(dataset.get("color_role", "non_semantic_visual_variation")),
                                "surface_cells": [dict(cell) for cell in dataset["surface_cells"]],
                            }
                            for label, dataset in option_datasets.items()
                        },
                        "question_format": str(selected_query),
                        "solver_trace": {
                            "operation": "select_extremum_option_by_visible_element_count",
                            "extremum_kind": str(extremum_kind),
                            "option_counts_by_label": dict(rendered.option_counts_by_label),
                            "answer_label": str(answer_label),
                            "answer_count": int(answer_count),
                            "unique_option_counts": True,
                        },
                    },
                    "witness_symbolic": {
                        "type": "selected_surface_fixture_option_panel",
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
    "DEFAULT_QUERY_ID",
    "HIGHEST_QUERY_ID",
    "LOWEST_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSurfaceFixtureElementCountExtremumLabelTask",
]
