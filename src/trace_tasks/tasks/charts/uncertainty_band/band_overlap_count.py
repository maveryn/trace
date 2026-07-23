"""Public task for `task_charts__uncertainty_band__band_overlap_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.uncertainty_band.shared.annotations import overlap_center_points
from trace_tasks.tasks.charts.uncertainty_band.shared.prompts import (
    ANNOTATION_HINT_OVERLAP,
    ANSWER_HINT_COUNT,
    JSON_EXAMPLE_ANSWER_ONLY_OVERLAP,
    JSON_EXAMPLE_OVERLAP,
    build_prompt_artifacts,
)
from trace_tasks.tasks.charts.uncertainty_band.shared.rendering import render_dataset
from trace_tasks.tasks.charts.uncertainty_band.shared.sampling import (
    choose_from_values,
    generation_default,
    interval_from_mid_width,
    probability_map,
    sample_base_band_scene,
)
from trace_tasks.tasks.charts.uncertainty_band.shared.state import BandSeries, DOMAIN, Dataset, SCENE_ID, SCENE_NAMESPACE
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import point_set_annotation_artifacts
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


TASK_ID = "task_charts__uncertainty_band__band_overlap_count"
OBJECTIVE_CONTRACT = "band_overlap_count"
PROMPT_QUERY_KEY = "band_overlap_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)


def _series_payload(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "series_id": str(series.series_id),
            "label": str(series.label),
            "color_rgb": [int(value) for value in series.color_rgb],
            "lower_values": [int(value) for value in series.lower_values],
            "mid_values": [int(value) for value in series.mid_values],
            "upper_values": [int(value) for value in series.upper_values],
            "band_widths": [
                int(high) - int(low)
                for low, high in zip(series.lower_values, series.upper_values)
            ],
        }
        for series in dataset.series
    ]


def _build_overlap_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
) -> tuple[Dataset, int, tuple[int, ...], dict[str, Any]]:
    """Build a dataset with a controlled number of overlapping x positions.

    The invariant is that only the sampled `overlap_indices` produce intersecting
    vertical band intervals, so the integer answer is fixed before rendering.
    """

    base = sample_base_band_scene(params, instance_seed=int(instance_seed))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.overlap")
    answer_min = int(params.get("overlap_answer_count_min", generation_default("overlap_answer_count_min", 1)))
    answer_max = int(params.get("overlap_answer_count_max", generation_default("overlap_answer_count_max", 5)))
    support = tuple(range(max(1, int(answer_min)), min(int(answer_max), int(base.x_count)) + 1))
    overlap_count = int(
        choose_from_values(
            params,
            values=support,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.overlap_answer_count",
        )
    )
    overlap_indices = tuple(sorted(rng.sample(list(range(int(base.x_count))), k=int(overlap_count))))
    overlap_index_set = set(int(index) for index in overlap_indices)

    lower_a: list[int] = []
    mid_a: list[int] = []
    upper_a: list[int] = []
    lower_b: list[int] = []
    mid_b: list[int] = []
    upper_b: list[int] = []
    overlap_intervals: dict[str, dict[str, int]] = {}

    for index in range(int(base.x_count)):
        if int(index) in overlap_index_set:
            width_a = int(rng.randint(16, 26))
            width_b = int(rng.randint(16, 26))
            base_mid = int(rng.randint(34, 66))
            half_sum = int(width_a // 2) + int(width_b // 2)
            delta_limit = max(1, int(half_sum) - 4)
            b_mid = max(18, min(82, int(base_mid + rng.randint(-delta_limit, delta_limit))))
            a_low, a_mid, a_high = interval_from_mid_width(base_mid, width_a)
            b_low, b_mid, b_high = interval_from_mid_width(b_mid, width_b)
        else:
            width_a = int(rng.randint(12, 18))
            width_b = int(rng.randint(12, 18))
            if bool(rng.randint(0, 1)):
                a_low, a_mid, a_high = interval_from_mid_width(int(rng.randint(62, 78)), width_a)
                b_low, b_mid, b_high = interval_from_mid_width(int(rng.randint(22, 38)), width_b)
            else:
                a_low, a_mid, a_high = interval_from_mid_width(int(rng.randint(22, 38)), width_a)
                b_low, b_mid, b_high = interval_from_mid_width(int(rng.randint(62, 78)), width_b)
        lower_a.append(int(a_low))
        mid_a.append(int(a_mid))
        upper_a.append(int(a_high))
        lower_b.append(int(b_low))
        mid_b.append(int(b_mid))
        upper_b.append(int(b_high))
        if max(int(a_low), int(b_low)) <= min(int(a_high), int(b_high)):
            overlap_intervals[str(base.x_labels[index])] = {
                "x_index": int(index),
                "lower": max(int(a_low), int(b_low)),
                "upper": min(int(a_high), int(b_high)),
            }

    if len(overlap_intervals) != int(overlap_count):
        raise ValueError("constructed overlap support does not match target count")

    series = (
        BandSeries("series_a", str(base.series_labels[0]), tuple(base.colors[0]), tuple(lower_a), tuple(mid_a), tuple(upper_a)),
        BandSeries("series_b", str(base.series_labels[1]), tuple(base.colors[1]), tuple(lower_b), tuple(mid_b), tuple(upper_b)),
    )
    dataset = Dataset(
        x_labels=tuple(base.x_labels),
        x_label_meta=dict(base.x_label_meta),
        series=series,
        series_label_meta=dict(base.series_label_meta),
        title=str(base.title),
    )
    trace_params = {
        "x_count": int(base.x_count),
        "x_count_probabilities": dict(base.x_count_probabilities),
        "x_label_meta": dict(base.x_label_meta),
        "series_label_meta": dict(base.series_label_meta),
        "overlap_count": int(overlap_count),
        "overlap_answer_count_probabilities": probability_map(support),
        "overlap_x_labels": [str(base.x_labels[index]) for index in overlap_indices],
        "overlap_intervals_by_label": dict(overlap_intervals),
    }
    return dataset, int(overlap_count), tuple(int(index) for index in overlap_indices), trace_params


@register_task
class ChartsUncertaintyBandOverlapCountTask:
    """Count x-axis positions where two uncertainty bands overlap."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _generate_once(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> TaskOutput:
        """Bind overlap answer/annotation from one sampled execution trace."""

        if str(selected_query_id) != DEFAULT_QUERY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        dataset, answer_value, annotation_x_indices, relation_params = _build_overlap_dataset(
            params=params,
            instance_seed=int(instance_seed),
        )
        artifacts = render_dataset(dataset, params=params, instance_seed=int(instance_seed))
        annotation = point_set_annotation_artifacts(
            overlap_center_points(
                dataset=dataset,
                rendered=artifacts.rendered,
                x_indices=annotation_x_indices,
            )
        )
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=PROMPT_QUERY_KEY,
            answer_hint=ANSWER_HINT_COUNT,
            annotation_hint=ANNOTATION_HINT_OVERLAP,
            json_example=JSON_EXAMPLE_OVERLAP,
            json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY_OVERLAP,
            dynamic_slot_values={
                "series_a_label": str(dataset.series[0].label),
                "series_b_label": str(dataset.series[1].label),
            },
            instance_seed=int(instance_seed),
        )
        query_params = {
            "query_id": str(selected_query_id),
            "answer_value": int(answer_value),
            "answer_type": "integer",
            **dict(relation_params),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": SCENE_ID,
                "entities": [dict(entity) for entity in artifacts.rendered.entities],
                "relations": {
                    "query_id": str(selected_query_id),
                    "answer_value": int(answer_value),
                    "annotation_x_indices": [int(value) for value in annotation_x_indices],
                    "target_series_id": None,
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=query_params,
            ),
            "render_spec": {
                "canvas_width": int(artifacts.rendered.image.size[0]),
                "canvas_height": int(artifacts.rendered.image.size[1]),
                "x_labels": [str(label) for label in dataset.x_labels],
                "series_labels": [str(series.label) for series in dataset.series],
                "render_meta": dict(artifacts.rendered.render_meta),
                "information_scene_style": dict(artifacts.rendered.render_meta.get("information_style", {})),
                "post_image_noise": dict(artifacts.post_image_noise),
            },
            "render_map": {
                "plot_bbox_px": list(artifacts.rendered.plot_bbox_px),
                "panel_bbox_px": list(artifacts.rendered.render_meta.get("panel_bbox_px", [])),
                "series_band_bboxes_px": dict(artifacts.rendered.series_band_bboxes_px),
                "point_map_px": dict(artifacts.rendered.point_map_px),
                "overlap_points_px": dict(artifacts.rendered.overlap_points_px),
            },
            "execution_trace": {
                "query_id": str(selected_query_id),
                "question_format": "uncertainty_band",
                "x_count": int(len(dataset.x_labels)),
                "x_labels": [str(label) for label in dataset.x_labels],
                "series": _series_payload(dataset),
                "answer_value": int(answer_value),
                "answer_type": "integer",
                "annotation_x_indices": [int(value) for value in annotation_x_indices],
                "target_series_id": None,
                **dict(relation_params),
            },
            "witness_symbolic": {
                "type": "uncertainty_band_overlap_witness",
                "answer_value": int(answer_value),
                "annotation_x_indices": [int(value) for value in annotation_x_indices],
            },
            "projected_annotation": {
                **dict(annotation.projected_annotation),
                "annotation_x_indices": [int(value) for value in annotation_x_indices],
            },
            "post_image_noise": dict(artifacts.post_image_noise),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=annotation.annotation_gt,
            image=artifacts.rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if int(attempt_index) == 0
                else int(hash64(int(instance_seed), self.task_id, int(attempt_index)))
            )
            try:
                attempt_params = {**dict(task_params), "_attempt_index": int(attempt_index)}
                return self._generate_once(
                    int(attempt_seed),
                    params=attempt_params,
                    selected_query_id=str(selected_query_id),
                )
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"failed to generate {self.task_id}: {last_error}")


__all__ = ["ChartsUncertaintyBandOverlapCountTask", "SUPPORTED_QUERY_IDS"]
