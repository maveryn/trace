"""Public task for `task_charts__uncertainty_band__band_width_extremum_x_label`."""

from __future__ import annotations

import math
from typing import Any

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.uncertainty_band.shared.annotations import band_width_segment
from trace_tasks.tasks.charts.uncertainty_band.shared.prompts import (
    ANNOTATION_HINT_WIDTH,
    ANSWER_HINT_LABEL,
    JSON_EXAMPLE_ANSWER_ONLY_WIDTH,
    JSON_EXAMPLE_WIDTH,
    build_prompt_artifacts,
)
from trace_tasks.tasks.charts.uncertainty_band.shared.rendering import render_dataset
from trace_tasks.tasks.charts.uncertainty_band.shared.sampling import (
    choose_from_values,
    interval_from_mid_width,
    sample_base_band_scene,
)
from trace_tasks.tasks.charts.uncertainty_band.shared.state import BandSeries, DOMAIN, Dataset, SCENE_ID, SCENE_NAMESPACE
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


TASK_ID = "task_charts__uncertainty_band__band_width_extremum_x_label"
OBJECTIVE_CONTRACT = "band_width_extremum_x_label"
WIDEST_QUERY_ID = "widest_band_x_label"
NARROWEST_QUERY_ID = "narrowest_band_x_label"
SUPPORTED_QUERY_IDS = (WIDEST_QUERY_ID, NARROWEST_QUERY_ID)
DEFAULT_QUERY_ID = WIDEST_QUERY_ID


def _rank_phrase(selected_query_id: str) -> str:
    if str(selected_query_id) == WIDEST_QUERY_ID:
        return "widest"
    if str(selected_query_id) == NARROWEST_QUERY_ID:
        return "narrowest"
    raise ValueError(f"unsupported query_id: {selected_query_id}")


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


def _build_width_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
    selected_query_id: str,
) -> tuple[Dataset, str, int, str, dict[str, Any]]:
    """Build a dataset with one unique extremal band width.

    The target series and answer x-index are sampled first, then the selected
    band width is forced to be uniquely widest or narrowest for the public query.
    """

    base = sample_base_band_scene(params, instance_seed=int(instance_seed))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.width.{selected_query_id}")
    answer_index = int(
        choose_from_values(
            params,
            values=tuple(range(int(base.x_count))),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.width.answer_index.{selected_query_id}",
        )
    )
    target_series_index = int(
        choose_from_values(
            params,
            values=(0, 1),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.width.target_series.{selected_query_id}",
        )
    )

    series_values: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for series_index in range(2):
        widths = [int(rng.randint(12, 22)) for _ in range(int(base.x_count))]
        if int(series_index) == int(target_series_index):
            if str(selected_query_id) == WIDEST_QUERY_ID:
                widths = [int(rng.randint(12, 22)) for _ in range(int(base.x_count))]
                widths[int(answer_index)] = int(rng.randint(28, 34))
            else:
                widths = [int(rng.randint(14, 26)) for _ in range(int(base.x_count))]
                widths[int(answer_index)] = int(rng.randint(6, 9))
        lowers: list[int] = []
        mids: list[int] = []
        uppers: list[int] = []
        for index, width in enumerate(widths):
            mid = int(rng.randint(30, 70))
            mid = max(18, min(82, int(mid + round(4.0 * math.sin((index + series_index) * 0.9)))))
            lower, midpoint, upper = interval_from_mid_width(mid, int(width))
            lowers.append(int(lower))
            mids.append(int(midpoint))
            uppers.append(int(upper))
        series_values.append((list(widths), lowers, mids, uppers))

    series = (
        BandSeries("series_a", str(base.series_labels[0]), tuple(base.colors[0]), tuple(series_values[0][1]), tuple(series_values[0][2]), tuple(series_values[0][3])),
        BandSeries("series_b", str(base.series_labels[1]), tuple(base.colors[1]), tuple(series_values[1][1]), tuple(series_values[1][2]), tuple(series_values[1][3])),
    )
    target_series = series[int(target_series_index)]
    target_widths = [
        int(high) - int(low)
        for low, high in zip(target_series.lower_values, target_series.upper_values)
    ]
    if str(selected_query_id) == WIDEST_QUERY_ID:
        resolved_index = max(range(len(target_widths)), key=lambda index: target_widths[index])
    elif str(selected_query_id) == NARROWEST_QUERY_ID:
        resolved_index = min(range(len(target_widths)), key=lambda index: target_widths[index])
    else:
        raise ValueError(f"unsupported query_id: {selected_query_id}")
    if int(resolved_index) != int(answer_index):
        raise ValueError("constructed width support does not match target index")

    dataset = Dataset(
        x_labels=tuple(base.x_labels),
        x_label_meta=dict(base.x_label_meta),
        series=series,
        series_label_meta=dict(base.series_label_meta),
        title=str(base.title),
    )
    answer_label = str(base.x_labels[int(answer_index)])
    trace_params = {
        "x_count": int(base.x_count),
        "x_count_probabilities": dict(base.x_count_probabilities),
        "x_label_meta": dict(base.x_label_meta),
        "series_label_meta": dict(base.series_label_meta),
        "target_series_id": str(target_series.series_id),
        "target_series_label": str(target_series.label),
        "target_x_index": int(answer_index),
        "target_x_label": str(answer_label),
        "rank_phrase": _rank_phrase(str(selected_query_id)),
        "target_band_widths": [int(value) for value in target_widths],
        "answer_band_width": int(target_widths[int(answer_index)]),
        "target_series_index_probabilities": {"series_a": 0.5, "series_b": 0.5},
    }
    return dataset, answer_label, int(answer_index), str(target_series.series_id), trace_params


@register_task
class ChartsUncertaintyBandWidthExtremumXLabelTask:
    """Return the x-axis label with the widest or narrowest band."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _generate_once(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> TaskOutput:
        """Bind width-extremum answer/segment annotation from one execution trace."""

        dataset, answer_label, annotation_x_index, target_series_id, relation_params = _build_width_dataset(
            params=params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
        )
        artifacts = render_dataset(dataset, params=params, instance_seed=int(instance_seed))
        annotation = segment_annotation_artifacts(
            band_width_segment(
                rendered=artifacts.rendered,
                target_series_id=str(target_series_id),
                x_label=str(answer_label),
            )
        )
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            answer_hint=ANSWER_HINT_LABEL,
            annotation_hint=ANNOTATION_HINT_WIDTH,
            json_example=JSON_EXAMPLE_WIDTH,
            json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY_WIDTH,
            dynamic_slot_values={
                "target_series_label": str(relation_params["target_series_label"]),
                "rank_phrase": str(relation_params["rank_phrase"]),
                "series_a_label": str(dataset.series[0].label),
                "series_b_label": str(dataset.series[1].label),
            },
            instance_seed=int(instance_seed),
        )
        query_params = {
            "query_id": str(selected_query_id),
            "answer_value": str(answer_label),
            "answer_type": "string",
            **dict(relation_params),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": SCENE_ID,
                "entities": [dict(entity) for entity in artifacts.rendered.entities],
                "relations": {
                    "query_id": str(selected_query_id),
                    "answer_value": str(answer_label),
                    "annotation_x_indices": [int(annotation_x_index)],
                    "target_series_id": str(target_series_id),
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
                "answer_value": str(answer_label),
                "answer_type": "string",
                "annotation_x_indices": [int(annotation_x_index)],
                "target_series_id": str(target_series_id),
                **dict(relation_params),
            },
            "witness_symbolic": {
                "type": "uncertainty_band_width_witness",
                "answer_value": str(answer_label),
                "annotation_x_indices": [int(annotation_x_index)],
                "target_series_id": str(target_series_id),
            },
            "projected_annotation": {
                **dict(annotation.projected_annotation),
                "annotation_x_indices": [int(annotation_x_index)],
                "target_series_id": str(target_series_id),
            },
            "post_image_noise": dict(artifacts.post_image_noise),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=TypedValue(type="string", value=str(answer_label)),
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


__all__ = [
    "ChartsUncertaintyBandWidthExtremumXLabelTask",
    "NARROWEST_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "WIDEST_QUERY_ID",
]
