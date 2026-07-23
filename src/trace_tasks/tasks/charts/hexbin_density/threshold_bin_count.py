"""Threshold-count task for hexbin-density charts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from trace_tasks.tasks.charts.hexbin_density.shared.annotations import (
    annotation_point_set,
    projected_annotation_payload,
)
from trace_tasks.tasks.charts.hexbin_density.shared.defaults import (
    DOMAIN,
    SCENE_ID,
    SUPPORTED_DENSITY_PALETTE_SCHEMES,
)
from trace_tasks.tasks.charts.hexbin_density.shared.prompts import build_prompt
from trace_tasks.tasks.charts.hexbin_density.shared.rendering import render_dataset
from trace_tasks.tasks.charts.hexbin_density.shared.sampling import build_threshold_dataset
from trace_tasks.tasks.charts.hexbin_density.shared.state import HexbinDataset, RenderedHexbinScene
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family


TASK_ID = "task_charts__hexbin_density__threshold_bin_count"
ABOVE_QUERY_ID = "above_threshold_bin_count"
BELOW_QUERY_ID = "below_threshold_bin_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    ABOVE_QUERY_ID,
    BELOW_QUERY_ID,
)
THRESHOLD_DIRECTION_BY_QUERY_ID: Mapping[str, str] = {
    ABOVE_QUERY_ID: "above",
    BELOW_QUERY_ID: "below",
}
DEFAULT_QUERY_ID = ABOVE_QUERY_ID


def _build_trace_payload(
    *,
    dataset: HexbinDataset,
    rendered: RenderedHexbinScene,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    prompt_trace: Mapping[str, Any],
    annotation: list[list[float]],
    projected_annotation: Mapping[str, Any],
    answer_value: int,
    chart_font_family: str,
) -> dict[str, Any]:
    """Bind task-owned answer/query semantics to one rendered density scene."""

    query_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "row_count": int(dataset.row_count),
        "column_count": int(dataset.column_count),
        "occupied_bin_count": int(len(dataset.bins)),
        "query_id_probabilities": dict(query_probabilities),
        **dict(dataset.query.trace),
    }
    return {
        "scene_ir": {
            "scene_kind": "chart_hexbin_density",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": str(selected_query_id),
                "answer": int(answer_value),
                "annotation_bin_ids": list(dataset.query.annotation_bin_ids),
                "density_threshold_level": int(dataset.query.threshold_level),
                "density_threshold_direction": str(dataset.query.threshold_direction),
            },
        },
        "query_spec": {
            "query_id": str(selected_query_id),
            **dict(prompt_trace),
            "params": dict(query_params),
        },
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "legend_bbox_px": list(rendered.legend_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "threshold_guide_bbox_px": list(rendered.threshold_guide_bbox_px),
            "font_assets": chart_font_asset_metadata(str(chart_font_family)),
            **dict(rendered.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "legend_bbox_px": list(rendered.legend_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "threshold_guide_bbox_px": list(rendered.threshold_guide_bbox_px),
            "bin_bboxes_px": dict(rendered.bin_bboxes_px),
            "bin_centers_px": dict(rendered.bin_centers_px),
        },
        "execution_trace": {
            "query_id": str(selected_query_id),
            "scene_id": SCENE_ID,
            "question_format": "hexbin_density_threshold_query",
            "answer": int(answer_value),
            "answer_type": "integer",
            "annotation_type": "point_set",
            "row_count": int(dataset.row_count),
            "column_count": int(dataset.column_count),
            "occupied_bin_count": int(len(dataset.bins)),
            "density_class_count": 5,
            "density_palette_scheme": str(dataset.density_palette_scheme),
            "density_palette_rgb": [list(color) for color in dataset.density_palette_rgb],
            "annotation_bin_ids": list(dataset.query.annotation_bin_ids),
            "bins": {
                str(bin_item.bin_id): {
                    "row_index": int(bin_item.row_index),
                    "column_index": int(bin_item.column_index),
                    "density_level": int(bin_item.density_level),
                }
                for bin_item in dataset.bins
            },
            **dict(dataset.query.trace),
            "query_id_probabilities": dict(query_probabilities),
        },
        "witness_symbolic": {
            "type": "hexbin_density_threshold_witness",
            "annotation_type": "point_set",
            "annotation_bin_ids": list(dataset.query.annotation_bin_ids),
            "answer": int(answer_value),
        },
        "projected_annotation": dict(projected_annotation),
        "annotation_refs": {
            "type": "point_set",
            "items": [
                {"bin_id": str(bin_id), "point_px": list(point)}
                for bin_id, point in zip(dataset.query.annotation_bin_ids, annotation, strict=True)
            ],
        },
    }


@register_task
class ChartsHexbinDensityThresholdBinCountTask:
    """Count visible hex bins satisfying a discrete density threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "threshold_bin_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_dataset(self, instance_seed: int, *, params: Mapping[str, Any], selected_query_id: str) -> HexbinDataset:
        if str(selected_query_id) not in THRESHOLD_DIRECTION_BY_QUERY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        return build_threshold_dataset(
            params=dict(params),
            instance_seed=int(instance_seed),
            threshold_direction=str(THRESHOLD_DIRECTION_BY_QUERY_ID[str(selected_query_id)]),
        )

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the threshold relation, sample one scene, and bind output fields."""

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=self.task_id,
        )
        dataset: HexbinDataset | None = None
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            try:
                attempt_params = {**dict(task_params), "_attempt_index": int(attempt_index)}
                dataset = self._build_dataset(
                    int(instance_seed) + int(attempt_index),
                    params=attempt_params,
                    selected_query_id=str(selected_query_id),
                )
                break
            except Exception as exc:
                last_error = exc
        if dataset is None:
            raise RuntimeError(f"failed to generate {self.task_id} instance") from last_error

        chart_font_family = sample_chart_font_family(
            instance_seed=int(instance_seed),
            namespace=f"{self.task_id}.chart_font",
            params=params,
        )
        with temporary_default_font_family(str(chart_font_family)):
            rendered = render_dataset(dataset, params=params, instance_seed=int(instance_seed))
            prompt, prompt_variants, prompt_trace = build_prompt(
                dataset,
                prompt_query_key=str(selected_query_id),
                instance_seed=int(instance_seed),
            )
        annotation = annotation_point_set(dataset, rendered)
        answer_value = int(dataset.query.answer)
        projected_annotation = projected_annotation_payload(dataset, annotation)
        trace_payload = _build_trace_payload(
            dataset=dataset,
            rendered=rendered,
            selected_query_id=str(selected_query_id),
            query_probabilities=dict(query_probabilities),
            prompt_trace=dict(prompt_trace),
            annotation=annotation,
            projected_annotation=projected_annotation,
            answer_value=int(answer_value),
            chart_font_family=str(chart_font_family),
        )
        return TaskOutput(
            prompt=str(prompt),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=TypedValue(type="point_set", value=[list(value) for value in annotation]),
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
            prompt_variants=dict(prompt_variants),
        )


__all__ = [
    "ChartsHexbinDensityThresholdBinCountTask",
    "SUPPORTED_DENSITY_PALETTE_SCHEMES",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
