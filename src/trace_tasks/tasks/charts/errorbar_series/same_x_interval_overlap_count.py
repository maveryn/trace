"""Public task for `task_charts__errorbar_series__same_x_interval_overlap_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.errorbar_series._lifecycle import ErrorbarSeriesTaskPlan, run_errorbar_series_plan
from trace_tasks.tasks.charts.errorbar_series.shared.defaults import choose_from_values, group_generation_default
from trace_tasks.tasks.charts.errorbar_series.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.errorbar_series.shared.sampling import (
    make_series,
    palette,
    random_series_triples,
    sample_base_scene,
)
from trace_tasks.tasks.charts.errorbar_series.shared.state import DOMAIN, ErrorbarDataset, ErrorbarQuery, SCENE_ID, SCENE_NAMESPACE
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID


TASK_ID = "task_charts__errorbar_series__same_x_interval_overlap_count"
OBJECTIVE_CONTRACT = "same_x_interval_overlap_count"
PROMPT_KEY = "overlap_target_errorbar_at_x_count"


@register_task
class ChartsErrorbarSeriesSameXIntervalOverlapCountTask:
    """Count same-x error bars overlapping a target series interval."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'spatial_relations')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (DEFAULT_QUERY_ID,)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> ErrorbarSeriesTaskPlan:
        """Bind the same-x interval-overlap count objective."""

        if str(selected_query_id) != DEFAULT_QUERY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.overlap")
        base = sample_base_scene(params, instance_seed=int(instance_seed), series_mode="overlap")
        target_series_index = int(
            choose_from_values(
                params,
                values=tuple(range(int(base.series_count))),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.overlap.target_series",
            )
        )
        target_x_index = int(
            choose_from_values(
                params,
                values=tuple(range(int(base.x_count))),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.overlap.target_x",
            )
        )
        max_overlap = min(
            int(base.series_count) - 1,
            int(params.get("overlap_answer_count_max", group_generation_default("overlap_answer_count_max", 3))),
        )
        min_overlap = min(
            max_overlap,
            int(params.get("overlap_answer_count_min", group_generation_default("overlap_answer_count_min", 1))),
        )
        answer_count = int(
            choose_from_values(
                params,
                values=tuple(range(int(min_overlap), int(max_overlap) + 1)),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.overlap.answer_count",
            )
        )
        other_indices = [index for index in range(int(base.series_count)) if int(index) != int(target_series_index)]
        rng.shuffle(other_indices)
        overlapping_indices = set(other_indices[: int(answer_count)])
        target_low = int(rng.randint(34, 48))
        target_high = int(target_low + rng.randint(18, 28))
        target_mid = int(round((target_low + target_high) / 2))
        colors = palette(params)
        series_rows = []
        for series_index in range(int(base.series_count)):
            triples = random_series_triples(rng, x_count=int(base.x_count))
            adjusted = list(triples)
            if int(series_index) == int(target_series_index):
                adjusted[int(target_x_index)] = (int(target_low), int(target_mid), int(target_high))
            elif int(series_index) in overlapping_indices:
                low = int(rng.randint(max(0, target_low - 12), min(target_high - 2, target_low + 10)))
                high = int(rng.randint(max(low + 6, target_low + 2), min(100, target_high + 14)))
                mid = int(round((low + high) / 2))
                adjusted[int(target_x_index)] = (int(low), int(mid), int(high))
            else:
                if rng.random() < 0.5:
                    high = max(6, int(target_low) - int(rng.randint(4, 14)))
                    low = max(0, high - int(rng.randint(8, 18)))
                else:
                    low = min(92, int(target_high) + int(rng.randint(4, 14)))
                    high = min(100, low + int(rng.randint(8, 18)))
                mid = int(round((low + high) / 2))
                adjusted[int(target_x_index)] = (int(low), int(mid), int(high))
            series_rows.append(
                make_series(
                    series_id=f"series_{series_index}",
                    label=str(base.series_labels[int(series_index)]),
                    color_rgb=colors[int(series_index)],
                    triples=adjusted,
                )
            )
        target = series_rows[int(target_series_index)]
        target_x_label = str(base.x_labels[int(target_x_index)])
        counted_labels = [str(series_rows[index].label) for index in sorted(overlapping_indices)]
        annotation_keys = tuple(f"{label}:{target_x_label}" for label in counted_labels)
        relation_params = {
            "query_id": str(selected_query_id),
            "internal_query_id": PROMPT_KEY,
            "scene_id": SCENE_ID,
            "scene_variant": str(base.scene_variant),
            "scene_variant_probabilities": dict(base.scene_variant_probabilities),
            "x_count": int(base.x_count),
            "series_count": int(base.series_count),
            "target_series_id": str(target.series_id),
            "target_series_label": str(target.label),
            "target_x_index": int(target_x_index),
            "target_x_label": str(target_x_label),
            "overlapping_series_labels": list(counted_labels),
            "overlapping_series_ids": [str(series_rows[index].series_id) for index in sorted(overlapping_indices)],
            "answer_value": int(answer_count),
        }
        dataset = ErrorbarDataset(
            x_labels=tuple(base.x_labels),
            x_label_meta=dict(base.x_label_meta),
            series=tuple(series_rows),
            series_label_meta=dict(base.series_label_meta),
            scene_variant=str(base.scene_variant),
            scene_variant_probabilities=dict(base.scene_variant_probabilities),
            prompt_key=PROMPT_KEY,
            prompt_key_probabilities={PROMPT_KEY: 1.0},
            threshold_value=None,
            target_series_id=str(target.series_id),
            target_x_index=int(target_x_index),
            title=str(base.title),
            query=ErrorbarQuery(
                prompt_key=PROMPT_KEY,
                answer=int(answer_count),
                answer_type="integer",
                annotation_kind="segment_set",
                annotation_item_keys=tuple(annotation_keys),
                params=dict(relation_params),
            ),
        )
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=PROMPT_KEY,
            dynamic_slot_values=dynamic_slots(dataset),
            instance_seed=int(instance_seed),
        )
        return ErrorbarSeriesTaskPlan(
            dataset=dataset,
            params=dict(params),
            answer_gt=TypedValue(type="integer", value=int(answer_count)),
            prompt_artifacts=prompt_artifacts,
            relation_params=relation_params,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_errorbar_series_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsErrorbarSeriesSameXIntervalOverlapCountTask"]
