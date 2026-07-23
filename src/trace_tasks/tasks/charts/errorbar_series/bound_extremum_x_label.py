"""Public task for `task_charts__errorbar_series__bound_extremum_x_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.errorbar_series._lifecycle import ErrorbarSeriesTaskPlan, run_errorbar_series_plan
from trace_tasks.tasks.charts.errorbar_series.shared.defaults import choose_from_values
from trace_tasks.tasks.charts.errorbar_series.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.errorbar_series.shared.sampling import (
    make_series,
    palette,
    random_series_triples,
    sample_base_scene,
)
from trace_tasks.tasks.charts.errorbar_series.shared.state import DOMAIN, ErrorbarDataset, ErrorbarQuery, SCENE_ID, SCENE_NAMESPACE
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__errorbar_series__bound_extremum_x_label"
OBJECTIVE_CONTRACT = "bound_extremum_x_label"
QUERY_IDS = (
    "highest_upper_bound_x_label",
    "lowest_lower_bound_x_label",
)
DEFAULT_QUERY_ID = "highest_upper_bound_x_label"


@register_task
class ChartsErrorbarSeriesBoundExtremumXLabelTask:
    """Find the x-axis label with an extremal error-bar bound."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def _build_plan(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> ErrorbarSeriesTaskPlan:
        """Bind the extremal-bound objective for the requested direction."""

        if str(selected_query_id) not in QUERY_IDS:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.bound_extremum")
        base = sample_base_scene(params, instance_seed=int(instance_seed), series_mode="standard")
        target_series_index = int(
            choose_from_values(
                params,
                values=tuple(range(int(base.series_count))),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.bound.target_series",
            )
        )
        answer_index = int(
            choose_from_values(
                params,
                values=tuple(range(int(base.x_count))),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.bound.answer_x",
            )
        )
        bound_kind = "upper" if str(selected_query_id) == "highest_upper_bound_x_label" else "lower"
        extremum_direction = "highest" if str(selected_query_id) == "highest_upper_bound_x_label" else "lowest"
        colors = palette(params)
        series_rows = []
        for series_index in range(int(base.series_count)):
            triples = random_series_triples(rng, x_count=int(base.x_count))
            if int(series_index) == int(target_series_index):
                triples = self._bound_extremum_triples(
                    rng,
                    x_count=int(base.x_count),
                    answer_index=int(answer_index),
                    bound_kind=str(bound_kind),
                )
            series_rows.append(
                make_series(
                    series_id=f"series_{series_index}",
                    label=str(base.series_labels[int(series_index)]),
                    color_rgb=colors[int(series_index)],
                    triples=triples,
                )
            )
        target = series_rows[int(target_series_index)]
        answer_label = str(base.x_labels[int(answer_index)])
        annotation_key = f"{target.label}:{answer_label}"
        relation_params = {
            "query_id": str(selected_query_id),
            "scene_id": SCENE_ID,
            "scene_variant": str(base.scene_variant),
            "scene_variant_probabilities": dict(base.scene_variant_probabilities),
            "x_count": int(base.x_count),
            "series_count": int(base.series_count),
            "target_series_id": str(target.series_id),
            "target_series_label": str(target.label),
            "answer_x_index": int(answer_index),
            "answer_x_label": str(answer_label),
            "bound_kind": str(bound_kind),
            "extremum_direction": str(extremum_direction),
            "answer_value": str(answer_label),
        }
        dataset = ErrorbarDataset(
            x_labels=tuple(base.x_labels),
            x_label_meta=dict(base.x_label_meta),
            series=tuple(series_rows),
            series_label_meta=dict(base.series_label_meta),
            scene_variant=str(base.scene_variant),
            scene_variant_probabilities=dict(base.scene_variant_probabilities),
            prompt_key=str(selected_query_id),
            prompt_key_probabilities={str(selected_query_id): 1.0},
            threshold_value=None,
            target_series_id=str(target.series_id),
            target_x_index=int(answer_index),
            title=str(base.title),
            query=ErrorbarQuery(
                prompt_key=str(selected_query_id),
                answer=str(answer_label),
                answer_type="string",
                annotation_kind="point",
                annotation_item_keys=(str(annotation_key),),
                params=dict(relation_params),
            ),
        )
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slot_values=dynamic_slots(
                dataset,
                bound_phrase=f"{bound_kind} error-bar endpoint",
                extremum_phrase=str(extremum_direction),
            ),
            instance_seed=int(instance_seed),
        )
        return ErrorbarSeriesTaskPlan(
            dataset=dataset,
            params=dict(params),
            answer_gt=TypedValue(type="string", value=str(answer_label)),
            prompt_artifacts=prompt_artifacts,
            relation_params=relation_params,
        )

    @staticmethod
    def _bound_extremum_triples(
        rng: Any,
        *,
        x_count: int,
        answer_index: int,
        bound_kind: str,
    ) -> list[tuple[int, int, int]]:
        """Construct intervals with a unique selected upper or lower endpoint."""

        adjusted: list[tuple[int, int, int]] = []
        for x_index in range(int(x_count)):
            if str(bound_kind) == "upper":
                if int(x_index) == int(answer_index):
                    upper = 95
                    lower = int(rng.randint(58, 72))
                else:
                    upper = int(rng.randint(45, 82))
                    lower = max(0, upper - int(rng.randint(7, 20)))
            else:
                if int(x_index) == int(answer_index):
                    lower = 5
                    upper = int(rng.randint(26, 42))
                else:
                    lower = int(rng.randint(18, 55))
                    upper = min(100, lower + int(rng.randint(7, 20)))
            mid = int(round((int(lower) + int(upper)) / 2))
            adjusted.append((int(lower), int(mid), int(upper)))
        return adjusted

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


__all__ = ["ChartsErrorbarSeriesBoundExtremumXLabelTask"]
