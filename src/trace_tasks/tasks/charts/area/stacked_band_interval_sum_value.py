"""Public task for `task_charts__area__stacked_band_interval_sum_value`."""
from __future__ import annotations
from typing import Any
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.area._lifecycle import AreaTaskPlan, area_attempt_seed, materialize_area_plan
from trace_tasks.tasks.charts.area.shared.defaults import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.area.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.area.shared.sampling import sample_categories, sample_interval, sample_point_count, sample_stacked_values, sample_x_labels
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

def _stacked_series_sample(params: dict[str, Any], *, instance_seed: int) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[int, ...]], tuple[int, int], tuple[int, int], tuple[int, int]]:
    point_count, point_count_range = sample_point_count(params, instance_seed=int(instance_seed))
    x_labels = sample_x_labels(count=int(point_count), instance_seed=int(instance_seed))
    category_labels, category_count_range = sample_categories(params=params, instance_seed=int(instance_seed))
    stacked_values = sample_stacked_values(category_count=len(category_labels), point_count=int(point_count), params=params, instance_seed=int(instance_seed))
    series_labels = tuple((str(label) for label in category_labels))
    series_values = {str(label): tuple((int(value) for value in stacked_values[index])) for index, label in enumerate(series_labels)}
    return (tuple((str(label) for label in x_labels)), series_labels, series_values, tuple((int(value) for value in point_count_range)), tuple((int(value) for value in category_count_range)), (int(point_count), int(point_count)))

@register_task
class ChartsAreaStackedBandIntervalSumValueTask:
    """Compute an interval sum for one band in a stacked area chart."""
    task_id = 'task_charts__area__stacked_band_interval_sum_value'
    reasoning_operations = ('aggregation',)
    domain = DOMAIN
    objective_contract = 'stacked_band_interval_sum_value'
    supported_query_ids = (DEFAULT_QUERY_ID,)
    default_dataset_enabled = True

    def _resolve_sum_target(self, *, params: dict[str, Any], instance_seed: int, x_labels: tuple[str, ...], series_labels: tuple[str, ...], series_values: dict[str, tuple[int, ...]]) -> dict[str, Any]:
        start_index, end_index, interval_span_range = sample_interval(point_count=len(x_labels), params=params, instance_seed=int(instance_seed))
        rng = spawn_rng(int(instance_seed), 'charts.area.stacked_band_interval_sum.category')
        selected_category = str(series_labels[int(rng.randrange(len(series_labels)))])
        interval_values = tuple((int(series_values[selected_category][index]) for index in range(int(start_index), int(end_index) + 1)))
        return {'selected_category': str(selected_category), 'start_index': int(start_index), 'end_index': int(end_index), 'interval_span_range': tuple((int(value) for value in interval_span_range)), 'interval_values': interval_values, 'answer_value': int(sum(interval_values))}

    def _build_band_sum_plan(self, params: dict[str, Any], instance_seed: int, selected_query_id: str) -> AreaTaskPlan:
        """Bind the selected-band interval-sum objective before rendering.

        This hook owns category selection, interval summing, semantic
        annotation pairs, prompt slots, and relation fields for the public task.
        """
        x_labels, series_labels, series_values, point_count_range, category_count_range, _ = _stacked_series_sample(params, instance_seed=int(instance_seed))
        target = self._resolve_sum_target(params=params, instance_seed=int(instance_seed), x_labels=x_labels, series_labels=series_labels, series_values=series_values)
        start_index = int(target['start_index'])
        end_index = int(target['end_index'])
        selected_category = str(target['selected_category'])
        annotation_pairs = tuple(((selected_category, str(x_labels[index])) for index in range(start_index, end_index + 1)))
        relations = {'query_id': str(selected_query_id), 'point_count': int(len(x_labels)), 'point_count_range': list(point_count_range), 'x_labels': [str(label) for label in x_labels], 'series_labels': [str(label) for label in series_labels], 'start_index': int(start_index), 'end_index': int(end_index), 'start_label': str(x_labels[start_index]), 'end_label': str(x_labels[end_index]), 'interval_span': int(end_index) - int(start_index), 'interval_span_range': list(target['interval_span_range']), 'stacked': True, 'category_label': str(selected_category), 'category_count': int(len(series_labels)), 'category_count_range': list(category_count_range), 'interval_values': [int(value) for value in target['interval_values']], 'interval_sum': int(target['answer_value'])}
        prompt_artifacts = build_prompt_artifacts(domain=self.domain, scene_variant='stacked_area', prompt_query_key=self.objective_contract, dynamic_slots={'category_label': f'"{selected_category}"', 'start_label': f'''"{relations['start_label']}"''', 'end_label': f'''"{relations['end_label']}"'''}, instance_seed=int(instance_seed))
        return AreaTaskPlan(x_labels=x_labels, series_labels=series_labels, series_values=series_values, stacked=True, scene_variant='stacked_area', answer_gt=TypedValue(type='integer', value=int(target['answer_value'])), answer_value=int(target['answer_value']), question_format='numeric_open', annotation_pairs=annotation_pairs, relations=relations, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the query locally, then delegate neutral lifecycle plumbing."""
        selected_query_id, _query_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=DEFAULT_QUERY_ID, task_id=self.task_id)
        attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_seed = area_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = self._build_band_sum_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_area_plan(instance_seed=int(attempt_seed), params=task_params, selected_query_id=str(selected_query_id), plan=plan)
                return TaskOutput(prompt=materialized.prompt, answer_gt=materialized.answer_gt, annotation_gt=materialized.annotation_gt, image=materialized.image, image_id='img0', trace_payload=materialized.trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=materialized.query_id, prompt_variants=materialized.prompt_variants)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f'failed to generate {self.task_id}: {last_error}')
__all__ = ['ChartsAreaStackedBandIntervalSumValueTask']
