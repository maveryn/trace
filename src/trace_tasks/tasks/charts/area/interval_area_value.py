"""Public task for `task_charts__area__interval_area_value`."""
from __future__ import annotations
from typing import Any
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.area._lifecycle import AreaTaskPlan, area_attempt_seed, materialize_area_plan
from trace_tasks.tasks.charts.area.shared.defaults import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.area.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.area.shared.sampling import sample_interval, sample_point_count, sample_single_values, sample_x_labels, trapezoid_interval_area
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

def _interval_area_relations(*, query_id: str, x_labels: tuple[str, ...], values: tuple[int, ...], start_index: int, end_index: int, point_count_range: tuple[int, int], interval_span_range: tuple[int, int], answer_value: int) -> dict[str, Any]:
    terms = [{'left_label': str(x_labels[index]), 'right_label': str(x_labels[index + 1]), 'left_value': int(values[index]), 'right_value': int(values[index + 1]), 'area': int((int(values[index]) + int(values[index + 1])) // 2)} for index in range(int(start_index), int(end_index))]
    return {'query_id': str(query_id), 'point_count': int(len(x_labels)), 'point_count_range': list(point_count_range), 'x_labels': [str(label) for label in x_labels], 'series_labels': ['Series'], 'start_index': int(start_index), 'end_index': int(end_index), 'start_label': str(x_labels[int(start_index)]), 'end_label': str(x_labels[int(end_index)]), 'interval_span': int(end_index) - int(start_index), 'interval_span_range': list(interval_span_range), 'stacked': False, 'interval_area': int(answer_value), 'unit_spacing': 1, 'trapezoid_terms': terms}

@register_task
class ChartsAreaIntervalAreaValueTask:
    """Compute interval area over a filled area chart."""
    task_id = 'task_charts__area__interval_area_value'
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = 'interval_area_value'
    supported_query_ids = (DEFAULT_QUERY_ID,)
    default_dataset_enabled = True

    def _build_interval_area_plan(self, params: dict[str, Any], instance_seed: int, selected_query_id: str) -> AreaTaskPlan:
        """Bind the single-area interval objective before shared rendering.

        This hook owns interval selection, trapezoid-area calculation, semantic
        annotation pairs, prompt slots, and relation fields for the public task.
        """
        point_count, point_count_range = sample_point_count(params, instance_seed=int(instance_seed))
        x_labels = sample_x_labels(count=int(point_count), instance_seed=int(instance_seed))
        start_index, end_index, interval_span_range = sample_interval(point_count=int(point_count), params=params, instance_seed=int(instance_seed))
        values = sample_single_values(point_count=int(point_count), params=params, instance_seed=int(instance_seed))
        answer_value = trapezoid_interval_area(values, int(start_index), int(end_index))
        annotation_pairs = tuple((('Series', str(x_labels[index])) for index in range(int(start_index), int(end_index) + 1)))
        relations = _interval_area_relations(query_id=str(selected_query_id), x_labels=tuple((str(label) for label in x_labels)), values=tuple((int(value) for value in values)), start_index=int(start_index), end_index=int(end_index), point_count_range=tuple((int(value) for value in point_count_range)), interval_span_range=tuple((int(value) for value in interval_span_range)), answer_value=int(answer_value))
        prompt_artifacts = build_prompt_artifacts(domain=self.domain, scene_variant='single_area', prompt_query_key=self.objective_contract, dynamic_slots={'start_label': f'''"{relations['start_label']}"''', 'end_label': f'''"{relations['end_label']}"'''}, instance_seed=int(instance_seed))
        return AreaTaskPlan(x_labels=tuple((str(label) for label in x_labels)), series_labels=('Series',), series_values={'Series': tuple((int(value) for value in values))}, stacked=False, scene_variant='area', answer_gt=TypedValue(type='integer', value=int(answer_value)), answer_value=int(answer_value), question_format='numeric_open', annotation_pairs=annotation_pairs, relations=relations, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the query locally, then delegate neutral lifecycle plumbing."""
        selected_query_id, _query_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=DEFAULT_QUERY_ID, task_id=self.task_id)
        attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_seed = area_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = self._build_interval_area_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_area_plan(instance_seed=int(attempt_seed), params=task_params, selected_query_id=str(selected_query_id), plan=plan)
                return TaskOutput(prompt=materialized.prompt, answer_gt=materialized.answer_gt, annotation_gt=materialized.annotation_gt, image=materialized.image, image_id='img0', trace_payload=materialized.trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=materialized.query_id, prompt_variants=materialized.prompt_variants)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f'failed to generate {self.task_id}: {last_error}')
__all__ = ['ChartsAreaIntervalAreaValueTask']
