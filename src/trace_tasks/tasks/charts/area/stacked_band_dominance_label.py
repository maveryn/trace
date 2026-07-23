"""Public task for `task_charts__area__stacked_band_dominance_label`."""
from __future__ import annotations
from typing import Any
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.area._lifecycle import AreaTaskPlan, area_attempt_seed, materialize_area_plan
from trace_tasks.tasks.charts.area.shared.defaults import DOMAIN, SCENE_ID, generation_int
from trace_tasks.tasks.charts.area.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.area.shared.sampling import sample_categories, sample_interval, sample_point_count, sample_stacked_values, sample_x_labels
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

def _rank_category_totals(*, series_labels: tuple[str, ...], series_values: dict[str, tuple[int, ...]], start_index: int, end_index: int) -> tuple[dict[str, int], tuple[tuple[str, int], ...]]:
    totals = {str(label): int(sum((int(series_values[str(label)][index]) for index in range(int(start_index), int(end_index) + 1)))) for label in series_labels}
    ranked = tuple(sorted(totals.items(), key=lambda item: (-int(item[1]), str(item[0]))))
    return (totals, ranked)

@register_task
class ChartsAreaStackedDominanceLabelTask:
    """Find the dominant stacked-area category over a labeled interval."""
    task_id = 'task_charts__area__stacked_band_dominance_label'
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = 'stacked_band_dominance_label'
    supported_query_ids = (DEFAULT_QUERY_ID,)
    default_dataset_enabled = True

    def _build_dominance_plan(self, params: dict[str, Any], instance_seed: int, selected_query_id: str) -> AreaTaskPlan:
        """Bind the dominant-category interval objective before rendering.

        This hook owns stacked-band total ranking, uniqueness enforcement,
        semantic annotation pairs, prompt slots, and relation fields for the
        public task.
        """
        point_count, point_count_range = sample_point_count(params, instance_seed=int(instance_seed))
        x_labels = sample_x_labels(count=int(point_count), instance_seed=int(instance_seed))
        start_index, end_index, interval_span_range = sample_interval(point_count=int(point_count), params=params, instance_seed=int(instance_seed))
        category_labels, category_count_range = sample_categories(params=params, instance_seed=int(instance_seed))
        stacked_values = sample_stacked_values(category_count=len(category_labels), point_count=int(point_count), params=params, instance_seed=int(instance_seed))
        series_labels = tuple((str(label) for label in category_labels))
        series_values = {str(label): tuple((int(value) for value in stacked_values[index])) for index, label in enumerate(series_labels)}
        totals_by_category, ranked = _rank_category_totals(series_labels=series_labels, series_values=series_values, start_index=int(start_index), end_index=int(end_index))
        winning_category, winning_total = ranked[0]
        runner_up_total = int(ranked[1][1]) if len(ranked) > 1 else -1
        margin_min = generation_int(params, 'dominance_margin_min', 3)
        dominance_margin = int(winning_total) - int(runner_up_total)
        if dominance_margin < int(margin_min):
            raise ValueError('stacked dominance task did not have a unique enough winner')
        annotation_pairs = tuple(((str(winning_category), str(x_labels[index])) for index in range(int(start_index), int(end_index) + 1)))
        relations = {'query_id': str(selected_query_id), 'point_count': int(point_count), 'point_count_range': list(point_count_range), 'x_labels': [str(label) for label in x_labels], 'series_labels': [str(label) for label in series_labels], 'start_index': int(start_index), 'end_index': int(end_index), 'start_label': str(x_labels[int(start_index)]), 'end_label': str(x_labels[int(end_index)]), 'interval_span': int(end_index) - int(start_index), 'interval_span_range': list(interval_span_range), 'stacked': True, 'category_count': int(len(series_labels)), 'category_count_range': list(category_count_range), 'totals_by_category': {str(key): int(value) for key, value in totals_by_category.items()}, 'winning_category': str(winning_category), 'winning_total': int(winning_total), 'runner_up_total': int(runner_up_total), 'dominance_margin': int(dominance_margin), 'dominance_margin_min': int(margin_min)}
        prompt_artifacts = build_prompt_artifacts(domain=self.domain, scene_variant='stacked_area', prompt_query_key=self.objective_contract, dynamic_slots={'start_label': f'''"{relations['start_label']}"''', 'end_label': f'''"{relations['end_label']}"'''}, instance_seed=int(instance_seed))
        return AreaTaskPlan(x_labels=tuple((str(label) for label in x_labels)), series_labels=series_labels, series_values=series_values, stacked=True, scene_variant='stacked_area', answer_gt=TypedValue(type='string', value=str(winning_category)), answer_value=str(winning_category), question_format='label_open', annotation_pairs=annotation_pairs, relations=relations, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the query locally, then delegate neutral lifecycle plumbing."""
        selected_query_id, _query_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=DEFAULT_QUERY_ID, task_id=self.task_id)
        attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_seed = area_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = self._build_dominance_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_area_plan(instance_seed=int(attempt_seed), params=task_params, selected_query_id=str(selected_query_id), plan=plan)
                return TaskOutput(prompt=materialized.prompt, answer_gt=materialized.answer_gt, annotation_gt=materialized.annotation_gt, image=materialized.image, image_id='img0', trace_payload=materialized.trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=materialized.query_id, prompt_variants=materialized.prompt_variants)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f'failed to generate {self.task_id}: {last_error}')
__all__ = ['ChartsAreaStackedDominanceLabelTask']
