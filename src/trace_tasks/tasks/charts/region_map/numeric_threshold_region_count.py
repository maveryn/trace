"""Count map regions satisfying a one-bound numeric legend predicate."""

from __future__ import annotations

from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import RegionMapBoundObjective, run_region_map_lifecycle
from .shared.annotations import region_point_set_bundle
from .shared.defaults import resolve_scene_variant
from .shared.sampling import construct_numeric_threshold_dataset


TASK_ID = "task_charts__region_map__numeric_threshold_region_count"
GREATER_THAN_QUERY_ID = "greater_than_numeric_threshold_region_count"
LESS_THAN_QUERY_ID = "less_than_numeric_threshold_region_count"
SUPPORTED_QUERY_IDS = (GREATER_THAN_QUERY_ID, LESS_THAN_QUERY_ID)
DEFAULT_QUERY_ID = GREATER_THAN_QUERY_ID
THRESHOLD_DIRECTION_BY_QUERY_ID = {GREATER_THAN_QUERY_ID: "greater_than", LESS_THAN_QUERY_ID: "less_than"}


def _threshold_direction_probabilities(query_probabilities):
    return {
        THRESHOLD_DIRECTION_BY_QUERY_ID[str(query_id)]: float(probability)
        for query_id, probability in query_probabilities.items()
    }


@register_task
class ChartsMapNumericThresholdRegionCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "charts"
    objective_contract = "numeric_threshold_region_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _construct_numeric_threshold_dataset(self, instance_seed, params, selected_query_id, query_probabilities):
        scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=instance_seed)
        dataset = construct_numeric_threshold_dataset(
            scene_variant=scene_variant,
            threshold_direction=THRESHOLD_DIRECTION_BY_QUERY_ID[str(selected_query_id)],
            threshold_direction_probabilities=_threshold_direction_probabilities(query_probabilities),
            params=params,
            instance_seed=instance_seed,
        )
        dataset["_scene_variant_probabilities"] = scene_variant_probabilities
        return dataset

    def _bind_numeric_threshold_objective(self, dataset, rendered, selected_query_id):
        annotation = region_point_set_bundle(rendered, dataset["annotation_region_ids"])
        return RegionMapBoundObjective(
            answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
            annotation=annotation,
            relations=dict(
                query_id=str(selected_query_id),
                threshold_direction=str(dataset["question_params"]["threshold_direction"]),
                threshold_value=int(dataset["question_params"]["threshold_value"]),
                annotation_region_count=int(len(annotation.annotation_region_ids)),
            ),
            witness_symbolic=dict(
                type="region_map_numeric_threshold_count_witness",
                candidate_region_ids=list(annotation.annotation_region_ids),
                answer_value=int(dataset["answer_value"]),
            ),
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_region_map_lifecycle(
            task=self,
            instance_seed=instance_seed,
            params=dict(params),
            max_attempts=max_attempts,
            default_query_id=DEFAULT_QUERY_ID,
            prompt_query_key="numeric_threshold_region_count",
            answer_type="integer",
            question_format="map_region_count",
            categorical=False,
            show_region_value_labels=False,
            build_dataset=self._construct_numeric_threshold_dataset,
            bind_objective=self._bind_numeric_threshold_objective,
        )
