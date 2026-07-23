"""Count map regions assigned to one visible legend category."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import RegionMapBoundObjective, run_region_map_lifecycle
from .shared.annotations import region_point_set_bundle
from .shared.defaults import resolve_scene_variant
from .shared.sampling import construct_categorical_count_dataset


TASK_ID = "task_charts__region_map__categorical_region_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@register_task
class ChartsMapCategoricalRegionCountTask:
    """Count map regions assigned to one visible legend category."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "charts"
    objective_contract = "categorical_region_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _construct_categorical_dataset(self, instance_seed, params, selected_query_id, query_probabilities):
        scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=instance_seed)
        dataset = construct_categorical_count_dataset(
            scene_variant=scene_variant,
            params=params,
            instance_seed=instance_seed,
        )
        dataset["_scene_variant_probabilities"] = scene_variant_probabilities
        return dataset

    def _bind_categorical_objective(self, dataset, rendered, selected_query_id):
        annotation = region_point_set_bundle(rendered, dataset["annotation_region_ids"])
        return RegionMapBoundObjective(
            answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "category_label": str(dataset["question_params"]["category_label"]),
                "annotation_region_count": int(len(annotation.annotation_region_ids)),
            },
            witness_symbolic={
                "type": "region_map_category_count_witness",
                "candidate_region_ids": list(annotation.annotation_region_ids),
                "answer_value": int(dataset["answer_value"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_region_map_lifecycle(
            task=self,
            instance_seed=instance_seed,
            params=dict(params),
            max_attempts=max_attempts,
            default_query_id=QUERY_ID,
            prompt_query_key="categorical_region_count",
            answer_type="integer",
            question_format="map_region_count",
            categorical=True,
            show_region_value_labels=False,
            build_dataset=self._construct_categorical_dataset,
            bind_objective=self._bind_categorical_objective,
        )
