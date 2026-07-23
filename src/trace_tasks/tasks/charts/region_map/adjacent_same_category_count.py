"""Count neighboring regions with the same category as a labeled reference region."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import RegionMapBoundObjective, run_region_map_lifecycle
from .shared.annotations import region_bbox_set_bundle
from .shared.defaults import fixed_scene_variant_probabilities
from .shared.sampling import construct_adjacent_same_category_dataset


TASK_ID = "task_charts__region_map__adjacent_same_category_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@register_task
class ChartsMapAdjacentSameCategoryCountTask:
    """Count neighboring regions with the same category as a labeled reference region."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "charts"
    objective_contract = "adjacent_same_category_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _construct_adjacent_same_category_dataset(self, instance_seed, params, selected_query_id, query_probabilities):
        dataset = construct_adjacent_same_category_dataset(params=params, instance_seed=instance_seed)
        dataset["_scene_variant_probabilities"] = fixed_scene_variant_probabilities("synthetic_region_map")
        return dataset

    def _bind_adjacent_same_category_objective(self, dataset, rendered, selected_query_id):
        annotation = region_bbox_set_bundle(rendered, dataset["annotation_region_ids"])
        return RegionMapBoundObjective(
            answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "reference_region_id": str(dataset["question_params"]["reference_region_id"]),
                "reference_region_label": str(dataset["question_params"]["reference_region_label"]),
                "category_label": str(dataset["question_params"]["category_label"]),
                "adjacent_neighbor_region_ids": list(dataset["question_params"]["adjacent_neighbor_region_ids"]),
                "annotation_region_count": int(len(annotation.annotation_region_ids)),
            },
            witness_symbolic={
                "type": "region_map_adjacent_same_category_count_witness",
                "reference_region_id": str(dataset["question_params"]["reference_region_id"]),
                "reference_region_label": str(dataset["question_params"]["reference_region_label"]),
                "candidate_region_ids": list(annotation.annotation_region_ids),
                "answer_value": int(dataset["answer_value"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_region_map_lifecycle(
            task=self,
            instance_seed=instance_seed,
            params={**dict(params), "scene_variant": "synthetic_region_map"},
            max_attempts=max_attempts,
            default_query_id=QUERY_ID,
            prompt_query_key="adjacent_same_category_count",
            answer_type="integer",
            question_format="map_region_count",
            categorical=True,
            show_region_value_labels=False,
            build_dataset=self._construct_adjacent_same_category_dataset,
            bind_objective=self._bind_adjacent_same_category_objective,
        )
