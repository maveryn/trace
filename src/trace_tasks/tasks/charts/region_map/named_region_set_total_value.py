"""Sum visible integer values for a named set of map regions."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import RegionMapBoundObjective, run_region_map_lifecycle
from .shared.annotations import region_bbox_set_bundle
from .shared.defaults import fixed_scene_variant_probabilities
from .shared.sampling import construct_named_region_set_total_dataset


TASK_ID = "task_charts__region_map__named_region_set_total_value"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@register_task
class ChartsMapNamedRegionSetTotalValueTask:
    """Sum visible integer values for a named set of map regions."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
    domain = "charts"
    objective_contract = "named_region_set_total_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _construct_region_set_dataset(self, instance_seed, params, selected_query_id, query_probabilities):
        dataset = construct_named_region_set_total_dataset(
            scene_variant="synthetic_region_map",
            params=params,
            instance_seed=instance_seed,
        )
        dataset["_scene_variant_probabilities"] = fixed_scene_variant_probabilities("synthetic_region_map")
        return dataset

    def _bind_region_set_total_objective(self, dataset, rendered, selected_query_id):
        annotation = region_bbox_set_bundle(rendered, dataset["annotation_region_ids"])
        return RegionMapBoundObjective(
            answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "region_set_name": str(dataset["question_params"]["region_set_name"]),
                "region_set_region_ids": [str(item) for item in dataset["question_params"]["region_set_region_ids"]],
                "annotation_region_count": int(len(annotation.annotation_region_ids)),
            },
            witness_symbolic={
                "type": "region_map_named_set_total_witness",
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
            prompt_query_key="named_region_set_total_value",
            answer_type="integer",
            question_format="map_region_value",
            categorical=False,
            show_region_value_labels=True,
            build_dataset=self._construct_region_set_dataset,
            bind_objective=self._bind_region_set_total_objective,
        )
