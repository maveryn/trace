from __future__ import annotations

from ....core.types import TypedValue
from ...registry import register_task
from ._lifecycle import MarkerMapBoundObjective, run_marker_layer_lifecycle
from .shared.annotations import marker_point_bundle
from .shared.defaults import resolve_scene_variant
from .shared.sampling import construct_marker_extremum_dataset


TASK_ID = "task_charts__region_map__marker_region_extremum_label"
LARGEST_QUERY_ID = "largest_marker_region_extremum_label"
SMALLEST_QUERY_ID = "smallest_marker_region_extremum_label"
SUPPORTED_QUERY_IDS = (LARGEST_QUERY_ID, SMALLEST_QUERY_ID)
DEFAULT_QUERY_ID = LARGEST_QUERY_ID
EXTREMUM_DIRECTION_BY_QUERY_ID = {
    LARGEST_QUERY_ID: "largest",
    SMALLEST_QUERY_ID: "smallest",
}


def _extremum_direction_probabilities(query_probabilities):
    return {
        EXTREMUM_DIRECTION_BY_QUERY_ID[str(query_id)]: float(probability)
        for query_id, probability in query_probabilities.items()
    }


@register_task
class ChartsRegionMapMarkerRegionExtremumLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "marker_region_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _construct_extremum_dataset(
        self,
        instance_seed,
        params,
        selected_query_id,
        query_probabilities,
    ):
        scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=instance_seed)
        dataset = construct_marker_extremum_dataset(
            scene_variant=scene_variant,
            extremum_direction=EXTREMUM_DIRECTION_BY_QUERY_ID[str(selected_query_id)],
            extremum_direction_probabilities=_extremum_direction_probabilities(query_probabilities),
            params=params,
            instance_seed=instance_seed,
        )
        dataset["_scene_variant_probabilities"] = scene_variant_probabilities
        return dataset

    def _bind_extremum_objective(
        self,
        dataset,
        rendered,
        selected_query_id,
    ):
        region_id = str(dataset["annotation_region_ids"][0])
        annotation = marker_point_bundle(rendered, region_id)
        return MarkerMapBoundObjective(
            answer_gt=TypedValue(type="string", value=str(dataset["answer_value"])),
            annotation=annotation,
            relations={
                "query_id": str(selected_query_id),
                "extremum_direction": str(dataset["question_params"]["extremum_direction"]),
                "answer_region_id": str(region_id),
            },
            witness_symbolic={
                "type": "map_marker_extremum_label_witness",
                "candidate_region_ids": [str(region_id)],
                "answer_value": str(dataset["answer_value"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_marker_layer_lifecycle(
            task=self,
            instance_seed=instance_seed,
            params=params,
            max_attempts=max_attempts,
            default_query_id=DEFAULT_QUERY_ID,
            prompt_query_key="marker_region_extremum_label",
            answer_type="string",
            question_format="map_marker_query",
            build_dataset=self._construct_extremum_dataset,
            bind_objective=self._bind_extremum_objective,
        )
