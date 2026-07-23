"""Choose a section point on a coordinate-plane segment."""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register_task
from ...shared.fixed_query import select_task_query_id

from ._lifecycle import CoordinateAlgebraObjective, run_coordinate_algebra_entry

SECTION_POINT_TASK_ID = "task_geometry__coordinate_plane__section_point_label"
TASK_ID = "task_geometry__coordinate_plane__section_point_label"
SCENE_ID = "coordinate_plane"
SECTION_POINT_QUERY_IDS = ("one_third_from_p_to_q", "two_thirds_from_p_to_q")
SUPPORTED_QUERY_IDS = SECTION_POINT_QUERY_IDS
SCENE_KEY = "coordinate_algebra_section_scene"
SEMANTIC_OPERATION_BY_ID = {
    "one_third_from_p_to_q": "section_one_third",
    "two_thirds_from_p_to_q": "section_two_thirds",
}


def _prepare_section_point_objective(
    selected_query: str,
    _query_probabilities: Mapping[str, float],
    _task_params: Mapping[str, Any],
) -> CoordinateAlgebraObjective:
    """Bind the selected section-ratio query to coordinate section semantics."""

    return CoordinateAlgebraObjective(
        semantic_operation=str(SEMANTIC_OPERATION_BY_ID[str(selected_query)]),
        prompt_query_key=str(selected_query),
        scene_key=SCENE_KEY,
    )


@register_task
class GeometryCoordinateSectionPointLabelTask:
    """Choose the candidate point at the requested one-third or two-third position."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    prepare_objective = staticmethod(_prepare_section_point_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a segment-section candidate selection scene."""

        task_params = dict(params)
        task_params["algebra_candidate_count"] = 4
        if "winner_label" not in task_params and "answer_label" not in task_params:
            selected_query, _, _ = select_task_query_id(
                instance_seed=int(instance_seed),
                params=task_params,
                supported_query_ids=SUPPORTED_QUERY_IDS,
                default_query_id=SUPPORTED_QUERY_IDS[0],
                task_id=TASK_ID,
            )
            query_offset = 1 if str(selected_query) == "two_thirds_from_p_to_q" else 0
            task_params["winner_label"] = ("A", "B", "C", "D")[
                (int(instance_seed) + int(query_offset)) % 4
            ]
        return run_coordinate_algebra_entry(self, instance_seed, params=task_params, max_attempts=max_attempts)
