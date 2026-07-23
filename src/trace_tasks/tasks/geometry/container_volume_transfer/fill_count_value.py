from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import build_container_volume_result
from .shared.defaults import DOMAIN, load_container_volume_transfer_task_defaults
from .shared.annotations import CONTAINER_BBOX_ANNOTATION_KEYS
from .shared.measurements import json_answer_value, resolve_cone_fill_count, resolve_cylinder_fill_count
from .shared.relations import (
    ContainerVolumeQueryProgram,
    ContainerVolumeTaskBinding,
    resolve_container_volume_problem,
)
from .shared.sampling import CONE_FILL_CASES, CYLINDER_FILL_CASES, select_cone_fill_case, select_cylinder_fill_case

TASK_ID = "task_geometry__container_volume_transfer__fill_count_value"
TASK_ID_FILL_COUNT = TASK_ID
QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT = "cone_to_cylinder_fill_count"
QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT = "cylinder_to_cuboid_fill_count"
SUPPORTED_QUERY_IDS = (QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT, QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT)
FILL_COUNT_ANNOTATION_KEYS = CONTAINER_BBOX_ANNOTATION_KEYS
ANNOTATION_KEYS = FILL_COUNT_ANNOTATION_KEYS
TASK_BINDING = ContainerVolumeTaskBinding("fill_count_value_query", ANNOTATION_KEYS, "answer_hint_integer", "integer")
QUERY_PROGRAMS = {
    QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT: ContainerVolumeQueryProgram(
        select_cone_fill_case,
        resolve_cone_fill_count,
        CONE_FILL_CASES,
        "fill_count",
        "cone_fill_count",
    ),
    QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT: ContainerVolumeQueryProgram(
        select_cylinder_fill_case,
        resolve_cylinder_fill_count,
        CYLINDER_FILL_CASES,
        "fill_count",
        "cylinder_fill_count",
    ),
}


def _build_problem(*, selected_query, query_probabilities, instance_seed, params):
    program = QUERY_PROGRAMS.get(str(selected_query))
    if program is None:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")
    return resolve_container_volume_problem(
        program=program,
        query_probabilities=query_probabilities,
        instance_seed=int(instance_seed),
        params=params,
        random_namespace=f"{TASK_ID}.{program.namespace_suffix}.case",
    )

@register_task
class GeometryContainerVolumeTransferFillCountValueTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Bind the selected fill-count branch and return the final task output."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT,
            task_id=TASK_ID,
        )
        problem = _build_problem(
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            instance_seed=int(instance_seed),
            params=task_params,
        )
        render_defaults, prompt_defaults = load_container_volume_transfer_task_defaults(TASK_ID)
        return build_container_volume_result(
            str(selected_query),
            str(selected_query),
            query_probabilities,
            problem,
            TASK_BINDING,
            json_answer_value(problem.answer),
            render_defaults,
            prompt_defaults,
            int(instance_seed),
            task_params,
            int(max_attempts),
            f"{TASK_ID}.render",
        )
