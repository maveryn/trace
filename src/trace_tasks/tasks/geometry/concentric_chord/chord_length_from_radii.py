from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import ConcentricChordObjectivePlan, prepare_concentric_chord_parts
from .shared.defaults import SCENE_ID
from .shared.measurements import (
    chord_length_from_case,
    tangent_chord_diagram_spec,
)
from .shared.sampling import (
    group_concentric_chord_cases_by_answer,
    select_answer_balanced_concentric_chord_case,
)

TASK_ID = "task_geometry__concentric_chord__chord_length_from_radii"
INTERNAL_QUERY_ID = "chord_length_from_radii"
SUPPORTED_QUERY_IDS = ("single",)

_CASES_BY_ANSWER = group_concentric_chord_cases_by_answer(
    answer_fn=chord_length_from_case,
)


def _prepare_chord_length_objective(*, instance_seed, task_params, selected_query, query_probabilities):
    """Bind the chord-length answer before rendering."""

    case, case_index, answer_probabilities = select_answer_balanced_concentric_chord_case(
        answer_cases=_CASES_BY_ANSWER,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}.case",
    )
    answer = chord_length_from_case(case)
    return ConcentricChordObjectivePlan(
        spec=tangent_chord_diagram_spec(
            case,
            answer=answer,
            inner_radius_label=f"r={case.inner_radius}",
            chord_label="c=?",
            formula_family="chord_length_from_radii",
            unknown_measure="chord_length",
        ),
        case_index=int(case_index),
        answer_probabilities=dict(answer_probabilities),
        prompt_query_key=INTERNAL_QUERY_ID,
        random_namespace=f"{TASK_ID}.render",
    )


@register_task
class GeometryConcentricChordLengthFromRadiiTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = "single"
    prepare_objective = staticmethod(_prepare_chord_length_objective)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate a chord-length instance from a task-bound Pythagorean case."""

        parts = prepare_concentric_chord_parts(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
        answer_gt = TypedValue(type="integer", value=int(parts.answer_value))
        annotation_gt = TypedValue(
            type=parts.annotation_artifacts.annotation_type,
            value=parts.annotation_artifacts.value,
        )
        trace_payload = dict(parts.trace_payload)
        return TaskOutput(
            prompt=parts.prompt,
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=parts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=parts.task_versions,
            scene_id=parts.scene_id,
            query_id=parts.selected_query,
            prompt_variants=parts.prompt_variants,
        )
