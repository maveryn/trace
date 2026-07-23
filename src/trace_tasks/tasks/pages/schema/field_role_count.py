"""Schema task for counting field rows by role in one table."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle
from .shared.annotations import bbox_set_projection, round_box


TASK_ID = "task_pages__schema__field_role_count"
SUPPORTED_QUERY_IDS = ("all_field_count", "attribute_field_count")
TASK_NAMESPACE = "pages.schema.field_role_count"


def _build_field_role_case(instance_seed: int, params: Mapping[str, Any]):
    return _lifecycle.build_schema_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=TASK_NAMESPACE,
        table_count_min=6,
        table_count_max=8,
    )


def _choose_table_for_count(
    *,
    instance_seed: int,
    selected_branch: str,
    tables,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if str(selected_branch) == "attribute_field_count":
        target_count = 2 + (abs(int(instance_seed)) % 5)
        role_description = "ordinary attribute fields"

        def field_selector(table):
            return [field for field in table["fields"] if str(field["role"]) == "ATTR"]

    else:
        target_count = 3 + (abs(int(instance_seed)) % 7)
        role_description = "fields"

        def field_selector(table):
            return list(table["fields"])

    ranked = sorted(
        tables,
        key=lambda candidate: (
            abs(len(field_selector(candidate)) - int(target_count)),
            str(candidate["label"]),
        ),
    )
    best_distance = abs(len(field_selector(ranked[0])) - int(target_count))
    eligible = [
        table
        for table in ranked
        if abs(len(field_selector(table)) - int(target_count)) == best_distance
    ]
    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.table")
    table = dict(eligible[int(rng.randrange(max(1, len(eligible)))) % len(eligible)])
    return table, field_selector(table), role_description


def _bind_field_role_count(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: _lifecycle.SchemaCase,
    rendered: _lifecycle.RenderedSchema,
):
    """Bind counted field-row boxes for the selected field-role predicate."""

    del branch_probabilities
    table, fields, role_description = _choose_table_for_count(
        instance_seed=int(instance_seed),
        selected_branch=str(selected_branch),
        tables=case.tables,
    )
    field_ids = [str(field["field_id"]) for field in fields]
    annotation_ids = [f"field:{field_id}" for field_id in field_ids]
    boxes = [
        round_box(rendered.render_map["field_bboxes_px"][field_id])
        for field_id in field_ids
    ]
    projection = bbox_set_projection(boxes=boxes, ids=annotation_ids)
    return (
        _lifecycle.SchemaPromptBinding(
            prompt_query_key=str(selected_branch),
            dynamic_slots={"table_label": str(table["label"])},
        ),
        _lifecycle.SchemaAnswerBinding(
            answer_gt=TypedValue(type="integer", value=int(len(fields))),
            annotation_gt=TypedValue(type="bbox_set", value=list(boxes)),
            annotation_projection=projection,
            annotation_ids=annotation_ids,
            witness_symbolic={
                "type": "bbox_id_set",
                "ids": list(annotation_ids),
                "bboxes": list(boxes),
            },
            supporting_bbox_ids=annotation_ids,
            extra_params={
                "table_id": str(table["table_id"]),
                "table_label": str(table["label"]),
                "field_role_description": str(role_description),
                "answer": int(len(fields)),
                "annotation_field_ids": list(field_ids),
            },
        ),
    )


@register_task
class PagesSchemaFieldRoleCountTask:
    """Count field rows in one named table."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
        )
        case = _build_field_role_case(int(instance_seed), task_params)
        return _lifecycle.render_bound_schema(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case=case,
            binding_factory=_bind_field_role_count,
            question_format="schema_field_role_count",
            source_query_name=str(selected_branch),
        )


__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesSchemaFieldRoleCountTask",
]
