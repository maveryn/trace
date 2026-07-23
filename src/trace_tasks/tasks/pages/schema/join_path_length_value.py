"""Schema task for finding the shortest join path length between two tables."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle
from .shared.annotations import round_segment


TASK_ID = "task_pages__schema__join_path_length_value"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_NAMESPACE = "pages.schema.join_path_length_value"
PROMPT_QUERY_KEY = "join_path_length_value"
PATH_LENGTH_SUPPORT = (1, 2, 3, 4)
_RELATION_LABELS = (
    "joins_to",
    "references",
    "links_to",
    "maps_to",
    "relates_to",
    "uses",
    "connects_to",
    "depends_on",
    "belongs_to",
)


def _resolve_path_length(instance_seed: int, params: Mapping[str, Any]) -> int:
    explicit = params.get("join_path_length")
    if explicit is not None:
        value = int(explicit)
        if value not in set(PATH_LENGTH_SUPPORT):
            raise ValueError(f"join_path_length must be one of {list(PATH_LENGTH_SUPPORT)}")
        return int(value)
    return int(PATH_LENGTH_SUPPORT[int(abs(int(instance_seed))) % len(PATH_LENGTH_SUPPORT)])


def _resolve_context_id(instance_seed: int, params: Mapping[str, Any]) -> str:
    context_ids = tuple(sorted(str(value) for value in _lifecycle._CONTEXTS))
    explicit = params.get("context_id")
    if explicit is not None:
        if str(explicit) not in set(context_ids):
            raise ValueError(f"unsupported schema context_id: {explicit}")
        return str(explicit)
    index = int(hash64(int(instance_seed), f"{TASK_NAMESPACE}.context")) % len(context_ids)
    return str(context_ids[index])


def _relation_label(index: int) -> str:
    return str(_RELATION_LABELS[int(index) % len(_RELATION_LABELS)])


def _build_relation_plan(
    *,
    instance_seed: int,
    context_id: str,
    path_length: int,
) -> tuple[tuple[tuple[str, str, str], ...], tuple[str, ...], tuple[str, ...]]:
    """Build a unique shortest path plus safe distractor relationships."""

    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.relation_plan")
    table_labels = [str(label) for label in _lifecycle._CONTEXTS[str(context_id)]["tables"].keys()]
    if len(table_labels) < int(path_length) + 1:
        raise ValueError("schema context does not have enough tables for requested path length")
    rng.shuffle(table_labels)
    path_tables = tuple(table_labels[: int(path_length) + 1])
    remaining = tuple(table_labels[int(path_length) + 1 :])

    relation_specs: list[tuple[str, str, str]] = []
    for index in range(int(path_length)):
        relation_specs.append((path_tables[index], path_tables[index + 1], _relation_label(index)))

    extra_specs: list[tuple[str, str, str]] = []
    if remaining:
        attachment = path_tables[max(0, min(len(path_tables) - 1, int(path_length) // 2))]
        extra_specs.append((remaining[0], attachment, _relation_label(len(relation_specs))))
        for offset, (source, target) in enumerate(zip(remaining, remaining[1:]), start=1):
            extra_specs.append((source, target, _relation_label(len(relation_specs) + offset)))
        if len(remaining) >= 3:
            extra_specs.append((remaining[0], remaining[2], _relation_label(len(relation_specs) + len(extra_specs))))
        if len(remaining) >= 4:
            extra_specs.append((remaining[1], remaining[3], _relation_label(len(relation_specs) + len(extra_specs))))

    target_total = min(9, max(5, int(path_length) + 4))
    relation_specs.extend(extra_specs[: max(0, int(target_total) - len(relation_specs))])
    path_relation_ids = tuple(f"r{index}" for index in range(int(path_length)))
    return tuple(relation_specs), tuple(path_tables), path_relation_ids


def _build_join_path_case(instance_seed: int, params: Mapping[str, Any]):
    path_length = _resolve_path_length(int(instance_seed), params)
    context_id = _resolve_context_id(int(instance_seed), params)
    relation_specs, path_tables, path_relationship_ids = _build_relation_plan(
        instance_seed=int(instance_seed),
        context_id=str(context_id),
        path_length=int(path_length),
    )
    task_params = {**dict(params), "context_id": str(context_id)}
    case = _lifecycle.build_schema_case(
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=TASK_NAMESPACE,
        table_count_min=8,
        table_count_max=8,
        relationship_count_min=max(4, int(path_length)),
        relationship_count_max=9,
        explicit_relation_specs=relation_specs,
    )
    return case, {
        "path_length": int(path_length),
        "path_table_labels": tuple(str(label) for label in path_tables),
        "path_relationship_ids": tuple(str(value) for value in path_relationship_ids),
    }


def _table_id_by_label(tables: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(table["label"]): str(table["table_id"]) for table in tables}


def _bind_join_path_length(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: _lifecycle.SchemaCase,
    rendered: _lifecycle.RenderedSchema,
):
    """Bind the unique relationship path connecting two named tables."""

    del selected_branch, branch_probabilities
    path_length = int(case.relationships[0].get("_path_length", 0))
    path_table_labels = tuple(str(value) for value in case.relationships[0].get("_path_table_labels", ()))
    path_relationship_ids = tuple(str(value) for value in case.relationships[0].get("_path_relationship_ids", ()))
    if not path_table_labels or not path_relationship_ids:
        raise ValueError("join path metadata missing from schema case")

    table_ids = _table_id_by_label(case.tables)
    source_label = str(path_table_labels[0])
    target_label = str(path_table_labels[-1])
    path_table_ids = tuple(table_ids[str(label)] for label in path_table_labels)
    segments = [
        round_segment(rendered.render_map["relationship_point_pairs_px"][str(relationship_id)])
        for relationship_id in path_relationship_ids
    ]
    annotation_ids = [
        f"relationship_segment:{relationship_id}"
        for relationship_id in path_relationship_ids
    ]
    return (
        _lifecycle.SchemaPromptBinding(
            prompt_query_key=PROMPT_QUERY_KEY,
            dynamic_slots={
                "source_table_label": source_label,
                "target_table_label": target_label,
            },
        ),
        _lifecycle.segment_set_answer_binding(
            answer_value=int(path_length),
            segments=segments,
            annotation_ids=annotation_ids,
            extra_params={
                "answer": int(path_length),
                "source_table_label": source_label,
                "target_table_label": target_label,
                "source_table_id": str(path_table_ids[0]),
                "target_table_id": str(path_table_ids[-1]),
                "path_length": int(path_length),
                "path_table_labels": list(path_table_labels),
                "path_table_ids": list(path_table_ids),
                "path_relationship_ids": list(path_relationship_ids),
                "annotation_relationship_ids": list(path_relationship_ids),
                "answer_support": list(PATH_LENGTH_SUPPORT),
                "context_bbox_ids": {
                    "source_table": f"table:{path_table_ids[0]}",
                    "target_table": f"table:{path_table_ids[-1]}",
                },
            },
        ),
    )


def _attach_path_metadata(case: _lifecycle.SchemaCase, path_meta: Mapping[str, Any]) -> _lifecycle.SchemaCase:
    """Attach task-local path metadata to copied relationship records."""

    from dataclasses import replace

    relationships = []
    for relationship in case.relationships:
        updated = dict(relationship)
        if str(updated["relationship_id"]) == "r0":
            updated["_path_length"] = int(path_meta["path_length"])
            updated["_path_table_labels"] = tuple(str(value) for value in path_meta["path_table_labels"])
            updated["_path_relationship_ids"] = tuple(str(value) for value in path_meta["path_relationship_ids"])
        relationships.append(updated)
    return replace(case, relationships=tuple(relationships))


@register_task
class PagesSchemaJoinPathLengthValueTask:
    """Return the shortest relationship-path length between two tables."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
        )
        case, path_meta = _build_join_path_case(int(instance_seed), task_params)
        case = _attach_path_metadata(case, path_meta)
        return _lifecycle.render_bound_schema(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case=case,
            binding_factory=_bind_join_path_length,
            question_format="schema_join_path_length_value",
            source_query_name=PROMPT_QUERY_KEY,
        )


__all__ = [
    "PATH_LENGTH_SUPPORT",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesSchemaJoinPathLengthValueTask",
]
