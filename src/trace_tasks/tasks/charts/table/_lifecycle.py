"""Private neutral materialization helpers for styled table chart tasks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.table.shared.annotations import (
    annotation_value_from_projection,
    boxed_map_projection,
    boxed_projection,
    boxed_set_map_projection,
    boxed_set_projection,
    cell_box,
    cell_boxes,
    column_box,
)
from trace_tasks.tasks.charts.shared.grid.geometry import bbox_union
from trace_tasks.tasks.charts.table.shared.defaults import (
    GENERATION_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_DEFAULTS,
)
from trace_tasks.tasks.charts.table.shared.rendering import (
    render_table_scene,
    resolve_table_render_params,
    table_render_style_spec,
)
from trace_tasks.tasks.charts.table.shared.prompts import (
    ANNOTATION_HINT_CELL_SET,
    ANNOTATION_HINT_COLUMN_BOX,
    ANNOTATION_HINT_FILTER_MAP,
    ANNOTATION_HINT_RANK_CELL,
    ANNOTATION_HINT_TEMPORAL,
    ANNOTATION_HINT_TEMPORAL_ROW_SPAN_MAP,
    ANSWER_HINT_COUNT,
    ANSWER_HINT_INTEGER,
    ANSWER_HINT_ROW_LABEL,
    COUNTING_BUNDLE_ID,
    RANKING_BUNDLE_ID,
    SCENE_KEY_COUNTING,
    SCENE_KEY_RANKING,
    SCENE_KEY_STATISTICS,
    SCENE_KEY_TEMPORAL,
    STATISTICS_BUNDLE_ID,
    TASK_KEY_COUNTING,
    TASK_KEY_FILTERED,
    TASK_KEY_RANKING,
    TASK_KEY_SUMMARY,
    TASK_KEY_TEMPORAL,
    TEMPORAL_BUNDLE_ID,
    object_description,
    render_prompt_artifacts,
)
from trace_tasks.tasks.charts.table.shared.sampling import (
    build_counting_value_dataset_for_variant,
    build_ranking_label_dataset_for_variant,
    build_statistics_filtered_subset_dataset_for_variant,
    build_summary_value_dataset_for_variant,
    build_temporal_value_dataset_for_variant,
    render_table_filter_condition,
    table_value_cell_id,
)
from trace_tasks.tasks.charts.table.shared.state import SCENE_ID, SUPPORTED_TABLE_SCENE_VARIANTS, TableDefaults
from trace_tasks.tasks.charts.table.shared.styles import sample_table_font_family, table_font_asset_metadata
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise


@dataclass(frozen=True)
class TableTaskPlan:
    """Task-owned semantic plan consumed by neutral table rendering."""

    row_labels: Sequence[str]
    column_headers: Sequence[str]
    values_by_row: Mapping[str, Mapping[str, Any]]
    scene_variant: str
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    annotation_kind: str
    annotation_cell_ids: Sequence[str]
    annotation_cell_id: str
    annotation_column_header: str
    annotation_cell_id_map: Mapping[str, Sequence[str]]
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    witness_type: str
    witness_calculation: Mapping[str, Any]
    render_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedTableTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def table_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for table-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.table.retry", int(attempt)))
    )


def select_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    """Select one visual table variant without public task/query routing."""

    explicit = params.get("scene_variant")
    supported = tuple(str(value) for value in SUPPORTED_TABLE_SCENE_VARIANTS)
    if explicit is not None:
        value = str(explicit)
        if value not in set(supported):
            raise ValueError(f"unsupported table scene_variant: {value}")
        return value, {item: (1.0 if item == value else 0.0) for item in supported}
    weights_raw = GENERATION_DEFAULTS.get("scene_variant_weights", {})
    weights = {
        item: float(weights_raw.get(item, 1.0)) if isinstance(weights_raw, Mapping) else 1.0
        for item in supported
    }
    total = float(sum(max(0.0, value) for value in weights.values()))
    if total <= 0:
        weights = {item: 1.0 for item in supported}
        total = float(len(supported))
    probabilities = {item: float(max(0.0, weights[item]) / total) for item in supported}
    rng = spawn_rng(int(instance_seed), "charts.table.scene_variant")
    threshold = float(rng.random())
    cumulative = 0.0
    selected = supported[-1]
    for item in supported:
        cumulative += float(probabilities[item])
        if threshold <= cumulative:
            selected = item
            break
    return str(selected), dict(probabilities)


def _project_annotation(plan: TableTaskPlan, rendered) -> dict[str, Any]:
    if plan.annotation_kind == "bbox_set":
        return boxed_set_projection(cell_boxes(rendered, plan.annotation_cell_ids))
    if plan.annotation_kind == "bbox":
        if plan.annotation_cell_id:
            return boxed_projection(cell_box(rendered, str(plan.annotation_cell_id)))
        return boxed_projection(column_box(rendered, str(plan.annotation_column_header)))
    if plan.annotation_kind == "bbox_set_map":
        return boxed_set_map_projection(
            {
                str(key): cell_boxes(rendered, [str(cell_id) for cell_id in cell_ids])
                for key, cell_ids in plan.annotation_cell_id_map.items()
            }
        )
    if plan.annotation_kind == "bbox_map":
        return boxed_map_projection(
            {
                str(key): bbox_union(cell_boxes(rendered, [str(cell_id) for cell_id in cell_ids]))
                for key, cell_ids in plan.annotation_cell_id_map.items()
            }
        )
    raise ValueError(f"unsupported table annotation kind: {plan.annotation_kind}")


def _values_by_row_trace(values_by_row: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row_label, row_values in values_by_row.items():
        output[str(row_label)] = {}
        for header, value in row_values.items():
            output[str(row_label)][str(header)] = int(value) if isinstance(value, int) and not isinstance(value, bool) else str(value)
    return output


def materialize_table_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: TableTaskPlan,
) -> MaterializedTableTask:
    """Render one task-owned table plan and build common trace sections."""

    defaults = TableDefaults()
    render_params = resolve_table_render_params(
        params,
        render_defaults=plan.render_defaults,
        defaults=defaults,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="table",
        render_params=render_params,
    )
    table_font_family = sample_table_font_family(
        instance_seed=int(instance_seed),
        namespace="charts.table.font",
        params=params,
    )
    with temporary_default_font_family(str(table_font_family)):
        rendered = render_table_scene(
            background,
            scene_variant=str(plan.scene_variant),
            row_labels=list(plan.row_labels),
            column_headers=list(plan.column_headers),
            values_by_row=dict(plan.values_by_row),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    projected_annotation = _project_annotation(plan, rendered)
    annotation_gt = annotation_value_from_projection(projected_annotation)
    cell_bbox_map = {
        str(cell_trace["cell_id"]): list(cell_trace["bbox_px"])
        for cell_trace in rendered.cell_traces
    }
    relation_fields = {
        "query_id": str(selected_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(plan.scene_variant),
        **dict(plan.relations),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"table_{str(plan.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relation_fields),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=plan.prompt_artifacts,
            query_id=str(selected_query_id),
            params=relation_fields,
        ),
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(plan.scene_variant),
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "font_assets": table_font_asset_metadata(str(table_font_family)),
            "table_bbox_px": list(rendered.table_bbox_px),
            "table_style": table_render_style_spec(render_params),
            "text_style": {
                "label_font_size_px": int(render_params.label_font_size_px),
                "value_font_size_px": int(render_params.value_font_size_px),
            },
            "grid_style": {
                "border_width_px": int(render_params.border_width_px),
                "grid_width_px": int(render_params.grid_width_px),
            },
        },
        "render_map": {
            "image_id": "img0",
            "table_bbox_px": list(rendered.table_bbox_px),
            "numeric_table_region_bbox_px": list(rendered.numeric_table_region_bbox),
            "row_region_bboxes_px": dict(rendered.row_region_bboxes),
            "column_region_bboxes_px": dict(rendered.column_region_bboxes),
            "row_label_bboxes_px": dict(rendered.row_label_bboxes),
            "header_bboxes_px": dict(rendered.header_bboxes),
            "cell_bboxes_px": dict(cell_bbox_map),
        },
        "execution_trace": {
            "query_id": str(selected_query_id),
            "scene_variant": str(plan.scene_variant),
            "answer_value": plan.answer_value,
            "row_labels": [str(label) for label in plan.row_labels],
            "column_headers": [str(header) for header in plan.column_headers],
            "values_by_row": _values_by_row_trace(plan.values_by_row),
            "question_format": str(plan.question_format),
            **dict(relation_fields),
        },
        "verifier_spec": {
            "answer_type": str(plan.answer_gt.type),
            "annotation_type": str(annotation_gt.type),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "value": annotation_gt.value,
            "calculation": dict(plan.witness_calculation),
        },
        "projected_annotation": dict(projected_annotation),
    }
    return MaterializedTableTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_table_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    build_plan: Callable[[int, Mapping[str, Any], str], TableTaskPlan],
) -> MaterializedTableTask:
    """Retry task-owned table plan construction, then render the payload."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = table_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), {**dict(params), "_attempt_index": int(attempt)}, str(selected_query_id))
            return materialize_table_plan(
                instance_seed=int(attempt_seed),
                params={**dict(params), "_attempt_index": int(attempt)},
                selected_query_id=str(selected_query_id),
                query_probabilities=query_probabilities,
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize table plan: {last_error}")


def table_task_output_fields(materialized: MaterializedTableTask) -> dict[str, Any]:
    """Return final-output fields from an already materialized table payload."""

    return {
        "prompt": materialized.prompt,
        "answer_gt": materialized.answer_gt,
        "annotation_gt": materialized.annotation_gt,
        "image": materialized.image,
        "image_id": "img0",
        "trace_payload": materialized.trace_payload,
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": materialized.query_id,
        "prompt_variants": materialized.prompt_variants,
    }


def run_table_task_from_public_class(
    task: Any,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Callable[[int, Mapping[str, Any], str], TableTaskPlan],
) -> TaskOutput:
    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=task.default_query_id,
        task_id=task.task_id,
    )
    materialized = materialize_table_plan_with_retries(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        query_probabilities=query_probabilities,
        build_plan=build_plan,
    )
    return TaskOutput(**table_task_output_fields(materialized))


def table_default_groups(public_task_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return task-specific config groups for one public table task."""

    generation, rendering, prompt = split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
        task_id=str(public_task_id),
    )
    return dict(generation), dict(rendering), dict(prompt)


def cell_id_for_row_and_column(*, data_row_index: int, rendered_column_index: int) -> str:
    """Return a value-cell id for one zero-indexed data row and rendered data column."""

    return table_value_cell_id(
        data_row_index=int(data_row_index),
        numeric_column_index=int(rendered_column_index),
    )


def _table_build_context(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, float]]:
    """Resolve config and visual-scene variation shared by table objectives."""

    generation_defaults, rendering_defaults, _ = table_default_groups(str(public_task_id))
    scene_variant, scene_probabilities = select_scene_variant(params, instance_seed=int(instance_seed))
    return dict(generation_defaults), dict(rendering_defaults), str(scene_variant), dict(scene_probabilities)


def _ordinal(value: int) -> str:
    """Format a positive integer rank for prompt slots."""

    number = int(value)
    if 10 <= (number % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _cell_ids_for_rows(*, dataset: Mapping[str, Any], row_key: str, column_index: int) -> tuple[str, ...]:
    """Bind row indices from one dataset to rendered table cell ids."""

    return tuple(
        cell_id_for_row_and_column(
            data_row_index=int(row_index),
            rendered_column_index=int(column_index),
        )
        for row_index in dataset[str(row_key)]
    )


def build_table_numeric_filter_count_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    operation: str,
    prompt_key: str,
    program_code: str,
    question_format: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build a count objective over numeric table cells selected by one filter predicate."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_counting_value_dataset_for_variant(
        operation=str(operation),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    dynamic_slots = {
        "object_description": object_description(scene_variant),
        "query_column": str(dataset["query_column"]),
    }
    relation_extra: dict[str, Any]
    if str(operation) in {"above_threshold", "below_threshold"}:
        dynamic_slots["threshold_value"] = str(int(dataset["threshold_value"]))
        relation_extra = {"threshold_value": int(dataset["threshold_value"])}
    else:
        dynamic_slots["interval_min"] = str(int(dataset["interval_min"]))
        dynamic_slots["interval_max"] = str(int(dataset["interval_max"]))
        relation_extra = {
            "interval_min": int(dataset["interval_min"]),
            "interval_max": int(dataset["interval_max"]),
        }
    prompt = render_prompt_artifacts(
        bundle_id=COUNTING_BUNDLE_ID,
        scene_key=SCENE_KEY_COUNTING,
        task_key=TASK_KEY_COUNTING,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_COUNT,
        annotation_hint=ANNOTATION_HINT_CELL_SET,
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    cell_ids = _cell_ids_for_rows(
        dataset=dataset,
        row_key="matching_row_indices",
        column_index=int(dataset["query_column_index"]),
    )
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": str(operation),
        "query_column": str(dataset["query_column"]),
        "query_column_index": int(dataset["query_column_index"]),
        "matching_row_indices": [int(index) for index in dataset["matching_row_indices"]],
        "matching_row_labels": [str(label) for label in dataset["matching_row_labels"]],
        "supporting_cell_ids": [str(cell_id) for cell_id in cell_ids],
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "value_range": list(dataset["value_range"]),
        "scene_variant_probabilities": dict(scene_probabilities),
        **relation_extra,
    }
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        answer_value=int(dataset["answer_value"]),
        question_format=str(question_format),
        annotation_kind="bbox_set",
        annotation_cell_ids=tuple(str(cell_id) for cell_id in cell_ids),
        annotation_cell_id="",
        annotation_column_header="",
        annotation_cell_id_map={},
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox_set",
        witness_calculation={"operation": "count", "matching_row_indices": list(dataset["matching_row_indices"])},
        render_defaults=rendering_defaults,
    )


def build_table_category_membership_count_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    prompt_key: str,
    program_code: str,
    question_format: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build a count objective over one categorical table column."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_counting_value_dataset_for_variant(
        operation="category_membership",
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    prompt = render_prompt_artifacts(
        bundle_id=COUNTING_BUNDLE_ID,
        scene_key=SCENE_KEY_COUNTING,
        task_key=TASK_KEY_COUNTING,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_COUNT,
        annotation_hint=ANNOTATION_HINT_CELL_SET,
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values={
            "object_description": object_description(scene_variant),
            "query_column": str(dataset["query_column"]),
            "target_category": str(dataset["target_category"]),
        },
        instance_seed=int(instance_seed),
    )
    cell_ids = _cell_ids_for_rows(dataset=dataset, row_key="matching_row_indices", column_index=0)
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": "category_membership",
        "query_column": str(dataset["query_column"]),
        "query_column_index": int(dataset["query_column_index"]),
        "target_category": str(dataset["target_category"]),
        "matching_row_indices": [int(index) for index in dataset["matching_row_indices"]],
        "matching_row_labels": [str(label) for label in dataset["matching_row_labels"]],
        "supporting_cell_ids": [str(cell_id) for cell_id in cell_ids],
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "column_count": int(dataset["column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "value_range": list(dataset["value_range"]),
        "category_values_by_row": dict(dataset["category_values_by_row"]),
        "scene_variant_probabilities": dict(scene_probabilities),
    }
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        answer_value=int(dataset["answer_value"]),
        question_format=str(question_format),
        annotation_kind="bbox_set",
        annotation_cell_ids=tuple(str(cell_id) for cell_id in cell_ids),
        annotation_cell_id="",
        annotation_column_header="",
        annotation_cell_id_map={},
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox_set",
        witness_calculation={"operation": "count", "matching_row_indices": list(dataset["matching_row_indices"])},
        render_defaults=rendering_defaults,
    )


def build_table_rank_label_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    operation: str,
    prompt_key: str,
    rank_direction: str,
    program_code: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build a row-label answer for one ranked numeric table column."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_ranking_label_dataset_for_variant(
        operation=str(operation),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    prompt = render_prompt_artifacts(
        bundle_id=RANKING_BUNDLE_ID,
        scene_key=SCENE_KEY_RANKING,
        task_key=TASK_KEY_RANKING,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_ROW_LABEL,
        annotation_hint=ANNOTATION_HINT_RANK_CELL,
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values={
            "object_description": object_description(scene_variant),
            "query_column": str(dataset["query_column"]),
            "query_rank": _ordinal(int(dataset["query_rank"])),
            "rank_direction": str(rank_direction),
        },
        instance_seed=int(instance_seed),
    )
    answer_cell_id = cell_id_for_row_and_column(
        data_row_index=int(dataset["answer_row_index"]),
        rendered_column_index=int(dataset["query_column_index"]),
    )
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": str(operation),
        "rank_direction": str(rank_direction),
        "query_rank": int(dataset["query_rank"]),
        "query_column": str(dataset["query_column"]),
        "query_column_index": int(dataset["query_column_index"]),
        "answer_row_label": str(dataset["answer_row_label"]),
        "answer_row_index": int(dataset["answer_row_index"]),
        "answer_value": int(dataset["answer_value"]),
        "supporting_cell_id": str(answer_cell_id),
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "value_range": list(dataset["value_range"]),
        "scene_variant_probabilities": dict(scene_probabilities),
    }
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="string", value=str(dataset["answer_row_label"])),
        answer_value=str(dataset["answer_row_label"]),
        question_format="table_column_rank_label",
        annotation_kind="bbox",
        annotation_cell_ids=(),
        annotation_cell_id=str(answer_cell_id),
        annotation_column_header="",
        annotation_cell_id_map={},
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox",
        witness_calculation={"operation": str(operation), "answer_value": int(dataset["answer_value"])},
        render_defaults=rendering_defaults,
    )


def build_table_column_summary_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    operation: str,
    prompt_key: str,
    program_code: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build one numeric aggregate answer over a full table column."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_summary_value_dataset_for_variant(
        operation=str(operation),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    prompt = render_prompt_artifacts(
        bundle_id=STATISTICS_BUNDLE_ID,
        scene_key=SCENE_KEY_STATISTICS,
        task_key=TASK_KEY_SUMMARY,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_INTEGER,
        annotation_hint=ANNOTATION_HINT_COLUMN_BOX,
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values={
            "object_description": object_description(scene_variant),
            "query_column": str(dataset["query_column"]),
        },
        instance_seed=int(instance_seed),
    )
    query_values = [
        int(dataset["values_by_row"][str(row_label)][str(dataset["query_column"])])
        for row_label in dataset["row_labels"]
    ]
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": str(operation),
        "query_column": str(dataset["query_column"]),
        "query_column_index": int(dataset["query_column_index"]),
        "query_values": [int(value) for value in query_values],
        "answer_value": int(dataset["answer_value"]),
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "value_range": list(dataset["value_range"]),
        "scene_variant_probabilities": dict(scene_probabilities),
    }
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        answer_value=int(dataset["answer_value"]),
        question_format="table_column_summary_value",
        annotation_kind="bbox",
        annotation_cell_ids=(),
        annotation_cell_id="",
        annotation_column_header=str(dataset["query_column"]),
        annotation_cell_id_map={},
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox",
        witness_calculation={"operation": str(operation), "query_values": [int(value) for value in query_values]},
        render_defaults=rendering_defaults,
    )


def build_table_filtered_mean_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    filter_variant: str,
    prompt_key: str,
    program_code: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build a mean answer over target cells whose rows pass one filter predicate."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_statistics_filtered_subset_dataset_for_variant(
        operation="mean",
        params={**dict(params), "filter_variant": str(filter_variant)},
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    prompt = render_prompt_artifacts(
        bundle_id=STATISTICS_BUNDLE_ID,
        scene_key=SCENE_KEY_STATISTICS,
        task_key=TASK_KEY_FILTERED,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_INTEGER,
        annotation_hint=ANNOTATION_HINT_FILTER_MAP,
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values={
            "object_description": object_description(scene_variant),
            "query_filter_column": str(dataset["filter_column"]),
            "query_target_column": str(dataset["target_column"]),
            "filter_condition": render_table_filter_condition(dataset),
        },
        instance_seed=int(instance_seed),
    )
    filter_cell_ids = _cell_ids_for_rows(
        dataset=dataset,
        row_key="selected_row_indices",
        column_index=int(dataset["filter_column_index"]),
    )
    target_cell_ids = _cell_ids_for_rows(
        dataset=dataset,
        row_key="selected_row_indices",
        column_index=int(dataset["target_column_index"]),
    )
    relation_extra = (
        {"threshold_value": int(dataset["threshold_value"])}
        if str(dataset["filter_variant"]) in {"above_threshold", "below_threshold"}
        else {"interval_min": int(dataset["interval_min"]), "interval_max": int(dataset["interval_max"])}
    )
    selected_values = [
        int(dataset["values_by_row"][str(row_label)][str(dataset["target_column"])])
        for row_label in dataset["selected_row_labels"]
    ]
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": "mean",
        "filter_variant": str(dataset["filter_variant"]),
        "filter_column": str(dataset["filter_column"]),
        "filter_column_index": int(dataset["filter_column_index"]),
        "target_column": str(dataset["target_column"]),
        "target_column_index": int(dataset["target_column_index"]),
        "selected_row_indices": [int(index) for index in dataset["selected_row_indices"]],
        "selected_row_labels": [str(label) for label in dataset["selected_row_labels"]],
        "selected_row_count": int(len(dataset["selected_row_indices"])),
        "selected_row_count_range": list(dataset["selected_row_count_range"]),
        "filter_cell_ids": [str(cell_id) for cell_id in filter_cell_ids],
        "target_cell_ids": [str(cell_id) for cell_id in target_cell_ids],
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "value_range": list(dataset["value_range"]),
        "scene_variant_probabilities": dict(scene_probabilities),
        **relation_extra,
    }
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        answer_value=int(dataset["answer_value"]),
        question_format="table_filtered_column_mean",
        annotation_kind="bbox_set_map",
        annotation_cell_ids=(),
        annotation_cell_id="",
        annotation_column_header="",
        annotation_cell_id_map={
            "filter_cells": tuple(str(cell_id) for cell_id in filter_cell_ids),
            "target_cells": tuple(str(cell_id) for cell_id in target_cell_ids),
        },
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox_set_map",
        witness_calculation={"operation": "mean", "target_values": [int(value) for value in selected_values]},
        render_defaults=rendering_defaults,
    )


def build_table_temporal_two_row_plan(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    operation: str,
    prompt_key: str,
    program_code: str,
    question_format: str,
    json_example: str,
    json_example_answer_only: str,
) -> TableTaskPlan:
    """Build a two-row temporal interval objective over chronological table columns."""

    generation_defaults, rendering_defaults, scene_variant, scene_probabilities = _table_build_context(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
    )
    dataset = build_temporal_value_dataset_for_variant(
        operation=str(operation),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=generation_defaults,
        defaults=TableDefaults(),
        namespace=str(public_task_id),
    )
    query_cells = [dict(cell) for cell in dataset["query_cells"]]
    cell_ids = tuple(str(cell["cell_id"]) for cell in query_cells)
    use_row_span_map = str(operation) == "row_interval_sum_difference_abs"
    row_span_cell_ids = {
        str(dataset["query_row_label_a"]): tuple(
            str(cell["cell_id"]) for cell in query_cells if str(cell.get("row_role")) == "row_a"
        ),
        str(dataset["query_row_label_b"]): tuple(
            str(cell["cell_id"]) for cell in query_cells if str(cell.get("row_role")) == "row_b"
        ),
    }
    row_span_json_example = json.dumps(
        {
            "annotation": {
                str(dataset["query_row_label_a"]): [260, 180, 444, 220],
                str(dataset["query_row_label_b"]): [260, 236, 444, 276],
            },
            "answer": 7,
        },
        separators=(",", ":"),
    )
    prompt = render_prompt_artifacts(
        bundle_id=TEMPORAL_BUNDLE_ID,
        scene_key=SCENE_KEY_TEMPORAL,
        task_key=TASK_KEY_TEMPORAL,
        prompt_key=str(prompt_key),
        answer_hint=ANSWER_HINT_INTEGER,
        annotation_hint=ANNOTATION_HINT_TEMPORAL_ROW_SPAN_MAP if use_row_span_map else ANNOTATION_HINT_TEMPORAL,
        json_example=str(row_span_json_example if use_row_span_map else json_example),
        json_example_answer_only=str(json_example_answer_only),
        dynamic_slot_values={
            "object_description": object_description(scene_variant, temporal=True),
            "query_row_label_a": str(dataset["query_row_label_a"]),
            "query_row_label_b": str(dataset["query_row_label_b"]),
            "query_year_start": str(dataset["query_year_start"]),
            "query_year_end": str(dataset["query_year_end"]),
        },
        instance_seed=int(instance_seed),
    )
    relations = {
        "program_code": str(program_code),
        "operation": str(prompt_key),
        "semantic_operation": str(operation),
        "query_row_label_a": str(dataset["query_row_label_a"]),
        "query_row_index_a": int(dataset["query_row_index_a"]),
        "query_row_label_b": str(dataset["query_row_label_b"]),
        "query_row_index_b": int(dataset["query_row_index_b"]),
        "query_row_labels": [str(label) for label in dataset["query_row_labels"]],
        "query_years": [str(year) for year in dataset["query_years"]],
        "query_year_start": str(dataset["query_year_start"]),
        "query_year_end": str(dataset["query_year_end"]),
        "query_cells": query_cells,
        "row_interval_sums": dict(dataset["row_interval_sums"]),
        "paired_absolute_differences": [int(value) for value in dataset["paired_absolute_differences"]],
        "supporting_cell_ids": [str(cell_id) for cell_id in cell_ids],
        "supporting_cell_ids_by_row": {
            str(row_label): [str(cell_id) for cell_id in ids]
            for row_label, ids in row_span_cell_ids.items()
        },
        "row_count": int(dataset["row_count"]),
        "numeric_column_count": int(dataset["numeric_column_count"]),
        "row_count_range": list(dataset["row_count_range"]),
        "numeric_column_count_range": list(dataset["numeric_column_count_range"]),
        "interval_length_range": list(dataset["interval_length_range"]),
        "value_range": list(dataset["value_range"]),
        "scene_variant_probabilities": dict(scene_probabilities),
    }
    witness_calculation = (
        {"operation": "absolute_difference", "row_interval_sums": dict(dataset["row_interval_sums"])}
        if str(operation) == "row_interval_sum_difference_abs"
        else {
            "operation": "sum_absolute_differences",
            "paired_absolute_differences": list(dataset["paired_absolute_differences"]),
        }
    )
    return TableTaskPlan(
        row_labels=tuple(str(label) for label in dataset["row_labels"]),
        column_headers=tuple(str(header) for header in dataset["column_headers"]),
        values_by_row=dict(dataset["values_by_row"]),
        scene_variant=str(scene_variant),
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        answer_value=int(dataset["answer_value"]),
        question_format=str(question_format),
        annotation_kind="bbox_map" if use_row_span_map else "bbox_set",
        annotation_cell_ids=() if use_row_span_map else tuple(str(cell_id) for cell_id in cell_ids),
        annotation_cell_id="",
        annotation_column_header="",
        annotation_cell_id_map=row_span_cell_ids if use_row_span_map else {},
        relations=relations,
        prompt_artifacts=prompt,
        witness_type="bbox_map" if use_row_span_map else "bbox_set",
        witness_calculation=witness_calculation,
        render_defaults=rendering_defaults,
    )


__all__ = [
    "TableTaskPlan",
    "build_table_category_membership_count_plan",
    "build_table_column_summary_plan",
    "build_table_filtered_mean_plan",
    "build_table_numeric_filter_count_plan",
    "build_table_rank_label_plan",
    "build_table_temporal_two_row_plan",
    "cell_id_for_row_and_column",
    "run_table_task_from_public_class",
    "select_scene_variant",
    "table_default_groups",
]
