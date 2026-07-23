"""Count grid rows or columns satisfying a named-icon count condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ...shared.variant_sampling import resolve_variant
from ..shared.annotation import bbox_set_annotation
from ..shared.icon_scene import sort_bboxes_reading_order
from ..shared.procedural_named_icons import procedural_named_icon_display_name
from .shared.defaults import SCENE_ID, NamedGridDefaults
from .shared.metrics import (
    active_line_counts,
    axis_line_capacity,
    axis_line_count,
    column_target_counts as grid_column_target_counts,
    line_cells,
    qualifying_line_indices as matching_line_indices,
    row_target_counts as grid_row_target_counts,
)
from .shared.output import (
    build_named_grid_trace_payload,
    cells_to_trace,
    counts_to_trace,
    shape_counts_from_icons,
    task_output_prompt_variants,
)
from .shared.prompts import build_named_grid_prompt_artifacts
from .shared.rendering import render_named_grid_scene, serialize_named_grid_icon
from .shared.sampling import (
    fill_style_probability_map as resolve_fill_style_probability_map,
    fill_style_support as resolve_fill_style_support,
    grid_size_label,
    grid_size_support,
    int_bounds,
    resolve_target_shape,
    shape_support as resolve_shape_support,
    string_probability_map,
)
from .shared.state import NamedGridScenePayload
from .shared.annotations import all_line_region_bboxes as _all_line_region_bboxes, line_region_bbox as _line_region_bbox
from .shared.styles import resolve_named_grid_render_params


DOMAIN = "icons"


TASK_ID = "task_icons__named_grid__group_predicate_count"
QUERY_IDS: Tuple[str, ...] = (
    "row_at_least_shape_count",
    "column_at_least_shape_count",
    "row_exactly_shape_count",
    "column_exactly_shape_count",
    "row_no_shape_count",
    "column_no_shape_count",
)


@dataclass(frozen=True)
class _SampleSpec:
    """Symbolic named-grid line-condition sample."""

    query_id: str
    target_shape_id: str
    target_shape_name: str
    answer_count: int
    grid_rows: int
    grid_cols: int
    queried_axis: str
    condition: str
    threshold: int
    shape_ids_by_cell: Tuple[Tuple[str, ...], ...]
    counted_cells: Tuple[Tuple[int, int], ...]
    off_line_target_cells: Tuple[Tuple[int, int], ...]
    qualifying_line_indices: Tuple[int, ...]
    row_target_counts: Tuple[int, ...]
    column_target_counts: Tuple[int, ...]
    query_probabilities: Dict[str, float]
    answer_probabilities: Dict[str, float]
    grid_size_probabilities: Dict[str, float]
    threshold_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]

_DEFAULTS = NamedGridDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class IconsCountingNamedGridLineConditionCountTask:
    """Count numbered rows or columns satisfying a target-shape count condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one named-grid row/column condition-counting instance."""

        render_params = resolve_named_grid_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene: NamedGridScenePayload | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                scene = render_named_grid_scene(
                    sample=sample,  # type: ignore[arg-type]
                    instance_seed=int(instance_seed),
                    render_params=render_params,
                    params=params,
                    rng=scene_rng,
                )
                break
            except Exception as exc:  # pragma: no cover - covered by smoke tests.
                last_error = exc
                sample = None
                scene = None
        if sample is None or scene is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        qualifying_line_bboxes = tuple(
            _line_region_bbox(scene, axis=str(sample.queried_axis), line_index=int(index))
            for index in sample.qualifying_line_indices
        )
        annotation_bboxes = sort_bboxes_reading_order(qualifying_line_bboxes)
        if len(annotation_bboxes) != int(sample.answer_count):
            raise RuntimeError("rendered named-grid line-condition annotation count does not match answer")
        annotation_payload = bbox_set_annotation(annotation_bboxes)

        question_key = f"question_text_{sample.query_id}"
        line_kind_plural = "rows" if str(sample.queried_axis) == "row" else "columns"
        prompt_artifacts, _prompt_defaults = build_named_grid_prompt_artifacts(
            domain=DOMAIN,
            run_namespace=self.task_id,
            prompt_defaults_map=_PROMPT_DEFAULTS,
            question_key=str(question_key),
            question_slots={
                "target_shape_name": str(sample.target_shape_name),
                "threshold": int(sample.threshold),
            },
            annotation_slots={
                "target_shape_name": str(sample.target_shape_name),
                "line_kind": str(sample.queried_axis),
                "line_kind_plural": str(line_kind_plural),
                "threshold": int(sample.threshold),
            },
            answer_slots={"line_kind_plural": str(line_kind_plural)},
            instance_seed=int(instance_seed),
            annotation_hint_key="group_annotation_hint",
            answer_hint_key="group_answer_hint",
            json_example_key="group_json_example",
            json_example_answer_only_key="group_json_example_answer_only",
        )

        serialized_icons = [serialize_named_grid_icon(icon) for icon in scene.icons]
        shape_counts = shape_counts_from_icons(scene.icons)
        active_counts = active_line_counts(
            axis=str(sample.queried_axis),
            row_counts=sample.row_target_counts,
            column_counts=sample.column_target_counts,
        )
        qualifying_line_numbers = tuple(int(index) + 1 for index in sample.qualifying_line_indices)
        line_region_bboxes = _all_line_region_bboxes(scene, axis=str(sample.queried_axis))
        qualifying_region_map = {
            f"{sample.queried_axis}_{int(index) + 1}": [int(value) for value in _line_region_bbox(scene, axis=str(sample.queried_axis), line_index=int(index))]
            for index in sample.qualifying_line_indices
        }
        selected_cells_trace = cells_to_trace(sample.counted_cells)
        other_cells_trace = cells_to_trace(sample.off_line_target_cells)
        active_counts_trace = counts_to_trace(active_counts)
        row_counts_trace = counts_to_trace(sample.row_target_counts)
        column_counts_trace = counts_to_trace(sample.column_target_counts)
        qualifying_numbers_trace = counts_to_trace(qualifying_line_numbers)
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(sample.query_id),
            params={
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "answer_count": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "queried_axis": str(sample.queried_axis),
                "condition": str(sample.condition),
                "threshold": int(sample.threshold),
                "query_id_probabilities": dict(sample.query_probabilities),
                "answer_probabilities": dict(sample.answer_probabilities),
                "grid_size_probabilities": dict(sample.grid_size_probabilities),
                "threshold_probabilities": dict(sample.threshold_probabilities),
                "shape_id_support": list(resolve_shape_support(params, _GEN_DEFAULTS)),
                "shape_probabilities": dict(sample.shape_probabilities),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "fill_style_probabilities": dict(sample.fill_style_probabilities),
            },
        )
        trace_payload = build_named_grid_trace_payload(
            scene=scene,
            scene_kind="icons_named_grid_line_condition_count",
            entities=serialized_icons,
            relations={
                "counting_rule": "grid_line_shape_count_condition",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "queried_axis": str(sample.queried_axis),
                "condition": str(sample.condition),
                "threshold": int(sample.threshold),
                "answer_count": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "shape_counts": dict(shape_counts),
                "row_target_counts": row_counts_trace,
                "column_target_counts": column_counts_trace,
                "active_line_target_counts": active_counts_trace,
                "qualifying_line_numbers": qualifying_numbers_trace,
                "qualifying_line_regions": dict(qualifying_region_map),
                "selected_line_target_cells": selected_cells_trace,
                "other_target_cells": other_cells_trace,
            },
            query_spec=query_spec,
            render_params=render_params,
            rows=int(sample.grid_rows),
            cols=int(sample.grid_cols),
            render_map_extra={
                "line_region_bboxes_px": dict(line_region_bboxes),
                "qualifying_line_region_bboxes_px": dict(qualifying_region_map),
            },
            execution_trace={
                "scene_variant": "single_panel_named_grid",
                "query_id": str(sample.query_id),
                "question_format": "count_grid_lines_satisfying_named_shape_count_condition",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "answer": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "queried_axis": str(sample.queried_axis),
                "condition": str(sample.condition),
                "threshold": int(sample.threshold),
                "shape_ids_by_cell": [list(row) for row in sample.shape_ids_by_cell],
                "row_target_counts": row_counts_trace,
                "column_target_counts": column_counts_trace,
                "active_line_target_counts": active_counts_trace,
                "qualifying_line_indices": counts_to_trace(sample.qualifying_line_indices),
                "qualifying_line_numbers": qualifying_numbers_trace,
                "selected_line_target_cells": selected_cells_trace,
                "other_target_cells": other_cells_trace,
            },
            witness_symbolic={
                "query_id": str(sample.query_id),
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "answer": int(sample.answer_count),
                "qualifying_line_numbers": qualifying_numbers_trace,
                "qualifying_line_region_bboxes": dict(qualifying_region_map),
            },
            annotation_payload=annotation_payload,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(sample.answer_count)),
            annotation_gt=TypedValue(type=str(annotation_payload["annotation_type"]), value=list(annotation_payload["annotation_value"])),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
            prompt_variants=task_output_prompt_variants(prompt_artifacts),
        )



def _query_axis_condition(query_id: str) -> Tuple[str, str]:
    """Resolve the row/column axis and line-count predicate from the public query."""

    query = str(query_id)
    if query.startswith("row_"):
        axis = "row"
    elif query.startswith("column_"):
        axis = "column"
    else:
        raise ValueError(f"unsupported named-grid line-condition query_id: {query_id}")
    if "_at_least_" in query:
        condition = "at_least"
    elif "_exactly_" in query:
        condition = "exactly"
    elif "_no_" in query:
        condition = "none"
    else:
        raise ValueError(f"unsupported named-grid line-condition query_id: {query_id}")
    return str(axis), str(condition)


def _choose_answer_count(
    rng,
    *,
    params: Mapping[str, Any],
    axis: str,
    grid_size_support: Sequence[Tuple[int, int]],
) -> Tuple[int, Dict[str, float]]:
    low, high = int_bounds(params, _GEN_DEFAULTS, "answer_count_min", "answer_count_max", 0, 5)
    explicit_rows = params.get("grid_rows")
    explicit_cols = params.get("grid_cols")
    if explicit_rows is not None and explicit_cols is not None:
        axis_count = axis_line_count(axis=str(axis), rows=int(explicit_rows), cols=int(explicit_cols))
        high = min(int(high), int(axis_count))
    else:
        max_axis_count = max(axis_line_count(axis=str(axis), rows=int(size[0]), cols=int(size[1])) for size in grid_size_support)
        high = min(int(high), int(max_axis_count))
    support = tuple(range(int(low), int(high) + 1))
    if not support:
        raise ValueError("answer_count support is empty")
    explicit = params.get("answer_count", params.get("target_count", params.get("answer")))
    if explicit is not None:
        answer_count = int(explicit)
        if answer_count not in set(support):
            raise ValueError("answer_count is outside configured support")
        return int(answer_count), uniform_probability_map(support, selected=int(answer_count))
    answer_count = int(rng.choice(support))
    return int(answer_count), uniform_probability_map(support)


def _choose_grid_size(
    rng,
    *,
    params: Mapping[str, Any],
    axis: str,
    answer_count: int,
) -> Tuple[int, int, Dict[str, float]]:
    support = grid_size_support(params, _GEN_DEFAULTS)
    explicit_rows = params.get("grid_rows")
    explicit_cols = params.get("grid_cols")
    if explicit_rows is not None or explicit_cols is not None:
        if explicit_rows is None or explicit_cols is None:
            raise ValueError("grid_rows and grid_cols must be provided together")
        size = (int(explicit_rows), int(explicit_cols))
        if size not in set(support):
            raise ValueError("explicit grid size is outside grid_size_support")
        axis_count = axis_line_count(axis=str(axis), rows=int(size[0]), cols=int(size[1]))
        if int(answer_count) > int(axis_count):
            raise ValueError("explicit grid size cannot support answer_count")
        labels = tuple(grid_size_label(value) for value in support)
        return int(size[0]), int(size[1]), string_probability_map(labels, selected=grid_size_label(size))

    feasible = tuple(
        size
        for size in support
        if int(answer_count) <= axis_line_count(axis=str(axis), rows=int(size[0]), cols=int(size[1]))
    )
    if not feasible:
        raise ValueError("grid_size_support cannot support answer_count")
    selected = tuple(int(value) for value in rng.choice(feasible))
    labels = tuple(grid_size_label(value) for value in feasible)
    return int(selected[0]), int(selected[1]), string_probability_map(labels)


def _choose_threshold(
    rng,
    *,
    params: Mapping[str, Any],
    condition: str,
    line_capacity: int,
) -> Tuple[int, Dict[str, float]]:
    if str(condition) == "none":
        return 0, {"0": 1.0}
    if str(condition) == "at_least":
        low, high = int_bounds(params, _GEN_DEFAULTS, "at_least_threshold_min", "at_least_threshold_max", 2, 3)
    else:
        low, high = int_bounds(params, _GEN_DEFAULTS, "exactly_threshold_min", "exactly_threshold_max", 1, 3)
    high = min(int(high), int(line_capacity))
    if high < low:
        raise ValueError("threshold support is empty for selected grid")
    support = tuple(range(int(low), int(high) + 1))
    explicit = params.get("threshold")
    if explicit is not None:
        threshold = int(explicit)
        if threshold not in set(support):
            raise ValueError("threshold is outside configured support")
        return int(threshold), uniform_probability_map(support, selected=int(threshold))
    threshold = int(rng.choice(support))
    return int(threshold), uniform_probability_map(support)


def _sample_line_target_count(rng, *, qualifies: bool, condition: str, threshold: int, line_capacity: int) -> int:
    if bool(qualifies):
        if str(condition) == "at_least":
            return int(rng.randint(int(threshold), int(line_capacity)))
        if str(condition) == "exactly":
            return int(threshold)
        return 0
    if str(condition) == "at_least":
        return int(rng.randint(0, max(0, int(threshold) - 1)))
    if str(condition) == "exactly":
        support = tuple(value for value in range(int(line_capacity) + 1) if int(value) != int(threshold))
        return int(rng.choice(support))
    return int(rng.randint(1, int(line_capacity)))


def _construct_grid_shapes(
    rng,
    *,
    support: Sequence[str],
    target_shape_id: str,
    axis: str,
    condition: str,
    threshold: int,
    answer_count: int,
    grid_rows: int,
    grid_cols: int,
) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Construct target-shape counts so exactly the sampled lines qualify.

    The queried axis is assigned line counts that satisfy or fail the selected
    predicate by construction, while non-target icons fill all remaining cells.
    """

    axis_count = axis_line_count(axis=str(axis), rows=int(grid_rows), cols=int(grid_cols))
    line_capacity = axis_line_capacity(axis=str(axis), rows=int(grid_rows), cols=int(grid_cols))
    if int(answer_count) > int(axis_count):
        raise ValueError("answer_count exceeds number of available grid lines")

    line_indices = list(range(int(axis_count)))
    rng.shuffle(line_indices)
    qualifying_line_indices = tuple(sorted(int(value) for value in line_indices[: int(answer_count)]))

    target_cells: set[Tuple[int, int]] = set()
    counted_cells: set[Tuple[int, int]] = set()
    for line_index in range(int(axis_count)):
        qualifies = int(line_index) in set(qualifying_line_indices)
        target_count = _sample_line_target_count(
            rng,
            qualifies=bool(qualifies),
            condition=str(condition),
            threshold=int(threshold),
            line_capacity=int(line_capacity),
        )
        cells = list(line_cells(axis=str(axis), line_index=int(line_index), rows=int(grid_rows), cols=int(grid_cols)))
        rng.shuffle(cells)
        selected_cells = set(cells[: int(target_count)])
        target_cells.update(selected_cells)
        if bool(qualifies):
            counted_cells.update(selected_cells)

    distractor_support = tuple(str(value) for value in support if str(value) != str(target_shape_id))
    rows_out: List[Tuple[str, ...]] = []
    for row in range(int(grid_rows)):
        row_values: List[str] = []
        for col in range(int(grid_cols)):
            if (int(row), int(col)) in target_cells:
                row_values.append(str(target_shape_id))
            else:
                row_values.append(str(rng.choice(distractor_support)))
        rows_out.append(tuple(row_values))

    shape_ids = tuple(rows_out)
    row_counts = grid_row_target_counts(shape_ids, target_shape_id=str(target_shape_id))
    column_counts = grid_column_target_counts(shape_ids, target_shape_id=str(target_shape_id))
    active_counts = active_line_counts(
        axis=str(axis),
        row_counts=row_counts,
        column_counts=column_counts,
    )
    realized = matching_line_indices(active_counts, condition=str(condition), threshold=int(threshold))
    if realized != qualifying_line_indices:
        raise RuntimeError("constructed grid does not realize requested qualifying-line count")

    return (
        tuple(tuple(str(value) for value in row) for row in shape_ids),
        tuple(sorted((int(row), int(col)) for row, col in counted_cells)),
        tuple(sorted((int(row), int(col)) for row, col in target_cells if (int(row), int(col)) not in counted_cells)),
        tuple(int(value) for value in qualifying_line_indices),
        tuple(int(value) for value in row_counts),
        tuple(int(value) for value in column_counts),
    )


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample one complete symbolic line-predicate contract before rendering."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id, query_probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    axis, condition = _query_axis_condition(str(query_id))
    grid_support = grid_size_support(params, _GEN_DEFAULTS)
    answer_count, answer_probabilities = _choose_answer_count(
        rng,
        params=params,
        axis=str(axis),
        grid_size_support=grid_support,
    )
    grid_rows, grid_cols, grid_size_probabilities = _choose_grid_size(
        rng,
        params=params,
        axis=str(axis),
        answer_count=int(answer_count),
    )
    line_capacity = axis_line_capacity(axis=str(axis), rows=int(grid_rows), cols=int(grid_cols))
    threshold, threshold_probabilities = _choose_threshold(
        rng,
        params=params,
        condition=str(condition),
        line_capacity=int(line_capacity),
    )
    shape_support = resolve_shape_support(params, _GEN_DEFAULTS)
    target_shape_id, shape_probabilities = resolve_target_shape(rng, params=params, defaults=_GEN_DEFAULTS, support=shape_support)
    shape_ids_by_cell, counted_cells, off_line_target_cells, qualifying_line_indices, row_counts, column_counts = _construct_grid_shapes(
        rng,
        support=shape_support,
        target_shape_id=str(target_shape_id),
        axis=str(axis),
        condition=str(condition),
        threshold=int(threshold),
        answer_count=int(answer_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    fill_style_support = resolve_fill_style_support(params, _GEN_DEFAULTS)
    fill_style_probabilities = resolve_fill_style_probability_map(params, _GEN_DEFAULTS, fill_style_support)
    return _SampleSpec(
        query_id=str(query_id),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        answer_count=int(answer_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        queried_axis=str(axis),
        condition=str(condition),
        threshold=int(threshold),
        shape_ids_by_cell=tuple(tuple(str(value) for value in row) for row in shape_ids_by_cell),
        counted_cells=tuple((int(row), int(col)) for row, col in counted_cells),
        off_line_target_cells=tuple((int(row), int(col)) for row, col in off_line_target_cells),
        qualifying_line_indices=tuple(int(value) for value in qualifying_line_indices),
        row_target_counts=tuple(int(value) for value in row_counts),
        column_target_counts=tuple(int(value) for value in column_counts),
        query_probabilities=dict(query_probabilities),
        answer_probabilities=dict(answer_probabilities),
        grid_size_probabilities=dict(grid_size_probabilities),
        threshold_probabilities=dict(threshold_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )

__all__ = ["IconsCountingNamedGridLineConditionCountTask"]
