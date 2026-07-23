"""Count unordered adjacent icon-type pairs in one named-grid row or column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import segment_set_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ...shared.variant_sampling import resolve_variant
from ..shared.procedural_named_icons import procedural_named_icon_display_name
from .shared.defaults import SCENE_ID, NamedGridDefaults
from .shared.metrics import axis_line_capacity, axis_line_count, grid_cells, line_cells
from .shared.output import (
    build_named_grid_trace_payload,
    cells_to_trace,
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
    shape_support as resolve_shape_support,
    string_probability_map,
)
from .shared.state import NamedGridScenePayload
from .shared.styles import resolve_named_grid_render_params


DOMAIN = "icons"

TASK_ID = "task_icons__named_grid__line_adjacency_pair_count"
QUERY_IDS: Tuple[str, ...] = (
    "row_unordered_adjacent_pair_count",
    "column_unordered_adjacent_pair_count",
)


GridCell = Tuple[int, int]
GridPair = Tuple[GridCell, GridCell]


@dataclass(frozen=True)
class _SampleSpec:
    """Symbolic named-grid adjacent-pair count sample."""

    query_id: str
    first_shape_id: str
    first_shape_name: str
    second_shape_id: str
    second_shape_name: str
    target_shape_id: str
    answer_count: int
    grid_rows: int
    grid_cols: int
    queried_axis: str
    queried_index: int
    shape_ids_by_cell: Tuple[Tuple[str, ...], ...]
    counted_cells: Tuple[GridCell, ...]
    counted_pair_cells: Tuple[GridPair, ...]
    query_probabilities: Dict[str, float]
    answer_probabilities: Dict[str, float]
    grid_size_probabilities: Dict[str, float]
    line_index_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


_DEFAULTS = NamedGridDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _query_axis(query_id: str) -> str:
    """Resolve the row/column axis from the public query id."""

    query = str(query_id)
    if query.startswith("row_"):
        return "row"
    if query.startswith("column_"):
        return "column"
    raise ValueError(f"unsupported named-grid adjacency query_id: {query_id}")


def _choose_answer_count(rng, *, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    low, high = int_bounds(
        params,
        _GEN_DEFAULTS,
        "answer_count_min",
        "answer_count_max",
        1,
        5,
    )
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
        line_capacity = axis_line_capacity(axis=str(axis), rows=int(size[0]), cols=int(size[1]))
        if int(answer_count) > max(0, int(line_capacity) - 1):
            raise ValueError("explicit grid size cannot support answer_count")
        labels = tuple(grid_size_label(value) for value in support)
        return int(size[0]), int(size[1]), string_probability_map(labels, selected=grid_size_label(size))

    feasible = tuple(
        size
        for size in support
        if int(answer_count) <= max(0, axis_line_capacity(axis=str(axis), rows=int(size[0]), cols=int(size[1])) - 1)
    )
    if not feasible:
        raise ValueError("grid_size_support cannot support answer_count")
    selected = tuple(int(value) for value in rng.choice(feasible))
    labels = tuple(grid_size_label(value) for value in feasible)
    return int(selected[0]), int(selected[1]), string_probability_map(labels)


def _choose_line_index(
    rng,
    *,
    params: Mapping[str, Any],
    axis: str,
    grid_rows: int,
    grid_cols: int,
) -> Tuple[int, Dict[str, float]]:
    axis_count = axis_line_count(axis=str(axis), rows=int(grid_rows), cols=int(grid_cols))
    key = "target_row_number" if str(axis) == "row" else "target_column_number"
    explicit = params.get(key, params.get("line_number"))
    support = tuple(range(1, int(axis_count) + 1))
    if explicit is not None:
        number = int(explicit)
        if number not in set(support):
            raise ValueError(f"{key} must be in 1..{axis_count}")
        return int(number) - 1, uniform_probability_map(support, selected=int(number))
    number = int(rng.choice(support))
    return int(number) - 1, uniform_probability_map(support)


def _sample_shape_pair(rng, *, params: Mapping[str, Any]) -> Tuple[str, str, Dict[str, float]]:
    support = tuple(str(value) for value in resolve_shape_support(params, _GEN_DEFAULTS))
    if len(support) < 2:
        raise ValueError("named-grid adjacent-pair task needs at least two supported shapes")

    explicit_first = params.get("first_shape_id", params.get("shape_id_a"))
    explicit_second = params.get("second_shape_id", params.get("shape_id_b"))
    if explicit_first is not None:
        first = str(explicit_first)
        if first not in set(support):
            raise ValueError(f"unsupported first_shape_id: {first}")
    else:
        first = str(rng.choice(support))

    remaining = tuple(value for value in support if str(value) != str(first))
    if explicit_second is not None:
        second = str(explicit_second)
        if second not in set(remaining):
            raise ValueError("second_shape_id must be supported and distinct from first_shape_id")
    else:
        second = str(rng.choice(remaining))

    if explicit_first is not None or explicit_second is not None:
        probabilities = {value: (1.0 if value in {str(first), str(second)} else 0.0) for value in support}
    else:
        probabilities = string_probability_map(support)
    return str(first), str(second), dict(probabilities)


def _line_pair_cells(
    shape_ids_by_cell: Sequence[Sequence[str]],
    *,
    axis: str,
    line_index: int,
    first_shape_id: str,
    second_shape_id: str,
) -> Tuple[GridPair, ...]:
    rows = len(shape_ids_by_cell)
    cols = len(shape_ids_by_cell[0]) if shape_ids_by_cell else 0
    cells = line_cells(axis=str(axis), line_index=int(line_index), rows=int(rows), cols=int(cols))
    target_pair = {str(first_shape_id), str(second_shape_id)}
    pairs: List[GridPair] = []
    for index in range(max(0, len(cells) - 1)):
        left = cells[int(index)]
        right = cells[int(index) + 1]
        observed = {
            str(shape_ids_by_cell[int(left[0])][int(left[1])]),
            str(shape_ids_by_cell[int(right[0])][int(right[1])]),
        }
        if observed == target_pair:
            pairs.append((left, right))
    return tuple(pairs)


def _construct_grid_shapes(
    rng,
    *,
    support: Sequence[str],
    first_shape_id: str,
    second_shape_id: str,
    axis: str,
    answer_count: int,
    grid_rows: int,
    grid_cols: int,
    queried_index: int,
) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[GridCell, ...], Tuple[GridPair, ...]]:
    """Construct one grid line with exactly the requested unordered pair count."""

    line_len = axis_line_capacity(axis=str(axis), rows=int(grid_rows), cols=int(grid_cols))
    if int(answer_count) > int(line_len) - 1:
        raise ValueError("answer_count exceeds adjacent-pair capacity for selected grid")
    filler_support = tuple(
        str(value)
        for value in support
        if str(value) not in {str(first_shape_id), str(second_shape_id)}
    )
    if int(answer_count) < int(line_len) - 1 and not filler_support:
        raise ValueError("at least one filler shape is needed for non-full adjacent-pair lines")

    line_values = [str(rng.choice(filler_support or support)) for _ in range(int(line_len))]
    block_len = int(answer_count) + 1
    block_start = int(rng.randint(0, int(line_len) - int(block_len)))
    start_first = bool(rng.randint(0, 1))
    for offset in range(int(block_len)):
        use_first = (int(offset) % 2 == 0) == bool(start_first)
        line_values[int(block_start) + int(offset)] = str(first_shape_id if use_first else second_shape_id)

    rows_out: List[List[str]] = []
    for row in range(int(grid_rows)):
        row_values: List[str] = []
        for col in range(int(grid_cols)):
            row_values.append(str(rng.choice(support)))
        rows_out.append(row_values)

    target_line_cells = line_cells(
        axis=str(axis),
        line_index=int(queried_index),
        rows=int(grid_rows),
        cols=int(grid_cols),
    )
    for cell, shape_id in zip(target_line_cells, line_values):
        rows_out[int(cell[0])][int(cell[1])] = str(shape_id)

    shape_ids = tuple(tuple(str(value) for value in row) for row in rows_out)
    counted_pair_cells = _line_pair_cells(
        shape_ids,
        axis=str(axis),
        line_index=int(queried_index),
        first_shape_id=str(first_shape_id),
        second_shape_id=str(second_shape_id),
    )
    if len(counted_pair_cells) != int(answer_count):
        raise RuntimeError("constructed grid does not realize requested adjacent-pair count")

    counted_cells = tuple(
        sorted(
            {
                (int(row), int(col))
                for pair in counted_pair_cells
                for row, col in pair
            }
        )
    )
    return shape_ids, counted_cells, counted_pair_cells


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample one complete symbolic adjacent-pair contract before rendering."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id, query_probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    axis = _query_axis(str(query_id))
    answer_count, answer_probabilities = _choose_answer_count(rng, params=params)
    grid_rows, grid_cols, grid_size_probabilities = _choose_grid_size(
        rng,
        params=params,
        axis=str(axis),
        answer_count=int(answer_count),
    )
    queried_index, line_index_probabilities = _choose_line_index(
        rng,
        params=params,
        axis=str(axis),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    first_shape_id, second_shape_id, shape_probabilities = _sample_shape_pair(rng, params=params)
    shape_support = resolve_shape_support(params, _GEN_DEFAULTS)
    shape_ids_by_cell, counted_cells, counted_pair_cells = _construct_grid_shapes(
        rng,
        support=shape_support,
        first_shape_id=str(first_shape_id),
        second_shape_id=str(second_shape_id),
        axis=str(axis),
        answer_count=int(answer_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        queried_index=int(queried_index),
    )
    fill_style_support = resolve_fill_style_support(params, _GEN_DEFAULTS)
    fill_style_probabilities = resolve_fill_style_probability_map(params, _GEN_DEFAULTS, fill_style_support)
    return _SampleSpec(
        query_id=str(query_id),
        first_shape_id=str(first_shape_id),
        first_shape_name=procedural_named_icon_display_name(str(first_shape_id)),
        second_shape_id=str(second_shape_id),
        second_shape_name=procedural_named_icon_display_name(str(second_shape_id)),
        target_shape_id=str(first_shape_id),
        answer_count=int(answer_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        queried_axis=str(axis),
        queried_index=int(queried_index),
        shape_ids_by_cell=tuple(tuple(str(value) for value in row) for row in shape_ids_by_cell),
        counted_cells=tuple((int(row), int(col)) for row, col in counted_cells),
        counted_pair_cells=tuple(
            (
                (int(left[0]), int(left[1])),
                (int(right[0]), int(right[1])),
            )
            for left, right in counted_pair_cells
        ),
        query_probabilities=dict(query_probabilities),
        answer_probabilities=dict(answer_probabilities),
        grid_size_probabilities=dict(grid_size_probabilities),
        line_index_probabilities=dict(line_index_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )


def _cell_center(scene: NamedGridScenePayload, cell: GridCell) -> list[float]:
    bbox = scene.cell_bboxes_xyxy[int(cell[0])][int(cell[1])]
    return [
        round(0.5 * float(int(bbox[0]) + int(bbox[2])), 3),
        round(0.5 * float(int(bbox[1]) + int(bbox[3])), 3),
    ]


def _pair_segments(scene: NamedGridScenePayload, pairs: Sequence[GridPair]) -> list[list[list[float]]]:
    return [
        [_cell_center(scene, left), _cell_center(scene, right)]
        for left, right in pairs
    ]


@register_task
class IconsNamedGridLineAdjacencyPairCountTask:
    """Count unordered adjacent pairs of two icon types in one grid line."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one named-grid adjacent-pair counting instance."""

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

        annotation_segments = _pair_segments(scene, sample.counted_pair_cells)
        if len(annotation_segments) != int(sample.answer_count):
            raise RuntimeError("rendered named-grid adjacent-pair annotation count does not match answer")
        annotation_artifacts = segment_set_annotation_artifacts(annotation_segments)
        annotation_payload = {
            "annotation_type": str(annotation_artifacts.annotation_type),
            "annotation_value": annotation_artifacts.value,
            "projected_annotation": annotation_artifacts.projected_annotation,
        }

        question_key = f"question_text_{sample.query_id}"
        prompt_artifacts, _prompt_defaults = build_named_grid_prompt_artifacts(
            domain=DOMAIN,
            run_namespace=self.task_id,
            prompt_defaults_map=_PROMPT_DEFAULTS,
            question_key=str(question_key),
            question_slots={
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_name": str(sample.second_shape_name),
                "line_number": int(sample.queried_index) + 1,
            },
            annotation_slots={
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_name": str(sample.second_shape_name),
                "line_kind": str(sample.queried_axis),
                "line_number": int(sample.queried_index) + 1,
            },
            instance_seed=int(instance_seed),
            annotation_hint_key="adjacency_annotation_hint",
            answer_hint_key="adjacency_answer_hint",
            json_example_key="adjacency_json_example",
            json_example_answer_only_key="adjacency_json_example_answer_only",
        )

        serialized_icons = [serialize_named_grid_icon(icon) for icon in scene.icons]
        shape_counts = shape_counts_from_icons(scene.icons)
        counted_cells_trace = cells_to_trace(sample.counted_cells)
        counted_pair_cells_trace = [
            [list(left), list(right)]
            for left, right in sample.counted_pair_cells
        ]
        counted_pair_segments_px = [[list(point) for point in segment] for segment in annotation_segments]
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(sample.query_id),
            params={
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "answer_count": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "queried_axis": str(sample.queried_axis),
                "queried_index": int(sample.queried_index),
                "queried_number": int(sample.queried_index) + 1,
                "query_id_probabilities": dict(sample.query_probabilities),
                "answer_probabilities": dict(sample.answer_probabilities),
                "grid_size_probabilities": dict(sample.grid_size_probabilities),
                "line_index_probabilities": dict(sample.line_index_probabilities),
                "shape_id_support": list(resolve_shape_support(params, _GEN_DEFAULTS)),
                "shape_probabilities": dict(sample.shape_probabilities),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "fill_style_probabilities": dict(sample.fill_style_probabilities),
            },
        )
        trace_payload = build_named_grid_trace_payload(
            scene=scene,
            scene_kind="icons_named_grid_line_adjacency_pair_count",
            entities=serialized_icons,
            relations={
                "counting_rule": "unordered_adjacent_shape_pair_in_prompt_addressed_grid_line",
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "queried_axis": str(sample.queried_axis),
                "queried_index": int(sample.queried_index),
                "queried_number": int(sample.queried_index) + 1,
                "pair_order": "unordered",
                "overlapping_pairs_count_separately": True,
                "answer_count": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "shape_counts": dict(shape_counts),
                "counted_cells": counted_cells_trace,
                "counted_pair_cells": counted_pair_cells_trace,
            },
            query_spec=query_spec,
            render_params=render_params,
            rows=int(sample.grid_rows),
            cols=int(sample.grid_cols),
            render_map_extra={
                "counted_pair_cells": counted_pair_cells_trace,
                "counted_pair_segments_px": counted_pair_segments_px,
            },
            execution_trace={
                "scene_variant": "single_panel_named_grid",
                "query_id": str(sample.query_id),
                "question_format": "count_unordered_adjacent_named_shape_pairs_in_grid_row_or_column",
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "answer": int(sample.answer_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "queried_axis": str(sample.queried_axis),
                "queried_index": int(sample.queried_index),
                "queried_number": int(sample.queried_index) + 1,
                "shape_ids_by_cell": [list(row) for row in sample.shape_ids_by_cell],
                "counted_cells": counted_cells_trace,
                "counted_pair_cells": counted_pair_cells_trace,
                "counted_pair_segments_px": counted_pair_segments_px,
                "pair_order": "unordered",
                "overlapping_pairs_count_separately": True,
            },
            witness_symbolic={
                "query_id": str(sample.query_id),
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "answer": int(sample.answer_count),
                "counted_cells": counted_cells_trace,
                "counted_pair_cells": counted_pair_cells_trace,
            },
            annotation_payload=annotation_payload,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(sample.answer_count)),
            annotation_gt=annotation_artifacts.annotation_gt,
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
            prompt_variants=task_output_prompt_variants(prompt_artifacts),
        )


__all__ = ["IconsNamedGridLineAdjacencyPairCountTask"]
