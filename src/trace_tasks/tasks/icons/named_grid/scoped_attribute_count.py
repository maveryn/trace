"""Count prompt-named procedural icons in a specified grid row or column."""

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
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_scene import sort_bboxes_reading_order
from ..shared.procedural_named_icons import procedural_named_icon_display_name
from .shared.defaults import SCENE_ID, NamedGridDefaults
from .shared.metrics import axis_line_capacity, axis_line_count, grid_cells, line_cells, line_shape_counts as grid_line_shape_counts
from .shared.output import (
    build_named_grid_trace_payload,
    cells_to_trace,
    instance_ids,
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
from .shared.styles import resolve_named_grid_render_params


DOMAIN = "icons"

TASK_ID = "task_icons__named_grid__scoped_attribute_count"
QUERY_IDS: Tuple[str, ...] = ("row_shape_count", "column_shape_count")


@dataclass(frozen=True)
class _SampleSpec:
    """Symbolic named-grid count sample."""

    query_id: str
    target_shape_id: str
    target_shape_name: str
    target_count: int
    grid_rows: int
    grid_cols: int
    queried_axis: str
    queried_index: int
    shape_ids_by_cell: Tuple[Tuple[str, ...], ...]
    counted_cells: Tuple[Tuple[int, int], ...]
    off_line_target_cells: Tuple[Tuple[int, int], ...]
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


def _choose_grid_size(
    rng,
    *,
    params: Mapping[str, Any],
    axis: str,
    target_count: int,
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
        if int(target_count) > int(line_capacity):
            raise ValueError("explicit grid size cannot support requested target_count")
        labels = tuple(grid_size_label(value) for value in support)
        return int(size[0]), int(size[1]), string_probability_map(labels, selected=grid_size_label(size))

    feasible = tuple(
        size
        for size in support
        if int(target_count) <= axis_line_capacity(axis=str(axis), rows=int(size[0]), cols=int(size[1]))
    )
    if not feasible:
        raise ValueError("grid_size_support cannot support requested target_count")
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
        if int(number) not in set(support):
            raise ValueError(f"{key} must be in 1..{axis_count}")
        return int(number) - 1, uniform_probability_map(support, selected=int(number))
    number = int(rng.choice(support))
    return int(number) - 1, uniform_probability_map(support)


def _construct_grid_shapes(
    rng,
    *,
    support: Sequence[str],
    target_shape_id: str,
    axis: str,
    target_count: int,
    grid_rows: int,
    grid_cols: int,
    queried_index: int,
    params: Mapping[str, Any],
) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    """Place the requested target shape exactly in the queried line.

    The queried row or column realizes the sampled answer count. Extra target
    shapes are allowed only outside that line so localization remains necessary.
    """

    queried_line_cells = line_cells(
        axis=str(axis),
        line_index=int(queried_index),
        rows=int(grid_rows),
        cols=int(grid_cols),
    )
    if int(target_count) > len(queried_line_cells):
        raise ValueError("target_count exceeds queried row/column capacity")
    shuffled_line = list(queried_line_cells)
    rng.shuffle(shuffled_line)
    counted_cells = tuple(sorted(shuffled_line[: int(target_count)]))

    off_line_cells = [cell for cell in grid_cells(rows=int(grid_rows), cols=int(grid_cols)) if cell not in set(queried_line_cells)]
    off_min, off_max = int_bounds(
        params,
        _GEN_DEFAULTS,
        "off_line_target_count_min",
        "off_line_target_count_max",
        _DEFAULTS.off_line_target_count_min,
        _DEFAULTS.off_line_target_count_max,
    )
    max_off = min(int(off_max), len(off_line_cells))
    min_off = min(int(off_min), max_off)
    explicit_off = params.get("off_line_target_count")
    if explicit_off is not None:
        off_count = int(explicit_off)
        if off_count < 0 or off_count > len(off_line_cells):
            raise ValueError("off_line_target_count is outside feasible support")
    else:
        off_count = int(rng.randint(int(min_off), int(max_off))) if max_off > 0 else 0
    rng.shuffle(off_line_cells)
    off_line_target_cells = tuple(sorted(off_line_cells[: int(off_count)]))

    target_cells = set(counted_cells) | set(off_line_target_cells)
    distractor_support = tuple(str(value) for value in support if str(value) != str(target_shape_id))
    rows: List[Tuple[str, ...]] = []
    for row in range(int(grid_rows)):
        row_shapes: List[str] = []
        for col in range(int(grid_cols)):
            if (int(row), int(col)) in target_cells:
                row_shapes.append(str(target_shape_id))
            else:
                row_shapes.append(str(rng.choice(distractor_support)))
        rows.append(tuple(row_shapes))

    realized_count = sum(
        1
        for row, col in queried_line_cells
        if rows[int(row)][int(col)] == str(target_shape_id)
    )
    if int(realized_count) != int(target_count):
        raise RuntimeError("constructed grid does not realize requested row/column target count")
    return tuple(rows), tuple(counted_cells), tuple(off_line_target_cells)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample one complete symbolic contract before rendering."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id, query_probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    axis = "row" if str(query_id) == "row_shape_count" else "column"
    count_min, count_max = int_bounds(params, _GEN_DEFAULTS, "target_count_min", "target_count_max", _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    answer_support = tuple(range(int(count_min), int(count_max) + 1))
    explicit_count = params.get("target_count", params.get("answer"))
    if explicit_count is not None:
        target_count = int(explicit_count)
        if int(target_count) not in set(answer_support):
            raise ValueError("explicit target_count is outside configured support")
        answer_probabilities = uniform_probability_map(answer_support, selected=int(target_count))
    else:
        target_count = int(rng.choice(answer_support))
        answer_probabilities = uniform_probability_map(answer_support)

    shape_support = resolve_shape_support(params, _GEN_DEFAULTS)
    target_shape_id, shape_probabilities = resolve_target_shape(rng, params=params, defaults=_GEN_DEFAULTS, support=shape_support)
    grid_rows, grid_cols, grid_size_probabilities = _choose_grid_size(
        rng,
        params=params,
        axis=str(axis),
        target_count=int(target_count),
    )
    queried_index, line_index_probabilities = _choose_line_index(
        rng,
        params=params,
        axis=str(axis),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    shape_ids_by_cell, counted_cells, off_line_target_cells = _construct_grid_shapes(
        rng,
        support=shape_support,
        target_shape_id=str(target_shape_id),
        axis=str(axis),
        target_count=int(target_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        queried_index=int(queried_index),
        params=params,
    )
    fill_style_support = resolve_fill_style_support(params, _GEN_DEFAULTS)
    fill_style_probabilities = resolve_fill_style_probability_map(params, _GEN_DEFAULTS, fill_style_support)
    return _SampleSpec(
        query_id=str(query_id),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        target_count=int(target_count),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        queried_axis=str(axis),
        queried_index=int(queried_index),
        shape_ids_by_cell=tuple(tuple(str(value) for value in row) for row in shape_ids_by_cell),
        counted_cells=tuple((int(row), int(col)) for row, col in counted_cells),
        off_line_target_cells=tuple((int(row), int(col)) for row, col in off_line_target_cells),
        query_probabilities=dict(query_probabilities),
        answer_probabilities=dict(answer_probabilities),
        grid_size_probabilities=dict(grid_size_probabilities),
        line_index_probabilities=dict(line_index_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )
@register_task
class IconsCountingNamedGridRowColumnShapeCountTask:
    """Count named icons in a prompt-addressed grid row or column."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one named-grid row/column scoped counting instance."""

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
                    sample=sample,
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

        counted_icons = tuple(icon for icon in scene.icons if bool(icon.is_counted))
        annotation_bboxes = sort_bboxes_reading_order(icon.bbox_xyxy for icon in counted_icons)
        if len(annotation_bboxes) != int(sample.target_count):
            raise RuntimeError("rendered named-grid annotation count does not match answer")
        annotation_payload = icon_bbox_set_annotation(annotation_bboxes)

        question_key = f"question_text_{sample.query_id}"
        prompt_artifacts, _prompt_defaults = build_named_grid_prompt_artifacts(
            domain=DOMAIN,
            run_namespace=self.task_id,
            prompt_defaults_map=_PROMPT_DEFAULTS,
            question_key=str(question_key),
            question_slots={
                "target_shape_name": str(sample.target_shape_name),
                "line_number": int(sample.queried_index) + 1,
            },
            annotation_slots={
                "target_shape_name": str(sample.target_shape_name),
                "line_kind": str(sample.queried_axis),
                "line_number": int(sample.queried_index) + 1,
            },
            instance_seed=int(instance_seed),
            annotation_hint_key="scoped_annotation_hint",
            answer_hint_key="scoped_answer_hint",
            json_example_key="scoped_json_example",
            json_example_answer_only_key="scoped_json_example_answer_only",
        )

        serialized_icons = [serialize_named_grid_icon(icon) for icon in scene.icons]
        counted_instance_ids = instance_ids(counted_icons)
        shape_counts = shape_counts_from_icons(scene.icons)
        queried_line_shape_counts = grid_line_shape_counts(
            sample.shape_ids_by_cell,
            axis=str(sample.queried_axis),
            line_index=int(sample.queried_index),
        )
        counted_cells_trace = cells_to_trace(sample.counted_cells)
        off_line_cells_trace = cells_to_trace(sample.off_line_target_cells)
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(sample.query_id),
            params={
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "target_count": int(sample.target_count),
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
            scene_kind="icons_named_grid_row_column_shape_count",
            entities=serialized_icons,
            relations={
                "counting_rule": "shape_id_in_prompt_addressed_grid_line",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "queried_axis": str(sample.queried_axis),
                "queried_index": int(sample.queried_index),
                "queried_number": int(sample.queried_index) + 1,
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "shape_counts": dict(shape_counts),
                "queried_line_shape_counts": dict(queried_line_shape_counts),
                "counted_cells": counted_cells_trace,
                "off_line_target_cells": off_line_cells_trace,
            },
            query_spec=query_spec,
            render_params=render_params,
            rows=int(sample.grid_rows),
            cols=int(sample.grid_cols),
            render_map_extra={"counted_instance_ids": list(counted_instance_ids)},
            execution_trace={
                "scene_variant": "single_panel_named_grid",
                "query_id": str(sample.query_id),
                "question_format": "count_named_shape_in_grid_row_or_column",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "answer": int(sample.target_count),
                "grid_rows": int(sample.grid_rows),
                "grid_cols": int(sample.grid_cols),
                "queried_axis": str(sample.queried_axis),
                "queried_index": int(sample.queried_index),
                "queried_number": int(sample.queried_index) + 1,
                "shape_ids_by_cell": [list(row) for row in sample.shape_ids_by_cell],
                "counted_cells": counted_cells_trace,
                "off_line_target_cells": off_line_cells_trace,
                "counted_instance_ids": list(counted_instance_ids),
                "queried_line_shape_counts": dict(queried_line_shape_counts),
            },
            witness_symbolic={
                "query_id": str(sample.query_id),
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "answer": int(sample.target_count),
                "counted_cells": counted_cells_trace,
                "counted_instance_ids": list(counted_instance_ids),
            },
            annotation_payload=annotation_payload,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(sample.target_count)),
            annotation_gt=TypedValue(type=str(annotation_payload["annotation_type"]), value=list(annotation_payload["annotation_value"])),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
            prompt_variants=task_output_prompt_variants(prompt_artifacts),
        )


__all__ = ["IconsCountingNamedGridRowColumnShapeCountTask"]
