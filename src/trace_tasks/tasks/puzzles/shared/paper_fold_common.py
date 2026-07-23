"""Shared dataset builders and render defaults for paper-fold result puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.config_defaults import group_default
from ...shared.render_variation import resolve_render_int, resolve_render_rgb
from .common import resolve_puzzle_axis_variant
from .unit_size_jitter import resolve_puzzle_unit_size_scale, scale_puzzle_px


SUPPORTED_PUZZLE_FOLD_SCENE_VARIANTS: Tuple[str, ...] = (
    "fold_strip",
    "fold_card",
    "fold_outline",
)
SUPPORTED_PUZZLE_FOLD_RESULT_VARIANTS: Tuple[str, ...] = (
    "vertical_fold_result",
    "horizontal_fold_result",
)
SUPPORTED_PUZZLE_FOLD_CUT_HOLE_SHAPES: Tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "rounded_square",
)
FOLD_RESULT_MARK_TYPES: Tuple[str, ...] = (
    "circle",
    "diamond",
    "square",
    "hexagon",
    "star",
)


def _resolve_puzzle_int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer parameter with task params taking precedence."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


@dataclass(frozen=True)
class PuzzleFoldResultDefaults:
    """Default generation bounds for the fold-result spatial puzzle."""

    option_count_min: int = 6
    option_count_max: int = 6
    grid_size: int = 6
    mark_count_min: int = 3
    mark_count_max: int = 5


@dataclass(frozen=True)
class PuzzleFoldResultRenderParams:
    """Resolved rendering knobs for fold-result scenes."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    reference_panel_height_px: int
    reference_panel_padding_px: int
    reference_to_options_gap_px: int
    option_gap_px: int
    option_row_gap_px: int
    option_label_gap_px: int
    paper_corner_radius_px: int
    panel_corner_radius_px: int
    border_width_px: int
    option_label_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    paper_fill_rgb: Tuple[int, int, int]
    paper_shadow_rgb: Tuple[int, int, int]
    border_color_rgb: Tuple[int, int, int]
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    fold_line_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    arrow_rgb: Tuple[int, int, int]
    instruction_fill_rgb: Tuple[int, int, int]
    cut_hole_fill_rgb: Tuple[int, int, int]
    cut_hole_outline_rgb: Tuple[int, int, int]
    cut_hole_shape: str
    unit_size_scale: float
    unit_size_jitter: Dict[str, Any]


def resolve_fold_result_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    task_id: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the active fold-result scene variant."""

    return resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_PUZZLE_FOLD_SCENE_VARIANTS,
        task_id=str(task_id),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_fold_result_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int | None = None,
) -> PuzzleFoldResultRenderParams:
    """Resolve rendering params for the fold-result puzzle."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="puzzles.fold",
        )

    def _rgb(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return resolve_render_rgb(
            params,
            render_defaults,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="puzzles.fold",
        )

    def _choice(key: str, fallback: str, supported: Sequence[str]) -> str:
        if params.get(str(key)) is not None:
            raw = str(params[str(key)])
        else:
            options = params.get(f"{str(key)}_options", group_default(render_defaults, f"{str(key)}_options", None))
            if options is not None:
                option_list = [str(item).strip() for item in options if str(item).strip()]
                if not option_list:
                    raise ValueError(f"{key}_options must contain at least one string option")
                seed = 0 if instance_seed is None else int(instance_seed)
                rng = spawn_rng(seed, f"puzzles.fold:{str(key)}", 12841)
                raw = option_list[int(rng.randrange(len(option_list)))]
            else:
                raw = str(group_default(render_defaults, str(key), str(fallback)))
        value = str(raw).strip()
        if value not in set(str(item) for item in supported):
            raise ValueError(f"{key} must be one of {tuple(supported)}")
        return value

    unit_size_scale, unit_size_jitter = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=instance_seed,
        namespace="puzzles.fold.unit_size",
    )

    return PuzzleFoldResultRenderParams(
        canvas_width=_int("canvas_width", 1200),
        canvas_height=_int("canvas_height", 840),
        scene_margin_left_px=_int("scene_margin_left_px", 64),
        scene_margin_right_px=_int("scene_margin_right_px", 64),
        scene_margin_top_px=_int("scene_margin_top_px", 56),
        scene_margin_bottom_px=_int("scene_margin_bottom_px", 56),
        reference_panel_height_px=scale_puzzle_px(_int("reference_panel_height_px", 332), unit_size_scale, min_px=180),
        reference_panel_padding_px=scale_puzzle_px(_int("reference_panel_padding_px", 28), unit_size_scale, min_px=12),
        reference_to_options_gap_px=scale_puzzle_px(_int("reference_to_options_gap_px", 44), unit_size_scale, min_px=18),
        option_gap_px=scale_puzzle_px(_int("option_gap_px", 18), unit_size_scale, min_px=10),
        option_row_gap_px=scale_puzzle_px(_int("option_row_gap_px", 18), unit_size_scale, min_px=10),
        option_label_gap_px=scale_puzzle_px(_int("option_label_gap_px", 16), unit_size_scale, min_px=8),
        paper_corner_radius_px=scale_puzzle_px(_int("paper_corner_radius_px", 18), unit_size_scale, min_px=8),
        panel_corner_radius_px=_int("panel_corner_radius_px", 28),
        border_width_px=scale_puzzle_px(_int("border_width_px", 3), unit_size_scale, min_px=2),
        option_label_font_size_px=scale_puzzle_px(_int("option_label_font_size_px", 30), unit_size_scale, min_px=20),
        panel_fill_rgb=_rgb("panel_fill_rgb", (248, 249, 252)),
        paper_fill_rgb=_rgb("paper_fill_rgb", (255, 252, 245)),
        paper_shadow_rgb=_rgb("paper_shadow_rgb", (236, 230, 215)),
        border_color_rgb=_rgb("border_color_rgb", (86, 94, 108)),
        text_color_rgb=_rgb("text_color_rgb", (30, 34, 40)),
        text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
        fold_line_rgb=_rgb("fold_line_rgb", (100, 116, 145)),
        grid_line_rgb=_rgb("grid_line_rgb", (209, 214, 223)),
        arrow_rgb=_rgb("arrow_rgb", (54, 102, 180)),
        instruction_fill_rgb=_rgb("instruction_fill_rgb", (238, 243, 250)),
        cut_hole_fill_rgb=_rgb("cut_hole_fill_rgb", (38, 45, 56)),
        cut_hole_outline_rgb=_rgb("cut_hole_outline_rgb", (255, 255, 255)),
        cut_hole_shape=_choice("cut_hole_shape", "circle", SUPPORTED_PUZZLE_FOLD_CUT_HOLE_SHAPES),
        unit_size_scale=float(unit_size_scale),
        unit_size_jitter=dict(unit_size_jitter),
    )


def folded_result_grid_dimensions(*, axis: str, grid_size: int) -> Tuple[int, int]:
    """Return the folded result grid dimensions for the active fold axis."""

    half = int(grid_size) // 2
    if str(axis) == "vertical":
        return int(half), int(grid_size)
    if str(axis) == "horizontal":
        return int(grid_size), int(half)
    raise ValueError(f"unsupported fold axis: {axis}")


def _canonical_mark_specs(
    mark_specs: Iterable[Mapping[str, Any]],
    *,
    include_source_side: bool,
) -> List[Dict[str, Any]]:
    """Canonicalize mark dictionaries into deterministic row-major order."""

    items = []
    for raw_mark in mark_specs:
        cell = raw_mark["cell"]
        cell_x = int(cell[0])
        cell_y = int(cell[1])
        record: Dict[str, Any] = {
            "mark_id": str(raw_mark["mark_id"]),
            "object_type": str(raw_mark["object_type"]),
            "cell": [int(cell_x), int(cell_y)],
        }
        if include_source_side and "source_side" in raw_mark:
            record["source_side"] = str(raw_mark["source_side"])
        items.append(record)
    items.sort(key=lambda item: (int(item["cell"][1]), int(item["cell"][0]), str(item["object_type"]), str(item["mark_id"])))
    return items


def _mark_signature(mark_specs: Iterable[Mapping[str, Any]]) -> Tuple[Tuple[str, int, int], ...]:
    """Return one hashable signature for folded-result marks."""

    return tuple(
        sorted(
            (
                str(mark["object_type"]),
                int(mark["cell"][0]),
                int(mark["cell"][1]),
            )
            for mark in mark_specs
        )
    )


def _sample_source_sides(rng, *, mark_count: int) -> List[str]:
    """Sample kept-versus-folded provenance while ensuring folding matters."""

    if int(mark_count) <= 1:
        return ["folded"]
    folded_count = int(rng.randint(1, int(mark_count) - 1))
    sides = ["folded"] * int(folded_count) + ["kept"] * int(mark_count - folded_count)
    rng.shuffle(sides)
    return [str(item) for item in sides]


def _map_local_to_original(
    *,
    axis: str,
    direction: str,
    source_side: str,
    cell_x: int,
    cell_y: int,
    grid_size: int,
) -> Tuple[int, int]:
    """Map one folded-result local cell back to the original full-sheet cell."""

    half = int(grid_size) // 2
    if str(axis) == "vertical":
        if str(direction) == "left_to_right":
            if str(source_side) == "kept":
                return int(half + cell_x), int(cell_y)
            return int((half - 1) - cell_x), int(cell_y)
        if str(direction) == "right_to_left":
            if str(source_side) == "kept":
                return int(cell_x), int(cell_y)
            return int((grid_size - 1) - cell_x), int(cell_y)
        raise ValueError(f"unsupported vertical fold direction: {direction}")
    if str(axis) == "horizontal":
        if str(direction) == "top_to_bottom":
            if str(source_side) == "kept":
                return int(cell_x), int(half + cell_y)
            return int(cell_x), int((half - 1) - cell_y)
        if str(direction) == "bottom_to_top":
            if str(source_side) == "kept":
                return int(cell_x), int(cell_y)
            return int(cell_x), int((grid_size - 1) - cell_y)
        raise ValueError(f"unsupported horizontal fold direction: {direction}")
    raise ValueError(f"unsupported fold axis: {axis}")


def _reflect_result_marks(
    mark_specs: Iterable[Mapping[str, Any]],
    *,
    axis: str,
    cols: int,
    rows: int,
) -> List[Dict[str, Any]]:
    """Reflect folded-result marks across the kept packet."""

    reflected: List[Dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        cell_x = int(mark["cell"][0])
        cell_y = int(mark["cell"][1])
        if str(axis) == "vertical":
            target = [int((cols - 1) - cell_x), int(cell_y)]
        elif str(axis) == "horizontal":
            target = [int(cell_x), int((rows - 1) - cell_y)]
        else:
            raise ValueError(f"unsupported fold axis: {axis}")
        reflected.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(mark["object_type"]),
                "cell": target,
            }
        )
    return _canonical_mark_specs(reflected, include_source_side=False)


def _shift_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    mark_index: int,
    dx: int,
    dy: int,
    cols: int,
    rows: int,
) -> List[Dict[str, Any]] | None:
    """Shift one mark inside the folded packet if the target cell is free."""

    occupied = {
        (int(mark["cell"][0]), int(mark["cell"][1]))
        for i, mark in enumerate(mark_specs)
        if int(i) != int(mark_index)
    }
    candidate: List[Dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        cell_x = int(mark["cell"][0])
        cell_y = int(mark["cell"][1])
        if int(index - 1) == int(mark_index):
            new_x = int(cell_x + dx)
            new_y = int(cell_y + dy)
            if not (0 <= int(new_x) < int(cols) and 0 <= int(new_y) < int(rows)):
                return None
            if (int(new_x), int(new_y)) in occupied:
                return None
            cell = [int(new_x), int(new_y)]
        else:
            cell = [int(cell_x), int(cell_y)]
        candidate.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(mark["object_type"]),
                "cell": cell,
            }
        )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _replace_type_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    mark_index: int,
    object_type: str,
) -> List[Dict[str, Any]]:
    """Replace one mark type while keeping positions fixed."""

    candidate: List[Dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        candidate.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(object_type if int(index - 1) == int(mark_index) else mark["object_type"]),
                "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
            }
        )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _remove_mark_candidate(mark_specs: Sequence[Mapping[str, Any]], *, mark_index: int) -> List[Dict[str, Any]] | None:
    """Remove one mark if the result still contains at least two marks."""

    if len(mark_specs) <= 2:
        return None
    candidate = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(mark["object_type"]),
            "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
        }
        for index, mark in enumerate(mark_specs, start=1)
        if int(index - 1) != int(mark_index)
    ]
    return _canonical_mark_specs(candidate, include_source_side=False)


def _add_mark_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    object_type: str,
    cell: Tuple[int, int],
) -> List[Dict[str, Any]] | None:
    """Add one mark at a free location if the type and cell are both unused."""

    occupied = {(int(mark["cell"][0]), int(mark["cell"][1])) for mark in mark_specs}
    used_types = {str(mark["object_type"]) for mark in mark_specs}
    if tuple(int(v) for v in cell) in occupied or str(object_type) in used_types:
        return None
    candidate = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(mark["object_type"]),
            "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
        }
        for index, mark in enumerate(mark_specs, start=1)
    ]
    candidate.append(
        {
            "mark_id": f"candidate_mark_{int(len(candidate) + 1)}",
            "object_type": str(object_type),
            "cell": [int(cell[0]), int(cell[1])],
        }
    )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _build_distractor_candidates(
    correct_result_marks: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    cols: int,
    rows: int,
) -> List[List[Dict[str, Any]]]:
    """Generate structured distractor candidates for a fold-result puzzle."""

    candidates: List[List[Dict[str, Any]]] = []
    candidates.append(
        _reflect_result_marks(
            correct_result_marks,
            axis=str(axis),
            cols=int(cols),
            rows=int(rows),
        )
    )
    for mark_index in range(len(correct_result_marks)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = _shift_candidate(
                correct_result_marks,
                mark_index=int(mark_index),
                dx=int(dx),
                dy=int(dy),
                cols=int(cols),
                rows=int(rows),
            )
            if shifted is not None:
                candidates.append(shifted)
    used_types = {str(mark["object_type"]) for mark in correct_result_marks}
    unused_types = [mark_type for mark_type in FOLD_RESULT_MARK_TYPES if str(mark_type) not in used_types]
    for mark_index in range(len(correct_result_marks)):
        for object_type in unused_types[:2]:
            candidates.append(
                _replace_type_candidate(
                    correct_result_marks,
                    mark_index=int(mark_index),
                    object_type=str(object_type),
                )
            )
    for mark_index in range(len(correct_result_marks)):
        removed = _remove_mark_candidate(correct_result_marks, mark_index=int(mark_index))
        if removed is not None:
            candidates.append(removed)
    if unused_types:
        occupied = {(int(mark["cell"][0]), int(mark["cell"][1])) for mark in correct_result_marks}
        for cell_y in range(int(rows)):
            for cell_x in range(int(cols)):
                if (int(cell_x), int(cell_y)) in occupied:
                    continue
                added = _add_mark_candidate(
                    correct_result_marks,
                    object_type=str(unused_types[0]),
                    cell=(int(cell_x), int(cell_y)),
                )
                if added is not None:
                    candidates.append(added)
                if len(candidates) >= 12:
                    return candidates
    return candidates


def _random_result_marks(
    rng,
    *,
    cols: int,
    rows: int,
    mark_count: int,
) -> List[Dict[str, Any]]:
    """Sample a fallback random folded-result mark set."""

    cells = rng.sample([(int(x), int(y)) for y in range(int(rows)) for x in range(int(cols))], int(mark_count))
    object_types = rng.sample(list(FOLD_RESULT_MARK_TYPES), int(mark_count))
    raw = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(object_type),
            "cell": [int(cell[0]), int(cell[1])],
        }
        for index, (cell, object_type) in enumerate(zip(cells, object_types), start=1)
    ]
    return _canonical_mark_specs(raw, include_source_side=False)


def _resolve_correct_option_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    option_count: int,
) -> int:
    """Resolve a stable correct-option slot from the instance seed."""

    explicit_index = params.get("correct_option_index")
    if explicit_index is not None:
        correct_index = int(explicit_index)
        if not 0 <= int(correct_index) < int(option_count):
            raise ValueError("correct_option_index must fall inside the option-count range")
        return int(correct_index)

    # Use a short cycle larger than the option count so balanced query-id
    # seed partitions do not alias directly into one answer letter.
    cycle_modulus = (int(option_count) * 3) + 2
    selection_index = abs(int(instance_seed + int(option_count) + 2)) % int(cycle_modulus)
    return int(selection_index % int(option_count))


def build_fold_result_dataset_for_variant(
    *,
    query_id: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: PuzzleFoldResultDefaults,
    task_id: str,
) -> Dict[str, Any]:
    """Build one single-fold paper puzzle with a unique correct folded result."""

    if str(query_id) not in SUPPORTED_PUZZLE_FOLD_RESULT_VARIANTS:
        raise ValueError(f"unsupported fold-result query id: {query_id}")

    rng = spawn_rng(int(instance_seed), f"{task_id}.dataset")
    option_count_min = int(_resolve_puzzle_int_param(params, gen_defaults, "option_count_min", defaults.option_count_min))
    option_count_max = int(_resolve_puzzle_int_param(params, gen_defaults, "option_count_max", defaults.option_count_max))
    option_count = int(rng.randint(option_count_min, max(option_count_min, option_count_max)))
    grid_size = int(_resolve_puzzle_int_param(params, gen_defaults, "grid_size", defaults.grid_size))
    if int(grid_size) % 2 != 0:
        raise ValueError("fold-result puzzles require an even grid_size")
    mark_count_min = int(_resolve_puzzle_int_param(params, gen_defaults, "mark_count_min", defaults.mark_count_min))
    mark_count_max = int(_resolve_puzzle_int_param(params, gen_defaults, "mark_count_max", defaults.mark_count_max))
    mark_count = int(rng.randint(mark_count_min, max(mark_count_min, mark_count_max)))

    axis = "vertical" if str(query_id) == "vertical_fold_result" else "horizontal"
    if str(axis) == "vertical":
        direction = str(rng.choice(("left_to_right", "right_to_left")))
    else:
        direction = str(rng.choice(("top_to_bottom", "bottom_to_top")))
    result_cols, result_rows = folded_result_grid_dimensions(axis=str(axis), grid_size=int(grid_size))

    all_local_cells = [(int(x), int(y)) for y in range(int(result_rows)) for x in range(int(result_cols))]
    max_attempts = 80
    original_mark_specs: List[Dict[str, Any]] = []
    result_mark_specs: List[Dict[str, Any]] = []
    for _ in range(max_attempts):
        local_cells = rng.sample(all_local_cells, int(mark_count))
        object_types = rng.sample(list(FOLD_RESULT_MARK_TYPES), int(mark_count))
        source_sides = _sample_source_sides(rng, mark_count=int(mark_count))
        raw_result_marks: List[Dict[str, Any]] = []
        raw_original_marks: List[Dict[str, Any]] = []
        for mark_index, (cell, object_type, source_side) in enumerate(
            zip(local_cells, object_types, source_sides),
            start=1,
        ):
            cell_x, cell_y = int(cell[0]), int(cell[1])
            mark_id = f"mark_{int(mark_index)}"
            raw_result_marks.append(
                {
                    "mark_id": str(mark_id),
                    "object_type": str(object_type),
                    "cell": [int(cell_x), int(cell_y)],
                    "source_side": str(source_side),
                }
            )
            original_x, original_y = _map_local_to_original(
                axis=str(axis),
                direction=str(direction),
                source_side=str(source_side),
                cell_x=int(cell_x),
                cell_y=int(cell_y),
                grid_size=int(grid_size),
            )
            raw_original_marks.append(
                {
                    "mark_id": str(mark_id),
                    "object_type": str(object_type),
                    "cell": [int(original_x), int(original_y)],
                    "source_side": str(source_side),
                }
            )
        candidate_result = _canonical_mark_specs(raw_result_marks, include_source_side=True)
        mirrored_signature = _mark_signature(
            _reflect_result_marks(
                candidate_result,
                axis=str(axis),
                cols=int(result_cols),
                rows=int(result_rows),
            )
        )
        if _mark_signature(candidate_result) == mirrored_signature:
            continue
        original_mark_specs = _canonical_mark_specs(raw_original_marks, include_source_side=True)
        result_mark_specs = candidate_result
        break
    else:
        raise ValueError("failed to sample a non-symmetric fold-result mark layout")

    correct_option_marks = _canonical_mark_specs(result_mark_specs, include_source_side=False)
    distractor_candidates = _build_distractor_candidates(
        correct_option_marks,
        axis=str(axis),
        cols=int(result_cols),
        rows=int(result_rows),
    )
    chosen_distractors: List[List[Dict[str, Any]]] = []
    seen_signatures = {_mark_signature(correct_option_marks)}
    for candidate in distractor_candidates:
        signature = _mark_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen_distractors.append(candidate)
        if len(chosen_distractors) >= int(option_count) - 1:
            break
    while len(chosen_distractors) < int(option_count) - 1:
        fallback = _random_result_marks(
            rng,
            cols=int(result_cols),
            rows=int(result_rows),
            mark_count=int(mark_count),
        )
        signature = _mark_signature(fallback)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen_distractors.append(fallback)

    correct_index = _resolve_correct_option_index(
        params=params,
        instance_seed=int(instance_seed),
        option_count=int(option_count),
    )
    option_mark_sets: List[List[Dict[str, Any]]] = []
    distractor_iter = iter(chosen_distractors)
    for option_index in range(int(option_count)):
        if int(option_index) == int(correct_index):
            option_mark_sets.append(correct_option_marks)
        else:
            option_mark_sets.append(next(distractor_iter))

    option_specs: List[Dict[str, Any]] = []
    for option_index, mark_specs in enumerate(option_mark_sets):
        option_label = chr(ord("A") + int(option_index))
        option_choice_id = f"option_choice_{option_label}"
        option_specs.append(
            {
                "option_label": str(option_label),
                "option_choice_id": str(option_choice_id),
                "mark_specs": [
                    {
                        "mark_id": f"{str(option_choice_id)}_mark_{int(index)}",
                        "object_type": str(mark["object_type"]),
                        "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
                    }
                    for index, mark in enumerate(mark_specs, start=1)
                ],
                "is_correct": bool(int(option_index) == int(correct_index)),
            }
        )

    folded_mark_count = sum(1 for mark in result_mark_specs if str(mark["source_side"]) == "folded")
    kept_mark_count = int(mark_count - folded_mark_count)
    return {
        "grid_size": int(grid_size),
        "result_grid_cols": int(result_cols),
        "result_grid_rows": int(result_rows),
        "option_count": int(option_count),
        "mark_count": int(mark_count),
        "fold_axis": str(axis),
        "fold_direction": str(direction),
        "original_mark_specs": original_mark_specs,
        "folded_result_mark_specs": result_mark_specs,
        "option_specs": option_specs,
        "answer_option_label": str(option_specs[int(correct_index)]["option_label"]),
        "correct_option_choice_id": str(option_specs[int(correct_index)]["option_choice_id"]),
        "correct_option_index": int(correct_index),
        "question_format": "fold_result_mcq",
        "view_family": "paper_fold_result_mcq",
        "folded_mark_count": int(folded_mark_count),
        "kept_mark_count": int(kept_mark_count),
    }


__all__ = [
    "FOLD_RESULT_MARK_TYPES",
    "PuzzleFoldResultDefaults",
    "PuzzleFoldResultRenderParams",
    "SUPPORTED_PUZZLE_FOLD_RESULT_VARIANTS",
    "SUPPORTED_PUZZLE_FOLD_SCENE_VARIANTS",
    "build_fold_result_dataset_for_variant",
    "folded_result_grid_dimensions",
    "resolve_fold_result_render_params",
    "resolve_fold_result_scene_variant",
]
