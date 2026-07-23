"""Neutral sampling helpers for counterfactual-board game cases."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.deterministic_sampling import (
    resolve_selection_index,
    uniform_probability_map,
)

from .rules import canonical_answer_for_style, target_answer_for_axis
from .state import CounterfactualBoardCase, STYLE_SPECS


def _uniform_string_probability_map(values: Sequence[str]) -> dict[str, float]:
    """Return a uniform probability map for non-numeric style labels."""

    labels = tuple(str(value) for value in values)
    if not labels:
        return {}
    probability = 1.0 / float(len(labels))
    return {label: probability for label in labels}


def select_board_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported_styles: Sequence[str],
    namespace: str,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Resolve one style from task-owned style support without query routing."""

    styles = tuple(str(style) for style in supported_styles)
    if not styles:
        raise ValueError("supported_styles must be non-empty")
    explicit = params.get("board_style")
    if explicit is not None:
        style = str(explicit)
        if style not in styles:
            raise ValueError(f"unsupported board_style {style!r}; supported: {styles}")
        return style, {value: (1.0 if value == style else 0.0) for value in styles}, dict(params)

    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        cursor = abs(int(sample_cursor))
        style = styles[cursor % len(styles)]
        next_params = dict(params)
        next_params["_sample_cursor"] = cursor // len(styles)
        return style, _uniform_string_probability_map(styles), next_params

    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(rng.choice(styles)), _uniform_string_probability_map(styles), dict(params)


def _select_dimension(
    *,
    support: Sequence[int],
    params: Mapping[str, Any],
    explicit_keys: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    for key in explicit_keys:
        if key not in params:
            continue
        value = int(params[key])
        if value not in set(int(item) for item in support):
            raise ValueError(f"{key}={value} outside supported values {tuple(support)}")
        return int(value), dict(uniform_probability_map(support, selected=value))

    values = tuple(int(value) for value in support)
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    selected = values[int(index) % len(values)]
    return int(selected), dict(uniform_probability_map(values))


def sample_board_dimensions(
    *,
    style: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[int, int, dict[str, float], dict[str, float]]:
    """Sample visible row and column counts from one board style support."""

    spec = STYLE_SPECS[str(style)]
    rows, row_probs = _select_dimension(
        support=spec.row_support,
        params=params,
        explicit_keys=("visible_rows",),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.rows",
    )
    cols, col_probs = _select_dimension(
        support=spec.col_support,
        params=params,
        explicit_keys=("visible_columns",),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.cols",
    )
    return int(rows), int(cols), dict(row_probs), dict(col_probs)


def build_counterfactual_board_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported_styles: Sequence[str],
    board_kind: str,
    counted_axis: str,
    prompt_query_key: str,
    style_namespace: str,
    dimension_namespace: str,
) -> CounterfactualBoardCase:
    """Build one count case from task-owned style and axis choices."""

    style, style_probs, style_params = select_board_style(
        instance_seed=int(instance_seed),
        params=params,
        supported_styles=supported_styles,
        namespace=str(style_namespace),
    )
    rows, cols, row_probs, col_probs = sample_board_dimensions(
        style=str(style),
        instance_seed=int(instance_seed),
        params=style_params,
        namespace=str(dimension_namespace),
    )
    spec = STYLE_SPECS[str(style)]
    answer = target_answer_for_axis(
        counted_axis=str(counted_axis),
        rows=int(rows),
        cols=int(cols),
    )
    canonical = canonical_answer_for_style(
        counted_axis=str(counted_axis),
        style=str(style),
    )
    case_params = {
        "board_style": str(style),
        "board_style_probabilities": dict(style_probs),
        "board_kind": str(board_kind),
        "visible_rows": int(rows),
        "visible_columns": int(cols),
        "row_count_probabilities": dict(row_probs),
        "column_count_probabilities": dict(col_probs),
        "canonical_rows": int(spec.canonical_rows),
        "canonical_columns": int(spec.canonical_cols),
        "counted_axis": str(counted_axis),
        "answer_value": int(answer),
        "canonical_bias_answer": int(canonical),
    }
    return CounterfactualBoardCase(
        style=str(style),
        rows=int(rows),
        cols=int(cols),
        board_kind=str(board_kind),
        counted_axis=str(counted_axis),
        prompt_query_key=str(prompt_query_key),
        answer_value=int(answer),
        canonical_bias_answer=int(canonical),
        case_params=case_params,
        execution_trace={
            "style_support": [str(style_id) for style_id in supported_styles],
            "row_support": [int(value) for value in spec.row_support],
            "column_support": [int(value) for value in spec.col_support],
        },
    )


__all__ = [
    "build_counterfactual_board_case",
    "sample_board_dimensions",
    "select_board_style",
]
