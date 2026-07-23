"""Shared axis and support sampling helpers for Bingo scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.style import SUPPORTED_BINGO_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .rendering import SUPPORTED_BINGO_CELL_FILL_PATTERNS, SUPPORTED_BINGO_MARK_SHAPES
from .state import SUPPORTED_BINGO_LINE_AXES, SUPPORTED_BINGO_SCENE_VARIANTS
from .defaults import SCENE_ID


_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class ResolvedBingoSceneAxes:
    """Resolved visual/style axes shared by Bingo tasks."""

    scene_variant: str
    style_variant: str
    mark_shape: str
    cell_fill_pattern: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    mark_shape_probabilities: Dict[str, float]
    cell_fill_pattern_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBingoTarget:
    """Resolved integer target and support metadata."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"games.bingo.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=tuple(str(value) for value in supported),
        explicit_key=explicit_key,
        weights_key=weights_key,
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=balance_flag_key,
        explicit_key=explicit_key,
        weights_key=weights_key,
        sampling_namespace=f"games.bingo.{namespace}",
    )
    return str(selected), dict(probabilities)


def resolve_bingo_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> ResolvedBingoSceneAxes:
    """Resolve shared visual axes for one Bingo scene."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BINGO_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BINGO_STYLE_VARIANTS,
    )
    mark_shape, mark_shape_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="mark_shape",
        explicit_key="mark_shape",
        weights_key="mark_shape_weights",
        balance_flag_key="balanced_mark_shape_sampling",
        supported=SUPPORTED_BINGO_MARK_SHAPES,
    )
    cell_fill_pattern, cell_fill_pattern_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="cell_fill_pattern",
        explicit_key="cell_fill_pattern",
        weights_key="cell_fill_pattern_weights",
        balance_flag_key="balanced_cell_fill_pattern_sampling",
        supported=SUPPORTED_BINGO_CELL_FILL_PATTERNS,
    )
    return ResolvedBingoSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        mark_shape=str(mark_shape),
        cell_fill_pattern=str(cell_fill_pattern),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        mark_shape_probabilities=dict(mark_shape_probabilities),
        cell_fill_pattern_probabilities=dict(cell_fill_pattern_probabilities),
    )


def resolve_bingo_line_axis(instance_seed: int, *, params: Mapping[str, Any]) -> tuple[str, Dict[str, float]]:
    """Resolve a row/column line axis."""

    return _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="line_axis",
        explicit_key="line_axis",
        weights_key="line_axis_weights",
        balance_flag_key="balanced_line_axis_sampling",
        supported=SUPPORTED_BINGO_LINE_AXES,
    )


def resolve_bingo_integer_target(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    explicit_key: str = "target_answer",
    balanced_flag_key: str = "balanced_target_answer_sampling",
) -> ResolvedBingoTarget:
    """Resolve a task-owned integer target and support metadata."""

    defaults = gen_defaults if gen_defaults is not None else _GEN_DEFAULTS
    target_answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return ResolvedBingoTarget(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in support),
        target_answer_probabilities=dict(probabilities),
    )


def resolve_bingo_float_param(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any] | None = None,
    key: str,
    fallback: float,
) -> float:
    """Resolve one float generation parameter from params/defaults/fallback."""

    defaults = gen_defaults if gen_defaults is not None else _GEN_DEFAULTS
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


__all__ = [
    "ResolvedBingoSceneAxes",
    "ResolvedBingoTarget",
    "resolve_bingo_float_param",
    "resolve_bingo_integer_target",
    "resolve_bingo_line_axis",
    "resolve_bingo_scene_axes",
]
