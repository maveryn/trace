"""Defaults and neutral config helpers for part-whole charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.balanced_sampling import balanced_int_from_support
from trace_tasks.tasks.charts.shared.labeled_chart_defaults import LabeledChartDefaults
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant
from trace_tasks.tasks.charts.shared.visual_defaults import load_chart_scene_background_defaults, load_chart_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults


DOMAIN = "charts"
SCENE_ID = "part_whole"
SCENE_NAMESPACE = "charts_part_whole"
SAMPLING_NAMESPACE = "charts_part_whole"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("pie", "donut")
ORDER_DIRECTIONS: tuple[str, ...] = ("clockwise", "counterclockwise")

DEFAULTS = LabeledChartDefaults(canvas_width=1280, canvas_height=920)
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    """Select pie or donut scene grammar with shared chart-axis policy."""

    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        task_id=SCENE_NAMESPACE,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def scene_axis_stride(params: Mapping[str, Any]) -> int:
    """Return the cursor stride consumed by scene-variant cycling."""

    if params.get("scene_variant") is not None or params.get("scene_variant_weights") is not None:
        return 1
    return len(SUPPORTED_SCENE_VARIANTS)


def shifted_cursor_params(params: Mapping[str, Any], *, divisor: int) -> dict[str, Any]:
    """Return params after removing the scene-axis cursor slot."""

    shifted = dict(params)
    if "_sample_cursor" not in shifted:
        return shifted
    shifted["_sample_cursor"] = abs(int(shifted["_sample_cursor"])) // max(1, int(divisor))
    return shifted


def resolve_count_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    """Resolve one integer min/max pair from scene generation defaults."""

    min_value = int(params.get(str(min_key), group_default(GEN_DEFAULTS, str(min_key), int(fallback_min))))
    max_value = int(params.get(str(max_key), group_default(GEN_DEFAULTS, str(max_key), int(fallback_max))))
    if int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(min_value), int(max_value)


def balanced_int(
    values: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Sample one integer with the repo's balanced-support helper."""

    return balanced_int_from_support(
        [int(value) for value in values],
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def configured_int_values(params: Mapping[str, Any], *, key: str, fallback: Sequence[int]) -> tuple[int, ...]:
    """Resolve a configured integer support list."""

    raw = params.get(str(key), group_default(GEN_DEFAULTS, str(key), None))
    if raw is None:
        raw = tuple(int(value) for value in fallback)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        values = []
    parsed = tuple(sorted({int(value) for value in values if str(value).strip()}))
    if not parsed:
        raise ValueError(f"{key} must contain at least one integer")
    return tuple(int(value) for value in parsed)


def format_quoted(values: Sequence[str]) -> str:
    """Format visible category labels for prompt slots."""

    return ", ".join(f'"{str(value)}"' for value in values)


def chart_order_phrase(scene_variant: str) -> str:
    """Return the default chart-order wording for a circular chart."""

    if str(scene_variant) in {"pie", "donut"}:
        return "clockwise"
    return "in chart order"
