"""Defaults and neutral variant resolution for single-series charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.labeled_chart_defaults import LabeledChartDefaults
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import SCENE_ID, SCENE_NAMESPACE


DEFAULTS = LabeledChartDefaults()
SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {}
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

COUNT_SCENE_VARIANTS: tuple[str, ...] = (
    "area",
    "bar",
    "line",
    "scatter",
    "horizontal_bar",
    "dot_plot",
    "lollipop",
)
ORDERED_SCENE_VARIANTS: tuple[str, ...] = (
    "area",
    "bar",
    "horizontal_bar",
    "line",
    "dot_plot",
    "lollipop",
)
THRESHOLD_SCENE_VARIANTS: tuple[str, ...] = (
    "area",
    "bar",
    "line",
    "dot_plot",
    "lollipop",
)
SUMMARY_SCENE_VARIANTS: tuple[str, ...] = (
    "area",
    "bar",
    "line",
    "scatter",
    "horizontal_bar",
    "dot_plot",
    "lollipop",
)
COUNTERFACTUAL_SCENE_VARIANTS: tuple[str, ...] = ("bar", "dot_plot", "lollipop")


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    supported: Sequence[str],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one visual chart variant without inspecting public task identity."""

    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in supported),
        namespace=f"{SCENE_NAMESPACE}.{str(namespace)}",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


__all__ = [
    "COUNTERFACTUAL_SCENE_VARIANTS",
    "COUNT_SCENE_VARIANTS",
    "DEFAULTS",
    "GEN_DEFAULTS",
    "ORDERED_SCENE_VARIANTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SUMMARY_SCENE_VARIANTS",
    "THRESHOLD_SCENE_VARIANTS",
    "resolve_scene_variant",
]
