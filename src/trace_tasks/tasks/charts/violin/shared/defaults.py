"""Shared defaults for violin chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.distribution.config import DistributionChartDefaults
from trace_tasks.tasks.charts.shared.labeled_chart_defaults import LabeledChartDefaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default as _group_default,
    load_scene_generation_rendering_prompt_defaults,
)


DOMAIN = "charts"
SCENE_ID = "violin"
SCENE_NAMESPACE = "charts.violin"
DATASET_NAMESPACE = "charts_violin_base"
SCENE_VARIANT = "violin"
PROMPT_BUNDLE_ID = "charts_violin_v1"

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)
DISTRIBUTION_DEFAULTS = DistributionChartDefaults()
RENDER_FALLBACKS = LabeledChartDefaults()
POST_IMAGE_BACKGROUND_DEFAULTS: dict[str, Any] = load_chart_scene_background_defaults(
    scene_id=SCENE_ID
)
POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = load_chart_scene_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.0,
)


def generation_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one violin generation default."""

    return params.get(str(key), _group_default(GENERATION_DEFAULTS, str(key), fallback))


__all__ = [
    "DATASET_NAMESPACE",
    "DISTRIBUTION_DEFAULTS",
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "RENDER_FALLBACKS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "generation_value",
]
