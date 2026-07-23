"""Configuration and deterministic default helpers for population-pyramid charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .....core.sampling import sample_without_replacement, uniform_choice
from .....core.scene_config import get_scene_defaults
from .....core.seed import spawn_rng
from ....shared.color_distance import sample_color_palette_with_distance_constraints
from ....shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.visual_defaults import load_chart_scene_noise_defaults

from .state import PROMPT_BUNDLE_ID, RGB, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def support_probability_map(values: Sequence[int | str]) -> dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): float(weight) for value in support}


def choose_from_values(
    params: Mapping[str, Any],
    *,
    values: Sequence[int | str],
    instance_seed: int,
    namespace: str,
) -> int | str:
    candidates = tuple(values)
    if not candidates:
        raise ValueError(f"empty support for {namespace}")
    return uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        candidates,
        sort_keys=True,
    )


def resolve_row_count(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, dict[str, float]]:
    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="row_count_min",
        max_key="row_count_max",
        fallback_min=8,
        fallback_max=14,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    support = tuple(range(int(low), int(high) + 1))
    selected = int(
        choose_from_values(
            params,
            values=support,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.row_count",
        )
    )
    return selected, support_probability_map(support)


def resolve_series_labels(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, str, dict[str, Any]]:
    label_pairs = params.get("series_label_pairs", group_default(GEN_DEFAULTS, "series_label_pairs", (("Female", "Male"),)))
    normalized: list[tuple[str, str]] = []
    if isinstance(label_pairs, Sequence) and not isinstance(label_pairs, (str, bytes)):
        for pair in label_pairs:
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) >= 2:
                normalized.append((str(pair[0]), str(pair[1])))  # type: ignore[index]
    if not normalized:
        normalized = [("Female", "Male")]
    selected = uniform_choice(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_labels"),
        tuple(normalized),
        sort_keys=True,
    )
    return str(selected[0]), str(selected[1]), {
        "series_label_pairs": [[left, right] for left, right in normalized],
        "series_label_pair_probabilities": {
            f"{left}|{right}": 1.0 / float(len(normalized))
            for left, right in normalized
        },
    }


def resolve_series_colors(params: Mapping[str, Any], *, instance_seed: int) -> tuple[RGB, RGB]:
    configured = params.get("series_palette_rgb", group_default(RENDER_DEFAULTS, "series_palette_rgb", None))
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        colors: list[RGB] = []
        for raw in configured:
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
                colors.append(tuple(max(0, min(255, int(channel))) for channel in raw[:3]))  # type: ignore[arg-type]
        if len(colors) >= 2:
            left, right = sample_without_replacement(
                spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette"),
                tuple(colors),
                2,
            )
            return left, right
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette")
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=2,
        channel_min=20,
        channel_max=210,
        anchor_colors=((255, 255, 255), (20, 20, 20)),
        min_distance=46.0,
        distance_space="lab",
    )
    return tuple(palette[0]), tuple(palette[1])


def sample_age_labels(params: Mapping[str, Any], *, row_count: int, instance_seed: int) -> tuple[tuple[str, ...], dict[str, Any]]:
    band_width_options = tuple(
        int(value)
        for value in params.get("age_band_width_options", group_default(GEN_DEFAULTS, "age_band_width_options", (5, 10)))
    )
    start_options = tuple(
        int(value)
        for value in params.get("age_start_options", group_default(GEN_DEFAULTS, "age_start_options", (0, 5, 10)))
    )
    band_width = int(
        choose_from_values(
            params,
            values=band_width_options,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.age_band_width",
        )
    )
    start_age = int(
        choose_from_values(
            params,
            values=start_options,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.age_start",
        )
    )
    labels = [
        f"{start_age + (index * band_width)}-{start_age + ((index + 1) * band_width) - 1}"
        for index in range(int(row_count))
    ]
    labels = list(reversed(labels))
    return tuple(labels), {
        "age_band_width": int(band_width),
        "age_start": int(start_age),
        "age_band_width_probabilities": support_probability_map(band_width_options),
        "age_start_probabilities": support_probability_map(start_options),
    }


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    title_options = params.get("title_options", group_default(RENDER_DEFAULTS, "title_options", ("Population Pyramid",)))
    if not isinstance(title_options, Sequence) or isinstance(title_options, (str, bytes)) or not title_options:
        title_options = ("Population Pyramid",)
    return str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title"),
            tuple(str(value) for value in title_options),
        )
    )


def prompt_bundle_id() -> str:
    return str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID))
