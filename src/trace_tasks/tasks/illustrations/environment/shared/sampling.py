"""Sampling and render-default helpers for environment illustrations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....shared.config_defaults import group_default
from ...shared.object_library import STYLE_IDS
from ...shared.canvas_profiles import resolve_profile_render_params
from ...shared.style_registry import resolve_art_style_weights
from ...shared.task_support import uniform_string_probability_map

from .defaults import CountContractDefaults
from .rendering import ENVIRONMENT_THEME_IDS, effective_environment_object_count
from .state import EnvironmentChoice


FEATURE_TYPES_BY_THEME: Dict[str, tuple[str, ...]] = {
    "park_road": ("road",),
    "river_meadow": ("river",),
    "road_and_river": ("road", "river"),
    "canal_city": ("river",),
    "skyline_street": ("road",),
}
RELATION_SUPPORT: Tuple[str, ...] = ("above", "below", "on")
CROSSING_THEME_SUPPORT: Dict[str, Tuple[str, ...]] = {
    "bridge": ("river_meadow", "road_and_river", "canal_city"),
    "crosswalk": ("park_road", "road_and_river", "skyline_street"),
}
CITY_THEME_SUPPORT: Tuple[str, ...] = ("canal_city", "skyline_street")
WINDOW_MODE_SUPPORT: Tuple[str, ...] = ("lit",)

ENVIRONMENT_SETTING_NAMES: Dict[str, str] = {
    "park_road": "a park road setting",
    "river_meadow": "a meadow river setting",
    "road_and_river": "an outdoor setting with both a road and a river",
    "canal_city": "a city canal setting",
    "skyline_street": "a city street setting",
}


def style_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    """Resolve render-only art-style weights for the environment scene."""

    return resolve_art_style_weights(params, render_defaults, style_ids=STYLE_IDS)


def environment_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
    instance_seed: int | None = None,
    namespace: str = "environment:canvas_profile",
) -> Dict[str, Any]:
    """Resolve scene-level environment rendering parameters from task and scene defaults."""

    profile_params = resolve_profile_render_params(
        params,
        render_defaults,
        prefix="environment",
        fallback_width=int(fallback["canvas_width"]),
        fallback_height=int(fallback["canvas_height"]),
        fallback_scale=int(fallback["render_scale"]),
        instance_seed=instance_seed,
        namespace=namespace,
    )
    return {
        "canvas_width": int(profile_params["canvas_width"]),
        "canvas_height": int(profile_params["canvas_height"]),
        "canvas_profile": str(profile_params["canvas_profile"]),
        "canvas_profile_size": list(profile_params["canvas_profile_size"]),
        "canvas_profile_probabilities": dict(profile_params["canvas_profile_probabilities"]),
        "object_size_min_px": int(
            params.get(
                "object_size_min_px",
                group_default(render_defaults, "environment_object_size_min_px", int(fallback["object_size_min_px"])),
            )
        ),
        "object_size_max_px": int(
            params.get(
                "object_size_max_px",
                group_default(render_defaults, "environment_object_size_max_px", int(fallback["object_size_max_px"])),
            )
        ),
        "min_gap_px": int(params.get("min_gap_px", group_default(render_defaults, "environment_min_gap_px", int(fallback["min_gap_px"])))),
        "max_overlap_fraction": float(
            params.get(
                "max_overlap_fraction",
                group_default(render_defaults, "environment_max_overlap_fraction", float(fallback["max_overlap_fraction"])),
            )
        ),
        "placement_max_attempts": int(
            params.get(
                "placement_max_attempts",
                group_default(render_defaults, "environment_placement_max_attempts", int(fallback["placement_max_attempts"])),
            )
        ),
        "render_scale": int(profile_params["render_scale"]),
        "skyline_building_min": int(params.get("skyline_building_min", group_default(render_defaults, "skyline_building_min", int(fallback.get("skyline_building_min", 7))))),
        "skyline_building_max": int(params.get("skyline_building_max", group_default(render_defaults, "skyline_building_max", int(fallback.get("skyline_building_max", 14))))),
    }


def environment_setting_name(theme_id: str) -> str:
    """Return prompt-facing text for one environment theme."""

    return ENVIRONMENT_SETTING_NAMES.get(str(theme_id), "an illustrated outdoor scene")


def theme_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    fallback: Sequence[str] = ENVIRONMENT_THEME_IDS,
) -> tuple[str, ...]:
    """Resolve supported environment themes from params/defaults."""

    raw = params.get("theme_support", group_default(generation_defaults, "theme_support", tuple(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("theme_support must be a sequence")
    supported = tuple(str(value) for value in raw if str(value) in set(ENVIRONMENT_THEME_IDS))
    if not supported:
        raise ValueError("theme_support resolved no supported environment themes")
    return tuple(dict.fromkeys(supported))


def global_feature_type_probabilities(themes: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return road/river probabilities induced by the configured theme support."""

    if selected is not None:
        return {str(selected): 1.0}
    weights = {"road": 0.0, "river": 0.0}
    for theme in themes:
        support = FEATURE_TYPES_BY_THEME[str(theme)]
        probability = 1.0 / float(len(support))
        for feature_type in support:
            weights[str(feature_type)] += probability
    total = sum(weights.values())
    return {key: value / total for key, value in sorted(weights.items()) if value > 0.0}


def int_bounds(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> tuple[int, int]:
    """Resolve an inclusive integer range from params/defaults/fallbacks."""

    if "target_count_min" in params or "target_count_max" in params:
        low = int(params.get("target_count_min", fallback_low))
        high = int(params.get("target_count_max", fallback_high))
    else:
        low = int(params.get(low_key, group_default(generation_defaults, low_key, fallback_low)))
        high = int(params.get(high_key, group_default(generation_defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} range")
    return int(low), int(high)


def sample_count_support(
    *,
    params: Mapping[str, Any],
    support: Sequence[int],
    explicit_key: str,
    instance_seed: int,
    namespace: str,
) -> tuple[int, Dict[str, float]]:
    """Sample one count from a configured support range."""

    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError(f"{explicit_key} has empty support")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(values):
            raise ValueError(f"{explicit_key} is outside configured support")
        return int(value), dict(support_probability_map(values, selected=int(value), sort_keys=True))
    rng = spawn_rng(int(instance_seed), str(namespace))
    value, probabilities = uniform_choice_with_probabilities(rng, values, sort_keys=True)
    return int(value), dict(probabilities)


def sample_object_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
) -> tuple[int, Dict[str, float]]:
    """Resolve and sample the requested environment foreground-object count."""

    low = int(params.get("object_count_min", group_default(generation_defaults, "object_count_min", int(fallback_min))))
    high = int(params.get("object_count_max", group_default(generation_defaults, "object_count_max", int(fallback_max))))
    if low < 0 or high < low:
        raise ValueError("invalid object_count_min/object_count_max range")
    return sample_count_support(
        params=params,
        support=tuple(range(int(low), int(high) + 1)),
        explicit_key="object_count",
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def sample_scene_object_count(
    params: Mapping[str, Any],
    instance_seed: int,
    _choice: EnvironmentChoice,
    generation_defaults: Mapping[str, Any],
    *,
    public_id: str,
    defaults: CountContractDefaults,
) -> tuple[int, Dict[str, float]]:
    """Resolve the foreground clutter count independent of the answer count."""

    return sample_object_count(
        params,
        generation_defaults,
        fallback_min=int(defaults.object_count_min),
        fallback_max=int(defaults.object_count_max),
        instance_seed=int(instance_seed),
        namespace=f"{public_id}:object_count",
    )


def sample_target_count_by_keys(
    params: Mapping[str, Any],
    instance_seed: int,
    choice: EnvironmentChoice,
    generation_defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    defaults: CountContractDefaults,
) -> tuple[int, Dict[str, float]]:
    """Sample the requested answer support from task-specific count bounds."""

    low, high = int_bounds(
        params,
        generation_defaults,
        low_key=str(low_key),
        high_key=str(high_key),
        fallback_low=int(defaults.target_count_min),
        fallback_high=int(defaults.target_count_max),
    )
    return sample_count_support(
        params=params,
        support=tuple(range(int(low), int(high) + 1)),
        explicit_key="target_count",
        instance_seed=int(instance_seed),
        namespace=f"{choice.theme_id}:{low_key}:{high_key}:target_count",
    )


def relation_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the allowed feature-relation operands for foreground object counts."""

    raw = params.get("relation_support", group_default(generation_defaults, "relation_support", RELATION_SUPPORT))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("relation_support must be a sequence")
    support = tuple(str(value) for value in raw if str(value) in set(RELATION_SUPPORT))
    if not support:
        raise ValueError("relation_support resolved no supported relations")
    return tuple(dict.fromkeys(support))


def resolve_feature_choice(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    public_id: str,
    include_relation: bool,
) -> EnvironmentChoice:
    """Resolve the sampled theme, road/river feature, and optional side relation."""

    themes = theme_support(params, generation_defaults)
    branch_index = int(spawn_rng(int(instance_seed), f"{public_id}:cycle").randrange(1 << 30))
    explicit_theme = params.get("theme_id")
    if explicit_theme is not None:
        theme_id = str(explicit_theme)
        if theme_id not in set(themes):
            raise ValueError(f"theme_id must be one of {themes}")
        theme_probabilities = uniform_string_probability_map(themes, selected=theme_id)
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:theme")
        theme_id, theme_probabilities = uniform_choice_with_probabilities(rng, themes, sort_keys=False)
        theme_id = str(theme_id)

    feature_values = FEATURE_TYPES_BY_THEME[str(theme_id)]
    explicit_feature = params.get("feature_type")
    if explicit_feature is not None:
        feature_type = str(explicit_feature)
        if feature_type not in set(feature_values):
            raise ValueError(f"feature_type {feature_type!r} is not available for theme {theme_id!r}")
        feature_probabilities = global_feature_type_probabilities(themes, selected=feature_type)
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:feature_type")
        feature_type, _feature_local_probabilities = uniform_choice_with_probabilities(
            rng,
            feature_values,
            sort_keys=False,
        )
        feature_type = str(feature_type)
        feature_probabilities = global_feature_type_probabilities(themes)

    relation = None
    relation_probabilities = None
    if include_relation:
        relations = relation_support(params, generation_defaults)
        explicit_relation = params.get("relation")
        if explicit_relation is not None:
            relation = str(explicit_relation)
            if relation not in set(relations):
                raise ValueError(f"relation must be one of {relations}")
            relation_probabilities = uniform_string_probability_map(relations, selected=relation)
        else:
            rng = spawn_rng(int(instance_seed), f"{public_id}:relation")
            relation, relation_probabilities = uniform_choice_with_probabilities(
                rng,
                relations,
                sort_keys=False,
            )
            relation = str(relation)

    return EnvironmentChoice(
        branch_index=int(branch_index),
        theme_id=str(theme_id),
        theme_probabilities=dict(theme_probabilities),
        feature_type=str(feature_type),
        feature_type_probabilities=dict(feature_probabilities),
        relation=relation,
        relation_probabilities=relation_probabilities,
    )


def crossing_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve supported bridge/crosswalk operands for crossing counts."""

    raw = params.get(
        "crossing_type_support",
        group_default(generation_defaults, "crossing_type_support", tuple(CROSSING_THEME_SUPPORT)),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("crossing_type_support must be a sequence")
    support = tuple(str(value) for value in raw if str(value) in set(CROSSING_THEME_SUPPORT))
    if not support:
        raise ValueError("crossing_type_support resolved no supported crossing types")
    return tuple(dict.fromkeys(support))


def resolve_crossing_choice(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    public_id: str,
) -> EnvironmentChoice:
    """Resolve the crossing type and compatible environment theme."""

    crossing_values = crossing_support(params, generation_defaults)
    branch_index = int(spawn_rng(int(instance_seed), f"{public_id}:cycle").randrange(1 << 30))
    explicit_crossing = params.get("crossing_type")
    if explicit_crossing is not None:
        crossing_type = str(explicit_crossing)
        if crossing_type not in set(crossing_values):
            raise ValueError(f"crossing_type must be one of {crossing_values}")
        crossing_probabilities = uniform_string_probability_map(crossing_values, selected=crossing_type)
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:crossing_type")
        crossing_type, crossing_probabilities = uniform_choice_with_probabilities(
            rng,
            crossing_values,
            sort_keys=False,
        )
        crossing_type = str(crossing_type)

    theme_options = CROSSING_THEME_SUPPORT[str(crossing_type)]
    explicit_theme = params.get("theme_id")
    if explicit_theme is not None:
        theme_id = str(explicit_theme)
        if theme_id not in set(theme_options):
            raise ValueError(f"theme_id must be one of {theme_options} for crossing_type {crossing_type!r}")
        theme_probabilities = {theme_id: 1.0}
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:theme")
        theme_id, _theme_local_probabilities = uniform_choice_with_probabilities(
            rng,
            theme_options,
            sort_keys=False,
        )
        theme_id = str(theme_id)
        theme_probabilities: Dict[str, float] = {}
        for crossing in crossing_values:
            crossing_probability = 1.0 / float(len(crossing_values))
            options = CROSSING_THEME_SUPPORT[str(crossing)]
            theme_probability = crossing_probability / float(len(options))
            for theme in options:
                theme_probabilities[str(theme)] = float(theme_probabilities.get(str(theme), 0.0)) + float(theme_probability)

    return EnvironmentChoice(
        branch_index=int(branch_index),
        theme_id=str(theme_id),
        theme_probabilities=dict(sorted(theme_probabilities.items())),
        crossing_type=str(crossing_type),
        crossing_type_probabilities=dict(crossing_probabilities),
    )


def window_mode_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve supported building-window modes for the window count contract."""

    raw = params.get("window_mode_support", group_default(generation_defaults, "window_mode_support", WINDOW_MODE_SUPPORT))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("window_mode_support must be a sequence")
    supported = tuple(str(value) for value in raw if str(value) == "lit")
    if not supported:
        raise ValueError("window_mode_support resolved no supported modes")
    return tuple(dict.fromkeys(supported))


def resolve_window_choice(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    public_id: str,
) -> EnvironmentChoice:
    """Resolve a city theme and the fixed lit-window predicate."""

    themes = theme_support(params, generation_defaults, fallback=CITY_THEME_SUPPORT)
    themes = tuple(theme for theme in themes if theme in set(CITY_THEME_SUPPORT))
    if not themes:
        raise ValueError("lit-window count requires canal_city or skyline_street theme support")
    modes = window_mode_support(params, generation_defaults)
    branch_index = int(spawn_rng(int(instance_seed), f"{public_id}:cycle").randrange(1 << 30))

    explicit_theme = params.get("theme_id")
    if explicit_theme is not None:
        theme_id = str(explicit_theme)
        if theme_id not in set(themes):
            raise ValueError(f"theme_id must be one of {themes}")
        theme_probabilities = uniform_string_probability_map(themes, selected=theme_id)
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:theme")
        theme_id, theme_probabilities = uniform_choice_with_probabilities(rng, themes, sort_keys=False)
        theme_id = str(theme_id)

    explicit_mode = params.get("window_mode")
    if explicit_mode is not None:
        window_mode = str(explicit_mode)
        if window_mode not in set(modes):
            raise ValueError(f"window_mode must be one of {modes}")
        window_mode_probabilities = uniform_string_probability_map(modes, selected=window_mode)
    else:
        rng = spawn_rng(int(instance_seed), f"{public_id}:window_mode")
        window_mode, window_mode_probabilities = uniform_choice_with_probabilities(
            rng,
            modes,
            sort_keys=False,
        )
        window_mode = str(window_mode)

    return EnvironmentChoice(
        branch_index=int(branch_index),
        theme_id=str(theme_id),
        theme_probabilities=dict(theme_probabilities),
        window_mode=str(window_mode),
        window_mode_probabilities=dict(window_mode_probabilities),
    )


def capped_object_count_probabilities(
    requested_probabilities: Mapping[str, float],
    theme_probabilities: Mapping[str, float],
) -> Dict[str, float]:
    """Map requested object-count probabilities through theme-specific caps."""

    normalized_theme_probabilities = {
        str(theme): max(0.0, float(probability))
        for theme, probability in theme_probabilities.items()
        if float(probability) > 0.0
    }
    if not normalized_theme_probabilities:
        normalized_theme_probabilities = {str(ENVIRONMENT_THEME_IDS[0]): 1.0}
    theme_total = sum(float(value) for value in normalized_theme_probabilities.values())
    capped: Dict[str, float] = {}
    for theme, theme_probability in normalized_theme_probabilities.items():
        theme_weight = float(theme_probability) / max(1e-9, float(theme_total))
        for requested_count, probability in requested_probabilities.items():
            actual_count = effective_environment_object_count(str(theme), int(requested_count))
            capped[str(actual_count)] = float(capped.get(str(actual_count), 0.0)) + float(theme_weight) * float(probability)
    return dict(sorted(capped.items(), key=lambda item: int(item[0])))


__all__ = [
    "ENVIRONMENT_SETTING_NAMES",
    "CITY_THEME_SUPPORT",
    "CROSSING_THEME_SUPPORT",
    "FEATURE_TYPES_BY_THEME",
    "RELATION_SUPPORT",
    "WINDOW_MODE_SUPPORT",
    "capped_object_count_probabilities",
    "crossing_support",
    "environment_render_params",
    "environment_setting_name",
    "global_feature_type_probabilities",
    "int_bounds",
    "relation_support",
    "sample_count_support",
    "sample_object_count",
    "sample_scene_object_count",
    "sample_target_count_by_keys",
    "resolve_crossing_choice",
    "resolve_feature_choice",
    "resolve_window_choice",
    "style_weights",
    "theme_support",
    "window_mode_support",
]
