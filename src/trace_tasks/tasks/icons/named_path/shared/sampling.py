"""Scene-neutral sampling helpers for the named-path icons scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.named_colors import available_named_colors, named_color
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icon_field_scene import resolve_named_icon_fill_style_probabilities
from ...shared.procedural_named_icons import (
    DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS,
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    sample_procedural_named_icon_fill_style,
    validate_procedural_named_icon_fill_style_support,
)

from .defaults import NamedPathDefaults
from .state import IconPlan


_DEFAULTS = NamedPathDefaults()


def string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a JSON-stable uniform or one-hot probability map."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def shape_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the procedural shape support for path-stop icons."""

    raw = params.get("shape_id_support", group_default(gen_defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    if len(values) < 12:
        raise ValueError("named-path neighbor task needs at least twelve supported icon shapes")
    return values


def color_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the semantic named-color support for path-stop icons."""

    available = {str(name): tuple(int(channel) for channel in rgb) for name, rgb in available_named_colors()}
    raw = params.get("named_color_support", group_default(gen_defaults, "named_color_support", tuple(available)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("named_color_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip().lower() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(available))
    if unsupported:
        raise ValueError(f"unsupported named colors: {unsupported}")
    if len(values) < 4:
        raise ValueError("named-path neighbor task needs at least four named colors")
    return values


def fill_style_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the procedural fill-style support for path-stop icons."""

    raw = params.get(
        "named_icon_fill_style_support",
        group_default(gen_defaults, "named_icon_fill_style_support", _DEFAULTS.named_icon_fill_style_support),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = _DEFAULTS.named_icon_fill_style_support
    return validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in raw))


def fill_style_probability_map(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support: Sequence[str],
) -> Dict[str, float]:
    """Resolve fill-style probabilities over the selected support."""

    return dict(
        resolve_named_icon_fill_style_probabilities(
            params,
            gen_defaults,
            tuple(str(value) for value in support),
            default_weights=DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS,
        )
    )


def int_support_value(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    explicit_key: str,
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one integer from an inclusive support."""

    low = int(params.get(low_key, group_default(gen_defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(gen_defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key}")
    support = tuple(range(int(low), int(high) + 1))
    explicit = params.get(explicit_key)
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} is outside configured support")
        return int(value), dict(uniform_probability_map(support, selected=value))
    value = int(rng.choice(support))
    return int(value), dict(uniform_probability_map(support))


def sample_target_positions(
    rng,
    *,
    stop_count: int,
    target_occurrence_count: int,
    occurrence_rank: int,
    neighbor_direction: str,
) -> Tuple[Tuple[int, ...], int, int]:
    """Sample non-adjacent target occurrences and the requested neighbor stop."""

    if int(occurrence_rank) < 0 or int(occurrence_rank) >= int(target_occurrence_count):
        raise ValueError("occurrence rank is outside target occurrence support")
    direction = str(neighbor_direction)
    if direction not in {"before", "after"}:
        raise ValueError("neighbor_direction must be before or after")
    all_positions = tuple(range(1, int(stop_count) - 1))
    for _ in range(1000):
        target_positions = tuple(sorted(int(value) for value in rng.sample(all_positions, int(target_occurrence_count))))
        if any(int(b) - int(a) <= 1 for a, b in zip(target_positions, target_positions[1:])):
            continue
        selected_position = int(target_positions[int(occurrence_rank)])
        if direction == "before" and selected_position <= 1:
            continue
        if direction == "after" and selected_position >= int(stop_count) - 2:
            continue
        answer_position = int(selected_position + (1 if direction == "after" else -1))
        if answer_position < 0 or answer_position >= int(stop_count):
            continue
        if answer_position in (0, int(stop_count) - 1):
            continue
        if answer_position in set(target_positions):
            continue
        return target_positions, int(selected_position), int(answer_position)
    raise RuntimeError("failed to sample non-adjacent target positions for named path")


def sample_icon_plans(
    *,
    rng,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    stop_count: int,
    target_shape_id: str,
    target_positions: Sequence[int],
    selected_position: int,
    answer_position: int,
    labels_by_position: Mapping[int, str],
    fill_style_values: Sequence[str],
    fill_style_probabilities: Mapping[str, float],
    noise_namespace: str,
) -> Tuple[Tuple[IconPlan, ...], Tuple[Tuple[int, int, int], ...]]:
    """Sample path-stop icon semantics before rendering."""

    shapes = shape_support(params, gen_defaults)
    colors = color_support(params, gen_defaults)
    distractor_shapes = tuple(str(value) for value in shapes if str(value) != str(target_shape_id))
    if len(distractor_shapes) < 8:
        raise ValueError("named-path neighbor task needs at least eight non-target shapes")

    target_rank_by_position = {int(position): int(rank) for rank, position in enumerate(target_positions)}
    plans = []
    for position in range(int(stop_count)):
        if int(position) in target_rank_by_position:
            role = "query_occurrence" if int(position) == int(selected_position) else "target_occurrence"
            shape_id = str(target_shape_id)
        else:
            role = "answer_option" if int(position) == int(answer_position) else (
                "option" if int(position) in labels_by_position else "distractor"
            )
            shape_id = str(rng.choice(distractor_shapes))
        color_name = str(rng.choice(colors))
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{noise_namespace}:path_stop:{int(position)}",
            render_params=render_params,
        )
        low = int(render_params["scene_icon_size_min_px"])
        high = int(render_params["scene_icon_size_max_px"])
        nominal_size = int(rng.randint(min(low, high), max(low, high)))
        from ...shared.procedural_named_icon_field_scene import rotation_for_named_shape

        plans.append(
            IconPlan(
                position_index=int(position),
                role=str(role),
                label=str(labels_by_position.get(int(position), "")),
                shape_id=str(shape_id),
                color_name=str(color_name),
                tint_rgb=tuple(int(channel) for channel in named_color(str(color_name))),
                fill_style=sample_procedural_named_icon_fill_style(
                    rng,
                    support=tuple(str(value) for value in fill_style_values),
                    probabilities=dict(fill_style_probabilities),
                ),
                nominal_size_px=int(nominal_size),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    sampled_palette_rgb = tuple(tuple(int(channel) for channel in named_color(color_name)) for color_name in colors)
    return tuple(plans), sampled_palette_rgb


def display_shape_name(shape_id: str) -> str:
    """Return the prompt-facing procedural shape name."""

    return procedural_named_icon_display_name(str(shape_id))
