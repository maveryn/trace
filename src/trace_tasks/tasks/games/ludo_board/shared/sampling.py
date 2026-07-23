"""Identity-free sampling helpers for Ludo board scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .rules import roll_option_text
from .state import (
    MAIN_PATH,
    OPTION_LABELS,
    PLAYER_COLORS,
    START_COORDS,
    Coord,
    LudoDestinationOption,
    LudoRollOption,
    LudoSceneAxes,
    STYLE_VARIANTS,
)


@dataclass(frozen=True)
class IntegerAxisSelection:
    """Resolved integer value plus support metadata for trace payloads."""

    value: int
    support: Tuple[int, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class LabelAxisSelection:
    """Resolved option label plus support metadata for trace payloads."""

    value: str
    support: Tuple[str, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class LabeledOptionAxes:
    """Resolved option-count and answer-label axes for visual option tasks."""

    option_count: int
    option_count_support: Tuple[int, ...]
    option_count_probabilities: Mapping[str, float]
    answer_label: str
    option_labels: Tuple[str, ...]
    answer_label_probabilities: Mapping[str, float]

    def trace_params(self) -> dict[str, Any]:
        """Return option-axis metadata with stable trace field names."""

        return {
            "option_count": int(self.option_count),
            "option_count_support": [int(value) for value in self.option_count_support],
            "option_count_probabilities": dict(self.option_count_probabilities),
            "answer_option_label": str(self.answer_label),
            "answer_option_label_support": list(self.option_labels),
            "answer_option_label_probabilities": dict(self.answer_label_probabilities),
        }


@dataclass(frozen=True)
class ValueOptionAxisConfig:
    """Config keys for one integer target axis plus visual option axes."""

    value_support_key: str
    value_explicit_key: str
    value_fallback_support: Tuple[int, ...]
    value_namespace: str
    value_balance_flag_key: str
    option_count_support_key: str
    option_count_fallback_support: Tuple[int, ...]
    option_namespace: str
    option_count_balance_flag_key: str


@dataclass(frozen=True)
class ValueOptionAxes:
    """Resolved integer target axis bundled with visual option axes."""

    value_axis: IntegerAxisSelection
    option_axes: LabeledOptionAxes


def make_ludo_value_option_axis_config(
    *,
    value: tuple[str, str, Sequence[int], str, str],
    options: tuple[str, Sequence[int], str, str],
) -> ValueOptionAxisConfig:
    """Build value+option axis config from ordered task-owned key tuples."""

    value_support_key, value_explicit_key, value_fallback_support, value_namespace, value_balance_flag_key = value
    option_count_support_key, option_count_fallback_support, option_namespace, option_count_balance_flag_key = options
    return ValueOptionAxisConfig(
        value_support_key=str(value_support_key),
        value_explicit_key=str(value_explicit_key),
        value_fallback_support=tuple(int(item) for item in value_fallback_support),
        value_namespace=str(value_namespace),
        value_balance_flag_key=str(value_balance_flag_key),
        option_count_support_key=str(option_count_support_key),
        option_count_fallback_support=tuple(int(item) for item in option_count_fallback_support),
        option_namespace=str(option_namespace),
        option_count_balance_flag_key=str(option_count_balance_flag_key),
    )


def resolve_ludo_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    axis_name: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve a scene-local named axis using the shared games sampler."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.{axis_name}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def resolve_ludo_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> LudoSceneAxes:
    """Resolve shared visual and player-color axes used by every Ludo task."""

    style_variant, style_probs = resolve_ludo_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    query_color, query_color_probs = resolve_ludo_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="query_color",
        explicit_key="query_color",
        weights_key="query_color_weights",
        balance_flag_key="balanced_query_color_sampling",
        supported=PLAYER_COLORS,
    )
    target_color, target_color_probs = resolve_ludo_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="target_color",
        explicit_key="target_color",
        weights_key="target_color_weights",
        balance_flag_key="balanced_target_color_sampling",
        supported=PLAYER_COLORS,
    )
    if str(target_color) == str(query_color):
        rng = spawn_rng(int(instance_seed), f"{str(namespace)}.fallback_target_color")
        alternatives = tuple(str(color) for color in PLAYER_COLORS if str(color) != str(query_color))
        target_color = str(rng.choice(alternatives))
    return LudoSceneAxes(
        style_variant=str(style_variant),
        query_color=str(query_color),
        target_color=str(target_color),
        style_variant_probabilities=dict(style_probs),
        query_color_probabilities=dict(query_color_probs),
        target_color_probabilities=dict(target_color_probs),
    )


def resolve_ludo_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> IntegerAxisSelection:
    """Resolve one integer axis and preserve its support/probability metadata."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(item) for item in fallback_support),
    )
    return IntegerAxisSelection(value=int(value), support=tuple(int(item) for item in support), probabilities=dict(probabilities))


def resolve_ludo_option_label_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    option_count: int,
    namespace: str,
) -> LabelAxisSelection:
    """Resolve the correct option letter from the first N Ludo option labels."""

    support = tuple(OPTION_LABELS[: int(option_count)]) if int(option_count) > 0 else tuple(OPTION_LABELS)
    value, probabilities = resolve_ludo_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="answer_option_label",
        explicit_key="answer_option_label",
        weights_key="answer_option_label_weights",
        balance_flag_key="balanced_answer_option_label_sampling",
        supported=support,
    )
    return LabelAxisSelection(value=str(value), support=tuple(support), probabilities=dict(probabilities))


def resolve_ludo_labeled_option_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    option_count_support_key: str,
    option_count_explicit_key: str,
    fallback_option_count_support: Sequence[int],
    namespace: str,
    balanced_option_count_flag_key: str,
) -> LabeledOptionAxes:
    """Resolve option count and answer letter for Ludo visual-option objectives."""

    count_axis = resolve_ludo_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(option_count_support_key),
        explicit_key=str(option_count_explicit_key),
        fallback_support=tuple(int(value) for value in fallback_option_count_support),
        namespace=f"{namespace}.option_count",
        balanced_flag_key=str(balanced_option_count_flag_key),
    )
    label_axis = resolve_ludo_option_label_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        option_count=int(count_axis.value),
        namespace=f"{namespace}.option_label",
    )
    return LabeledOptionAxes(
        option_count=int(count_axis.value),
        option_count_support=tuple(int(value) for value in count_axis.support),
        option_count_probabilities=dict(count_axis.probabilities),
        answer_label=str(label_axis.value),
        option_labels=tuple(str(value) for value in label_axis.support),
        answer_label_probabilities=dict(label_axis.probabilities),
    )


def resolve_ludo_value_option_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    config: ValueOptionAxisConfig,
) -> ValueOptionAxes:
    """Resolve one integer target axis and the associated labeled option axes."""

    value_axis = resolve_ludo_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(config.value_support_key),
        explicit_key=str(config.value_explicit_key),
        fallback_support=tuple(int(value) for value in config.value_fallback_support),
        namespace=str(config.value_namespace),
        balanced_flag_key=str(config.value_balance_flag_key),
    )
    option_axes = resolve_ludo_labeled_option_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        option_count_support_key=str(config.option_count_support_key),
        option_count_explicit_key="option_count",
        fallback_option_count_support=tuple(int(value) for value in config.option_count_fallback_support),
        namespace=str(config.option_namespace),
        balanced_option_count_flag_key=str(config.option_count_balance_flag_key),
    )
    return ValueOptionAxes(value_axis=value_axis, option_axes=option_axes)


def sample_other_token_coords(*, rng: Any, occupied: set[Coord], colors: Sequence[str]) -> dict[str, Coord]:
    """Place non-target Ludo tokens on distinct visible main-track cells."""

    out: dict[str, Coord] = {}
    path = list(MAIN_PATH)
    for color in colors:
        rng.shuffle(path)
        for coord in path:
            if tuple(coord) not in occupied:
                occupied.add(tuple(coord))
                out[str(color)] = tuple(coord)
                break
        if str(color) not in out:
            raise ValueError("failed to place Ludo token")
    return out


def make_roll_options(*, rng: Any, correct_distance: int, answer_label: str, option_labels: Sequence[str]) -> Tuple[LudoRollOption, ...]:
    """Create unique visible roll options with the correct distance under the answer label."""

    correct_distance = int(correct_distance)
    labels = [str(label) for label in option_labels]
    correct_index = labels.index(str(answer_label))
    distractor_distances = [distance for distance in range(1, 12) if int(distance) != int(correct_distance)]
    rng.shuffle(distractor_distances)
    selected_distances = distractor_distances[: len(labels) - 1]
    options: list[LudoRollOption] = []
    distractor_cursor = 0
    for index, label in enumerate(labels):
        if index == correct_index:
            distance = int(correct_distance)
        else:
            distance = int(selected_distances[distractor_cursor])
            distractor_cursor += 1
        options.append(LudoRollOption(label=str(label), distance=int(distance), text=roll_option_text(distance)))
    return tuple(options)


def make_destination_options(
    *,
    rng: Any,
    route: Sequence[Coord],
    current_index: int,
    final_coord: Coord,
    occupied: set[Coord],
    answer_label: str,
    option_labels: Sequence[str],
) -> Tuple[LudoDestinationOption, ...]:
    """Create unique board-cell destination options around the true final cell."""

    labels = [str(label) for label in option_labels]
    correct_index = labels.index(str(answer_label))
    candidates: list[Coord] = []
    seen = {tuple(final_coord), *{tuple(coord) for coord in occupied}}
    preferred_indices = [int(current_index) + offset for offset in (-8, -6, -4, -2, 2, 4, 6, 8, 10, 12)]
    for index in preferred_indices:
        if 0 <= int(index) < len(route):
            coord = tuple(route[int(index)])
            if coord not in seen:
                candidates.append(coord)
                seen.add(coord)
    random_pool = [tuple(coord) for coord in route if tuple(coord) not in seen]
    rng.shuffle(random_pool)
    candidates.extend(random_pool[: max(0, len(labels) - 1 - len(candidates))])
    if len(candidates) < len(labels) - 1:
        raise ValueError("failed to construct enough Ludo destination options")
    options: list[LudoDestinationOption] = []
    distractor_cursor = 0
    for index, label in enumerate(labels):
        if int(index) == int(correct_index):
            coord = tuple(final_coord)
        else:
            coord = tuple(candidates[distractor_cursor])
            distractor_cursor += 1
        options.append(LudoDestinationOption(label=str(label), coord=coord))
    return tuple(options)


__all__ = [
    "IntegerAxisSelection",
    "LabelAxisSelection",
    "make_destination_options",
    "make_roll_options",
    "resolve_ludo_integer_axis",
    "resolve_ludo_named_axis",
    "resolve_ludo_option_label_axis",
    "resolve_ludo_scene_axes",
    "sample_other_token_coords",
]
