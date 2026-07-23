"""Scene-neutral sampling helpers for marble-chain tasks."""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .defaults import COLOR_KEYS, DEFAULTS, OPTION_LABELS, SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS
from .rules import all_outcomes
from .state import MarbleIntegerAxis, MarbleOutcome, MarbleSceneAxes, ShotOption


def resolve_marble_integer_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> MarbleIntegerAxis:
    """Resolve one integer generation axis with explicit-param override support."""

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
    support = group_default(gen_defaults, str(support_key), tuple(int(item) for item in fallback_support))
    return MarbleIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_marble_scene_axes(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> MarbleSceneAxes:
    """Sample scene/style axes that are independent of the task objective."""

    scene_variant, scene_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_STYLE_VARIANTS,
    )
    return MarbleSceneAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_probs),
    )


def sample_color_keys(rng: Any, *, color_count: int) -> Tuple[str, ...]:
    """Sample the visible color palette for one marble chain."""

    colors = list(COLOR_KEYS)
    rng.shuffle(colors)
    return tuple(colors[: int(color_count)])


def sample_chain(rng: Any, *, length: int, color_keys: Sequence[str], shooter_color: str) -> Tuple[str, ...]:
    """Create a chain with local runs but without identical adjacent run starts."""

    chain: List[str] = []
    while len(chain) < int(length):
        allowed = [str(color) for color in color_keys if not chain or str(color) != str(chain[-1])]
        color = str(allowed[int(rng.randrange(len(allowed)))])
        if color == str(shooter_color):
            run_len = int(rng.choice((1, 1, 2, 2, 3, 4, 5)))
        else:
            run_len = int(rng.choice((1, 1, 2, 2, 3)))
        remaining = int(length) - len(chain)
        chain.extend([str(color)] * min(int(run_len), int(remaining)))
    return tuple(chain[: int(length)])


def answer_option_label(instance_seed: int, *, params: Mapping[str, Any], option_count: int) -> str:
    """Resolve the displayed option label that should hold the correct shot."""

    labels = OPTION_LABELS[: int(option_count)]
    raw = params.get("answer_option_label")
    if raw is not None:
        value = str(raw).strip().upper()
        if value not in labels:
            raise ValueError(f"answer_option_label={value!r} is not available for {option_count} options")
        return str(value)
    rng = spawn_rng(int(instance_seed), "games.marble_chain.answer_label")
    return str(uniform_choice(rng, labels))


def pick_slots_with_spacing(
    rng: Any,
    candidates: Sequence[int],
    *,
    count: int,
    blocked: Sequence[int] = (),
    min_gap: int = 2,
) -> List[int]:
    """Choose slot ids while spreading visible arrows where possible."""

    blocked_set = {int(value) for value in blocked}
    pool = [int(value) for value in candidates if int(value) not in blocked_set]
    rng.shuffle(pool)
    selected: List[int] = []
    for slot in pool:
        if all(abs(int(slot) - int(existing)) >= int(min_gap) for existing in selected) and all(
            abs(int(slot) - int(existing)) >= int(min_gap) for existing in blocked_set
        ):
            selected.append(int(slot))
            if len(selected) == int(count):
                return selected
    for slot in pool:
        if int(slot) not in selected:
            selected.append(int(slot))
            if len(selected) == int(count):
                return selected
    return selected


def sample_chain_state(
    rng: Any,
    *,
    chain_length: int,
    color_count: int,
) -> tuple[Tuple[str, ...], str, dict[int, MarbleOutcome]]:
    """Sample chain colors, shooter color, and all immediate insertion outcomes."""

    color_keys = sample_color_keys(rng, color_count=int(color_count))
    shooter_color = str(color_keys[int(rng.randrange(len(color_keys)))])
    chain_colors = sample_chain(
        rng,
        length=int(chain_length),
        color_keys=color_keys,
        shooter_color=str(shooter_color),
    )
    return tuple(chain_colors), str(shooter_color), all_outcomes(chain_colors, shooter_color=str(shooter_color))


def build_shot_options(
    *,
    labels: Sequence[str],
    answer_label: str,
    answer_slot: int,
    distractor_slots: Sequence[int],
    outcomes: Mapping[int, MarbleOutcome],
) -> tuple[ShotOption, ...]:
    """Bind labels to answer/distractor slots without choosing objective semantics."""

    specs: list[ShotOption] = []
    distractor_cursor = 0
    for label in labels:
        if str(label) == str(answer_label):
            slot = int(answer_slot)
        else:
            slot = int(distractor_slots[distractor_cursor])
            distractor_cursor += 1
        specs.append(
            ShotOption(
                label=str(label),
                slot_index=int(slot),
                outcome=outcomes[int(slot)],
                is_answer=str(label) == str(answer_label),
            )
        )
    return tuple(specs)


def is_interior_slot(slot_index: int, *, chain_length: int) -> bool:
    """Return whether an insertion slot is between two existing marbles."""

    return 0 < int(slot_index) < int(chain_length)


def interior_slots(candidates: Sequence[int], *, chain_length: int) -> list[int]:
    """Filter candidate slots to visible gaps between existing marbles."""

    return [int(slot) for slot in candidates if is_interior_slot(int(slot), chain_length=int(chain_length))]


SlotGroupBuilder = Callable[[Mapping[int, MarbleOutcome]], tuple[Sequence[int], Sequence[int]]]
StateSlotGroupBuilder = Callable[[Sequence[str], Mapping[int, MarbleOutcome]], tuple[Sequence[int], Sequence[int]]]
DisplayValidator = Callable[[Sequence[ShotOption]], bool]


def sample_direction_option_scene_from_state_groups(
    rng: Any,
    *,
    chain_length: int,
    color_count: int,
    option_count: int,
    answer_label: str,
    slot_group_builder: StateSlotGroupBuilder,
    display_validator: DisplayValidator,
) -> tuple[Tuple[str, ...], str, tuple[ShotOption, ...]]:
    """Construct labeled arrows from callbacks that can inspect chain colors."""

    labels = OPTION_LABELS[: int(option_count)]
    for _attempt in range(800):
        chain_colors, shooter_color, outcomes = sample_chain_state(
            rng,
            chain_length=int(chain_length),
            color_count=int(color_count),
        )
        answer_candidates, distractor_candidates = slot_group_builder(chain_colors, outcomes)
        answer_candidates = interior_slots(answer_candidates, chain_length=len(chain_colors))
        distractor_candidates = interior_slots(distractor_candidates, chain_length=len(chain_colors))
        if not answer_candidates or len(distractor_candidates) < int(option_count) - 1:
            continue
        answer_slot = int(answer_candidates[int(rng.randrange(len(answer_candidates)))])
        distractor_slots = pick_slots_with_spacing(
            rng,
            distractor_candidates,
            count=int(option_count) - 1,
            blocked=(answer_slot,),
            min_gap=3,
        )
        if len(distractor_slots) < int(option_count) - 1:
            continue
        option_specs = build_shot_options(
            labels=labels,
            answer_label=str(answer_label),
            answer_slot=int(answer_slot),
            distractor_slots=distractor_slots,
            outcomes=outcomes,
        )
        if not display_validator(option_specs):
            continue
        return tuple(chain_colors), str(shooter_color), tuple(option_specs)
    raise ValueError("failed to sample marble-chain direction-option scene")


def sample_direction_option_scene(
    rng: Any,
    *,
    chain_length: int,
    color_count: int,
    option_count: int,
    answer_label: str,
    slot_group_builder: SlotGroupBuilder,
    display_validator: DisplayValidator,
) -> tuple[Tuple[str, ...], str, tuple[ShotOption, ...]]:
    """Construct labeled arrows once task-owned callbacks define valid slots."""

    return sample_direction_option_scene_from_state_groups(
        rng,
        chain_length=int(chain_length),
        color_count=int(color_count),
        option_count=int(option_count),
        answer_label=str(answer_label),
        slot_group_builder=lambda _chain_colors, outcomes: slot_group_builder(outcomes),
        display_validator=display_validator,
    )


def resolve_chain_length_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> MarbleIntegerAxis:
    """Resolve the chain-length axis used by all marble objectives."""

    return resolve_marble_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="chain_length_support",
        explicit_key="chain_length",
        fallback_support=DEFAULTS.chain_length_support,
        namespace=f"{namespace}.chain_length",
        balanced_flag_key="balanced_chain_length_sampling",
    )


def resolve_color_count_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> MarbleIntegerAxis:
    """Resolve how many colors appear in the generated marble chain."""

    return resolve_marble_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="color_count_support",
        explicit_key="color_count",
        fallback_support=DEFAULTS.color_count_support,
        namespace=f"{namespace}.color_count",
        balanced_flag_key="balanced_color_count_sampling",
    )


def resolve_option_count_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> MarbleIntegerAxis:
    """Resolve the number of visible shot-option arrows."""

    return resolve_marble_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.option_count_support,
        namespace=f"{namespace}.option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )


def resolve_target_pop_axis(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str = "target_pop_count",
) -> MarbleIntegerAxis:
    """Resolve the requested pop count for target-conditioned tasks."""

    return resolve_marble_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=params,
        support_key="target_pop_count_support",
        explicit_key=str(explicit_key),
        fallback_support=DEFAULTS.target_pop_count_support,
        namespace=f"{namespace}.target_pop_count",
        balanced_flag_key="balanced_target_answer_sampling",
    )


__all__ = [
    "answer_option_label",
    "build_shot_options",
    "interior_slots",
    "is_interior_slot",
    "pick_slots_with_spacing",
    "resolve_chain_length_axis",
    "resolve_color_count_axis",
    "resolve_marble_scene_axes",
    "resolve_option_count_axis",
    "resolve_target_pop_axis",
    "sample_chain",
    "sample_chain_state",
    "sample_color_keys",
    "sample_direction_option_scene",
    "sample_direction_option_scene_from_state_groups",
]
