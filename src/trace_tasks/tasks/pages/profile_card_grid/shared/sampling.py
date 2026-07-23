"""Sampling helpers for profile-card-grid scene packages."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.pages.shared.sampling import (
    resolve_int_support as resolve_pages_int_support,
    resolve_named_axis as resolve_pages_named_axis,
    resolve_supported_int as resolve_pages_supported_int,
)
from trace_tasks.tasks.pages.shared.page_text_resources import (
    page_text_resource_metadata,
    sample_page_context_batch,
    sample_page_label_batch,
)

from .defaults import GENERATION_DEFAULTS, NAMESPACE_ROOT
from .state import (
    ACCENTS,
    PROFILE_FILTER_FIELDS,
    PROFILE_NUMERIC_FIELDS,
    PROFILE_RANK_ORDINALS,
    PROFILE_RANK_POSITION_SUPPORT,
    PROFILE_TEXT_FIELDS,
    SCENE_VARIANTS,
    ProfileCard,
    ProfileCardGridCase,
    ProfileCardGridSpec,
)


def resolve_named_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named scene axis."""

    return resolve_pages_named_axis(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace_root=NAMESPACE_ROOT,
        supported=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
        namespace=namespace,
    )


def resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve an integer support list from config or explicit params."""

    return resolve_pages_int_support(params=params, gen_defaults=GENERATION_DEFAULTS, key=key, fallback=fallback)


def resolve_supported_int(
    *,
    params: Mapping[str, Any],
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve one supported integer operand and probability map."""

    return resolve_pages_supported_int(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        namespace_root=NAMESPACE_ROOT,
        explicit_key=explicit_key,
        support_key=support_key,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )


def profile_numeric_field_labels() -> Tuple[str, ...]:
    """Return visible numeric field labels that support ordering tasks."""

    return tuple(str(label) for label, _values in PROFILE_NUMERIC_FIELDS)


def profile_filter_field_labels() -> Tuple[str, ...]:
    """Return visible categorical field labels that support filtered tasks."""

    return tuple(str(label) for label, _values in PROFILE_FILTER_FIELDS)


def _filter_values_for_field(field_label: str) -> Tuple[str, ...]:
    for label, values in PROFILE_FILTER_FIELDS:
        if str(label) == str(field_label):
            return tuple(str(value) for value in values)
    raise ValueError(f"filter field {field_label!r} is not supported")


def resolve_profile_numeric_field(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> str:
    """Select one numeric profile-card field."""

    field_labels = profile_numeric_field_labels()
    explicit_field = params.get("field_label")
    if explicit_field is not None:
        target_field = str(explicit_field)
        if target_field not in set(field_labels):
            raise ValueError(f"field_label must be one of {list(field_labels)}")
        return str(target_field)
    field_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.{namespace}.numeric_field_label",
    ) % len(field_labels)
    return str(field_labels[int(field_index)])


def resolve_profile_filter_field(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> str:
    """Select one repeated categorical profile-card field."""

    field_labels = profile_filter_field_labels()
    explicit_field = params.get("filter_field_label")
    if explicit_field is not None:
        target_field = str(explicit_field)
        if target_field not in set(field_labels):
            raise ValueError(f"filter_field_label must be one of {list(field_labels)}")
        return str(target_field)
    field_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.{namespace}.filter_field_label",
    ) % len(field_labels)
    return str(field_labels[int(field_index)])


def profile_rank_ordinal(rank_position: int) -> str:
    """Return prompt text for a supported profile-card rank."""

    return str(PROFILE_RANK_ORDINALS.get(int(rank_position), f"{int(rank_position)}th"))


def resolve_profile_rank_position(
    *,
    params: Mapping[str, Any],
    card_count: int,
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve a non-extremal rank position."""

    rank_position, support, probabilities = resolve_supported_int(
        params=params,
        explicit_key="rank_position",
        support_key="rank_position_support",
        fallback=PROFILE_RANK_POSITION_SUPPORT,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.rank_position.{int(card_count)}",
    )
    if int(rank_position) > int(card_count):
        raise ValueError("rank_position cannot exceed card_count")
    return int(rank_position), tuple(int(value) for value in support), dict(probabilities)


def build_profile_card_spec(
    *,
    card_count: int,
    instance_seed: int,
    include_numeric_fields: bool,
    filter_field_label: str = "",
) -> ProfileCardGridSpec:
    """Build visible profile cards with unique labels and field values."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.cards")
    title_batch = sample_page_context_batch(
        rng,
        role="profile_card_grid_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="profile_card_grid_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
    )
    name_batch = sample_page_label_batch(
        rng,
        role="profile_card_grid_profile_name",
        count=int(card_count),
        manifest_name="people/first_names_ssa.txt",
        min_chars=3,
        max_chars=12,
        allow_spaces=False,
        allow_punctuation=False,
    )
    names = list(name_batch.values)
    field_values_by_label: Dict[str, List[str]] = {}
    profile_field_specs = () if str(filter_field_label) else (PROFILE_TEXT_FIELDS[:2] if bool(include_numeric_fields) else PROFILE_TEXT_FIELDS)
    for field_label, values in profile_field_specs:
        shuffled = list(values)
        rng.shuffle(shuffled)
        field_values_by_label[str(field_label)] = shuffled
    filter_values_by_card: List[str] = []
    if str(filter_field_label):
        filter_values = list(_filter_values_for_field(str(filter_field_label)))
        rng.shuffle(filter_values)
        group_count = min(3, len(filter_values), int(card_count))
        selected_groups = filter_values[: int(group_count)]
        filter_values_by_card = [str(selected_groups[index % int(group_count)]) for index in range(int(card_count))]
        rng.shuffle(filter_values_by_card)
    numeric_values_by_label: Dict[str, List[int]] = {}
    if bool(include_numeric_fields):
        for field_label, values in PROFILE_NUMERIC_FIELDS:
            shuffled_numeric = [int(value) for value in values]
            rng.shuffle(shuffled_numeric)
            numeric_values_by_label[str(field_label)] = shuffled_numeric
    cards: List[ProfileCard] = []
    color_offset = int(rng.randrange(len(ACCENTS)))
    for index in range(int(card_count)):
        fields = {
            str(field_label): str(field_values_by_label[str(field_label)][int(index)])
            for field_label, _values in profile_field_specs
        }
        if str(filter_field_label):
            fields[str(filter_field_label)] = str(filter_values_by_card[int(index)])
        numeric_fields = {
            str(field_label): int(values[int(index)])
            for field_label, values in numeric_values_by_label.items()
        }
        fields.update({str(label): str(value) for label, value in numeric_fields.items()})
        cards.append(
            ProfileCard(
                profile_id=f"profile_{int(index) + 1}",
                name=str(names[int(index)]),
                fields=fields,
                numeric_fields=dict(numeric_fields),
                accent_rgb=ACCENTS[(int(index) + int(color_offset)) % len(ACCENTS)],
            )
        )
    return ProfileCardGridSpec(
        cards=tuple(cards),
        title=str(title_batch.values[0]),
        subtitle=str(subtitle_batch.values[0]),
        text_resource_metadata=page_text_resource_metadata(title_batch, subtitle_batch, name_batch),
        filter_field_label=str(filter_field_label),
    )


def build_profile_card_case(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    include_numeric_fields: bool,
    include_filter_field: bool = False,
) -> ProfileCardGridCase:
    """Sample the scene-level profile-card grid state."""

    scene_variant, scene_variant_probabilities = resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    card_count, card_count_support, card_count_probabilities = resolve_supported_int(
        params=params,
        explicit_key="card_count",
        support_key="card_count_support",
        fallback=(6, 9),
        instance_seed=int(instance_seed),
        namespace="card_count",
    )
    filter_field_label = (
        resolve_profile_filter_field(
            params=params,
            instance_seed=int(instance_seed),
            namespace="filter_field",
        )
        if bool(include_filter_field)
        else ""
    )
    spec = build_profile_card_spec(
        card_count=int(card_count),
        instance_seed=int(instance_seed),
        include_numeric_fields=bool(include_numeric_fields),
        filter_field_label=str(filter_field_label),
    )
    return ProfileCardGridCase(
        spec=spec,
        scene_variant=str(scene_variant),
        card_count=int(card_count),
        card_count_support=tuple(int(value) for value in card_count_support),
        card_count_probabilities=dict(card_count_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )


def select_text_field_target(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    cards: Sequence[ProfileCard],
    namespace: str,
) -> Tuple[ProfileCard, str]:
    """Select one profile card and one non-ranking field."""

    field_labels = [str(label) for label, _values in PROFILE_TEXT_FIELDS]
    explicit_field = params.get("field_label")
    if explicit_field is not None:
        target_field = str(explicit_field)
        if target_field not in field_labels:
            raise ValueError(f"field_label must be one of {field_labels}")
    else:
        field_index = resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.{namespace}.field_label",
        ) % len(field_labels)
        target_field = str(field_labels[int(field_index)])
    explicit_profile_index = params.get("profile_index")
    if explicit_profile_index is not None:
        target_index = int(explicit_profile_index)
        if target_index < 0 or target_index >= len(cards):
            raise ValueError("profile_index out of range")
    else:
        target_index = int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}.{namespace}.profile_index.{target_field}",
            )
            % len(cards)
        )
    return cards[int(target_index)], str(target_field)


def numeric_candidates(
    *,
    cards: Sequence[ProfileCard],
    target_field: str,
) -> List[Dict[str, Any]]:
    """Return sorted-capable numeric candidate payloads for one field."""

    candidates: List[Dict[str, Any]] = []
    for card in cards:
        if str(target_field) not in card.numeric_fields:
            raise ValueError(f"numeric field {target_field} missing from profile card")
        value = int(card.numeric_fields[str(target_field)])
        candidates.append(
            {
                "profile_id": str(card.profile_id),
                "profile_name": str(card.name),
                "field_label": str(target_field),
                "field_value": str(card.fields[str(target_field)]),
                "numeric_value": int(value),
            }
        )
    values = [int(candidate["numeric_value"]) for candidate in candidates]
    if len(values) != len(set(values)):
        raise ValueError(f"numeric field {target_field} values must be unique")
    return candidates


def sort_numeric_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rank_direction: str,
) -> List[Dict[str, Any]]:
    """Sort candidates for highest-first or lowest-first reasoning."""

    want_highest = str(rank_direction) == "highest"
    return [
        dict(candidate)
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                -int(candidate["numeric_value"]) if want_highest else int(candidate["numeric_value"]),
                str(candidate["profile_name"]),
            ),
        )
    ]


def card_by_profile_id(cards: Sequence[ProfileCard], profile_id: str) -> ProfileCard:
    """Return the card with a matching stable profile id."""

    for card in cards:
        if str(card.profile_id) == str(profile_id):
            return card
    raise ValueError(f"missing profile card {profile_id}")
